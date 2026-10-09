"""
Independent math audit via the OpenAI API.

Sends the simulation math (with line numbers) plus the README's stated
methodology to an OpenAI reasoning model and asks it to verify the math
from first principles. The auditor sees only source code and the spec —
never another model's conclusions — so its findings are independent.

Usage:
    export OPENAI_API_KEY=sk-...
    python math_auditor.py                          # audit the default files
    python math_auditor.py simulation.py            # audit specific files
    python math_auditor.py --focus "bequest units"  # steer attention
    python math_auditor.py --dry-run                # print the request, no API call

Environment:
    OPENAI_API_KEY       required (except --dry-run)
    OPENAI_AUDIT_MODEL   model id, default below
    OPENAI_AUDIT_EFFORT  reasoning effort (low|medium|high|xhigh), or "none"
                         to omit the reasoning block for non-reasoning models

Writes audits/<timestamp>_<model>.{md,json}. Exits 1 if any critical or
major finding is reported, so it can gate a commit or CI step.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT          = Path(__file__).parent
AUDIT_DIR     = ROOT / "audits"
API_URL       = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = os.environ.get("OPENAI_AUDIT_MODEL", "gpt-5.5")
DEFAULT_EFFORT = os.environ.get("OPENAI_AUDIT_EFFORT", "high")
DEFAULT_FILES = ["simulation.py", "data_loader.py"]
SPEC_FILE     = "README.md"
BLOCKING      = {"critical", "major"}

INSTRUCTIONS = """\
You are an independent mathematical auditor for a quantitative finance codebase:
a CRRA-utility lifecycle portfolio optimizer (Monte Carlo accumulation +
drawdown with stochastic couple mortality and bootstrapped real returns).

Your job is to find mathematical errors, not to praise the code.

Rules:
- Verify every formula from first principles. Do NOT trust comments or
  docstrings that claim the math is "verbatim", "unchanged" or "correct".
- Check units and time scales (monthly vs annual, real vs nominal, per-capita
  vs household), sign conventions, off-by-one errors in month/age indexing,
  boundary conditions (death at retirement, ruin, truncation of the age
  horizon), RNG stream independence, and numerical stability (e.g. large
  negative powers when gamma is large).
- Check the code against the stated methodology in the spec. A mismatch
  between spec and code is a finding.
- For each finding, show the derivation or a concrete numeric counterexample.
  If you cannot show it, downgrade it to "note" and say it is unconfirmed.
- Severity: critical = results are wrong / optimizer conclusion likely
  changes; major = material bias in reported numbers; minor = small or
  edge-case error; note = style, unconfirmed suspicion, or documentation.
- For each finding give a concrete verification: an input and the expected
  output a correct implementation must produce, computed by hand.
- List what you checked and found correct in checked_and_ok, so silence on
  a component is never ambiguous.
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "summary", "findings", "checked_and_ok"],
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "pass_with_issues", "fail"]},
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "severity", "file", "lines", "title",
                             "issue", "derivation", "suggested_fix", "verification"],
                "properties": {
                    "id":            {"type": "string"},
                    "severity":      {"type": "string",
                                      "enum": ["critical", "major", "minor", "note"]},
                    "file":          {"type": "string"},
                    "lines":         {"type": "string"},
                    "title":         {"type": "string"},
                    "issue":         {"type": "string"},
                    "derivation":    {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "verification":  {"type": "string"},
                },
            },
        },
        "checked_and_ok": {"type": "array", "items": {"type": "string"}},
    },
}


def _numbered(path: Path) -> str:
    lines = path.read_text().splitlines()
    return "\n".join(f"{i:4d}| {line}" for i, line in enumerate(lines, 1))


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_input(files: list[str], focus: str | None) -> str:
    parts = [f"## Stated methodology ({SPEC_FILE})\n\n{(ROOT / SPEC_FILE).read_text()}"]
    for f in files:
        parts.append(f"## File: {f}\n\n```python\n{_numbered(ROOT / f)}\n```")
    if focus:
        parts.append(f"## Requested focus\n\n{focus}\n\n"
                     "Prioritise this area, but still report anything else you find.")
    return "\n\n".join(parts)


def build_request(model: str, effort: str, user_input: str) -> dict:
    body = {
        "model": model,
        "instructions": INSTRUCTIONS,
        "input": user_input,
        "text": {"format": {"type": "json_schema", "name": "math_audit",
                            "strict": True, "schema": SCHEMA}},
        "store": False,
    }
    if effort != "none":
        body["reasoning"] = {"effort": effort}
    return body


def call_openai(body: dict, api_key: str, timeout: int) -> dict:
    r = requests.post(API_URL, json=body, timeout=timeout,
                      headers={"Authorization": f"Bearer {api_key}"})
    if r.status_code != 200:
        sys.exit(f"OpenAI API error {r.status_code}: {r.text[:2000]}")
    resp = r.json()
    if resp.get("status") != "completed":
        sys.exit(f"Response not completed: status={resp.get('status')} "
                 f"details={resp.get('incomplete_details')}")
    for item in resp.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "refusal":
                sys.exit(f"Auditor refused: {part.get('refusal')}")
            if part.get("type") == "output_text":
                return json.loads(part["text"])
    sys.exit("No output_text in response:\n" + json.dumps(resp, indent=2)[:2000])


def render_markdown(audit: dict, meta: dict) -> str:
    out = [
        f"# Math audit — {meta['timestamp']}",
        "",
        f"- Model: `{meta['model']}` (effort: {meta['effort']})",
        f"- Commit: `{meta['commit']}`",
        f"- Files: {', '.join(meta['files'])}",
        f"- Focus: {meta['focus'] or '—'}",
        f"- **Verdict: {audit['verdict']}**",
        "",
        "## Summary",
        "",
        audit["summary"],
        "",
        "## Findings",
        "",
    ]
    order = {"critical": 0, "major": 1, "minor": 2, "note": 3}
    findings = sorted(audit["findings"], key=lambda f: order[f["severity"]])
    if not findings:
        out.append("None reported.")
    for f in findings:
        out += [
            f"### [{f['severity'].upper()}] {f['id']}: {f['title']}",
            f"`{f['file']}` lines {f['lines']}",
            "",
            f"**Issue.** {f['issue']}",
            "",
            f"**Derivation.** {f['derivation']}",
            "",
            f"**Suggested fix.** {f['suggested_fix']}",
            "",
            f"**Verification.** {f['verification']}",
            "",
        ]
    out += ["## Checked and OK", ""]
    out += [f"- {c}" for c in audit["checked_and_ok"]] or ["- (none listed)"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--focus", help="area to prioritise, e.g. 'bequest utility units'")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT)
    ap.add_argument("--timeout", type=int, default=900, help="seconds")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the request body and exit without calling the API")
    args = ap.parse_args()

    body = build_request(args.model, args.effort, build_input(args.files, args.focus))
    if args.dry_run:
        print(json.dumps(body, indent=2))
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY is not set.")

    audit = call_openai(body, api_key, args.timeout)
    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ"),
        "model": args.model, "effort": args.effort, "commit": _git_sha(),
        "files": args.files, "focus": args.focus,
    }
    AUDIT_DIR.mkdir(exist_ok=True)
    stem = f"{meta['timestamp']}_{args.model}"
    md_path, json_path = AUDIT_DIR / f"{stem}.md", AUDIT_DIR / f"{stem}.json"
    json_path.write_text(json.dumps({"meta": meta, "audit": audit}, indent=2))
    report = render_markdown(audit, meta)
    md_path.write_text(report)
    print(report)
    print(f"Wrote {md_path} and {json_path}")

    blocking = [f for f in audit["findings"] if f["severity"] in BLOCKING]
    sys.exit(1 if blocking else 0)


if __name__ == "__main__":
    main()
