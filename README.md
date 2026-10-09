# CRRA Lifecycle Optimizer — Streamlit Demo

Interactive demo of CRRA-utility-optimal equity/bond allocation across a 40-year accumulation + ≤34-year drawdown lifecycle, with stochastic SSA mortality and IID bootstrap return paths.

This is a **standalone demo** — not the QFE5315 paper engine. Methodology diverges from the paper (yfinance vs CRSP, IID vs block bootstrap, fixed flat income).

## Quick start

```bash
pip install -r requirements.txt

# 1) Generate the SPY baseline (one-time, ~5–10 hours, run overnight)
python precompute.py

# 2) Run the app
streamlit run app.py
```

## Files

| File | Purpose |
|---|---|
| `simulation.py` | CRRA utility, lifecycle sim, mortality, bootstrap, optimizer |
| `data_loader.py` | yfinance equity/bond + Shiller CPI fetchers |
| `precompute.py` | Generate `precomputed/spy_baseline_g{2,3p84,7}.npz` baselines |
| `app.py` | Streamlit UI |
| `math_auditor.py` | Independent math audit via the OpenAI API (see below) |
| `audits/` | Audit reports (`.md` + raw `.json`), committed as an audit trail |
| `precomputed/` | Baseline results + Shiller CPI snapshot + death-month draws |

## Configuration

Locked simulation parameters live at the top of `simulation.py`:
- Accumulation: 40 years (age 25–65), 10% contribution rate
- Drawdown: 4% rule (Bengen), inflation-adjusted
- Mortality: SSA 2022 stochastic, both members of equal-age couple
- CRRA: γ ∈ {2, 3.84, 7}, θ = 2360 × 12^γ, k = $490,000
- Bootstrap: IID, seed=42, 50,000 paths (baseline) / 5,000 (live)

## Independent math audit (ChatGPT)

`math_auditor.py` sends `simulation.py` and `data_loader.py` (line-numbered) plus this README's stated methodology to an OpenAI reasoning model. The model sees only code and spec, never Claude's own conclusions, and checks the math from first principles: units, indexing, boundary conditions, RNG independence, and whether the code matches the spec. Each finding must carry a derivation and a hand-computed check.

```bash
export OPENAI_API_KEY=sk-...                      # never commit this
python math_auditor.py                            # default files
python math_auditor.py --focus "bequest utility units"
python math_auditor.py simulation.py --model gpt-5.5 --effort xhigh
python math_auditor.py --dry-run                  # inspect the request only
```

Model and effort default to `gpt-5.5` / `high`. Override them with `OPENAI_AUDIT_MODEL` / `OPENAI_AUDIT_EFFORT` or the flags. Reports go to `audits/`. The command exits 1 if any `critical` or `major` finding is reported. `openai` is not in `requirements.txt` (the script uses `requests`), so the Streamlit deploy is unaffected.

## Deployment

Streamlit Cloud:
1. Push this directory (with `precomputed/` populated) to GitHub.
2. Connect repo at share.streamlit.io, point at `app.py`.
3. The `.npz` baselines are required at runtime — do NOT regenerate them on the cloud (insufficient memory/runtime).
