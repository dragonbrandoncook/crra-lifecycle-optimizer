"""
Run locally — generates the SPY baseline at 50,000 paths × 21 weight points,
for each of three γ values (Aggressive=2, Moderate=3.84, Conservative=7).

Outputs to precomputed/:
    spy_baseline_g2.npz
    spy_baseline_g3p84.npz
    spy_baseline_g7.npz
    death_months_50k.npy   (first, last) stacked, shape (2, 50000)
    death_months_5k.npy    (2, 5000)  — for live runs to reuse
    shiller_cpi_snapshot.xls — auto-cached by data_loader
    metadata.json

Usage:
    cd crra_app
    python precompute.py
    # ~5–10 hours total at 50k × 21 × 3γ on a typical laptop.
    # Run overnight.

Do NOT run this inside Streamlit. Streamlit Cloud cannot allocate the memory
or runtime needed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from data_loader import load_equity_returns, load_bond_returns, load_shiller_cpi
from simulation import (
    GAMMA_BASE,
    SS_FLOOR_MONTHLY,
    couple_death_months,
    run_crra_grid,
)

OUT_DIR = Path(__file__).parent / "precomputed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GAMMAS_TO_RUN = [
    ("g2",    2.0),
    ("g3p84", 3.84),
    ("g7",    7.0),
]
WEIGHT_GRID_FINE = np.round(np.arange(0.0, 1.0001, 0.05), 3).tolist()  # 21 points
N_PATHS_BASELINE = 50_000
N_PATHS_LIVE     = 5_000


def main() -> None:
    print("=" * 72)
    print("CRRA LIFECYCLE OPTIMIZER — PRECOMPUTE BASELINES")
    print("=" * 72)

    # ── 1. Load data once (snapshots Shiller CPI to disk for app deployment) ──
    print("\n[1/4] Loading Shiller CPI...")
    cpi = load_shiller_cpi()
    print(f"      CPI: {len(cpi)} months  [{cpi.index[0]} → {cpi.index[-1]}]")

    print("\n[2/4] Loading SPY (equity) and ^TNX (bonds) from yfinance...")
    eq = load_equity_returns("SPY", cpi=cpi, start="1993-01-01")
    bd = load_bond_returns(cpi=cpi, start="1993-01-01")
    print(f"      SPY:  {len(eq)} months  [{eq.index[0]} → {eq.index[-1]}]")
    print(f"      ^TNX: {len(bd)} months  [{bd.index[0]} → {bd.index[-1]}]")

    # ── 2. Precompute and save death months (reused across γ values + live) ──
    print("\n[3/4] Precomputing mortality draws (SSA 2022 stochastic)...")
    first_50k, last_50k = couple_death_months(N_PATHS_BASELINE)
    first_5k,  last_5k  = couple_death_months(N_PATHS_LIVE)
    np.save(OUT_DIR / "death_months_50k.npy", np.stack([first_50k, last_50k]))
    np.save(OUT_DIR / "death_months_5k.npy",  np.stack([first_5k,  last_5k]))
    mean_age_at_last = 25 + last_50k.mean() / 12
    print(f"      Mean age at last-survivor death: {mean_age_at_last:.2f}  (target ~86.8)")

    # ── 3. Sweep γ ────────────────────────────────────────────────────────
    print("\n[4/4] Running CRRA grid for each γ — this is the long part.")
    for tag, gamma in GAMMAS_TO_RUN:
        out_file = OUT_DIR / f"spy_baseline_{tag}.npz"
        print(f"\n  → γ = {gamma}  ({tag})")
        t0 = time.time()
        result = run_crra_grid(
            equity_returns = eq,
            bond_returns   = bd,
            n_paths        = N_PATHS_BASELINE,
            weight_grid    = WEIGHT_GRID_FINE,
            gamma          = gamma,
            ss_monthly     = SS_FLOOR_MONTHLY,
            death_months   = (first_50k, last_50k),
        )
        elapsed = time.time() - t0
        print(f"     w*={result['optimal_w']:.4f}   "
              f"E[U]={result['optimal_eu']:.4e}   "
              f"ruin@w*≈{np.interp(result['optimal_w'], result['weights'], result['ruin'])*100:.2f}%   "
              f"({elapsed/60:.1f} min)")

        # np.savez doesn't take Python scalars cleanly; wrap them
        np.savez_compressed(
            out_file,
            weights      = np.asarray(result["weights"]),
            eu           = np.asarray(result["eu"]),
            ruin         = np.asarray(result["ruin"]),
            median_tw    = np.asarray(result["median_tw"]),
            optimal_w    = np.asarray(result["optimal_w"]),
            optimal_eu   = np.asarray(result["optimal_eu"]),
            gamma        = np.asarray(result["gamma"]),
            ss_monthly   = np.asarray(result["ss_monthly"]),
            n_paths      = np.asarray(result["n_paths"]),
            equity_n     = np.asarray(result["equity_n"]),
            equity_start = np.asarray(result["equity_start"]),
            equity_end   = np.asarray(result["equity_end"]),
        )
        print(f"     wrote {out_file.name}")

    # ── 4. Metadata ───────────────────────────────────────────────────────
    metadata = {
        "ticker":          "SPY",
        "equity_source":   "yfinance Adj Close (1993+)",
        "bond_source":     "yfinance ^TNX (10-yr UST yield) -> Macaulay duration",
        "cpi_source":      "Shiller Yale ie_data.xls",
        "bootstrap":       "IID, seed=42",
        "n_paths":         N_PATHS_BASELINE,
        "n_paths_live":    N_PATHS_LIVE,
        "weight_grid":     WEIGHT_GRID_FINE,
        "ss_monthly":      SS_FLOOR_MONTHLY,
        "gammas":          [g for _, g in GAMMAS_TO_RUN],
        "default_gamma":   GAMMA_BASE,
        "files": {
            f"spy_baseline_{tag}.npz": {"gamma": g}
            for tag, g in GAMMAS_TO_RUN
        },
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'metadata.json'}")
    print("\nPrecompute complete.")


if __name__ == "__main__":
    main()
