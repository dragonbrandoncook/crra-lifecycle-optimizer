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
| `precomputed/` | Baseline results + Shiller CPI snapshot + death-month draws |

## Configuration

Locked simulation parameters live at the top of `simulation.py`:
- Accumulation: 40 years (age 25–65), 10% contribution rate
- Drawdown: 4% rule (Bengen), inflation-adjusted
- Mortality: SSA 2022 stochastic, both members of equal-age couple
- CRRA: γ ∈ {2, 3.84, 7}, θ = 2360 × 12^γ, k = $490,000
- Bootstrap: IID, seed=42, 50,000 paths (baseline) / 5,000 (live)

## Deployment

Streamlit Cloud:
1. Push this directory (with `precomputed/` populated) to GitHub.
2. Connect repo at share.streamlit.io, point at `app.py`.
3. The `.npz` baselines are required at runtime — do NOT regenerate them on the cloud (insufficient memory/runtime).
