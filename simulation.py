"""
CRRA Lifecycle Simulation Engine.

Functions are extracted from finance_project3.1.ipynb (Cells 5, 8, 14, 16, 20, 24)
and adapted to take explicit arguments instead of relying on notebook globals.
Math is unchanged.

The pipeline:
    yfinance returns
        -> iid_bootstrap()              -> (n_paths, n_months, 2) real-return paths
        -> draw_death_months() x2       -> stochastic SSA mortality per path
        -> simulate_lifecycle()         -> consumption + terminal wealth per path
        -> crra_expected_utility()      -> scalar E[U]
    repeated across a weight grid -> run_crra_grid() -> optimal w*

All long-running work uses pure NumPy. No Streamlit imports here so this module
is callable from precompute.py and app.py alike.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ============================================================================
# LOCKED CONSTANTS — must match the research notebook exactly
# ============================================================================
GAMMA_BASE        = 3.84            # De Nardi, French, Jones (2010)
K_BEQUEST         = 490_000.0       # bequest scale, 2022 USD
DELTA_DISC        = 1.0             # no time discount within retirement
N_MONTHS          = 888             # 480 accum + 360 drawdown + 48 mortality buffer
ACCUM_MONTHS      = 480             # age 25–65
DRAWDOWN_MAX      = 408             # age 65–99 cap (= N_MONTHS - ACCUM_MONTHS)
ANNUAL_INCOME     = 60_000.0        # flat normalized income
CONTRIB_RATE      = 0.10            # 10% of income
WITHDRAW_RATE_ANN = 0.04            # Bengen 4% rule
SS_FLOOR_MONTHLY  = 1_261.0         # SSA floor, 2022 USD (default; overridable)
SEED              = 42

MEAN_BLOCK_LEN = 120                # kept for completeness; this build uses IID

def theta_of(gamma: float) -> float:
    """Bequest scale parameter, as in Step 4 Cell 1."""
    return 2360.0 * (12.0 ** gamma)

# ============================================================================
# SSA 2022 PERIOD LIFE TABLE — verbatim from notebook Cell 14
# Anchors at 5-yr ages, log-linearly interpolated to single year of age 25–119.
# ============================================================================
ANCHOR_AGES = np.array(
    [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 119]
)
QX_MALE_ANCH = np.array([
    0.00151, 0.00190, 0.00237, 0.00287, 0.00381, 0.00540, 0.00760, 0.01090,
    0.01580, 0.02340, 0.03600, 0.05900, 0.10100, 0.17100, 0.27200, 0.38500,
    0.47800, 0.55000, 0.65000, 1.00000,
])
QX_FEM_ANCH = np.array([
    0.00050, 0.00071, 0.00108, 0.00161, 0.00237, 0.00347, 0.00491, 0.00710,
    0.01030, 0.01580, 0.02550, 0.04450, 0.07900, 0.14100, 0.23300, 0.34000,
    0.47800, 0.55000, 0.65000, 1.00000,
])

_AGES_FINE = np.arange(25, 120)
QX_MALE   = np.exp(np.interp(_AGES_FINE, ANCHOR_AGES, np.log(QX_MALE_ANCH)))
QX_FEMALE = np.exp(np.interp(_AGES_FINE, ANCHOR_AGES, np.log(QX_FEM_ANCH)))
QX_MALE[-1] = QX_FEMALE[-1] = 1.0   # force certain death at age 119


def draw_death_months(qx_annual: np.ndarray, n_paths: int, seed: int) -> np.ndarray:
    """Draw stochastic month-of-death per path. (Notebook Cell 14, verbatim.)"""
    rng   = np.random.default_rng(seed)
    n_age = len(qx_annual)
    u     = rng.random(size=(n_paths, n_age))
    dies  = u < qx_annual[None, :]
    death_age_idx = dies.argmax(axis=1)
    month_within  = rng.integers(0, 12, size=n_paths)
    return death_age_idx * 12 + month_within


def couple_death_months(n_paths: int, seed: int = SEED, n_months: int = N_MONTHS):
    """Return (first_death_month, last_death_month) clipped to n_months-1."""
    male   = draw_death_months(QX_MALE,   n_paths, seed=seed)
    female = draw_death_months(QX_FEMALE, n_paths, seed=seed + 1)
    first_death = np.minimum(male, female).clip(max=n_months - 1)
    last_death  = np.maximum(male, female).clip(max=n_months - 1)
    return first_death, last_death

# ============================================================================
# BOOTSTRAP — IID only for this build (notebook Cell 8, FULL_PATHS removed)
# ============================================================================
def iid_bootstrap(returns_arr: np.ndarray,
                  n_paths: int,
                  n_months: int = N_MONTHS,
                  seed: int = SEED) -> np.ndarray:
    """IID bootstrap. returns_arr shape (T, k); output shape (n_paths, n_months, k)."""
    rng = np.random.default_rng(seed + 1)
    T   = returns_arr.shape[0]
    idx = rng.integers(0, T, size=(n_paths, n_months))
    return returns_arr[idx]

# ============================================================================
# BOND CONSTRUCTION — used by data_loader; lives here so simulation.py owns
# all of the pricing math (Notebook Cell 5)
# ============================================================================
def macaulay_duration(ytm_pct: float, T: int = 10, freq: int = 2) -> float:
    """Macaulay duration of a par-coupon bond at given YTM (in percent)."""
    ytm = ytm_pct / 100.0
    c   = ytm / freq
    n   = int(T * freq)
    if c == 0:
        return float(T)
    times     = np.arange(1, n + 1) / freq
    cashflows = np.full(n, c); cashflows[-1] += 1
    discount  = (1 + c) ** -np.arange(1, n + 1)
    pv        = cashflows * discount
    return float(np.sum(times * pv) / np.sum(pv))


def build_bond_real(yields_s: pd.Series, cpi_s: pd.Series) -> pd.Series:
    """Monthly real bond returns from a yield series + CPI level series.

    Both series should have a PeriodIndex[M] (or DatetimeIndex convertible to one).
    """
    y_idx = (yields_s.index if isinstance(yields_s.index, pd.PeriodIndex)
             else pd.to_datetime(yields_s.index).to_period('M'))
    c_idx = (cpi_s.index if isinstance(cpi_s.index, pd.PeriodIndex)
             else pd.to_datetime(cpi_s.index).to_period('M'))
    y = pd.Series(yields_s.values, index=y_idx).sort_index()
    c = pd.Series(cpi_s.values,    index=c_idx).sort_index()
    y = y[~y.index.duplicated(keep='last')]
    c = c[~c.index.duplicated(keep='last')]

    common = y.index.intersection(c.index).sort_values()
    y, c = y.loc[common], c.loc[common]
    y_arr, c_arr = y.values.astype(float), c.values.astype(float)

    nominal = np.empty(len(y_arr) - 1)
    for t in range(1, len(y_arr)):
        D_mac  = macaulay_duration(y_arr[t-1])
        D_mod  = D_mac / (1 + y_arr[t-1] / 100.0 / 2.0)
        income = y_arr[t-1] / 100.0 / 12.0
        cgain  = -D_mod * ((y_arr[t] - y_arr[t-1]) / 100.0)
        nominal[t-1] = income + cgain

    cpi_infl = np.diff(c_arr) / c_arr[:-1]
    real     = (1 + nominal) / (1 + cpi_infl) - 1.0
    return pd.Series(real, index=common[1:], name='bond_real')

# ============================================================================
# LIFECYCLE SIMULATION — adapted from notebook Cell 16 to take explicit args
# ============================================================================
def simulate_lifecycle(paths: np.ndarray,
                       w_eq_schedule: np.ndarray,
                       last_death_month: np.ndarray,
                       ss_monthly: float = SS_FLOOR_MONTHLY,
                       annual_income: float = ANNUAL_INCOME,
                       contrib_rate: float = CONTRIB_RATE,
                       withdraw_rate_ann: float = WITHDRAW_RATE_ANN,
                       accum_months: int = ACCUM_MONTHS,
                       drawdown_max: int = DRAWDOWN_MAX) -> dict:
    """
    Simulate accumulation + drawdown for n_paths households.

    Parameters
    ----------
    paths : (n_paths, n_months, 2) bootstrapped real returns [equity, bond]
    w_eq_schedule : (n_months,) equity weight per month
    last_death_month : (n_paths,) absolute month index of last-survivor death
    ss_monthly : Social Security floor (real $/mo). Acts as both a retirement
                 income floor and the post-ruin consumption level.

    Returns dict identical to notebook Cell 16.
    """
    n_paths = paths.shape[0]
    monthly_contrib = annual_income * contrib_rate / 12.0

    # ── Accumulation ────────────────────────────────────────────────────────
    wealth = np.zeros(n_paths, dtype=np.float64)
    for t in range(accum_months):
        r_eq   = paths[:, t, 0]
        r_bd   = paths[:, t, 1]
        w_eq   = w_eq_schedule[t]
        r_port = w_eq * r_eq + (1.0 - w_eq) * r_bd
        wealth = (wealth + monthly_contrib) * (1.0 + r_port)

    accumulated_wealth = wealth.copy()
    died_pre_retire    = last_death_month <= accum_months
    wealth_at_retire   = np.where(died_pre_retire, 0.0, wealth)

    # ── Drawdown ────────────────────────────────────────────────────────────
    monthly_withdraw   = wealth_at_retire * withdraw_rate_ann / 12.0
    ruin_flag          = np.zeros(n_paths, dtype=bool)
    ruin_month         = np.full(n_paths, -1, dtype=np.int32)
    alive              = ~died_pre_retire
    wealth             = np.where(alive, wealth_at_retire, 0.0).astype(np.float64)
    consumption_stream = np.zeros((n_paths, drawdown_max), dtype=np.float64)

    for t_draw in range(drawdown_max):
        t_abs  = accum_months + t_draw
        r_eq   = paths[:, t_abs, 0]
        r_bd   = paths[:, t_abs, 1]
        w_eq   = w_eq_schedule[t_abs]
        r_port = w_eq * r_eq + (1.0 - w_eq) * r_bd

        # Detect ruin this month
        cannot_withdraw = alive & (wealth < monthly_withdraw) & (~ruin_flag)
        ruin_flag      |= cannot_withdraw
        ruin_month      = np.where(cannot_withdraw & (ruin_month == -1), t_abs, ruin_month)

        solvent      = alive & (~ruin_flag)
        ruined_alive = alive &  ruin_flag
        cm  = np.where(solvent,      monthly_withdraw + ss_monthly, 0.0)
        cm  = np.where(ruined_alive, ss_monthly,                    cm)
        consumption_stream[:, t_draw] = cm

        # Solvent paths: withdraw, then grow
        wealth = np.where(solvent, (wealth - monthly_withdraw) * (1.0 + r_port), wealth)
        wealth = np.where(ruin_flag, 0.0, wealth)

        # Mortality this month
        just_died = alive & (last_death_month == t_abs)
        alive     = alive & ~just_died

    return {
        "terminal_wealth":    wealth,
        "ruin_flag":          ruin_flag,
        "ruin_month":         ruin_month,
        "wealth_at_retire":   wealth_at_retire,
        "monthly_withdraw":   monthly_withdraw,
        "consumption_stream": consumption_stream,
        "accumulated_wealth": accumulated_wealth,
        "died_pre_retire":    died_pre_retire,
    }


def bequest_wealth(outcome: dict) -> np.ndarray:
    """Bequest per path: accumulated_wealth if died pre-retire, else terminal_wealth."""
    return np.where(outcome["died_pre_retire"],
                    outcome["accumulated_wealth"],
                    outcome["terminal_wealth"])


def build_H_t(first_death_month: np.ndarray,
              last_death_month: np.ndarray,
              accum_months: int = ACCUM_MONTHS,
              drawdown_max: int = DRAWDOWN_MAX) -> np.ndarray:
    """Equivalence-scale matrix: H=2 before first death, 1 between deaths, 0 after."""
    t_abs_draw = accum_months + np.arange(drawdown_max)
    H = np.where(t_abs_draw[None, :] < first_death_month[:, None], 2,
         np.where(t_abs_draw[None, :] < last_death_month[:, None], 1, 0)).astype(np.float64)
    return H


def crra_expected_utility(consumption_stream: np.ndarray,
                          W_bequest: np.ndarray,
                          H: np.ndarray,
                          gamma: float = GAMMA_BASE,
                          delta: float = DELTA_DISC) -> float:
    """Mean CRRA utility across paths. (Notebook Step 4 Cell 1.)"""
    one_minus_g = 1.0 - gamma
    alive = H > 0
    H_s   = np.where(alive, H, 1.0)
    C_s   = np.where(alive, consumption_stream, 1.0)
    per_cap = C_s / np.sqrt(H_s)
    flow = H_s * (per_cap ** one_minus_g) / one_minus_g
    flow = np.where(alive, flow, 0.0)
    if delta != 1.0:
        disc = delta ** np.arange(consumption_stream.shape[1], dtype=np.float64)
        flow = flow * disc[None, :]
    flow_total = flow.sum(axis=1)
    bequest = theta_of(gamma) * (W_bequest / 12.0 + K_BEQUEST) ** one_minus_g / one_minus_g
    return float((flow_total + bequest).mean())

# ============================================================================
# ORCHESTRATOR — the only new function. Live computation on user-selected ticker.
# ============================================================================
def _quadratic_peak(weights: np.ndarray, eu: np.ndarray) -> tuple[float, float]:
    """Fit y = a x² + b x + c and return (x*, y*) of the parabola peak,
    clipped to [0, 1]. Falls back to the discrete argmax if the fit isn't concave.
    """
    a, b, c = np.polyfit(weights, eu, 2)
    if a >= 0:  # not concave — quadratic doesn't have a maximum
        i = int(np.argmax(eu))
        return float(weights[i]), float(eu[i])
    x_star = -b / (2 * a)
    x_star = float(np.clip(x_star, weights.min(), weights.max()))
    y_star = float(a * x_star**2 + b * x_star + c)
    return x_star, y_star


def run_crra_grid(equity_returns: pd.Series,
                  bond_returns: pd.Series,
                  n_paths: int = 5_000,
                  weight_grid: list | np.ndarray | None = None,
                  gamma: float = GAMMA_BASE,
                  ss_monthly: float = SS_FLOOR_MONTHLY,
                  seed: int = SEED,
                  death_months: tuple[np.ndarray, np.ndarray] | None = None,
                  ) -> dict:
    """
    Run the full CRRA optimization for a single ticker, across a coarse weight grid.

    Live-app default: 6-point grid + quadratic fit gives a continuous w*.
    Precompute use:   pass a fine 21-point grid via weight_grid for the baseline.

    Parameters
    ----------
    equity_returns, bond_returns : pd.Series with PeriodIndex[M]
    n_paths       : bootstrap sample size
    weight_grid   : list of equity weights to evaluate (default 6-pt)
    gamma         : CRRA risk aversion
    ss_monthly    : Social Security floor in real $/mo
    death_months  : optional precomputed (first, last) tuple. If provided,
                    skips drawing and reuses the same death realizations
                    across calls (reduces variance across grid points).

    Returns dict ready for json/npz serialization with:
        weights      : (G,)  list of grid weights evaluated
        eu           : (G,)  expected CRRA utility per weight
        ruin         : (G,)  ruin probability per weight (0–1)
        median_tw    : (G,)  median terminal wealth per weight
        optimal_w    : float continuous w* from quadratic fit
        optimal_eu   : float E[U] at w*
        gamma        : float
        ss_monthly   : float
        n_paths      : int
        equity_n     : int historical observations used
        equity_start : str  PeriodIndex[M] start as 'YYYY-MM'
        equity_end   : str  PeriodIndex[M] end   as 'YYYY-MM'
    """
    if weight_grid is None:
        weight_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    weight_grid = np.asarray(weight_grid, dtype=np.float64)

    # ── Align inputs on a common PeriodIndex[M] ────────────────────────────
    if not isinstance(equity_returns.index, pd.PeriodIndex):
        equity_returns = equity_returns.copy()
        equity_returns.index = pd.to_datetime(equity_returns.index).to_period('M')
    if not isinstance(bond_returns.index, pd.PeriodIndex):
        bond_returns = bond_returns.copy()
        bond_returns.index = pd.to_datetime(bond_returns.index).to_period('M')

    common = equity_returns.index.intersection(bond_returns.index).sort_values()
    if len(common) < 60:
        raise ValueError(
            f"Only {len(common)} overlapping monthly observations between equity and bonds — "
            "need at least 60 (5 years) to bootstrap meaningfully."
        )
    eq = equity_returns.loc[common].values.astype(np.float64)
    bd = bond_returns.loc[common].values.astype(np.float64)
    returns_arr = np.column_stack([eq, bd])  # (T, 2)

    # ── Bootstrap (IID, common-random-numbers across grid) ─────────────────
    paths = iid_bootstrap(returns_arr, n_paths=n_paths, seed=seed)

    # ── Mortality ──────────────────────────────────────────────────────────
    if death_months is None:
        first_death, last_death = couple_death_months(n_paths, seed=seed)
    else:
        first_death, last_death = death_months
        first_death = first_death[:n_paths]
        last_death  = last_death[:n_paths]

    H = build_H_t(first_death, last_death)

    # ── Sweep the weight grid ──────────────────────────────────────────────
    eu_arr        = np.empty(len(weight_grid), dtype=np.float64)
    ruin_arr      = np.empty(len(weight_grid), dtype=np.float64)
    median_tw_arr = np.empty(len(weight_grid), dtype=np.float64)

    for i, w in enumerate(weight_grid):
        sched = np.full(N_MONTHS, w)
        out   = simulate_lifecycle(paths, sched, last_death, ss_monthly=ss_monthly)
        W_beq = bequest_wealth(out)
        eu_arr[i]        = crra_expected_utility(out["consumption_stream"], W_beq, H, gamma=gamma)
        ruin_arr[i]      = float(out["ruin_flag"].mean())
        median_tw_arr[i] = float(np.median(out["terminal_wealth"]))

    # ── Continuous optimum via quadratic fit ───────────────────────────────
    optimal_w, optimal_eu = _quadratic_peak(weight_grid, eu_arr)

    return {
        "weights":      weight_grid,
        "eu":           eu_arr,
        "ruin":         ruin_arr,
        "median_tw":    median_tw_arr,
        "optimal_w":    optimal_w,
        "optimal_eu":   optimal_eu,
        "gamma":        float(gamma),
        "ss_monthly":   float(ss_monthly),
        "n_paths":      int(n_paths),
        "equity_n":     int(len(common)),
        "equity_start": str(common[0]),
        "equity_end":   str(common[-1]),
    }
