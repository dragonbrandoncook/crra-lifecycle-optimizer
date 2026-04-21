"""
Data fetchers for the CRRA Lifecycle Optimizer.

Pure functions — no Streamlit decorators here. The app.py module wraps these
with @st.cache_data so precompute.py can import the same functions without
needing Streamlit installed at compute time.

Three sources:
    1. Shiller Yale `ie_data.xls`           -> monthly CPI level for deflation
    2. yfinance `Adj Close` for any ticker  -> monthly nominal equity returns
    3. yfinance `^TNX` (10-yr UST yield)    -> monthly real bond returns
                                                via Macaulay-duration construction

CPI deflation uses the Fisher identity:  real = (1 + nom) / (1 + cpi_pct) - 1
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from simulation import build_bond_real

SHILLER_URL      = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
SHILLER_SNAPSHOT = Path(__file__).parent / "precomputed" / "shiller_cpi_snapshot.xls"

# ============================================================================
# Shiller CPI (notebook Cells 3 & 4)
# ============================================================================
def load_shiller_cpi(prefer_snapshot: bool = False) -> pd.Series:
    """Return monthly CPI levels indexed by Period[M] (name='cpi').

    Tries the live Shiller URL first; if it fails (or `prefer_snapshot=True`),
    falls back to the snapshot at precomputed/shiller_cpi_snapshot.xls.
    Raises FileNotFoundError if neither source works.
    """
    raw_bytes = None
    if not prefer_snapshot:
        try:
            resp = requests.get(SHILLER_URL, timeout=30)
            resp.raise_for_status()
            raw_bytes = resp.content
        except Exception:
            raw_bytes = None

    if raw_bytes is None:
        if SHILLER_SNAPSHOT.exists():
            raw_bytes = SHILLER_SNAPSHOT.read_bytes()
        else:
            raise FileNotFoundError(
                f"Could not fetch Shiller data from {SHILLER_URL} and no snapshot at "
                f"{SHILLER_SNAPSHOT}. Run precompute.py with network access to create "
                "the snapshot."
            )

    df = pd.read_excel(
        io.BytesIO(raw_bytes),
        sheet_name="Data",
        skiprows=7,
        engine="xlrd",
    )
    df = df.dropna(axis=1, how='all')

    # Find columns by name (column order in the Shiller file shifts year-to-year)
    date_col = df.columns[0]
    cpi_col  = next(c for c in df.columns if 'CPI' in str(c).upper())

    df = df[[date_col, cpi_col]].copy()
    df.columns = ['date_raw', 'cpi']
    df = df[pd.to_numeric(df['date_raw'], errors='coerce').notna()]
    df['date_raw'] = df['date_raw'].astype(float)

    # Shiller date format YYYY.MM (e.g., 1926.10 = Oct 1926)
    def _shiller_date(x: float) -> pd.Timestamp:
        year  = int(x)
        month = int(round((x - year) * 100))
        if month < 1 or month > 12:
            month = max(1, min(12, month))
        return pd.Timestamp(year=year, month=month, day=1)

    df['date'] = df['date_raw'].apply(_shiller_date)
    df = df.set_index('date').sort_index()
    df = df[df['cpi'].notna()]

    cpi = df['cpi'].astype(float)
    cpi.index = pd.to_datetime(cpi.index).to_period('M')
    cpi.name = 'cpi'

    # Try to write the snapshot for next time (best-effort, ignore failures)
    if not SHILLER_SNAPSHOT.exists() and raw_bytes is not None:
        try:
            SHILLER_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            SHILLER_SNAPSHOT.write_bytes(raw_bytes)
        except Exception:
            pass

    return cpi

# ============================================================================
# Equity returns (yfinance Adj Close, deflated by Shiller CPI)
# ============================================================================
def load_equity_returns(ticker: str,
                        cpi: pd.Series | None = None,
                        start: str = "1993-01-01",
                        end: str | None = None) -> pd.Series:
    """Real monthly returns for any yfinance ticker, indexed by Period[M].

    Uses Adj Close pct_change as a total-return proxy. For ETFs this captures
    distributions; for individual stocks it does not include cash dividends.
    """
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"yfinance returned no data for ticker '{ticker}'.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if "Adj Close" not in raw.columns:
        raise ValueError(f"yfinance data for '{ticker}' has no 'Adj Close' column.")

    nom = raw["Adj Close"].pct_change().dropna()
    nom.index = pd.to_datetime(nom.index).to_period('M')
    nom = nom[~nom.index.duplicated(keep='last')]
    nom.name = ticker

    if cpi is None:
        cpi = load_shiller_cpi()
    cpi_pct = cpi.pct_change().dropna()
    if not isinstance(cpi_pct.index, pd.PeriodIndex):
        cpi_pct.index = pd.to_datetime(cpi_pct.index).to_period('M')

    common = nom.index.intersection(cpi_pct.index)
    if len(common) < 60:
        raise ValueError(
            f"Only {len(common)} overlapping months between {ticker} and CPI — "
            "need at least 60 (5 years)."
        )

    real = (1.0 + nom.loc[common]) / (1.0 + cpi_pct.loc[common]) - 1.0
    real.name = ticker
    return real

# ============================================================================
# Bond returns (10-yr Treasury yield via ^TNX -> Macaulay duration -> deflate)
# ============================================================================
def load_bond_returns(cpi: pd.Series | None = None,
                      start: str = "1993-01-01",
                      end: str | None = None) -> pd.Series:
    """Monthly real returns of a 10-yr Treasury position via duration math."""
    raw = yf.download(
        "^TNX",
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError("yfinance returned no data for ^TNX.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    yield_pct = raw["Close"].dropna()  # ^TNX 'Close' is the yield in percent
    yield_pct.index = pd.to_datetime(yield_pct.index).to_period('M')
    yield_pct = yield_pct[~yield_pct.index.duplicated(keep='last')]
    yield_pct.name = 'gs10'

    if cpi is None:
        cpi = load_shiller_cpi()

    real = build_bond_real(yield_pct, cpi)
    real.name = 'bond_real'
    return real
