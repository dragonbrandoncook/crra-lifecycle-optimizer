"""
Streamlit entry point for the CRRA Lifecycle Optimizer.

UI only — all numerics live in simulation.py and data_loader.py. Layout
mirrors crra_ui_mockup.jsx as closely as Streamlit's component model allows.
Streamlit's CSS reach is limited; structural parity > pixel parity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_equity_returns, load_bond_returns, load_shiller_cpi
from simulation import (
    GAMMA_BASE,
    SS_FLOOR_MONTHLY,
    run_crra_grid,
)

# ============================================================================
# Page config — must be first Streamlit call
# ============================================================================
st.set_page_config(
    page_title="CRRA Lifecycle Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRECOMPUTED_DIR = Path(__file__).parent / "precomputed"

GAMMA_PRESETS = [
    {"id": "agg", "label": "Aggressive",   "gamma": 2.0,  "color": "#FFC72C", "file": "spy_baseline_g2.npz"},
    {"id": "mod", "label": "Moderate",     "gamma": 3.84, "color": "#8E1B34", "file": "spy_baseline_g3p84.npz"},
    {"id": "con", "label": "Conservative", "gamma": 7.0,  "color": "#AC9155", "file": "spy_baseline_g7.npz"},
]

ASSET_PRESETS = [
    ("SPY",  "S&P 500 ETF",   "#FFC72C"),
    ("QQQ",  "NASDAQ ETF",    "#8E1B34"),
    ("VTI",  "Total Market",  "#AC9155"),
    ("AAPL", "Apple Inc.",    "#E97A26"),
    ("TLT",  "20yr Treasury", "#5E1219"),
    ("VXUS", "Intl. ex-US",   "#E0C56E"),
]
ASSET_NAMES  = {t: n for t, n, _ in ASSET_PRESETS}
ASSET_COLORS = {t: c for t, _, c in ASSET_PRESETS}

# ============================================================================
# Custom CSS — closer to mockup feel
# ============================================================================
st.markdown("""
<style>
    /* Light main panel + maroon sidebar */
    .stApp { background: #FFFFFF; }
    section[data-testid="stSidebar"] > div { background: #2C0F12; }
    /* Force light text inside the maroon sidebar (textColor is now dark globally) */
    section[data-testid="stSidebar"] * { color: #F1F5F9; }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #E0C56E !important; }
    /* Sidebar buttons — keep them maroon-themed instead of cream */
    section[data-testid="stSidebar"] .stButton > button {
        background: #4A1318 !important;
        color: #F1F5F9 !important;
        border: 1px solid #6B1B22 !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #6B1B22 !important;
        border-color: #FFC72C !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #8E1B34 !important;
        border: 1px solid #FFC72C !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background: #A52440 !important;
    }
    /* Sidebar text input (Custom ticker) */
    section[data-testid="stSidebar"] input[type="text"],
    section[data-testid="stSidebar"] textarea {
        background: #4A1318 !important;
        color: #F1F5F9 !important;
        border: 1px solid #6B1B22 !important;
    }
    section[data-testid="stSidebar"] input[type="text"]::placeholder {
        color: #B89B6E !important;
    }
    /* Tighter buttons */
    .stButton > button {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        border-radius: 5px;
        font-size: 11px;
        font-weight: 600;
    }
    /* Metric card look */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(44,15,18,0.95), rgba(20,5,7,0.6));
        border: 1px solid #3D1116;
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 9px !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700; }
    /* Section headers — maroon for the light main area */
    h1, h2, h3 { color: #501214; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #FFC72C; }
    /* Tab labels — maroon on the light main panel */
    .stTabs [data-baseweb="tab-list"] button p,
    .stTabs [data-baseweb="tab-list"] button span,
    button[role="tab"] p, button[role="tab"] span {
        color: #501214 !important;
        font-weight: 700 !important;
    }
    button[role="tab"][aria-selected="true"] p,
    button[role="tab"][aria-selected="true"] span {
        color: #8E1B34 !important;
    }
    /* Code/mono accents */
    code { color: #FFC72C; background: rgba(255,199,44,0.10); }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session state
# ============================================================================
def _init_state() -> None:
    defaults = {
        "ticker":         "SPY",
        "gamma_preset":   "mod",
        "ss_monthly":     int(SS_FLOOR_MONTHLY),
        "live_results":   None,
        "live_ticker":    None,
        "live_gamma":     None,
        "live_ss":        None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
_init_state()

# ============================================================================
# Cached data layer
# ============================================================================
@st.cache_data(ttl=86400, show_spinner=False)
def cached_cpi() -> pd.Series:
    return load_shiller_cpi()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_equity(ticker: str) -> pd.Series:
    return load_equity_returns(ticker, cpi=cached_cpi())

@st.cache_data(ttl=3600, show_spinner=False)
def cached_bonds() -> pd.Series:
    return load_bond_returns(cpi=cached_cpi())

@st.cache_data(show_spinner=False)
def cached_baseline(file_name: str) -> dict:
    path = PRECOMPUTED_DIR / file_name
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}

@st.cache_data(show_spinner=False)
def cached_metadata() -> dict | None:
    path = PRECOMPUTED_DIR / "metadata.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def cached_death_5k() -> tuple[np.ndarray, np.ndarray] | None:
    path = PRECOMPUTED_DIR / "death_months_5k.npy"
    if not path.exists():
        return None
    arr = np.load(path)
    return arr[0], arr[1]

# ============================================================================
# Helpers
# ============================================================================
def fmt_money(n: float) -> str:
    if n >= 1e6: return f"${n/1e6:.2f}M"
    if n >= 1e3: return f"${n/1e3:.0f}K"
    return f"${n:.0f}"

def get_preset(pid: str) -> dict:
    return next(p for p in GAMMA_PRESETS if p["id"] == pid)

def baseline_metrics(baseline: dict) -> dict:
    """Pull the optimal-w slice from a precomputed baseline dict."""
    weights = np.asarray(baseline["weights"])
    w_star  = float(baseline["optimal_w"])
    eu      = float(baseline["optimal_eu"])
    ruin    = float(np.interp(w_star, weights, baseline["ruin"]))
    medtw   = float(np.interp(w_star, weights, baseline["median_tw"]))
    return {"opt_w": w_star, "ruin": ruin, "med_tw": medtw, "eu": eu}

def live_metrics(result: dict) -> dict:
    weights = np.asarray(result["weights"])
    w_star  = float(result["optimal_w"])
    eu      = float(result["optimal_eu"])
    ruin    = float(np.interp(w_star, weights, result["ruin"]))
    medtw   = float(np.interp(w_star, weights, result["median_tw"]))
    return {"opt_w": w_star, "ruin": ruin, "med_tw": medtw, "eu": eu}

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">'
        '<div style="width:26px;height:26px;border-radius:6px;'
        'background:linear-gradient(135deg,#501214,#FFC72C);'
        'display:flex;align-items:center;justify-content:center;'
        'box-shadow:0 0 12px rgba(56,189,248,0.4);'
        'color:#F8FAFC;font-weight:800;font-size:13px;">γ</div>'
        '<div>'
        '<div style="color:#F8FAFC;font-weight:700;font-size:12px;letter-spacing:0.1em;">CRRA OPTIMIZER</div>'
        '<div style="color:#64748B;font-size:9px;letter-spacing:0.15em;">LIFECYCLE ALLOCATION ENGINE</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Asset selector ─────────────────────────────────────────────────────
    st.markdown('<div style="font-size:9px;color:#64748B;letter-spacing:0.15em;margin-bottom:8px;">◆ EQUITY ASSET</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (t, _, _) in enumerate(ASSET_PRESETS):
        if cols[i % 2].button(t, key=f"tk_{t}", use_container_width=True,
                              type="primary" if st.session_state.ticker == t else "secondary"):
            st.session_state.ticker = t
            st.session_state.live_results = None
    custom = st.text_input("Custom ticker", value="", placeholder="Custom ticker...",
                           label_visibility="collapsed")
    if custom:
        st.session_state.ticker = custom.upper().strip()
        st.session_state.live_results = None
    st.caption(f"Active: **{st.session_state.ticker}** — {ASSET_NAMES.get(st.session_state.ticker, 'Custom')}")

    st.divider()

    # ── Risk profile ───────────────────────────────────────────────────────
    st.markdown('<div style="font-size:9px;color:#64748B;letter-spacing:0.15em;margin-bottom:8px;">◐ RISK PROFILE (γ)</div>', unsafe_allow_html=True)
    g_cols = st.columns(3)
    for i, p in enumerate(GAMMA_PRESETS):
        label = f"{p['label']}\nγ={p['gamma']}"
        if g_cols[i].button(label, key=f"g_{p['id']}", use_container_width=True,
                            type="primary" if st.session_state.gamma_preset == p["id"] else "secondary"):
            st.session_state.gamma_preset = p["id"]
            st.session_state.live_results = None
    st.caption("Snaps to nearest precomputed baseline.")

    st.divider()

    # ── SS slider ──────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:9px;color:#64748B;letter-spacing:0.15em;margin-bottom:4px;">● SOCIAL SECURITY FLOOR</div>', unsafe_allow_html=True)
    ss = st.slider("Real $/month",
                   min_value=0, max_value=3000, value=st.session_state.ss_monthly,
                   step=50, label_visibility="collapsed")
    if ss != st.session_state.ss_monthly:
        st.session_state.ss_monthly = ss
        st.session_state.live_results = None
    ss_changed = st.session_state.ss_monthly != int(SS_FLOOR_MONTHLY)
    st.caption(f"**${st.session_state.ss_monthly:,}/mo** "
               + ("(default $1,261)" if not ss_changed
                  else "⚠ Baseline locked at $1,261; slider applies to live ticker only."))

    st.divider()

    # ── Active params readout ──────────────────────────────────────────────
    p = get_preset(st.session_state.gamma_preset)
    theta_val = 2360 * (12 ** p["gamma"])
    st.markdown(
        f"""<div style="background:rgba(44,15,18,0.55);border:1px solid #3D1116;border-radius:6px;padding:10px 12px;font-size:10px;line-height:1.7;color:#94A3B8;">
        <div style="color:#475569;letter-spacing:0.12em;font-size:9px;margin-bottom:6px;">ACTIVE PARAMS</div>
        γ = <span style="color:{p['color']};font-weight:700">{p['gamma']:.2f}</span><br>
        θ = {theta_val:.2e}<br>
        k = $490K<br>
        SS = <span style="color:{'#FFC72C' if ss_changed else '#94A3B8'};font-weight:{700 if ss_changed else 400}">${st.session_state.ss_monthly:,}/mo</span><br>
        δ = 1.000<br>
        n = 5,000 paths (live)
        </div>""",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Run button ─────────────────────────────────────────────────────────
    run_btn = st.button("▶  RUN OPTIMIZATION", type="primary", use_container_width=True)
    st.caption("~30–60 sec · 5k paths · 6 weights + quadratic fit")

# ============================================================================
# Main panel
# ============================================================================
gamma_preset = get_preset(st.session_state.gamma_preset)
gamma        = gamma_preset["gamma"]
ticker       = st.session_state.ticker
ticker_color = ASSET_COLORS.get(ticker, "#FFC72C")
ss_monthly   = float(st.session_state.ss_monthly)

# Trigger compute
if run_btn:
    try:
        with st.spinner(f"Running CRRA optimization for {ticker}... (~30–60 sec)"):
            cpi = cached_cpi()
            eq  = cached_equity(ticker)
            bd  = cached_bonds()
            deaths = cached_death_5k()
            n_live = int(deaths[0].shape[0])
            result = run_crra_grid(
                equity_returns = eq,
                bond_returns   = bd,
                n_paths        = n_live,
                gamma          = gamma,
                ss_monthly     = ss_monthly,
                death_months   = deaths,
            )
        st.session_state.live_results = result
        st.session_state.live_ticker  = ticker
        st.session_state.live_gamma   = gamma
        st.session_state.live_ss      = ss_monthly
    except Exception as e:
        st.error(f"Run failed: {e}")

# Load baseline for current γ preset
baseline = cached_baseline(gamma_preset["file"])
metadata = cached_metadata()

# ── Header ────────────────────────────────────────────────────────────────
hdr_l, hdr_r = st.columns([3, 1])
with hdr_l:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:12px;">'
        '<div style="color:#501214;font-weight:800;font-size:18px;letter-spacing:0.08em;">CRRA LIFECYCLE OPTIMIZER</div>'
        '<div style="color:#8E1B34;font-size:10px;letter-spacing:0.15em;">WELFARE-MAXIMIZING ALLOCATION ENGINE</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with hdr_r:
    st.markdown(
        '<div style="text-align:right;font-size:10px;color:#8E1B34;font-weight:700;letter-spacing:0.08em;">'
        'QFE5315 · TEXAS STATE UNIVERSITY'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_results, tab_surface, tab_explainer = st.tabs([
    "📊 Results",
    "🗺 Utility Surface",
    "📖 How It Works",
])

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — RESULTS
# ──────────────────────────────────────────────────────────────────────────
with tab_results:
    st.warning(
        "⚠ Live runs use 5,000 IID-bootstrap paths (vs 50,000 for SPY baseline). "
        "Equity = yfinance Adj Close. CPI deflation via Shiller Yale series. "
        "Demo only — not investment advice.",
        icon=None,
    )

    if baseline is None:
        st.error(
            f"Baseline file `{gamma_preset['file']}` not found in precomputed/. "
            "Run `python precompute.py` locally first, then commit the precomputed/ "
            "directory to the deployment repo."
        )
    else:
        b = baseline_metrics(baseline)

        if st.session_state.live_results is None:
            # Pre-run state — show baseline only
            st.markdown(
                f'<div style="font-size:11px;color:#64748B;letter-spacing:0.1em;margin:8px 0 14px;">'
                f'SPY BASELINE · 50,000 PATHS · γ = {gamma:.2f}'
                f'</div>',
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Optimal Eq. Weight",  f"{b['opt_w']*100:.1f}%")
            c2.metric("Ruin Probability",    f"{b['ruin']*100:.2f}%")
            c3.metric("Median Term. Wealth", fmt_money(b['med_tw']))
            c4.metric("E[U] Score",          f"{b['eu']:,.0f}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.info("▶  Select an asset and click **RUN OPTIMIZATION** to compare a live ticker against the baseline.")
        else:
            # Post-run — comparison
            r = st.session_state.live_results
            l = live_metrics(r)
            live_t  = st.session_state.live_ticker
            live_g  = st.session_state.live_gamma
            live_ss = st.session_state.live_ss
            ss_changed = live_ss != float(SS_FLOOR_MONTHLY)

            col_l, col_mid, col_r = st.columns([10, 1, 10])
            with col_l:
                st.markdown(f"#### SPY BASELINE")
                st.caption(f"50k paths · γ-locked · SS=$1,261")
                st.metric("Optimal Equity Weight",    f"{b['opt_w']*100:.1f}%")
                st.metric("Ruin Probability",         f"{b['ruin']*100:.2f}%")
                st.metric("Median Terminal Wealth",   fmt_money(b['med_tw']))
                st.metric("E[U] Score",               f"{b['eu']:,.0f}")
            with col_mid:
                st.markdown('<div style="text-align:center;color:#475569;font-weight:700;letter-spacing:0.2em;padding-top:80px;">VS</div>', unsafe_allow_html=True)
            with col_r:
                tag = f"5k paths · γ={live_g:.2f}" + (f" · SS=${int(live_ss)}" if ss_changed else "")
                st.markdown(f"#### {live_t} — {ASSET_NAMES.get(live_t, 'Custom')}")
                st.caption(tag)
                st.metric(
                    "Optimal Equity Weight",
                    f"{l['opt_w']*100:.1f}%",
                    delta=f"{(l['opt_w']-b['opt_w'])*100:+.1f} pp",
                    delta_color="normal",
                )
                st.metric(
                    "Ruin Probability",
                    f"{l['ruin']*100:.2f}%",
                    delta=f"{(l['ruin']-b['ruin'])*100:+.2f} pp",
                    delta_color="inverse",
                )
                st.metric(
                    "Median Terminal Wealth",
                    fmt_money(l['med_tw']),
                    delta=f"{(l['med_tw']-b['med_tw'])/abs(b['med_tw'])*100:+.1f}%",
                    delta_color="normal",
                )
                st.metric(
                    "E[U] Score",
                    f"{l['eu']:,.0f}",
                    delta=f"{(l['eu']-b['eu'])/abs(b['eu'])*100:+.2f}%",
                    delta_color="normal",
                )

            # Interpretation
            direction = "down" if b['opt_w'] > l['opt_w'] else "up"
            vol_note  = ("higher volatility increasing lifecycle ruin risk, prompting CRRA to reduce equity exposure"
                         if l['ruin'] > b['ruin']
                         else "similar or lower volatility profile, allowing CRRA to maintain or increase equity exposure")
            ss_note   = (f" A higher SS floor (${int(live_ss):,}/mo) reduces the consequence of portfolio shortfalls — typically pushes optimal weight upward."
                         if ss_changed else "")
            st.markdown(
                f"""<div style="margin-top:16px;background:linear-gradient(135deg,rgba(20,5,7,0.7),rgba(20,5,7,0.4));
                border:1px solid #3D1116;border-left:3px solid {ticker_color};border-radius:6px;padding:14px 18px;
                font-size:12px;color:#94A3B8;line-height:1.7;">
                <div style="color:#E2E8F0;font-weight:700;font-size:11px;margin-bottom:6px;letter-spacing:0.05em;">INTERPRETATION</div>
                Optimal equity weight shifted <strong>{direction}</strong> from
                <strong style="color:#FFC72C">{b['opt_w']*100:.1f}%</strong> (SPY) to
                <strong style="color:{ticker_color}">{l['opt_w']*100:.1f}%</strong> ({live_t}).
                This reflects {live_t}'s {vol_note}.{ss_note}
                </div>""",
                unsafe_allow_html=True,
            )

# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — UTILITY SURFACE
# ──────────────────────────────────────────────────────────────────────────
with tab_surface:
    st.markdown(
        '<div style="font-size:11px;color:#64748B;letter-spacing:0.1em;margin-bottom:12px;">'
        'CRRA UTILITY SURFACE  ·  E[U] vs EQUITY WEIGHT'
        '</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    if baseline is not None:
        fig.add_trace(go.Scatter(
            x=baseline["weights"]*100, y=baseline["eu"],
            mode="lines", name=f"SPY baseline (50k, γ={gamma:.2f})",
            line=dict(color="#FFC72C", width=2.5),
        ))
        fig.add_vline(x=float(baseline["optimal_w"])*100, line_dash="dash",
                      line_color="#FFC72C", opacity=0.6,
                      annotation_text=f"SPY w*={float(baseline['optimal_w'])*100:.1f}%",
                      annotation_font_color="#FFC72C")

    if st.session_state.live_results is not None:
        r = st.session_state.live_results
        live_t = st.session_state.live_ticker
        c = ASSET_COLORS.get(live_t, "#FFC72C")
        fig.add_trace(go.Scatter(
            x=np.asarray(r["weights"])*100, y=r["eu"],
            mode="lines+markers", name=f"{live_t} (5k + quadratic)",
            line=dict(color=c, width=2.5, dash="dash"),
            marker=dict(size=7),
        ))
        # Continuous quadratic fit overlay
        ws = np.asarray(r["weights"])
        eus = np.asarray(r["eu"])
        try:
            a, bcoef, ccoef = np.polyfit(ws, eus, 2)
            xs_fit = np.linspace(ws.min(), ws.max(), 100)
            ys_fit = a*xs_fit**2 + bcoef*xs_fit + ccoef
            fig.add_trace(go.Scatter(
                x=xs_fit*100, y=ys_fit,
                mode="lines", name=f"{live_t} quadratic fit",
                line=dict(color=c, width=1, dash="dot"), showlegend=False, opacity=0.5,
            ))
        except Exception:
            pass
        fig.add_vline(x=float(r["optimal_w"])*100, line_dash="dash", line_color=c, opacity=0.6,
                      annotation_text=f"{live_t} w*={float(r['optimal_w'])*100:.1f}%",
                      annotation_font_color=c)

    fig.update_layout(
        plot_bgcolor="rgba(20,5,7,0.6)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#94A3B8", size=11),
        xaxis=dict(title="Equity weight (%)", gridcolor="#3D1116", zerolinecolor="#3D1116"),
        yaxis=dict(title="E[U]",              gridcolor="#3D1116", zerolinecolor="#3D1116"),
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        margin=dict(t=10, b=40, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """<div style="margin-top:8px;font-size:11px;color:#94A3B8;line-height:1.7;
        background:rgba(20,5,7,0.5);border:1px solid #3D1116;border-radius:6px;padding:12px 16px;">
        <strong style="color:#E2E8F0;">What you're looking at:</strong>
        Each point represents a full lifecycle simulation — 480 months of accumulation (age 25–65)
        followed by up to 408 months of drawdown, with stochastic mortality from SSA 2022 tables.
        The peak is <strong style="color:#FFC72C;">w*</strong>, the equity weight that maximizes
        expected CRRA utility across all simulated lifetimes. A flatter curve means the model is
        less sensitive to the exact allocation.
        </div>""",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — HOW IT WORKS
# ──────────────────────────────────────────────────────────────────────────
with tab_explainer:
    items = [
        ("01", "What does CRRA optimization actually do?",
         "CRRA (Constant Relative Risk Aversion) is a utility function that models how much a household values consumption at different wealth levels. The model simulates 74 years of financial life — 40 years of saving, 34 years of spending — across thousands of possible market futures. It finds the equity allocation that maximizes expected lifetime wellbeing, trading off higher average wealth against the risk of running out of money."),
        ("02", "Why does changing the ticker change the optimal allocation?",
         "The model uses the selected asset's historical return distribution to generate future scenarios. A more volatile asset (e.g., AAPL) produces a wider range of outcomes — potentially much more wealth, but also a higher chance of severe shortfalls. CRRA utility penalizes shortfalls more than it rewards windfalls at the same magnitude, so higher volatility typically pushes the optimal weight down."),
        ("03", "What does the Social Security slider do?",
         "Social Security acts as a guaranteed income floor in retirement. Raising it reduces the consequence of running out of portfolio wealth — the household never falls below that floor. CRRA responds by tolerating more equity risk, since the downside is bounded. Lowering SS does the opposite: more reliance on the portfolio means the model wants safer assets. The slider only affects the live ticker — the SPY baseline is locked at the SSI floor of $1,261/mo."),
        ("04", "What does the Risk Profile selector do?",
         "γ (gamma) is the household's risk aversion parameter. Higher γ means a stronger preference for stable consumption over expected returns. The selector snaps to one of three precomputed baselines — Aggressive (γ=2), Moderate (γ=3.84, the De Nardi/French/Jones 2010 estimate), or Conservative (γ=7) — so the SPY benchmark stays methodologically consistent."),
        ("05", "What are the data limitations?",
         "Equity data comes from yfinance Adjusted Close prices (1993+). This is a total-return proxy for ETFs, but excludes dividends for individual stocks. Bond returns are constructed from the 10-yr Treasury yield (^TNX) using Macaulay duration approximation. CPI deflation uses Shiller Yale data. Bootstrap is IID; the source paper's full methodology uses block bootstrap to preserve serial dependence — results here may show slightly higher optimal equity weights as a result. Live runs use 5,000 paths vs 50,000 for the SPY baseline — directionally reliable, not precise estimates."),
        ("06", "What is the SPY baseline?",
         "The SPY baseline was pre-computed locally using 50,000 IID-bootstrap paths and a fine 21-point weight grid — the full research-grade simulation. It serves as the reference point. Live results (5,000 paths, 6-point grid + quadratic fit) are shown alongside for comparison. Three baselines are stored, one for each γ preset."),
    ]
    for num, q, a in items:
        st.markdown(
            f"""<div style="margin-bottom:14px;background:linear-gradient(135deg,rgba(20,5,7,0.92),rgba(20,5,7,0.55));
            border:1px solid #3D1116;border-radius:8px;overflow:hidden;">
            <div style="background:linear-gradient(90deg,#3D1116,transparent);padding:10px 16px;
            font-size:12px;color:#E2E8F0;font-weight:700;">
            <span style="color:#FFC72C;margin-right:10px;">{num}</span>{q}
            </div>
            <div style="padding:13px 16px;font-size:12px;color:#94A3B8;line-height:1.75;">{a}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown(
        """<div style="background:linear-gradient(90deg,rgba(217,119,6,0.1),transparent);
        border:1px solid rgba(142,27,52,0.30);border-left:3px solid #8E1B34;
        border-radius:6px;padding:12px 16px;font-size:11px;color:#E0C56E;line-height:1.7;">
        <strong style="color:#FFC72C;">Disclaimer:</strong>
        This tool is a class demonstration of CRRA lifecycle optimization methodology.
        It does not constitute investment advice. All results are based on historical
        return simulations and should not be used for financial planning decisions.
        </div>""",
        unsafe_allow_html=True,
    )
