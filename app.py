"""
VVER-1200 Fuel Rod Cladding Failure — Interactive Detection Dashboard
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from data.generator import SensorDataGenerator
from models.preprocessor import CoolantSignalPreprocessor
from models.detector import LeakDetector
from visualization.reactor_3d import build_reactor_3d, compute_component_scores
from utils.physics import (
    BASELINE_I131, BASELINE_I134, BASELINE_CS137,
    BASELINE_DNS, BASELINE_PH, BASELINE_CONDUCTIVITY,
    LIMIT_I131, LIMIT_I134, LIMIT_I134_I131_RATIO,
    LIMIT_DNS, LIMIT_CS137, PH_MIN, PH_MAX,
    T_HALF_I134_S, T_HALF_I131_S,
    ISOTOPES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VVER-1200 | Cladding Failure Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design system
# ─────────────────────────────────────────────────────────────────────────────

C = dict(
    bg0      = "#06080d",   # deepest background
    bg1      = "#0d1117",   # page background
    bg2      = "#161b22",   # card / panel
    bg3      = "#1c2333",   # elevated card
    border   = "#21262d",
    border2  = "#30363d",
    text0    = "#e6edf3",   # primary text
    text1    = "#8b949e",   # secondary text
    text2    = "#484f58",   # muted text
    blue     = "#388bfd",
    blue_d   = "#1f4e8f",
    green    = "#3fb950",
    green_d  = "#196c2e",
    amber    = "#d29922",
    amber_d  = "#5a3e05",
    red      = "#f85149",
    red_d    = "#5a1e1e",
    purple   = "#a371f7",
    teal     = "#39c5cf",
    i131     = "#58a6ff",
    i134     = "#ff7b72",
    cs137    = "#ffa657",
    dns      = "#56d364",
    ratio    = "#e3b341",
    score    = "#bc8cff",
    ph       = "#79c0ff",
    cond     = "#d2a8ff",
)

PLOT_BG    = C["bg2"]
PAPER_BG   = C["bg1"]
GRID_COL   = C["border2"]
FONT_FAM   = "'JetBrains Mono', 'Fira Code', 'Courier New', monospace"
SANS_FAM   = "'Inter', 'Segoe UI', system-ui, sans-serif"


def hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color string to an rgba() string Plotly 6 accepts."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Reset & base ─────────────────────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: {SANS_FAM};
    color: {C["text0"]};
}}
.main .block-container {{
    padding: 0 2rem 3rem 2rem;
    max-width: 100%;
}}
[data-testid="stSidebar"] > div:first-child {{
    background: {C["bg2"]};
    border-right: 1px solid {C["border"]};
    padding-top: 0;
}}

/* ── Hide default chrome ──────────────────────────────────────────────────── */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}

/* ── Header banner ────────────────────────────────────────────────────────── */
.site-header {{
    background: linear-gradient(180deg, #0d1f35 0%, {C["bg1"]} 100%);
    border-bottom: 1px solid {C["blue_d"]};
    padding: 18px 2rem 14px 2rem;
    margin: 0 -2rem 1.5rem -2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.site-header-left {{
    display: flex;
    flex-direction: column;
    gap: 3px;
}}
.site-title {{
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {C["text0"]};
}}
.site-subtitle {{
    font-size: 0.72rem;
    color: {C["text1"]};
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.site-header-right {{
    display: flex;
    align-items: center;
    gap: 18px;
}}
.header-badge {{
    font-family: {FONT_FAM};
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    padding: 3px 9px;
    border-radius: 3px;
    border: 1px solid {C["border2"]};
    color: {C["text1"]};
    text-transform: uppercase;
}}
.header-clock {{
    font-family: {FONT_FAM};
    font-size: 0.78rem;
    color: {C["text1"]};
}}

/* ── Reactor status bar ───────────────────────────────────────────────────── */
.status-bar {{
    display: flex;
    gap: 2px;
    margin: 0 -2rem 1.8rem -2rem;
    padding: 0 2rem;
}}
.status-segment {{
    flex: 1;
    height: 3px;
    border-radius: 0;
}}

/* ── Section titles ───────────────────────────────────────────────────────── */
.section-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2rem 0 0.8rem 0;
    padding-bottom: 8px;
    border-bottom: 1px solid {C["border"]};
}}
.section-num {{
    font-family: {FONT_FAM};
    font-size: 0.65rem;
    color: {C["text2"]};
    letter-spacing: 0.06em;
    min-width: 28px;
}}
.section-title {{
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: {C["text0"]};
}}
.section-desc {{
    font-size: 0.75rem;
    color: {C["text1"]};
    margin-left: auto;
}}

/* ── Metric cards ─────────────────────────────────────────────────────────── */
.metrics-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    margin-bottom: 1rem;
}}
.metric-card {{
    background: {C["bg2"]};
    border: 1px solid {C["border"]};
    border-radius: 6px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
}}
.metric-card::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}}
.mc-blue::before  {{ background: {C["blue"]}; }}
.mc-green::before {{ background: {C["green"]}; }}
.mc-amber::before {{ background: {C["amber"]}; }}
.mc-red::before   {{ background: {C["red"]}; }}
.mc-purple::before{{ background: {C["purple"]}; }}
.mc-teal::before  {{ background: {C["teal"]}; }}
.metric-label {{
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {C["text1"]};
    margin-bottom: 6px;
    font-weight: 500;
}}
.metric-value {{
    font-family: {FONT_FAM};
    font-size: 1.45rem;
    font-weight: 600;
    line-height: 1;
}}
.metric-sub {{
    font-size: 0.68rem;
    color: {C["text2"]};
    margin-top: 5px;
    font-family: {FONT_FAM};
}}
.mv-green  {{ color: {C["green"]}; }}
.mv-amber  {{ color: {C["amber"]}; }}
.mv-red    {{ color: {C["red"]}; }}
.mv-blue   {{ color: {C["blue"]}; }}
.mv-purple {{ color: {C["purple"]}; }}
.mv-white  {{ color: {C["text0"]}; }}

/* ── Detection result card ────────────────────────────────────────────────── */
.result-card {{
    border-radius: 6px;
    padding: 12px 18px;
    margin: 1rem 0;
    border: 1px solid;
    font-size: 0.82rem;
    line-height: 1.6;
}}
.result-ok    {{ background:{C["green_d"]}22; border-color:{C["green"]}55; color:{C["text0"]}; }}
.result-warn  {{ background:{C["amber_d"]}22; border-color:{C["amber"]}55; color:{C["text0"]}; }}
.result-alarm {{ background:{C["red_d"]}22;   border-color:{C["red"]}55;   color:{C["text0"]}; }}
.result-label {{ font-weight: 700; letter-spacing: 0.05em; font-size: 0.75rem; text-transform: uppercase; }}

/* ── Physics info box ─────────────────────────────────────────────────────── */
.physics-panel {{
    background: {C["bg2"]};
    border: 1px solid {C["border2"]};
    border-left: 3px solid {C["blue"]};
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.82rem;
    line-height: 1.7;
    color: {C["text1"]};
}}
.physics-panel b {{ color: {C["text0"]}; }}
.physics-panel code {{
    font-family: {FONT_FAM};
    background: {C["bg3"]};
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.78rem;
    color: {C["teal"]};
}}
.ref-tag {{
    display: inline-block;
    font-size: 0.62rem;
    font-family: {FONT_FAM};
    background: {C["bg3"]};
    border: 1px solid {C["border2"]};
    color: {C["text2"]};
    padding: 1px 6px;
    border-radius: 3px;
    letter-spacing: 0.04em;
    margin-left: 4px;
    vertical-align: middle;
}}

/* ── Data table ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {C["border"]};
    border-radius: 6px;
    overflow: hidden;
}}

/* ── Tab bar ──────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {C["border"]};
    gap: 0;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 600;
    color: {C["text1"]};
    background: transparent;
    border-bottom: 2px solid transparent;
    padding: 10px 18px;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: {C["text0"]};
    border-bottom-color: {C["blue"]};
    background: transparent;
}}

/* ── Sidebar typography ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] .sidebar-logo {{
    padding: 16px 16px 0 16px;
    border-bottom: 1px solid {C["border"]};
    margin-bottom: 16px;
}}
[data-testid="stSidebar"] .sidebar-section {{
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {C["text2"]};
    padding: 8px 16px 4px 16px;
    font-weight: 600;
}}
.stSlider label, .stSelectbox label, .stCheckbox label {{
    font-size: 0.78rem !important;
    color: {C["text1"]} !important;
    letter-spacing: 0.02em;
}}

/* ── Expander ─────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: 1px solid {C["border"]} !important;
    border-radius: 6px !important;
    background: {C["bg2"]} !important;
}}
[data-testid="stExpander"] summary {{
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-weight: 600;
    color: {C["text1"]};
}}

/* ── Dividers ─────────────────────────────────────────────────────────────── */
hr {{ border-color: {C["border"]}; margin: 1.5rem 0; }}

/* ── Progress bar override ────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {{
    background: {C["blue"]};
}}

/* ── Reference table ──────────────────────────────────────────────────────── */
.ref-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
    font-family: {FONT_FAM};
}}
.ref-table th {{
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {C["text2"]};
    border-bottom: 1px solid {C["border2"]};
    padding: 6px 10px;
    text-align: left;
    font-weight: 500;
}}
.ref-table td {{
    padding: 5px 10px;
    border-bottom: 1px solid {C["border"]};
    color: {C["text1"]};
}}
.ref-table td:first-child {{ color: {C["text0"]}; font-weight: 500; }}
.ref-table tr:last-child td {{ border-bottom: none; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-logo">
        <div style="font-size:0.95rem;font-weight:700;letter-spacing:0.1em;
                    text-transform:uppercase;color:{C['text0']}">
            VVER-1200
        </div>
        <div style="font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;
                    color:{C['text2']};margin-top:3px">
            Cladding Failure Detection System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-section">Simulation</div>', unsafe_allow_html=True)
    hours = st.slider("Duration (hours)", 36, 120, 72, 6)

    auto_leak = st.checkbox("Random leak time", value=True)
    if auto_leak:
        leak_hour_arg = None
        st.caption("Leak injected at a random hour after h 24")
    else:
        leak_hour_arg = st.slider(
            "Leak start (hour)", 5, hours - 6, min(40, hours - 6)
        )

    seed = st.number_input("Random seed", 0, 9999, 42, 1)

    st.markdown(f'<div class="sidebar-section" style="margin-top:8px">Model</div>',
                unsafe_allow_html=True)
    sensitivity = st.slider("Alert threshold", 0.30, 0.95, 0.70, 0.05,
        help="Composite score above this triggers an alarm")
    train_hours = st.slider("Training window (hours)", 6, 24, 20, 2,
        help="Only normal-operation data is used for training")
    physics_weight = st.slider("Physics rule weight α", 0.0, 1.0, 0.60, 0.05,
        help="alert = α·physics + (1−α)·ML")

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Simulation", use_container_width=True, type="primary")

    st.markdown(f"""
    <div style="margin-top:auto;padding:16px 0 0 0;border-top:1px solid {C['border']};
                font-size:0.62rem;color:{C['text2']};line-height:1.6;text-transform:uppercase;
                letter-spacing:0.04em">
        Synthetic data only<br>
        Physics: IAEA-TECDOC-1328<br>
        IAEA NS-G-2.2 &nbsp;·&nbsp; NUREG/CR-6365
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline(hours, leak_hour_arg, seed, sensitivity, train_hours, physics_weight):
    gen = SensorDataGenerator(
        duration_hours=hours,
        leak_hour=leak_hour_arg,
        random_seed=seed,
    )
    raw_df = gen.generate()

    preprocessor = CoolantSignalPreprocessor(rolling_window_minutes=30)
    processed_df = preprocessor.transform(raw_df)
    X_all, feature_names = preprocessor.get_feature_matrix(processed_df)

    train_n = min(train_hours * 60, gen.leak_sample)
    detector = LeakDetector(
        contamination=0.05,
        physics_weight=physics_weight,
        alert_threshold=sensitivity,
    )
    detector.fit(X_all[:train_n])

    results = detector.predict(X_all, processed_df)
    metrics = detector.compute_performance_metrics(
        results["alert"], raw_df["label"].values, gen.leak_sample
    )
    return raw_df, processed_df, results, metrics, gen.leak_hour, gen.leak_sample, feature_names

if "results" not in st.session_state or run:
    with st.spinner("Running simulation pipeline..."):
        (
            st.session_state.raw_df,
            st.session_state.processed_df,
            st.session_state.results,
            st.session_state.metrics,
            st.session_state.leak_hour,
            st.session_state.leak_sample,
            st.session_state.feature_names,
        ) = run_pipeline(hours, leak_hour_arg, seed, sensitivity, train_hours, physics_weight)

raw_df       = st.session_state.raw_df
processed_df = st.session_state.processed_df
results      = st.session_state.results
metrics      = st.session_state.metrics
leak_hour    = st.session_state.leak_hour
leak_sample  = st.session_state.leak_sample
feature_names = st.session_state.feature_names

ts  = pd.to_datetime(raw_df["timestamp"])
t_h = (ts - ts.iloc[0]).dt.total_seconds().values / 3600.0
fa  = metrics["first_alert_idx"]
lt  = metrics["lead_time_minutes"]

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

ts_now = datetime.now().strftime("%Y-%m-%d  %H:%M")
sim_range = f"T+00:00 — T+{hours:02d}:00"

if fa is None:
    status_label, status_color = "MISSED", C["red"]
elif lt is not None and lt > 0:
    status_label, status_color = "EARLY WARNING", C["green"]
elif lt == 0:
    status_label, status_color = "DETECTED", C["green"]
else:
    status_label, status_color = "LATE DETECTION", C["amber"]

st.markdown(f"""
<div class="site-header">
  <div class="site-header-left">
    <div class="site-title">Nuclear Fuel Rod Cladding Failure Detection</div>
    <div class="site-subtitle">VVER-1200 Primary Coolant Radiochemistry Monitor &nbsp;/&nbsp; Anomaly Detection</div>
  </div>
  <div class="site-header-right">
    <span class="header-badge">{sim_range}</span>
    <span class="header-badge" style="color:{status_color};border-color:{status_color}55">
      {status_label}
    </span>
    <span class="header-clock">{ts_now}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Status colour strip
n_seg = 80
segs = []
for i in range(n_seg):
    frac = i / n_seg
    t_at = frac * hours
    if t_at < leak_hour:
        col = C["green_d"]
    elif fa is not None and t_at >= t_h[fa]:
        col = C["red_d"]
    else:
        col = C["amber_d"]
    segs.append(f'<div class="status-segment" style="background:{col}"></div>')

st.markdown(
    f'<div class="status-bar">{"".join(segs)}</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics row
# ─────────────────────────────────────────────────────────────────────────────

def _result_html(lt, fa):
    if fa is None:
        return f'<div class="metric-value mv-red">MISSED</div>'
    if lt is not None and lt > 0:
        return f'<div class="metric-value mv-green">+{lt} min</div>'
    if lt == 0:
        return f'<div class="metric-value mv-green">At onset</div>'
    return f'<div class="metric-value mv-amber">{abs(lt)} min late</div>'

fp_color = "mv-green" if metrics["false_positives"] == 0 else "mv-amber"
fa_str   = f"{t_h[fa]:.1f} h" if fa is not None else "—"

st.markdown(f"""
<div class="metrics-row">
  <div class="metric-card mc-blue">
    <div class="metric-label">Leak injected</div>
    <div class="metric-value mv-blue">{leak_hour:.1f} h</div>
    <div class="metric-sub">sample {leak_sample}</div>
  </div>
  <div class="metric-card mc-{'green' if fa is not None and lt is not None and lt > 0 else 'red' if fa is None else 'amber'}">
    <div class="metric-label">Detection result</div>
    {_result_html(lt, fa)}
    <div class="metric-sub">first alert at {fa_str}</div>
  </div>
  <div class="metric-card mc-{'green' if metrics['false_positives'] == 0 else 'amber'}">
    <div class="metric-label">False positives</div>
    <div class="metric-value {fp_color}">{metrics['false_positives']}</div>
    <div class="metric-sub">FP rate {metrics['false_positive_rate']:.4f}</div>
  </div>
  <div class="metric-card mc-purple">
    <div class="metric-label">Precision</div>
    <div class="metric-value mv-purple">{metrics['precision']:.3f}</div>
    <div class="metric-sub">TP / (TP + FP)</div>
  </div>
  <div class="metric-card mc-purple">
    <div class="metric-label">Recall</div>
    <div class="metric-value mv-purple">{metrics['recall']:.3f}</div>
    <div class="metric-sub">TP / (TP + FN)</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Detection result banner
if fa is None:
    st.markdown(
        f'<div class="result-card result-alarm">'
        f'<span class="result-label" style="color:{C["red"]}">Missed Detection</span>'
        f'<br>No sustained alert was generated over the {hours}-hour window. '
        f'Consider lowering the alert threshold or increasing the physics rule weight.'
        f'</div>', unsafe_allow_html=True)
elif lt is not None and lt > 0:
    st.markdown(
        f'<div class="result-card result-ok">'
        f'<span class="result-label" style="color:{C["green"]}">Early Warning Achieved</span>'
        f'<br>First alert at hour {t_h[fa]:.2f} — {lt} minute(s) before cladding failure onset at hour {leak_hour:.2f}. '
        f'Operators have {lt} minutes to initiate diagnostic sampling and evaluate '
        f'the I-134/I-131 ratio trend before the formal action limit is breached.'
        f'</div>', unsafe_allow_html=True)
elif lt == 0:
    st.markdown(
        f'<div class="result-card result-ok">'
        f'<span class="result-label" style="color:{C["green"]}">Detected at Onset</span>'
        f'<br>Alert triggered coincident with cladding failure at hour {leak_hour:.2f}. '
        f'Zero lead time — consider lowering the threshold for earlier warning.</div>',
        unsafe_allow_html=True)
else:
    late = abs(lt) if lt is not None else "?"
    st.markdown(
        f'<div class="result-card result-warn">'
        f'<span class="result-label" style="color:{C["amber"]}">Late Detection</span>'
        f'<br>First alert {late} minute(s) after cladding failure onset. '
        f'Reduce the alert threshold to improve lead time.</div>',
        unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Shared chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_theme(fig, height=380, title=""):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=11, color=C["text1"]),
                   x=0, xref="paper") if title else None,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=SANS_FAM, color=C["text0"], size=11),
        legend=dict(
            bgcolor="rgba(13,17,23,0.85)",
            bordercolor=C["border2"],
            borderwidth=1,
            font=dict(size=10, family=FONT_FAM),
        ),
        margin=dict(l=8, r=8, t=42 if title else 18, b=8),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=C["bg3"],
            bordercolor=C["border2"],
            font=dict(family=FONT_FAM, size=10, color=C["text0"]),
        ),
        xaxis=dict(
            gridcolor=GRID_COL, gridwidth=0.5,
            zerolinecolor=GRID_COL,
            tickfont=dict(family=FONT_FAM, size=9, color=C["text1"]),
            title=dict(text="Time (hours)", font=dict(size=10, color=C["text1"])),
        ),
        yaxis=dict(
            gridcolor=GRID_COL, gridwidth=0.5,
            zerolinecolor=GRID_COL,
            tickfont=dict(family=FONT_FAM, size=9, color=C["text1"]),
        ),
    )
    return fig

def add_event_vlines(fig, row=None, col=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_vline(x=t_h[leak_sample], line_color=C["green"],
                  line_dash="dash", line_width=1.4, **kw)
    if fa is not None:
        fig.add_vline(x=t_h[fa], line_color=C["red"],
                      line_dash="dash", line_width=1.4, **kw)

def section_hdr(num, title, desc=""):
    st.markdown(f"""
    <div class="section-header">
      <span class="section-num">{num}</span>
      <span class="section-title">{title}</span>
      {"<span class='section-desc'>" + desc + "</span>" if desc else ""}
    </div>
    """, unsafe_allow_html=True)

def physics_box(html):
    st.markdown(f'<div class="physics-panel">{html}</div>', unsafe_allow_html=True)

def ref(code):
    return f'<span class="ref-tag">{code}</span>'

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_3d, tab_radio, tab_detect, tab_chem, tab_data, tab_ref = st.tabs([
    "3D Reactor", "Radiological", "Detection", "Chemistry", "Data Explorer", "Reference",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 0 — 3D Reactor
# ═════════════════════════════════════════════════════════════════════════════

# Component metadata: key → (display name, sensor list, description)
COMPONENTS = {
    "core": (
        "Reactor Core (ТВЭЛ)",
        ["I134", "I131", "DNS"],
        "Fuel rod cladding is the primary barrier. Failure releases fission products "
        "directly from the fuel-cladding gap into the primary coolant. "
        "Governed by I-134/I-131 ratio and delayed neutron signal.",
    ),
    "loop": (
        "Primary Coolant Loops & Steam Generators",
        ["I131", "I134", "Cs137"],
        "Four primary loops carry contaminated coolant from the RPV to the steam generators. "
        "Absolute activity levels of I-131, I-134, and Cs-137 are monitored here. "
        "Rising activity indicates cladding breach somewhere in the core.",
    ),
    "activity": (
        "Coolant Activity Monitor",
        ["I131", "I134"],
        "A continuous bypass sample is taken from the cold leg and passed through "
        "a gamma spectrometer. The I-134/I-131 ratio computed here is the primary "
        "early-warning indicator — can precede absolute limit exceedance by 30–90 min.",
    ),
    "dns": (
        "Delayed Neutron Monitor (DNS)",
        ["DNS"],
        "A neutron detector on the primary coolant bypass line detects delayed-neutron "
        "precursors (Br-87, Kr-87/88). These only appear if the fuel matrix is directly "
        "exposed to coolant — the most specific indicator of cladding failure.",
    ),
    "chemistry": (
        "Pressurizer / Chemical & Volume Control",
        ["pH", "conductivity"],
        "The pressurizer maintains primary circuit pressure (~160 bar). "
        "The CVCS controls coolant chemistry. Iodine hydrolysis from leaking fuel "
        "slightly lowers pH; dissolved fission products raise electrical conductivity.",
    ),
}

with tab_3d:

    # ── Sidebar additions for 3D tab ──────────────────────────────────────────
    # (component selector lives in sidebar, conditionally shown)

    col_3d_ctrl, col_3d_main = st.columns([1, 3])

    with col_3d_ctrl:
        st.markdown(f"""
        <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:{C['text2']};padding:0 0 8px 0;font-weight:600;
                    border-bottom:1px solid {C['border']};margin-bottom:12px">
            Component Selection
        </div>
        """, unsafe_allow_html=True)

        comp_labels = {k: v[0] for k, v in COMPONENTS.items()}
        selected_comp = st.radio(
            "Select reactor component",
            options=list(comp_labels.keys()),
            format_func=lambda k: comp_labels[k],
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:{C['text2']};padding:0 0 8px 0;font-weight:600;
                    border-bottom:1px solid {C['border']};margin-bottom:12px">
            Time Navigator
        </div>
        """, unsafe_allow_html=True)

        time_idx = st.slider(
            "Simulation time",
            min_value=0,
            max_value=len(raw_df) - 1,
            value=len(raw_df) - 1,
            step=1,
            format=f"h %d",
            label_visibility="collapsed",
            help="Scrub through simulation time — the 3D model updates in real time",
        )
        t_current = t_h[time_idx]
        st.markdown(
            f'<div style="font-family:{FONT_FAM};font-size:0.75rem;color:{C["text1"]};'
            f'text-align:center;margin-top:-8px">T + {t_current:.2f} h</div>',
            unsafe_allow_html=True,
        )

        # Current component scores at selected time
        comp_scores = compute_component_scores(
            raw_df, processed_df, time_idx, sensitivity
        )

        # Status legend
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;
                    color:{C['text2']};margin-bottom:8px;font-weight:600">
            Component Status
        </div>
        """, unsafe_allow_html=True)

        for k, (name, _, _) in COMPONENTS.items():
            sc = comp_scores[k]
            if sc >= sensitivity:
                dot_col, state_label = C["red"],   "ALARM"
            elif sc >= 0.30:
                dot_col, state_label = C["amber"],  "CAUTION"
            else:
                dot_col, state_label = C["green"],  "NORMAL"
            sel_bg = f"background:{C['bg3']};border-left:2px solid {C['blue']};" if k == selected_comp else ""
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;padding:5px 8px;
                        border-radius:4px;{sel_bg}margin-bottom:3px">
                <div style="width:8px;height:8px;border-radius:50%;
                            background:{dot_col};flex-shrink:0"></div>
                <div style="font-size:0.72rem;color:{C['text1']};flex:1;
                            line-height:1.3">{name.split(' (')[0]}</div>
                <div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;
                            color:{dot_col}">{state_label}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_3d_main:
        # Build 3D figure at selected time
        latest_row = {
            **raw_df.iloc[time_idx].to_dict(),
            "ratio_I134_I131": float(processed_df["ratio_I134_I131"].iloc[time_idx]),
        }
        fig3d = build_reactor_3d(
            component_scores=comp_scores,
            selected=selected_comp,
            alert_threshold=sensitivity,
            latest=latest_row,
        )
        st.plotly_chart(fig3d, width="stretch")

    # ── Selected component detail panel ───────────────────────────────────────
    st.markdown(f"""
    <div style="border-top:1px solid {C['border']};margin:0.5rem 0 1rem 0"></div>
    """, unsafe_allow_html=True)

    comp_name, comp_sensors, comp_desc = COMPONENTS[selected_comp]
    comp_score_now = comp_scores[selected_comp]

    if comp_score_now >= sensitivity:
        band_col, band_label = C["red"],   "ALARM"
    elif comp_score_now >= 0.30:
        band_col, band_label = C["amber"],  "CAUTION"
    else:
        band_col, band_label = C["green"],  "NORMAL"

    # Component header
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;gap:16px;margin-bottom:1rem">
        <div style="width:4px;border-radius:2px;background:{band_col};
                    align-self:stretch;flex-shrink:0"></div>
        <div style="flex:1">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
                <span style="font-size:0.9rem;font-weight:700;letter-spacing:0.04em;
                             text-transform:uppercase;color:{C['text0']}">{comp_name}</span>
                <span style="font-family:JetBrains Mono,monospace;font-size:0.68rem;
                             padding:2px 8px;border-radius:3px;border:1px solid {band_col}55;
                             color:{band_col};letter-spacing:0.06em">{band_label}</span>
                <span style="font-family:JetBrains Mono,monospace;font-size:0.68rem;
                             color:{C['text2']}">score {comp_score_now:.4f}</span>
            </div>
            <div style="font-size:0.80rem;color:{C['text1']};line-height:1.7;
                        max-width:900px">{comp_desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sensor metric cards for current timestep
    sensor_meta = {
        "I131":        ("I-131",        "Bq/cm³",  C["i131"],  LIMIT_I131,  "{:.3e}"),
        "I134":        ("I-134",        "Bq/cm³",  C["i134"],  LIMIT_I134,  "{:.3e}"),
        "Cs137":       ("Cs-137",       "Bq/cm³",  C["cs137"], LIMIT_CS137, "{:.3e}"),
        "DNS":         ("DNS",          "norm.",   C["dns"],   LIMIT_DNS,   "{:.4f}"),
        "pH":          ("pH",           "",        C["ph"],    PH_MAX,      "{:.4f}"),
        "conductivity":("Conductivity", "µS/cm",   C["cond"],  25.0,        "{:.3f}"),
    }

    metric_cols = st.columns(len(comp_sensors))
    for i, sensor in enumerate(comp_sensors):
        if sensor not in sensor_meta:
            continue
        s_name, s_unit, s_col, s_limit, s_fmt = sensor_meta[sensor]
        val = float(raw_df[sensor].iloc[time_idx])
        formatted = s_fmt.format(val)
        pct = min(100, int(val / s_limit * 100)) if s_limit > 0 else 0
        bar_col = C["red"] if pct >= 100 else C["amber"] if pct >= 50 else C["green"]
        with metric_cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="border-top:2px solid {s_col}">
                <div class="metric-label">{s_name}</div>
                <div class="metric-value" style="color:{s_col};font-size:1.15rem">
                    {formatted}
                </div>
                <div class="metric-sub">{s_unit}</div>
                <div style="margin-top:8px;height:3px;border-radius:2px;
                            background:{C['border2']}">
                    <div style="width:{pct}%;height:100%;border-radius:2px;
                                background:{bar_col};transition:width 0.3s"></div>
                </div>
                <div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;
                            color:{C['text2']};margin-top:3px">{pct}% of action limit</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Time-series chart for this component's sensors
    section_hdr("R3D", f"{comp_name} — Time Series",
                f"T+{t_current:.2f} h  |  vertical line = current time")

    n_sensor_cols = min(len(comp_sensors), 2)
    chart_cols = st.columns(n_sensor_cols) if n_sensor_cols > 1 else [st.container()]

    chart_color_map = {
        "I131": C["i131"], "I134": C["i134"], "Cs137": C["cs137"],
        "DNS": C["dns"], "pH": C["ph"], "conductivity": C["cond"],
    }
    chart_limit_map = {
        "I131": LIMIT_I131, "I134": LIMIT_I134, "Cs137": LIMIT_CS137,
        "DNS": LIMIT_DNS, "pH": PH_MAX, "conductivity": 25.0,
    }
    log_sensors = {"I131", "I134", "Cs137"}

    for i, sensor in enumerate(comp_sensors):
        col_idx = i % n_sensor_cols
        with chart_cols[col_idx]:
            s_name, s_unit, s_col, _, _ = sensor_meta.get(
                sensor, (sensor, "", C["blue"], None, "{:.4f}")
            )
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(
                x=t_h, y=raw_df[sensor].values,
                name=s_name,
                line=dict(color=s_col, width=1.2),
                fill="tozeroy",
                fillcolor=hex_rgba(s_col, 0.07),
                hovertemplate=f"<b>{s_name}</b>: %{{y:.4g}} {s_unit}",
            ))
            if sensor in chart_limit_map:
                fig_s.add_hline(
                    y=chart_limit_map[sensor],
                    line_color=C["red"], line_dash="dash",
                    line_width=0.9, opacity=0.7,
                    annotation_text="Action limit",
                    annotation_font_color=C["red"], annotation_font_size=9,
                )
            # Current time marker
            fig_s.add_vline(x=t_current, line_color=C["blue"],
                            line_dash="dash", line_width=1.4)
            # Leak + alert lines
            fig_s.add_vline(x=t_h[leak_sample], line_color=C["green"],
                            line_dash="dash", line_width=1.2, opacity=0.7)
            if fa is not None:
                fig_s.add_vline(x=t_h[fa], line_color=C["red"],
                                line_dash="dash", line_width=1.2, opacity=0.7)

            if sensor in log_sensors:
                fig_s.update_yaxes(type="log")
            fig_s.update_yaxes(
                title_text=f"{s_name} ({s_unit})" if s_unit else s_name,
                tickfont=dict(family=FONT_FAM, size=9),
                gridcolor=GRID_COL,
            )
            apply_theme(fig_s, height=260, title=s_name)
            st.plotly_chart(fig_s, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Radiological
# ═════════════════════════════════════════════════════════════════════════════

with tab_radio:

    # ── 1.1 Iodine concentrations ─────────────────────────────────────────────
    section_hdr("01", "Primary Coolant Iodine Activity",
                "I-131 · I-134 · Bq/cm³ · log scale")

    with st.expander("Physical basis"):
        t_half_i134_min = T_HALF_I134_S / 60
        t_half_i131_d = T_HALF_I131_S / 86400
        physics_box(f"""
        Iodine isotopes are produced in high yield by uranium fission and serve as the primary
        indicators of fuel rod cladding integrity. Under intact cladding the fuel matrix retains
        these products; a through-wall crack releases them into the primary coolant.<br><br>

        <b>I-131</b> &nbsp;(T½ = {t_half_i131_d:.2f} d) &nbsp;&nbsp; Fission yield 2.77 % &nbsp;·&nbsp;
        Gamma 364.5 keV &nbsp;·&nbsp; Baseline <code>{BASELINE_I131:.1e}</code> Bq/cm³ &nbsp;·&nbsp;
        Action limit <code>{LIMIT_I131:.1e}</code> Bq/cm³ {ref("NS-G-2.2")}<br>
        Long half-life means concentration builds up gradually after a breach — a <b>lagging indicator</b>.<br><br>

        <b>I-134</b> &nbsp;(T½ = {t_half_i134_min:.0f} min) &nbsp;&nbsp; Fission yield 7.80 % &nbsp;·&nbsp;
        Gamma 847 keV &nbsp;·&nbsp; Baseline <code>{BASELINE_I134:.1e}</code> Bq/cm³ &nbsp;·&nbsp;
        Action limit <code>{LIMIT_I134:.1e}</code> Bq/cm³<br>
        Short half-life means only <b>freshly produced</b> I-134 reaches the coolant in detectable
        quantities — it spikes immediately after a breach and decays within hours.
        This is the <b>leading indicator</b>.<br><br>

        <b>Leak sequence</b>: I-134 spikes at t = 0 (gap inventory burst), I-131 follows at t ≈ 2–4 h
        (matrix diffusion). The time gap between the two rises confirms cladding breach rather than
        a tramp-uranium transient. {ref("IAEA-TECDOC-1328")}
        """)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_h, y=raw_df["I131"], name="I-131",
        line=dict(color=C["i131"], width=1.1),
        hovertemplate="<b>I-131</b>: %{y:.3e} Bq/cm³",
    ))
    fig.add_trace(go.Scatter(
        x=t_h, y=raw_df["I134"], name="I-134",
        line=dict(color=C["i134"], width=1.1),
        hovertemplate="<b>I-134</b>: %{y:.3e} Bq/cm³",
    ))
    for lim, col, name in [
        (LIMIT_I131, C["i131"], "I-131 limit"),
        (LIMIT_I134, C["i134"], "I-134 limit"),
    ]:
        fig.add_hline(y=lim, line_color=col, line_dash="dot", line_width=0.9, opacity=0.55,
                      annotation_text=f"{name}  {lim:.0e}", annotation_font_size=9,
                      annotation_font_color=col)
    add_event_vlines(fig)
    fig.update_yaxes(type="log", title_text="Activity (Bq/cm³)",
                     tickfont=dict(family=FONT_FAM, size=9))
    apply_theme(fig, height=380, title="I-131 and I-134 Concentrations in Primary Coolant")
    st.plotly_chart(fig, width="stretch")

    # ── 1.2 Isotope ratio ─────────────────────────────────────────────────────
    section_hdr("02", "I-134 / I-131 Isotope Ratio",
                "Primary early-warning indicator · threshold = 10")

    with st.expander("Physical basis"):
        physics_box(f"""
        The I-134/I-131 activity ratio is the most powerful early-warning signal available in
        primary coolant radiochemistry. {ref("IAEA-TECDOC-1328 §4.3")}<br><br>

        <b>Mechanism</b>: both isotopes are released simultaneously when cladding fails, in
        proportion to their fission yields (ratio ≈ 7.8/2.8 ≈ <code>2.82</code> at t = 0).
        I-134 then decays with T½ = {t_half_i134_min:.0f} min while I-131 stays nearly constant
        (T½ = 8 days), so the ratio collapses predictably:<br>
        &nbsp;&nbsp;t = 0 h: ratio ≈ 2–5 &nbsp;(first rise above normal background of 0.5–1.0)<br>
        &nbsp;&nbsp;t = 1 h: ratio ≈ <b>10–50</b> — action threshold breached<br>
        &nbsp;&nbsp;t = 6 h: ratio ≈ 1–3<br>
        &nbsp;&nbsp;t = 24 h: ratio < 0.05 (I-134 essentially absent)<br><br>

        <b>Diagnostic value</b>: the ratio provides 30–90 minutes of warning before absolute
        level limits are exceeded. It also distinguishes fresh damage (high ratio) from
        pre-existing equilibrium tramp-uranium activity (low ratio ~1). {ref("NS-G-2.2")}
        """)

    ratio = processed_df["ratio_I134_I131"].values

    fig = go.Figure()
    fig.add_hrect(
        y0=LIMIT_I134_I131_RATIO, y1=max(ratio.max() * 2, 30),
        fillcolor=C["red"], opacity=0.07, line_width=0,
    )
    fig.add_trace(go.Scatter(
        x=t_h, y=np.maximum(ratio, 0.01), name="I-134 / I-131",
        line=dict(color=C["ratio"], width=1.3),
        fill="tozeroy", fillcolor=hex_rgba(C["ratio"], 0.08),
        hovertemplate="<b>Ratio</b>: %{y:.3f}",
    ))
    fig.add_hline(
        y=LIMIT_I134_I131_RATIO, line_color=C["red"],
        line_dash="dash", line_width=1.1,
        annotation_text=f"Action limit = {LIMIT_I134_I131_RATIO:.0f}",
        annotation_font_color=C["red"], annotation_font_size=9,
    )
    add_event_vlines(fig)
    fig.update_yaxes(type="log", title_text="I-134 / I-131 ratio",
                     tickfont=dict(family=FONT_FAM, size=9))
    apply_theme(fig, height=320, title="I-134 / I-131 Activity Ratio — Fresh Fuel Damage Indicator")
    st.plotly_chart(fig, width="stretch")

    # ── 1.3 Cs-137 ────────────────────────────────────────────────────────────
    section_hdr("03", "Cs-137 Activity",
                "Long-lived tracer · lagging indicator")

    with st.expander("Physical basis"):
        physics_box(f"""
        Caesium-137 (T½ = 30.2 years) diffuses from the UO₂ fuel matrix significantly more
        slowly than iodine isotopes due to its larger ionic radius and lower diffusivity in
        UO₂ at reactor temperatures.<br><br>
        <b>Role</b>: Cs-137 rise is a <b>confirmatory and cumulative</b> signal — it appears
        hours to days after the initial iodine transient and continues rising as long as the
        breach exists. Monitoring Cs-137 over the fuel cycle provides a quantitative measure
        of the total fractional release from the damaged rod(s).<br><br>
        Baseline <code>{BASELINE_CS137:.1e}</code> Bq/cm³ &nbsp;·&nbsp;
        Action limit <code>{LIMIT_CS137:.1e}</code> Bq/cm³ &nbsp;·&nbsp;
        Gamma 661.7 keV {ref("NUREG/CR-6365")}
        """)

    fig = go.Figure()
    fig.add_hline(y=LIMIT_CS137, line_color=C["red"], line_dash="dot",
                  line_width=0.9, opacity=0.55,
                  annotation_text=f"Action limit  {LIMIT_CS137:.0e}",
                  annotation_font_color=C["red"], annotation_font_size=9)
    fig.add_trace(go.Scatter(
        x=t_h, y=raw_df["Cs137"], name="Cs-137",
        line=dict(color=C["cs137"], width=1.1),
        fill="tozeroy", fillcolor=hex_rgba(C["cs137"], 0.07),
        hovertemplate="<b>Cs-137</b>: %{y:.3e} Bq/cm³",
    ))
    add_event_vlines(fig)
    fig.update_yaxes(type="log", title_text="Activity (Bq/cm³)",
                     tickfont=dict(family=FONT_FAM, size=9))
    apply_theme(fig, height=280, title="Cs-137 Concentration in Primary Coolant")
    st.plotly_chart(fig, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Detection
# ═════════════════════════════════════════════════════════════════════════════

with tab_detect:

    # ── 2.1 Composite alert score ─────────────────────────────────────────────
    section_hdr("04", "Composite Anomaly Score",
                f"Isolation Forest + Physics Rules · threshold {sensitivity:.2f}")

    with st.expander("Model architecture"):
        physics_box(f"""
        The alert score combines two independent anomaly detectors:<br><br>

        <code>alert_score = α · physics_score + (1−α) · ml_score</code><br>
        &nbsp;&nbsp;α = <code>{physics_weight:.2f}</code> (set in sidebar)<br><br>

        <b>Physics score</b> — three rules derived from IAEA radiochemistry limits:<br>
        &nbsp;&nbsp;50 % — Rule 1: I-134/I-131 ratio sigmoid centred on threshold {LIMIT_I134_I131_RATIO:.0f} {ref("TECDOC-1328")}<br>
        &nbsp;&nbsp;35 % — Rule 2: 30-min rate of change of I-134 (catches rising slope before peak)<br>
        &nbsp;&nbsp;15 % — Rule 3: DNS signal normalised to action level {LIMIT_DNS:.1f} (most specific)<br><br>

        <b>ML score</b> — Isolation Forest trained on <code>{train_hours}</code> hours of normal data.
        Anomalous samples produce shorter average path lengths in random binary partition trees,
        mapped to [0, 1] after affine rescaling.<br><br>

        <b>Alert zones</b><br>
        <code style="color:{C['green']}">0.00 – 0.30</code> &nbsp; Normal operation<br>
        <code style="color:{C['amber']}">0.30 – {sensitivity:.2f}</code> &nbsp; Caution — elevated activity, investigate<br>
        <code style="color:{C['red']}">{sensitivity:.2f} – 1.00</code> &nbsp; Alarm — initiate response procedure per Tech Specs
        """)

    score  = results["alert_score"]
    phys_s = results["physics_score"]
    ml_s   = results["ml_score"]
    ci_lo  = results["ci_lower"]
    ci_hi  = results["ci_upper"]

    fig = go.Figure()

    fig.add_hrect(y0=0.0,         y1=0.30,        fillcolor=C["green_d"], opacity=0.20, line_width=0)
    fig.add_hrect(y0=0.30,        y1=sensitivity, fillcolor=C["amber_d"], opacity=0.20, line_width=0)
    fig.add_hrect(y0=sensitivity, y1=1.05,        fillcolor=C["red_d"],   opacity=0.20, line_width=0)

    fig.add_trace(go.Scatter(
        x=np.concatenate([t_h, t_h[::-1]]),
        y=np.concatenate([ci_hi, ci_lo[::-1]]),
        fill="toself", fillcolor=hex_rgba(C["score"], 0.09),
        line=dict(color="rgba(0,0,0,0)"),
        name="90% confidence interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=t_h, y=phys_s, name="Physics score",
        line=dict(color=C["ratio"], width=0.9, dash="dot"),
        hovertemplate="<b>Physics</b>: %{y:.4f}",
    ))
    fig.add_trace(go.Scatter(
        x=t_h, y=ml_s, name="ML score",
        line=dict(color=C["i131"], width=0.9, dash="dot"),
        hovertemplate="<b>ML</b>: %{y:.4f}",
    ))
    fig.add_trace(go.Scatter(
        x=t_h, y=score, name="Composite score",
        line=dict(color=C["score"], width=1.6),
        hovertemplate="<b>Score</b>: %{y:.4f}",
    ))
    fig.add_hline(
        y=sensitivity, line_color=C["red"], line_dash="dash", line_width=1.2,
        annotation_text=f"Alarm threshold  {sensitivity:.2f}",
        annotation_font_color=C["red"], annotation_font_size=9,
    )
    add_event_vlines(fig)

    if lt is not None and lt > 0 and fa is not None:
        fig.add_annotation(
            x=t_h[fa], y=sensitivity + 0.07,
            text=f"{lt} min early warning",
            showarrow=True, arrowhead=2, arrowsize=1,
            arrowcolor=C["red"], arrowwidth=1.5,
            font=dict(color=C["red"], size=10, family=FONT_FAM),
            ax=50, ay=-45,
        )

    # Zone labels on right axis
    for y_pos, txt, col in [
        (0.15, "NORMAL",  C["green"]),
        (0.44, "CAUTION", C["amber"]),
        (0.80, "ALARM",   C["red"]),
    ]:
        fig.add_annotation(
            x=1.0, y=y_pos, xref="paper", yref="y",
            text=txt, showarrow=False,
            font=dict(size=8, color=col, family=FONT_FAM),
            xanchor="right",
        )

    fig.update_yaxes(range=[0.0, 1.08], title_text="Alert Score",
                     tickfont=dict(family=FONT_FAM, size=9))
    apply_theme(fig, height=400, title="Composite Anomaly Score — Isolation Forest + Physics Rules")
    st.plotly_chart(fig, width="stretch")

    # ── 2.2 Alert timeline ────────────────────────────────────────────────────
    section_hdr("05", "Binary Alert Timeline",
                "Sustained alert requires 3 consecutive samples above threshold")

    alerts = results["alert"]

    fig = go.Figure()
    fig.add_hrect(y0=0.5, y1=1.3, fillcolor=C["red_d"], opacity=0.25, line_width=0)
    fig.add_trace(go.Scatter(
        x=t_h, y=alerts.astype(float),
        name="Alert state",
        line=dict(color=C["red"], width=1.4),
        fill="tozeroy", fillcolor=hex_rgba(C["red"], 0.10),
        hovertemplate="<b>Alert</b>: %{y}",
    ))
    add_event_vlines(fig)
    fig.update_yaxes(
        range=[-0.05, 1.35],
        tickvals=[0, 1], ticktext=["NORMAL", "ALARM"],
        tickfont=dict(family=FONT_FAM, size=9),
    )
    apply_theme(fig, height=200, title="Alert State (0 = Normal, 1 = Alarm)")
    st.plotly_chart(fig, width="stretch")

    # ── 2.3 Score distribution ────────────────────────────────────────────────
    section_hdr("06", "Score Distribution",
                "Normal vs post-leak period")

    col_a, col_b = st.columns(2)

    normal_scores = score[raw_df["label"].values == 0]
    leak_scores   = score[raw_df["label"].values == 1]

    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=normal_scores, name="Normal",
            nbinsx=40, marker_color=C["green"], opacity=0.7,
            hovertemplate="Score: %{x:.3f}<br>Count: %{y}",
        ))
        fig.add_vline(x=sensitivity, line_color=C["red"], line_dash="dash", line_width=1.2)
        fig.update_xaxes(title_text="Alert score", tickfont=dict(family=FONT_FAM, size=9))
        fig.update_yaxes(title_text="Count", tickfont=dict(family=FONT_FAM, size=9))
        apply_theme(fig, height=260, title="Normal Period Score Distribution")
        st.plotly_chart(fig, width="stretch")

    with col_b:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=leak_scores, name="Post-leak",
            nbinsx=40, marker_color=C["red"], opacity=0.7,
            hovertemplate="Score: %{x:.3f}<br>Count: %{y}",
        ))
        fig.add_vline(x=sensitivity, line_color=C["red"], line_dash="dash", line_width=1.2)
        fig.update_xaxes(title_text="Alert score", tickfont=dict(family=FONT_FAM, size=9))
        fig.update_yaxes(title_text="Count", tickfont=dict(family=FONT_FAM, size=9))
        apply_theme(fig, height=260, title="Post-Leak Period Score Distribution")
        st.plotly_chart(fig, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Chemistry
# ═════════════════════════════════════════════════════════════════════════════

with tab_chem:

    # ── 3.1 DNS ───────────────────────────────────────────────────────────────
    section_hdr("07", "Delayed Neutron Signal",
                "Direct fuel-coolant contact indicator")

    with st.expander("Physical basis"):
        physics_box(f"""
        The Delayed Neutron Signal (DNS) monitor measures neutron emission from short-lived
        delayed-neutron precursors (DNPs) in a continuous bypass coolant sample passed past
        a neutron detector.<br><br>

        <b>Key DNP species</b>:<br>
        &nbsp;&nbsp;Br-87 &nbsp; T½ = 55.6 s<br>
        &nbsp;&nbsp;Kr-87 &nbsp; T½ = 76.3 s<br>
        &nbsp;&nbsp;Kr-88 &nbsp; T½ = 170.4 min<br>
        &nbsp;&nbsp;Rb-87 &nbsp; T½ = 4.7 × 10¹⁰ y (stable, no contribution)<br><br>

        Under intact cladding these precursors remain in the fuel matrix; they cannot
        reach the coolant in significant quantities through an undamaged Zr tube.
        A detectable DNS rise above baseline (<code>{BASELINE_DNS:.1f}</code>) therefore
        constitutes near-certain evidence of direct fuel-coolant contact.<br><br>

        DNS is the <b>most specific</b> indicator (lowest false-positive rate) but requires
        a relatively significant breach before DNP flux into the coolant exceeds detection
        threshold. Action level: <code>{LIMIT_DNS}</code> (normalized). {ref("IAEA-TECDOC-1328")}
        """)

    fig = go.Figure()
    fig.add_hrect(y0=LIMIT_DNS, y1=max(raw_df["DNS"].max() * 1.1, LIMIT_DNS * 1.5),
                  fillcolor=C["red"], opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(
        x=t_h, y=raw_df["DNS"], name="DNS",
        line=dict(color=C["dns"], width=1.1),
        fill="tozeroy", fillcolor=hex_rgba(C["dns"], 0.07),
        hovertemplate="<b>DNS</b>: %{y:.4f}",
    ))
    fig.add_hline(y=LIMIT_DNS, line_color=C["red"], line_dash="dash", line_width=1.0,
                  annotation_text=f"Action level {LIMIT_DNS}",
                  annotation_font_color=C["red"], annotation_font_size=9)
    add_event_vlines(fig)
    fig.update_yaxes(title_text="DNS (normalized)", tickfont=dict(family=FONT_FAM, size=9))
    apply_theme(fig, height=300, title="Delayed Neutron Signal — Primary Coolant Bypass Monitor")
    st.plotly_chart(fig, width="stretch")

    # ── 3.2 pH & Conductivity ─────────────────────────────────────────────────
    section_hdr("08", "Coolant Chemistry",
                f"pH target {PH_MIN}–{PH_MAX}  ·  conductivity baseline {BASELINE_CONDUCTIVITY} µS/cm")

    with st.expander("Physical basis"):
        physics_box(f"""
        <b>pH</b><br>
        VVER-1200 primary coolant is buffered with boric acid (H₃BO₃) for reactivity control
        and potassium hydroxide (KOH) for pH adjustment, maintained at {PH_MIN}–{PH_MAX}.
        This range minimises corrosion of the Zr-1%Nb (E110) cladding and stainless steel
        internals while preventing magnetite deposition on fuel assemblies.<br><br>
        When iodine leaks from damaged fuel, it partially hydrolyses in water:<br>
        &nbsp;&nbsp;&nbsp;<code>I₂ + H₂O → HIO + H⁺ + I⁻</code><br>
        producing H⁺ ions and causing a subtle pH drop of ≈ 0.05–0.15 units. {ref("NS-G-2.2")}<br><br>

        <b>Conductivity</b><br>
        Proportional to total dissolved ionic strength (primarily boric acid concentration).
        Released fission products — dissolved as iodide (I⁻), iodate (IO₃⁻), and caesium
        ions (Cs⁺) — add to the ionic strength, raising conductivity by 1–3 µS/cm above
        the baseline of ≈ {BASELINE_CONDUCTIVITY} µS/cm.
        """)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=["Coolant pH", "Electrical Conductivity (µS/cm)"])

    fig.add_hrect(y0=PH_MIN, y1=PH_MAX, fillcolor=C["green"], opacity=0.08,
                  line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_h, y=raw_df["pH"], name="pH",
        line=dict(color=C["ph"], width=1.1),
        hovertemplate="<b>pH</b>: %{y:.4f}",
    ), row=1, col=1)
    for lim in [PH_MIN, PH_MAX]:
        fig.add_hline(y=lim, line_color=C["amber"], line_dash="dot",
                      line_width=0.8, opacity=0.6, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=t_h, y=raw_df["conductivity"], name="Conductivity",
        line=dict(color=C["cond"], width=1.1),
        hovertemplate="<b>Cond.</b>: %{y:.3f} µS/cm",
    ), row=1, col=2)

    for c in [1, 2]:
        fig.add_vline(x=t_h[leak_sample], line_color=C["green"],
                      line_dash="dash", line_width=1.4, row=1, col=c)
        if fa is not None:
            fig.add_vline(x=t_h[fa], line_color=C["red"],
                          line_dash="dash", line_width=1.4, row=1, col=c)

    fig.update_xaxes(title_text="Time (hours)", gridcolor=GRID_COL,
                     tickfont=dict(family=FONT_FAM, size=9))
    fig.update_yaxes(gridcolor=GRID_COL, tickfont=dict(family=FONT_FAM, size=9))
    fig.update_layout(
        height=300, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(family=SANS_FAM, color=C["text0"], size=11),
        legend=dict(bgcolor="rgba(13,17,23,0.85)", bordercolor=C["border2"],
                    font=dict(family=FONT_FAM, size=10)),
        margin=dict(l=8, r=8, t=42, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["bg3"], font=dict(family=FONT_FAM, size=10)),
    )
    st.plotly_chart(fig, width="stretch")

    # ── 3.3 Multi-signal overview ─────────────────────────────────────────────
    section_hdr("09", "Multi-Signal Overview",
                "All channels normalised to baseline — deviation from 1.0")

    norm_cols = {
        "I131": ("I-131", C["i131"], BASELINE_I131),
        "I134": ("I-134", C["i134"], BASELINE_I134),
        "Cs137": ("Cs-137", C["cs137"], BASELINE_CS137),
        "DNS": ("DNS", C["dns"], BASELINE_DNS),
    }

    fig = go.Figure()
    for col, (label, color, base) in norm_cols.items():
        fig.add_trace(go.Scatter(
            x=t_h, y=raw_df[col] / base,
            name=label, line=dict(color=color, width=0.9),
            hovertemplate=f"<b>{label}</b>: %{{y:.3f}} × baseline",
        ))
    fig.add_hline(y=1.0, line_color=C["text2"], line_dash="dot", line_width=0.7)
    add_event_vlines(fig)
    fig.update_yaxes(type="log", title_text="Activity / baseline (×)",
                     tickfont=dict(family=FONT_FAM, size=9))
    apply_theme(fig, height=320, title="All Channels Normalised to Baseline (log scale)")
    st.plotly_chart(fig, width="stretch")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Data Explorer
# ═════════════════════════════════════════════════════════════════════════════

with tab_data:

    section_hdr("10", "Raw Sensor Data & Scores")

    # Build display table
    disp = raw_df.copy()
    disp["alert_score"]    = np.round(results["alert_score"], 4)
    disp["physics_score"]  = np.round(results["physics_score"], 4)
    disp["ml_score"]       = np.round(results["ml_score"], 4)
    disp["alert"]          = results["alert"].astype(int)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        show_normal = st.checkbox("Show normal samples", value=True)
    with col2:
        show_leak = st.checkbox("Show post-leak samples", value=True)
    with col3:
        show_alert_only = st.checkbox("Alerts only", value=False)

    mask = np.ones(len(disp), dtype=bool)
    if not show_normal:
        mask &= disp["label"].values == 1
    if not show_leak:
        mask &= disp["label"].values == 0
    if show_alert_only:
        mask &= disp["alert"].values == 1

    disp_filtered = disp[mask].copy()

    st.dataframe(
        disp_filtered,
        width="stretch",
        height=420,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm"),
            "I131": st.column_config.NumberColumn("I-131 (Bq/cm³)", format="%.3e"),
            "I134": st.column_config.NumberColumn("I-134 (Bq/cm³)", format="%.3e"),
            "Cs137": st.column_config.NumberColumn("Cs-137 (Bq/cm³)", format="%.3e"),
            "pH": st.column_config.NumberColumn("pH", format="%.4f"),
            "conductivity": st.column_config.NumberColumn("Cond. (µS/cm)", format="%.3f"),
            "DNS": st.column_config.NumberColumn("DNS", format="%.4f"),
            "label": st.column_config.NumberColumn("Ground truth", format="%d"),
            "alert": st.column_config.NumberColumn("Alert", format="%d"),
            "alert_score": st.column_config.ProgressColumn(
                "Alert score", min_value=0.0, max_value=1.0, format="%.4f"),
            "physics_score": st.column_config.NumberColumn("Physics score", format="%.4f"),
            "ml_score": st.column_config.NumberColumn("ML score", format="%.4f"),
        },
    )

    csv = disp_filtered.to_csv(index=False).encode()
    st.download_button(
        "Download CSV", csv,
        file_name=f"vver1200_leak_h{hours}_seed{seed}.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    section_hdr("11", "Engineered Features",
                f"{len(feature_names)} features fed to Isolation Forest")

    with st.expander("Show feature list"):
        cols = st.columns(3)
        for i, name in enumerate(sorted(feature_names)):
            cols[i % 3].markdown(
                f'<span style="font-family:{FONT_FAM};font-size:0.75rem;'
                f'color:{C["text1"]}">{name}</span>',
                unsafe_allow_html=True,
            )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — Reference
# ═════════════════════════════════════════════════════════════════════════════

with tab_ref:

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        section_hdr("A", "Isotope Properties")
        st.markdown(f"""
        <table class="ref-table">
          <thead>
            <tr>
              <th>Isotope</th><th>T½</th><th>Fission yield</th>
              <th>Gamma (keV)</th><th>Baseline (Bq/cm³)</th><th>Limit (Bq/cm³)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>I-131</td>
              <td>{T_HALF_I131_S/86400:.2f} d</td>
              <td>2.77 %</td><td>364.5</td>
              <td>{BASELINE_I131:.1e}</td><td>{LIMIT_I131:.1e}</td>
            </tr>
            <tr>
              <td>I-134</td>
              <td>{T_HALF_I134_S/60:.1f} min</td>
              <td>7.80 %</td><td>847.0</td>
              <td>{BASELINE_I134:.1e}</td><td>{LIMIT_I134:.1e}</td>
            </tr>
            <tr>
              <td>Cs-137</td>
              <td>30.17 yr</td>
              <td>6.19 %</td><td>661.7</td>
              <td>{BASELINE_CS137:.1e}</td><td>{LIMIT_CS137:.1e}</td>
            </tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_hdr("B", "VVER-1200 Reactor Parameters")
        st.markdown(f"""
        <table class="ref-table">
          <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td>Thermal power</td><td>3200 MW</td></tr>
            <tr><td>Electrical output</td><td>1200 MWe</td></tr>
            <tr><td>Primary pressure</td><td>~160 bar</td></tr>
            <tr><td>Coolant inlet temp.</td><td>291 °C</td></tr>
            <tr><td>Coolant outlet temp.</td><td>321 °C</td></tr>
            <tr><td>Primary coolant volume</td><td>~200 m³</td></tr>
            <tr><td>Fuel assemblies</td><td>163</td></tr>
            <tr><td>Fuel rods (твэл) per assembly</td><td>312</td></tr>
            <tr><td>Total fuel rods</td><td>~50,000</td></tr>
            <tr><td>Cladding material</td><td>Zr-1%Nb alloy E110</td></tr>
            <tr><td>Coolant chemistry</td><td>H₃BO₃ + KOH buffer</td></tr>
            <tr><td>Coolant pH target</td><td>{PH_MIN}–{PH_MAX}</td></tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)

    with col_r2:
        section_hdr("C", "Detection Signal Priority")
        st.markdown(f"""
        <table class="ref-table">
          <thead>
            <tr><th>Signal</th><th>Sensitivity</th><th>Specificity</th><th>Typical lead time</th></tr>
          </thead>
          <tbody>
            <tr><td>I-134/I-131 ratio</td><td>High</td><td>High</td><td>30–90 min</td></tr>
            <tr><td>dI-134/dt (30 min)</td><td>Highest</td><td>Medium</td><td>15–60 min</td></tr>
            <tr><td>DNS elevation</td><td>Low</td><td>Highest</td><td>Near zero</td></tr>
            <tr><td>Cs-137 trend</td><td>Low</td><td>Medium</td><td>Lagging (hours)</td></tr>
            <tr><td>pH drop</td><td>Very low</td><td>Low</td><td>Lagging</td></tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_hdr("D", "Cladding Failure Sequence")
        st.markdown(f"""
        <table class="ref-table">
          <thead><tr><th>Phase</th><th>Time</th><th>Observable signal</th></tr></thead>
          <tbody>
            <tr><td>Micro-crack initiation</td><td>t − 2 h to t = 0</td>
                <td>Slight I-134 rise, DNS baseline drift</td></tr>
            <tr><td>Through-wall penetration</td><td>t = 0</td>
                <td>I-134 spike (gap release), DNS jump</td></tr>
            <tr><td>I-134/I-131 ratio peak</td><td>t + 0.5 to 2 h</td>
                <td>Ratio > 10 — primary alarm criterion</td></tr>
            <tr><td>I-131 matrix diffusion</td><td>t + 2 to 6 h</td>
                <td>I-131 rises; absolute level may breach limit</td></tr>
            <tr><td>Cs-137 accumulation</td><td>t + 6 h+</td>
                <td>Slow Cs-137 rise; quantifies breach size</td></tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_hdr("E", "Sources")
        st.markdown(f"""
        <table class="ref-table">
          <thead><tr><th>Reference</th><th>Title</th></tr></thead>
          <tbody>
            <tr><td>IAEA-TECDOC-1328</td><td>Fuel Failure in Water Cooled Reactors</td></tr>
            <tr><td>IAEA NS-G-2.2</td><td>Operational Limits and Conditions</td></tr>
            <tr><td>NUREG/CR-6365</td><td>Fission Product Release from Fuel</td></tr>
            <tr><td>IAEA-TECDOC-1195</td><td>Assessment and Management of Fuel Failures</td></tr>
          </tbody>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.68rem;color:{C["text2"]};letter-spacing:0.04em;'
        f'text-transform:uppercase;text-align:center;padding:1rem 0 0.5rem 0;">'
        f'Synthetic data only — for educational and research purposes</div>',
        unsafe_allow_html=True,
    )
