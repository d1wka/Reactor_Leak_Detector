"""
3D schematic model of the VVER-1200 reactor for the Streamlit monitoring dashboard.

All geometry is built from Plotly Surface (cylinders) and Scatter3d (pipes, labels).
Each major component is color-coded in real time based on the alert score derived
from the sensors that physically belong to that part of the reactor circuit.

VVER-1200 primary circuit layout (top view):
                    SG-N
                     |
         SG-W ---  RPV  --- SG-E
                     |
                    SG-S

4 primary loops, 4 steam generators, 4 main coolant pumps, 1 pressurizer,
1 coolant activity monitor (bypass line), 1 DNS monitor (bypass line).
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# State → color mapping
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = {
    "normal":   dict(surface="#0b2233", accent="#1a4f6a", text="#39c5cf", glow="#39c5cf"),
    "caution":  dict(surface="#251a00", accent="#5c3c00", text="#d29922", glow="#d29922"),
    "alarm":    dict(surface="#280808", accent="#661010", text="#f85149", glow="#f85149"),
    "selected": dict(surface="#0f2660", accent="#1e4fc2", text="#79c0ff", glow="#388bfd"),
    "inactive": dict(surface="#0d1520", accent="#1a2535", text="#484f58", glow="#30363d"),
}


def _state(score: float, threshold: float, is_selected: bool) -> dict:
    if is_selected:
        return _COLORS["selected"]
    if score >= threshold:
        return _COLORS["alarm"]
    if score >= 0.30:
        return _COLORS["caution"]
    return _COLORS["normal"]


# ─────────────────────────────────────────────────────────────────────────────
# Geometry primitives
# ─────────────────────────────────────────────────────────────────────────────

def _cylinder(
    cx: float, cy: float, z0: float, z1: float, r: float,
    surf_col: str, cap_col: str,
    opacity: float = 0.88, n: int = 36,
) -> list:
    """
    Closed cylinder = lateral surface + top cap + bottom cap.
    Returns a list of go.Surface traces.
    """
    traces = []
    theta = np.linspace(0, 2 * np.pi, n)
    z_arr = np.array([z0, z1])
    T, Z = np.meshgrid(theta, z_arr)
    X = cx + r * np.cos(T)
    Y = cy + r * np.sin(T)
    C = np.zeros_like(X)

    traces.append(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=C,
        colorscale=[[0, surf_col], [1, surf_col]],
        showscale=False, opacity=opacity,
        hoverinfo="skip", showlegend=False,
    ))

    # Circular caps
    r_arr = np.linspace(0, r, 6)
    T_c, R_c = np.meshgrid(np.linspace(0, 2 * np.pi, n), r_arr)
    X_c = cx + R_c * np.cos(T_c)
    Y_c = cy + R_c * np.sin(T_c)
    C_c = np.zeros_like(X_c)
    for z_val in (z0, z1):
        Z_c = np.full_like(X_c, z_val)
        traces.append(go.Surface(
            x=X_c, y=Y_c, z=Z_c,
            surfacecolor=C_c,
            colorscale=[[0, cap_col], [1, cap_col]],
            showscale=False, opacity=min(1.0, opacity + 0.08),
            hoverinfo="skip", showlegend=False,
        ))
    return traces


def _pipe(points: list[tuple], color: str, width: int = 8) -> go.Scatter3d:
    """Pipe segment as a thick 3D line."""
    xs, ys, zs = zip(*points)
    return go.Scatter3d(
        x=list(xs), y=list(ys), z=list(zs),
        mode="lines",
        line=dict(color=color, width=width),
        showlegend=False, hoverinfo="skip",
    )


def _label(
    x: float, y: float, z: float,
    lines: list[str], color: str,
    marker_size: int = 5,
    font_size: int = 10,
) -> go.Scatter3d:
    """Floating text label at a 3D point."""
    text = "<br>".join(lines)
    return go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="markers+text",
        text=[text],
        textposition="top center",
        textfont=dict(
            color=color, size=font_size,
            family="JetBrains Mono, Courier New, monospace",
        ),
        marker=dict(size=marker_size, color=color, opacity=0.9,
                    symbol="circle"),
        showlegend=False, hoverinfo="skip",
    )


def _clickable(x: float, y: float, z: float, name: str, color: str) -> go.Scatter3d:
    """Invisible large marker that registers hover/click for a component."""
    return go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="markers",
        marker=dict(size=18, color=color, opacity=0.001),
        name=name,
        hovertemplate=f"<b>{name}</b><extra></extra>",
        showlegend=False,
    )


def _sphere_surface(
    cx: float, cy: float, cz: float, r: float,
    color: str, opacity: float = 0.08,
    n_phi: int = 24, n_theta: int = 32,
) -> go.Surface:
    """Translucent containment sphere/dome."""
    phi = np.linspace(0, np.pi, n_phi)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    P, T = np.meshgrid(phi, theta)
    X = cx + r * np.sin(P) * np.cos(T)
    Y = cy + r * np.sin(P) * np.sin(T)
    Z = cz + r * np.cos(P)
    C = np.zeros_like(X)
    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=C,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=opacity,
        hoverinfo="skip", showlegend=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-component score calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_component_scores(
    raw_df,
    processed_df,
    time_idx: int,
    alert_threshold: float,
) -> dict[str, float]:
    """
    Derive per-component alert scores from sensor readings at a single timestep.

    Mapping follows the physical role of each component:
        core     — fuel rods are the source of fission products →
                   governed by I-134/I-131 ratio and DNS
        loop     — primary coolant pipes carry the released activity →
                   governed by absolute I-131 and I-134 levels
        activity — coolant activity monitor (bypass line) →
                   combines ratio + derivative signal
        dns      — delayed neutron monitor (bypass line) →
                   governed by DNS alone
        chemistry — pressurizer + CVCS control pH and conductivity →
                   governed by pH deviation and conductivity rise
    """
    from utils.physics import (
        LIMIT_I134_I131_RATIO, LIMIT_DNS,
        BASELINE_I131, BASELINE_I134, BASELINE_CONDUCTIVITY, BASELINE_PH,
        LIMIT_I131, LIMIT_I134,
    )

    rp = processed_df.iloc[time_idx]
    rr = raw_df.iloc[time_idx]

    def clamp(v):
        return float(np.clip(v, 0.0, 1.0))

    # I-134/I-131 ratio → sigmoid centred on action limit 10
    ratio = float(rp.get("ratio_I134_I131", 0.0))
    ratio_sc = clamp(1.0 / (1.0 + np.exp(-0.30 * (ratio - LIMIT_I134_I131_RATIO))))

    # 30-min I-134 derivative
    deriv = abs(float(rp.get("I134_clipped_deriv_30min", 0.0)))
    deriv_sc = clamp(deriv / (BASELINE_I134 * 0.10))

    # DNS absolute level
    dns_raw = float(rr.get("DNS", 1.0))
    dns_sc = clamp((dns_raw - LIMIT_DNS) / LIMIT_DNS)

    # Absolute I-131 / I-134 fractional exceedance
    i131_sc = clamp((float(rr.get("I131", 0)) - BASELINE_I131) /
                    max(1.0, LIMIT_I131 - BASELINE_I131))
    i134_sc = clamp((float(rr.get("I134", 0)) - BASELINE_I134) /
                    max(1.0, LIMIT_I134 - BASELINE_I134))

    # pH deviation from target 7.2
    ph_sc = clamp(abs(float(rr.get("pH", 7.2)) - BASELINE_PH) / 0.20)

    # Conductivity rise above baseline
    cond_sc = clamp((float(rr.get("conductivity", BASELINE_CONDUCTIVITY))
                     - BASELINE_CONDUCTIVITY) / 5.0)

    return {
        "core":      clamp(0.55 * ratio_sc + 0.30 * dns_sc + 0.15 * deriv_sc),
        "loop":      clamp(0.50 * i134_sc + 0.35 * i131_sc + 0.15 * deriv_sc),
        "activity":  clamp(0.50 * ratio_sc + 0.35 * deriv_sc + 0.15 * i134_sc),
        "dns":       dns_sc,
        "chemistry": clamp(0.60 * ph_sc + 0.40 * cond_sc),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main figure builder
# ─────────────────────────────────────────────────────────────────────────────

def build_reactor_3d(
    component_scores: dict[str, float],
    selected: Optional[str],
    alert_threshold: float,
    latest: dict,         # raw sensor values at current timestep
    camera: Optional[dict] = None,
) -> go.Figure:
    """
    Construct the full 3D VVER-1200 schematic figure.

    Args:
        component_scores : scores from compute_component_scores()
        selected         : currently selected component key (or None)
        alert_threshold  : threshold for alarm coloring
        latest           : raw sensor values dict for label annotations
        camera           : optional Plotly camera dict (preserves user rotation)

    Returns:
        go.Figure ready for st.plotly_chart()
    """
    fig = go.Figure()

    def col(key: str) -> dict:
        return _state(component_scores.get(key, 0.0), alert_threshold, selected == key)

    # ── Geometry constants (metres, schematic scale) ─────────────────────────
    RPV_R    = 2.20; RPV_Z0 = 0.0; RPV_Z1 = 12.0
    FUEL_R   = 1.00; FUEL_Z0 = 0.8; FUEL_Z1 = 10.5
    SG_R     = 1.50; SG_Z0  = 0.0; SG_Z1  = 15.0
    SG_DIST  = 11.0                          # SG centre-to-RPV-centre distance
    PUMP_R   = 0.65; PUMP_Z0 = 0.5; PUMP_Z1 = 3.8
    PUMP_D   = 6.50                          # pump distance from origin
    PZ_R     = 0.85; PZ_Z0  = 8.5; PZ_Z1  = 22.0; PZ_X = 4.5; PZ_Y = 3.5
    AM_R     = 0.55; AM_Z0  = 3.0; AM_Z1  = 7.0;  AM_X = -4.5; AM_Y = -4.5
    DNS_R    = 0.40; DNS_Z0 = 2.0; DNS_Z1 = 6.0;  DNS_X = -3.0; DNS_Y = -6.0
    DOME_R   = 21.0; DOME_CZ = 5.0

    # Cardinal loop positions (label, cx, cy, direction unit vector)
    SG_POS = [
        ("SG-N",  0,        SG_DIST,  0,  1),
        ("SG-E",  SG_DIST,  0,        1,  0),
        ("SG-S",  0,       -SG_DIST,  0, -1),
        ("SG-W", -SG_DIST,  0,       -1,  0),
    ]
    PUMP_POS = [
        (0, PUMP_D), (PUMP_D, 0), (0, -PUMP_D), (-PUMP_D, 0)
    ]

    # ── Containment dome (drawn first — behind everything) ───────────────────
    fig.add_trace(_sphere_surface(
        0, 0, DOME_CZ, DOME_R,
        color="#1a3a5c", opacity=0.07,
    ))

    # ── Reactor Pressure Vessel (RPV) ─────────────────────────────────────────
    c = col("core")
    for tr in _cylinder(0, 0, RPV_Z0, RPV_Z1, RPV_R,
                        c["surface"], c["accent"], opacity=0.90):
        fig.add_trace(tr)

    # Inner fuel zone (semi-transparent, shows through RPV)
    cf = col("core")
    for tr in _cylinder(0, 0, FUEL_Z0, FUEL_Z1, FUEL_R,
                        cf["surface"], cf["glow"], opacity=0.70):
        fig.add_trace(tr)

    # Fuel rod bundle hint (vertical lines inside fuel zone)
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        xf = 0.60 * np.cos(angle)
        yf = 0.60 * np.sin(angle)
        fig.add_trace(_pipe(
            [(xf, yf, FUEL_Z0 + 0.3), (xf, yf, FUEL_Z1 - 0.3)],
            cf["glow"], width=3,
        ))

    # ── Steam generators ──────────────────────────────────────────────────────
    cl = col("loop")
    for sg_name, sx, sy, _, _ in SG_POS:
        for tr in _cylinder(sx, sy, SG_Z0, SG_Z1, SG_R,
                             cl["surface"], cl["accent"], opacity=0.88):
            fig.add_trace(tr)

    # ── Main coolant pumps ────────────────────────────────────────────────────
    for px, py in PUMP_POS:
        for tr in _cylinder(px, py, PUMP_Z0, PUMP_Z1, PUMP_R,
                             cl["surface"], cl["accent"], opacity=0.88):
            fig.add_trace(tr)

    # ── Pressurizer ───────────────────────────────────────────────────────────
    cc = col("chemistry")
    for tr in _cylinder(PZ_X, PZ_Y, PZ_Z0, PZ_Z1, PZ_R,
                        cc["surface"], cc["accent"], opacity=0.92):
        fig.add_trace(tr)

    # ── Coolant Activity Monitor ───────────────────────────────────────────────
    ca = col("activity")
    for tr in _cylinder(AM_X, AM_Y, AM_Z0, AM_Z1, AM_R,
                        ca["surface"], ca["accent"], opacity=0.92):
        fig.add_trace(tr)

    # ── DNS Monitor ───────────────────────────────────────────────────────────
    cd = col("dns")
    for tr in _cylinder(DNS_X, DNS_Y, DNS_Z0, DNS_Z1, DNS_R,
                        cd["surface"], cd["accent"], opacity=0.92):
        fig.add_trace(tr)

    # ── Primary loop pipes ────────────────────────────────────────────────────
    pc = cl["accent"]  # pipe color follows loop state
    hot_z  = 9.0       # hot leg elevation
    cold_z = 3.2       # cold leg elevation

    for _, sx, sy, nx, ny in SG_POS:
        # Hot leg: RPV nozzle → SG inlet
        fig.add_trace(_pipe([
            (nx * RPV_R, ny * RPV_R, hot_z),
            (nx * (SG_DIST - SG_R), ny * (SG_DIST - SG_R), hot_z),
        ], pc, width=9))

        # Cold leg: SG outlet → pump inlet
        fig.add_trace(_pipe([
            (nx * (SG_DIST - SG_R), ny * (SG_DIST - SG_R), cold_z),
            (nx * (PUMP_D + PUMP_R), ny * (PUMP_D + PUMP_R), cold_z),
        ], pc, width=9))

        # Pump → RPV inlet
        fig.add_trace(_pipe([
            (nx * (PUMP_D - PUMP_R), ny * (PUMP_D - PUMP_R), cold_z),
            (nx * RPV_R, ny * RPV_R, cold_z),
        ], pc, width=9))

        # Vertical drop inside SG (hot → cold level)
        fig.add_trace(_pipe([
            (nx * (SG_DIST - SG_R), ny * (SG_DIST - SG_R), hot_z),
            (nx * (SG_DIST - SG_R), ny * (SG_DIST - SG_R), cold_z),
        ], pc, width=6))

    # Pressurizer surge line (RPV nozzle → pressurizer base)
    fig.add_trace(_pipe([
        (RPV_R * 0.7, RPV_R * 0.7, hot_z),
        (PZ_X, PZ_Y, hot_z),
        (PZ_X, PZ_Y, PZ_Z0),
    ], cc["accent"], width=5))

    # Activity monitor bypass lines
    fig.add_trace(_pipe([
        (0, -RPV_R, 5.0),
        (AM_X, AM_Y, 5.0),
        (AM_X, AM_Y, AM_Z0),
    ], ca["accent"], width=5))

    # DNS bypass branch from activity monitor
    fig.add_trace(_pipe([
        (AM_X, AM_Y, 4.5),
        (DNS_X, DNS_Y, 4.5),
        (DNS_X, DNS_Y, DNS_Z0),
    ], cd["accent"], width=4))

    # ── Component labels ──────────────────────────────────────────────────────
    ratio_v = float(latest.get("ratio_I134_I131", 0.0))
    i131_v  = float(latest.get("I131",  0.0))
    i134_v  = float(latest.get("I134",  0.0))
    dns_v   = float(latest.get("DNS",   1.0))
    ph_v    = float(latest.get("pH",    7.2))
    cond_v  = float(latest.get("conductivity", 15.0))

    # Core
    fig.add_trace(_label(0, 0, RPV_Z1 + 1.8, [
        "REACTOR CORE",
        f"I-134/I-131  {ratio_v:>7.2f}",
        f"DNS          {dns_v:>7.4f}",
    ], col("core")["text"], marker_size=6, font_size=10))

    # Steam generators (label the north one in detail, rest abbreviated)
    fig.add_trace(_label(0, SG_DIST, SG_Z1 + 1.5, [
        "SG-N / PRIMARY LOOP",
        f"I-134  {i134_v:.2e} Bq/cm³",
        f"I-131  {i131_v:.2e} Bq/cm³",
    ], col("loop")["text"], marker_size=4, font_size=9))
    for sg_name, sx, sy, _, _ in SG_POS[1:]:
        fig.add_trace(_label(sx, sy, SG_Z1 + 1.5, [sg_name],
                             col("loop")["text"], marker_size=4, font_size=9))

    # Pressurizer
    fig.add_trace(_label(PZ_X, PZ_Y, PZ_Z1 + 1.2, [
        "PRESSURIZER",
        f"pH     {ph_v:.3f}",
        f"σ      {cond_v:.2f} µS/cm",
    ], col("chemistry")["text"], marker_size=4, font_size=9))

    # Activity monitor
    fig.add_trace(_label(AM_X, AM_Y, AM_Z1 + 1.2, [
        "ACTIVITY MONITOR",
        f"I-134/I-131  {ratio_v:.2f}",
    ], col("activity")["text"], marker_size=4, font_size=9))

    # DNS monitor
    fig.add_trace(_label(DNS_X, DNS_Y, DNS_Z1 + 1.2, [
        "DNS MONITOR",
        f"{dns_v:.4f}",
    ], col("dns")["text"], marker_size=4, font_size=9))

    # ── Clickable invisible markers (for hover names) ─────────────────────────
    for key, x, y, z, name in [
        ("core",      0,     0,     7.0,  "Reactor Core (ТВЭЛ)"),
        ("loop",      0,  SG_DIST,  7.5,  "Primary Coolant Loop"),
        ("chemistry", PZ_X, PZ_Y,  15.0,  "Pressurizer / Chemistry"),
        ("activity",  AM_X, AM_Y,   5.0,  "Coolant Activity Monitor"),
        ("dns",      DNS_X, DNS_Y,  4.0,  "DNS Monitor"),
    ]:
        fig.add_trace(_clickable(x, y, z, name, col(key)["glow"]))

    # ── Scene layout ──────────────────────────────────────────────────────────
    eye = camera.get("eye", dict(x=1.7, y=1.4, z=0.75)) if camera else dict(x=1.7, y=1.4, z=0.75)

    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title="",
                       backgroundcolor="rgba(0,0,0,0)", showline=False,
                       zeroline=False, showspikes=False),
            yaxis=dict(showgrid=False, showticklabels=False, title="",
                       backgroundcolor="rgba(0,0,0,0)", showline=False,
                       zeroline=False, showspikes=False),
            zaxis=dict(showgrid=False, showticklabels=False, title="",
                       backgroundcolor="rgba(0,0,0,0)", showline=False,
                       zeroline=False, showspikes=False),
            bgcolor="rgba(6,8,13,0.0)",
            camera=dict(eye=eye, up=dict(x=0, y=0, z=1)),
            aspectratio=dict(x=1.0, y=1.0, z=0.65),
            dragmode="orbit",
        ),
        paper_bgcolor="#06080d",
        margin=dict(l=0, r=0, t=0, b=0),
        height=640,
        showlegend=False,
        uirevision="reactor",   # keeps camera stable between re-renders
    )
    return fig
