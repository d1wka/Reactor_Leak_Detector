"""
Real-time matplotlib monitoring dashboard for VVER-1200 coolant radiochemistry.

Four-panel layout (shared time axis):
  Panel 0 — I-131 and I-134 absolute concentrations (log scale, Bq/cm³)
  Panel 1 — I-134/I-131 isotope ratio (early-warning indicator, log scale)
  Panel 2 — Composite alert score [0–1] with GREEN / CAUTION / ALARM zones
  Panel 3 — Delayed Neutron Signal (DNS) normalized count rate

Vertical annotations:
  Green dashed line  — true leak start (ground truth from generator)
  Red dashed line    — first sustained alert (model detection)
  Lead-time label    — minutes of early warning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

from utils.physics import (
    LIMIT_I131,
    LIMIT_I134,
    LIMIT_I134_I131_RATIO,
    LIMIT_DNS,
    BASELINE_DNS,
)

# ── Dark-theme palette ────────────────────────────────────────────────────────
_BG_FIGURE = "#0d1117"
_BG_AXES = "#161b22"
_GRID = "#30363d"
_I131_COLOR = "#58a6ff"   # blue
_I134_COLOR = "#ff7b72"   # red-orange
_RATIO_COLOR = "#e3b341"  # amber
_SCORE_COLOR = "#bc8cff"  # purple
_DNS_COLOR = "#56d364"    # green
_LEAK_LINE = "#39d353"    # bright green — true event
_ALERT_LINE = "#f85149"   # bright red   — model alert


def _style_ax(ax: plt.Axes) -> None:
    """Apply dark control-room styling to a single axes."""
    ax.set_facecolor(_BG_AXES)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.grid(axis="y", color=_GRID, linewidth=0.4, linestyle="--")


def _add_event_lines(
    ax: plt.Axes,
    t_hours: np.ndarray,
    leak_idx: int,
    alert_idx: Optional[int],
) -> None:
    """Overlay the true-leak and first-alert vertical lines on an axes."""
    ax.axvline(
        t_hours[leak_idx],
        color=_LEAK_LINE,
        linewidth=1.4,
        linestyle="--",
        alpha=0.85,
        label="True leak start",
        zorder=5,
    )
    if alert_idx is not None:
        ax.axvline(
            t_hours[alert_idx],
            color=_ALERT_LINE,
            linewidth=1.4,
            linestyle="--",
            alpha=0.85,
            label="First alert",
            zorder=5,
        )


def build_dashboard(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    detection_results: dict[str, np.ndarray],
    leak_sample_idx: int,
    first_alert_idx: Optional[int],
    lead_time_minutes: Optional[int],
    alert_threshold: float = 0.5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render the four-panel monitoring dashboard.

    Args:
        raw_df: original sensor data (from SensorDataGenerator).
        processed_df: feature-engineered data (from CoolantSignalPreprocessor).
        detection_results: output of LeakDetector.predict().
        leak_sample_idx: index of the true cladding failure onset.
        first_alert_idx: index of the first sustained model alert; None if missed.
        lead_time_minutes: positive = alert before leak; None if missed.
        alert_threshold: threshold value drawn on the alert-score panel.
        save_path: optional file path to save the figure as PNG.

    Returns:
        matplotlib Figure (caller is responsible for plt.show() if needed).
    """
    # ── Time axis ─────────────────────────────────────────────────────────────
    ts = pd.to_datetime(raw_df["timestamp"].values)
    t_hours = (ts - ts[0]).total_seconds().values / 3600.0

    fig, axes = plt.subplots(4, 1, figsize=(17, 14), sharex=True)
    fig.patch.set_facecolor(_BG_FIGURE)

    for ax in axes:
        _style_ax(ax)

    # ── Panel 0 · I-131 and I-134 absolute concentrations ────────────────────
    ax = axes[0]
    ax.semilogy(t_hours, raw_df["I131"].values, color=_I131_COLOR,
                linewidth=0.75, alpha=0.8, label="I-131")
    ax.semilogy(t_hours, raw_df["I134"].values, color=_I134_COLOR,
                linewidth=0.75, alpha=0.8, label="I-134")

    # Operational action limits (dashed, matching isotope colours)
    ax.axhline(LIMIT_I131, color=_I131_COLOR, linewidth=0.8,
               linestyle=":", alpha=0.55, label=f"I-131 limit {LIMIT_I131:.1e}")
    ax.axhline(LIMIT_I134, color=_I134_COLOR, linewidth=0.8,
               linestyle=":", alpha=0.55, label=f"I-134 limit {LIMIT_I134:.1e}")

    _add_event_lines(ax, t_hours, leak_sample_idx, first_alert_idx)

    ax.set_ylabel("Activity (Bq/cm³)", fontsize=9)
    ax.set_title(
        "Primary Coolant Iodine Activity  ·  VVER-1200 Radiochemistry Monitor",
        fontsize=10, pad=4,
    )
    ax.legend(loc="upper left", fontsize=7, framealpha=0.25,
              labelcolor="white", facecolor=_BG_AXES, edgecolor=_GRID)

    # ── Panel 1 · I-134/I-131 ratio ──────────────────────────────────────────
    ax = axes[1]
    ratio = processed_df["ratio_I134_I131"].values

    ax.semilogy(t_hours, np.maximum(ratio, 1e-3), color=_RATIO_COLOR,
                linewidth=0.85, label="I-134 / I-131 ratio")

    # Threshold line at ratio = 10 (IAEA early-warning criterion)
    ax.axhline(LIMIT_I134_I131_RATIO, color=_ALERT_LINE, linewidth=1.0,
               linestyle="--", alpha=0.8,
               label=f"Threshold = {LIMIT_I134_I131_RATIO:.0f}")

    # Shade the area above the threshold to make exceedances visible at a glance
    above = np.maximum(ratio, LIMIT_I134_I131_RATIO)
    ax.fill_between(t_hours, LIMIT_I134_I131_RATIO, above,
                    where=(ratio >= LIMIT_I134_I131_RATIO),
                    color=_ALERT_LINE, alpha=0.18, label="Threshold exceeded")

    _add_event_lines(ax, t_hours, leak_sample_idx, first_alert_idx)

    ax.set_ylabel("I-134 / I-131", fontsize=9)
    ax.set_title(
        "Isotope Ratio  ·  Fresh Fuel Damage Early-Warning Indicator",
        fontsize=10, pad=4,
    )
    ax.legend(loc="upper left", fontsize=7, framealpha=0.25,
              labelcolor="white", facecolor=_BG_AXES, edgecolor=_GRID)

    # ── Panel 2 · Composite alert score ──────────────────────────────────────
    ax = axes[2]
    score = detection_results["alert_score"]
    ci_lo = detection_results["ci_lower"]
    ci_hi = detection_results["ci_upper"]

    # Colour zones: green (normal), amber (caution), red (alarm)
    ax.axhspan(0.0,             0.30,            color="#1a4a1a", alpha=0.35)
    ax.axhspan(0.30,            alert_threshold, color="#4a3a00", alpha=0.35)
    ax.axhspan(alert_threshold, 1.05,            color="#4a1a1a", alpha=0.35)

    ax.fill_between(t_hours, ci_lo, ci_hi,
                    color=_SCORE_COLOR, alpha=0.22, label="90 % CI")
    ax.plot(t_hours, score, color=_SCORE_COLOR, linewidth=0.9, label="Alert score")
    ax.axhline(alert_threshold, color=_ALERT_LINE, linewidth=1.0, linestyle="--",
               alpha=0.85, label=f"Alert threshold = {alert_threshold:.2f}")

    _add_event_lines(ax, t_hours, leak_sample_idx, first_alert_idx)

    # Zone text labels at right edge
    ax.text(0.99, 0.13, "NORMAL",  transform=ax.transAxes, ha="right",
            fontsize=8, color="#3fb950", alpha=0.85)
    ax.text(0.99, 0.44, "CAUTION", transform=ax.transAxes, ha="right",
            fontsize=8, color="#e3b341", alpha=0.85)
    ax.text(0.99, 0.82, "ALARM",   transform=ax.transAxes, ha="right",
            fontsize=8, color=_ALERT_LINE, alpha=0.85)

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Alert Score [0–1]", fontsize=9)
    ax.set_title(
        "Composite Anomaly Score  ·  Isolation Forest + Physics Rules",
        fontsize=10, pad=4,
    )
    ax.legend(loc="upper left", fontsize=7, framealpha=0.25,
              labelcolor="white", facecolor=_BG_AXES, edgecolor=_GRID)

    # Lead-time callout arrow
    if lead_time_minutes is not None and lead_time_minutes > 0 and first_alert_idx is not None:
        t_a = t_hours[first_alert_idx]
        ax.annotate(
            f"⚠ {lead_time_minutes} min\nearly warning",
            xy=(t_a, alert_threshold + 0.04),
            xytext=(t_a - max(1.5, t_hours[-1] * 0.06), alert_threshold + 0.22),
            color=_ALERT_LINE,
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color=_ALERT_LINE, lw=1.2),
        )

    # ── Panel 3 · Delayed Neutron Signal ─────────────────────────────────────
    ax = axes[3]
    dns_vals = raw_df["DNS"].values

    ax.plot(t_hours, dns_vals, color=_DNS_COLOR, linewidth=0.75,
            alpha=0.85, label="DNS (normalized)")
    ax.axhline(LIMIT_DNS, color=_ALERT_LINE, linewidth=0.9, linestyle="--",
               alpha=0.75, label=f"Action level = {LIMIT_DNS}")

    ax.fill_between(t_hours, BASELINE_DNS,
                    np.where(dns_vals > LIMIT_DNS, dns_vals, BASELINE_DNS),
                    color=_ALERT_LINE, alpha=0.22,
                    label="Above action level")

    _add_event_lines(ax, t_hours, leak_sample_idx, first_alert_idx)

    ax.set_ylabel("DNS (norm.)", fontsize=9)
    ax.set_title(
        "Delayed Neutron Signal  ·  Direct Fuel-Coolant Contact Indicator",
        fontsize=10, pad=4,
    )
    ax.set_xlabel("Time (hours from simulation start)", fontsize=9, color="white")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.25,
              labelcolor="white", facecolor=_BG_AXES, edgecolor=_GRID)

    # ── Figure title ─────────────────────────────────────────────────────────
    fig.suptitle(
        "VVER-1200  ·  Fuel Rod Cladding Failure (ТВЭЛ Разгерметизация) — Early Detection",
        color="white", fontsize=11, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.993])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[dashboard] Saved → {save_path}")

    return fig
