"""
Synthetic sensor data generator for VVER-1200 primary coolant monitoring.

Simulates up to 72 hours of radiological and chemistry sensor data,
then injects a realistic cladding failure (твэл разгерметизация) event
with physics-accurate time evolution of each fission product.

Leak progression modelled in three phases:
  Phase 1 — Pre-leak micro-crack drift  (~2 h before full penetration)
  Phase 2 — Gap inventory burst         (minutes 0–60 post-failure)
  Phase 3 — Fuel matrix diffusion       (hours 1–8+ post-failure)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from utils.physics import (
    LAMBDA_I131, LAMBDA_I134, LAMBDA_CS137,
    BASELINE_I131, BASELINE_I134, BASELINE_CS137,
    BASELINE_DNS, BASELINE_PH, BASELINE_CONDUCTIVITY,
)


class SensorDataGenerator:
    """
    Generates multivariate time-series sensor data for a VVER-1200 coolant loop.

    Produces a pandas DataFrame with one row per minute containing:
        timestamp, I131, I134, Cs137, pH, conductivity, DNS, label
    where label = 0 (normal) or 1 (post-leak-onset).
    """

    def __init__(
        self,
        duration_hours: int = 72,
        sample_interval_minutes: int = 1,
        leak_hour: Optional[float] = None,
        random_seed: int = 42,
    ) -> None:
        """
        Args:
            duration_hours: total simulation length in hours.
            sample_interval_minutes: acquisition interval (default 1 min).
            leak_hour: hour at which cladding failure starts;
                       None → random within [24, duration-12].
            random_seed: for reproducible runs.
        """
        self.duration_hours = duration_hours
        self.dt_min = sample_interval_minutes
        self.rng = np.random.default_rng(random_seed)

        self.n_samples: int = int(duration_hours * 60 / sample_interval_minutes)

        # Determine when the leak occurs
        if leak_hour is None:
            self.leak_hour = float(
                self.rng.uniform(24.0, max(25.0, duration_hours - 12.0))
            )
        else:
            self.leak_hour = float(leak_hour)

        self.leak_sample: int = int(self.leak_hour * 60 / sample_interval_minutes)

        # Build timestamp index
        t0 = datetime(2024, 1, 15, 0, 0, 0)
        self.timestamps = [
            t0 + timedelta(minutes=i * sample_interval_minutes)
            for i in range(self.n_samples)
        ]
        # Fractional hours for each sample (used in drift formulas)
        self.t_h = np.arange(self.n_samples) * sample_interval_minutes / 60.0

    # ── Normal-operation signal generators ───────────────────────────────────

    def _normal_i131(self) -> np.ndarray:
        """
        I-131 at secular equilibrium.

        Production in coolant (via neutron activation of Te-131, decay of
        I-131 precursors, and tramp uranium) equals removal by decay and
        continuous purification. The net result is a slow sinusoidal drift
        tracking reactor power fluctuations (~12-hour turbine-following cycle)
        on top of the steady-state baseline.
        """
        drift = 0.03 * BASELINE_I131 * np.sin(2 * np.pi * self.t_h / 12.0)
        noise = self.rng.normal(0.0, 0.05 * BASELINE_I131, self.n_samples)
        return BASELINE_I131 + drift + noise

    def _normal_i134(self) -> np.ndarray:
        """
        I-134 at secular equilibrium.

        Tracks neutron flux variations more closely than I-131 because its
        52.5-min half-life makes it respond quickly to power changes.
        Higher fractional noise (8 %) reflects rapid ingrowth/decay cycling.
        """
        drift = 0.05 * BASELINE_I134 * np.sin(2 * np.pi * self.t_h / 6.0)
        noise = self.rng.normal(0.0, 0.08 * BASELINE_I134, self.n_samples)
        return BASELINE_I134 + drift + noise

    def _normal_cs137(self) -> np.ndarray:
        """
        Cs-137 during normal operation.

        With T½ = 30.2 years, concentration is nearly constant over 72 hours.
        A very slow upward trend (~0.1 % per hour) models accumulation from
        activated corrosion products and tramp Cs from fuel assembly gaps.
        """
        trend = 1e-3 * BASELINE_CS137 * self.t_h
        noise = self.rng.normal(0.0, 0.02 * BASELINE_CS137, self.n_samples)
        return BASELINE_CS137 + trend + noise

    def _normal_dns(self) -> np.ndarray:
        """
        Delayed Neutron Signal (DNS) baseline.

        DNS measures delayed-neutron precursors (Br-87, Kr-87, Kr-88, etc.)
        in a bypass coolant flow. At normal levels these come from activation
        of O-17 and dissolved uranium impurities — not from the fuel matrix.
        Baseline ≈ 1.0 normalized count rate.
        """
        noise = self.rng.normal(0.0, 0.05, self.n_samples)
        return BASELINE_DNS + noise

    def _normal_ph(self) -> np.ndarray:
        """
        Coolant pH.

        Controlled by KOH injection against a boric acid background.
        24-hour sinusoidal drift models the dilution/boration cycle for
        load-following operation.
        """
        drift = 0.005 * np.sin(2 * np.pi * self.t_h / 24.0)
        noise = self.rng.normal(0.0, 0.02, self.n_samples)
        return BASELINE_PH + drift + noise

    def _normal_conductivity(self) -> np.ndarray:
        """
        Electrical conductivity.

        Proportional to total dissolved ionic species (mainly boric acid).
        Minor fluctuations from temperature and chemistry adjustments.
        """
        noise = self.rng.normal(0.0, 0.3, self.n_samples)
        return BASELINE_CONDUCTIVITY + noise

    # ── Leak event injection ──────────────────────────────────────────────────

    def _inject_leak(
        self,
        i131: np.ndarray,
        i134: np.ndarray,
        cs137: np.ndarray,
        dns: np.ndarray,
        ph: np.ndarray,
        conductivity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Superimpose a realistic cladding failure event on the baseline signals.

        ── Phase 1 · Pre-leak micro-crack drift (t = −2 h to 0) ──────────────
        Zirconium cladding develops through-wall micro-cracks before the major
        penetration occurs. Tiny amounts of fission gas + short-lived iodines
        leak out, causing a subtle pre-event rise in I-134 and DNS.

        ── Phase 2 · Gap inventory burst (t = 0 to ~1 h) ─────────────────────
        The fuel-to-cladding gap accumulates fission products over the fuel
        cycle lifetime. Upon penetration this inventory releases nearly
        instantaneously into the coolant. I-134 (high yield 7.8 %) dominates
        the initial spike. DNS jumps immediately due to delayed-neutron
        precursors flushing out of the gap.

        ── Phase 3 · Matrix diffusion (t = 1 h to 8 h+) ──────────────────────
        Fission products diffuse from the UO₂ fuel pellet bulk into the
        coolant. Diffusion is much slower than gap release. I-131 builds up
        over 2–4 hours (long diffusion path + re-trapping). Cs-137 rises even
        more slowly (lower diffusivity in UO₂). I-134 has mostly decayed by
        the time the matrix component is significant.
        """
        ls = self.leak_sample  # index alias

        # ── Phase 1: pre-leak drift ───────────────────────────────────────────
        pre_start = max(0, ls - int(120 / self.dt_min))  # 2 h before leak
        for i in range(pre_start, ls):
            frac = (i - pre_start) / max(1, ls - pre_start)  # 0 → 1 over 2 h
            # I-134 is most sensitive to micro-cracks (short half-life, high yield)
            i134[i] += frac * 0.4 * BASELINE_I134 * (
                1.0 + self.rng.normal(0.0, 0.08)
            )
            dns[i] += frac * 0.2 * (1.0 + self.rng.normal(0.0, 0.05))

        # ── Phases 2 & 3: post-failure transient ─────────────────────────────
        # Time in seconds measured from the leak start for every sample
        t_s = (np.arange(self.n_samples) - ls) * self.dt_min * 60.0
        post = np.arange(self.n_samples) >= ls   # boolean mask

        # I-134 spike: rise with τ_rise = 30 min (gap transport through coolant),
        # then decays with its own λ (52.5-min half-life dominates)
        tau_rise_i134 = 30.0 * 60.0   # s
        I134_peak = 20.0 * BASELINE_I134
        i134_spike = (
            I134_peak
            * (1.0 - np.exp(-t_s / tau_rise_i134))
            * np.exp(-LAMBDA_I134 * t_s)
        )
        i134_spike = np.where(post, i134_spike, 0.0)
        # Add proportional noise on the spike itself
        i134_noise = self.rng.normal(0.0, 0.05, self.n_samples) * (i134 + i134_spike)

        # I-131 matrix component: rises with τ = 3 h (bulk UO₂ diffusion)
        tau_rise_i131 = 3.0 * 3600.0  # s
        I131_peak = 8.0 * BASELINE_I131
        i131_leak = I131_peak * (1.0 - np.exp(-t_s / tau_rise_i131))
        i131_leak = np.where(post, i131_leak, 0.0)
        i131_noise = self.rng.normal(0.0, 0.03, self.n_samples) * (i131 + i131_leak)

        # DNS: sharp rise from gap release, sustained elevation from ongoing
        # precursor production at exposed fuel surface, then gradual decay
        DNS_peak = 4.0 * BASELINE_DNS
        dns_spike = (
            DNS_peak
            * (1.0 - np.exp(-t_s / (20.0 * 60.0)))     # 20-min rise
            * np.exp(-t_s / (4.0 * 3600.0))             # 4-h decay (precursors clear)
        )
        dns_spike = np.where(post, dns_spike, 0.0)

        # Cs-137: very slow matrix release (τ = 8 h) — lags everything else
        CS137_peak = 3.0 * BASELINE_CS137
        cs137_leak = CS137_peak * (1.0 - np.exp(-t_s / (8.0 * 3600.0)))
        cs137_leak = np.where(post, cs137_leak, 0.0)

        # pH drops slightly: iodine hydrolysis in water reduces pH
        # I₂ + H₂O → HIO + HI, producing H⁺
        ph_drop = -0.10 * (1.0 - np.exp(-t_s / (2.0 * 3600.0)))
        ph_drop = np.where(post, ph_drop, 0.0)

        # Conductivity rises: dissolved iodine species add ionic strength
        cond_rise = 1.5 * (1.0 - np.exp(-t_s / (3.0 * 3600.0)))
        cond_rise = np.where(post, cond_rise, 0.0)

        return (
            i131 + i131_leak + i131_noise,
            i134 + i134_spike + i134_noise,
            cs137 + cs137_leak,
            dns + dns_spike,
            ph + ph_drop,
            conductivity + cond_rise,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        """
        Build the complete sensor DataFrame.

        Returns:
            DataFrame with columns:
                timestamp   – UTC datetime (1-min cadence)
                I131        – Bq/cm³
                I134        – Bq/cm³
                Cs137       – Bq/cm³
                pH          – dimensionless
                conductivity – μS/cm
                DNS         – normalized count rate
                label       – 0 (normal) | 1 (anomalous, post-leak)
        """
        i131 = self._normal_i131()
        i134 = self._normal_i134()
        cs137 = self._normal_cs137()
        dns = self._normal_dns()
        ph = self._normal_ph()
        conductivity = self._normal_conductivity()

        i131, i134, cs137, dns, ph, conductivity = self._inject_leak(
            i131, i134, cs137, dns, ph, conductivity
        )

        # Physical floor: activity and DNS cannot be negative
        i131 = np.maximum(i131, 0.0)
        i134 = np.maximum(i134, 0.0)
        cs137 = np.maximum(cs137, 0.0)
        dns = np.maximum(dns, 0.0)

        labels = np.zeros(self.n_samples, dtype=np.int8)
        labels[self.leak_sample:] = 1

        return pd.DataFrame({
            "timestamp": self.timestamps,
            "I131": i131,
            "I134": i134,
            "Cs137": cs137,
            "pH": ph,
            "conductivity": conductivity,
            "DNS": dns,
            "label": labels,
        })
