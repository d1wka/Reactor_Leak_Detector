"""
Hybrid anomaly detector for VVER-1200 fuel cladding failure detection.

Architecture:
  ┌──────────────────────────────────────────────────────┐
  │   Raw + engineered features                          │
  │        │                     │                       │
  │  Isolation Forest      Physics Rule Engine           │
  │  (unsupervised ML)    (domain-expert rules)          │
  │        │                     │                       │
  │    ml_score (0-1)      physics_score (0-1)           │
  │        └──────── weighted blend ──────────┘          │
  │                   alert_score (0-1)                  │
  └──────────────────────────────────────────────────────┘

The physics rules fire on the three strongest physical signatures of cladding
failure; the ML model catches multivariate patterns that no single rule covers.
Weighting physics at 60 % reflects the high reliability of isotope ratios when
a well-calibrated gamma spectrometer is available.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Optional

from utils.physics import (
    LIMIT_I134_I131_RATIO,
    LIMIT_DNS,
    BASELINE_I134,
)


class LeakDetector:
    """
    Hybrid detector combining Isolation Forest and physics-based rules.

    Training:
        Call fit(X_normal) with the feature matrix from the normal-operation
        window only (never include post-leak samples in training).

    Inference:
        Call predict(X, processed_df) on the full time series.
        Returns a dict of score arrays and binary alerts.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        physics_weight: float = 0.6,
        alert_threshold: float = 0.5,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            contamination: expected anomaly fraction for Isolation Forest.
                           Influences the decision boundary, not the score itself.
            physics_weight: α ∈ [0,1]; weight of physics score in the blend.
            alert_threshold: composite score above which an alert is issued.
            n_estimators: number of isolation trees.
            random_state: reproducibility seed.
        """
        self.contamination = contamination
        self.physics_weight = physics_weight
        self.alert_threshold = alert_threshold

        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X_normal: np.ndarray) -> "LeakDetector":
        """
        Train Isolation Forest on a clean normal-operation feature matrix.

        StandardScaler is fitted here so that inference uses the same
        normalization parameters as training — important for scores to stay
        consistent over the entire 72-hour window.
        """
        X_scaled = self._scaler.fit_transform(X_normal)
        self._model.fit(X_scaled)
        self._fitted = True
        return self

    # ── Physics Rule Engine (vectorized) ─────────────────────────────────────

    def _rule_ratio(self, df: pd.DataFrame) -> np.ndarray:
        """
        Rule 1 — I-134/I-131 ratio exceedance.

        A sigmoid centred on LIMIT_I134_I131_RATIO = 10 maps the ratio to [0,1].
        The sigmoid gradient (0.3) gives ~5 % score at ratio=5 and ~95 % at
        ratio=20, providing a smooth response rather than a hard step function.

        Physics: ratio > 10 indicates fresh fission-product burst into coolant,
        meaning cladding penetration occurred within the last few half-lives of
        I-134 (i.e. within ~3–4 hours).
        """
        ratio = df.get("ratio_I134_I131", pd.Series(np.zeros(len(df)), index=df.index))
        return (1.0 / (1.0 + np.exp(-0.3 * (ratio.values - LIMIT_I134_I131_RATIO)))).clip(0.0, 1.0)

    def _rule_i134_derivative(self, df: pd.DataFrame) -> np.ndarray:
        """
        Rule 2 — 30-minute rate of change of I-134.

        Normal fluctuations: |dI134/dt| < 5 % baseline / 30 min
        Gap-release event:   dI134/dt ≈ several hundred percent / 30 min

        We normalize the derivative by (10 % of baseline / 30 min) so that
        a rate equal to 10 % baseline/30 min produces a score of 1.0.
        This threshold is conservative — real events are 10–100× larger.
        """
        col = "I134_clipped_deriv_30min"
        if col not in df.columns:
            return np.zeros(len(df))
        norm_ref = BASELINE_I134 * 0.10   # 10 % of baseline per 30-min window
        score = np.abs(df[col].values) / norm_ref
        return np.clip(score, 0.0, 1.0)

    def _rule_dns(self, df: pd.DataFrame) -> np.ndarray:
        """
        Rule 3 — Delayed Neutron Signal elevation.

        DNS > action level (2.5 normalized) means delayed-neutron precursors
        (Br-87, Kr-87/88, Rb-87) are present in the coolant bulk.  These
        isotopes have half-lives of 0.3–56 s and can only reach the coolant
        in significant quantities if the fuel matrix is DIRECTLY exposed.

        DNS is the most SPECIFIC rule (very few false positives) but least
        SENSITIVE (only fires at moderate-to-large breaches).  Weighted low.
        """
        dns_col = "DNS_clipped" if "DNS_clipped" in df.columns else "DNS"
        dns = df[dns_col].values if dns_col in df.columns else np.ones(len(df))
        score = (dns - LIMIT_DNS) / LIMIT_DNS
        return np.clip(score, 0.0, 1.0)

    def _physics_score(self, processed_df: pd.DataFrame) -> np.ndarray:
        """
        Combine the three physics rules into a single score per time step.

        Weights reflect expert knowledge of signal characteristics:
          0.50 × ratio score      — most reliable, 30–90 min early warning
          0.35 × derivative score — earliest signal, catches the initial slope
          0.15 × DNS score        — most specific, fewest false positives
        """
        r1 = self._rule_ratio(processed_df)
        r2 = self._rule_i134_derivative(processed_df)
        r3 = self._rule_dns(processed_df)
        return np.clip(0.50 * r1 + 0.35 * r2 + 0.15 * r3, 0.0, 1.0)

    # ── ML Score ──────────────────────────────────────────────────────────────

    def _ml_score(self, X: np.ndarray) -> np.ndarray:
        """
        Map Isolation Forest output to [0, 1] where 1 = most anomalous.

        IsolationForest.score_samples() returns the negative average path length.
        Typical values: −0.55 to −0.35 for normal, −0.75 to −0.55 for anomalies.
        We shift and scale so that −0.35 → 0.0 and −0.75 → 1.0.
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fit() before predict().")
        X_scaled = self._scaler.transform(X)
        raw = self._model.score_samples(X_scaled)  # lower (more negative) = more anomalous
        # Linear rescale: normal ≈ -0.35 → 0, strong anomaly ≈ -0.75 → 1
        score = (-raw - 0.35) / 0.40
        return np.clip(score, 0.0, 1.0)

    # ── Composite Detection ───────────────────────────────────────────────────

    def predict(
        self, X: np.ndarray, processed_df: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """
        Compute per-sample alert scores and binary alerts.

        Args:
            X: feature matrix from CoolantSignalPreprocessor.get_feature_matrix()
            processed_df: the processed DataFrame (needed for named physics rules)

        Returns:
            Dictionary with keys:
              physics_score  – [0,1] array, rule-engine score
              ml_score       – [0,1] array, Isolation Forest score
              alert_score    – [0,1] composite blend
              alert          – binary (0/1) per sample
              ci_lower       – lower bound of 90 % confidence interval
              ci_upper       – upper bound of 90 % confidence interval
        """
        phys = self._physics_score(processed_df)
        ml = self._ml_score(X)

        alpha = self.physics_weight
        composite = alpha * phys + (1.0 - alpha) * ml

        # Propagate flat ±10 % uncertainty through the linear combination
        half_ci = 0.10 * alpha + 0.10 * (1.0 - alpha)   # = 0.10 always here
        ci_lo = np.clip(composite - half_ci, 0.0, 1.0)
        ci_hi = np.clip(composite + half_ci, 0.0, 1.0)

        alerts = (composite >= self.alert_threshold).astype(np.int8)

        return {
            "physics_score": phys,
            "ml_score": ml,
            "alert_score": composite,
            "alert": alerts,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
        }

    # ── Performance evaluation ─────────────────────────────────────────────────

    def find_first_alert(
        self, alerts: np.ndarray, min_consecutive: int = 3
    ) -> Optional[int]:
        """
        Return the index of the first SUSTAINED alert.

        Requires min_consecutive consecutive alert samples to filter out
        single-sample false positives caused by noise spikes.
        Cost: ≤ min_consecutive minutes of additional latency (acceptable).
        """
        streak = 0
        for i, flag in enumerate(alerts):
            if flag:
                streak += 1
                if streak >= min_consecutive:
                    return i - min_consecutive + 1
            else:
                streak = 0
        return None

    def compute_performance_metrics(
        self,
        alerts: np.ndarray,
        true_labels: np.ndarray,
        leak_start_idx: int,
    ) -> dict:
        """
        Compute detection performance against ground-truth labels.

        Args:
            alerts: binary alert array from predict()
            true_labels: ground-truth 0/1 array from the generator
            leak_start_idx: sample index of the true leak start

        Returns:
            Dict with: first_alert_idx, lead_time_minutes, false_positives,
                       false_positive_rate, precision, recall.
        """
        first_alert_idx = self.find_first_alert(alerts)

        pre_leak = alerts[:leak_start_idx]
        post_leak = alerts[leak_start_idx:]

        fp = int(pre_leak.sum())
        tp = int(post_leak.sum())
        fn = int((1 - post_leak).sum())

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        fp_rate = fp / max(1, leak_start_idx)

        lead_time: Optional[int] = None
        if first_alert_idx is not None:
            # Positive lead time = alert fired before the leak started
            lead_time = int(leak_start_idx - first_alert_idx)

        return {
            "first_alert_idx": first_alert_idx,
            "lead_time_minutes": lead_time,
            "false_positives": fp,
            "false_positive_rate": round(fp_rate, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }
