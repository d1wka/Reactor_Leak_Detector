"""
Signal preprocessing pipeline for VVER-1200 coolant sensor data.

All transformations are physics-motivated:
  1. 3-sigma outlier clipping       → removes detector glitches / cosmic-ray hits
  2. Rolling z-score normalization  → removes slow power-level drift, makes
                                      signal stationary for the ML model
  3. I-134/I-131 ratio              → primary physics indicator of fresh fuel damage
  4. Rate-of-change features        → captures the rapid I-134 rise at leak onset
  5. EWMA trend baseline            → sensitive deviation signal (faster than rolling mean)
"""

import numpy as np
import pandas as pd


# Columns that come directly from the sensor hardware
_RAW_SENSOR_COLS = ["I131", "I134", "Cs137", "DNS", "pH", "conductivity"]


class CoolantSignalPreprocessor:
    """
    Feature engineering pipeline for VVER-1200 primary coolant monitoring.

    Call transform(df) on the raw generator output to produce a richer
    DataFrame ready for the anomaly detector.
    """

    def __init__(
        self,
        rolling_window_minutes: int = 30,
        derivative_windows_minutes: tuple[int, int] = (15, 30),
        ewma_halflife_minutes: float = 20.0,
        sigma_clip: float = 3.0,
    ) -> None:
        """
        Args:
            rolling_window_minutes: window for rolling mean/std (used in
                normalization and outlier detection).
            derivative_windows_minutes: windows for rate-of-change features.
            ewma_halflife_minutes: EWMA half-life in minutes.
            sigma_clip: outlier threshold in standard deviations.
        """
        self.rolling_window = rolling_window_minutes
        self.deriv_windows = derivative_windows_minutes
        self.ewma_halflife = ewma_halflife_minutes
        self.sigma_clip = sigma_clip

    # ── Step-level helpers ────────────────────────────────────────────────────

    def _clip_outliers(self, s: pd.Series) -> pd.Series:
        """
        Clip values beyond ±sigma_clip standard deviations of the rolling mean.

        Individual-sample spikes from electronic noise or gamma-ray pile-up in
        the NaI(Tl) detector look identical to a real activity excursion if you
        look at a single point — clipping removes them before feature engineering.
        Uses a backward-looking rolling window so no future information leaks in.
        """
        rm = s.rolling(self.rolling_window, min_periods=5).mean()
        rs = s.rolling(self.rolling_window, min_periods=5).std()

        upper = (rm + self.sigma_clip * rs).bfill().fillna(s.max())
        lower = (rm - self.sigma_clip * rs).bfill().fillna(0.0)

        return s.clip(lower=lower, upper=upper)

    def _rolling_zscore(self, s: pd.Series) -> pd.Series:
        """
        Normalize each point relative to its local rolling mean and std.

        z(t) = (x(t) − μ_rolling(t)) / σ_rolling(t)

        Removes slow baseline drift caused by fuel-cycle burn-up changes,
        power manoeuvres, and temperature variations — all of which shift the
        absolute activity without indicating cladding failure.
        The resulting signal is approximately stationary, which is required
        for Isolation Forest to work well.
        """
        rm = s.rolling(self.rolling_window, min_periods=5).mean()
        rs = s.rolling(self.rolling_window, min_periods=5).std()

        rs = rs.replace(0.0, np.nan).fillna(1.0)
        rm = rm.fillna(s.iloc[0] if len(s) > 0 else 0.0)

        return (s - rm) / rs

    def _derivative(self, s: pd.Series, window: int) -> pd.Series:
        """
        Finite-difference rate of change over `window` samples (1 sample = 1 min).

        dX/dt ≈ (X[t] − X[t−window]) / window  [units: activity / minute]

        For I-134, a normal fluctuation gives |dI134/dt| ≪ 1 % baseline / min.
        A gap-release event produces dI134/dt ≈ several percent baseline / min
        within the first 30 minutes — visible well before absolute limits breach.
        """
        return s.diff(periods=window) / window

    def _ewma(self, s: pd.Series) -> pd.Series:
        """
        Exponential Weighted Moving Average.

        EWMA weights recent observations more heavily than a symmetric rolling
        average. This makes it faster to track a rising trend, so the gap
        between the current reading and its EWMA grows earlier in a leak event.
        Half-life of 20 min means a step change reaches ~87 % of its new level
        within 60 minutes — well matched to the I-134 rise time constant.
        """
        return s.ewm(halflife=self.ewma_halflife, adjust=False).mean()

    def _i134_i131_ratio(self, i134: pd.Series, i131: pd.Series) -> pd.Series:
        """
        I-134 / I-131 activity ratio.

        This is the single most reliable early-warning indicator:
          · Normal coolant:         ratio ≈ 0.3–1.0
          · Fresh gap burst (t=0):  ratio >> 10  (I-134 not yet decayed)
          · 6 h post-failure:       ratio ≈ 1–3
          · 24 h post-failure:      ratio < 0.05

        The ratio can exceed the threshold of 10 some 30–90 minutes BEFORE the
        absolute I-134 limit (1×10⁵ Bq/cm³) is breached, giving critical
        early warning time for operator action.

        Source: IAEA-TECDOC-1328, §4.3 — "The I-134/I-131 ratio method"
        """
        safe_denom = i131.clip(lower=1.0)   # avoid division by zero
        return (i134 / safe_denom).clip(upper=500.0)

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full preprocessing pipeline.

        Input  : raw sensor DataFrame from SensorDataGenerator.generate()
        Output : same rows, many more columns (clipped, z-scored, derived features).
                 timestamp and label are preserved exactly.
        """
        out = df.sort_values("timestamp").reset_index(drop=True).copy()

        # 1. Clip sensor glitches on raw channels
        for col in _RAW_SENSOR_COLS:
            out[f"{col}_clipped"] = self._clip_outliers(out[col])

        # 2. I-134/I-131 ratio on clipped values
        out["ratio_I134_I131"] = self._i134_i131_ratio(
            out["I134_clipped"], out["I131_clipped"]
        )

        # 3. Rolling z-scores (stationarity for Isolation Forest)
        for col in _RAW_SENSOR_COLS:
            out[f"{col}_z"] = self._rolling_zscore(out[f"{col}_clipped"])

        # 4. Rate-of-change features for the most leak-sensitive channels
        for col in ["I131_clipped", "I134_clipped", "DNS_clipped"]:
            for w in self.deriv_windows:
                out[f"{col}_deriv_{w}min"] = self._derivative(out[col], w)

        # 5. EWMA trend baselines
        for col in ["I131_clipped", "I134_clipped", "ratio_I134_I131"]:
            out[f"{col}_ewma"] = self._ewma(out[col])

        # 6. Deviation above EWMA — negative deviation is uninteresting for leaks;
        #    positive deviation is the early warning signal
        out["I134_above_ewma"] = (
            out["I134_clipped"] - out["I134_clipped_ewma"]
        ).clip(lower=0.0)
        out["ratio_above_ewma"] = (
            out["ratio_I134_I131"] - out["ratio_I134_I131_ewma"]
        ).clip(lower=0.0)

        # Fill NaNs that arise from rolling/diff operations at the window edges
        numeric_cols = out.select_dtypes(include=[np.number]).columns
        out[numeric_cols] = out[numeric_cols].fillna(0.0)

        # Restore non-numeric columns (fillna above doesn't touch them, but be safe)
        out["timestamp"] = df["timestamp"].values
        out["label"] = df["label"].values

        return out

    def get_feature_matrix(
        self, processed_df: pd.DataFrame
    ) -> tuple[np.ndarray, list[str]]:
        """
        Extract the numeric feature matrix for ML training / inference.

        Excludes:
          - timestamp (non-numeric)
          - label (target variable — must not be fed to the model)
          - raw sensor columns (use clipped versions instead)

        Returns:
            (X, feature_names) where X has shape (n_samples, n_features).
        """
        exclude = {"timestamp", "label"} | set(_RAW_SENSOR_COLS)
        feature_cols = [
            c for c in processed_df.columns
            if c not in exclude
            and pd.api.types.is_float_dtype(processed_df[c])
        ]
        return processed_df[feature_cols].to_numpy(dtype=np.float64), feature_cols
