"""
VVER-1200 Fuel Rod Cladding Failure (ТВЭЛ) — Early Detection System
====================================================================

End-to-end pipeline: synthetic data generation → preprocessing →
anomaly detection → dashboard visualisation → text report.

Usage
-----
  python main.py                                  # defaults (72 h, random leak)
  python main.py --hours 72 --sensitivity 0.7
  python main.py --hours 48 --leak-hour 30 --sensitivity 0.6 --no-show
  python main.py --hours 72 --seed 7 --output results/run7.png
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VVER-1200 Fuel Rod Cladding Failure Early Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --hours 72 --sensitivity 0.7
  python main.py --hours 48 --leak-hour 30 --output plot.png --no-show
  python main.py --hours 72 --seed 7 --train-hours 18 --sensitivity 0.65
""",
    )
    p.add_argument(
        "--hours", type=int, default=72,
        help="Total simulation duration in hours (default: 72)",
    )
    p.add_argument(
        "--leak-hour", type=float, default=None,
        help="Hour at which cladding failure starts; random after hour 24 if omitted",
    )
    p.add_argument(
        "--sensitivity", type=float, default=0.7,
        help="Alert threshold [0–1]: higher = fewer false alarms, lower = earlier "
             "detection; used directly as the composite-score cut-off (default: 0.7)",
    )
    p.add_argument(
        "--train-hours", type=int, default=20,
        help="Hours of normal data used to train Isolation Forest (default: 20)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for data generation (default: 42)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="File path for the output PNG dashboard; auto-named if omitted",
    )
    p.add_argument(
        "--no-show", action="store_true",
        help="Skip the interactive plot window (for headless / CI runs)",
    )
    return p.parse_args()


def _separator(char: str = "─", width: int = 62) -> str:
    return char * width


def print_report(
    leak_hour: float,
    train_hours: int,
    alert_threshold: float,
    metrics: dict,
) -> None:
    """Print a formatted detection performance summary to stdout."""
    sep = _separator()
    print(f"\n{sep}")
    print("   VVER-1200 Fuel Cladding Failure Detection  ·  Run Report")
    print(sep)
    print(f"   Injected leak at hour :  {leak_hour:.2f}")
    print(f"   Training window       :  0 h – {train_hours} h  (normal operation only)")
    print(f"   Alert threshold       :  {alert_threshold:.2f}  (sensitivity flag)")
    print(sep)

    fa = metrics["first_alert_idx"]
    lt = metrics["lead_time_minutes"]

    if fa is None:
        print("   RESULT  ▶  MISSED DETECTION — no sustained alert generated")
    elif lt is not None and lt > 0:
        print(f"   RESULT  ▶  EARLY WARNING — {lt} minute(s) before leak onset")
        print(f"   First alert at sample  :  {fa}  (minute {fa})")
    elif lt == 0:
        print("   RESULT  ▶  Alert triggered at leak onset (zero lead time)")
    else:
        late = abs(lt) if lt is not None else "?"
        print(f"   RESULT  ▶  Late detection — {late} minute(s) after leak onset")

    print()
    print(f"   False positive rate    :  {metrics['false_positive_rate']:.4f}")
    print(f"   Precision              :  {metrics['precision']:.4f}")
    print(f"   Recall                 :  {metrics['recall']:.4f}")
    print(f"   False positive count   :  {metrics['false_positives']}")
    print(f"{sep}\n")


def main() -> None:
    args = parse_args()

    # ── 1 · Synthetic sensor data ─────────────────────────────────────────────
    # Late imports so that --help remains instant and matplotlib backend can
    # be set before any pyplot import (needed for --no-show / headless mode).
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")

    from data.generator import SensorDataGenerator
    from models.preprocessor import CoolantSignalPreprocessor
    from models.detector import LeakDetector
    from visualization.dashboard import build_dashboard
    import matplotlib.pyplot as plt

    print("\n[1/5] Generating synthetic VVER-1200 sensor data …")
    gen = SensorDataGenerator(
        duration_hours=args.hours,
        leak_hour=args.leak_hour,
        random_seed=args.seed,
    )
    raw_df = gen.generate()
    leak_hour_actual = gen.leak_hour
    leak_sample = gen.leak_sample
    print(f"      Leak start : hour {leak_hour_actual:.2f}  (sample index {leak_sample})")
    print(f"      Data shape : {raw_df.shape}  ({raw_df.shape[0]} minutes, "
          f"{raw_df.shape[1]} channels)")

    # ── 2 · Preprocessing & feature engineering ───────────────────────────────
    print("\n[2/5] Preprocessing signals (normalization, feature engineering) …")
    preprocessor = CoolantSignalPreprocessor(rolling_window_minutes=30)
    processed_df = preprocessor.transform(raw_df)
    X_all, feature_names = preprocessor.get_feature_matrix(processed_df)
    print(f"      Feature matrix : {X_all.shape}  ({len(feature_names)} features)")

    # ── 3 · Train Isolation Forest ────────────────────────────────────────────
    # Training window must be entirely within the normal-operation period.
    # We cap it at leak_sample to guarantee no contamination from anomalous data.
    train_n = min(args.train_hours * 60, leak_sample)
    if train_n < 60:
        print(
            f"\n[WARNING] Only {train_n} training samples available "
            f"(leak starts at sample {leak_sample}). "
            "Detection quality may be degraded.",
            file=sys.stderr,
        )

    print(f"\n[3/5] Training Isolation Forest on {train_n} normal samples "
          f"({train_n // 60} h {train_n % 60} min) …")

    detector = LeakDetector(
        contamination=0.05,
        physics_weight=0.6,
        alert_threshold=args.sensitivity,   # sensitivity is used directly as threshold
    )
    detector.fit(X_all[:train_n])
    print("      Training complete.")

    # ── 4 · Anomaly detection on full time series ─────────────────────────────
    print("\n[4/5] Running detection on full time series …")
    results = detector.predict(X_all, processed_df)
    metrics = detector.compute_performance_metrics(
        results["alert"],
        raw_df["label"].values,
        leak_sample,
    )
    print_report(leak_hour_actual, args.train_hours, detector.alert_threshold, metrics)

    # ── 5 · Dashboard ─────────────────────────────────────────────────────────
    print("[5/5] Building monitoring dashboard …")

    output_path = args.output
    if output_path is None:
        output_path = (
            f"leak_detection_h{args.hours}"
            f"_leak{leak_hour_actual:.0f}"
            f"_sens{args.sensitivity}.png"
        )

    build_dashboard(
        raw_df=raw_df,
        processed_df=processed_df,
        detection_results=results,
        leak_sample_idx=leak_sample,
        first_alert_idx=metrics["first_alert_idx"],
        lead_time_minutes=metrics["lead_time_minutes"],
        alert_threshold=detector.alert_threshold,
        save_path=output_path,
    )

    if not args.no_show:
        plt.show()

    print("Done.\n")


if __name__ == "__main__":
    main()
