# VVER-1200 Fuel Rod Cladding Failure — Early Detection System

A simulation and anomaly-detection pipeline that demonstrates how primary-coolant
radiochemistry data can be used to detect a VVER-1200 fuel rod cladding failure
(a "твэл" leak) **before** absolute regulatory limits are exceeded.

The system generates physics-accurate **synthetic** sensor data for a VVER-1200
primary coolant loop, injects a realistic cladding-failure event, and runs a
hybrid (physics rules + machine learning) detector against it. Results are
visualised either as a static PNG dashboard (`main.py`) or an interactive
Streamlit application with a 3D reactor schematic (`app.py`).

> ⚠️ **All sensor data is synthetically generated.** This project does not
> connect to any real reactor, plant, or SCADA/monitoring system. It is a
> research / educational simulation of detection methodology, built on public
> reference values from IAEA and NRC documents (see [References](#references)).

---

## Table of Contents

- [Why this matters](#why-this-matters)
- [How detection works](#how-detection-works)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI pipeline (`main.py`)](#cli-pipeline-mainpy)
  - [Interactive dashboard (`app.py`)](#interactive-dashboard-apppy)
- [Sensor channels](#sensor-channels)
- [Pipeline stages](#pipeline-stages)
- [Output metrics](#output-metrics)
- [Configuration reference](#configuration-reference)
- [References](#references)

---

## Why this matters

Fuel rod (ТВЭЛ) cladding is the primary barrier between fission products and
the primary coolant. A through-wall crack releases fission gases and iodine
isotopes into the coolant. Detecting this **early** — minutes to hours before
absolute activity limits are breached — gives operators time to reduce power,
sample diagnostically, and plan a response before the situation escalates.

The key physical insight exploited here is the **I‑134 / I‑131 activity
ratio**:

- Both isotopes are released simultaneously and in proportion to their fission
  yields (ratio ≈ 2.8 at the moment of release).
- I‑134 has a short half-life (52.5 min) and decays away quickly.
- I‑131 has a long half-life (8 days) and stays essentially flat over the
  relevant time window.

So a fresh cladding breach produces a brief **spike** in the I‑134/I‑131
ratio that collapses again within hours — a signature that can precede
absolute-limit exceedance by **30–90 minutes**.

## How detection works

```
Raw sensor data (I-131, I-134, Cs-137, pH, conductivity, DNS)
        │
        ▼
Preprocessing (outlier clipping, rolling z-score, derivatives, EWMA)
        │
        ├──────────────► Physics Rule Engine ─────► physics_score  [0,1]
        │                 (ratio, dI134/dt, DNS)
        │
        └──────────────► Isolation Forest ─────────► ml_score      [0,1]
                          (unsupervised, trained on normal data)
        │
        ▼
alert_score = α · physics_score + (1 − α) · ml_score
        │
        ▼
alert = 1  if alert_score ≥ threshold
```

The **physics rule engine** ([`models/detector.py`](models/detector.py))
combines three domain-expert rules:

| Rule | Weight | Signal | Rationale |
|---|---|---|---|
| I‑134/I‑131 ratio | 50% | sigmoid centred on ratio = 10 | most reliable, 30–90 min early warning |
| I‑134 rate of change (30 min) | 35% | normalized `dI134/dt` | earliest signal, catches the initial slope |
| DNS (delayed neutron signal) | 15% | normalized to action level 2.5 | most specific, fewest false positives |

The **ML model** is an `IsolationForest` (scikit-learn) trained exclusively on
a window of normal-operation data, so it never sees leak samples during
training. Its anomaly score is rescaled into `[0, 1]` and blended with the
physics score using a configurable weight `α` (`physics_weight`, default 0.6).

## Project structure

```
reactor_leak_detector/
├── main.py                    CLI entry point — end-to-end pipeline + PNG dashboard
├── app.py                     Streamlit interactive dashboard (run with `streamlit run app.py`)
├── requirements.txt           Python dependencies
├── data/
│   └── generator.py           SensorDataGenerator — synthetic sensor data + leak injection
├── models/
│   ├── preprocessor.py         CoolantSignalPreprocessor — feature engineering
│   └── detector.py             LeakDetector — Isolation Forest + physics rule engine
├── utils/
│   └── physics.py               Nuclear physics constants, limits, decay formulas
└── visualization/
    ├── dashboard.py              Static matplotlib 4-panel dashboard (used by main.py)
    └── reactor_3d.py             3D Plotly reactor schematic (used by app.py)
```

## Installation

Requires Python 3.10+ (uses `tuple[int, int]` / `dict[str, ...]` type hints).

```bash
cd reactor_leak_detector
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `scipy`,
`streamlit`, `plotly`.

## Usage

### CLI pipeline (`main.py`)

Runs the full pipeline once and produces a static PNG dashboard plus a
text report in the terminal.

```bash
python main.py                                          # defaults: 72 h, random leak time
python main.py --hours 72 --sensitivity 0.7
python main.py --hours 48 --leak-hour 30 --sensitivity 0.6 --no-show
python main.py --hours 72 --seed 7 --output results/run7.png
```

| Flag | Default | Description |
|---|---|---|
| `--hours` | `72` | Total simulated duration in hours |
| `--leak-hour` | random after hour 24 | Hour at which cladding failure begins |
| `--sensitivity` | `0.7` | Alert threshold `[0–1]`; higher = fewer false alarms, lower = earlier detection |
| `--train-hours` | `20` | Hours of normal data used to train the Isolation Forest |
| `--seed` | `42` | Random seed for data generation |
| `--output` | auto-named | Output PNG file path |
| `--no-show` | off | Skip the interactive plot window (for headless/CI runs) |

The terminal report includes lead time (or missed/late detection), false
positive rate, precision, and recall against the synthetic ground truth.

### Interactive dashboard (`app.py`)

A full Streamlit application with sidebar controls (duration, leak timing,
alert threshold, training window, physics/ML blend weight) and six tabs:

- **3D Reactor** — interactive 3D schematic of the VVER-1200 primary circuit,
  color-coded per component by live alert score, with a time-scrubber.
- **Radiological** — I‑131/I‑134 concentrations, the I‑134/I‑131 ratio, and
  Cs‑137 activity, each with an expandable "Physical basis" explanation.
- **Detection** — composite alert score with confidence band and alert zones.
- **Chemistry** — pH and conductivity trends.
- **Data Explorer** — raw/processed data tables.
- **Reference** — physics constants and regulatory limits used by the model.

Run it with:

```bash
streamlit run app.py
```

## Sensor channels

| Channel | Unit | Description |
|---|---|---|
| `I131` | Bq/cm³ | Iodine‑131 activity — long half-life (8 d), lagging indicator |
| `I134` | Bq/cm³ | Iodine‑134 activity — short half-life (52.5 min), leading indicator |
| `Cs137` | Bq/cm³ | Caesium‑137 activity — very long half-life (30 yr), confirmatory/cumulative signal |
| `pH` | – | Coolant pH; drops slightly during a leak (iodine hydrolysis) |
| `conductivity` | µS/cm | Electrical conductivity; rises during a leak (dissolved ionic species) |
| `DNS` | normalized | Delayed Neutron Signal — most specific indicator, requires direct fuel exposure |

Baselines and action limits for each channel are defined in
[`utils/physics.py`](utils/physics.py).

## Pipeline stages

1. **Synthetic data generation** ([`data/generator.py`](data/generator.py))
   Produces per-minute sensor data with realistic normal-operation drift and
   noise, then injects a three-phase leak event: pre-leak micro-crack drift,
   gap-inventory burst, and fuel-matrix diffusion.
2. **Preprocessing** ([`models/preprocessor.py`](models/preprocessor.py))
   Outlier clipping (3σ), rolling z-score normalization, I‑134/I‑131 ratio,
   rate-of-change features, and EWMA trend baselines.
3. **Training** — the `IsolationForest` is trained only on the normal-operation
   window (capped so it never includes post-leak samples).
4. **Detection** ([`models/detector.py`](models/detector.py))
   Produces per-sample physics score, ML score, composite alert score, and
   binary alert flags, plus lead-time/precision/recall metrics against ground
   truth.
5. **Visualization** ([`visualization/dashboard.py`](visualization/dashboard.py),
   [`visualization/reactor_3d.py`](visualization/reactor_3d.py))
   Static PNG dashboard or interactive Streamlit dashboard with a 3D reactor
   schematic.

## Output metrics

| Metric | Meaning |
|---|---|
| **Lead time** | Minutes the first sustained alert precedes (positive) or follows (negative) true leak onset |
| **False positive rate** | Fraction of pre-leak samples incorrectly flagged |
| **Precision** | `TP / (TP + FP)` on the full time series |
| **Recall** | `TP / (TP + FN)` on the full time series |

A "sustained" alert requires 3 consecutive alarmed samples
(`LeakDetector.find_first_alert`), filtering out single-sample noise spikes.

## Configuration reference

Key tunable parameters (CLI flags in `main.py`, sidebar sliders in `app.py`):

- **`sensitivity` / `alert_threshold`** — composite score cut-off for raising
  an alert. Lower = earlier but noisier detection.
- **`train_hours`** — length of the normal-operation window used to fit the
  Isolation Forest.
- **`physics_weight` (α)** — blend weight between the physics rule engine and
  the ML model (`alert_score = α·physics + (1−α)·ml`).
- **`contamination`** — expected anomaly fraction passed to `IsolationForest`
  (affects its internal decision boundary).

## References

- IAEA-TECDOC-1328 — *Fuel Failure in Water Cooled Reactors*
- IAEA Safety Guide NS-G-2.2 — *Operational Limits and Conditions*
- NRC NUREG/CR-6365 — *Fission Product Release from Fuel*
- Rosatom VVER-1200 Technical Specifications (АЭС-2006)
