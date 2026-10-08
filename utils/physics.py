"""
Nuclear physics constants and formulas for VVER-1200 fuel rod monitoring.

Sources:
- IAEA-TECDOC-1328: Fuel Failure in Water Cooled Reactors
- IAEA Safety Guide NS-G-2.2: Operational Limits and Conditions
- NRC NUREG/CR-6365: Fission Product Release from Fuel
- Rosatom VVER-1200 Technical Specifications (АЭС-2006)
"""

import numpy as np
from dataclasses import dataclass


# ─── Radioactive Decay Constants (λ = ln(2) / T½) ────────────────────────────
# These govern how fast each isotope disappears from the coolant after release.
# Short-lived I-134 is the KEY early warning signal precisely because it must
# have been FRESHLY produced — it cannot survive long transport from fuel to coolant
# unless cladding failure is recent.

T_HALF_I131_S: float = 8.0207 * 24 * 3600     # 8.02 days in seconds
T_HALF_I134_S: float = 52.5 * 60              # 52.5 minutes in seconds
T_HALF_CS137_S: float = 30.17 * 365.25 * 24 * 3600  # 30.17 years in seconds

LAMBDA_I131: float = np.log(2) / T_HALF_I131_S   # s⁻¹ ≈ 1.00e-6
LAMBDA_I134: float = np.log(2) / T_HALF_I134_S   # s⁻¹ ≈ 2.20e-4  (rapid decay!)
LAMBDA_CS137: float = np.log(2) / T_HALF_CS137_S  # s⁻¹ ≈ 7.27e-10


# ─── VVER-1200 Normal Coolant Activity Baselines ─────────────────────────────
# These represent secular equilibrium concentrations in the primary coolant
# of an operating VVER-1200 at full power (1200 MWe).
# Tramp uranium on fuel assembly surfaces and small defects in fuel pellets
# produce a continuous low-level background even in "intact" cladding operation.
# Source: IAEA-TECDOC-1195, Table IV

BASELINE_I131: float = 3.7e3    # Bq/cm³, ~normal steady-state
BASELINE_I134: float = 1.5e4    # Bq/cm³, higher yield but partially decayed in transit
BASELINE_CS137: float = 5.0e2   # Bq/cm³, very slow accumulation over fuel cycle
BASELINE_DNS: float = 1.0       # normalized units (detector counts relative to reference)
BASELINE_PH: float = 7.2        # dimensionless, pH of VVER coolant with KOH+H3BO3 buffer
BASELINE_CONDUCTIVITY: float = 15.0  # μS/cm, proportional to boric acid concentration


# ─── Operational Action Limits ────────────────────────────────────────────────
# Exceeding these requires operator notification and evaluation.
# Source: IAEA Safety Guide NS-G-2.2, Table 3; typical VVER-1200 Tech Specs

LIMIT_I131: float = 3.7e4       # Bq/cm³, Level 1 action limit (10× baseline)
LIMIT_I134: float = 1.0e5       # Bq/cm³, Level 1 action limit
LIMIT_CS137: float = 5.0e3      # Bq/cm³, action limit

# The I-134/I-131 ratio > 10 is the primary EARLY WARNING indicator.
# Physics basis: I-134 is produced in fission at ~2.8× the rate of I-131,
# but its 52.5-min half-life means it decays rapidly. In the coolant at
# equilibrium, the ratio settles to ~0.5-1.0. A sudden ratio spike to >10
# means a FRESH burst of fission products just entered the coolant — the
# I-134 hasn't had time to decay yet. This can precede absolute limit
# exceedances by 30-90 minutes.
# Source: IAEA-TECDOC-1328, Section 4.3
LIMIT_I134_I131_RATIO: float = 10.0

LIMIT_DNS: float = 2.5          # normalized units; action level


# ─── Coolant Chemistry Bounds ────────────────────────────────────────────────
# VVER coolant uses boric acid (H₃BO₃) for reactivity control and KOH for pH.
# pH must stay above 6.8 to prevent Zr cladding dissolution;
# below 7.5 to prevent magnetite deposition on fuel assemblies.

PH_MIN: float = 6.8
PH_MAX: float = 7.5
CONDUCTIVITY_MIN: float = 10.0  # μS/cm
CONDUCTIVITY_MAX: float = 25.0  # μS/cm


# ─── Detector and Sampling Parameters ────────────────────────────────────────
# Parameters for the gamma spectrometry system monitoring the primary coolant.

DETECTOR_EFFICIENCY_I131: float = 0.35  # NaI(Tl) efficiency at 364.5 keV
DETECTOR_EFFICIENCY_I134: float = 0.28  # NaI(Tl) efficiency at 847 keV
SAMPLING_VOLUME_CM3: float = 10.0       # continuous bypass sample flow in cm³
COUNTING_TIME_S: float = 60.0           # integration time per measurement point


# ─── Isotope Data Container ───────────────────────────────────────────────────

@dataclass(frozen=True)
class IsotopeProperties:
    """Physical properties of a fission product isotope."""
    name: str
    symbol: str
    half_life_s: float         # seconds
    decay_constant: float      # λ in s⁻¹
    gamma_energy_keV: float    # primary gamma line
    fission_yield: float       # thermal-neutron fission yield from U-235
    baseline_Bq_cm3: float     # typical normal coolant concentration


ISOTOPES: dict[str, IsotopeProperties] = {
    "I131": IsotopeProperties(
        name="Iodine-131",
        symbol="I-131",
        half_life_s=T_HALF_I131_S,
        decay_constant=LAMBDA_I131,
        gamma_energy_keV=364.5,
        fission_yield=0.0277,   # 2.77% per U-235 fission
        baseline_Bq_cm3=BASELINE_I131,
    ),
    "I134": IsotopeProperties(
        name="Iodine-134",
        symbol="I-134",
        half_life_s=T_HALF_I134_S,
        decay_constant=LAMBDA_I134,
        gamma_energy_keV=847.0,
        fission_yield=0.0780,   # 7.80% — highest yield of all iodine isotopes
        baseline_Bq_cm3=BASELINE_I134,
    ),
    "Cs137": IsotopeProperties(
        name="Cesium-137",
        symbol="Cs-137",
        half_life_s=T_HALF_CS137_S,
        decay_constant=LAMBDA_CS137,
        gamma_energy_keV=661.7,
        fission_yield=0.0619,   # 6.19% per fission
        baseline_Bq_cm3=BASELINE_CS137,
    ),
}


# ─── Physics Utility Functions ────────────────────────────────────────────────

def bateman_decay(A0: float, decay_constant: float, t_seconds: float) -> float:
    """
    Radioactive decay law: A(t) = A₀ · exp(−λt).

    Args:
        A0: initial activity in Bq
        decay_constant: λ in s⁻¹
        t_seconds: elapsed time in seconds

    Returns:
        Activity at time t in Bq.
    """
    return A0 * np.exp(-decay_constant * t_seconds)


def i134_i131_ratio_age(t_hours: float) -> float:
    """
    Expected I-134/I-131 ratio as a function of time since cladding damage.

    At the moment of damage, both isotopes escape at a rate proportional to
    their fission yields (ratio ≈ yield_I134/yield_I131 ≈ 2.82). I-134 then
    decays with its 52.5-min half-life while I-131 remains essentially constant
    over the first few hours (T½ = 8 days). The ratio collapses over 6-12 hours.

    Args:
        t_hours: time since damage began, in hours

    Returns:
        Estimated I-134/I-131 ratio at that point in time.
    """
    t_sec = t_hours * 3600.0
    initial_ratio = ISOTOPES["I134"].fission_yield / ISOTOPES["I131"].fission_yield
    # I-134 decays; I-131 barely changes over hours → ratio tracks I-134 decay
    i134_fraction = np.exp(-LAMBDA_I134 * t_sec)
    return initial_ratio * i134_fraction


def coolant_dilution_factor(coolant_volume_m3: float = 200.0) -> float:
    """
    Dilution factor for fission products released into the primary circuit.

    VVER-1200 primary coolant volume ≈ 200 m³.
    Converts a total released activity (Bq) into concentration (Bq/cm³).

    Args:
        coolant_volume_m3: total primary coolant inventory in m³

    Returns:
        Dilution factor in (Bq/cm³) per Bq released.
    """
    volume_cm3 = coolant_volume_m3 * 1.0e6   # m³ → cm³
    return 1.0 / volume_cm3
