# vitality_score.py
"""
Vitality Score (testosterone + lifestyle), age-fair with diminishing returns.

Run doctests:
    python -m doctest -v vitality_score.py

Or from Python:
    import importlib.util, doctest
    spec = importlib.util.spec_from_file_location("vitality_score", "vitality_score.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    doctest.testmod(m, verbose=True)
"""
from __future__ import annotations
import math
from typing import Optional, Literal, Dict, Callable

# -----------------------------
# Helpers
# -----------------------------

def clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def safediv(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default

def safe_sigmoid(x: float) -> float:
    """Numerically stable logistic.

    >>> round(safe_sigmoid(0.0), 6)
    0.5
    >>> round(safe_sigmoid(20), 6)
    1.0
    >>> round(safe_sigmoid(-20), 6)
    0.0
    """
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    else:
        e = math.exp(x)
        return e / (1.0 + e)

NGDL_PER_NMOL = 28.84  # ng/dL = nmol/L * 28.84

def to_ngdl(x: float, unit: Literal["ng/dL","nmol/L"] = "ng/dL") -> float:
    """Convert to ng/dL if unit is nmol/L, else passthrough.

    >>> round(to_ngdl(20, "nmol/L"))  # ~577 ng/dL
    577
    >>> to_ngdl(400, "ng/dL")
    400
    """
    if unit == "nmol/L":
        return x * NGDL_PER_NMOL
    return x

# -----------------------------
# 1) Hormone component (age-fair, stable slope, diminishing returns)
# -----------------------------

def hormone_component(
    total_T: float,
    age: int,
    T50_fn: Callable[[int], float],
    T95_fn: Callable[[int], float],
    unit: Literal["ng/dL", "nmol/L"] = "ng/dL",
    morning_sample: bool = True,
) -> float:
    """
    Map total testosterone to an age-fair 0..1 score with diminishing returns.

    - Uses quantile-derived SD: SD ≈ (T95 - T50) / 1.645
    - Uses logistic matched to normal CDF slope via scale k≈1.702
    - Small bonus above T95 (2%) to avoid glamorizing extremes
    - If sample not morning/fasted, a mild discount is applied.

    >>> # Toy curves: see below for definitions; quick sanity with 45y/o
    >>> T50 = toy_T50(45); T95 = toy_T95(45)
    >>> # Median should map near 0.5
    >>> round(hormone_component(T50, 45, toy_T50, toy_T95), 2)
    0.49
    >>> # Big jump from low->normal, small from normal->very high
    >>> low = hormone_component(T50 - 250, 45, toy_T50, toy_T95)
    >>> mid = hormone_component(T50 + 150, 45, toy_T50, toy_T95)
    >>> high = hormone_component(T95 + 200, 45, toy_T50, toy_T95)
    >>> (mid - low) > (high - mid)
    True
    >>> # Age fairness: 25 and 65 at their medians -> ~same H
    >>> round(hormone_component(toy_T50(25), 25, toy_T50, toy_T95), 2) == round(hormone_component(toy_T50(65), 65, toy_T50, toy_T95), 2)
    True
    """
    # Ensure consistent units (assumes T50/T95 provided in same unit argument)
    total_T = to_ngdl(total_T, unit) if unit != "ng/dL" else total_T
    T50 = to_ngdl(T50_fn(age), unit) if unit != "ng/dL" else T50_fn(age)
    T95 = to_ngdl(T95_fn(age), unit) if unit != "ng/dL" else T95_fn(age)

    # Guard rails
    def _clipT(x: float) -> float:
        return max(1e-6, min(x, 5000.0))  # wide physiological caps (ng/dL scale)
    total_T = _clipT(total_T); T50 = _clipT(T50); T95 = _clipT(T95)

    # Quantile-derived SD (approx normal near center)
    sd = max((T95 - T50) / 1.645, 1e-6)
    k = 1.702  # logistic scale factor to match normal slope at 0
    z = (total_T - T50) / sd
    H_base = safe_sigmoid(z / k)
    H_tail = clip((total_T - T95) / max(T95, 1e-9), 0.0, 1.0)
    H = 0.98 * H_base + 0.02 * H_tail

    # Apply small discount if not morning sample (uncertainty)
    if not morning_sample:
        H *= 0.97

    return clip(H, 0.0, 1.0)

# -----------------------------
# 2) Body composition factor (tiered signals with tunables)
# -----------------------------

def body_comp_factor(
    sex: Literal["male", "female"],
    fat_mass_kg: Optional[float] = None,
    lean_mass_kg: Optional[float] = None,
    bodyfat_pct: Optional[float] = None,     # 0..100
    waist_cm: Optional[float] = None,
    height_cm: Optional[float] = None,
    BMI: Optional[float] = None,
    # Tunables
    center_fatratio_male: float = 0.15,
    center_fatratio_female: float = 0.25,
    spread_fatratio: float = 0.10,
    center_wht: float = 0.45,
    spread_wht: float = 0.10,
    center_bmi: float = 23.0,
    spread_bmi: float = 7.0,
) -> float:
    """
    Return 0..1 body comp factor using best available metric (A->D).

    >>> round(body_comp_factor("male", BMI=23.0), 3)
    1.0
    >>> round(body_comp_factor("male", BMI=30.0), 3)  # away from center -> lower
    0.0
    """
    center_fatratio = center_fatratio_male if sex == "male" else center_fatratio_female

    # (A) fat + lean available
    if fat_mass_kg is not None and lean_mass_kg is not None and (fat_mass_kg + lean_mass_kg) > 0:
        fat_ratio = safediv(fat_mass_kg, fat_mass_kg + lean_mass_kg)
        return 1.0 - clip(abs(fat_ratio - center_fatratio) / spread_fatratio, 0.0, 1.0)

    # (B) body-fat % available
    if bodyfat_pct is not None:
        fat_ratio = clip(bodyfat_pct / 100.0, 0.0, 1.0)
        return 1.0 - clip(abs(fat_ratio - center_fatratio) / spread_fatratio, 0.0, 1.0)

    # (C) waist:height ratio
    if waist_cm is not None and height_cm is not None and height_cm > 0:
        wht = waist_cm / height_cm
        return 1.0 - clip(abs(wht - center_wht) / spread_wht, 0.0, 1.0)

    # (D) BMI fallback
    if BMI is not None:
        return 1.0 - clip(abs(BMI - center_bmi) / spread_bmi, 0.0, 1.0)

    # No signal → neutral-ish
    return 0.5

# -----------------------------
# 3) Training factor (self-report or device)
# -----------------------------

def training_factor(
    training_days_per_week: Optional[float] = None,
    training_minutes_per_week: Optional[int] = None,
    device_strain_0_21: Optional[float] = None
) -> float:
    """
    Saturating transforms; prefer objective signal if provided.

    >>> round(training_factor(training_days_per_week=6), 3)
    1.0
    >>> round(training_factor(training_days_per_week=3), 3)  # sqrt saturation
    0.707
    """
    if device_strain_0_21 is not None:
        return math.sqrt(clip(device_strain_0_21, 0.0, 21.0) / 16.0)

    if training_minutes_per_week is not None:
        return math.sqrt(clip(float(training_minutes_per_week), 0.0, 300.0) / 300.0)

    if training_days_per_week is not None:
        return math.sqrt(clip(training_days_per_week, 0.0, 6.0) / 6.0)

    return 0.5

# -----------------------------
# 4) Sleep factor (nonlinear penalty when <6h, plateau near 8.5h)
# -----------------------------

def sleep_factor(avg_sleep_hours: Optional[float]) -> float:
    """
    >>> round(sleep_factor(9.0), 3)   # near plateau
    1.0
    >>> round(sleep_factor(8.0), 3)
    0.941
    >>> round(sleep_factor(5.0), 3)   # penalty below 6h
    0.188
    """
    if avg_sleep_hours is None:
        return 0.5
    base = clip(avg_sleep_hours / 8.5, 0.0, 1.0)
    penalty = 0.0
    if avg_sleep_hours < 6.0:
        penalty = clip((6.0 - avg_sleep_hours) / 2.0, 0.0, 0.4)
    return clip(base - penalty, 0.0, 1.0)

# -----------------------------
# 5) Physiological anchors (RHR/HRV) -> consistency multiplier
# -----------------------------

def rhr_factor(rhr_bpm: Optional[float], sex: Literal["male","female"]) -> Optional[float]:
    if rhr_bpm is None:
        return None
    center = 55.0 if sex == "male" else 60.0
    spread = 15.0
    return 1.0 - clip(abs(rhr_bpm - center) / spread, 0.0, 1.0)

def hrv_factor(
    hrv_ms: Optional[float],
    age: int,
    sex: Literal["male","female"],
    HRV50_fn: Optional[Callable[[int, Literal["male","female"]], float]] = None,
    HRVspread: float = 12.0
) -> Optional[float]:
    if hrv_ms is None or HRV50_fn is None:
        return None
    ref = max(HRV50_fn(age, sex), 1e-9)
    z = (hrv_ms - ref) / HRVspread
    return safe_sigmoid(z)

def combined_recovery_anchor(
    sex: Literal["male","female"],
    age: int,
    rhr_bpm: Optional[float] = None,
    hrv_ms: Optional[float] = None,
    HRV50_fn: Optional[Callable[[int, Literal["male","female"]], float]] = None
) -> Optional[float]:
    Rf = rhr_factor(rhr_bpm, sex)
    Hf = hrv_factor(hrv_ms, age, sex, HRV50_fn=HRV50_fn) if HRV50_fn else None

    if Rf is not None and Hf is not None:
        return 0.5 * Rf + 0.5 * Hf
    if Rf is not None:
        return Rf
    if Hf is not None:
        return Hf
    return None

# -----------------------------
# 6) Coverage-aware lifestyle + anchor multiplier
# -----------------------------

def _weighted_mean_with_prior(values, weights, prior=0.5, prior_weight=0.5):
    num = sum(v*w for v, w in zip(values, weights))
    den = sum(weights)
    num += prior * prior_weight
    den += prior_weight
    return num / max(den, 1e-9)

def lifestyle_component(
    sex: Literal["male","female"],
    training_days_per_week: Optional[float] = None,
    training_minutes_per_week: Optional[int] = None,
    device_strain_0_21: Optional[float] = None,
    avg_sleep_hours: Optional[float] = None,
    fat_mass_kg: Optional[float] = None,
    lean_mass_kg: Optional[float] = None,
    bodyfat_pct: Optional[float] = None,
    waist_cm: Optional[float] = None,
    height_cm: Optional[float] = None,
    BMI: Optional[float] = None,
    age: Optional[int] = None,
    rhr_bpm: Optional[float] = None,
    hrv_ms: Optional[float] = None,
    HRV50_fn: Optional[Callable[[int, Literal["male","female"]], float]] = None
) -> float:
    """
    Coverage-aware mean with neutral prior; anchors act as a mild multiplier.
    """
    Tf = training_factor(training_days_per_week, training_minutes_per_week, device_strain_0_21)
    Sf = sleep_factor(avg_sleep_hours)
    Bf = body_comp_factor(sex, fat_mass_kg, lean_mass_kg, bodyfat_pct, waist_cm, height_cm, BMI)

    # Determine which signals are truly present (None means absent)
    weights = [
        1.0 if (training_days_per_week is not None or training_minutes_per_week is not None or device_strain_0_21 is not None) else 0.0,
        1.0 if avg_sleep_hours is not None else 0.0,
        1.0 if any(v is not None for v in [fat_mass_kg, lean_mass_kg, bodyfat_pct, waist_cm, height_cm, BMI]) else 0.0,
    ]
    values = [Tf, Sf, Bf]
    L_base = _weighted_mean_with_prior(values, weights, prior=0.5, prior_weight=0.5)

    Rf = combined_recovery_anchor(sex, age or 40, rhr_bpm, hrv_ms, HRV50_fn=HRV50_fn)
    if Rf is not None:
        mult = 0.9 + 0.15 * Rf  # ~[0.9, 1.05]
        L = clip(L_base * mult, 0.0, 1.0)
    else:
        L = L_base

    return L

# -----------------------------
# 7) Confidence curve (matters with sparse/self-report data)
# -----------------------------

def apply_confidence(x: float, confidence_c: float = 1.0, k: float = 0.20, gamma: float = 1.5) -> float:
    """
    multiplier = 1 - k * (1-c)^gamma

    >>> round(apply_confidence(0.8, confidence_c=0.6), 3)
    0.76
    """
    c = clip(confidence_c, 0.0, 1.0)
    m = 1.0 - k * ((1.0 - c) ** gamma)
    return clip(x * m, 0.0, 1.0)

# -----------------------------
# 8) Final score with interaction (hybrid) + explainability outputs
# -----------------------------

def synergy_score(
    total_T: float,
    age: int,
    sex: Literal["male","female"],
    T50_fn: Callable[[int], float],
    T95_fn: Callable[[int], float],
    training_days_per_week: Optional[float] = None,
    training_minutes_per_week: Optional[int] = None,
    device_strain_0_21: Optional[float] = None,
    avg_sleep_hours: Optional[float] = None,
    fat_mass_kg: Optional[float] = None,
    lean_mass_kg: Optional[float] = None,
    bodyfat_pct: Optional[float] = None,
    waist_cm: Optional[float] = None,
    height_cm: Optional[float] = None,
    BMI: Optional[float] = None,
    rhr_bpm: Optional[float] = None,
    hrv_ms: Optional[float] = None,
    HRV50_fn: Optional[Callable[[int, Literal["male","female"]], float]] = None,
    confidence_c: float = 1.0,
    unit: Literal["ng/dL","nmol/L"] = "ng/dL",
    morning_sample: bool = True,
    a: float = 0.25,
    b: float = 0.25
) -> Dict[str, float]:
    """
    Returns dict with H, L_raw, L, interaction, Score.
    """
    H = hormone_component(total_T, age, T50_fn, T95_fn, unit=unit, morning_sample=morning_sample)
    L_raw = lifestyle_component(
        sex=sex,
        training_days_per_week=training_days_per_week,
        training_minutes_per_week=training_minutes_per_week,
        device_strain_0_21=device_strain_0_21,
        avg_sleep_hours=avg_sleep_hours,
        fat_mass_kg=fat_mass_kg,
        lean_mass_kg=lean_mass_kg,
        bodyfat_pct=bodyfat_pct,
        waist_cm=waist_cm,
        height_cm=height_cm,
        BMI=BMI,
        age=age,
        rhr_bpm=rhr_bpm,
        hrv_ms=hrv_ms,
        HRV50_fn=HRV50_fn
    )
    L = apply_confidence(L_raw, confidence_c=confidence_c)

    mult_w = max(1.0 - a - b, 0.0)
    S_star = a * H + b * L + mult_w * (H * L)
    Score = 100.0 * clip(S_star, 0.0, 1.0)

    return {
        "H": H,
        "L_raw": L_raw,
        "L": L,
        "interaction": H * L,
        "Score": Score
    }

def synergy_score_geomean(*args, **kwargs) -> Dict[str, float]:
    out = synergy_score(*args, **kwargs)
    H = out["H"]; L = out["L"]
    out["Score"] = 100.0 * clip(math.sqrt(max(H, 0.0) * max(L, 0.0)), 0.0, 1.0)
    return out

# -----------------------------
# 9) Toy reference curves (for demos/tests)
# -----------------------------

def toy_T50(age: int) -> float:
    """
    Age-median total T (ng/dL), synthetic but realistic-ish.
    Peak ~650 at 25; smooth decline with age.

    >>> round(toy_T50(25))
    650
    >>> 450 <= toy_T50(65) <= 500
    True
    """
    # 650 * exp(-0.008 * (age-25))
    return 650.0 * math.exp(-0.008 * max(age - 25, 0))

def toy_sd(age: int) -> float:
    """Synthetic SD that slowly tightens with age."""
    # ~110 at 25, gradually down to ~90 by 65
    return 110.0 * math.exp(-0.006 * max(age - 25, 0))

def toy_T95(age: int) -> float:
    """95th percentile assuming near-normal around center: T95 = T50 + 1.645*SD"""
    return toy_T50(age) + 1.645 * toy_sd(age)

def toy_HRV50(age: int, sex: Literal["male","female"]) -> float:
    """Toy RMSSD median (ms), higher in females and younger adults."""
    base_25 = 50.0 if sex == "male" else 58.0
    decline = 0.35 * max(age - 25, 0)  # ms drop from 25y baseline
    return max(base_25 - decline, 18.0)

# -----------------------------
# 10) Demonstration helpers
# -----------------------------

def demo_flattening_and_age_fairness():
    """
    Quick numeric demo for docs/README.

    >>> # Show diminishing returns: 45y male, strong lifestyle
    >>> age=45; sex="male"
    >>> base_kwargs = dict(age=age, sex=sex, T50_fn=toy_T50, T95_fn=toy_T95,
    ...                    training_days_per_week=6, avg_sleep_hours=9.0, BMI=23,
    ...                    confidence_c=1.0, morning_sample=True)
    >>> s300 = synergy_score(total_T=300, **base_kwargs)["Score"]
    >>> s600 = synergy_score(total_T=600, **base_kwargs)["Score"]
    >>> s900 = synergy_score(total_T=900, **base_kwargs)["Score"]
    >>> (s600 - s300) > (s900 - s600)
    True
    >>> # Age fairness: 25 vs 65 at their T50 with same lifestyle → similar scores
    >>> young = synergy_score(total_T=toy_T50(25), age=25, sex="male", T50_fn=toy_T50, T95_fn=toy_T95,
    ...                       training_days_per_week=5, avg_sleep_hours=8.5, BMI=23)["Score"]
    >>> older = synergy_score(total_T=toy_T50(65), age=65, sex="male", T50_fn=toy_T50, T95_fn=toy_T95,
    ...                       training_days_per_week=5, avg_sleep_hours=8.5, BMI=23)["Score"]
    >>> abs(round(young - older, 1)) <= 2.0
    True
    """
    pass

if __name__ == "__main__":
    # Simple printout demo
    for age in [25, 45, 65]:
        sex = "male"
        lifestyle = dict(training_days_per_week=6, avg_sleep_hours=8.8, BMI=23)
        for T in [300, 500, 700, 900]:
            s = synergy_score(total_T=T, age=age, sex=sex,
                              T50_fn=toy_T50, T95_fn=toy_T95,
                              **lifestyle)
            print(f"age={age} T={T} -> Score={s['Score']:.1f} (H={s['H']:.2f}, L={s['L']:.2f})")
