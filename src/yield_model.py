"""Disease-specific potential yield-impact scenarios.

The coefficients are explicit sensitivity-scenario assumptions. They are kept
separate from measured model metrics because the bundled datasets contain no
field-level yield observations from which these coefficients could be fitted.
"""

from __future__ import annotations

from typing import Any

import numpy as np


YIELD_PROFILES: dict[str, dict[str, Any]] = {
    "early_blight": {
        "display_name": "Early Blight",
        "beta": 0.6,
        "beta_min": 0.5,
        "beta_max": 0.7,
        "citation": "Saha, P., & Das, S. (2012). Assessment of Yield Loss Due to Early Blight (Alternaria solani) in Tomato. Indian Journal of Plant Protection, 40(3), 195–198.",
        "citation_status": "Study reports an absolute slope in t/ha per severity point; it does not validate beta=0.6 as a universal percentage coefficient.",
    },
    "late_blight": {
        "display_name": "Late Blight",
        "beta": 1.1,
        "beta_min": 1.0,
        "beta_max": 1.2,
        "citation": "Fontem, D. A. (2003). Quantitative Effects of Early and Late Blights on Tomato Yields in Cameroon. Tropicultura, 21(1), 36–41; Nowicki et al. (2012), Plant Disease 96(1), 4–17, doi:10.1094/PDIS-05-11-0458.",
        "citation_status": "Field evidence supports potentially severe losses but not a universal beta=1.1 mapping.",
    },
    "bacterial_spot": {
        "display_name": "Bacterial Spot",
        "beta": 0.4,
        "beta_min": 0.3,
        "beta_max": 0.45,
        "citation": "Pohronezny, K., & Volin, R. B. (1983). The effect of bacterial spot on yield and quality of fresh market tomatoes. HortScience, 18(1), 69–70. https://doi.org/10.21273/HORTSCI.18.1.69",
        "citation_status": "The paper reports yield effects but does not by itself validate beta=0.4 for this dataset.",
    },
}


def normalize_disease(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def calculate_yield_loss(disease: str, severity: float, beta: float | None = None) -> float:
    """Return a bounded potential yield-loss percentage for one scenario."""
    key = normalize_disease(disease)
    if key == "healthy" or key not in YIELD_PROFILES:
        return 0.0
    coefficient = float(YIELD_PROFILES[key]["beta"] if beta is None else beta)
    return float(np.clip(coefficient * float(severity), 0.0, 100.0))


def yield_loss_interval(disease: str, severity: float) -> dict[str, float]:
    key = normalize_disease(disease)
    if key not in YIELD_PROFILES:
        return {"low": 0.0, "central": 0.0, "high": 0.0}
    profile = YIELD_PROFILES[key]
    return {
        "low": calculate_yield_loss(key, severity, float(profile["beta_min"])),
        "central": calculate_yield_loss(key, severity, float(profile["beta"])),
        "high": calculate_yield_loss(key, severity, float(profile["beta_max"])),
    }


def metadata() -> dict[str, Any]:
    return {
        "model": "critical-point linear sensitivity scenario",
        "formula": "YL = clip(beta * severity_percent, 0, 100)",
        "scientific_status": "Potential impact scenario; not calibrated against field-yield ground truth.",
        "severity_source": "Semantic lesion mask divided by a documented heuristic leaf mask.",
        "profiles": YIELD_PROFILES,
    }
