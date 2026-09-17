from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


EPS = 1e-9


@dataclass(frozen=True)
class PhenotypeResult:
    label: str
    confidence: float
    murmur_score: float
    rhythm_score: float
    explanation: str


def derived_markers(features: Mapping[str, float], typical_rate: float) -> dict[str, float]:
    """Return interpretable acoustic markers used by the second-stage screen."""
    delta_std = [float(features.get(f"delta_{i}_std", 0.0)) for i in range(1, 21)]
    rms_mean = abs(float(features.get("rms_mean", 0.0)))
    rate = float(features.get("rate_bpm_proxy", typical_rate))
    return {
        "spectral_entropy": float(features.get("spectral_entropy", 0.0)),
        "spectral_flatness": float(features.get("flatness_mean", 0.0)),
        "zero_crossing_rate": float(features.get("zcr_mean", 0.0)),
        "spectral_centroid": float(features.get("centroid_mean", 0.0)),
        "high_frequency_fraction": sum(
            float(features.get(name, 0.0))
            for name in ("bandpower_120_200", "bandpower_200_400", "bandpower_400_800")
        ),
        "low_periodicity": -float(features.get("envelope_periodicity", 0.0)),
        "rate_deviation": abs(rate - typical_rate),
        "energy_variation": float(features.get("rms_std", 0.0)) / (rms_mean + EPS),
        "mfcc_change": float(np.mean(np.abs(delta_std))),
    }


def _robust_z(value: float, center: float, scale: float) -> float:
    return float(np.clip((value - center) / max(scale, EPS), -4.0, 4.0))


def classify_abnormal_pattern(
    features: Mapping[str, float], artifact: Mapping[str, object]
) -> PhenotypeResult:
    """Classify an already-abnormal recording into one of two acoustic patterns.

    This is a descriptive phenotype, not a disease diagnosis. The source dataset has
    no record-level disease or murmur labels.
    """
    typical_rate = float(artifact["typical_rate"])
    markers = derived_markers(features, typical_rate)
    calibration = artifact["calibration"]

    murmur_names = artifact["axes"]["murmur_like"]
    rhythm_names = artifact["axes"]["rhythm_timing_like"]
    murmur_z = [
        _robust_z(markers[name], calibration[name]["median"], calibration[name]["scale"])
        for name in murmur_names
    ]
    rhythm_z = [
        _robust_z(markers[name], calibration[name]["median"], calibration[name]["scale"])
        for name in rhythm_names
    ]
    murmur_score = float(np.mean(murmur_z))
    rhythm_score = float(np.mean(rhythm_z))
    probability_murmur = float(1.0 / (1.0 + np.exp(-(murmur_score - rhythm_score))))

    if probability_murmur >= 0.5:
        label = "Murmur-like acoustic pattern"
        confidence = probability_murmur
        explanation = "Broad, noise-like energy is more dominant than timing irregularity."
    else:
        label = "Rhythm/timing-like acoustic pattern"
        confidence = 1.0 - probability_murmur
        explanation = "Cycle timing irregularity is more dominant than broad murmur-like energy."

    return PhenotypeResult(
        label=label,
        confidence=confidence,
        murmur_score=murmur_score,
        rhythm_score=rhythm_score,
        explanation=explanation,
    )
