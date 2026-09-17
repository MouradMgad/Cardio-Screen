from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heart_sound.phenotypes import classify_abnormal_pattern, derived_markers


def training_indices(frame: pd.DataFrame, seed: int = 42) -> np.ndarray:
    strat = frame["source"].astype(str) + "_" + frame["label"].astype(str)
    train, _ = train_test_split(
        np.arange(len(frame)), test_size=0.30, random_state=seed, stratify=strat
    )
    return train


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.nanmedian(values))
    q25, q75 = np.nanpercentile(values, [25, 75])
    scale = float(max((q75 - q25) / 1.349, 1e-6))
    return median, scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.features)
    train = frame.iloc[training_indices(frame)]
    abnormal = train[train["label"].astype(str).str.lower().isin({"1", "true"})].copy()
    if len(abnormal) < 100:
        raise RuntimeError("Too few abnormal training examples for phenotype calibration")

    typical_rate = float(np.nanmedian(train["rate_bpm_proxy"].to_numpy(float)))
    rows = [derived_markers(row, typical_rate) for row in abnormal.to_dict("records")]
    marker_frame = pd.DataFrame(rows)
    calibration = {}
    for name in marker_frame.columns:
        median, scale = robust_scale(marker_frame[name].to_numpy(float))
        calibration[name] = {"median": median, "scale": scale}

    artifact = {
        "version": "1.0.0",
        "type": "descriptive_abnormal_acoustic_phenotype",
        "typical_rate": typical_rate,
        "calibration": calibration,
        "axes": {
            "murmur_like": [
                "spectral_entropy",
                "spectral_flatness",
                "zero_crossing_rate",
                "spectral_centroid",
                "high_frequency_fraction",
            ],
            "rhythm_timing_like": [
                "low_periodicity",
                "rate_deviation",
                "energy_variation",
                "mfcc_change",
            ],
        },
        "warning": (
            "Descriptive acoustic pattern only. The PhysioNet 2016 records do not have "
            "record-level disease, murmur, S1/S2, or auscultation-position labels."
        ),
    }

    labels, confidences = [], []
    for row in abnormal.to_dict("records"):
        result = classify_abnormal_pattern(row, artifact)
        labels.append(result.label)
        confidences.append(result.confidence)

    counts = pd.Series(labels).value_counts().to_dict()
    summary = {
        "calibration_records": int(len(abnormal)),
        "training_records_total": int(len(train)),
        "pattern_counts": {str(k): int(v) for k, v in counts.items()},
        "mean_pattern_confidence": float(np.mean(confidences)),
        "important_note": artifact["warning"],
        "accuracy_precision_recall": "Not computable without subtype ground-truth labels.",
    }
    joblib.dump(artifact, args.output / "phenotype_model.joblib", compress=3)
    (args.output / "phenotype_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
