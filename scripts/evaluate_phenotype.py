from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heart_sound.phenotypes import classify_abnormal_pattern, derived_markers


def split_indices(frame: pd.DataFrame, seed: int = 42):
    strat = frame["source"].astype(str) + "_" + frame["label"].astype(str)
    train, temporary = train_test_split(
        np.arange(len(frame)), test_size=0.30, random_state=seed, stratify=strat
    )
    temporary_strat = strat.iloc[temporary]
    validation, test = train_test_split(
        temporary, test_size=0.50, random_state=seed, stratify=temporary_strat
    )
    return train, validation, test


def is_abnormal(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"1", "true"})


def robust_calibration(marker_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for name in marker_frame.columns:
        values = marker_frame[name].to_numpy(float)
        median = float(np.nanmedian(values))
        q25, q75 = np.nanpercentile(values, [25, 75])
        result[name] = {
            "median": median,
            "scale": float(max((q75 - q25) / 1.349, 1e-6)),
        }
    return result


def label_rows(frame: pd.DataFrame, artifact: dict) -> tuple[np.ndarray, np.ndarray]:
    results = [classify_abnormal_pattern(row, artifact) for row in frame.to_dict("records")]
    labels = np.asarray([result.label for result in results])
    confidence = np.asarray([result.confidence for result in results], dtype=float)
    return labels, confidence


def perturb_row(row: dict, rng: np.random.Generator) -> dict:
    perturbed = dict(row)
    positive_names = [
        "spectral_entropy", "flatness_mean", "zcr_mean", "centroid_mean",
        "bandpower_120_200", "bandpower_200_400", "bandpower_400_800",
        "rms_mean", "rms_std", *[f"delta_{i}_std" for i in range(1, 21)],
    ]
    for name in positive_names:
        value = float(perturbed.get(name, 0.0))
        perturbed[name] = value * float(np.exp(rng.normal(0.0, 0.05)))
    perturbed["envelope_periodicity"] = float(np.clip(
        float(perturbed.get("envelope_periodicity", 0.0)) + rng.normal(0.0, 0.02), 0.0, 1.0
    ))
    perturbed["rate_bpm_proxy"] = float(
        float(perturbed.get("rate_bpm_proxy", 0.0)) + rng.normal(0.0, 2.0)
    )
    return perturbed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-runs", type=int, default=200)
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    artifact = joblib.load(args.model)
    binary = json.loads(args.binary_metrics.read_text())
    train_idx, _, test_idx = split_indices(frame)
    training_abnormal = frame.iloc[train_idx][is_abnormal(frame.iloc[train_idx]["label"])]
    test_abnormal = frame.iloc[test_idx][is_abnormal(frame.iloc[test_idx]["label"])]

    baseline_labels, baseline_confidence = label_rows(test_abnormal, artifact)
    murmur_name = "Murmur-like acoustic pattern"
    murmur_count = int(np.sum(baseline_labels == murmur_name))

    rng = np.random.default_rng(42)
    perturb_agreement = []
    rows = test_abnormal.to_dict("records")
    for _ in range(30):
        perturbed_labels = np.asarray([
            classify_abnormal_pattern(perturb_row(row, rng), artifact).label for row in rows
        ])
        perturb_agreement.append(float(np.mean(perturbed_labels == baseline_labels)))

    marker_frame = pd.DataFrame([
        derived_markers(row, float(artifact["typical_rate"]))
        for row in training_abnormal.to_dict("records")
    ])
    bootstrap_agreement = []
    for _ in range(args.bootstrap_runs):
        sample = marker_frame.iloc[rng.integers(0, len(marker_frame), len(marker_frame))]
        candidate = deepcopy(artifact)
        candidate["calibration"] = robust_calibration(sample)
        candidate_labels, _ = label_rows(test_abnormal, candidate)
        bootstrap_agreement.append(float(np.mean(candidate_labels == baseline_labels)))

    source_table = pd.crosstab(test_abnormal["source"], baseline_labels)
    source_rates = {}
    for source, row in source_table.iterrows():
        total = int(row.sum())
        source_rates[str(source)] = {
            "n": total,
            "murmur_like_fraction": float(row.get(murmur_name, 0) / max(total, 1)),
        }

    result = {
        "binary_classifier_real_ground_truth": {
            "test_n": int(binary["test"]["n"]),
            "accuracy": float(binary["test"]["accuracy"]),
            "precision": float(binary["test"]["precision"]),
            "recall_sensitivity": float(binary["test"]["recall_sensitivity"]),
            "specificity": float(binary["test"]["specificity"]),
            "roc_auc": float(binary["test"]["roc_auc"]),
        },
        "pattern_classifier_held_out_checks": {
            "held_out_abnormal_n": int(len(test_abnormal)),
            "murmur_like_n": murmur_count,
            "rhythm_timing_like_n": int(len(test_abnormal) - murmur_count),
            "mean_confidence": float(np.mean(baseline_confidence)),
            "confidence_at_least_70_percent_fraction": float(np.mean(baseline_confidence >= 0.70)),
            "bootstrap_calibration_agreement_mean": float(np.mean(bootstrap_agreement)),
            "bootstrap_calibration_agreement_95pct_interval": [
                float(np.percentile(bootstrap_agreement, 2.5)),
                float(np.percentile(bootstrap_agreement, 97.5)),
            ],
            "small_feature_perturbation_agreement_mean": float(np.mean(perturb_agreement)),
            "small_feature_perturbation_agreement_min": float(np.min(perturb_agreement)),
            "source_pattern_distribution": source_rates,
        },
        "validity_statement": {
            "accuracy_precision_recall_available": False,
            "reason": (
                "The PhysioNet 2016 records provide normal/abnormal ground truth but no "
                "record-level labels for murmur-like versus rhythm/timing-like patterns. "
                "Stability is not diagnostic accuracy."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
