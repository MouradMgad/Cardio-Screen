"""Train a research-only, patient-held-out valvular-disease PCG classifier.

The BMD-HS data contain echocardiography-derived labels for AS, AR, MR and MS.
They also contain eight highly correlated recordings per patient, so this script
*never* splits recordings from the same patient across train and test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heart_sound.features import extract_features

LABELS = ["AS", "AR", "MR", "MS"]
DISPLAY = {"AS": "Aortic stenosis", "AR": "Aortic regurgitation",
           "MR": "Mitral regurgitation", "MS": "Mitral stenosis"}


def build_rows(dataset: Path) -> tuple[pd.DataFrame, list[str]]:
    labels = pd.read_csv(dataset / "train.csv", dtype=str).replace("N/A", np.nan)
    rows, names = [], None
    for _, patient in labels.dropna(subset=LABELS).iterrows():
        target = {key: int(patient[key]) for key in LABELS}
        for recording in patient.filter(regex=r"^recording_").dropna():
            wav = dataset / "train" / f"{recording}.wav"
            if not wav.exists():
                continue
            values, current_names = extract_features(wav)
            names = current_names if names is None else names
            rows.append({"patient_id": patient.patient_id, "recording": recording,
                         **target, **dict(zip(current_names, values, strict=True))})
    if not rows or names is None:
        raise RuntimeError("No labelled WAV recordings found in the BMD-HS dataset.")
    return pd.DataFrame(rows), names


def score(y: np.ndarray, p: np.ndarray) -> dict:
    result = {}
    for index, label in enumerate(LABELS):
        # A class can be absent in a small patient-held-out split.
        positives = int(y[:, index].sum())
        if 0 < positives < len(y):
            result[DISPLAY[label]] = {
                "test_positive_patients": positives,
                "auroc": float(roc_auc_score(y[:, index], p[:, index])),
                "average_precision": float(average_precision_score(y[:, index], p[:, index])),
            }
        else:
            result[DISPLAY[label]] = {"test_positive_patients": positives,
                                      "auroc": None, "average_precision": None}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "work" / "data" / "bmd_hs")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "disease_model")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    frame, features = build_rows(args.dataset)
    # Pick exactly one representative recording per patient for honest patient-level scoring.
    patient_rows = frame.sort_values("recording").groupby("patient_id", as_index=False).first()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_pat, test_pat = next(splitter.split(patient_rows, groups=patient_rows.patient_id))
    train_ids = set(patient_rows.iloc[train_pat].patient_id)
    test_ids = set(patient_rows.iloc[test_pat].patient_id)
    train = frame[frame.patient_id.isin(train_ids)]
    test = patient_rows[patient_rows.patient_id.isin(test_ids)]
    model = MultiOutputClassifier(ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
        n_jobs=-1, random_state=42
    ))
    X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train, y_test = train[LABELS].to_numpy(int), test[LABELS].to_numpy(int)
    model.fit(X_train, y_train)
    probabilities = np.column_stack([est.predict_proba(X_test)[:, 1] for est in model.estimators_])
    metrics = score(y_test, probabilities)
    predictions = test[["patient_id", "recording", *LABELS]].copy()
    for index, label in enumerate(LABELS): predictions[f"probability_{label}"] = probabilities[:, index]
    predictions.to_csv(args.output / "patient_held_out_predictions.csv", index=False)
    artifact = {"version": "1.0.0", "type": "research_valvular_disease_multilabel",
                "labels": LABELS, "display_labels": DISPLAY, "feature_names": features,
                "model": model,
                "warning": "Research only: small single-site dataset; not a diagnosis or clinical decision tool."}
    joblib.dump(artifact, args.output / "valve_disease_model.joblib", compress=3)
    summary = {"dataset": "BMD-HS", "dataset_license": "CC BY-NC-ND 4.0",
               "records_used": int(len(frame)), "labelled_patients": int(frame.patient_id.nunique()),
               "train_patients": int(len(train_ids)), "test_patients": int(len(test_ids)),
               "evaluation": "One representative recording per held-out patient; no patient leakage.",
               "metrics": metrics, "limitation": artifact["warning"]}
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
