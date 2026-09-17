from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]


def split_indices(meta: pd.DataFrame, seed: int = 42):
    strat = meta["source"].astype(str) + "_" + meta["label"].astype(str)
    train_idx, temp_idx = train_test_split(np.arange(len(meta)), test_size=0.30,
                                            random_state=seed, stratify=strat)
    temp_strat = strat.iloc[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed,
                                          stratify=temp_strat)
    return train_idx, val_idx, test_idx


def metrics(y, p, threshold: float) -> dict:
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold), "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def choose_threshold(y, p) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    predictions = p[None, :] >= candidates[:, None]
    positive = y[None, :] == 1
    negative = ~positive
    tp = np.sum(predictions & positive, axis=1)
    fp = np.sum(predictions & negative, axis=1)
    fn = np.sum(~predictions & positive, axis=1)
    tn = np.sum(~predictions & negative, axis=1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    balanced = 0.5 * (tp / np.maximum(tp + fn, 1) + tn / np.maximum(tn + fp, 1))
    index = np.lexsort((candidates, balanced, f1))[-1]
    return float(candidates[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "work" / "features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    metadata = ["record", "path", "label", "source"]
    feature_names = [c for c in df.columns if c not in metadata]
    X = df[feature_names].replace([np.inf, -np.inf], np.nan)
    y = df["label"].astype(int).to_numpy()
    train_idx, val_idx, test_idx = split_indices(df)

    extra = ExtraTreesClassifier(n_estimators=700, min_samples_leaf=2, max_features="sqrt",
                                  class_weight="balanced", n_jobs=-1, random_state=42)
    hist = HistGradientBoostingClassifier(max_iter=350, learning_rate=0.055, max_leaf_nodes=21,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=42)
    svm = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scale", RobustScaler()),
                    ("svc", SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True,
                                random_state=42))])
    base = VotingClassifier([("extra", extra), ("hist", hist), ("svm", svm)],
                            voting="soft", weights=[2, 1, 2], n_jobs=-1)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", base)])
    model.fit(X.iloc[train_idx], y[train_idx])
    val_p = model.predict_proba(X.iloc[val_idx])[:, 1]
    threshold = choose_threshold(y[val_idx], val_p)
    test_p = model.predict_proba(X.iloc[test_idx])[:, 1]
    result = {
        "dataset": {"total": len(df), "train": len(train_idx), "validation": len(val_idx),
                    "test": len(test_idx), "positive_rate": float(y.mean()),
                    "sources": df["source"].value_counts().to_dict()},
        "validation": metrics(y[val_idx], val_p, threshold),
        "test": metrics(y[test_idx], test_p, threshold),
    }
    test_out = df.iloc[test_idx][metadata].copy()
    test_out["probability_abnormal"] = test_p
    test_out["prediction"] = (test_p >= threshold).astype(int)
    test_out.to_csv(args.output / "test_predictions.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    artifact = {"model": model, "threshold": threshold, "feature_names": feature_names,
                "version": "1.0.0"}
    joblib.dump(artifact, args.output / "heart_sound_model.joblib", compress=3)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ConfusionMatrixDisplay.from_predictions(y[test_idx], test_p >= threshold,
                                            display_labels=["Normal", "Abnormal"], ax=ax,
                                            colorbar=False, cmap="Blues")
    fig.tight_layout(); fig.savefig(args.output / "confusion_matrix.png", dpi=180); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
