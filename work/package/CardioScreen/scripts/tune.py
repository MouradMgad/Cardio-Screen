from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train import choose_threshold, metrics, split_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "work" / "features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_tuned")
    args = parser.parse_args()
    df = pd.read_csv(args.features)
    metadata = ["record", "path", "label", "source"]
    feature_names = [c for c in df.columns if c not in metadata]
    X = df[feature_names].replace([np.inf, -np.inf], np.nan)
    y = df["label"].astype(int).to_numpy()
    train_idx, val_idx, test_idx = split_indices(df)

    candidates = {
        "svm_c3": Pipeline([("impute", SimpleImputer(strategy="median")),
                            ("scale", RobustScaler()),
                            ("model", SVC(C=3, class_weight="balanced", probability=True,
                                          random_state=42))]),
        "svm_c10": Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", RobustScaler()),
                             ("model", SVC(C=10, class_weight="balanced", probability=True,
                                           random_state=42))]),
        "extra_deep": Pipeline([("impute", SimpleImputer(strategy="median")),
                                ("model", ExtraTreesClassifier(n_estimators=900, min_samples_leaf=1,
                                 max_features=0.45, class_weight="balanced", n_jobs=-1,
                                 random_state=42))]),
        "lgb_balanced": LGBMClassifier(n_estimators=700, learning_rate=0.025, num_leaves=15,
                              min_child_samples=20, subsample=0.85, colsample_bytree=0.75,
                              reg_lambda=2.0, class_weight="balanced", n_jobs=6,
                              random_state=42, verbosity=-1),
        "lgb_moderate": LGBMClassifier(n_estimators=700, learning_rate=0.025, num_leaves=15,
                              min_child_samples=20, subsample=0.85, colsample_bytree=0.75,
                              reg_lambda=2.0, scale_pos_weight=2.5, n_jobs=6,
                              random_state=42, verbosity=-1),
    }
    val_predictions, fitted, candidate_metrics = {}, {}, {}
    for name, model in candidates.items():
        print(f"training {name}", flush=True)
        model.fit(X.iloc[train_idx], y[train_idx])
        p = model.predict_proba(X.iloc[val_idx])[:, 1]
        threshold = choose_threshold(y[val_idx], p)
        val_predictions[name] = p
        fitted[name] = model
        candidate_metrics[name] = metrics(y[val_idx], p, threshold)

    # Select a blend only with validation predictions. Integer weights keep the search auditable.
    best = None
    names = list(candidates)
    for weights in itertools.product(range(4), repeat=len(names)):
        if sum(weights) == 0 or sum(w > 0 for w in weights) < 2:
            continue
        p = sum(w * val_predictions[n] for w, n in zip(weights, names)) / sum(weights)
        threshold = choose_threshold(y[val_idx], p)
        m = metrics(y[val_idx], p, threshold)
        rank = (m["f1"], m["balanced_accuracy"], m["roc_auc"])
        if best is None or rank > best[0]:
            best = (rank, weights, threshold, m)
    assert best is not None
    _, weights, threshold, val_metrics = best
    active = {n: int(w) for n, w in zip(names, weights) if w}
    print(f"selected blend {active} threshold={threshold:.3f}", flush=True)
    test_prob = sum(active[n] * fitted[n].predict_proba(X.iloc[test_idx])[:, 1]
                    for n in active) / sum(active.values())
    test_metrics = metrics(y[test_idx], test_prob, threshold)
    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    artifact = {"models": {n: fitted[n] for n in active}, "weights": active,
                "threshold": threshold, "feature_names": feature_names, "version": "2.0.0"}
    joblib.dump(artifact, outdir / "heart_sound_model.joblib", compress=3)
    predictions = df.iloc[test_idx][metadata].copy()
    predictions["probability_abnormal"] = test_prob
    predictions["prediction"] = (test_prob >= threshold).astype(int)
    predictions.to_csv(outdir / "test_predictions.csv", index=False)
    result = {"candidate_validation": candidate_metrics, "selected_weights": active,
              "validation": val_metrics, "test": test_metrics}
    (outdir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"selected_weights": active, "validation": val_metrics,
                      "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
