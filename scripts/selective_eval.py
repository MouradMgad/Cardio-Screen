from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train import split_indices


def predict(artifact, X):
    return sum(artifact["weights"][name] * model.predict_proba(X)[:, 1]
               for name, model in artifact["models"].items()) / sum(artifact["weights"].values())


def selective_metrics(y, p, low, high):
    accepted = (p <= low) | (p >= high)
    pred = p[accepted] >= high
    yt = y[accepted]
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return {
        "low_threshold": float(low), "high_threshold": float(high),
        "coverage": float(accepted.mean()), "uncertain_count": int((~accepted).sum()),
        "accepted_n": int(accepted.sum()),
        "precision": float(precision_score(yt, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(yt, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        # Under a screening policy, abnormal and uncertain both trigger expert referral.
        "referral_sensitivity": float(np.mean(p[y == 1] > low)),
        "referral_rate": float(np.mean(p > low)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=ROOT / "work" / "features_v2.csv")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "outputs" / "model_final")
    args = parser.parse_args()
    df = pd.read_csv(args.features)
    meta = ["record", "path", "label", "source"]
    names = [c for c in df.columns if c not in meta]
    X = df[names].replace([np.inf, -np.inf], np.nan)
    y = df["label"].astype(int).to_numpy()
    _, val_idx, test_idx = split_indices(df)
    artifact = joblib.load(args.model_dir / "heart_sound_model.joblib")
    val_p, test_p = predict(artifact, X.iloc[val_idx]), predict(artifact, X.iloc[test_idx])
    feasible = []
    for low in np.linspace(0.05, 0.40, 71):
        for high in np.linspace(0.45, 0.90, 91):
            m = selective_metrics(y[val_idx], val_p, low, high)
            if min(m["precision"], m["recall_sensitivity"], m["specificity"]) >= 0.99:
                feasible.append((m["coverage"], -m["referral_rate"], low, high, m))
    if not feasible:
        raise RuntimeError("No validation operating point satisfies the prespecified 90% constraints")
    _, _, low, high, val_result = max(feasible)
    test_result = selective_metrics(y[test_idx], test_p, low, high)
    result = {"selection_rule": "maximize validation coverage subject to precision, recall, and specificity >= 0.99 (conservative safety margin for a 0.90 test target)",
              "validation": val_result, "test": test_result}
    out = args.model_dir / "selective_metrics.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
