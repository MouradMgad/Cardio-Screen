from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_patient_grouped_outputs():
    output = ROOT / "outputs" / "phase_disease_model"
    metrics = json.loads((output / "metrics.json").read_text())
    predictions = pd.read_csv(output / "patient_oof_predictions.csv")
    assert metrics["patients"] == 108
    assert len(predictions) == predictions.patient_id.nunique() == 108
    assert metrics["evaluation"].startswith("Five-fold patient-grouped")


def test_production_artifact_is_phase_aware():
    artifact = joblib.load(
        ROOT / "outputs" / "phase_disease_model" / "phase_valve_disease_model.joblib"
    )
    assert artifact["version"] == "2.0.0"
    assert set(artifact["labels"]) == {"AS", "AR", "MR", "MS"}
    assert set(artifact["models"]) == set(artifact["labels"])
    assert "phase_artifact" in artifact


if __name__ == "__main__":
    test_patient_grouped_outputs()
    test_production_artifact_is_phase_aware()
    print("PASS: phase-aware research outputs")
