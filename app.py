from __future__ import annotations

import os
import json
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from heart_sound.features import extract_features
from heart_sound.phenotypes import classify_abnormal_pattern
from heart_sound.phases import extract_phase_features

MODEL_DIR = ROOT / "outputs" / "model_final"
MODEL_PATH = MODEL_DIR / "heart_sound_model.joblib"
PHENOTYPE_PATH = MODEL_DIR / "phenotype_model.joblib"
DISEASE_PATH = ROOT / "outputs" / "disease_model" / "valve_disease_model.joblib"
PHASE_DISEASE_PATH = ROOT / "outputs" / "phase_disease_model" / "phase_valve_disease_model.joblib"
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        upload = request.files.get("audio")
        if not upload or not upload.filename.lower().endswith(".wav"):
            error = "Please provide a WAV heart-sound recording."
        elif not MODEL_PATH.exists():
            error = "The trained model has not been generated yet."
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    upload.save(tmp); tmp_path = tmp.name
                artifact = joblib.load(MODEL_PATH)
                values, names = extract_features(tmp_path)
                row = pd.DataFrame([dict(zip(names, values, strict=True))])
                row = row.reindex(columns=artifact["feature_names"])
                probability = float(sum(artifact["weights"][name] * model.predict_proba(row)[0, 1]
                                        for name, model in artifact["models"].items()) /
                                    sum(artifact["weights"].values()))
                selective = json.loads((MODEL_DIR / "selective_metrics.json").read_text())
                low = float(selective["validation"]["low_threshold"])
                high = float(selective["validation"]["high_threshold"])
                confidence = max(probability, 1 - probability)
                phenotype = None
                disease_estimates = None
                if low < probability < high:
                    label = "Uncertain — record again"
                else:
                    label = "Abnormal — expert review advised" if probability >= high else "Likely normal"
                if probability >= high and PHENOTYPE_PATH.exists():
                    phenotype_artifact = joblib.load(PHENOTYPE_PATH)
                    pattern = classify_abnormal_pattern(
                        dict(zip(names, values, strict=True)), phenotype_artifact
                    )
                    phenotype = {
                        "label": pattern.label,
                        "confidence": round(pattern.confidence * 100, 1),
                        "explanation": pattern.explanation,
                    }
                if probability >= high and PHASE_DISEASE_PATH.exists():
                    disease_artifact = joblib.load(PHASE_DISEASE_PATH)
                    phase_values, segmentation = extract_phase_features(
                        tmp_path, disease_artifact["phase_artifact"]
                    )
                    location = request.form.get("location", "unknown")
                    posture = request.form.get("posture", "unknown")
                    for name in ("Mit", "Tri", "Pul", "Aor"):
                        phase_values[f"site_{name}"] = float(location == name)
                    for name in ("sup", "sit"):
                        phase_values[f"posture_{name}"] = float(posture == name)
                    estimates = []
                    for code in disease_artifact["labels"]:
                        columns = disease_artifact["feature_names_by_label"][code]
                        disease_row = pd.DataFrame([phase_values]).reindex(
                            columns=columns, fill_value=0
                        )
                        estimate = float(disease_artifact["models"][code].predict_proba(
                            disease_row
                        )[0, 1])
                        limits = disease_artifact["selective_limits"][code]
                        status = ("pattern present" if estimate >= limits["high"] else
                                  "pattern absent" if estimate <= limits["low"] else "uncertain")
                        estimates.append({"label": disease_artifact["display_labels"][code],
                                          "probability": round(estimate * 100, 1),
                                          "status": status})
                    disease_estimates = sorted(
                        estimates, key=lambda item: item["probability"], reverse=True
                    )
                    disease_estimates.append({
                        "label": "S1/S2 segmentation confidence",
                        "probability": round(segmentation.confidence * 100, 1),
                        "status": "inferred boundary quality",
                    })
                elif probability >= high and DISEASE_PATH.exists():
                    disease_artifact = joblib.load(DISEASE_PATH)
                    disease_row = row.reindex(columns=disease_artifact["feature_names"], fill_value=0)
                    estimates = []
                    for code, estimator in zip(disease_artifact["labels"], disease_artifact["model"].estimators_):
                        estimate = float(estimator.predict_proba(disease_row)[0, 1])
                        estimates.append({"label": disease_artifact["display_labels"][code],
                                          "probability": round(estimate * 100, 1)})
                    disease_estimates = sorted(estimates, key=lambda item: item["probability"], reverse=True)
                result = {"label": label, "probability": round(probability * 100, 1),
                          "confidence": round(confidence * 100, 1),
                          "phenotype": phenotype, "disease_estimates": disease_estimates}
            except Exception as exc:
                error = f"Unable to analyze this recording: {exc}"
            finally:
                if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)
    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
