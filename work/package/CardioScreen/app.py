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

MODEL_DIR = ROOT / "outputs" / "model_final"
MODEL_PATH = MODEL_DIR / "heart_sound_model.joblib"
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
                if low < probability < high:
                    label = "Uncertain — record again"
                else:
                    label = "Abnormal — expert review advised" if probability >= high else "Likely normal"
                result = {"label": label, "probability": round(probability * 100, 1),
                          "confidence": round(confidence * 100, 1)}
            except Exception as exc:
                error = f"Unable to analyze this recording: {exc}"
            finally:
                if tmp_path and os.path.exists(tmp_path): os.unlink(tmp_path)
    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
