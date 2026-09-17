from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import app


def main():
    predictions = pd.read_csv(ROOT / "outputs" / "model_final" / "test_predictions.csv")
    cases = [
        (predictions[(predictions.label == 0) & (predictions.probability_abnormal <= 0.215)].iloc[0],
         b"Likely normal"),
        (predictions[(predictions.label == 1) & (predictions.probability_abnormal >= 0.825)].iloc[0],
         b"Abnormal"),
        (predictions[(predictions.probability_abnormal > 0.215) &
                     (predictions.probability_abnormal < 0.825)].iloc[0], b"Uncertain"),
    ]
    client = app.test_client()
    for row, expected in cases:
        audio_path = Path(str(row.path).strip())
        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path
        if not audio_path.exists():
            audio_path = ROOT / "work" / "data" / "training" / row.source / Path(str(row.path).strip()).name
        with audio_path.open("rb") as audio:
            response = client.post("/", data={"audio": (audio, audio_path.name)},
                                   content_type="multipart/form-data")
        assert response.status_code == 200
        assert expected in response.data, (row.record, expected, response.data[:500])
        print(f"PASS {row.record}: {expected.decode()}")


if __name__ == "__main__":
    main()
