from __future__ import annotations

import numpy as np
from scipy.io import wavfile

from heart_sound.features import extract_features


def test_feature_extraction_returns_finite_named_values(tmp_path):
    sample_rate = 2000
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    audio = (
        0.25 * np.sin(2 * np.pi * 55 * time)
        + 0.1 * np.sin(2 * np.pi * 120 * time)
    )
    path = tmp_path / "synthetic-heart-sound.wav"
    wavfile.write(path, sample_rate, np.asarray(audio * 30_000, dtype=np.int16))

    values, names = extract_features(path)

    assert values.ndim == 1
    assert values.size == len(names)
    assert values.size >= 300
    assert len(names) == len(set(names))
    assert np.isfinite(values).all()
