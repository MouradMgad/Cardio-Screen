"""Train a lightweight S1/systole/S2/diastole segmenter from CirCor annotations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heart_sound.features import TARGET_SR, bandpass
from heart_sound.phases import HOP_LENGTH, PHASES, frame_features


def annotation_labels(tsv: Path, n_frames: int) -> np.ndarray:
    labels = np.full(n_frames, -1, dtype=np.int8)
    table = pd.read_csv(tsv, sep="\t", header=None, names=["start", "stop", "state"])
    times = np.arange(n_frames) * HOP_LENGTH / TARGET_SR
    for row in table.itertuples(index=False):
        state = int(row.state)
        if 1 <= state <= 4:
            labels[(times >= float(row.start)) & (times < float(row.stop))] = state - 1
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "phase_model")
    parser.add_argument("--max-recordings", type=int, default=500)
    parser.add_argument("--frames-per-class", type=int, default=25000)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    wavs = sorted(args.data.rglob("*.wav"))
    rng.shuffle(wavs); wavs = wavs[:args.max_recordings]
    chunks, targets, groups = [], [], []
    used = 0
    for wav in wavs:
        tsv = wav.with_suffix(".tsv")
        if not tsv.exists():
            continue
        audio, _ = librosa.load(wav, sr=TARGET_SR, mono=True)
        audio = bandpass(np.nan_to_num(audio.astype(np.float32)), TARGET_SR)
        x = frame_features(audio)
        y = annotation_labels(tsv, len(x))
        keep = y >= 0
        if not np.any(keep):
            continue
        chunks.append(x[keep]); targets.append(y[keep])
        groups.extend([wav.stem.split("_")[0]] * int(np.sum(keep)))
        used += 1
    if not chunks:
        raise RuntimeError(f"No paired WAV/TSV annotations found under {args.data}")
    X, y, group = np.vstack(chunks), np.concatenate(targets), np.asarray(groups)
    selected = []
    for state in range(4):
        indices = np.flatnonzero(y == state)
        selected.extend(rng.choice(indices, min(len(indices), args.frames_per_class), replace=False))
    selected = np.asarray(selected); rng.shuffle(selected)
    X, y, group = X[selected], y[selected], group[selected]
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, group))
    model = HistGradientBoostingClassifier(max_iter=220, learning_rate=.07,
                                           max_leaf_nodes=31, min_samples_leaf=30,
                                           l2_regularization=1.0, random_state=42)
    model.fit(X[train], y[train], sample_weight=compute_sample_weight("balanced", y[train]))
    prediction = model.predict(X[test])
    result = {
        "source": "CirCor DigiScope 1.0.3",
        "recordings_used": used,
        "train_frames": int(len(train)), "test_frames": int(len(test)),
        "held_out_subject_balanced_accuracy": float(balanced_accuracy_score(y[test], prediction)),
        "confusion_matrix": confusion_matrix(y[test], prediction, labels=range(4)).tolist(),
        "phases": list(PHASES),
        "note": "Frame metric before cyclic decoding; transferred BMD boundaries remain inferred.",
    }
    artifact = {"version": "1.0.0", "model": model, "phases": list(PHASES),
                "target_sr": TARGET_SR, "hop_length": HOP_LENGTH,
                "transition_penalty": 2.2, "metrics": result}
    joblib.dump(artifact, args.output / "phase_segmenter.joblib", compress=3)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
