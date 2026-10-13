"""Cardiac-phase segmentation and phase-specific PCG features.

The segmenter is trained on CirCor's S1/systole/S2/diastole annotations.  Its
predictions on another dataset are inferred boundaries, not ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import librosa
import numpy as np
from scipy.signal import hilbert, welch

from .features import EPS, TARGET_SR, bandpass, load_audio

PHASES = ("s1", "systole", "s2", "diastole")
FRAME_LENGTH = 128
HOP_LENGTH = 40  # 20 ms at 2 kHz


@dataclass(frozen=True)
class PhaseSegmentation:
    audio: np.ndarray
    sample_rate: int
    frame_labels: np.ndarray
    frame_probabilities: np.ndarray
    confidence: float


def frame_features(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Return compact frame-level features used by the transferred segmenter."""
    y = bandpass(np.asarray(audio, dtype=np.float32), sr)
    envelope = np.abs(hilbert(y)).astype(np.float32)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0] / (sr / 2)
    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    env_rms = librosa.feature.rms(
        y=envelope, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    length = min(map(len, (rms, zcr, centroid, flatness, env_rms)))
    matrix = np.column_stack((rms[:length], env_rms[:length], zcr[:length],
                              centroid[:length], flatness[:length]))
    matrix[:, :2] = np.log1p(20.0 * matrix[:, :2])
    for col in range(matrix.shape[1]):
        median = np.median(matrix[:, col])
        scale = np.median(np.abs(matrix[:, col] - median)) * 1.4826 + EPS
        matrix[:, col] = np.clip((matrix[:, col] - median) / scale, -6, 6)
    delta = np.gradient(matrix, axis=0)
    return np.nan_to_num(np.column_stack((matrix, delta))).astype(np.float32)


def _cyclic_decode(probabilities: np.ndarray, transition_penalty: float = 2.2) -> np.ndarray:
    """Viterbi decode constrained to stay in a phase or advance around the cycle."""
    logp = np.log(np.clip(probabilities, 1e-8, 1.0))
    frames, states = logp.shape
    score = np.full((frames, states), -np.inf)
    back = np.zeros((frames, states), dtype=np.int8)
    score[0] = logp[0]
    for t in range(1, frames):
        for state in range(states):
            previous = (state - 1) % states
            stay_score = score[t - 1, state]
            advance_score = score[t - 1, previous] - transition_penalty
            if advance_score > stay_score:
                score[t, state] = advance_score + logp[t, state]
                back[t, state] = previous
            else:
                score[t, state] = stay_score + logp[t, state]
                back[t, state] = state
    labels = np.empty(frames, dtype=np.int8)
    labels[-1] = int(np.argmax(score[-1]))
    for t in range(frames - 1, 0, -1):
        labels[t - 1] = back[t, labels[t]]
    return labels


def segment_audio(path: str | Path, artifact: Mapping[str, object]) -> PhaseSegmentation:
    raw, sr = load_audio(path)
    audio = bandpass(raw, sr)
    features = frame_features(audio, sr)
    model = artifact["model"]
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    labels = _cyclic_decode(probabilities, float(artifact.get("transition_penalty", 2.2)))
    confidence = float(np.mean(probabilities[np.arange(len(labels)), labels]))
    return PhaseSegmentation(audio, sr, labels, probabilities, confidence)


def labels_to_sample_mask(labels: np.ndarray, n_samples: int, phase: int) -> np.ndarray:
    # Assign each sample to exactly one frame label. Using the full overlapping
    # analysis windows here would let a boundary sample belong to two phases.
    frame = np.minimum(np.arange(n_samples) // HOP_LENGTH, len(labels) - 1)
    return labels[frame] == phase


def _phase_summary(signal: np.ndarray, sr: int, prefix: str) -> dict[str, float]:
    if len(signal) < FRAME_LENGTH:
        signal = np.pad(signal, (0, FRAME_LENGTH - len(signal)))
    abs_y = np.abs(signal)
    rms = float(np.sqrt(np.mean(signal * signal)) + EPS)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(signal)))))
    freq, psd = welch(signal, fs=sr, nperseg=min(512, len(signal)))
    total = float(np.trapezoid(psd, freq)) + EPS
    probability = psd / (np.sum(psd) + EPS)
    entropy = float(-np.sum(probability * np.log(probability + EPS)) / np.log(len(probability)))
    centroid = float(np.sum(freq * psd) / (np.sum(psd) + EPS))
    bands = {}
    for lo, hi in ((20, 60), (60, 120), (120, 200), (200, 400), (400, 800)):
        use = (freq >= lo) & (freq < hi)
        bands[f"band_{lo}_{hi}"] = float(np.trapezoid(psd[use], freq[use]) / total)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=8, n_fft=256,
                                hop_length=64, n_mels=24, fmin=20, fmax=800)
    result = {
        f"{prefix}_rms": rms,
        f"{prefix}_rms_variation": float(np.std(abs_y) / (np.mean(abs_y) + EPS)),
        f"{prefix}_zcr": zcr,
        f"{prefix}_centroid": centroid,
        f"{prefix}_entropy": entropy,
        f"{prefix}_crest": float(np.max(abs_y) / rms),
    }
    result.update({f"{prefix}_{name}": value for name, value in bands.items()})
    for index in range(mfcc.shape[0]):
        result[f"{prefix}_mfcc_{index + 1}_mean"] = float(np.mean(mfcc[index]))
        result[f"{prefix}_mfcc_{index + 1}_std"] = float(np.std(mfcc[index]))
    return result


def extract_phase_features(
    path: str | Path,
    artifact: Mapping[str, object],
    mask_phase: str | None = None,
    noise_seed: int | None = None,
    segmentation: PhaseSegmentation | None = None,
) -> tuple[dict[str, float], PhaseSegmentation]:
    """Extract phase-specific features, optionally applying a counterfactual edit."""
    segmentation = segmentation or segment_audio(path, artifact)
    audio = segmentation.audio.copy()
    masks = [labels_to_sample_mask(segmentation.frame_labels, len(audio), i) for i in range(4)]
    if mask_phase is not None:
        phase_index = PHASES.index(mask_phase)
        # Replace the selected phase by low-level matched noise instead of silence.
        target = masks[phase_index]
        rng = np.random.default_rng(noise_seed if noise_seed is not None else 42)
        scale = float(np.std(audio[target])) if np.any(target) else 0.0
        audio[target] = rng.normal(0.0, max(scale * 0.05, EPS), int(np.sum(target)))
    result: dict[str, float] = {"segmentation_confidence": segmentation.confidence}
    total_energy = float(np.sum(audio * audio)) + EPS
    for index, phase in enumerate(PHASES):
        samples = audio[masks[index]]
        result.update(_phase_summary(samples, segmentation.sample_rate, phase))
        result[f"{phase}_duration_fraction"] = float(np.mean(masks[index]))
        result[f"{phase}_energy_fraction"] = float(np.sum(samples * samples) / total_energy)
    result["systolic_to_diastolic_energy"] = (
        result["systole_energy_fraction"] / (result["diastole_energy_fraction"] + EPS)
    )
    result["s1_to_s2_energy"] = result["s1_energy_fraction"] / (
        result["s2_energy_fraction"] + EPS
    )
    return result, segmentation


def load_phase_artifact(path: str | Path) -> dict:
    return joblib.load(path)
