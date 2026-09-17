from __future__ import annotations

import warnings
from pathlib import Path

import librosa
import numpy as np
import pywt
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from scipy.stats import kurtosis, skew


TARGET_SR = 2000
EPS = 1e-9


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = np.nan_to_num(y.astype(np.float32))
    if y.size < target_sr:
        y = np.pad(y, (0, target_sr - y.size))
    y -= np.median(y)
    peak = np.percentile(np.abs(y), 99.5)
    if peak > EPS:
        y = np.clip(y / peak, -1.0, 1.0)
    return y, target_sr


def bandpass(y: np.ndarray, sr: int, low: float = 20.0, high: float = 800.0) -> np.ndarray:
    sos = butter(4, [low, min(high, sr * 0.49)], btype="bandpass", fs=sr, output="sos")
    try:
        return sosfiltfilt(sos, y).astype(np.float32)
    except ValueError:
        return y.astype(np.float32)


def _frame_features(y: np.ndarray, sr: int) -> tuple[np.ndarray, list[str]]:
    n_fft, hop = 256, 64
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop,
                                    n_mels=40, fmin=20, fmax=800)
        delta = librosa.feature.delta(mfcc)
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                                   roll_percent=0.85)
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                                     n_bands=4, fmin=20)
    matrix = np.vstack([mfcc, delta, rms, zcr, centroid, bandwidth, rolloff, flatness, contrast])
    names = ([f"mfcc_{i+1}" for i in range(20)] + [f"delta_{i+1}" for i in range(20)] +
             ["rms", "zcr", "centroid", "bandwidth", "rolloff", "flatness"] +
             [f"contrast_{i+1}" for i in range(contrast.shape[0])])
    return matrix, names


def _summary(matrix: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    stats = {
        "mean": np.mean(matrix, axis=1),
        "std": np.std(matrix, axis=1),
        "median": np.median(matrix, axis=1),
        "p10": np.percentile(matrix, 10, axis=1),
        "p90": np.percentile(matrix, 90, axis=1),
    }
    values, out_names = [], []
    for stat, vector in stats.items():
        values.extend(vector.tolist())
        out_names.extend([f"{name}_{stat}" for name in names])
    return np.asarray(values, dtype=np.float32), out_names


def extract_features(path: str | Path) -> tuple[np.ndarray, list[str]]:
    y_raw, sr = load_audio(path)
    y = bandpass(y_raw, sr)
    matrix, base_names = _frame_features(y, sr)
    values, names = _summary(matrix, base_names)

    duration = len(y) / sr
    abs_y = np.abs(y)
    clip_fraction = float(np.mean(np.abs(y_raw) >= 0.999))
    silence_fraction = float(np.mean(abs_y < max(np.percentile(abs_y, 20), EPS)))
    crest = float(np.max(abs_y) / (np.sqrt(np.mean(y * y)) + EPS))
    q = np.percentile(y, [1, 10, 25, 75, 90, 99])
    extra = np.asarray([duration, clip_fraction, silence_fraction, crest, *q], dtype=np.float32)
    extra_names = ["duration", "clip_fraction", "silence_fraction", "crest_factor",
                   "amp_p01", "amp_p10", "amp_p25", "amp_p75", "amp_p90", "amp_p99"]

    # Physiologically motivated frequency bands and periodicity of the amplitude envelope.
    freq, psd = welch(y, fs=sr, nperseg=min(1024, len(y)))
    total_power = float(np.trapezoid(psd, freq)) + EPS
    bands = [(20, 60), (60, 120), (120, 200), (200, 400), (400, 800)]
    band_values = []
    for lo, hi in bands:
        mask = (freq >= lo) & (freq < hi)
        band_values.append(float(np.trapezoid(psd[mask], freq[mask]) / total_power))
    psd_prob = psd / (np.sum(psd) + EPS)
    spectral_entropy = float(-np.sum(psd_prob * np.log(psd_prob + EPS)) / np.log(len(psd_prob)))

    envelope = np.abs(hilbert(y))
    envelope = envelope - np.mean(envelope)
    max_lag = min(int(1.5 * sr), len(envelope) - 1)
    min_lag = min(int(0.30 * sr), max_lag)
    if max_lag > min_lag:
        # FFT autocorrelation is substantially faster than direct correlation on long records.
        n = 1 << (2 * len(envelope) - 1).bit_length()
        spectrum = np.fft.rfft(envelope, n=n)
        ac = np.fft.irfft(spectrum * np.conj(spectrum), n=n)[:max_lag + 1]
        ac /= ac[0] + EPS
        lag = int(np.argmax(ac[min_lag:max_lag + 1]) + min_lag)
        periodicity = float(ac[lag])
        rate_bpm = float(60 * sr / lag)
    else:
        periodicity, rate_bpm = 0.0, 0.0

    coeffs = pywt.wavedec(y, "db4", level=min(6, pywt.dwt_max_level(len(y), 8)))
    wavelet_values, wavelet_names = [], []
    coeff_energy = np.asarray([np.sum(c * c) for c in coeffs], dtype=float)
    coeff_energy /= np.sum(coeff_energy) + EPS
    for i, (c, energy) in enumerate(zip(coeffs, coeff_energy)):
        wavelet_values.extend([energy, np.std(c), skew(c), kurtosis(c, fisher=True)])
        wavelet_names.extend([f"wavelet_{i}_energy", f"wavelet_{i}_std",
                              f"wavelet_{i}_skew", f"wavelet_{i}_kurtosis"])

    physio = np.asarray([*band_values, spectral_entropy, periodicity, rate_bpm,
                         *wavelet_values], dtype=np.float32)
    physio_names = ([f"bandpower_{lo}_{hi}" for lo, hi in bands] +
                    ["spectral_entropy", "envelope_periodicity", "rate_bpm_proxy"] +
                    wavelet_names)
    return (np.nan_to_num(np.concatenate([values, extra, physio])),
            names + extra_names + physio_names)
