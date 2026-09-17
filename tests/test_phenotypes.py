from heart_sound.phenotypes import classify_abnormal_pattern


def artifact():
    names = [
        "spectral_entropy", "spectral_flatness", "zero_crossing_rate",
        "spectral_centroid", "high_frequency_fraction", "low_periodicity",
        "rate_deviation", "energy_variation", "mfcc_change",
    ]
    return {
        "typical_rate": 70.0,
        "calibration": {name: {"median": 0.0, "scale": 1.0} for name in names},
        "axes": {
            "murmur_like": names[:5],
            "rhythm_timing_like": names[5:],
        },
    }


def base_features():
    values = {
        "spectral_entropy": 0.0, "flatness_mean": 0.0, "zcr_mean": 0.0,
        "centroid_mean": 0.0, "bandpower_120_200": 0.0,
        "bandpower_200_400": 0.0, "bandpower_400_800": 0.0,
        "envelope_periodicity": 0.0, "rate_bpm_proxy": 70.0,
        "rms_mean": 1.0, "rms_std": 0.0,
    }
    values.update({f"delta_{i}_std": 0.0 for i in range(1, 21)})
    return values


def test_murmur_like_axis_wins():
    features = base_features()
    features.update({
        "spectral_entropy": 3.0, "flatness_mean": 3.0, "zcr_mean": 3.0,
        "centroid_mean": 3.0, "bandpower_200_400": 3.0,
    })
    result = classify_abnormal_pattern(features, artifact())
    assert result.label == "Murmur-like acoustic pattern"


def test_rhythm_timing_axis_wins():
    features = base_features()
    features.update({
        "envelope_periodicity": -3.0, "rate_bpm_proxy": 74.0,
        "rms_std": 4.0,
    })
    features.update({f"delta_{i}_std": 4.0 for i in range(1, 21)})
    result = classify_abnormal_pattern(features, artifact())
    assert result.label == "Rhythm/timing-like acoustic pattern"


if __name__ == "__main__":
    test_murmur_like_axis_wins()
    test_rhythm_timing_axis_wins()
    print("PASS: both abnormal acoustic-pattern paths")
