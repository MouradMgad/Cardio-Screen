from __future__ import annotations

import numpy as np

from heart_sound.phases import PHASES, _cyclic_decode, labels_to_sample_mask


def test_cyclic_decoder_respects_phase_order():
    probabilities = np.full((40, 4), 0.01)
    expected = np.repeat(np.arange(4), 10)
    probabilities[np.arange(40), expected] = 0.97
    decoded = _cyclic_decode(probabilities)
    assert np.array_equal(decoded, expected)


def test_phase_masks_are_present_and_distinct():
    labels = np.repeat(np.arange(4), 5)
    masks = [labels_to_sample_mask(labels, 1000, index) for index in range(4)]
    assert PHASES == ("s1", "systole", "s2", "diastole")
    assert all(np.any(mask) for mask in masks)
    assert not np.array_equal(masks[0], masks[1])


if __name__ == "__main__":
    test_cyclic_decoder_respects_phase_order()
    test_phase_masks_are_present_and_distinct()
    print("PASS: phase decoder and masks")
