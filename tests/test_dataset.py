from __future__ import annotations

import pytest

from heart_sound.dataset import discover_records


def test_discover_records_reads_valid_reference_rows(tmp_path):
    source = tmp_path / "training-a"
    source.mkdir()
    (source / "REFERENCE.csv").write_text(
        "\ufeffnormal,-1\nabnormal,1\ninvalid,0\nmissing,1\n",
        encoding="utf-8",
    )
    (source / "normal.wav").write_bytes(b"wav")
    (source / "abnormal.wav").write_bytes(b"wav")
    (source / "invalid.wav").write_bytes(b"wav")

    records = discover_records(tmp_path)

    assert [record["record"] for record in records] == ["normal", "abnormal"]
    assert [record["label"] for record in records] == [False, True]
    assert all(record["source"] == "training-a" for record in records)
    assert all(record["path"].endswith(".wav") for record in records)


def test_discover_records_requires_reference_and_wav_pairs(tmp_path):
    with pytest.raises(FileNotFoundError, match="No REFERENCE.csv/WAV pairs"):
        discover_records(tmp_path)
