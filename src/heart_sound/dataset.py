from __future__ import annotations

import csv
from pathlib import Path


def discover_records(root: str | Path) -> list[dict]:
    root = Path(root)
    rows: list[dict] = []
    for ref in root.rglob("REFERENCE.csv"):
        source = ref.parent.name
        with ref.open("r", encoding="utf-8-sig", newline="") as handle:
            for record, label, *_ in csv.reader(handle):
                wav = ref.parent / f"{record}.wav"
                if wav.exists() and label.strip() in {"-1", "1"}:
                    rows.append({"record": record, "path": str(wav.resolve()),
                                 "label": int(label) == 1, "source": source})
    if not rows:
        raise FileNotFoundError(f"No REFERENCE.csv/WAV pairs found below {root}")
    return rows

