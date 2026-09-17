from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_sound.dataset import discover_records
from heart_sound.features import extract_features


def process_record(row: dict):
    try:
        values, names = extract_features(row["path"])
        return {**row, **dict(zip(names, values, strict=True))}, None
    except Exception as exc:
        return None, {"record": row["record"], "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "work" / "data" / "training")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "features.csv")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    records = discover_records(args.data)
    completed = set()
    if args.output.exists():
        completed = set(pd.read_csv(args.output, usecols=["record"])["record"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    pending = [row for row in records if row["record"] not in completed]
    for start in range(0, len(pending), 100):
        batch = pending[start:start + 100]
        results = Parallel(n_jobs=args.jobs, prefer="processes")(
            delayed(process_record)(row) for row in batch)
        good = [out for out, failure in results if out is not None]
        failures.extend(failure for out, failure in results if failure is not None)
        if good:
            pd.DataFrame(good).to_csv(args.output, mode="a", index=False,
                                      header=not args.output.exists())
        done = len(completed) + min(start + len(batch), len(pending))
        print(f"features {done}/{len(records)}; failures={len(failures)}", flush=True)
    (args.output.parent / "feature_failures.json").write_text(json.dumps(failures, indent=2))
    print(f"complete: {len(records)-len(failures)} records, {len(failures)} failures")


if __name__ == "__main__":
    main()
