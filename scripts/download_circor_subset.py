"""Download a reproducible paired WAV/TSV subset of open CirCor 1.0.3."""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://physionet.org/files/circor-heart-sound/1.0.3/"


def fetch(url: str, destination: Path) -> tuple[str, bool]:
    if destination.exists() and destination.stat().st_size > 0:
        return destination.name, True
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(destination)
        return destination.name, True
    except Exception:
        if temporary.exists(): temporary.unlink()
        return destination.name, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recordings", type=int, default=400)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records = urllib.request.urlopen(BASE + "RECORDS").read().decode().splitlines()
    # Select subjects first, then locations, so the subset spans many independent people.
    by_subject: dict[str, list[str]] = {}
    for record in records:
        stem = record.rsplit("/", 1)[-1]
        by_subject.setdefault(stem.split("_")[0], []).append(record)
    subjects = sorted(by_subject); random.Random(42).shuffle(subjects)
    selected = []
    for subject in subjects:
        locations = sorted(by_subject[subject])
        selected.extend(locations)
        if len(selected) >= args.recordings:
            break
    selected = selected[:args.recordings]
    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in selected:
            stem = record.rsplit("/", 1)[-1]
            for extension in (".wav", ".tsv"):
                jobs.append(pool.submit(fetch, BASE + record + extension,
                                        args.output / f"{stem}{extension}"))
        completed = 0; failures = []
        for future in as_completed(jobs):
            name, ok = future.result(); completed += 1
            if not ok: failures.append(name)
            if completed % 100 == 0:
                print(f"downloaded {completed}/{len(jobs)} files; failures={len(failures)}", flush=True)
    manifest = {"source": BASE, "seed": 42, "requested_recordings": args.recordings,
                "selected_records": selected, "failures": failures}
    (args.output / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"recordings": len(selected), "failed_files": len(failures)}, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} CirCor subset files could not be downloaded")


if __name__ == "__main__":
    main()
