# Dataset provenance

This repository does not redistribute the source heart-sound datasets. Download them from
their original sources and place them under `work/data/`.

## PhysioNet/CinC 2016

Used for the normal-versus-abnormal screening benchmark.

- Source: https://physionet.org/content/challenge-2016/1.0.0/
- Expected local path: `work/data/training`
- Required structure: `training-a` through `training-f` folders with their `REFERENCE.csv`
  files directly below `work/data/training`.
- License: Open Data Commons Attribution License v1.0.

Citation:

> Liu C, Springer D, Li Q, et al. An open access database for the evaluation of heart
> sound algorithms. Physiological Measurement. 2016;37(12):2181-2213.
> https://doi.org/10.1088/0967-3334/37/12/2181

## CirCor subset

Used to train the S1/systole/S2/diastole segmenter from annotated heart-sound recordings.

Create the reproducible local subset with:

```powershell
.\.venv\Scripts\python.exe scripts\download_circor_subset.py
```

Expected local path: `work/data/circor_subset`.

## BMD-HS

Used for the research-only valvular-disease experiment covering AS, AR, MR, and MS labels.

Expected local path: `work/data/bmd_hs`.

The disease pipeline uses patient-grouped evaluation because the dataset contains multiple
recordings per patient. Do not split recordings from the same patient across train and test.
