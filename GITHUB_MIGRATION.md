# Moving CardioScreen to another computer

The GitHub repository contains the source code, trained models, cached feature tables,
reports, poster files, example WAV recordings, tests, and result files.

The following machine-specific or oversized folders are intentionally not stored in GitHub:

- `.venv/` and `.uv-cache/`: recreated by `setup.ps1`.
- `work/data/`: the original PhysioNet audio dataset. Download `training.zip` again from
  <https://physionet.org/content/challenge-2016/1.0.0/> and extract it as described in
  `DATASET.md` when complete feature re-extraction is required.
- temporary report-rendering and quality-assurance image folders.

## On the new Windows computer

```powershell
git clone REPOSITORY_URL
cd heart-abnormality-detection
.\setup.ps1
.\run_app.ps1
```

Open <http://127.0.0.1:7860>. The committed model in `outputs/model_final` lets the app run
without retraining. The committed `work/features_v2.csv` allows CPU retraining without
downloading the raw audio again.
