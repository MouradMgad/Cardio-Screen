# Project file guide

The runnable project root is:

`C:\Users\0\Downloads\new work`

## Commands

- `setup.ps1` creates `.venv` and installs the Python packages when needed.
- `run_app.ps1` starts the local web application at `http://127.0.0.1:7860`.
- `run_pipeline.ps1` trains, evaluates, builds the report, and runs the smoke test. It uses
  the cached feature table unless `-ReextractFeatures` is supplied.

## Application

- `app.py` receives a WAV upload, extracts the same features used during training, loads the
  saved ensemble, and returns likely normal, uncertain, or likely abnormal. High-confidence
  abnormal recordings also receive a descriptive murmur-like or rhythm/timing-like pattern.
- `templates/index.html` contains the web page displayed by Flask.

## Heart-sound code

- `src/heart_sound/dataset.py` finds the six PhysioNet source folders and reads their labels.
- `src/heart_sound/features.py` converts audio to mono, resamples it to 2 kHz, filters and
  normalizes it, then calculates the 301 numerical features.
- `src/heart_sound/phenotypes.py` calculates the two descriptive abnormal sound-pattern
  scores. It does not claim to diagnose a named disease.
- `src/heart_sound/__init__.py` marks the directory as the `heart_sound` Python package.

## Training and evaluation

- `scripts/extract_features.py` processes every WAV recording and creates
  `work/features_v2.csv`.
- `scripts/tune.py` creates the fixed train, validation, and test partitions, compares model
  candidates, selects the 3:1 Extra Trees and LightGBM ensemble, and saves the fitted model.
- `scripts/selective_eval.py` chooses the lower and upper validation thresholds for the
  three-way output and evaluates them on the held-out test recordings.
- `scripts/train_phenotype.py` calibrates the murmur-like and rhythm/timing-like acoustic
  axes on abnormal training recordings and saves the second-stage artifact.
- `scripts/evaluate_phenotype.py` evaluates held-out pattern balance, bootstrap calibration
  stability, small-perturbation stability, confidence, and source dependence. It deliberately
  does not report subtype accuracy because subtype ground truth does not exist.
- `scripts/build_report.py` creates `outputs/CardioScreen_Project_Report.pdf` from the saved
  metrics and predictions.
- `tests/smoke_test.py` checks the saved predictions and verifies examples from all three
  output categories.
- `tests/test_phenotypes.py` verifies both second-stage acoustic-pattern paths.

## Data and generated files

- `.venv/` is the isolated Python environment created by `setup.ps1`; the run scripts use it
  so they do not depend on packages installed elsewhere on the computer.
- `work/data/training/` contains the 3,240 extracted PhysioNet WAV recordings and reference
  label files. It is needed only for complete feature extraction or retraining.
- `work/features_v2.csv` is the cached 3,240 by 301 feature table. Keeping it makes retraining
  much faster.
- `outputs/model_final/heart_sound_model.joblib` is the trained Extra Trees and LightGBM
  ensemble together with its feature order, blend weights, and binary threshold.
- `outputs/model_final/phenotype_model.joblib` stores the robust calibration values and
  feature definitions for the two abnormal sound patterns.
- `outputs/model_final/phenotype_summary.json` records calibration counts and the essential
  limitation that subtype accuracy cannot be calculated without ground-truth subtype labels.
- `outputs/model_final/phenotype_validation.json` contains the honest quantitative validation
  checks for the second stage alongside the labeled binary-classifier metrics.
- `outputs/model_final/metrics.json` contains validation and binary test metrics.
- `outputs/model_final/selective_metrics.json` contains the 0.215 and 0.825 decision limits
  and the three-way test results.
- `outputs/model_final/test_predictions.csv` contains one prediction for every held-out test
  recording.
- `outputs/CardioScreen_Project_Report.pdf` is the detailed project report.
- `outputs/Heart-Abnormality_Detection_Poster_Mourad_Mohamed_Gad_THRESHOLDS_FINAL.pptx` is
  the editable scientific poster.
- `outputs/Heart-Abnormality_Detection_Poster_Preview_Thresholds_FINAL.png` is the poster
  preview.
- `outputs/Example_Normal_e01432.wav` and `outputs/Example_Abnormal_e01623.wav` are example
  recordings used to demonstrate the input format.

## Documentation

- `README.md` gives the quick start, method summary, results, and limitations.
- `DATASET.md` records the dataset source, license, and required citation.
- `requirements.txt` lists the Python dependencies.
- `LICENSE` contains the software license for this project.
- `.gitignore` prevents local environments, Python caches, and temporary report charts from
  being committed to source control.
