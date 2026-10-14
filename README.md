# CardioScreen: heart-sound screening research prototype

[![CI](https://github.com/MouradMgad/Cardio-Screen/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MouradMgad/Cardio-Screen/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

CardioScreen is a reproducible research prototype for binary screening of normal versus
abnormal phonocardiogram (PCG) recordings. It uses the open PhysioNet/CinC 2016 heart-sound
dataset and an ensemble of engineered audio representations and classical machine
learning models.

**Research use only.** CardioScreen is not a medical device and does not provide a diagnosis.
Results are recording-level benchmark estimates, not evidence of clinical validity.

When a recording passes the high-confidence abnormal threshold, a second stage now reports
one of two **descriptive acoustic patterns**:

- Murmur-like acoustic pattern: broad, noise-like frequency energy is dominant.
- Rhythm/timing-like acoustic pattern: irregular cycle timing is dominant.

These are not disease diagnoses. The PhysioNet 2016 training records do not contain reliable
record-level disease, murmur, S1/S2, or stethoscope-position labels, so accuracy, precision,
and recall cannot honestly be calculated for these two pattern names.

### Second-stage validation

On 99 held-out recordings that have genuine abnormal labels, the descriptive stage produced
55 murmur-like and 44 rhythm/timing-like results. Its mean agreement across 200 bootstrap
recalibrations was 97.7%, and mean agreement after small feature perturbations was 94.9%.
These numbers measure stability, not medical correctness. Pattern proportions also varied
substantially between source databases, so equipment and collection conditions may influence
the second-stage result. See `outputs/model_final/phenotype_validation.json`.

## Research question

Can an ensemble distinguish abnormal from normal digital-stethoscope recordings and identify
low-confidence cases for review?

## Run the finished application

Clone the repository, then run these commands in PowerShell from the project folder:

```powershell
git clone https://github.com/MouradMgad/Cardio-Screen.git
Set-Location Cardio-Screen
.\setup.ps1
.\run_app.ps1
```

Then open `http://127.0.0.1:7860` and upload a WAV recording. The trained binary model and
validation outputs are included in `outputs/model_final`, so training is not required to run
the screening application.

`.\setup.ps1` creates a local virtual environment and installs the runtime and development
packages. Python 3.12 is used for continuous integration.

## Quick checks

After running setup, run the source-level test suite and the application smoke test:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tests\smoke_test.py
```

The phase-research integration tests are skipped when the generated patient-level artifacts
are absent; reproduce them first with `.\run_research.ps1`. GitHub Actions runs the lightweight
unit tests and application smoke test on pushes and pull requests.

## Reproduce training

1. Download `training.zip` from the official PhysioNet/CinC 2016 page.
2. Extract it below `work/data/training` so the `training-a` through `training-f` folders
   and their `REFERENCE.csv` files are discoverable.
3. Run `.\setup.ps1` if the dependencies have not already been installed.
4. Run `.\run_pipeline.ps1` from PowerShell. It uses the existing feature cache for a fast
   retraining run.
5. Run `.\run_pipeline.ps1 -ReextractFeatures` to rebuild all 301 features from every WAV
   file before training.

The pipeline checkpoints extracted features record by record, so interrupted extraction can
resume without discarding completed work. Final artifacts are written to `outputs/model_final`.
The PhysioNet/CinC and CirCor recordings are not included in this repository; see
[`docs/DATASET.md`](docs/DATASET.md) for sources, expected paths, and citation requirements.

## Final held-out result

The forced binary model achieved 93.6% accuracy, 92.9% recall, 93.8% specificity, 79.3%
precision, and 0.975 AUROC on 486 held-out recordings. Because a forced answer did not meet
the precision target, the final validation-locked policy uses three outcomes:

- Likely normal: abnormality probability at or below 0.215.
- Uncertain / refer or record again: probability between 0.215 and 0.825.
- Abnormal / expert review advised: probability at or above 0.825.

On decisive held-out recordings this policy achieved 92.1% precision, 97.2% recall, 99.1%
specificity, and 78.6% coverage. If uncertain and abnormal results both trigger referral,
screening sensitivity was 99.0%. These are recording-level benchmark results, not clinical
validation.

## Method summary

- Fourth-order 20–800 Hz band-pass filtering and robust amplitude normalization.
- MFCC, delta-MFCC, spectral contrast, centroid, bandwidth, roll-off, flatness, energy,
  zero-crossing, duration, clipping, and amplitude-distribution features.
- Mean, standard deviation, median, 10th percentile, and 90th percentile temporal summaries.
- Stratification jointly by source database and class for 70/15/15 development splits.
- Final 3:1 probability blend of class-balanced Extra Trees and LightGBM, selected after
  comparing five candidate models on the validation split.
- Classification threshold chosen once on validation F1 and applied unchanged to the test set.
- Test reporting includes precision, recall/sensitivity, specificity, F1, balanced accuracy,
  AUROC, average precision, Brier score, confusion matrix, and per-record predictions.

## Important limitation

The released training collection does not provide reliable patient identifiers across every
source. The primary split therefore stratifies by source and class at recording level. This
can overestimate patient-level generalization if one patient contributed multiple recordings.
Results must be described as a recording-level benchmark, not clinical validation. Source-wise
robustness and external patient-level validation are recommended before any clinical use.

## Dataset citation

Liu C, Springer D, Li Q, et al. An open access database for the evaluation of heart sound
algorithms. *Physiological Measurement*. 2016;37(12):2181–2213.

Dataset: https://physionet.org/content/challenge-2016/1.0.0/

## Repository layout

- `app.py`, `templates/`: local WAV-upload demo.
- `src/heart_sound/`: audio features, data discovery, phase decoding, and descriptive pattern logic.
- `scripts/`: feature extraction, training, evaluation, and report generation.
- `tests/`: unit tests, optional research-artifact checks, and application smoke test.
- `outputs/model_final/`: bundled binary screening model and benchmark summaries.
- `docs/`: dataset instructions, research results, and project notes.

For bug reports, include the command you ran, Python version, and full error output. Do not
submit patient-identifiable recordings or sensitive health data as issues.
