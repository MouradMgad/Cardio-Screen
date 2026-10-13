# CardioScreen

CardioScreen is a research prototype for heart-sound screening from phonocardiogram
recordings. The project starts with the PhysioNet/CinC 2016 normal-versus-abnormal
benchmark, then extends it into a phase-aware research experiment that asks whether
S1, systole, S2, and diastole information can make valvular-disease prediction more
physiologically trustworthy.

This is not a medical device. It is a reproducible machine-learning project for
research, education, and method development.

## Why this project matters

Most heart-sound classifiers can learn strong dataset-specific shortcuts: recording
source, stethoscope position, patient overlap, noise profile, or global murmur energy.
Those shortcuts can look impressive in ordinary metrics while still failing the harder
question: is the model using the part of the cardiac cycle where the disease should
actually be audible?

CardioScreen is built around that problem. It combines:

- A CPU-friendly abnormal heart-sound screening baseline.
- A three-way selective decision policy: likely normal, uncertain, likely abnormal.
- A descriptive abnormal-sound stage for murmur-like versus rhythm/timing-like patterns.
- An experimental S1/S2-aware pipeline for valvular disease labels.
- A physiology audit using systolic and diastolic counterfactual masking.

## Current results

### PhysioNet/CinC 2016 abnormal screening

The forced binary model achieved the following held-out recording-level results:

| Metric | Value |
| --- | ---: |
| Accuracy | 93.6% |
| Recall / sensitivity | 92.9% |
| Specificity | 93.8% |
| Precision | 79.3% |
| AUROC | 0.975 |

Because forced binary decisions did not meet the precision target, the final policy uses
selective thresholds:

| Output | Probability rule |
| --- | --- |
| Likely normal | `p(abnormal) <= 0.215` |
| Uncertain / repeat or refer | `0.215 < p(abnormal) < 0.825` |
| Likely abnormal | `p(abnormal) >= 0.825` |

On decisive held-out recordings, the selective policy reached 92.1% precision, 97.2%
recall, 99.1% specificity, and 78.6% coverage. If uncertain and abnormal results both
trigger referral, screening sensitivity was 99.0%.

### Phase-aware disease research extension

The research extension trains an S1/systole/S2/diastole segmenter from CirCor annotations,
then evaluates AS, AR, MR, and MS prediction on the BMD-HS dataset with patient-grouped
cross-validation.

| Model | Macro AUROC | Macro average precision | Macro Brier |
| --- | ---: | ---: | ---: |
| Whole-recording features | 0.766 | 0.632 | 0.199 |
| Phase features | 0.772 | 0.645 | 0.193 |
| Phase features + physiologic constraints | 0.775 | 0.654 | 0.192 |

The counterfactual audit supported systolic dependence for aortic stenosis and mitral
regurgitation, but did not support the expected diastolic dependence for aortic
regurgitation and mitral stenosis. Those negative findings are preserved because they are
scientifically useful: they show where the current method is not yet reliable.

See [docs/PHASE_RESEARCH_RESULTS.md](docs/PHASE_RESEARCH_RESULTS.md) for the full
experiment summary.

## Repository layout

```text
.
+-- app.py                         # Flask demo app
+-- src/heart_sound/               # Feature extraction, phenotype, and phase utilities
+-- scripts/                       # Training, evaluation, download, and research pipelines
+-- tests/                         # Smoke and research integrity tests
+-- templates/                     # Web UI template
+-- outputs/                       # Reproducible model/metric artifacts
+-- docs/                          # Dataset, research, and project notes
+-- run_app.ps1                    # Start the local web app
+-- run_pipeline.ps1               # Reproduce the abnormal-screening pipeline
+-- run_research.ps1               # Reproduce the phase-aware research experiment
+-- requirements.txt
```

Generated datasets, feature caches, local environments, old exported reports, and package
archives are intentionally ignored by Git.

## Quick start

From PowerShell:

```powershell
cd "C:\Users\mourad\heartabnormaldetection"
.\setup.ps1
.\run_app.ps1
```

Then open `http://127.0.0.1:7860` and upload a WAV recording.

The app uses the saved artifacts in `outputs/model_final`. A full retraining run is not
required just to try the demo.

## Reproduce the abnormal-screening pipeline

1. Download `training.zip` from the official PhysioNet/CinC 2016 page.
2. Extract it to `work/data/training` so the `training-a` through `training-f` folders are
   directly below that directory.
3. Run:

```powershell
.\setup.ps1
.\run_pipeline.ps1
```

To rebuild features from every WAV file instead of using a local cache:

```powershell
.\run_pipeline.ps1 -ReextractFeatures
```

Final artifacts are written to `outputs/model_final`.

## Reproduce the phase-aware research experiment

Create the CirCor subset:

```powershell
.\.venv\Scripts\python.exe scripts\download_circor_subset.py
```

Place the BMD-HS dataset at `work/data/bmd_hs`, then run:

```powershell
.\run_research.ps1
```

The research pipeline writes:

- `outputs/phase_model`: compact S1/systole/S2/diastole segmenter.
- `outputs/phase_disease_model`: patient-level disease metrics, predictions, and
  counterfactual phase-masking results.

## Method summary

- Audio is converted to mono, resampled to 2 kHz, band-pass filtered from 20 to 800 Hz,
  and robustly normalized.
- Whole-recording features include MFCCs, delta MFCCs, spectral contrast, centroid,
  bandwidth, roll-off, flatness, energy, zero crossing, duration, clipping, and
  amplitude-distribution summaries.
- The binary screening model blends class-balanced Extra Trees and LightGBM.
- Selective thresholds are chosen on validation data and then locked for held-out testing.
- The phase-aware extension extracts cardiac-cycle-specific features and compares them
  against whole-recording baselines with patient-grouped evaluation.

## Limitations

- PhysioNet/CinC 2016 does not provide reliable disease-subtype, murmur-grade,
  stethoscope-position, S1, or S2 labels for every record. Subtype accuracy cannot be
  claimed on that benchmark.
- Primary PhysioNet results are recording-level, not patient-level clinical validation.
- BMD-HS disease labels are useful for research but the cohort is small; external
  validation is still needed.
- The phase-aware disease model should be treated as a research audit, not a deployable
  diagnostic system.

## Dataset notes

This repository does not redistribute the source heart-sound datasets. See
[docs/DATASET.md](docs/DATASET.md) for provenance, download notes, and citation details.

PhysioNet/CinC 2016 citation:

> Liu C, Springer D, Li Q, et al. An open access database for the evaluation of heart
> sound algorithms. Physiological Measurement. 2016;37(12):2181-2213.

## License

Project code is released under the license in [LICENSE](LICENSE). Dataset files are governed
by their original dataset licenses.
