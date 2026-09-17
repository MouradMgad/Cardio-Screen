# CardioScreen: CPU-first abnormal heart-sound screening

Prepared by Mourad Mohamed Gad.

CardioScreen is a reproducible research prototype for binary screening of normal versus
abnormal phonocardiogram (PCG) recordings. It uses the open PhysioNet/CinC 2016 heart-sound
dataset and an efficient ensemble of engineered audio representations and classical machine
learning models.

> Research use only. This project is not a medical device and does not provide a diagnosis.

## Research question

Can a fast CPU-trained ensemble distinguish abnormal from normal digital-stethoscope
recordings with strong precision, recall, specificity, and calibration while exposing an
uncertain result for low-confidence cases?

## Reproduce training

1. Download `training.zip` from the official PhysioNet/CinC 2016 page.
2. Extract it below `work/data/training` so the `training-a` through `training-f` folders
   and their `REFERENCE.csv` files are discoverable.
3. Install `requirements.txt` or set `PYTHONPATH` to the self-contained dependencies.
4. Run `./run_pipeline.ps1` from PowerShell.

The pipeline checkpoints extracted features record by record, so interrupted extraction can
resume without discarding completed work. Final artifacts are written to `outputs/model_final`.

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

## Run the local demo

Run `./run_app.ps1`, open `http://127.0.0.1:7860`, and upload a mono or stereo WAV file.
The service resamples audio to 2 kHz, applies the same preprocessing used in training, and
reports the calibrated abnormality probability and an uncertainty-aware screening result.

## Method summary

- Fourth-order 20–800 Hz band-pass filtering and robust amplitude normalization.
- MFCC, delta-MFCC, spectral contrast, centroid, bandwidth, roll-off, flatness, energy,
  zero-crossing, duration, clipping, and amplitude-distribution features.
- Mean, standard deviation, median, 10th percentile, and 90th percentile temporal summaries.
- Stratification jointly by source database and class for 70/15/15 development splits.
- Soft-voting ensemble of class-balanced Extra Trees, histogram gradient boosting, and RBF SVM.
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
