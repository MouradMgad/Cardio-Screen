# Proposed research direction

## Working title

**Does the model hear the valve lesion or the recording environment? A physiology-constrained,
phase-aware and shortcut-resistant system for multi-label valvular heart-disease screening**

## The unresolved problem

Published phonocardiogram models often report high classification scores, but a correct label
does not prove that the model used the clinically relevant sound. It may instead exploit recording
device, hospital noise, subject identity, posture, auscultation location, or preprocessing artifacts.

This is especially important for BMD-HS:

- diseased subjects were recorded in a cardiac hospital;
- healthy controls were recorded in a university laboratory, with recorded hospital noise played
  in an attempt to reproduce the environment;
- every patient contributes eight correlated recordings;
- the dataset supplies echocardiogram-confirmed valve labels but no S1/S2 boundary annotations;
- CirCor supplies S1/S2/systole/diastole annotations and murmur descriptions, but not matching
  valve-disease diagnoses.

Therefore, a model can perform well without demonstrating that its decision follows the expected
physiology: systolic murmur energy between S1 and S2 for aortic stenosis or mitral regurgitation,
and diastolic energy after S2 for aortic or mitral regurgitation/stenosis patterns.

## Research gap

Existing work separately addresses disease classification, cardiac-cycle segmentation,
explainability, synthetic heart sounds, and cross-domain normal/abnormal classification. The
specific gap proposed here is a unified disease-level study that:

1. transfers S1/S2 phase knowledge from an annotated murmur dataset to an
   echocardiogram-labelled multi-disease dataset;
2. measures whether disease decisions actually depend on the expected cardiac phase and
   auscultation location;
3. detects shortcut learning through counterfactual audio tests;
4. evaluates multi-label disease predictions strictly by patient, with uncertainty and
   calibration rather than accuracy alone.

The novelty claim should be phrased conservatively: a literature search did not identify a public
study combining all four elements on BMD-HS. It should not claim that no related work exists.

## Main research question

**Can physiology-constrained S1/S2-aware learning improve patient-level robustness and make
valvular-disease predictions depend more on clinically plausible murmur regions than on recording
shortcuts?**

## Hypotheses

- **H1:** A conventional whole-recording classifier will lose performance when background,
  location, or phase information is counterfactually altered, revealing shortcut use.
- **H2:** A phase-aware model will improve macro-AUROC, macro-average-precision, and calibration
  under patient-held-out testing, especially for lesions whose timing is diagnostically important.
- **H3:** Masking the clinically relevant phase will reduce the correct disease probability more
  than masking an irrelevant phase for the phase-aware model.
- **H4:** Combining all four auscultation sites for a patient will be more reliable than using one
  recording, while remaining interpretable through site-specific attention or feature attribution.
- **H5:** An explicit uncertain output will reduce false confident disease suggestions under
  cross-domain and noisy-input tests.

## Proposed system

### Stage 1: signal-quality gate

Reject or request a new recording when clipping, silence, extreme noise, or absent periodic heart
activity makes interpretation unreliable.

### Stage 2: cardiac-cycle segmentation

Train or adapt an S1/systole/S2/diastole segmenter on CirCor, which provides those annotations.
Apply confidence checks when transferring it to BMD-HS. Do not treat automatically inferred
boundaries as ground truth.

### Stage 3: phase-specific representation

For every cardiac cycle, calculate separate representations for:

- S1;
- systole;
- S2;
- diastole;
- full-cycle context.

Use spectral energy, entropy, duration, envelope shape, wavelets, MFCCs, and learned embeddings.
Retain auscultation location and posture as explicit inputs rather than letting filename or patient
identity leak into the model.

### Stage 4: physiology-constrained multi-label classifier

Predict AS, AR, MR, and MS independently so multi-valve disease is naturally supported. Add a
soft constraint or auxiliary loss that encourages attention to clinically plausible phases without
forcing every patient to match a textbook pattern.

### Stage 5: calibrated selective output

Return per-disease probabilities and an uncertain/refer result. Calibrate thresholds only on
validation patients. Never label the output as a confirmed diagnosis.

## Critical experiments

### E1. Leakage-safe baseline

Use grouped nested cross-validation. All eight recordings from a patient must remain in one fold.
Report patient-level results by aggregating site/posture recordings.

### E2. Phase-aware ablation

Compare:

1. whole-recording features;
2. automatically segmented phase features;
3. phase features plus location/posture;
4. phase-aware model with physiological constraints.

### E3. Counterfactual phase test

For each test recording:

- suppress systole;
- suppress diastole;
- suppress S1 or S2;
- swap systolic or diastolic intervals between patients;
- preserve total energy so the task is not trivial.

Measure the change in each disease probability. Expected examples include a larger AS/MR drop after
systolic masking and a larger AR/MS drop after diastolic masking.

### E4. Shortcut audit

- train a classifier to predict hospital-versus-control acquisition group from audio;
- test disease models after background replacement and band-limited noise injection;
- remove leading/trailing silence and normalize amplitude;
- test whether predictions change with posture or recording site more than clinically expected;
- compare performance with and without demographic metadata.

High acquisition-group predictability is evidence of confounding, not disease-detection success.

### E5. Cross-domain stress test

Use CirCor and PhysioNet for abnormal/murmur pretraining or stress testing, not as disease-label
ground truth. Test calibration under device, population, sampling-rate, and noise shifts.

### E6. Uncertainty evaluation

Report risk-coverage curves, calibration error, Brier score, and performance after abstaining on
low-confidence patients. Evaluate whether uncertainty rises under corrupted or out-of-domain audio.

## Evaluation measures

- Per-disease AUROC and average precision
- Macro and micro averages
- Sensitivity and specificity at validation-locked thresholds
- Multi-label F1 and Hamming loss
- Brier score and expected calibration error
- Risk versus coverage for selective prediction
- Bootstrap confidence intervals sampled by patient
- Counterfactual phase-sensitivity score
- Acquisition-shortcut predictability
- Results broken down by sex, age band, site, posture, single versus multi-valve disease

Because the disease classes are small, confidence intervals and fold-to-fold variance are essential.
A single train/test split is not enough evidence.

## Current baseline

The repository now contains a leakage-safe initial Extra Trees baseline trained on 864 BMD-HS
recordings from 108 labelled patients. One held-out recording per test patient produced:

| Disease | AUROC | Average precision |
|---|---:|---:|
| Aortic stenosis | 0.924 | 0.891 |
| Aortic regurgitation | 0.551 | 0.470 |
| Mitral regurgitation | 0.700 | 0.618 |
| Mitral stenosis | 0.852 | 0.878 |

These are feasibility results from only 27 held-out patients. They are not publication-ready and
do not establish clinical validity. AR is currently close to chance by AUROC.

## Minimum publishable contribution

A credible paper does not need to claim a clinically deployable diagnostic system. It can make the
more defensible contribution that:

1. exposes and quantifies acquisition shortcuts in disease-labelled PCG data;
2. introduces a reproducible patient-level counterfactual evaluation protocol;
3. tests whether transferred S1/S2 segmentation and physiological constraints reduce shortcut
   dependence;
4. releases code, folds, metrics, and failure analysis for reproducibility.

Even a negative result is useful: demonstrating that high internal accuracy collapses under
physiology-preserving counterfactual tests would be an important warning for this field.

## Data sources

- BMD-HS: https://github.com/mHealthBuet/BMD-HS-Dataset
- BMD-HS paper: https://arxiv.org/abs/2409.00724
- CirCor DigiScope: https://physionet.org/content/circor-heart-sound/1.0.3/
- PhysioNet/CinC 2016: https://physionet.org/content/challenge-2016/1.0.0/

The BMD-HS data are CC BY-NC-ND 4.0. Confirm the license requirements before redistributing data or
derived dataset packages. Code and numerical model outputs should be kept separate from raw data.
