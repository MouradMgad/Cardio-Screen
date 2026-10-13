# CardioScreen — AI-Based Heart-Sound Screening

**A machine learning research project for classifying normal and abnormal heart sounds from digital stethoscope recordings.**

CardioScreen is a research prototype built using the [PhysioNet/CinC 2016 heart-sound dataset](https://physionet.org/content/challenge-2016/1.0.0/). It combines audio signal processing, feature engineering, and ensemble learning to analyze phonocardiogram (PCG) recordings and identify recordings that may require expert review.

The project includes a trained classification model, a recording-upload interface, and a reproducible training pipeline.

> **Research use only.** CardioScreen is not a medical device and must not be used to diagnose medical conditions.

## Key Results

Evaluation on 486 held-out recordings:

| Metric | Result |
|---|---:|
| Accuracy | 93.6% |
| AUROC | 0.975 |
| Sensitivity | 92.9% |
| Specificity | 93.8% |
| Precision | 79.3% |

### Confidence-Aware Screening

Instead of forcing a classification for every recording, CardioScreen uses probability thresholds to produce three outcomes:

- **Likely normal:** Abnormality probability ≤ 0.215
- **Uncertain:** Probability between 0.215 and 0.825; further review or a new recording is recommended
- **Abnormal:** Abnormality probability ≥ 0.825; expert review is advised

On decisive predictions, the policy achieved **92.1% precision, 97.2% recall, and 78.6% coverage**. When uncertain and abnormal results both trigger referral, the reported screening sensitivity was 99.0%.

These results describe recording-level benchmark performance, not clinical effectiveness.

## How It Works

1. **Signal preprocessing** — Filters and normalizes heart-sound recordings.
2. **Feature extraction** — Computes 301 audio features, including MFCCs, spectral characteristics, wavelet-based features, and temporal statistics.
3. **Machine learning** — Combines Extra Trees and LightGBM models to estimate the probability of an abnormal recording.
4. **Confidence-based classification** — Uses validation-selected thresholds to distinguish likely normal, uncertain, and abnormal recordings.
5. **Interactive application** — Allows users to upload WAV recordings and inspect model predictions.

## Acoustic Pattern Exploration

CardioScreen also includes an experimental second stage that assigns descriptive acoustic patterns to abnormal recordings:

- **Murmur-like:** Dominated by broad, noise-like frequency energy.
- **Rhythm/timing-like:** Dominated by irregular cycle timing.

In a validation set of 99 recordings with abnormal labels, the stage showed 97.7% mean agreement across bootstrap recalibrations and 94.9% mean agreement under small feature perturbations.

These measurements reflect prediction stability, **not medical accuracy**. The patterns are exploratory acoustic descriptions, not diagnoses.

## Run the Application

From PowerShell, navigate to the project directory and run:

```powershell
.\run_app.ps1
```

Then open:

http://127.0.0.1:7860

Upload a WAV recording to test the trained model. The existing model artifacts are stored in `outputs/model_final`, so retraining is not required to run the application.

If the required Python environment is unavailable, run:

```powershell
.\setup.ps1
```

## Reproduce the Results

1. Download the training data from the [official PhysioNet/CinC 2016 dataset](https://physionet.org/content/challenge-2016/1.0.0/).
2. Extract the data into `work/data/training`, preserving the dataset folders and reference files.
3. Install the dependencies using `.\setup.ps1`, if necessary.
4. Run the training pipeline:

   ```powershell
   .\run_pipeline.ps1
   ```

5. To rebuild the feature cache from the WAV recordings, run:

   ```powershell
   .\run_pipeline.ps1 -ReextractFeatures
   ```

The pipeline saves extracted features and model artifacts under the project directories, allowing training to resume after interrupted feature extraction.

## Limitations

- The evaluation uses recording-level splits; reliable patient identifiers are not available across all source databases. Performance may therefore overestimate generalization to unseen patients.
- The secondary acoustic-pattern classifier has not been validated against reliable clinical pattern labels.
- Independent external validation and patient-level evaluation are needed before considering any clinical application.

CardioScreen is intended for research and experimentation in biomedical signal processing and machine learning.

## Dataset Reference

Liu C, Springer D, Li Q, et al. *An open access database for the evaluation of heart sound algorithms.* Physiological Measurement. 2016;37(12):2181–2213.

Dataset: [PhysioNet/CinC Challenge 2016](https://physionet.org/content/challenge-2016/1.0.0/)

**Author:** Mourad Mohamed Gad
