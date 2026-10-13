# Phase-aware valvular-disease experiment

## Status

Implemented and executed on September 24, 2026. This is a research feasibility experiment, not
clinical validation and not a diagnostic device.

## Data and protocol

- Disease data: all 864 usable BMD-HS recordings from 108 labelled patients.
- Disease targets: AS, AR, MR, and MS as independent labels.
- Phase-transfer data: a reproducible subset of 398 CirCor recordings with paired expert-reviewed
  S1/systole/S2/diastole annotations.
- Evaluation: five-fold out-of-fold evaluation grouped strictly by patient.
- Patient prediction: mean probability across the patient's eight site/posture recordings.
- Confidence intervals: 500 patient-level bootstrap samples.
- Comparisons: whole-recording features, unconstrained phase features, and physiologically
  constrained phase features.
- Counterfactual audit: replace inferred systole or diastole with low-amplitude matched noise and
  measure the change in disease probability.

No recording from an evaluation patient is used to train that fold's disease model.

## Phase-segmentation transfer

The compact CirCor phase model used 63,373 training frames and 16,627 frames from held-out
subjects. Its pre-decoding balanced frame accuracy was 0.548. This is modest, so BMD-HS phase
boundaries must be described as inferred and noisy. Cyclic Viterbi decoding enforces the cardiac
order S1 -> systole -> S2 -> diastole, but it does not turn predictions into ground truth.

## Disease results

### Patient-level AUROC

| Disease | Whole recording | Phase unconstrained | Phase constrained |
|---|---:|---:|---:|
| Aortic stenosis | 0.908 | 0.928 | 0.927 |
| Aortic regurgitation | 0.747 | 0.787 | 0.794 |
| Mitral regurgitation | 0.715 | 0.694 | 0.698 |
| Mitral stenosis | 0.693 | 0.677 | 0.683 |
| **Macro average** | **0.766** | **0.772** | **0.775** |

### Patient-level average precision

| Disease | Whole recording | Phase unconstrained | Phase constrained |
|---|---:|---:|---:|
| Aortic stenosis | 0.828 | 0.881 | 0.882 |
| Aortic regurgitation | 0.624 | 0.701 | 0.710 |
| Mitral regurgitation | 0.520 | 0.489 | 0.508 |
| Mitral stenosis | 0.554 | 0.510 | 0.516 |
| **Macro average** | **0.632** | **0.645** | **0.654** |

Macro Brier score improved from 0.199 for whole-recording features to 0.192 for the constrained
phase model. Lower is better.

## Counterfactual phase audit

This audit was performed on the unconstrained phase model so the answer was not predetermined by
hard feature selection.

| Disease | Expected phase | Probability drop when expected phase masked | Drop when other phase masked |
|---|---|---:|---:|
| Aortic stenosis | Systole | 0.129 | 0.068 |
| Aortic regurgitation | Diastole | 0.069 | 0.071 |
| Mitral regurgitation | Systole | 0.126 | 0.100 |
| Mitral stenosis | Diastole | 0.028 | 0.120 |

## Interpretation

The experiment supports only part of the original hypothesis:

- Phase-aware features modestly improved macro discrimination, calibration, AS, and AR.
- AS showed the clearest physiologically expected dependence on systole.
- MR also depended more on systole, although the separation was smaller.
- AR did not distinguish expected from non-expected phase masking.
- MS behaved contrary to the simple physiological hypothesis: masking systole changed the model
  more than masking diastole.

Therefore, the current system must not claim that all disease estimates are based on the expected
murmur phase. The AR/MS findings are evidence of unresolved segmentation error, correlated
non-murmur features, acquisition shortcuts, or oversimplified textbook constraints. This negative
result strengthens the research motivation rather than supporting deployment.

## Artifacts

- `outputs/phase_model/phase_segmenter.joblib`: compact transferred phase model.
- `outputs/phase_model/metrics.json`: held-out CirCor frame results.
- `outputs/phase_disease_model/phase_valve_disease_model.joblib`: production research artifact.
- `outputs/phase_disease_model/metrics.json`: complete metrics and confidence intervals.
- `outputs/phase_disease_model/patient_oof_predictions.csv`: one out-of-fold row per patient.
- `outputs/phase_disease_model/counterfactual_phase_predictions.csv`: patient-level masking audit.
- `work/phase_research_features/*.csv`: cached feature and counterfactual tables.
- `work/data/circor_subset/subset_manifest.json`: exact reproducible CirCor subset.

## Next experiment required for publication

The next iteration should improve phase transfer and directly test acquisition shortcuts:

1. train a temporal segmenter rather than an independent-frame classifier;
2. manually annotate a small blinded BMD-HS subset for S1/S2 transfer evaluation;
3. perform background replacement using recordings from different acquisition groups;
4. compare models with site/posture metadata removed;
5. obtain an independent disease-labelled cohort for external validation;
6. preregister thresholds and the counterfactual hypotheses before the final evaluation.

The present results establish the pipeline and reveal the central failure mode. They do not close
the research question.
