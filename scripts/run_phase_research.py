"""Run the patient-grouped, phase-aware valvular-disease research experiment."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heart_sound.features import extract_features
from heart_sound.phases import extract_phase_features

LABELS = ["AS", "AR", "MR", "MS"]
DISPLAY = {"AS": "Aortic stenosis", "AR": "Aortic regurgitation",
           "MR": "Mitral regurgitation", "MS": "Mitral stenosis"}
RELEVANT_PHASE = {"AS": "systole", "MR": "systole", "AR": "diastole", "MS": "diastole"}


def recordings(dataset: Path) -> pd.DataFrame:
    labels = pd.read_csv(dataset / "train.csv", dtype=str).replace("N/A", np.nan)
    rows = []
    for patient in labels.dropna(subset=LABELS).itertuples(index=False):
        target = {name: int(getattr(patient, name)) for name in LABELS}
        for index in range(1, 9):
            recording = getattr(patient, f"recording_{index}")
            if not isinstance(recording, str):
                continue
            path = dataset / "train" / f"{recording}.wav"
            if not path.exists():
                continue
            tokens = recording.split("_")
            rows.append({"patient_id": patient.patient_id, "recording": recording,
                         "path": str(path.resolve()), "posture": tokens[-2],
                         "site": tokens[-1], **target})
    return pd.DataFrame(rows)


def metadata_features(row: pd.Series) -> dict[str, float]:
    return {
        **{f"site_{name}": float(row.site == name) for name in ("Mit", "Tri", "Pul", "Aor")},
        **{f"posture_{name}": float(row.posture == name) for name in ("sup", "sit")},
    }


def build_feature_tables(meta: pd.DataFrame, phase_artifact: dict, cache: Path) -> dict[str, pd.DataFrame]:
    cache.mkdir(parents=True, exist_ok=True)
    paths = {name: cache / f"{name}.csv" for name in ("whole", "phase", "mask_systole", "mask_diastole")}
    if all(path.exists() for path in paths.values()):
        return {name: pd.read_csv(path) for name, path in paths.items()}
    outputs = {name: [] for name in paths}
    for number, row in meta.iterrows():
        identity = row[["patient_id", "recording", "path", "posture", "site", *LABELS]].to_dict()
        extra = metadata_features(row)
        values, names = extract_features(row.path)
        outputs["whole"].append({**identity, **extra, **dict(zip(names, values, strict=True))})
        phase, segmentation = extract_phase_features(row.path, phase_artifact)
        outputs["phase"].append({**identity, **extra, **phase})
        for phase_name in ("systole", "diastole"):
            changed, _ = extract_phase_features(
                row.path, phase_artifact, mask_phase=phase_name,
                noise_seed=42 + int(number), segmentation=segmentation,
            )
            outputs[f"mask_{phase_name}"].append({**identity, **extra, **changed})
        if (number + 1) % 50 == 0:
            print(f"features {number + 1}/{len(meta)}", flush=True)
    frames = {name: pd.DataFrame(rows) for name, rows in outputs.items()}
    for name, frame in frames.items(): frame.to_csv(paths[name], index=False)
    return frames


def feature_names(frame: pd.DataFrame) -> list[str]:
    metadata = {"patient_id", "recording", "path", "posture", "site", *LABELS}
    return [column for column in frame.columns if column not in metadata]


def constrained_names(names: list[str], label: str) -> list[str]:
    phase = RELEVANT_PHASE[label]
    allowed = ("s1_", "s2_", f"{phase}_", "segmentation_", "site_", "posture_")
    return [name for name in names if name.startswith(allowed)]


def estimator() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                class_weight="balanced", n_jobs=-1, random_state=42)


def patient_probabilities(frame: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    columns = [f"probability_{label}" for label in LABELS]
    temporary = frame[["patient_id", *LABELS]].copy()
    temporary[columns] = probabilities
    return temporary.groupby("patient_id", as_index=False).agg(
        {**{label: "first" for label in LABELS}, **{column: "mean" for column in columns}}
    )


def bootstrap_interval(y: np.ndarray, p: np.ndarray, metric, runs: int = 500) -> list[float]:
    rng = np.random.default_rng(42); values = []
    for _ in range(runs):
        take = rng.integers(0, len(y), len(y))
        if len(np.unique(y[take])) < 2:
            continue
        values.append(float(metric(y[take], p[take])))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def summarize(patient: pd.DataFrame) -> dict:
    result = {}
    for label in LABELS:
        y = patient[label].to_numpy(int)
        p = patient[f"probability_{label}"].to_numpy(float)
        result[DISPLAY[label]] = {
            "patients": int(len(y)), "positive_patients": int(np.sum(y)),
            "auroc": float(roc_auc_score(y, p)),
            "auroc_95pct_patient_bootstrap": bootstrap_interval(y, p, roc_auc_score),
            "average_precision": float(average_precision_score(y, p)),
            "average_precision_95pct_patient_bootstrap": bootstrap_interval(y, p, average_precision_score),
            "brier": float(brier_score_loss(y, p)),
        }
    return result


def choose_selective_limits(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    best = None
    for low in np.linspace(.1, .45, 36):
        for high in np.linspace(.55, .9, 36):
            accepted = (p <= low) | (p >= high)
            if np.sum(accepted) < max(10, int(.4 * len(y))):
                continue
            pred = p[accepted] >= high; truth = y[accepted]
            sensitivity = np.sum(pred & (truth == 1)) / max(np.sum(truth == 1), 1)
            specificity = np.sum(~pred & (truth == 0)) / max(np.sum(truth == 0), 1)
            rank = (min(sensitivity, specificity), float(np.mean(accepted)))
            if best is None or rank > best[0]: best = (rank, low, high)
    assert best is not None
    return {"low": float(best[1]), "high": float(best[2])}


def cross_validate(tables: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame, dict]:
    phase = tables["phase"]
    patients = phase[["patient_id", *LABELS]].drop_duplicates().reset_index(drop=True)
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    systems = ("whole_recording", "phase_unconstrained", "phase_constrained")
    oof = {system: [] for system in systems}
    counterfactual = []
    for fold, (train_patient, test_patient) in enumerate(folds.split(patients), start=1):
        train_ids = set(patients.iloc[train_patient].patient_id)
        test_ids = set(patients.iloc[test_patient].patient_id)
        fold_prob = {system: np.zeros((sum(phase.patient_id.isin(test_ids)), 4)) for system in systems}
        test_phase = phase[phase.patient_id.isin(test_ids)]
        for system, table_name in (("whole_recording", "whole"),
                                   ("phase_unconstrained", "phase"),
                                   ("phase_constrained", "phase")):
            table = tables[table_name]
            train = table[table.patient_id.isin(train_ids)]
            test = table[table.patient_id.isin(test_ids)]
            all_names = feature_names(table)
            for label_index, label in enumerate(LABELS):
                names = constrained_names(all_names, label) if system == "phase_constrained" else all_names
                model = estimator().fit(train[names].fillna(0), train[label].astype(int))
                fold_prob[system][:, label_index] = model.predict_proba(test[names].fillna(0))[:, 1]
                if system == "phase_unconstrained":
                    masked = {}
                    for changed_phase in ("systole", "diastole"):
                        changed = tables[f"mask_{changed_phase}"]
                        changed = changed[changed.patient_id.isin(test_ids)]
                        masked[changed_phase] = model.predict_proba(changed[names].fillna(0))[:, 1]
                    relevant = RELEVANT_PHASE[label]
                    other = "diastole" if relevant == "systole" else "systole"
                    for patient_id, indices in test.reset_index(drop=True).groupby("patient_id").groups.items():
                        idx = np.asarray(list(indices), dtype=int)
                        base = float(np.mean(fold_prob[system][idx, label_index]))
                        counterfactual.append({
                            "fold": fold, "patient_id": patient_id, "label": label,
                            "truth": int(test.iloc[idx[0]][label]), "base_probability": base,
                            "relevant_phase": relevant,
                            "relevant_mask_probability": float(np.mean(masked[relevant][idx])),
                            "irrelevant_mask_probability": float(np.mean(masked[other][idx])),
                        })
        for system in systems:
            oof[system].append(patient_probabilities(test_phase, fold_prob[system]))
    combined = {name: pd.concat(parts, ignore_index=True).sort_values("patient_id")
                for name, parts in oof.items()}
    results = {name: summarize(frame) for name, frame in combined.items()}
    predictions = combined["phase_constrained"].copy()
    predictions["fold_protocol"] = "5-fold patient-grouped out-of-fold"
    limits = {}
    for label in LABELS:
        limits[label] = choose_selective_limits(predictions[label].to_numpy(int),
                                                predictions[f"probability_{label}"].to_numpy(float))
    cf = pd.DataFrame(counterfactual)
    counter_summary = {}
    for label in LABELS:
        selected = cf[(cf.label == label) & (cf.truth == 1)]
        counter_summary[DISPLAY[label]] = {
            "positive_patients": int(len(selected)),
            "mean_probability_drop_masking_expected_phase": float(np.mean(
                selected.base_probability - selected.relevant_mask_probability)),
            "mean_probability_drop_masking_other_phase": float(np.mean(
                selected.base_probability - selected.irrelevant_mask_probability)),
        }
    return {"models": results, "counterfactual_phase_test": counter_summary}, predictions, {
        "limits": limits, "counterfactual_rows": cf
    }


def fit_final(phase: pd.DataFrame, phase_artifact: dict, limits: dict) -> dict:
    all_names = feature_names(phase); models = {}; names_by_label = {}
    for label in LABELS:
        names = constrained_names(all_names, label)
        models[label] = estimator().fit(phase[names].fillna(0), phase[label].astype(int))
        names_by_label[label] = names
    return {"version": "2.0.0", "type": "phase_aware_valvular_multilabel_research",
            "labels": LABELS, "display_labels": DISPLAY, "relevant_phase": RELEVANT_PHASE,
            "models": models, "feature_names_by_label": names_by_label,
            "selective_limits": limits, "phase_artifact": phase_artifact,
            "warning": "Research-only screening estimate; not a diagnosis. Echocardiography is required."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "work" / "data" / "bmd_hs")
    parser.add_argument("--phase-model", type=Path, default=ROOT / "outputs" / "phase_model" / "phase_segmenter.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "phase_disease_model")
    parser.add_argument("--cache", type=Path, default=ROOT / "work" / "phase_research_features")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    phase_artifact = joblib.load(args.phase_model)
    meta = recordings(args.dataset)
    tables = build_feature_tables(meta, phase_artifact, args.cache)
    results, predictions, details = cross_validate(tables)
    results.update({
        "dataset": "BMD-HS", "records": int(len(meta)),
        "patients": int(meta.patient_id.nunique()),
        "evaluation": "Five-fold patient-grouped out-of-fold evaluation; eight recordings aggregated per patient.",
        "segmentation_source": phase_artifact["metrics"],
        "limitations": [
            "Small, single-site disease dataset.",
            "BMD cardiac phases are inferred by a segmenter transferred from pediatric CirCor data.",
            "No external disease-labelled cohort was available for independent validation.",
        ],
    })
    predictions.to_csv(args.output / "patient_oof_predictions.csv", index=False)
    details["counterfactual_rows"].to_csv(args.output / "counterfactual_phase_predictions.csv", index=False)
    artifact = fit_final(tables["phase"], phase_artifact, details["limits"])
    joblib.dump(artifact, args.output / "phase_valve_disease_model.joblib", compress=3)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
