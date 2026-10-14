#!/usr/bin/env python3
"""Train and export the reproducible Dual-Rule Lung Model artifact.

Exact hyperparameters:
- Training Data File: data/processed/allcells_pseudobulk_counts.npz
- k: 500 features selected via ANOVA F-value (f_classif)
- Normalization: counts per million (CPM) -> log1p
- Preprocessing: StandardScaler()
- Base Estimator: LinearSVC(C=0.1, class_weight='balanced', max_iter=10000, random_state=42)
- Probability Calibrator: CalibratedClassifierCV(method='sigmoid', cv=5)
- Decision Rule: Dual-Rule (uncalibrated decision function argmax for discrete labels,
                 5-fold CV Platt sigmoid scaling for probability estimates)
- Output Artifact: models/lung_uncalibrated_dual_rule.joblib
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from pulmoai.service import DualRuleLungModel, PulmoAIService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_dual_rule")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "allcells_pseudobulk_counts.npz"
ARTIFACT_PATH = PROJECT_ROOT / "models" / "lung_uncalibrated_dual_rule.joblib"
LABELS = ("Donor", "COPD", "PF", "COVID19")

# Fixed Hyperparameters
HYPERPARAMS = {
    "k_features": 500,
    "feature_selection_metric": "f_classif",
    "C": 0.1,
    "class_weight": "balanced",
    "max_iter": 10000,
    "random_state": 42,
    "calibration_method": "sigmoid",
    "calibration_cv": 5,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def normalize_cpm_log1p(counts: np.ndarray) -> np.ndarray:
    lib_size = counts.sum(axis=1, keepdims=True)
    lib_size = np.where(lib_size == 0, 1.0, lib_size)
    cpm = (counts / lib_size) * 1e6
    return np.log1p(cpm)


def train_and_export() -> dict[str, str]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at: {DATA_PATH}")

    data_hash = sha256_file(DATA_PATH)
    logger.info("Training data: %s", DATA_PATH)
    logger.info("Training data SHA-256: %s", data_hash)

    with np.load(DATA_PATH, allow_pickle=False) as arch:
        counts = arch["counts"]
        labels = arch["labels"].astype(str)
        feature_names = arch["feature_names"].astype(str).tolist()
        sample_ids = arch["sample_ids"].astype(str)
        studies = arch["studies"].astype(str)

    label_to_code = {lbl: idx for idx, lbl in enumerate(LABELS)}
    y = np.array([label_to_code[lbl] for lbl in labels], dtype=np.int8)
    expr = normalize_cpm_log1p(counts).astype(np.float32)

    logger.info("Loaded %d profiles (%d genes) across %d studies", expr.shape[0], expr.shape[1], len(np.unique(studies)))

    # Construct and fit base pipeline
    base_selector = SelectKBest(f_classif, k=HYPERPARAMS["k_features"])
    base_svc = LinearSVC(
        C=HYPERPARAMS["C"],
        class_weight=HYPERPARAMS["class_weight"],
        max_iter=HYPERPARAMS["max_iter"],
        random_state=HYPERPARAMS["random_state"],
    )
    base_pipeline = Pipeline([
        ("select", base_selector),
        ("classifier", Pipeline([
            ("scale", StandardScaler()),
            ("classifier", base_svc),
        ])),
    ])

    logger.info("Fitting base pipeline on all %d profiles...", len(y))
    base_pipeline.fit(expr, y)

    # Construct and fit 5-fold CV Platt calibrator
    logger.info("Fitting 5-fold cross-validated Platt sigmoid calibrator...")
    calibrator = CalibratedClassifierCV(
        base_pipeline,
        method=HYPERPARAMS["calibration_method"],
        cv=HYPERPARAMS["calibration_cv"],
    )
    calibrator.fit(expr, y)

    # Wrap in DualRuleLungModel
    dual_model = DualRuleLungModel(base_pipeline, calibrator, list(LABELS))

    # Save artifact
    logger.info("Saving dual-rule model to: %s", ARTIFACT_PATH)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": dual_model,
        "base_pipeline": base_pipeline,
        "calibrated_pipeline": calibrator,
        "model_name": "Linear SVM (Dual-Rule Decision + Calibrated Probs)",
        "feature_names": feature_names,
        "label_order": list(LABELS),
        "normalization": "sample-level summed counts -> counts per million -> log1p",
        "sample_grouping": "gse_alias + orig.ident",
        "validation": "nested leave-one-study-out outer evaluation",
        "model_input_contract": "one pseudobulk count row in feature_names order; normalize before pipeline.predict",
        "calibration": "Dual-rule: uncalibrated argmax decision for discrete prediction, Platt scaling 5-fold CV for probabilities",
        "hyperparameters": HYPERPARAMS,
        "training_data_sha256": data_hash,
    }
    joblib.dump(bundle, ARTIFACT_PATH, compress=3)

    artifact_hash = sha256_file(ARTIFACT_PATH)
    logger.info("Saved dual-rule artifact SHA-256: %s", artifact_hash)

    # Verification: Load with PulmoAIService in a fresh instance
    logger.info("Verifying artifact loading with PulmoAIService...")
    loaded_bundle = joblib.load(ARTIFACT_PATH)
    service = PulmoAIService(loaded_bundle, artifact_id=artifact_hash[:16])

    # Smoke test validation and prediction
    dummy_csv = "gene,count\n" + "\n".join(f"{g},10" for g in feature_names[:10]) + "\n"
    # Provide complete profile for first sample
    sample_0_csv = "gene,count\n" + "\n".join(f"{g},{int(counts[0, i])}" for i, g in enumerate(feature_names)) + "\n"
    pred_res = service.predict_upload(sample_0_csv.encode("utf-8"))
    logger.info("Smoke test prediction successful: predicted=%s, probabilities=%s", pred_res["predicted_class"], pred_res["probabilities_available"])

    results = {
        "training_data_path": str(DATA_PATH),
        "training_data_sha256": data_hash,
        "artifact_path": str(ARTIFACT_PATH),
        "artifact_sha256": artifact_hash,
        "hyperparameters": HYPERPARAMS,
        "verified_loading": True,
        "sample_0_prediction": pred_res["predicted_class"],
    }
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    train_and_export()
