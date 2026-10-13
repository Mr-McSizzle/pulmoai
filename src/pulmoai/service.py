"""Trusted local model access and application-facing research services."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from pulmoai.preprocessing import read_single_profile_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ensemble_benchmark_best.joblib"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "ensemble_benchmark" / "analysis_summary.json"
DEFAULT_VERIFICATION_PATH = PROJECT_ROOT / "reports" / "ensemble_benchmark" / "independent_verification.json"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


class PulmoAIService:
    def __init__(self, bundle: dict[str, Any], artifact_id: str):
        if not isinstance(bundle, dict) or not callable(getattr(bundle.get("pipeline"), "predict", None)):
            raise ValueError("model bundle does not contain a usable fitted prediction pipeline")
        self.bundle = bundle
        self.pipeline = bundle["pipeline"]
        self.feature_names = tuple(str(name) for name in bundle.get("feature_names", ()))
        self.label_order = tuple(str(name) for name in bundle.get("label_order", ()))
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("model bundle has an empty or duplicate feature list")
        if not self.label_order or len(set(self.label_order)) != len(self.label_order):
            raise ValueError("model bundle has an empty or duplicate class list")
        classes = np.asarray(getattr(self.pipeline, "classes_", ()))
        if classes.size != len(self.label_order) or set(classes.tolist()) != set(range(len(self.label_order))):
            raise ValueError("model classes do not match the saved label-order mapping")
        self.model_name = str(bundle.get("model_name", "Unknown model"))
        self.artifact_id = artifact_id

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> "PulmoAIService":
        if not path.is_file():
            raise FileNotFoundError(f"model artifact not found: {path}")
        bundle = joblib.load(path)
        digest = hashlib.sha256()
        with path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return cls(bundle, digest.hexdigest()[:12])

    def parse_csv_bytes(self, contents: bytes) -> np.ndarray:
        if not contents:
            raise ValueError("the uploaded CSV is empty")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise ValueError(f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit")
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv") as temporary_file:
                temporary_file.write(contents)
                temporary_file.flush()
                return read_single_profile_csv(temporary_file.name, self.feature_names)
        except UnicodeDecodeError as error:
            raise ValueError("CSV must be UTF-8 encoded") from error

    def validate_upload(self, contents: bytes) -> dict[str, Any]:
        self.parse_csv_bytes(contents)
        return {
            "valid": True,
            "profile_count": 1,
            "recognized_gene_count": len(self.feature_names),
            "required_gene_count": len(self.feature_names),
            "normalization": "raw integer-like counts → CPM → log1p",
            "message": "One complete profile matches the model's gene set and input format.",
        }

    def predict_upload(self, contents: bytes) -> dict[str, Any]:
        expression = self.parse_csv_bytes(contents)
        class_code = int(self.pipeline.predict(expression)[0])
        if class_code < 0 or class_code >= len(self.label_order):
            raise RuntimeError("model returned a class code outside its saved label mapping")
        result: dict[str, Any] = {
            "predicted_class": self.label_order[class_code],
            "model_name": self.model_name,
            "artifact_id": self.artifact_id,
            "input_profile_count": 1,
            "input_gene_count": len(self.feature_names),
            "interpretation": "Exploratory research classification only; not a diagnosis or clinical recommendation.",
        }
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(expression)[0]
            classes = getattr(self.pipeline, "classes_", None)
            if classes is not None and len(classes) == len(self.label_order):
                result["probabilities_available"] = True
                result["class_probabilities"] = {
                    self.label_order[int(cls)]: round(float(proba[i]), 6)
                    for i, cls in enumerate(classes)
                }
            else:
                result["probabilities_available"] = False
                result["class_probabilities"] = None
        else:
            result["probabilities_available"] = False
            result["class_probabilities"] = None
        return result

    def dashboard_data(self) -> dict[str, Any]:
        with DEFAULT_SUMMARY_PATH.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        verification = None
        if DEFAULT_VERIFICATION_PATH.is_file():
            with DEFAULT_VERIFICATION_PATH.open(encoding="utf-8") as stream:
                verification = json.load(stream)
        verified = bool(verification and verification.get("status") == "passed")
        validation = summary.get("validation", {})
        return {
            "verification_status": "independently_verified" if verified else "not_verified",
            "cells": int(summary.get("cells_aggregated_into_profiles", 0)),
            "studies": int(summary.get("source_study_count", 0)),
            "profiles": int(summary.get("sample_group_count", 0)),
            "features": int(summary.get("feature_count", 0)),
            "classes": list(self.label_order),
            "class_distribution": summary.get("sample_group_counts", {}),
            "model_name": self.model_name,
            "artifact_id": self.artifact_id,
            "metrics": verification.get("selected_procedure_metrics") if verified else None,
            "per_class_metrics": verification.get("per_class_metrics") if verified else None,
            "confusion_matrix": verification.get("confusion_matrix") if verified else None,
            "candidate_metrics": validation.get("fixed_candidate_heldout_study_metrics", []) if verified else [],
            "methodology": {
                "aggregation": "Cells aggregated by source study + orig.ident into sample/library profiles.",
                "normalization": "Profile counts normalized to counts per million, then log1p transformed.",
                "evaluation": "Nested selection with each entire source study held out once; candidate choice uses only training-study folds.",
                "feature_selection": "Top 500 ANOVA features selected inside each training fold.",
                "calibration": "Final model probabilities calibrated with Platt scaling (sigmoid) using 5-fold cross-validation.",
            },
            "limitations": [
                "The 372 profiles are sample/library groups, not verified independent patients; the source metadata has no confirmed patient identifier.",
                "Disease labels are unevenly distributed across source studies; performance may reflect study and batch effects.",
                "COVID19 has 19 profiles from two studies and low held-out-study recall.",
                "This exploratory model is not validated for diagnosis, treatment, or future-risk prediction.",
            ],
        }

    def gene_insights(self, limit: int = 30) -> dict[str, Any]:
        pipeline = getattr(self.pipeline, "estimator", self.pipeline)
        named_steps = getattr(pipeline, "named_steps", None)
        if named_steps is None:
            return {"available": False, "message": "This model has no compatible feature-selection metadata."}
        selector = named_steps.get("select")
        estimator = named_steps.get("classifier")
        if selector is None or estimator is None:
            return {"available": False, "message": "This model has no compatible feature-selection metadata."}
        selected_names = np.asarray(self.feature_names, dtype=object)[selector.get_support()]
        while hasattr(estimator, "named_steps"):
            estimator = estimator.steps[-1][1]
        coefficients = getattr(estimator, "coef_", None)
        classes = getattr(estimator, "classes_", None)
        if coefficients is None or classes is None:
            return {"available": False, "message": "This model does not expose linear coefficients."}
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.ndim != 2 or coefficients.shape[1] != len(selected_names):
            return {"available": False, "message": "Model coefficient dimensions do not match selected genes."}
        rows = []
        for feature_index, feature_name in enumerate(selected_names):
            class_coefficients = {
                self.label_order[int(class_code)]: float(coefficients[class_index, feature_index])
                for class_index, class_code in enumerate(classes)
                if 0 <= int(class_code) < len(self.label_order)
            }
            rows.append({
                "gene": str(feature_name),
                "class_coefficients": class_coefficients,
                "max_absolute_coefficient": max(abs(value) for value in class_coefficients.values()),
            })
        rows.sort(key=lambda row: row["max_absolute_coefficient"], reverse=True)
        return {
            "available": True,
            "model_name": self.model_name,
            "method": "Absolute linear SVM coefficients on selected standardized features",
            "interpretation": "Model weights describe classifier behavior, not differential-expression significance, biological causality, or validated biomarkers.",
            "features": rows[:max(1, min(limit, len(rows)))],
        }
