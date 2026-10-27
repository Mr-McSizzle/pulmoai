"""Create audited sex-specific candidate models from explicitly annotated local counts.

Input: NPZ with counts (profiles x genes), feature_names and sample_ids.
Metadata CSV: sample_id, patient_id, study, reported_sex, disease.
Outputs are research candidates, never downloaded or automatically promoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pulmoai.service import PulmoAIService


def train(counts_path: Path, metadata_path: Path, output: Path) -> dict:
    pooled = PulmoAIService.load()
    metadata = pd.read_csv(metadata_path, dtype=str)
    required = {"sample_id", "patient_id", "study", "reported_sex", "disease"}
    if not required.issubset(metadata.columns) or metadata[list(required)].isna().any().any():
        raise ValueError("Complete verified sample, patient, study, sex and disease metadata is required.")
    if metadata.sample_id.duplicated().any() or metadata.patient_id.duplicated().any():
        raise ValueError("Use one profile per verified patient to prevent patient leakage.")
    if not metadata.reported_sex.isin(["female", "male", "intersex", "unknown"]).all():
        raise ValueError("Use explicit reported-sex labels; never infer them from expression.")
    if not metadata.disease.isin(pooled.label_order).all():
        raise ValueError("Disease labels do not match the deployed label order.")
    with np.load(counts_path, allow_pickle=False) as data:
        counts = np.asarray(data["counts"], dtype=np.float64)
        features = tuple(data["feature_names"].astype(str))
        ids = data["sample_ids"].astype(str)
    if features != pooled.feature_names or counts.shape != (len(ids), len(features)):
        raise ValueError("Counts must match the deployed complete gene set and sample IDs.")
    if len(set(ids)) != len(ids) or set(ids) != set(metadata.sample_id):
        raise ValueError("Count and metadata sample IDs must match exactly, without duplicates.")
    metadata = metadata.set_index("sample_id").loc[ids].reset_index()
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.allclose(counts, np.rint(counts), atol=1e-5, rtol=0):
        raise ValueError("Expected finite, nonnegative integer raw counts.")
    totals = counts.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("Empty profiles are not allowed.")
    X = np.log1p(counts / totals * 1e6).astype(np.float32)
    y = np.array([pooled.label_order.index(label) for label in metadata.disease])
    output.mkdir(parents=True, exist_ok=True)
    registry = {"schema_version": 1, "research_only": True, "models": {},
                "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                "counts_sha256": hashlib.sha256(counts_path.read_bytes()).hexdigest()}
    estimator = getattr(pooled.pipeline, "base_pipeline", pooled.pipeline)
    for sex in ["female", "male"]:
        mask = metadata.reported_sex.eq(sex).to_numpy()
        sx, sy = X[mask], y[mask]
        groups = metadata.study.to_numpy()[mask]
        support = {label: int((sy == code).sum()) for code, label in enumerate(pooled.label_order)}
        if len(set(groups)) < 3 or min(support.values()) < 10:
            raise ValueError(f"Insufficient {sex} data: need 3 studies and 10 profiles per class.")
        predictions = np.empty_like(sy)
        for train_idx, test_idx in LeaveOneGroupOut().split(sx, sy, groups):
            if len(set(sy[train_idx])) != len(pooled.label_order):
                raise ValueError(f"{sex}: a held-out-study fold leaves a class absent from training.")
            candidate = clone(estimator).fit(sx[train_idx], sy[train_idx])
            predictions[test_idx] = candidate.predict(sx[test_idx])
        evaluation = {"protocol": "leave_one_study_out", "source_studies": len(set(groups)),
                      "class_support": support, "macro_f1": float(f1_score(sy, predictions, average="macro", zero_division=0)),
                      "balanced_accuracy": float(balanced_accuracy_score(sy, predictions)),
                      "clinically_validated": False}
        fitted = clone(estimator).fit(sx, sy)
        bundle = {"pipeline": fitted, "feature_names": pooled.feature_names, "label_order": pooled.label_order,
                  "model_name": f"Reported-{sex} research Linear SVM", "evaluation": evaluation}
        path = output / f"lung_{sex}.joblib"
        joblib.dump(bundle, path)
        registry["models"][sex] = {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "evaluation": evaluation}
        pd.DataFrame({"sample_id": metadata.sample_id.to_numpy()[mask], "study": groups,
                      "true_class": sy, "predicted_class": predictions}).to_csv(output / f"{sex}_oof.csv", index=False)
    (output / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    return registry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "sex_candidates")
    args = parser.parse_args()
    print(json.dumps(train(args.counts, args.metadata, args.output), indent=2))
