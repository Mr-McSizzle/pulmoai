#!/usr/bin/env python3
"""Independently recompute reported metrics from saved study-held-out predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "ensemble_benchmark"
METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")


def compute_metrics(truth: list[str], predictions: list[str], labels: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predictions, labels=labels, average="weighted", zero_division=0)),
    }


def verify(report_dir: Path) -> dict:
    predictions_path = report_dir / "out_of_fold_predictions.csv"
    summary_path = report_dir / "analysis_summary.json"
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    labels = [str(label) for label in summary["label_order"]]
    with predictions_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("out-of-fold predictions file is empty")
    required_columns = {"sample_group_id", "study", "true_disease", "selected_procedure_prediction"}
    missing_columns = required_columns.difference(rows[0])
    if missing_columns:
        raise ValueError(f"prediction file is missing columns: {sorted(missing_columns)}")
    if any(not row["sample_group_id"] for row in rows):
        raise ValueError("out-of-fold file contains a blank sample-group identifier")
    group_ids = [row["sample_group_id"] for row in rows]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("out-of-fold file contains duplicate sample-group identifiers")

    truth = [row["true_disease"] for row in rows]
    observed_labels = set(truth)
    if observed_labels != set(labels):
        raise ValueError(f"prediction labels differ from model label order: {sorted(observed_labels)}")
    primary_predictions = [row["selected_procedure_prediction"] for row in rows]
    if not set(primary_predictions).issubset(labels):
        raise ValueError("selected procedure contains an unknown class label")

    computed_primary = compute_metrics(truth, primary_predictions, labels)
    expected_primary = summary["validation"]["reported_selected_procedure_metrics"]
    comparisons = {
        metric: {
            "computed": computed_primary[metric],
            "recorded": float(expected_primary[metric]),
            "matches": bool(np.isclose(computed_primary[metric], expected_primary[metric], atol=1e-12, rtol=0)),
        }
        for metric in METRIC_KEYS
    }
    if not all(result["matches"] for result in comparisons.values()):
        raise ValueError("independently recomputed selected-procedure metrics do not match summary")

    candidate_checks = {}
    expected_candidates = {
        item["model"]: item for item in summary["validation"]["fixed_candidate_heldout_study_metrics"]
    }
    for model_name, expected in expected_candidates.items():
        column = f"{model_name}_prediction"
        if column not in rows[0]:
            raise ValueError(f"prediction file is missing candidate column: {column}")
        candidate_predictions = [row[column] for row in rows]
        if not set(candidate_predictions).issubset(labels):
            raise ValueError(f"candidate {model_name} contains an unknown class label")
        computed = compute_metrics(truth, candidate_predictions, labels)
        candidate_checks[model_name] = {
            metric: {
                "computed": computed[metric],
                "recorded": float(expected[metric]),
                "matches": bool(np.isclose(computed[metric], expected[metric], atol=1e-12, rtol=0)),
            }
            for metric in METRIC_KEYS
        }
        if not all(result["matches"] for result in candidate_checks[model_name].values()):
            raise ValueError(f"independently recomputed metrics do not match for {model_name}")

    study_count = len({row["study"] for row in rows})
    if study_count != int(summary["source_study_count"]):
        raise ValueError("held-out prediction study count does not match summary")
    computed_confusion = confusion_matrix(truth, primary_predictions, labels=labels).tolist()
    if computed_confusion != summary["validation"]["confusion_matrix"]:
        raise ValueError("independently recomputed confusion matrix does not match summary")
    precision, recall, class_f1, support = precision_recall_fscore_support(
        truth, primary_predictions, labels=labels, zero_division=0,
    )
    per_class_metrics = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    recorded_class_metrics = summary["validation"]["classification_report"]
    for label, metrics in per_class_metrics.items():
        recorded = recorded_class_metrics[label]
        for key in ("precision", "recall", "f1-score"):
            calculated_key = "f1" if key == "f1-score" else key
            if not np.isclose(metrics[calculated_key], recorded[key], atol=1e-12, rtol=0):
                raise ValueError(f"independently recomputed {label} {key} does not match summary")
        if metrics["support"] != int(recorded["support"]):
            raise ValueError(f"independently recomputed {label} support does not match summary")

    return {
        "status": "passed",
        "method": "metrics recomputed directly from one out-of-fold row per study-scoped profile",
        "prediction_file": str(predictions_path),
        "profile_count": len(rows),
        "study_count": study_count,
        "class_counts": {label: truth.count(label) for label in labels},
        "selected_procedure_metrics": computed_primary,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix_labels_rows_true_columns_predicted": labels,
        "confusion_matrix": computed_confusion,
        "fixed_candidate_metrics_verified": candidate_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "reports" / "ensemble_benchmark" / "independent_verification.json",
    )
    args = parser.parse_args()
    try:
        report = verify(args.report_dir)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"benchmark verification failed: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "profile_count": report["profile_count"],
        "study_count": report["study_count"],
        "selected_procedure_metrics": report["selected_procedure_metrics"],
        "candidate_models_verified": len(report["fixed_candidate_metrics_verified"]),
    }, indent=2))


if __name__ == "__main__":
    main()
