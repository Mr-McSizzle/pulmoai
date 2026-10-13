#!/usr/bin/env python3
"""Compare exploratory donor-level classifiers on the labeled COVID-HC pilot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tarfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from train_baseline import (
    MATRIX_PREFIXES,
    aggregate_library,
    load_donor_metadata,
    open_text_member,
    target_for_disease,
)


LABELS = ("healthy", "COVID")


def make_pipeline(name: str, feature_count: int, top_features: int) -> Pipeline:
    steps = [
        ("remove_constant", VarianceThreshold(threshold=0.0)),
        ("select", SelectKBest(score_func=f_classif, k=min(top_features, feature_count))),
    ]
    if name in {"logistic_regression_scaled", "linear_svm", "rbf_svm"}:
        steps.append(("scale", StandardScaler()))
    if name == "logistic_regression":
        estimator = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=42,
        )
    elif name == "logistic_regression_scaled":
        estimator = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=42,
        )
    elif name == "linear_svm":
        estimator = LinearSVC(
            C=0.1,
            class_weight="balanced",
            dual="auto",
            max_iter=5000,
            random_state=42,
        )
    elif name == "rbf_svm":
        estimator = SVC(C=1.0, class_weight="balanced", gamma="scale", kernel="rbf")
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            class_weight="balanced",
            max_features="sqrt",
            n_estimators=400,
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"unknown model: {name}")
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def load_pilot(archive_path: Path):
    with tarfile.open(archive_path, mode="r:*") as archive:
        matrices = sorted(
            (
                member
                for member in archive.getmembers()
                if member.name.endswith("_matrix.mtx.gz")
                and Path(member.name).name.startswith(MATRIX_PREFIXES)
            ),
            key=lambda member: member.name,
        )
        if len(matrices) != 2:
            raise ValueError(f"expected two COVID-HC matrices, found {len(matrices)}")

        donor_ids = set()
        for member in archive.getmembers():
            if (
                member.name.endswith("_CMO.csv.gz")
                and Path(member.name).name.startswith(("GSM8335814_", "GSM8335816_"))
            ):
                with open_text_member(archive, member.name) as stream:
                    donor_ids.update(
                        row["sample_id"]
                        for row in csv.DictReader(stream)
                        if row.get("sample_id") and row.get("disease")
                    )
        donor_ids = sorted(donor_ids)
        if len(donor_ids) != 8:
            raise ValueError(f"expected eight labeled donors, found {len(donor_ids)}")
        donor_index = {donor: index for index, donor in enumerate(donor_ids)}

        donor_labels = {}
        cell_counts = Counter()
        combined_counts = None
        feature_ids = None
        matrix_stats = []
        for matrix_member in matrices:
            token = Path(matrix_member.name.removesuffix("_matrix.mtx.gz")).name.split("_", 1)[1]
            _, metadata = load_donor_metadata(archive, token)
            for row in metadata.values():
                if row.get("sample_id") and row.get("disease"):
                    disease = row["disease"]
                    label = target_for_disease(disease)
                    value = (disease, label)
                    previous = donor_labels.get(row["sample_id"])
                    if previous and previous != value:
                        raise ValueError(f"conflicting disease labels for {row['sample_id']}")
                    donor_labels[row["sample_id"]] = value
            features, counts, _, library_cell_counts, stats = aggregate_library(
                archive,
                matrix_member,
                donor_index,
                len(donor_ids),
                feature_ids,
            )
            feature_ids = features
            combined_counts = counts if combined_counts is None else combined_counts + counts
            cell_counts.update(library_cell_counts)
            matrix_stats.append(stats)

    if set(donor_labels) != set(donor_ids):
        raise ValueError("CMO donor IDs and disease labels do not match")
    y = np.asarray([LABELS.index(donor_labels[donor][1]) for donor in donor_ids], dtype=np.int32)
    class_counts = Counter(donor_labels[donor][1] for donor in donor_ids)
    if set(class_counts) != set(LABELS) or min(class_counts.values()) < 2:
        raise ValueError(f"both classes require at least two donors: {dict(class_counts)}")
    library_sizes = combined_counts.sum(axis=0)
    if np.any(library_sizes <= 0):
        raise ValueError("at least one donor has no assigned expression counts")
    expression = np.log1p((combined_counts / library_sizes[np.newaxis, :]) * 1_000_000).T
    return donor_ids, donor_labels, y, expression, feature_ids, cell_counts, class_counts, matrix_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to GSE248688_RAW.tar")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/model_comparison"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--top-features", type=int, default=100)
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"archive does not exist or is not a file: {args.archive}")
    if args.top_features < 1:
        parser.error("top-features must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    donor_ids, donor_labels, y, expression, feature_ids, cell_counts, class_counts, matrix_stats = load_pilot(args.archive)
    model_names = (
        "logistic_regression",
        "logistic_regression_scaled",
        "linear_svm",
        "rbf_svm",
        "random_forest",
    )
    results = []
    prediction_rows = []
    splitter = LeaveOneOut()
    for model_name in model_names:
        predictions = np.empty(len(donor_ids), dtype=np.int32)
        for train_indices, test_indices in splitter.split(expression, y):
            model = make_pipeline(model_name, expression.shape[1], args.top_features)
            model.fit(expression[train_indices], y[train_indices])
            predictions[test_indices] = model.predict(expression[test_indices])
        matrix = confusion_matrix(y, predictions, labels=np.arange(len(LABELS)))
        result = {
            "model": model_name,
            "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
            "accuracy": float(accuracy_score(y, predictions)),
            "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, predictions, average="weighted", zero_division=0)),
            "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(
                y, predictions, labels=np.arange(len(LABELS)), target_names=LABELS,
                output_dict=True, zero_division=0,
            ),
        }
        results.append(result)
        for index, donor in enumerate(donor_ids):
            prediction_rows.append(
                {
                    "model": model_name,
                    "donor_id": donor,
                    "true_disease": donor_labels[donor][0],
                    "true_label": donor_labels[donor][1],
                    "predicted_label": LABELS[int(predictions[index])],
                    "correct": bool(predictions[index] == y[index]),
                }
            )

    preference = {name: index for index, name in enumerate(model_names)}
    best_result = max(
        results,
        key=lambda result: (
            result["balanced_accuracy"],
            result["macro_f1"],
            -preference[result["model"]],
        ),
    )
    best_model = make_pipeline(best_result["model"], expression.shape[1], args.top_features)
    best_model.fit(expression, y)
    model_path = args.model_dir / f"pilot_best_{best_result['model']}.joblib"
    joblib.dump(
        {
            "pipeline": best_model,
            "feature_ids": feature_ids,
            "donor_ids_used_for_final_fit": donor_ids,
            "class_labels": list(LABELS),
            "normalization": "sum counts by donor across libraries; CPM then log1p",
            "selected_model": best_result["model"],
            "selection_cv": "leave-one-donor-out; exploratory only",
        },
        model_path,
        compress=3,
    )

    report = {
        "task": "exploratory donor-level COVID-versus-healthy classification",
        "dataset": str(args.archive),
        "donor_count": len(donor_ids),
        "class_counts": dict(class_counts),
        "donor_cell_counts": dict(sorted(cell_counts.items())),
        "source_matrices": matrix_stats,
        "preprocessing": "sum assigned raw counts by donor across both libraries; normalize each donor to counts per million; apply log1p",
        "feature_selection": f"fold-local ANOVA SelectKBest; k={args.top_features}; constant features removed",
        "cv": "leave-one-donor-out; every donor predicted once; no hyperparameter tuning",
        "model_comparison": results,
        "selected_model": best_result["model"],
        "selection_rule": "highest LOOCV balanced accuracy, then macro-F1; ties favor the simpler earlier-listed model",
        "selected_model_metrics_are_model_selection_cv_not_independent_test": True,
        "model_file": str(model_path),
        "limitations": [
            "Only eight independent donors are labeled in the locally usable cell-level data.",
            "LOOCV metrics are extremely unstable and are not an independent final test.",
            "The local GEO archive contains additional pooled libraries without cell-to-donor assignments; those were excluded rather than given speculative labels.",
            "This is a COVID-versus-healthy pilot, not the full multi-disease atlas model.",
            "No external cohort validation or clinical performance claim is supported.",
        ],
    }
    (args.output_dir / "algorithm_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "out_of_fold_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    names = [result["model"] for result in results]
    display_names = {
        "logistic_regression": "L2 Logistic",
        "logistic_regression_scaled": "Scaled Logistic",
        "linear_svm": "Linear SVM",
        "rbf_svm": "RBF SVM",
        "random_forest": "Random Forest",
    }
    scores = [result["balanced_accuracy"] for result in results]
    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(
        [display_names[name] for name in names],
        scores,
        color=["#2563eb" if name == best_result["model"] else "#94a3b8" for name in names],
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Leave-one-donor-out balanced accuracy")
    axis.set_title("Exploratory model comparison — 8 donors")
    axis.tick_params(axis="x", rotation=15)
    for bar, score in zip(bars, scores):
        axis.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.2f}", ha="center")
    figure.tight_layout()
    figure.savefig(args.output_dir / "model_comparison.png", dpi=160)
    plt.close(figure)

    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(best_result["confusion_matrix"]),
        display_labels=[label.title() for label in LABELS],
    )
    display.plot(cmap="Blues", colorbar=False, values_format="d")
    plt.title(f"{best_result['model']} — LOOCV, 8 donors")
    plt.tight_layout()
    plt.savefig(args.output_dir / "selected_model_confusion_matrix.png", dpi=160)
    plt.close()

    print(f"Donors: {len(donor_ids)} ({dict(class_counts)})")
    for result in results:
        print(
            f"{result['model']}: balanced_accuracy={result['balanced_accuracy']:.3f}, "
            f"macro_f1={result['macro_f1']:.3f}, weighted_f1={result['weighted_f1']:.3f}"
        )
    print(f"Selected exploratory model: {best_result['model']}")
    print(f"Model saved: {model_path}")
    print(f"Report saved: {args.output_dir / 'algorithm_comparison.json'}")


if __name__ == "__main__":
    main()
