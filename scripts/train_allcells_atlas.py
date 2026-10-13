#!/usr/bin/env python3
"""Train and evaluate a study-held-out lung disease classifier from an H5AD atlas."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import anndata as ad
import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ("Donor", "COPD", "PF", "COVID19")
MODEL_NAMES = ("Logistic Regression", "Linear SVM", "RBF SVM", "Extra Trees")
REQUIRED_OBS = ("orig.ident", "Disease", "gse_alias", "celltypeuse")
TOP_FEATURES = 500


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def make_pipeline(model_name: str, feature_count: int, seed: int) -> Pipeline:
    steps = [("select", SelectKBest(score_func=f_classif, k=min(TOP_FEATURES, feature_count)))]
    if model_name == "Logistic Regression":
        steps.extend([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=seed
            )),
        ])
    elif model_name == "Linear SVM":
        steps.extend([
            ("scale", StandardScaler()),
            ("classifier", LinearSVC(
                C=0.1, class_weight="balanced", max_iter=10000, random_state=seed
            )),
        ])
    elif model_name == "RBF SVM":
        steps.extend([
            ("scale", StandardScaler()),
            ("classifier", SVC(C=1.0, kernel="rbf", class_weight="balanced")),
        ])
    elif model_name == "Extra Trees":
        steps.append(("classifier", ExtraTreesClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=2,
        )))
    else:
        raise ValueError(f"unknown model: {model_name}")
    return Pipeline(steps)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_sample_index(adata: ad.AnnData):
    obs = adata.obs
    missing = [column for column in REQUIRED_OBS if column not in obs.columns]
    if missing:
        raise ValueError(f"required .obs metadata columns are absent: {missing}")

    orig = pd.Categorical(obs["orig.ident"])
    study = pd.Categorical(obs["gse_alias"])
    disease = pd.Categorical(obs["Disease"])
    celltype = pd.Categorical(obs["celltypeuse"])
    if (orig.codes < 0).any() or (study.codes < 0).any() or (disease.codes < 0).any():
        raise ValueError("orig.ident, gse_alias, and Disease must not contain missing values")

    disease_categories = [str(value) for value in disease.categories]
    unexpected = sorted(set(disease_categories) - set(LABELS))
    if unexpected or set(disease_categories) != set(LABELS):
        raise ValueError(f"expected exactly {list(LABELS)} disease labels; found {disease_categories}")
    label_index = {label: index for index, label in enumerate(LABELS)}
    disease_lookup = np.asarray([label_index[str(value)] for value in disease.categories], dtype=np.int8)
    cell_disease_codes = disease_lookup[disease.codes]

    orig_count = len(orig.categories)
    composite = study.codes.astype(np.int64) * orig_count + orig.codes.astype(np.int64)
    unique_composite, cell_sample_index = np.unique(composite, return_inverse=True)
    sample_study_codes = (unique_composite // orig_count).astype(np.int32)
    sample_orig_codes = (unique_composite % orig_count).astype(np.int32)
    n_samples = len(unique_composite)

    disease_by_sample = np.bincount(
        cell_sample_index.astype(np.int64) * len(LABELS) + cell_disease_codes,
        minlength=n_samples * len(LABELS),
    ).reshape(n_samples, len(LABELS))
    if np.any((disease_by_sample > 0).sum(axis=1) != 1):
        bad = np.flatnonzero((disease_by_sample > 0).sum(axis=1) != 1)[:10].tolist()
        raise ValueError(f"a source sample maps to inconsistent disease labels: {bad}")
    sample_label_codes = disease_by_sample.argmax(axis=1).astype(np.int8)

    study_names = np.asarray([str(value) for value in study.categories], dtype=object)
    orig_names = np.asarray([str(value) for value in orig.categories], dtype=object)
    sample_studies = study_names[sample_study_codes]
    sample_orig = orig_names[sample_orig_codes]
    sample_ids = np.asarray(
        [f"{study_id}::{orig_id}" for study_id, orig_id in zip(sample_studies, sample_orig)],
        dtype=object,
    )
    cell_counts = np.bincount(cell_sample_index, minlength=n_samples).astype(np.int64)
    celltype_codes = celltype.codes
    composition_codes = cell_sample_index.astype(np.int64) * max(1, len(celltype.categories)) + np.maximum(celltype_codes, 0)
    composition = np.bincount(
        composition_codes[celltype_codes >= 0],
        minlength=n_samples * max(1, len(celltype.categories)),
    ).reshape(n_samples, max(1, len(celltype.categories)))
    celltype_names = [str(value) for value in celltype.categories]
    return (
        cell_sample_index.astype(np.int32), sample_ids, sample_studies,
        sample_orig, np.asarray(LABELS, dtype=object)[sample_label_codes],
        sample_label_codes, cell_counts, composition, celltype_names,
    )


def aggregate_h5ad(path: Path, output_dir: Path, chunk_rows: int) -> dict:
    adata = ad.read_h5ad(path, backed="r")
    try:
        if not sparse.issparse(adata.X) and "CSRDataset" not in type(adata.X).__name__:
            raise ValueError(f"expected a sparse-backed expression matrix; found {type(adata.X).__name__}")
        (
            cell_to_sample, sample_ids, studies, orig_ids, labels,
            label_codes, cell_counts, composition, celltype_names,
        ) = build_sample_index(adata)
        n_samples = len(sample_ids)
        n_features = int(adata.n_vars)
        counts = np.zeros((n_samples, n_features), dtype=np.float64)
        fractional_entries = 0
        nonzero_entries = 0
        negative_entries = 0
        nonfinite_entries = 0
        min_nonzero = float("inf")
        max_nonzero = float("-inf")
        start_time = time.time()
        n_chunks = (adata.n_obs + chunk_rows - 1) // chunk_rows

        for chunk_number, start in enumerate(range(0, adata.n_obs, chunk_rows), start=1):
            end = min(start + chunk_rows, adata.n_obs)
            block = adata.X[start:end]
            if not sparse.issparse(block):
                block = sparse.csr_matrix(np.asarray(block))
            else:
                block = sparse.csr_matrix(block)
            values = block.data
            if values.size:
                nonzero_entries += int(values.size)
                nonfinite_entries += int((~np.isfinite(values)).sum())
                negative_entries += int((values < 0).sum())
                fractional_entries += int((~np.isclose(values, np.rint(values), rtol=1e-6, atol=1e-5)).sum())
                min_nonzero = min(min_nonzero, float(np.min(values)))
                max_nonzero = max(max_nonzero, float(np.max(values)))

            chunk_groups = cell_to_sample[start:end]
            unique_groups, inverse = np.unique(chunk_groups, return_inverse=True)
            row_aggregator = sparse.csr_matrix(
                (
                    np.ones(end - start, dtype=np.float32),
                    (inverse.astype(np.int32), np.arange(end - start, dtype=np.int32)),
                ),
                shape=(len(unique_groups), end - start),
            )
            partial = row_aggregator @ block
            counts[unique_groups] += partial.toarray()
            if chunk_number == 1 or chunk_number % 25 == 0 or chunk_number == n_chunks:
                print(f"Aggregated {end:,}/{adata.n_obs:,} cells ({chunk_number}/{n_chunks} chunks)", flush=True)

        if nonfinite_entries or negative_entries or fractional_entries:
            raise ValueError(
                "The expression matrix is not entirely finite, nonnegative, integer-like count data; "
                f"nonfinite={nonfinite_entries}, negative={negative_entries}, fractional={fractional_entries}. "
                "Stop and inspect source normalization before using count-based CPM."
            )
        if np.any(counts.sum(axis=1) <= 0):
            raise ValueError("at least one sample has zero aggregate expression")

        feature_names = np.asarray(adata.var_names.astype(str), dtype=object)
        if len(set(feature_names.tolist())) != len(feature_names):
            raise ValueError("gene/feature names are not unique")
        counts = np.rint(counts).astype(np.int64)
        output_dir.mkdir(parents=True, exist_ok=True)
        inventory = []
        for index in range(n_samples):
            inventory.append({
                "sample_group_id": str(sample_ids[index]),
                "orig_ident": str(orig_ids[index]),
                "study": str(studies[index]),
                "disease": str(labels[index]),
                "cells": int(cell_counts[index]),
                "pseudobulk_total_counts": int(counts[index].sum()),
                "patient_identity_verified": False,
            })
        write_csv(output_dir / "sample_inventory.csv", list(inventory[0]), inventory)

        composition_rows = []
        for sample_index, sample_id in enumerate(sample_ids):
            total_cells = int(cell_counts[sample_index])
            for type_index, type_name in enumerate(celltype_names):
                value = int(composition[sample_index, type_index])
                if value:
                    composition_rows.append({
                        "sample_group_id": str(sample_id),
                        "study": str(studies[sample_index]),
                        "disease": str(labels[sample_index]),
                        "cell_type": type_name,
                        "cells": value,
                        "percent_of_sample": round(100 * value / total_cells, 4),
                    })
        write_csv(
            output_dir / "celltype_composition.csv",
            ["sample_group_id", "study", "disease", "cell_type", "cells", "percent_of_sample"],
            composition_rows,
        )

        matrix_path = PROJECT_ROOT / "data" / "processed" / "allcells_pseudobulk_counts.npz"
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            matrix_path,
            counts=counts,
            sample_ids=sample_ids.astype(str),
            studies=studies.astype(str),
            orig_ids=orig_ids.astype(str),
            labels=labels.astype(str),
            cell_counts=cell_counts,
            feature_names=feature_names.astype(str),
        )
        elapsed = time.time() - start_time
        matrix_info = {
            "path": str(matrix_path),
            "shape": [int(n_samples), int(n_features)],
            "total_cells": int(adata.n_obs),
            "study_count": int(len(set(studies.tolist()))),
            "orig_ident_group_count": int(n_samples),
            "disease_counts_by_sample_group": {
                label: int(np.sum(labels == label)) for label in LABELS
            },
            "celltype_categories": celltype_names,
            "matrix_values": {
                "nonzero_entries": nonzero_entries,
                "integer_like_nonzero_entries": nonzero_entries - fractional_entries,
                "fractional_nonzero_entries": fractional_entries,
                "negative_nonzero_entries": negative_entries,
                "nonfinite_nonzero_entries": nonfinite_entries,
                "minimum_nonzero": min_nonzero if nonzero_entries else None,
                "maximum_nonzero": max_nonzero if nonzero_entries else None,
                "count_matrix_validation": (
                    "passed: all stored values finite, nonnegative, and integer-like within "
                    "float32 tolerance (rtol=1e-6, atol=1e-5)"
                ),
            },
            "aggregation_seconds": round(elapsed, 2),
            "patient_level_metadata_present": False,
            "sample_identity_note": (
                "orig.ident is a source sample/library identifier; the H5AD does not contain a verified patient ID. "
                "All records from each source study are held out together during the primary evaluation."
            ),
        }
        return {
            "adata": None,
            "counts": counts,
            "sample_ids": sample_ids.astype(str),
            "studies": studies.astype(str),
            "orig_ids": orig_ids.astype(str),
            "labels": labels.astype(str),
            "cell_counts": cell_counts,
            "feature_names": feature_names.astype(str),
            "composition": composition,
            "celltype_names": celltype_names,
            "sample_inventory": inventory,
            "matrix_info": matrix_info,
        }
    finally:
        adata.file.close()


def run_study_heldout_analysis(data: dict, report_dir: Path, seed: int) -> dict:
    counts = data["counts"]
    labels = data["labels"]
    studies = data["studies"]
    totals = counts.sum(axis=1, keepdims=True)
    expression = np.log1p(counts / totals * 1_000_000).astype(np.float32)
    unique_studies = sorted(np.unique(studies).tolist())
    if len(unique_studies) < 3:
        raise ValueError("at least three studies are required for study-held-out validation")
    for label in LABELS:
        if label not in set(labels.tolist()):
            raise ValueError(f"no sample groups for label {label}")
    report_dir.mkdir(parents=True, exist_ok=True)

    logo = LeaveOneGroupOut()
    oof_predictions = np.empty(len(labels), dtype=object)
    fold_rows = []
    selection_rows = []
    prediction_rows = []
    candidate_outer_predictions = {name: np.empty(len(labels), dtype=object) for name in MODEL_NAMES}

    for fold_number, (train_idx, test_idx) in enumerate(logo.split(expression, labels, groups=studies), start=1):
        heldout_study = str(studies[test_idx[0]])
        train_labels = set(labels[train_idx].tolist())
        if train_labels != set(LABELS):
            raise ValueError(
                f"holding out {heldout_study} removes a disease from training; "
                f"training labels are {sorted(train_labels)}"
            )
        print(f"\nStudy holdout {fold_number}/{len(unique_studies)}: {heldout_study}", flush=True)
        minimum_class = min(int(np.sum(labels[train_idx] == label)) for label in LABELS)
        n_splits = min(3, minimum_class)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + fold_number)
        inner_scores = {}
        for model_name in MODEL_NAMES:
            inner_oof = np.empty(len(train_idx), dtype=object)
            for inner_train_local, inner_valid_local in splitter.split(expression[train_idx], labels[train_idx]):
                inner_train = train_idx[inner_train_local]
                inner_valid = train_idx[inner_valid_local]
                if set(labels[inner_train].tolist()) != set(LABELS):
                    raise ValueError(f"inner fold lacks a disease class for {heldout_study}")
                estimator = make_pipeline(model_name, expression.shape[1], seed + fold_number)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(expression[inner_train], labels[inner_train])
                inner_oof[inner_valid_local] = estimator.predict(expression[inner_valid])
            scores = metric_dict(labels[train_idx], inner_oof)
            inner_scores[model_name] = scores
            selection_rows.append({
                "heldout_study": heldout_study,
                "candidate_model": model_name,
                **scores,
            })

        selected_model = max(
            MODEL_NAMES,
            key=lambda name: (
                inner_scores[name]["macro_f1"],
                inner_scores[name]["balanced_accuracy"],
                -MODEL_NAMES.index(name),
            ),
        )
        for model_name in MODEL_NAMES:
            estimator = make_pipeline(model_name, expression.shape[1], seed + fold_number)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(expression[train_idx], labels[train_idx])
            candidate_outer_predictions[model_name][test_idx] = estimator.predict(expression[test_idx])

        selected_predictions = candidate_outer_predictions[selected_model][test_idx]
        oof_predictions[test_idx] = selected_predictions
        heldout_scores = metric_dict(labels[test_idx], selected_predictions)
        fold_rows.append({
            "heldout_study": heldout_study,
            "test_sample_groups": int(len(test_idx)),
            "test_label_counts": json.dumps({label: int(np.sum(labels[test_idx] == label)) for label in LABELS}),
            "selected_model_from_inner_cv": selected_model,
            **heldout_scores,
        })
        for local_index, sample_index in enumerate(test_idx):
            prediction_rows.append({
                "sample_group_id": data["sample_ids"][sample_index],
                "study": studies[sample_index],
                "orig_ident": data["orig_ids"][sample_index],
                "cells": int(data["cell_counts"][sample_index]),
                "true_disease": labels[sample_index],
                "predicted_disease": selected_predictions[local_index],
                "selected_model": selected_model,
            })
        print(
            f"Selected {selected_model}; held-out-study macro-F1={heldout_scores['macro_f1']:.3f}",
            flush=True,
        )

    selected_metrics = metric_dict(labels, oof_predictions)
    model_comparison = []
    for model_name in MODEL_NAMES:
        model_comparison.append({
            "model": model_name,
            **metric_dict(labels, candidate_outer_predictions[model_name]),
        })
    final_model_name = max(
        MODEL_NAMES,
        key=lambda name: (
            float(np.mean([row["macro_f1"] for row in selection_rows if row["candidate_model"] == name])),
            float(np.mean([row["balanced_accuracy"] for row in selection_rows if row["candidate_model"] == name])),
            -MODEL_NAMES.index(name),
        ),
    )
    mean_inner_scores = {
        name: {
            metric: float(np.mean([
                row[metric] for row in selection_rows if row["candidate_model"] == name
            ]))
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        }
        for name in MODEL_NAMES
    }

    confusion = confusion_matrix(labels, oof_predictions, labels=list(LABELS))
    class_report = classification_report(
        labels, oof_predictions, labels=list(LABELS), output_dict=True, zero_division=0
    )
    write_csv(report_dir / "study_heldout_predictions.csv", list(prediction_rows[0]), prediction_rows)
    write_csv(report_dir / "study_heldout_folds.csv", list(fold_rows[0]), fold_rows)
    write_csv(report_dir / "inner_model_selection.csv", list(selection_rows[0]), selection_rows)
    write_csv(report_dir / "algorithm_comparison.csv", list(model_comparison[0]), model_comparison)
    (report_dir / "classification_report.txt").write_text(
        classification_report(labels, oof_predictions, labels=list(LABELS), zero_division=0),
        encoding="utf-8",
    )
    np.savetxt(report_dir / "study_heldout_confusion_matrix.csv", confusion, delimiter=",", fmt="%d")

    fig, ax = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("True disease")
    ax.set_title("Unseen-study held-out predictions")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            ax.text(column, row, str(confusion[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Sample groups")
    fig.savefig(report_dir / "study_heldout_confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    order = sorted(model_comparison, key=lambda row: row["macro_f1"], reverse=True)
    ax.bar([row["model"] for row in order], [row["macro_f1"] for row in order], color="#2d718e")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Candidate models on held-out studies (descriptive comparison)")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(report_dir / "algorithm_comparison.png", dpi=180)
    plt.close(fig)

    study_names = unique_studies
    counts_by_study = np.zeros((len(study_names), len(LABELS)), dtype=int)
    for study_index, study_name in enumerate(study_names):
        for label_index, label in enumerate(LABELS):
            counts_by_study[study_index, label_index] = int(np.sum((studies == study_name) & (labels == label)))
    write_csv(
        report_dir / "study_disease_sample_counts.csv",
        ["study", *LABELS],
        [{"study": study_name, **{label: int(counts_by_study[i, j]) for j, label in enumerate(LABELS)}}
         for i, study_name in enumerate(study_names)],
    )
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image = ax.imshow(counts_by_study, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(study_names)), study_names)
    ax.set_title("Source-study coverage by disease")
    for row in range(len(study_names)):
        for column in range(len(LABELS)):
            if counts_by_study[row, column]:
                ax.text(column, row, str(counts_by_study[row, column]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Sample groups")
    fig.savefig(report_dir / "study_disease_coverage.png", dpi=180)
    plt.close(fig)

    per_study = []
    for study_name in study_names:
        mask = studies == study_name
        present = [label for label in LABELS if np.any(labels[mask] == label)]
        per_study.append({
            "study": study_name,
            "sample_groups": int(mask.sum()),
            "labels_in_test": ";".join(present),
            "accuracy": float(accuracy_score(labels[mask], oof_predictions[mask])),
            "balanced_accuracy": float(balanced_accuracy_score(labels[mask], oof_predictions[mask])),
            "macro_f1_over_labels_present": float(f1_score(
                labels[mask], oof_predictions[mask], labels=present, average="macro", zero_division=0
            )),
        })
    write_csv(report_dir / "per_study_metrics.csv", list(per_study[0]), per_study)

    final_estimator = make_pipeline(final_model_name, expression.shape[1], seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_estimator.fit(expression, labels)
    selector = final_estimator.named_steps["select"]
    selected_indices = selector.get_support(indices=True)
    selected_names = data["feature_names"][selected_indices]
    selected_scores = selector.scores_[selected_indices]
    classifier = final_estimator.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        importance = np.mean(np.abs(classifier.coef_), axis=0)
        importance_kind = "mean absolute fitted coefficient; model-specific, not a biomarker claim"
    elif hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
        importance_kind = "Extra Trees impurity importance; exploratory, not a biomarker claim"
    else:
        importance = np.full(len(selected_indices), np.nan)
        importance_kind = "not available for selected nonlinear model; ANOVA scores are screening scores only"
    feature_rows = []
    for index in range(len(selected_indices)):
        feature_rows.append({
            "feature_name": str(selected_names[index]),
            "matrix_column_index": int(selected_indices[index]),
            "anova_f_score": float(selected_scores[index]) if np.isfinite(selected_scores[index]) else "",
            "model_importance": float(importance[index]) if np.isfinite(importance[index]) else "",
            "importance_definition": importance_kind,
        })
    feature_rows.sort(
        key=lambda row: float(
            row["model_importance"] if row["model_importance"] != "" else
            row["anova_f_score"] if row["anova_f_score"] != "" else -1
        ),
        reverse=True,
    )
    write_csv(report_dir / "top_features.csv", list(feature_rows[0]), feature_rows[:100])

    model_path = PROJECT_ROOT / "models" / "allcells_lung_disease_classifier.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": final_estimator,
        "model_name": final_model_name,
        "feature_names": data["feature_names"].tolist(),
        "label_order": list(LABELS),
        "normalization": "sample-level summed counts -> counts per million -> log1p",
        "sample_grouping": "gse_alias + orig.ident; patient ID is not provided/verified",
        "model_input_contract": "one pseudobulk count row in feature_names order; normalize as documented before pipeline.predict",
    }, model_path, compress=3)

    summary = {
        "status": "trained and evaluated; exploratory research model",
        "source_file": str(data["matrix_info"].get("source_file", "")),
        "dimensions": data["matrix_info"]["shape"],
        "total_cells": data["matrix_info"]["total_cells"],
        "studies": study_names,
        "study_count": len(study_names),
        "sample_group_count": len(labels),
        "label_order": list(LABELS),
        "sample_group_counts": {label: int(np.sum(labels == label)) for label in LABELS},
        "cell_counts_by_disease": {
            label: int(sum(int(row["cells"]) for row in data["sample_inventory"] if row["disease"] == label))
            for label in LABELS
        } if "sample_inventory" in data else {},
        "primary_evaluation": {
            "design": "nested study-held-out evaluation: each outer fold holds out one entire source study; model selection uses stratified sample-group folds only inside the training studies",
            "metrics_for_selected_procedure": selected_metrics,
            "confusion_matrix_labels_rows_true_columns_predicted": list(LABELS),
            "confusion_matrix": confusion.tolist(),
            "classification_report": class_report,
            "candidate_heldout_study_metrics_descriptive": model_comparison,
            "selected_model_by_outer_fold": [
                {"study": row["heldout_study"], "model": row["selected_model_from_inner_cv"]}
                for row in fold_rows
            ],
        },
        "final_development_model": {
            "algorithm": final_model_name,
            "selection_rule": "highest mean inner-CV macro-F1 across outer training folds; balanced accuracy breaks ties",
            "mean_inner_cv_scores_across_outer_training_folds": mean_inner_scores,
            "path": str(model_path),
            "fit_on_all_sample_groups": True,
            "performance_source": "Do not report training-set performance; use the held-out-study metrics above.",
        },
        "outputs": {
            "report_dir": str(report_dir),
            "pseudobulk_counts": str(PROJECT_ROOT / "data" / "processed" / "allcells_pseudobulk_counts.npz"),
            "model": str(model_path),
        },
        "limitations": [
            "The H5AD provides orig.ident, Disease, gse_alias, and celltypeuse, but no verified biological patient identifier.",
            "The number of orig.ident groups is a count of source sample/library groups, not a verified patient count.",
            "The Zenodo record describes 16 studies and 1,470,730 cells; study-held-out evaluation reduces source-study leakage but cannot establish patient-level or clinical validity.",
            "Disease and source study are imbalanced; disease-study confounding may limit transfer even with whole-study holdout.",
            "This is a retrospective research classifier, not a diagnostic tool.",
            "Feature rankings are exploratory and must not be presented as validated biomarkers.",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "anndata": ad.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with (report_dir / "analysis_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5ad", type=Path,
        default=Path.home() / "Downloads" / "atlas-001.h5ad",
        help="Input H5AD; opened read-only and processed in row chunks.",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "allcells_atlas",
    )
    parser.add_argument("--chunk-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.h5ad.is_file():
        parser.error(f"input H5AD does not exist: {args.h5ad}")
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")

    print(f"Opening {args.h5ad} in backed read-only mode", flush=True)
    data = aggregate_h5ad(args.h5ad, args.report_dir, args.chunk_rows)
    data["matrix_info"]["source_file"] = str(args.h5ad)
    data["matrix_info"]["source_file_size_bytes"] = args.h5ad.stat().st_size
    print("Pseudobulk aggregation complete; starting held-out-study model comparison", flush=True)
    summary = run_study_heldout_analysis(data, args.report_dir, args.seed)
    print(json.dumps({
        "summary": str(args.report_dir / "analysis_summary.json"),
        "sample_groups": summary["sample_group_count"],
        "studies": summary["study_count"],
        "selected_final_model": summary["final_development_model"]["algorithm"],
        "heldout_study_metrics": summary["primary_evaluation"]["metrics_for_selected_procedure"],
    }, indent=2))


if __name__ == "__main__":
    main()
