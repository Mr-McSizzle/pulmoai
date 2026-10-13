#!/usr/bin/env python3
"""Train and evaluate donor-level disease classifiers on GSE136831."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import shutil
import sys
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISEASES = ("Control", "COPD", "IPF")
SIMPLICITY_ORDER = ("Logistic Regression", "Linear SVM", "RBF SVM", "Extra Trees")


def read_matrix_header(path: Path) -> tuple[int, int, int]:
    with path.open("rt", encoding="ascii") as stream:
        header = stream.readline().strip().lower()
        if header != "%%matrixmarket matrix coordinate integer general":
            raise ValueError(f"unsupported Matrix Market header: {header}")
        for line in stream:
            if line.strip() and not line.lstrip().startswith("%"):
                dimensions = tuple(int(value) for value in line.split())
                if len(dimensions) != 3:
                    raise ValueError("matrix dimension line must contain rows, columns, and entries")
                return dimensions
    raise ValueError("matrix file contains no dimension line")


def load_metadata(metadata_path: Path, barcode_path: Path, matrix_path: Path):
    with gzip.open(metadata_path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {
        "CellBarcode_Identity", "Disease_Identity", "Subject_Identity",
        "CellType_Category", "nUMI", "nGene",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"metadata is missing required fields: {sorted(required - set(rows[0] if rows else []))}")

    with gzip.open(barcode_path, "rt", encoding="utf-8") as stream:
        barcodes = [line.strip() for line in stream if line.strip()]
    feature_count, matrix_cells, matrix_entries = read_matrix_header(matrix_path)
    if len(rows) != len(barcodes) or matrix_cells != len(barcodes):
        raise ValueError(
            f"cell counts disagree: metadata={len(rows)}, barcodes={len(barcodes)}, matrix={matrix_cells}"
        )
    mismatches = [index for index, (row, barcode) in enumerate(zip(rows, barcodes))
                  if row["CellBarcode_Identity"] != barcode]
    if mismatches:
        raise ValueError(f"metadata/barcode order mismatch at row {mismatches[0] + 1}")

    donor_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        disease = row["Disease_Identity"].strip()
        donor = row["Subject_Identity"].strip()
        if disease not in DISEASES or not donor:
            raise ValueError(f"unexpected disease/donor value: {disease!r}, {donor!r}")
        donor_labels[donor].add(disease)
    mixed_donors = {donor: labels for donor, labels in donor_labels.items() if len(labels) != 1}
    if mixed_donors:
        raise ValueError(f"donors have inconsistent disease labels: {mixed_donors}")

    donor_ids = sorted(donor_labels)
    label_by_donor = {donor: next(iter(donor_labels[donor])) for donor in donor_ids}
    donor_index = {donor: index for index, donor in enumerate(donor_ids)}
    cell_to_donor = np.fromiter(
        (donor_index[row["Subject_Identity"].strip()] for row in rows),
        dtype=np.int32,
        count=len(rows),
    )
    if set(label_by_donor.values()) != set(DISEASES):
        raise ValueError(f"expected all three disease classes; found {sorted(set(label_by_donor.values()))}")
    return rows, donor_ids, label_by_donor, cell_to_donor, feature_count, matrix_entries


def compile_aggregator(build_dir: Path) -> Path:
    compiler = shutil.which("clang++") or shutil.which("c++")
    if not compiler:
        raise RuntimeError("A C++17 compiler is required (Xcode Command Line Tools on macOS).")
    executable = build_dir / "aggregate_matrix"
    source = PROJECT_ROOT / "scripts" / "aggregate_matrix.cpp"
    subprocess.run(
        [compiler, "-O3", "-std=c++17", str(source), "-o", str(executable)],
        check=True,
    )
    return executable


def build_pipeline(name: str, feature_count: int, top_features: int, seed: int) -> Pipeline:
    classifiers = {
        "Logistic Regression": LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=seed
        ),
        "Linear SVM": LinearSVC(C=0.1, class_weight="balanced", max_iter=10000, random_state=seed),
        "RBF SVM": SVC(C=1.0, kernel="rbf", class_weight="balanced"),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
    }
    return Pipeline([
        ("remove_constant", VarianceThreshold(threshold=0.0)),
        ("select", SelectKBest(score_func=f_classif, k=min(top_features, feature_count))),
        ("scale", StandardScaler()),
        ("classifier", classifiers[name]),
    ])


def scores(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_cohort_reports(output_dir: Path, rows: list[dict], donor_ids: list[str],
                         label_by_donor: dict[str, str], cell_to_donor: np.ndarray,
                         counts: np.ndarray, count_totals: np.ndarray) -> None:
    donor_cell_counts = np.bincount(cell_to_donor, minlength=len(donor_ids))
    donor_umi = np.zeros(len(donor_ids), dtype=np.int64)
    donor_ngene = np.zeros(len(donor_ids), dtype=np.int64)
    celltypes: list[set[str]] = [set() for _ in donor_ids]
    composition: dict[tuple[str, str], int] = Counter()
    for index, row in enumerate(rows):
        donor_index = int(cell_to_donor[index])
        donor_umi[donor_index] += int(row["nUMI"])
        donor_ngene[donor_index] += int(row["nGene"])
        category = row["CellType_Category"].strip()
        celltypes[donor_index].add(category)
        composition[(donor_ids[donor_index], category)] += 1

    donor_records = []
    for index, donor in enumerate(donor_ids):
        donor_records.append({
            "donor_id": donor,
            "disease": label_by_donor[donor],
            "cells": int(donor_cell_counts[index]),
            "mean_cell_umi": round(donor_umi[index] / donor_cell_counts[index], 2),
            "mean_cell_genes": round(donor_ngene[index] / donor_cell_counts[index], 2),
            "pseudobulk_total_counts": int(count_totals[index]),
            "cell_type_categories": ";".join(sorted(celltypes[index])),
        })
    write_csv(output_dir / "donor_summary.csv", list(donor_records[0]), donor_records)

    comp_records = []
    for donor in donor_ids:
        total = sum(value for (this_donor, _), value in composition.items() if this_donor == donor)
        for (this_donor, category), count in sorted(composition.items()):
            if this_donor == donor:
                comp_records.append({
                    "donor_id": donor, "disease": label_by_donor[donor],
                    "cell_type": category, "cells": count,
                    "percent_of_donor": round(100 * count / total, 3),
                })
    write_csv(output_dir / "celltype_composition.csv",
              ["donor_id", "disease", "cell_type", "cells", "percent_of_donor"], comp_records)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    ordered_cells = [donor_cell_counts[[i for i, donor in enumerate(donor_ids)
                                       if label_by_donor[donor] == disease]] for disease in DISEASES]
    axes[0].boxplot(ordered_cells, tick_labels=DISEASES, showfliers=False)
    for pos, disease in enumerate(DISEASES, start=1):
        values = ordered_cells[pos - 1]
        axes[0].scatter(np.full(len(values), pos), values, s=22, alpha=0.75)
    axes[0].set(title="Cells per donor", ylabel="Number of cells")
    categories = sorted({category for _, category in composition})
    fractions = np.zeros((len(DISEASES), len(categories)), dtype=float)
    for disease_index, disease in enumerate(DISEASES):
        for category_index, category in enumerate(categories):
            numerator = sum(composition.get((donor, category), 0) for donor in donor_ids
                            if label_by_donor[donor] == disease)
            denominator = sum(donor_cell_counts[i] for i, donor in enumerate(donor_ids)
                              if label_by_donor[donor] == disease)
            fractions[disease_index, category_index] = 100 * numerator / denominator
    image = axes[1].imshow(fractions, aspect="auto", cmap="Blues")
    axes[1].set_yticks(range(len(DISEASES)), DISEASES)
    axes[1].set_xticks(range(len(categories)), categories, rotation=45, ha="right")
    axes[1].set(title="Cell-type composition by diagnosis", xlabel="Cell-type category")
    fig.colorbar(image, ax=axes[1], label="Percent of cells")
    fig.suptitle("GSE136831 cohort overview (descriptive; cells are not independent patients)")
    fig.savefig(output_dir / "cohort_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    downloads = Path.home() / "Downloads"
    parser.add_argument("--matrix", type=Path, default=downloads / "GSE136831_RawCounts_Sparse.mtx")
    parser.add_argument("--metadata", type=Path,
                        default=downloads / "GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz")
    parser.add_argument("--barcodes", type=Path,
                        default=downloads / "GSE136831_AllCells.cellBarcodes.txt.gz")
    parser.add_argument("--gene-ids", type=Path,
                        default=downloads / "GSE136831_AllCells.GeneIDs.txt.gz")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports" / "lung_atlas")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--model-path", type=Path,
                        default=PROJECT_ROOT / "models" / "lung_disease_classifier.joblib")
    parser.add_argument("--top-features", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for path in (args.matrix, args.metadata, args.barcodes):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    if args.top_features < 1:
        parser.error("--top-features must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    rows, donor_ids, label_by_donor, cell_to_donor, feature_count, matrix_entries = load_metadata(
        args.metadata, args.barcodes, args.matrix
    )
    donor_labels = np.asarray([label_by_donor[donor] for donor in donor_ids])
    cell_labels = Counter(row["Disease_Identity"].strip() for row in rows)
    donor_class_counts = Counter(donor_labels)

    print(f"Verified {len(rows):,} cells, {len(donor_ids)} donors, {feature_count:,} genes; "
          f"{matrix_entries:,} sparse entries.", flush=True)
    print("Donors per class: " + ", ".join(f"{key}={donor_class_counts[key]}" for key in DISEASES),
          flush=True)
    print("Streaming sparse matrix into donor pseudobulk counts (low-memory path)...", flush=True)

    with tempfile.TemporaryDirectory(prefix="lung_atlas_") as temporary:
        temporary_path = Path(temporary)
        mapping_path = temporary_path / "cell_to_donor.txt"
        binary_path = temporary_path / "pseudobulk_counts.bin"
        with mapping_path.open("w", encoding="ascii") as mapping:
            mapping.write(f"{len(rows)} {len(donor_ids)}\n")
            mapping.writelines(f"{int(value)}\n" for value in cell_to_donor)
        executable = compile_aggregator(temporary_path)
        subprocess.run([str(executable), str(args.matrix), str(mapping_path), str(binary_path)], check=True)
        raw = np.fromfile(binary_path, dtype=np.uint64)

    expected_size = feature_count * len(donor_ids)
    if raw.size != expected_size:
        raise RuntimeError(f"aggregator output has {raw.size} values; expected {expected_size}")
    counts = raw.reshape(feature_count, len(donor_ids)).T.copy()
    count_totals = counts.sum(axis=1, dtype=np.uint64)
    if np.any(count_totals == 0):
        raise ValueError("at least one donor has an empty pseudobulk profile")

    gene_ids: list[str]
    gene_ids_available = args.gene_ids.is_file()
    if gene_ids_available:
        with gzip.open(args.gene_ids, "rt", encoding="utf-8") as stream:
            gene_ids = [line.strip() for line in stream if line.strip()]
        if len(gene_ids) != feature_count:
            raise ValueError(f"gene ID file has {len(gene_ids)} rows; expected {feature_count}")
    else:
        gene_ids = [f"matrix_row_{index + 1:05d}" for index in range(feature_count)]
        print("Note: gene-ID sidecar not found; model remains trainable, but feature names "
              "will use matrix row numbers.", flush=True)

    np.savez_compressed(
        args.processed_dir / "pseudobulk_counts.npz",
        counts=counts,
        donor_ids=np.asarray(donor_ids),
        labels=donor_labels,
        gene_ids=np.asarray(gene_ids),
    )
    write_cohort_reports(args.output_dir, rows, donor_ids, label_by_donor, cell_to_donor,
                         counts, count_totals)

    library_sizes = counts.sum(axis=1, dtype=np.uint64).astype(np.float64)
    expression = np.log1p(counts.astype(np.float64) / library_sizes[:, None] * 1_000_000)
    minimum_class_count = min(donor_class_counts.values())
    outer_folds = min(5, minimum_class_count)
    if outer_folds < 3:
        raise ValueError("at least three donors per class are required for stratified evaluation")
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=args.seed)
    names = list(SIMPLICITY_ORDER)
    nested_predictions = np.empty(len(donor_ids), dtype=object)
    selected_by_outer_fold = []
    outer_fold_records = []
    nested_prediction_records = []

    print(f"Running nested, donor-stratified {outer_folds}-fold evaluation across "
          f"{len(names)} algorithms...", flush=True)
    for fold_number, (train_indices, test_indices) in enumerate(outer_cv.split(expression, donor_labels), 1):
        train_y = donor_labels[train_indices]
        inner_class_min = min(Counter(train_y).values())
        inner_folds = min(3, inner_class_min)
        if inner_folds < 2:
            raise ValueError("not enough donors in an outer training fold for inner CV")
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True,
                                   random_state=args.seed + fold_number)
        inner_scores = {}
        for name in names:
            estimator = build_pipeline(name, feature_count, args.top_features, args.seed)
            validation = cross_validate(
                estimator, expression[train_indices], train_y, cv=inner_cv,
                scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
                n_jobs=1, error_score="raise",
            )
            inner_scores[name] = (
                float(np.mean(validation["test_macro_f1"])),
                float(np.mean(validation["test_balanced_accuracy"])),
            )
        chosen = max(names, key=lambda name: (inner_scores[name][0], inner_scores[name][1], -names.index(name)))
        selected_by_outer_fold.append(chosen)
        outer_estimator = build_pipeline(chosen, feature_count, args.top_features, args.seed)
        outer_estimator.fit(expression[train_indices], train_y)
        fold_predictions = outer_estimator.predict(expression[test_indices])
        nested_predictions[test_indices] = fold_predictions
        fold_score = scores(donor_labels[test_indices], fold_predictions)
        outer_fold_records.append({
            "outer_fold": fold_number,
            "selected_model": chosen,
            "inner_macro_f1": inner_scores[chosen][0],
            "inner_balanced_accuracy": inner_scores[chosen][1],
            **fold_score,
        })
        for index, prediction in zip(test_indices, fold_predictions):
            nested_prediction_records.append({
                "donor_id": donor_ids[index], "true_disease": donor_labels[index],
                "predicted_disease": prediction, "outer_fold": fold_number,
                "selected_model": chosen,
            })
        print(f"Outer fold {fold_number}/{outer_folds}: selected {chosen}; "
              f"macro-F1={fold_score['macro_f1']:.3f}", flush=True)

    overall_nested = scores(donor_labels, nested_predictions)
    write_csv(args.output_dir / "nested_oof_predictions.csv",
              ["donor_id", "true_disease", "predicted_disease", "outer_fold", "selected_model"],
              nested_prediction_records)
    write_csv(args.output_dir / "nested_outer_folds.csv", list(outer_fold_records[0]), outer_fold_records)

    development_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=args.seed + 100)
    comparison_rows = []
    for name in names:
        estimator = build_pipeline(name, feature_count, args.top_features, args.seed)
        result = cross_validate(
            estimator, expression, donor_labels, cv=development_cv,
            scoring={"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy",
                     "accuracy": "accuracy"}, n_jobs=1, error_score="raise",
        )
        comparison_rows.append({
            "algorithm": name,
            "cv_macro_f1_mean": float(np.mean(result["test_macro_f1"])),
            "cv_macro_f1_sd": float(np.std(result["test_macro_f1"], ddof=1)),
            "cv_balanced_accuracy_mean": float(np.mean(result["test_balanced_accuracy"])),
            "cv_accuracy_mean": float(np.mean(result["test_accuracy"])),
        })
    winner = max(comparison_rows,
                 key=lambda row: (row["cv_macro_f1_mean"], row["cv_balanced_accuracy_mean"],
                                  -names.index(row["algorithm"])))["algorithm"]
    write_csv(args.output_dir / "algorithm_comparison.csv", list(comparison_rows[0]), comparison_rows)

    final_model = build_pipeline(winner, feature_count, args.top_features, args.seed)
    final_model.fit(expression, donor_labels)
    joblib.dump({
        "pipeline": final_model,
        "disease_classes": list(final_model.classes_),
        "gene_ids": gene_ids,
        "gene_ids_available": gene_ids_available,
        "normalization": "donor-summed raw counts -> counts per million -> log1p",
        "dataset": "GSE136831 Adams IPF Cell Atlas",
        "training_donors": donor_ids,
        "model_selection": "highest mean donor-stratified CV macro-F1; tie-break by balanced accuracy",
    }, args.model_path, compress=3)

    var_mask = final_model.named_steps["remove_constant"].get_support()
    selected_in_filtered = final_model.named_steps["select"].get_support()
    selected_rows = np.flatnonzero(var_mask)[selected_in_filtered]
    classifier = final_model.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        importances = np.mean(np.abs(classifier.coef_), axis=0)
    else:
        importances = classifier.feature_importances_
    ranked = np.argsort(importances)[::-1][:50]
    feature_records = [{
        "rank": rank + 1,
        "matrix_row_1based": int(selected_rows[index] + 1),
        "feature_id": gene_ids[selected_rows[index]],
        "importance": float(importances[index]),
    } for rank, index in enumerate(ranked)]
    write_csv(args.output_dir / "top_features.csv", list(feature_records[0]), feature_records)

    class_report = classification_report(donor_labels, nested_predictions, labels=DISEASES,
                                         target_names=DISEASES, zero_division=0, digits=3)
    (args.output_dir / "classification_report.txt").write_text(
        "Nested donor-level out-of-fold classification report\n"
        "Each donor is held out once; model selection occurs inside each outer training fold.\n\n"
        + class_report, encoding="utf-8")
    cm = confusion_matrix(donor_labels, nested_predictions, labels=DISEASES)
    fig, axis = plt.subplots(figsize=(6.8, 6.1), constrained_layout=True)
    ConfusionMatrixDisplay(cm, display_labels=DISEASES).plot(
        ax=axis, cmap="Blues", colorbar=False, values_format="d"
    )
    axis.set_title(f"Nested donor-level CV\nMacro-F1 {overall_nested['macro_f1']:.3f} · "
                   f"Balanced accuracy {overall_nested['balanced_accuracy']:.3f}")
    fig.savefig(args.output_dir / "nested_confusion_matrix.png", dpi=190)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ordered = sorted(comparison_rows, key=lambda row: row["cv_macro_f1_mean"], reverse=True)
    axis.bar([row["algorithm"] for row in ordered],
             [row["cv_macro_f1_mean"] for row in ordered],
             yerr=[row["cv_macro_f1_sd"] for row in ordered], capsize=4,
             color=["#0f766e" if row["algorithm"] == winner else "#94a3b8" for row in ordered])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Mean macro-F1 (5-fold donor CV)")
    axis.set_title("Algorithm comparison (error bars: fold SD; small cohort)")
    axis.tick_params(axis="x", rotation=15)
    fig.savefig(args.output_dir / "algorithm_comparison.png", dpi=190)
    plt.close(fig)

    variance = np.var(expression, axis=0)
    variable_features = np.argsort(variance)[-min(500, feature_count):]
    pca_input = StandardScaler().fit_transform(expression[:, variable_features])
    coordinates = PCA(n_components=2, random_state=args.seed).fit_transform(pca_input)
    fig, axis = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
    palette = {"Control": "#2b8cbe", "COPD": "#f28e2b", "IPF": "#59a14f"}
    for disease in DISEASES:
        mask = donor_labels == disease
        axis.scatter(coordinates[mask, 0], coordinates[mask, 1], label=disease,
                     color=palette[disease], alpha=0.85, s=42, edgecolor="white", linewidth=0.4)
    axis.set_xlabel("Principal component 1")
    axis.set_ylabel("Principal component 2")
    axis.set_title("Exploratory PCA of donor pseudobulk profiles\n(not a validation metric)")
    axis.legend(frameon=False)
    fig.savefig(args.output_dir / "pseudobulk_pca.png", dpi=190)
    plt.close(fig)

    audit = {
        "project": "PulmoAI — exploratory lung disease classification",
        "dataset": "GSE136831 (Adams IPF Cell Atlas)",
        "matrix_shape_features_by_cells": [feature_count, len(rows)],
        "matrix_nonzero_entries": matrix_entries,
        "source_files": {
            "matrix": {"path": str(args.matrix), "bytes": args.matrix.stat().st_size},
            "metadata": {"path": str(args.metadata), "bytes": args.metadata.stat().st_size},
            "barcodes": {"path": str(args.barcodes), "bytes": args.barcodes.stat().st_size},
            "gene_ids": ({"path": str(args.gene_ids), "bytes": args.gene_ids.stat().st_size}
                         if gene_ids_available else None),
        },
        "cell_count_by_disease": dict(cell_labels),
        "donor_count": len(donor_ids),
        "donor_count_by_disease": dict(donor_class_counts),
        "donor_level_nested_cv": {
            "outer_folds": outer_folds,
            "inner_folds": "up to 3, stratified, within outer training fold",
            "metrics": overall_nested,
            "confusion_matrix_labels_order": list(DISEASES),
            "confusion_matrix": cm.tolist(),
            "selected_models_by_outer_fold": selected_by_outer_fold,
        },
        "full_data_cv_algorithm_selection": {
            "selected_algorithm": winner,
            "candidates": comparison_rows,
            "selection_metric": "mean macro-F1; balanced accuracy used as tie-break",
        },
        "gene_ids_sidecar_available": gene_ids_available,
        "model_configuration": {
            "top_features": min(args.top_features, feature_count),
            "normalization": "donor-summed raw counts -> counts per million -> log1p",
            "seed": args.seed,
            "candidate_algorithms": names,
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "interpretation_note": (
            "Single-study exploratory cohort; donor-level CV prevents cell leakage but does not "
            "establish external clinical generalization. Algorithm ranking on this small cohort "
            "is uncertain. Feature selection was fit inside every evaluation fold."
        ),
        "model_file": str(args.model_path),
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\nNested outer-CV results (primary exploratory estimate):", flush=True)
    print(json.dumps(overall_nested, indent=2), flush=True)
    print(f"Development CV selected: {winner}", flush=True)
    print(f"Saved model: {args.model_path}", flush=True)
    print(f"Saved reports: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
