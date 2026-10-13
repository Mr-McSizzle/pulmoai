#!/usr/bin/env python3
"""Train an exploratory donor-level COVID-versus-healthy baseline."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import tarfile
from collections import Counter
from pathlib import Path

import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
from scipy.sparse import issparse
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


TARGET_LABELS = ("healthy", "COVID")
MATRIX_PREFIXES = ("GSM8335813_", "GSM8335815_")
CHUNK_SIZE = 2_000_000


def open_text_member(archive: tarfile.TarFile, member_name: str):
    member = archive.getmember(member_name)
    source = archive.extractfile(member)
    if source is None:
        raise OSError(f"could not read archive member: {member_name}")
    compressed = gzip.GzipFile(fileobj=source, mode="rb")
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def read_lines(archive: tarfile.TarFile, member_name: str) -> list[str]:
    with open_text_member(archive, member_name) as stream:
        return [line.rstrip("\r\n") for line in stream]


def load_donor_metadata(archive: tarfile.TarFile, token: str):
    metadata_member = next(
        member
        for member in archive.getmembers()
        if member.name.endswith("_CMO.csv.gz")
        and member.name.split("_", 1)[1].removesuffix("_CMO.csv.gz") == token
    )
    with open_text_member(archive, metadata_member.name) as stream:
        rows = list(csv.DictReader(stream))
    by_cmo = {row["CMO_id"]: row for row in rows if row.get("CMO_id")}
    return metadata_member, by_cmo


def load_assignments(archive: tarfile.TarFile, token: str):
    assignment_member = next(
        member
        for member in archive.getmembers()
        if member.name.endswith("_CMO_assignment_confidence_table.csv.gz")
        and member.name.split("_", 1)[1].removesuffix(
            "_CMO_assignment_confidence_table.csv.gz"
        )
        == token
    )
    with open_text_member(archive, assignment_member.name) as stream:
        return assignment_member, {
            row["Barcodes"]: row
            for row in csv.DictReader(stream)
            if row.get("Barcodes")
        }


def target_for_disease(disease: str) -> str:
    if disease == "healthy":
        return "healthy"
    if disease.startswith("COVID"):
        return "COVID"
    raise ValueError(f"unexpected disease label in selected subset: {disease}")


def aggregate_library(
    archive: tarfile.TarFile,
    matrix_member: tarfile.TarInfo,
    donor_index: dict[str, int],
    donor_count: int,
    expected_features: list[str] | None,
):
    base = matrix_member.name.removesuffix("_matrix.mtx.gz")
    token = Path(base).name.split("_", 1)[1]
    barcode_member_name = f"{base}_barcodes.tsv.gz"
    feature_member_name = f"{base}_features.tsv.gz"
    barcodes = read_lines(archive, barcode_member_name)
    feature_rows = read_lines(archive, feature_member_name)
    features = [row.split("\t", 1)[0] for row in feature_rows]
    if expected_features is not None and features != expected_features:
        raise ValueError(f"feature identifiers/order differ in {matrix_member.name}")

    metadata_member, cmo_metadata = load_donor_metadata(archive, token)
    assignment_member, assignments = load_assignments(archive, token)
    labels_by_cmo = {
        cmo: row
        for cmo, row in cmo_metadata.items()
        if row.get("sample_id") and row.get("disease")
    }

    donors_for_barcode = np.full(len(barcodes), -1, dtype=np.int32)
    cell_counts: Counter[str] = Counter()
    metadata_by_donor: dict[str, tuple[str, str]] = {}
    for column, barcode in enumerate(barcodes):
        assignment = assignments.get(barcode)
        if assignment is None:
            continue
        donor = labels_by_cmo.get(assignment.get("Assignment", ""))
        if donor is None:
            continue
        sample_id = donor["sample_id"]
        disease = donor["disease"]
        target = target_for_disease(disease)
        metadata_by_donor[sample_id] = (disease, target)
        donors_for_barcode[column] = donor_index[sample_id]
        cell_counts[sample_id] += 1

    matrix_source = archive.extractfile(matrix_member)
    if matrix_source is None:
        raise OSError(f"could not read matrix member: {matrix_member.name}")
    with gzip.GzipFile(fileobj=matrix_source, mode="rb") as compressed:
        matrix = mmread(compressed, spmatrix=False)
    if not issparse(matrix):
        raise ValueError(f"expected a sparse matrix in {matrix_member.name}")
    if matrix.shape != (len(features), len(barcodes)):
        raise ValueError(f"matrix dimensions do not match features/barcodes in {matrix_member.name}")

    coordinate_matrix = matrix.tocoo(copy=False)
    totals = np.zeros((len(features), donor_count), dtype=np.float64)
    for start in range(0, coordinate_matrix.nnz, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, coordinate_matrix.nnz)
        columns = coordinate_matrix.col[start:end]
        sample_indices = donors_for_barcode[columns]
        keep = sample_indices >= 0
        if not np.any(keep):
            continue
        flat_indices = (
            coordinate_matrix.row[start:end][keep].astype(np.int64) * donor_count
            + sample_indices[keep]
        )
        chunk_totals = np.bincount(
            flat_indices,
            weights=coordinate_matrix.data[start:end][keep],
            minlength=len(features) * donor_count,
        ).reshape(len(features), donor_count)
        totals += chunk_totals

    stats = {
        "matrix_member": matrix_member.name,
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "matrix_nonzero_entries": int(coordinate_matrix.nnz),
        "metadata_member": metadata_member.name,
        "assignment_member": assignment_member.name,
        "assignment_rows": len(assignments),
        "barcodes_in_matrix": len(barcodes),
        "matched_labeled_cells_by_donor": dict(sorted(cell_counts.items())),
    }
    return features, totals, metadata_by_donor, cell_counts, stats


def build_pipeline(feature_count: int, top_features: int):
    return Pipeline(
        [
            ("remove_constant", VarianceThreshold(threshold=0.0)),
            ("select", SelectKBest(score_func=f_classif, k=min(top_features, feature_count))),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to GSE248688_RAW.tar")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--top-features", type=int, default=100)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.archive.is_file():
        parser.error(f"archive does not exist or is not a file: {args.archive}")
    if args.top_features < 1 or args.folds < 2:
        parser.error("top-features must be positive and folds must be at least two")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.archive, mode="r:*") as archive:
        matrix_members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.name.endswith("_matrix.mtx.gz")
                and Path(member.name).name.startswith(MATRIX_PREFIXES)
            ),
            key=lambda member: member.name,
        )
        if len(matrix_members) != 2:
            parser.error(f"expected two COVID-HC matrices, found {len(matrix_members)}")

        donor_metadata: dict[str, tuple[str, str]] = {}
        donor_counts: Counter[str] = Counter()
        all_stats = []
        combined_counts = None
        feature_ids = None
        donor_id_set = set()
        for member in archive.getmembers():
            if not (
                member.name.endswith("_CMO.csv.gz")
                and Path(member.name).name.startswith(("GSM8335814_", "GSM8335816_"))
            ):
                continue
            with open_text_member(archive, member.name) as stream:
                donor_id_set.update(
                    row["sample_id"]
                    for row in csv.DictReader(stream)
                    if row.get("sample_id") and row.get("disease")
                )
        donor_ids = sorted(donor_id_set)
        if len(donor_ids) != 8:
            parser.error(f"expected eight labeled donors, found {len(donor_ids)}")
        donor_index = {sample_id: index for index, sample_id in enumerate(donor_ids)}

        for matrix_member in matrix_members:
            token = Path(matrix_member.name.removesuffix("_matrix.mtx.gz")).name.split("_", 1)[1]
            _, metadata_for_library = load_donor_metadata(archive, token)
            for row in metadata_for_library.values():
                if row.get("sample_id") and row.get("disease"):
                    target = target_for_disease(row["disease"])
                    previous = donor_metadata.get(row["sample_id"])
                    if previous and previous != (row["disease"], target):
                        parser.error(f"conflicting labels for donor {row['sample_id']}")
                    donor_metadata[row["sample_id"]] = (row["disease"], target)

            features, matrix_counts, metadata_by_donor, cell_counts, stats = aggregate_library(
                archive,
                matrix_member,
                donor_index,
                len(donor_ids),
                feature_ids,
            )
            feature_ids = features
            if combined_counts is None:
                combined_counts = matrix_counts
            else:
                combined_counts += matrix_counts
            donor_counts.update(cell_counts)
            all_stats.append(stats)
            del matrix_counts

    if combined_counts is None or feature_ids is None:
        parser.error("no expression counts were read")
    if set(donor_metadata) != set(donor_ids):
        parser.error("CMO donor IDs and labels do not match")

    y = np.array(
        [1 if donor_metadata[donor][1] == "COVID" else 0 for donor in donor_ids],
        dtype=np.int32,
    )
    class_counts = Counter(donor_metadata[donor][1] for donor in donor_ids)
    if set(class_counts) != set(TARGET_LABELS) or min(class_counts.values()) < args.folds:
        parser.error(f"each target class must have at least {args.folds} donors: {dict(class_counts)}")

    library_sizes = combined_counts.sum(axis=0)
    if np.any(library_sizes <= 0):
        parser.error("at least one donor has no assigned expression counts")
    expression = np.log1p((combined_counts / library_sizes[np.newaxis, :]) * 1_000_000).T

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    predictions = np.empty(len(donor_ids), dtype=np.int32)
    probabilities = np.empty(len(donor_ids), dtype=np.float64)
    fold_indices = np.empty(len(donor_ids), dtype=np.int32)
    fold_details = []
    for fold_number, (train_indices, test_indices) in enumerate(splitter.split(expression, y), start=1):
        pipeline = build_pipeline(expression.shape[1], args.top_features)
        pipeline.fit(expression[train_indices], y[train_indices])
        predictions[test_indices] = pipeline.predict(expression[test_indices])
        probabilities[test_indices] = pipeline.predict_proba(expression[test_indices])[:, 1]
        fold_indices[test_indices] = fold_number
        fold_details.append(
            {
                "fold": fold_number,
                "train_donors": [donor_ids[index] for index in train_indices],
                "test_donors": [donor_ids[index] for index in test_indices],
            }
        )

    labels = ["healthy", "COVID"]
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    report = classification_report(
        y, predictions, labels=[0, 1], target_names=labels, output_dict=True, zero_division=0
    )
    metrics = {
        "task": "donor-level COVID versus healthy classification",
        "status": "exploratory; eight independent donors only",
        "archive": str(args.archive),
        "donor_count": len(donor_ids),
        "class_counts": dict(class_counts),
        "donor_cell_counts": dict(sorted(donor_counts.items())),
        "preprocessing": "sum assigned raw counts by donor across both libraries; normalize each donor to counts per million; apply log1p",
        "cell_filtering": "keep CMO Assignment values mapped to a donor label; omit Blanks, Multiplet, Unassigned, and barcode mismatches; no additional probability threshold",
        "model": "VarianceThreshold, fold-local SelectKBest(f_classif, k=min(top_features, feature_count)), and L2-regularized LogisticRegression(C=0.1, class_weight=balanced)",
        "top_features": args.top_features,
        "cv": f"{args.folds}-fold stratified cross-validation over donor-level profiles; random_state={args.seed}",
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "weighted_f1": float(f1_score(y, predictions, average="weighted", zero_division=0)),
        "confusion_matrix_labels_order": labels,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "source_matrices": all_stats,
        "folds": fold_details,
        "limitations": [
            "Only eight independent donors are available, so cross-validation metrics are highly unstable.",
            "This is a COVID-versus-healthy pilot, not a multi-disease classifier.",
            "There is no external cohort validation; results do not establish clinical performance.",
        ],
    }

    with (args.output_dir / "fold_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["donor_id", "true_label", "predicted_label", "probability_covid", "fold"])
        for index, donor in enumerate(donor_ids):
            writer.writerow(
                [
                    donor,
                    donor_metadata[donor][1],
                    labels[predictions[index]],
                    f"{probabilities[index]:.8f}",
                    int(fold_indices[index]),
                ]
            )
    (args.output_dir / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Healthy", "COVID"])
    display.plot(cmap="Blues", colorbar=False, values_format="d")
    plt.title(f"Donor-level {args.folds}-fold cross-validation")
    plt.tight_layout()
    plt.savefig(args.output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    final_pipeline = build_pipeline(expression.shape[1], args.top_features)
    final_pipeline.fit(expression, y)
    joblib.dump(
        {
            "pipeline": final_pipeline,
            "feature_ids": feature_ids,
            "donor_ids_used_for_final_fit": donor_ids,
            "class_labels": labels,
            "normalization": "CPM then log1p",
        },
        args.model_dir / "covid_vs_healthy_logistic_regression.joblib",
        compress=3,
    )

    print(f"Donors: {len(donor_ids)} ({dict(class_counts)})")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.3f}")
    print("Confusion matrix (rows=true, columns=predicted; healthy, COVID):")
    print(matrix)
    print(f"Metrics: {args.output_dir / 'baseline_metrics.json'}")
    print(f"Predictions: {args.output_dir / 'fold_predictions.csv'}")
    print(f"Model: {args.model_dir / 'covid_vs_healthy_logistic_regression.joblib'}")


if __name__ == "__main__":
    main()
