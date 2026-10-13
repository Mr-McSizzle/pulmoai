#!/usr/bin/env python3
"""Validate atlas identifiers, disease labels, profile grouping, and model features."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LABELS = ("Donor", "COPD", "PF", "COVID19")
GROUPING_FIELDS = ("Disease", "gse_alias", "orig.ident")
CHUNK_ROWS = 50_000


def category_column(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    codes = np.asarray(group["codes"][:], dtype=np.int32)
    categories = np.asarray(group["categories"].asstr()[:], dtype=str)
    return codes, categories


def count_cell_identifier_duplicates(
    cell_ids: h5py.Dataset,
    study_codes: np.ndarray,
) -> dict[str, int]:
    blank_ids = 0
    inserted = 0
    with tempfile.TemporaryDirectory(prefix="pulmoai_metadata_validation_") as temp_dir:
        database = sqlite3.connect(Path(temp_dir) / "cell_ids.sqlite")
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute(
            "CREATE TABLE cell_ids (study INTEGER NOT NULL, cell_id TEXT NOT NULL, "
            "PRIMARY KEY (study, cell_id)) WITHOUT ROWID"
        )
        for start in range(0, len(study_codes), CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, len(study_codes))
            values = cell_ids.asstr()[start:end]
            rows = []
            for study_code, value in zip(study_codes[start:end], values):
                cell_id = str(value).strip()
                if not cell_id:
                    blank_ids += 1
                    continue
                rows.append((int(study_code), cell_id))
            prior_changes = database.total_changes
            database.executemany("INSERT OR IGNORE INTO cell_ids VALUES (?, ?)", rows)
            inserted += database.total_changes - prior_changes
        database.commit()
        duplicate_study_cell_pairs = len(study_codes) - blank_ids - inserted
        identifiers_reused_between_studies = int(database.execute(
            "SELECT COUNT(*) FROM (SELECT cell_id FROM cell_ids GROUP BY cell_id HAVING COUNT(*) > 1)"
        ).fetchone()[0])
        database.close()
    return {
        "missing_or_blank_cell_ids": int(blank_ids),
        "duplicate_cell_ids_within_study": int(duplicate_study_cell_pairs),
        "unique_study_cell_id_pairs": int(inserted),
        "cell_ids_reused_across_studies": int(identifiers_reused_between_studies),
    }


def validate(args: argparse.Namespace) -> dict:
    if not args.h5ad.is_file():
        raise FileNotFoundError(f"H5AD file not found: {args.h5ad}")
    if not args.profiles.is_file():
        raise FileNotFoundError(f"processed profiles not found: {args.profiles}")
    if not args.model.is_file():
        raise FileNotFoundError(f"model bundle not found: {args.model}")

    with h5py.File(args.h5ad, "r") as h5ad:
        obs = h5ad["obs"]
        obs_fields = [str(name) for name in obs.keys()]
        index_field = str(obs.attrs.get("_index", "_index"))
        required = (*GROUPING_FIELDS, "celltypeuse", index_field)
        missing_fields = sorted(set(required) - set(obs_fields))
        if missing_fields:
            raise ValueError(f"required observation fields are absent: {missing_fields}")

        n_cells = int(obs[index_field].shape[0])
        codes_by_field = {}
        categories_by_field = {}
        metadata_field_summary = {}
        for field in (*GROUPING_FIELDS, "celltypeuse"):
            codes, categories = category_column(obs[field])
            if len(codes) != n_cells:
                raise ValueError(f"{field} length does not match cell identifier count")
            missing_count = int(np.count_nonzero(codes < 0))
            if field in GROUPING_FIELDS and missing_count:
                raise ValueError(f"{field} contains {missing_count} missing category codes")
            counts = np.bincount(codes[codes >= 0], minlength=len(categories))
            metadata_field_summary[field] = {
                "category_count": int(len(categories)),
                "missing_cells": missing_count,
                "category_counts": {
                    str(category): int(count) for category, count in zip(categories, counts)
                },
            }
            codes_by_field[field] = codes
            categories_by_field[field] = categories

        label_names = categories_by_field["Disease"].tolist()
        if set(label_names) != set(EXPECTED_LABELS):
            raise ValueError(f"expected disease categories {EXPECTED_LABELS}; found {label_names}")
        study_codes = codes_by_field["gse_alias"]
        orig_codes = codes_by_field["orig.ident"]
        disease_codes = codes_by_field["Disease"]
        orig_count = len(categories_by_field["orig.ident"])
        group_keys = study_codes.astype(np.int64) * orig_count + orig_codes
        unique_group_keys, cell_to_group = np.unique(group_keys, return_inverse=True)
        disease_count_by_group = np.bincount(
            cell_to_group * len(label_names) + disease_codes,
            minlength=len(unique_group_keys) * len(label_names),
        ).reshape(len(unique_group_keys), len(label_names))
        inconsistent_groups = int(np.count_nonzero((disease_count_by_group > 0).sum(axis=1) != 1))
        if inconsistent_groups:
            raise ValueError(f"{inconsistent_groups} study-scoped sample groups have inconsistent disease labels")

        label_codes = disease_count_by_group.argmax(axis=1)
        group_counts = Counter(categories_by_field["Disease"][label_codes].tolist())
        study_names = categories_by_field["gse_alias"]
        study_group_counts: dict[str, Counter] = defaultdict(Counter)
        for group_key, label_code in zip(unique_group_keys, label_codes):
            study_code = int(group_key // orig_count)
            study_group_counts[str(study_names[study_code])][str(label_names[label_code])] += 1
        cell_identifier_summary = count_cell_identifier_duplicates(obs[index_field], study_codes)
        var_names = h5ad["var"][str(h5ad["var"].attrs.get("_index", "_index"))].asstr()[:].astype(str)
        unique_feature_count = len(set(var_names.tolist()))
        blank_feature_count = int(sum(not name.strip() for name in var_names))
        if unique_feature_count != len(var_names):
            raise ValueError("source feature identifiers are not unique")
        if blank_feature_count:
            raise ValueError(f"source contains {blank_feature_count} blank feature identifiers")

    with np.load(args.profiles, allow_pickle=False) as profiles:
        cached_counts = profiles["counts"]
        cached_sample_ids = profiles["sample_ids"].astype(str)
        cached_studies = profiles["studies"].astype(str)
        cached_labels = profiles["labels"].astype(str)
        cached_features = profiles["feature_names"].astype(str)
    if cached_counts.shape != (len(cached_sample_ids), len(cached_features)):
        raise ValueError("cached count matrix dimensions do not match metadata arrays")
    if len(cached_sample_ids) != len(unique_group_keys):
        raise ValueError("cached profile count does not match H5AD study-scoped sample groups")

    bundle = joblib.load(args.model)
    bundle_features = np.asarray(bundle.get("feature_names", []), dtype=str)
    feature_order_matches_cache = bool(np.array_equal(var_names, cached_features))
    feature_order_matches_model = bool(np.array_equal(cached_features, bundle_features))
    if not feature_order_matches_cache or not feature_order_matches_model:
        raise ValueError("source, cached, and model feature orders do not match exactly")

    cached_group_labels = Counter(cached_labels.tolist())
    if cached_group_labels != group_counts:
        raise ValueError("cached profile disease counts do not match H5AD-derived group labels")
    if Counter(cached_studies.tolist()) != Counter(
        {study: sum(counts.values()) for study, counts in study_group_counts.items()}
    ):
        raise ValueError("cached profile study counts do not match H5AD-derived study groups")

    patient_id_fields = [
        field for field in obs_fields
        if field.lower() in {"patient_id", "patient", "subject_id", "donor_id", "biological_sample_id"}
    ]
    return {
        "status": "passed_with_identity_limitation",
        "source_file": str(args.h5ad),
        "source_file_size_bytes": int(args.h5ad.stat().st_size),
        "matrix_shape_cells_by_features": [int(n_cells), int(len(var_names))],
        "obs_fields": obs_fields,
        "metadata_fields": metadata_field_summary,
        "cell_identifier_validation": cell_identifier_summary,
        "feature_validation": {
            "feature_count": int(len(var_names)),
            "unique_feature_count": int(unique_feature_count),
            "blank_feature_names": blank_feature_count,
            "matches_cached_feature_order": feature_order_matches_cache,
            "matches_model_feature_order": feature_order_matches_model,
        },
        "sample_grouping": {
            "key": "gse_alias + orig.ident",
            "profile_count": int(len(unique_group_keys)),
            "groups_with_inconsistent_disease_labels": inconsistent_groups,
            "group_counts_by_disease": {label: int(group_counts.get(label, 0)) for label in EXPECTED_LABELS},
            "group_counts_by_study_and_disease": {
                study: dict(counts) for study, counts in sorted(study_group_counts.items())
            },
        },
        "identity_limitation": {
            "explicit_patient_or_biological_sample_id_fields": patient_id_fields,
            "verified_patient_count": None,
            "note": "orig.ident is study-scoped source sample/library metadata; no explicit verified biological patient ID is present in .obs.",
        },
        "model_input_contract": {
            "model_name": bundle.get("model_name"),
            "class_labels": bundle.get("label_order"),
            "feature_count": int(len(bundle_features)),
            "required_representation": bundle.get("normalization"),
            "prediction_input_note": bundle.get("model_input_contract"),
            "preprocessing_inside_saved_pipeline": list(bundle["pipeline"].named_steps),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5ad", type=Path,
        default=Path.home() / "Downloads" / "atlas-001.h5ad",
    )
    parser.add_argument(
        "--profiles", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "allcells_pseudobulk_counts.npz",
    )
    parser.add_argument(
        "--model", type=Path,
        default=PROJECT_ROOT / "models" / "ensemble_benchmark_best.joblib",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "reports" / "atlas_metadata_validation.json",
    )
    args = parser.parse_args()
    try:
        report = validate(args)
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        parser.error(f"metadata validation failed: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "cells": report["matrix_shape_cells_by_features"][0],
        "features": report["matrix_shape_cells_by_features"][1],
        "sample_groups": report["sample_grouping"]["profile_count"],
        "group_labels_consistent": report["sample_grouping"]["groups_with_inconsistent_disease_labels"] == 0,
        "feature_order_matches_model": report["feature_validation"]["matches_model_feature_order"],
        "verified_patient_count": report["identity_limitation"]["verified_patient_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
