#!/usr/bin/env python3
"""Inspect H5AD structure and observation metadata without loading expression data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


KEYWORDS = (
    "disease", "condition", "diagnos", "status", "patient", "donor", "subject",
    "sample", "study", "cohort", "cell_type", "celltype", "annotation", "tissue",
    "species", "author", "orig.ident", "gse_alias", "gse", "geo", "accession",
    "dataset", "ident",
)
CELL_ID_TOKENS = ("barcode", "cell_id", "cellid", "cell_name")


def json_value(value):
    if value is None:
        return None
    try:
        if bool(value != value):
            return "<missing>"
    except (TypeError, ValueError):
        pass
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "reports"
        / "atlas_h5ad_inspection.json",
    )
    args = parser.parse_args()
    if not args.h5ad.is_file():
        parser.error(f"H5AD file does not exist: {args.h5ad}")

    try:
        import anndata as ad
    except ImportError as error:
        raise SystemExit(
            "Missing dependency 'anndata'. Install it in the project environment with "
            "`.venv/bin/python -m pip install anndata`, then rerun this script."
        ) from error

    adata = ad.read_h5ad(args.h5ad, backed="r")
    try:
        obs_columns = []
        candidates = {}
        for column in adata.obs.columns:
            series = adata.obs[column]
            name = str(column)
            column_info = {"name": name, "dtype": str(series.dtype)}
            lowered = name.lower()
            if any(keyword in lowered for keyword in KEYWORDS):
                if any(token in lowered for token in CELL_ID_TOKENS):
                    column_info["inspection"] = "skipped high-cardinality cell identifier"
                else:
                    nonmissing = series.dropna()
                    unique_count = int(nonmissing.nunique())
                    column_info["missing"] = int(series.isna().sum())
                    column_info["unique_nonmissing"] = unique_count
                    if unique_count <= 1000:
                        frequencies = series.value_counts(dropna=False)
                        column_info["top_values"] = [
                            {"value": json_value(value), "count": int(count)}
                            for value, count in frequencies.items()
                        ]
                    elif unique_count <= 10000:
                        frequencies = series.value_counts(dropna=False)
                        column_info["top_values"] = [
                            {"value": json_value(value), "count": int(count)}
                            for value, count in frequencies.head(25).items()
                        ]
                    else:
                        column_info["top_values"] = "skipped: more than 10,000 unique values"
                    candidates[name] = column_info
            obs_columns.append(column_info)

        matrix = adata.X
        matrix_description = {
            "shape": [int(value) for value in matrix.shape],
            "backed_type": f"{type(matrix).__module__}.{type(matrix).__name__}",
            "dtype": str(getattr(matrix, "dtype", "unknown")),
            "loaded_into_memory": False,
        }
        sampled_rows = np.linspace(0, adata.n_obs - 1, min(128, adata.n_obs), dtype=np.int64)
        sampled_matrix = matrix[sampled_rows]
        if sparse.issparse(sampled_matrix):
            sampled_values = sampled_matrix.data
            sampled_shape = sampled_matrix.shape
            nonzero_count = int(np.count_nonzero(sampled_values))
        else:
            sampled_array = np.asarray(sampled_matrix)
            sampled_values = sampled_array.ravel()
            sampled_shape = sampled_array.shape
            nonzero_count = int(np.count_nonzero(sampled_values))
        nonzero_values = np.asarray(sampled_values)[np.asarray(sampled_values) != 0]
        sampled_slots = int(np.prod(sampled_shape))
        if nonzero_values.size:
            quantiles = np.percentile(nonzero_values, [50, 90, 99, 99.9])
            expression_sample = {
                "rows_sampled": int(len(sampled_rows)),
                "nonzero_values_sampled": int(nonzero_values.size),
                "nonzero_fraction": float(nonzero_count / sampled_slots),
                "minimum_nonzero": float(np.min(nonzero_values)),
                "median_nonzero": float(quantiles[0]),
                "p90_nonzero": float(quantiles[1]),
                "p99_nonzero": float(quantiles[2]),
                "p99_9_nonzero": float(quantiles[3]),
                "maximum_nonzero": float(np.max(nonzero_values)),
                "integer_like_fraction": float(np.mean(np.isclose(nonzero_values, np.rint(nonzero_values)))),
                "negative_value_count": int(np.count_nonzero(nonzero_values < 0)),
            }
        else:
            expression_sample = {"rows_sampled": int(len(sampled_rows)), "nonzero_values_sampled": 0}
        raw = getattr(adata, "raw", None)
        report = {
            "source_file": str(args.h5ad),
            "file_size_bytes": args.h5ad.stat().st_size,
            "file_size_gib": round(args.h5ad.stat().st_size / (1024 ** 3), 3),
            "backed_mode": "read-only",
            "n_obs_cells": int(adata.n_obs),
            "n_vars_features": int(adata.n_vars),
            "obs_columns": obs_columns,
            "candidate_metadata_columns": candidates,
            "var_columns": [str(column) for column in adata.var.columns],
            "var_names_first_10": [str(value) for value in adata.var_names[:10]],
            "var_names_last_10": [str(value) for value in adata.var_names[-10:]],
            "var_names_unique": bool(adata.var_names.is_unique),
            "layer_names": [str(name) for name in adata.layers.keys()],
            "has_raw_slot": raw is not None,
            "raw_var_columns": [str(column) for column in raw.var.columns] if raw is not None else [],
            "uns_keys": [str(key) for key in adata.uns.keys()],
            "expression_matrix": matrix_description,
            "expression_sample_summary": expression_sample,
            "expression_scale": (
                "inference must be based on source documentation and the sampled value summary; "
                "do not treat the sample alone as proof"
            ),
            "training_status": (
                "not handled by this inspection script; consult "
                "reports/allcells_atlas/analysis_summary.json for training status"
            ),
        }
    finally:
        adata.file.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "cells": report["n_obs_cells"],
        "features": report["n_vars_features"],
        "matrix_type": matrix_description["backed_type"],
        "candidate_columns": list(candidates),
        "training_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
