#!/usr/bin/env python3
"""Stream the full backed H5AD matrix and audit values before count normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parents[1] / "reports" / "atlas_expression_scale.json",
    )
    parser.add_argument("--chunk-rows", type=int, default=2048)
    args = parser.parse_args()
    if not args.h5ad.is_file():
        parser.error(f"H5AD does not exist: {args.h5ad}")

    adata = ad.read_h5ad(args.h5ad, backed="r")
    try:
        if "gse_alias" not in adata.obs:
            raise ValueError("required study column 'gse_alias' is absent")
        study_categories = [str(value) for value in adata.obs["gse_alias"].cat.categories]
        study_codes = adata.obs["gse_alias"].cat.codes.to_numpy(dtype=np.int16, copy=True)
        nonzero_total = 0
        finite_errors = 0
        negative_total = 0
        threshold_counts = {str(threshold): 0 for threshold in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5)}
        fractional_samples = []
        study_nonzero = np.zeros(len(study_categories), dtype=np.int64)
        study_fractional = np.zeros(len(study_categories), dtype=np.int64)
        fractional_values = []
        examples = []
        min_value = float("inf")
        max_value = float("-inf")

        for chunk_number, start in enumerate(range(0, adata.n_obs, args.chunk_rows), start=1):
            end = min(start + args.chunk_rows, adata.n_obs)
            block = adata.X[start:end]
            if not sparse.issparse(block):
                block = sparse.csr_matrix(np.asarray(block))
            else:
                block = sparse.csr_matrix(block)
            values = block.data
            if not values.size:
                continue
            row_indices = np.repeat(np.arange(end - start), np.diff(block.indptr))
            studies_for_entries = study_codes[start:end][row_indices]
            deviations = np.abs(values - np.rint(values))
            nonzero_total += int(values.size)
            finite_errors += int((~np.isfinite(values)).sum())
            negative_total += int((values < 0).sum())
            min_value = min(min_value, float(np.min(values)))
            max_value = max(max_value, float(np.max(values)))
            np.add.at(study_nonzero, studies_for_entries, 1)
            for threshold in threshold_counts:
                mask = deviations > float(threshold)
                threshold_counts[threshold] += int(mask.sum())
                if threshold == "1e-05":
                    np.add.at(study_fractional, studies_for_entries[mask], 1)
                    if mask.any():
                        fractional_values.extend(values[mask][:1000].tolist())
                        fractional_samples.extend(deviations[mask][:1000].tolist())
                        if len(examples) < 30:
                            chosen = np.flatnonzero(mask)[:30 - len(examples)]
                            for value_index in chosen:
                                row_index = int(row_indices[value_index])
                                column_index = int(block.indices[value_index])
                                examples.append({
                                    "study": study_categories[int(study_codes[start + row_index])],
                                    "value": float(values[value_index]),
                                    "distance_to_integer": float(deviations[value_index]),
                                    "gene": str(adata.var_names[column_index]),
                                })
            if chunk_number == 1 or chunk_number % 100 == 0:
                print(f"Audited {end:,}/{adata.n_obs:,} cells", flush=True)

        study_rows = []
        for index, study_name in enumerate(study_categories):
            study_rows.append({
                "study": study_name,
                "stored_nonzero_values": int(study_nonzero[index]),
                "values_not_integer_like_at_1e-5": int(study_fractional[index]),
                "fractional_percent": round(
                    100 * study_fractional[index] / max(1, study_nonzero[index]), 6
                ),
            })
        study_rows.sort(key=lambda row: row["values_not_integer_like_at_1e-5"], reverse=True)
        report = {
            "source_file": str(args.h5ad),
            "shape": [int(adata.n_obs), int(adata.n_vars)],
            "backed_mode": "read-only",
            "stored_nonzero_values": nonzero_total,
            "finite_value_errors": finite_errors,
            "negative_values": negative_total,
            "minimum_stored_value": min_value,
            "maximum_stored_value": max_value,
            "absolute_distance_from_nearest_integer_counts": threshold_counts,
            "fractional_value_distribution_sample": (
                {
                    "sampled_values": len(fractional_values),
                    "value_quantiles": np.percentile(fractional_values, [0, 1, 10, 50, 90, 99, 100]).tolist(),
                    "distance_quantiles": np.percentile(fractional_samples, [0, 1, 10, 50, 90, 99, 100]).tolist(),
                } if fractional_values else None
            ),
            "examples_not_integer_like_at_1e-5": examples,
            "study_distribution": study_rows,
            "interpretation": (
                "Full-matrix scale audit only. Non-integer values require source/provenance review before "
                "treating the matrix as raw counts or applying count-based normalization."
            ),
        }
    finally:
        adata.file.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "nonzero_values": report["stored_nonzero_values"],
        "fractional_at_1e-5": threshold_counts["1e-05"],
        "fractional_by_study": [
            row for row in study_rows if row["values_not_integer_like_at_1e-5"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
