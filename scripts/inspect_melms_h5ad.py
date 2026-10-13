#!/usr/bin/env python3
"""Download and inspect Melms et al. H5AD dataset."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import anndata
import cellxgene_census
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = PROJECT_ROOT / "data" / "raw" / "census_temp"
TEMP_H5AD = TEMP_DIR / "melms_d8da613f.h5ad"

DATASET_ID = "d8da613f-e681-4c69-b463-e94f5e66847f"

def check_disk_space(min_gb: float = 3.0) -> None:
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / (1024 ** 3)
    print(f"Disk check: {free_gb:.2f} GB free (required: >{min_gb} GB)")
    if free_gb < min_gb:
        raise RuntimeError(f"ABORT: Free disk space ({free_gb:.2f} GB) is below minimum {min_gb} GB threshold!")

def main() -> None:
    check_disk_space(3.0)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if not TEMP_H5AD.exists():
        print(f"Downloading {DATASET_ID} via cellxgene_census.download_source_h5ad...")
        cellxgene_census.download_source_h5ad(DATASET_ID, to_path=str(TEMP_H5AD), progress_bar=True)
        print(f"Downloaded to {TEMP_H5AD} ({TEMP_H5AD.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"Found existing {TEMP_H5AD} ({TEMP_H5AD.stat().st_size / (1024*1024):.2f} MB)")

    check_disk_space(3.0)

    print("\nOpening H5AD in backed mode ('r')...")
    adata = anndata.read_h5ad(TEMP_H5AD, backed="r")
    print(f"AnnData shape: {adata.shape}")
    print(f"obs columns: {list(adata.obs.columns)}")
    print(f"var columns: {list(adata.var.columns)}")
    print(f"adata.raw is None: {adata.raw is None}")
    if adata.raw is not None:
        print(f"adata.raw.shape: {adata.raw.shape}")
        print(f"adata.raw.var columns: {list(adata.raw.var.columns)}")

    # Check a small slice of X or raw.X to verify integer counts
    if adata.raw is not None:
        raw_slice = adata.raw.X[:5, :10]
        if hasattr(raw_slice, "toarray"):
            raw_slice = raw_slice.toarray()
        print(f"adata.raw.X slice:\n{raw_slice}")
        is_int_raw = np.all(np.equal(np.mod(raw_slice.data if hasattr(raw_slice, "data") else raw_slice, 1), 0))
        print(f"adata.raw.X is integer counts: {is_int_raw}")

    x_slice = adata.X[:5, :10]
    if hasattr(x_slice, "toarray"):
        x_slice = x_slice.toarray()
    print(f"adata.X slice:\n{x_slice}")
    is_int_x = np.all(np.equal(np.mod(x_slice.data if hasattr(x_slice, "data") else x_slice, 1), 0))
    print(f"adata.X is integer counts: {is_int_x}")

    # Inspect donor IDs and L04cov / L04covaddon
    obs = adata.obs[["donor_id", "disease", "tissue", "assay", "suspension_type"]].copy()
    print("\nDonor value counts:")
    donor_counts = obs.groupby(["donor_id", "disease"]).size().reset_index(name="n_cells")
    print(donor_counts.to_string())

    adata.file.close()

if __name__ == "__main__":
    main()
