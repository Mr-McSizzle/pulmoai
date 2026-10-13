#!/usr/bin/env python3
"""Build Melms et al. (GSE171524) pseudobulk profiles.

Downloads/verifies Melms H5AD, verifies integer raw counts in adata.raw.X,
aggregates raw counts per donor (grouping L04covaddon with L04cov into 26 true donors),
deletes temporary H5AD immediately, applies HGNC alias remapping (no zero-padding),
identifies COMMON GENE SET across 16 atlas studies + Melms, normalizes via
CPM -> log1p, and saves to data/processed/melms_profiles.npz and metadata CSV.
"""

from __future__ import annotations

import os
import resource
import shutil
import sys
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TEMP_DIR = DATA_DIR / "raw" / "census_temp"
ATLAS_NPZ = PROCESSED_DIR / "allcells_pseudobulk_counts.npz"
HGNC_TXT = DATA_DIR / "raw" / "hgnc_complete_set.txt"
H5AD_PATH = TEMP_DIR / "melms_d8da613f.h5ad"

OUTPUT_NPZ = PROCESSED_DIR / "melms_profiles.npz"
OUTPUT_META_CSV = PROCESSED_DIR / "melms_metadata.csv"

INTEGER_COUNT_TOLERANCE = 1e-5


def check_disk(min_free_gb: float = 3.0) -> float:
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"ABORT: Free disk space ({free_gb:.2f} GiB) is below minimum {min_free_gb} GiB threshold!"
        )
    return free_gb


def get_peak_ram_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def build_hgnc_clean_maps() -> dict[str, str]:
    print("Loading HGNC complete set for alias remapping...")
    hgnc = pd.read_csv(HGNC_TXT, sep="\t", low_memory=False)
    prev_map: dict[str, set[str]] = {}
    for _, row in hgnc.iterrows():
        curr = str(row["symbol"]).strip()
        if pd.notna(row["prev_symbol"]):
            for p in str(row["prev_symbol"]).split("|"):
                p = p.strip().strip('"')
                if p:
                    prev_map.setdefault(p, set()).add(curr)
        if pd.notna(row["alias_symbol"]):
            for al in str(row["alias_symbol"]).split("|"):
                al = al.strip().strip('"')
                if al:
                    prev_map.setdefault(al, set()).add(curr)

    # Keep only unambiguous 1:1 mappings
    clean_map = {k: list(v)[0] for k, v in prev_map.items() if len(v) == 1}
    print(f"Loaded {len(clean_map):,} unambiguous 1:1 HGNC aliases.")
    return clean_map


def main() -> None:
    t_start = time.time()
    free_gb = check_disk(3.0)
    print(f"[Resource Check] Initial free disk space: {free_gb:.2f} GiB")

    if not H5AD_PATH.exists():
        raise FileNotFoundError(f"H5AD file not found at {H5AD_PATH}. Ensure download is complete.")

    h5ad_size_mb = H5AD_PATH.stat().st_size / (1024 * 1024)
    print(f"Found Melms H5AD: {h5ad_size_mb:.2f} MB")

    # 1. Open in backed mode
    print("\n[Step 1] Opening Melms H5AD in backed mode ('r')...")
    adata = anndata.read_h5ad(H5AD_PATH, backed="r")
    print(f"  Shape: {adata.shape[0]:,} cells x {adata.shape[1]:,} features")

    # 2. Check adata.raw.X vs adata.X
    use_raw = False
    if adata.raw is not None:
        raw_var = adata.raw.var
        raw_X = adata.raw.X
        if "feature_name" in raw_var.columns:
            melms_symbols = raw_var["feature_name"].astype(str).values
        else:
            melms_symbols = raw_var.index.astype(str).values
        use_raw = True
        print(f"  adata.raw present: shape {adata.raw.shape}")
    else:
        raw_var = adata.var
        raw_X = adata.X
        if "feature_name" in raw_var.columns:
            melms_symbols = raw_var["feature_name"].astype(str).values
        else:
            melms_symbols = raw_var.index.astype(str).values
        print("  adata.raw NOT present, using adata.X")

    # Verify integer counts on sample slice
    sample_slice = raw_X[:500]
    sample_data = sample_slice.data if hasattr(sample_slice, "data") else sample_slice
    is_int = np.all(np.equal(np.mod(sample_data, 1), 0))
    if not is_int and use_raw:
        print("  adata.raw.X had floats; checking adata.X...")
        sample_slice2 = adata.X[:500]
        sample_data2 = sample_slice2.data if hasattr(sample_slice2, "data") else sample_slice2
        if np.all(np.equal(np.mod(sample_data2, 1), 0)):
            raw_X = adata.X
            if "feature_name" in adata.var.columns:
                melms_symbols = adata.var["feature_name"].astype(str).values
            else:
                melms_symbols = adata.var.index.astype(str).values
            is_int = True
            use_raw = False

    if not is_int:
        raise ValueError("ABORT: Melms counts are not strictly integers!")
    print(f"  Integer count verification: PASSED (source: {'adata.raw.X' if use_raw else 'adata.X'}).")

    # 3. Donor resolution & mapping
    obs_df = adata.obs[["donor_id", "disease", "tissue", "assay", "suspension_type"]].copy()
    raw_donor_count = obs_df["donor_id"].nunique()
    print(f"\n[Step 2] Resolving donors. Raw donor_id unique count: {raw_donor_count}")

    # Group L04covaddon with L04cov
    addon_mask = (obs_df["donor_id"] == "L04covaddon")
    n_addon = addon_mask.sum()
    print(f"  L04covaddon cells: {n_addon:,}. Grouping into donor 'L04cov'...")
    obs_df.loc[addon_mask, "donor_id"] = "L04cov"

    unique_donors = sorted(obs_df["donor_id"].unique().tolist())
    n_true_donors = len(unique_donors)
    print(f"  True unique donors: {n_true_donors}")

    # Map disease labels:
    # COVID-19 -> COVID19, normal -> Donor
    disease_map = {"COVID-19": "COVID19", "normal": "Donor"}

    # 4. Aggregate counts per donor
    print(f"\n[Step 3] Aggregating raw counts for {n_true_donors} donors...")
    n_melms_genes = len(melms_symbols)
    aggregated_raw = np.zeros((n_true_donors, n_melms_genes), dtype=np.float64)

    donor_meta_rows = []
    for d_idx, donor in enumerate(unique_donors):
        donor_cell_mask = (obs_df["donor_id"] == donor).values
        cell_indices = np.where(donor_cell_mask)[0]
        n_cells = len(cell_indices)

        sub_obs = obs_df[donor_cell_mask].iloc[0]
        raw_dis = str(sub_obs["disease"])
        mapped_label = disease_map.get(raw_dis)
        if mapped_label is None:
            raise ValueError(f"Unexpected disease label for donor {donor}: {raw_dis}")

        # Slice backed matrix for this donor
        donor_slice = raw_X[cell_indices, :]
        donor_sum = np.array(donor_slice.sum(axis=0)).flatten()
        aggregated_raw[d_idx, :] = donor_sum

        lib_size = float(np.sum(donor_sum))
        donor_meta_rows.append({
            "profile_id": f"GSE171524_{donor}",
            "donor_id": donor,
            "study_name": "GSE171524",
            "mapped_class": mapped_label,
            "original_disease": raw_dis,
            "unit": "donor",
            "n_cells": n_cells,
            "library_size": lib_size,
            "suspension_type": str(sub_obs.get("suspension_type", "nucleus")),
            "assay": str(sub_obs.get("assay", "10x 3' v3")),
        })
        print(f"  Donor {donor:<12} ({mapped_label:<7}): {n_cells:>6,} cells, {lib_size:>10,.0f} UMIs")

    # Close backed file and delete immediately
    del raw_X
    adata.file.close()
    del adata
    print(f"\nDeleting temporary file {H5AD_PATH}...")
    H5AD_PATH.unlink()
    print(f"Temporary file deleted. Free disk space: {check_disk(3.0):.2f} GiB")
    print(f"Current peak RAM: {get_peak_ram_mb():.1f} MB")

    # 5. Apply HGNC Alias Remapping (No Zero-Padding)
    print("\n[Step 4] Applying HGNC legacy alias remapping to Melms genes...")
    hgnc_map = build_hgnc_clean_maps()

    # Build unique remapped Melms gene matrix
    remapped_symbols = []
    remapping_changes = 0
    for sym in melms_symbols:
        clean_sym = sym.strip()
        if clean_sym in hgnc_map:
            remapped_symbols.append(hgnc_map[clean_sym])
            remapping_changes += 1
        else:
            remapped_symbols.append(clean_sym)

    print(f"  Remapped {remapping_changes:,} gene symbols via HGNC 1:1 aliases.")

    # Combine duplicate gene columns if any alias mapped to an existing symbol
    unique_remap_symbols, inverse_indices = np.unique(remapped_symbols, return_inverse=True)
    n_unique_melms = len(unique_remap_symbols)
    print(f"  Unique Melms gene symbols after remapping: {n_unique_melms:,} (from {n_melms_genes:,})")

    melms_counts_dedup = np.zeros((n_true_donors, n_unique_melms), dtype=np.float64)
    for orig_col, new_col in enumerate(inverse_indices):
        melms_counts_dedup[:, new_col] += aggregated_raw[:, orig_col]

    melms_gene_map = {sym: idx for idx, sym in enumerate(unique_remap_symbols)}

    # 6. Define COMMON GENE SET
    print("\n[Step 5] Defining COMMON GENE SET across 16 Atlas Studies + Melms...")
    with np.load(ATLAS_NPZ, allow_pickle=True) as a:
        atlas_counts = a["counts"]
        atlas_studies = a["studies"].astype(str)
        atlas_features = a["feature_names"].astype(str)

    unique_atlas_studies = np.unique(atlas_studies)
    print(f"  Atlas has {len(atlas_features):,} genes across {len(unique_atlas_studies)} studies.")

    # Check which genes are measured in all 16 atlas studies (count > 0 in each study)
    atlas_study_measured = []
    for s in unique_atlas_studies:
        s_mask = (atlas_studies == s)
        s_counts = atlas_counts[s_mask]
        atlas_study_measured.append(s_counts.sum(axis=0) > 0)

    atlas_all_16_mask = np.all(atlas_study_measured, axis=0)
    atlas_all_16_genes = set(atlas_features[atlas_all_16_mask])
    print(f"  Genes measured in all 16 atlas studies (count > 0): {len(atlas_all_16_genes):,}")

    # Melms measured genes (count > 0 in Melms)
    melms_measured_mask = (melms_counts_dedup.sum(axis=0) > 0)
    melms_measured_genes = set(unique_remap_symbols[melms_measured_mask])
    print(f"  Genes measured in Melms (count > 0): {len(melms_measured_genes):,}")

    # Intersection = COMMON GENE SET
    common_genes = sorted(list(atlas_all_16_genes.intersection(melms_measured_genes)))
    n_common = len(common_genes)
    print(f"\n  >>> COMMON GENE SET SIZE: {n_common:,} genes <<<")

    # Also report pure symbol intersection (all 30,566 atlas genes that are in Melms)
    atlas_full_set = set(atlas_features)
    full_symbol_overlap = sorted(list(atlas_full_set.intersection(set(unique_remap_symbols))))
    print(f"  (Context: All Atlas genes present in Melms symbol space: {len(full_symbol_overlap):,} genes)")

    # 7. Normalize Melms: CPM -> log1p
    print("\n[Step 6] Normalizing Melms donor profiles (CPM -> log1p)...")
    melms_common_indices = [melms_gene_map[g] for g in common_genes]
    melms_counts_common = melms_counts_dedup[:, melms_common_indices]

    # Melms total library size per donor (computed on full measured counts)
    donor_lib_sizes = melms_counts_dedup.sum(axis=1, keepdims=True)
    cpm = (melms_counts_common / np.maximum(donor_lib_sizes, 1.0)) * 1e6
    melms_expression_common = np.log1p(cpm).astype(np.float32)

    # 8. Save melms_profiles.npz and metadata CSV
    print(f"\n[Step 7] Saving Melms profiles to {OUTPUT_NPZ}...")
    donor_labels = [row["mapped_class"] for row in donor_meta_rows]
    donor_cells = [row["n_cells"] for row in donor_meta_rows]
    donor_ids = [row["donor_id"] for row in donor_meta_rows]
    studies = ["GSE171524"] * n_true_donors

    # Update metadata with common gene coverage
    for i, row in enumerate(donor_meta_rows):
        row["measured_common_genes"] = int((melms_counts_common[i] > 0).sum())
        row["common_genes_fraction"] = round(float((melms_counts_common[i] > 0).sum()) / n_common, 4)

    meta_df = pd.DataFrame(donor_meta_rows)
    meta_df.to_csv(OUTPUT_META_CSV, index=False)
    print(f"Saved metadata to {OUTPUT_META_CSV}")

    np.savez_compressed(
        OUTPUT_NPZ,
        counts=melms_counts_common.astype(np.float64),
        expression=melms_expression_common,
        feature_names=np.array(common_genes, dtype=object),
        donors=np.array(donor_ids, dtype=object),
        studies=np.array(studies, dtype=object),
        labels=np.array(donor_labels, dtype=object),
        cell_counts=np.array(donor_cells, dtype=np.int64),
        library_sizes=donor_lib_sizes.flatten(),
        suspension_type=np.array(["nucleus"] * n_true_donors, dtype=object),
        assay=np.array(["10x 3' v3"] * n_true_donors, dtype=object),
        unit=np.array(["donor"] * n_true_donors, dtype=object),
    )
    print(f"Saved {OUTPUT_NPZ} ({OUTPUT_NPZ.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Donors summary: {pd.Series(donor_labels).value_counts().to_dict()}")
    print(f"Peak RAM: {get_peak_ram_mb():.1f} MB | Elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
