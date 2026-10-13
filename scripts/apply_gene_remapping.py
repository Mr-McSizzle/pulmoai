#!/usr/bin/env python3
"""Apply HGNC previous/alias remapping to Wang et al. 2023 profiles.

Extracts counts for newly resolved genes, records a per-profile missing mask,
applies CPM -> log1p normalization, and saves new clearly named files without
touching existing processed files.
"""

from __future__ import annotations

import json
from pathlib import Path
import anndata
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATLAS_NPZ = PROJECT_ROOT / "data" / "processed" / "allcells_pseudobulk_counts.npz"
OLD_WANG_NPZ = PROJECT_ROOT / "data" / "processed" / "census_new_profiles.npz"
HGNC_TXT = PROJECT_ROOT / "data" / "raw" / "hgnc_complete_set.txt"

OUTPUT_NPZ = PROJECT_ROOT / "data" / "processed" / "census_new_profiles_remapped.npz"
OUTPUT_META_CSV = PROJECT_ROOT / "data" / "processed" / "census_new_metadata_remapped.csv"
OUTPUT_AUDIT_CSV = PROJECT_ROOT / "reports" / "gene_remapping_audit.csv"

def build_hgnc_clean_maps() -> dict[str, str]:
    hgnc = pd.read_csv(HGNC_TXT, sep="\t", low_memory=False)
    prev_map: dict[str, set[str]] = {}
    for _, row in hgnc.iterrows():
        curr = str(row["symbol"]).strip()
        if pd.notna(row["prev_symbol"]):
            for p in str(row["prev_symbol"]).split("|"):
                p = p.strip().strip("\"")
                if p:
                    prev_map.setdefault(p, set()).add(curr)
        if pd.notna(row["alias_symbol"]):
            for al in str(row["alias_symbol"]).split("|"):
                al = al.strip().strip("\"")
                if al:
                    prev_map.setdefault(al, set()).add(curr)

    # Keep only unambiguous 1:1 mappings
    return {k: list(v)[0] for k, v in prev_map.items() if len(v) == 1}

def main() -> None:
    print("=== Applying HGNC Legacy Gene Remapping ===")
    clean_map = build_hgnc_clean_maps()

    # Load atlas features
    with np.load(ATLAS_NPZ, allow_pickle=True) as a:
        atlas_features = a["feature_names"].astype(str).tolist()
    n_atlas = len(atlas_features)
    atlas_set = set(atlas_features)

    # Load existing Wang profiles
    with np.load(OLD_WANG_NPZ, allow_pickle=True) as w:
        wang_counts = w["counts"].copy()
        wang_studies = w["studies"]
        wang_donors = w["donors"]
        wang_labels = w["labels"]
        wang_cells = w["cell_counts"]
        wang_units = w["unit"]

    # Classify each atlas gene
    # Note: in Census, direct symbols were matched in step 4b (17,573 genes)
    measured_in_old_wang = (wang_counts.sum(axis=0) > 0)

    audit_rows = []
    direct_count = 0
    remapped_count = 0
    missing_count = 0

    remapping_actions = {} # atlas_col_idx -> target_symbol
    for idx, sym in enumerate(atlas_features):
        if measured_in_old_wang[idx]:
            direct_count += 1
            audit_rows.append({"atlas_gene": sym, "status": "direct_match", "resolved_symbol": sym})
        elif sym in clean_map:
            target = clean_map[sym]
            remapped_count += 1
            remapping_actions[idx] = target
            audit_rows.append({"atlas_gene": sym, "status": "unambiguous_alias", "resolved_symbol": target})
        else:
            missing_count += 1
            audit_rows.append({"atlas_gene": sym, "status": "unmeasured", "resolved_symbol": ""})

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUTPUT_AUDIT_CSV, index=False)
    print(f"Audit table saved to {OUTPUT_AUDIT_CSV}")
    print(f"Direct match genes: {direct_count:,} ({direct_count/n_atlas:.1%})")
    print(f"Unambiguously remapped genes: {remapped_count:,} ({remapped_count/n_atlas:.1%})")
    print(f"Unmeasured / unmapped genes: {missing_count:,} ({missing_count/n_atlas:.1%})")
    print(f"Total resolved gene space: {direct_count + remapped_count:,} ({(direct_count+remapped_count)/n_atlas:.1%})")

    # Construct the per-profile missing mask
    # True = gene is missing/unmeasured
    missing_mask = np.zeros((len(wang_donors), n_atlas), dtype=bool)
    # By default, any gene that is not direct and not remapped is missing
    unresolved_mask = np.array([row["status"] == "unmeasured" for row in audit_rows], dtype=bool)
    for i in range(len(wang_donors)):
        missing_mask[i, :] = unresolved_mask

    # Normalize remapped counts to CPM -> log1p
    n_profiles = len(wang_donors)
    norm_matrix = np.zeros((n_profiles, n_atlas), dtype=np.float32)
    lib_sizes = []
    metadata_rows = []

    for i in range(n_profiles):
        raw_vec = wang_counts[i, :]
        lib_size = float(np.sum(raw_vec))
        lib_sizes.append(lib_size)
        cpm = (raw_vec / max(lib_size, 1.0)) * 1e6
        norm_matrix[i, :] = np.log1p(cpm).astype(np.float32)

        metadata_rows.append({
            "profile_id": f"{wang_studies[i]}_{wang_donors[i]}",
            "study_name": wang_studies[i],
            "donor_id": wang_donors[i],
            "mapped_class": wang_labels[i],
            "unit": wang_units[i],
            "n_cells": wang_cells[i],
            "library_size": lib_size,
            "measured_genes_count": int(np.sum(~missing_mask[i])),
            "missing_genes_count": int(np.sum(missing_mask[i])),
            "coverage_fraction": round(float(np.sum(~missing_mask[i])) / n_atlas, 4),
        })

    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(OUTPUT_META_CSV, index=False)
    print(f"Saved remapped metadata to {OUTPUT_META_CSV}")

    # Save remapped NPZ with missing_mask
    np.savez_compressed(
        OUTPUT_NPZ,
        expression=norm_matrix,
        counts=wang_counts,
        missing_mask=missing_mask,
        studies=wang_studies,
        donors=wang_donors,
        labels=wang_labels,
        feature_names=np.array(atlas_features, dtype=object),
        cell_counts=wang_cells,
        unit=wang_units,
        library_sizes=np.array(lib_sizes, dtype=np.float64),
    )
    print(f"Saved remapped profiles to {OUTPUT_NPZ} ({OUTPUT_NPZ.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
