#!/usr/bin/env python3
"""Compute profile correlations between Wang et al. 2023 and atlas profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATLAS_NPZ = PROJECT_ROOT / "data" / "processed" / "allcells_pseudobulk_counts.npz"
WANG_NPZ = PROJECT_ROOT / "data" / "processed" / "census_new_profiles.npz"

def main() -> None:
    with np.load(ATLAS_NPZ, allow_pickle=True) as a:
        atlas_counts = a["counts"]
        atlas_features = a["feature_names"].astype(str)
        atlas_studies = a["studies"].astype(str)
        atlas_labels = a["labels"].astype(str)
        atlas_orig = a["orig_ids"].astype(str)
        atlas_samples = a["sample_ids"].astype(str)

    with np.load(WANG_NPZ, allow_pickle=True) as w:
        wang_counts = w["counts"]
        wang_features = w["feature_names"].astype(str)
        wang_donors = w["donors"].astype(str)
        wang_labels = w["labels"].astype(str)

    # 1. Normalize atlas to log1p(CPM)
    atlas_lib = atlas_counts.sum(axis=1, keepdims=True)
    atlas_cpm = (atlas_counts / np.maximum(atlas_lib, 1)) * 1e6
    atlas_log = np.log1p(atlas_cpm)

    # 2. Normalize Wang to log1p(CPM)
    wang_lib = wang_counts.sum(axis=1, keepdims=True)
    wang_cpm = (wang_counts / np.maximum(wang_lib, 1)) * 1e6
    wang_log = np.log1p(wang_cpm)

    # Intersection of genes that are measured (>0 counts across both)
    atlas_active = atlas_counts.sum(axis=0) > 0
    wang_active = wang_counts.sum(axis=0) > 0
    shared_mask = atlas_active & wang_active
    print(f"Number of shared expressed genes for correlation: {np.sum(shared_mask):,}")

    atlas_sub = atlas_log[:, shared_mask]
    wang_sub = wang_log[:, shared_mask]

    # Compute correlation matrix between Wang (6) and Atlas (372)
    # Standardize rows
    atlas_centered = atlas_sub - atlas_sub.mean(axis=1, keepdims=True)
    atlas_norm = np.linalg.norm(atlas_centered, axis=1, keepdims=True)
    atlas_std = atlas_centered / np.maximum(atlas_norm, 1e-12)

    wang_centered = wang_sub - wang_sub.mean(axis=1, keepdims=True)
    wang_norm = np.linalg.norm(wang_centered, axis=1, keepdims=True)
    wang_std = wang_centered / np.maximum(wang_norm, 1e-12)

    corr_matrix = np.dot(wang_std, atlas_std.T) # shape (6, 372)

    print("\n" + "=" * 80)
    print("TOP 3 NEAREST ATLAS PROFILES FOR EACH WANG PROFILE")
    print("=" * 80)

    for i, donor in enumerate(wang_donors):
        label = wang_labels[i]
        corrs = corr_matrix[i]
        top_idx = np.argsort(corrs)[::-1][:3]
        print(f"\nProfile: Wang {donor} (Disease: {label})")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  Rank {rank}: r = {corrs[idx]:.4f} | Study: {atlas_studies[idx]} | Label: {atlas_labels[idx]} | Orig ID: {atlas_orig[idx]} | Sample ID: {atlas_samples[idx]}")

        # Specifically check GSE171541 and GSE136831
        mask_171541 = (atlas_studies == "GSE171541")
        if np.any(mask_171541):
            max_r_171541 = np.max(corrs[mask_171541])
            print(f"  -> Max r vs GSE171541 (COPD): {max_r_171541:.4f}")
        mask_136831 = (atlas_studies == "GSE136831")
        if np.any(mask_136831):
            max_r_136831 = np.max(corrs[mask_136831])
            print(f"  -> Max r vs GSE136831: {max_r_136831:.4f}")

    # 3. Typical correlation between different donors of same disease inside the atlas
    print("\n" + "=" * 80)
    print("WITHIN-ATLAS PAIRWISE CORRELATIONS (FOR COMPARISON)")
    print("=" * 80)
    # Compute atlas-atlas correlation matrix
    atlas_corr = np.dot(atlas_std, atlas_std.T)
    np.fill_diagonal(atlas_corr, np.nan)

    for dis in ["Donor", "COPD", "PF", "COVID19"]:
        dis_mask = (atlas_labels == dis)
        dis_corrs = atlas_corr[dis_mask, :][:, dis_mask]
        valid_corrs = dis_corrs[~np.isnan(dis_corrs)]
        print(f"Within-Atlas {dis:<8} (n={np.sum(dis_mask)}): Mean r = {np.mean(valid_corrs):.4f}, Median r = {np.median(valid_corrs):.4f}, Range = [{np.min(valid_corrs):.4f}, {np.max(valid_corrs):.4f}]")

    # Across different studies of same disease
    print("\nWithin-Atlas Cross-Study Same-Disease Correlations:")
    for dis in ["Donor", "COPD", "PF"]:
        cross_corrs = []
        dis_idx = np.where(atlas_labels == dis)[0]
        for idx1 in dis_idx:
            for idx2 in dis_idx:
                if idx1 < idx2 and atlas_studies[idx1] != atlas_studies[idx2]:
                    cross_corrs.append(atlas_corr[idx1, idx2])
        if cross_corrs:
            print(f"  Cross-study {dis:<6}: Mean r = {np.mean(cross_corrs):.4f}, Median r = {np.median(cross_corrs):.4f}, Max r = {np.max(cross_corrs):.4f}")

if __name__ == "__main__":
    main()
