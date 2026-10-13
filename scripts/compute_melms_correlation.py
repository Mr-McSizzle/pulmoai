#!/usr/bin/env python3
"""Compute correlation between Melms C51ctr sample and atlas profiles."""

from __future__ import annotations

import cellxgene_census
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATLAS_NPZ = PROJECT_ROOT / "data" / "processed" / "allcells_pseudobulk_counts.npz"

def main() -> None:
    print("Opening Census to get var mapping and cells for Melms C51ctr...")
    census = cellxgene_census.open_soma()
    human = census["census_data"]["homo_sapiens"]

    # 1. Get var mapping
    var_df = human.ms["RNA"].var.read(column_names=["soma_joinid", "feature_name"]).concat().to_pandas()
    soma_to_symbol = dict(zip(var_df["soma_joinid"], var_df["feature_name"]))

    # 2. Get cell joinids for C51ctr
    obs = human.obs.read(
        value_filter="dataset_id == 'd8da613f-e681-4c69-b463-e94f5e66847f' and donor_id == 'C51ctr'",
        column_names=["soma_joinid"]
    ).concat().to_pandas()
    sample_cell_ids = obs["soma_joinid"].values[:300]
    print(f"Reading counts for {len(sample_cell_ids)} cells of C51ctr...")

    tbl = human.ms["RNA"].X["raw"].read(coords=(sample_cell_ids, slice(None))).tables().concat().to_pandas()
    census.close()

    # Aggregate counts per gene
    gene_counts = tbl.groupby("soma_dim_1")["soma_data"].sum().to_dict()
    melms_dict = {soma_to_symbol[soma_id]: cnt for soma_id, cnt in gene_counts.items() if soma_id in soma_to_symbol}

    # 3. Load Atlas profiles
    with np.load(ATLAS_NPZ, allow_pickle=True) as a:
        atlas_counts = a["counts"]
        atlas_features = a["feature_names"].astype(str)
        atlas_studies = a["studies"].astype(str)
        atlas_labels = a["labels"].astype(str)
        atlas_orig = a["orig_ids"].astype(str)

    # Normalize Atlas
    atlas_lib = atlas_counts.sum(axis=1, keepdims=True)
    atlas_cpm = (atlas_counts / np.maximum(atlas_lib, 1)) * 1e6
    atlas_log = np.log1p(atlas_cpm)

    # Create Melms vector
    melms_raw = np.array([melms_dict.get(g, 0.0) for g in atlas_features], dtype=np.float64)
    active = (atlas_counts.sum(axis=0) > 0) & (melms_raw > 0)
    print(f"Shared active genes: {np.sum(active):,}")

    melms_lib = melms_raw.sum()
    melms_cpm = (melms_raw / max(melms_lib, 1)) * 1e6
    melms_log = np.log1p(melms_cpm)

    melms_sub = melms_log[active]
    atlas_sub = atlas_log[:, active]

    # Compute correlations
    m_cen = melms_sub - melms_sub.mean()
    m_std = m_cen / np.linalg.norm(m_cen)

    a_cen = atlas_sub - atlas_sub.mean(axis=1, keepdims=True)
    a_std = a_cen / np.linalg.norm(a_cen, axis=1, keepdims=True)

    corrs = np.dot(a_std, m_std)

    print("\n" + "=" * 80)
    print("TOP 5 NEAREST ATLAS PROFILES FOR MELMS DONOR C51ctr (Normal Control)")
    print("=" * 80)
    top_indices = np.argsort(corrs)[::-1][:5]
    for rank, idx in enumerate(top_indices, 1):
        print(f"Rank {rank}: r = {corrs[idx]:.4f} | Study: {atlas_studies[idx]} | Label: {atlas_labels[idx]} | Orig ID: {atlas_orig[idx]}")

    # Check COVID studies in atlas
    print("\nMax correlation vs atlas COVID-19 studies:")
    for st in ["GSE171668", "GSE158127"]:
        mask = (atlas_studies == st)
        if np.any(mask):
            print(f"  Max r vs {st}: {np.max(corrs[mask]):.4f}")

if __name__ == "__main__":
    main()
