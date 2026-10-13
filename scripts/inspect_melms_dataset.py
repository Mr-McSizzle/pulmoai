#!/usr/bin/env python3
"""Inspect Melms et al. (d8da613f-e681-4c69-b463-e94f5e66847f) and compare with atlas."""

from __future__ import annotations

import json
from pathlib import Path

import cellxgene_census
import certifi
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATLAS_NPZ = PROJECT_ROOT / "data" / "processed" / "allcells_pseudobulk_counts.npz"

def main() -> None:
    dataset_id = "d8da613f-e681-4c69-b463-e94f5e66847f"
    print(f"=== Inspecting Melms et al. ({dataset_id}) ===")

    # 1. Query Census datasets table for collection & URL
    census = cellxgene_census.open_soma()
    ds_df = census["census_info"]["datasets"].read().concat().to_pandas()
    row = ds_df[ds_df["dataset_id"] == dataset_id].iloc[0]
    print(f"Dataset Title: {row['dataset_title']}")
    print(f"Collection ID: {row['collection_id']}")
    print(f"Collection Name: {row['collection_name']}")
    print(f"Collection DOI: {row['collection_doi']}")
    print(f"Dataset H5AD path: {row['dataset_h5ad_path']}")

    # Check remote H5AD file size
    h5ad_url = f"https://datasets.cellxgene.cziscience.com/{dataset_id}.h5ad"
    try:
        r = requests.head(h5ad_url, verify=certifi.where())
        cl = r.headers.get("content-length")
        if cl:
            size_mb = int(cl) / (1024 * 1024)
            print(f"H5AD download size: {size_mb:.2f} MB")
        else:
            print("H5AD content-length not returned")
    except Exception as e:
        print(f"Could not fetch H5AD size: {e}")

    # 2. Query obs
    human = census["census_data"]["homo_sapiens"]
    obs = human.obs.read(
        value_filter=f"dataset_id == '{dataset_id}'",
        column_names=["donor_id", "disease", "tissue", "assay", "suspension_type"]
    ).concat().to_pandas()

    for col in ["donor_id", "disease", "tissue", "assay", "suspension_type"]:
        obs[col] = obs[col].astype(str)

    print(f"\nTotal cells: {len(obs):,}")
    print("\nTissues:\n", obs["tissue"].value_counts().to_string())
    print("\nDiseases:\n", obs["disease"].value_counts().to_string())
    print("\nAssays:\n", obs["assay"].value_counts().to_string())
    print("\nSuspension Types:\n", obs["suspension_type"].value_counts().to_string())

    donor_counts = obs.groupby(["donor_id", "disease"]).size().reset_index(name="cell_count")
    print(f"\nTotal unique donors: {len(donor_counts)}")
    print(donor_counts.to_string())

    # 3. Compare with atlas orig_ids for GSE171668, GSE168191, GSE158127
    print("\n=== Comparing Donor/Sample IDs with Atlas ===")
    with np.load(ATLAS_NPZ, allow_pickle=True) as a:
        atlas_studies = a["studies"]
        atlas_orig_ids = a["orig_ids"]
        atlas_labels = a["labels"]
        atlas_sample_ids = a["sample_ids"]

    for target_gse in ["GSE171668", "GSE168191", "GSE158127"]:
        mask = (atlas_studies == target_gse)
        gse_origs = set(atlas_orig_ids[mask])
        gse_samples = set(atlas_sample_ids[mask])
        print(f"\nAtlas study {target_gse}: {np.sum(mask)} libraries")
        print(f"  orig_ids ({len(gse_origs)}): {sorted(list(gse_origs))[:10]}")
        melms_donors = set(donor_counts["donor_id"])
        shared_origs = melms_donors.intersection(gse_origs)
        print(f"  Overlap with Melms donor_ids: {len(shared_origs)} shared IDs ({shared_origs})")

    census.close()

if __name__ == "__main__":
    main()
