#!/usr/bin/env python3
"""Build new pseudobulk profiles from verified clean Census datasets.

Streams/downloads verified clean H5AD files to temporary storage, processes
donors one by one in backed mode to strictly bound memory, verifies integer
counts (using adata.raw.X or adata.X), harmonizes gene symbols with the
existing 30,566 atlas genes, applies the exact CPM -> log1p transformation
from src/pulmoai/preprocessing.py, and saves new clearly named files
without touching existing processed files.
"""

from __future__ import annotations

import os
import resource
import shutil
import sys
import time
from pathlib import Path

import anndata
import certifi
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
TEMP_DIR = DATA_DIR / "raw" / "census_temp"
ATLAS_PROFILES_NPZ = PROCESSED_DIR / "allcells_pseudobulk_counts.npz"

OUTPUT_PROFILES_NPZ = PROCESSED_DIR / "census_new_profiles.npz"
OUTPUT_METADATA_CSV = PROCESSED_DIR / "census_new_metadata.csv"
OUTPUT_CELLTYPE_COMP_CSV = PROCESSED_DIR / "census_new_celltype_composition.csv"
OUTPUT_CELLTYPE_PB_NPZ = PROCESSED_DIR / "census_new_celltype_pseudobulk.npz"


def check_disk(min_free_gb: float = 3.0) -> float:
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Disk space too low: {free_gb:.2f} GiB free, minimum required is {min_free_gb} GiB. Aborting."
        )
    return free_gb


def get_peak_ram_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def download_file(url: str, dest: Path) -> None:
    check_disk(3.0)
    print(f"  Downloading {url.split('/')[-1]} to temporary storage...")
    t0 = time.time()
    r = requests.get(url, stream=True, verify=certifi.where())
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    elapsed = time.time() - t0
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  Downloaded {size_mb:.1f} MB in {elapsed:.1f}s ({size_mb/max(elapsed, 0.1):.1f} MB/s)")


def main() -> None:
    t_start = time.time()
    initial_free_gb = check_disk(3.0)
    print(f"[Resource Check] Initial free disk space: {initial_free_gb:.2f} GiB")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load existing atlas feature names
    print(f"\n[Gene Harmonization] Loading existing atlas feature list from {ATLAS_PROFILES_NPZ}...")
    with np.load(ATLAS_PROFILES_NPZ) as a:
        atlas_features = a["feature_names"].astype(str).tolist()
    n_atlas_features = len(atlas_features)
    atlas_feature_map = {feat: idx for idx, feat in enumerate(atlas_features)}
    print(f"  Atlas feature count: {n_atlas_features:,} genes.")

    # 2. Define clean candidate datasets
    CLEAN_DATASETS = [
        {
            "study_name": "Wang_Immunity_2023_Emphysema",
            "collection_id": "03cdc7f4-bd08-49d0-a395-4487c0e5a168",
            "doi": "10.1016/j.immuni.2023.01.032",
            "files": [
                {
                    "partition": "immune",
                    "dataset_id": "1e5bd3b8-6a0e-4959-8d69-cafed30fe814",
                    "url": "https://datasets.cellxgene.cziscience.com/b1348c02-123e-4e06-8511-def04849cb8d.h5ad",
                    "filename": "wang_immune.h5ad",
                },
                {
                    "partition": "non_immune",
                    "dataset_id": "4b6af54a-4a21-46e0-bc8d-673c0561a836",
                    "url": "https://datasets.cellxgene.cziscience.com/86d2948e-9ff9-47c7-9d83-999c4cfa1789.h5ad",
                    "filename": "wang_non_immune.h5ad",
                },
            ],
            "disease_mapping": {
                "pulmonary emphysema": "COPD",
                "normal": "Donor",
            },
        }
    ]

    all_profiles_raw: dict[str, np.ndarray] = {}
    donor_meta: dict[str, dict] = {}
    donor_celltypes: dict[str, dict[str, int]] = {}
    donor_celltype_pb: dict[tuple[str, str], np.ndarray] = {}
    gene_overlap_set: set[str] = set()

    for study_info in CLEAN_DATASETS:
        study_name = study_info["study_name"]
        print(f"\n[Processing Study] {study_name} (DOI: {study_info['doi']})...")

        for f_info in study_info["files"]:
            part_name = f_info["partition"]
            ds_id = f_info["dataset_id"]
            h5ad_path = TEMP_DIR / f_info["filename"]

            download_file(f_info["url"], h5ad_path)

            print(f"  Opening {part_name} partition in backed mode...")
            adata = anndata.read_h5ad(h5ad_path, backed="r")
            n_cells, n_vars = adata.shape
            print(f"  Shape: {n_cells:,} cells x {n_vars:,} genes.")

            # Check if raw counts are in raw or layers or X
            if adata.raw is not None:
                print("  Found adata.raw. Checking raw counts in adata.raw.X...")
                use_raw = True
                raw_var = adata.raw.var
                raw_X = adata.raw.X
                if "feature_name" in raw_var.columns:
                    symbols = raw_var["feature_name"].astype(str).values
                else:
                    symbols = raw_var.index.astype(str).values
            elif "raw_counts" in adata.layers:
                print("  Found adata.layers['raw_counts']...")
                use_raw = False
                raw_var = adata.var
                raw_X = adata.layers["raw_counts"]
                if "feature_name" in raw_var.columns:
                    symbols = raw_var["feature_name"].astype(str).values
                else:
                    symbols = raw_var.index.astype(str).values
            else:
                use_raw = False
                raw_var = adata.var
                raw_X = adata.X
                if "feature_name" in raw_var.columns:
                    symbols = raw_var["feature_name"].astype(str).values
                else:
                    symbols = raw_var.index.astype(str).values

            # Sample slice to verify integer counts
            sample_slice = raw_X[:500]
            sample_data = sample_slice.data if hasattr(sample_slice, "data") else sample_slice
            is_int = np.all(np.equal(np.mod(sample_data, 1), 0))
            if not is_int:
                print(f"  Warning: raw_X has floats (sample: {sample_data[:5]}). Checking adata.X...")
                if use_raw:
                    sample_slice2 = adata.X[:500]
                    sample_data2 = sample_slice2.data if hasattr(sample_slice2, "data") else sample_slice2
                    if np.all(np.equal(np.mod(sample_data2, 1), 0)):
                        print("  Found integer counts in adata.X!")
                        raw_X = adata.X
                        is_int = True

            if not is_int:
                h5ad_path.unlink()
                raise ValueError(f"Study {study_name} partition {part_name} contains non-integer counts! Rejected.")

            print(f"  Integer count verification: PASSED (all counts strictly integers).")

            # Map symbols to atlas feature indices
            matched_indices = []
            for col_idx, sym in enumerate(symbols):
                if sym in atlas_feature_map:
                    matched_indices.append((col_idx, atlas_feature_map[sym]))
                    gene_overlap_set.add(sym)

            print(f"  Matched genes: {len(matched_indices):,} / {len(symbols):,} ({len(matched_indices)/n_atlas_features:.1%} of atlas)")

            obs_df = adata.obs
            donors = obs_df["donor_id"].unique().tolist()
            col_idxs = np.array([m[0] for m in matched_indices], dtype=np.int32)
            atlas_idxs = np.array([m[1] for m in matched_indices], dtype=np.int32)
            sort_perm = np.argsort(col_idxs)
            col_idxs = col_idxs[sort_perm]
            atlas_idxs = atlas_idxs[sort_perm]

            for donor in donors:
                donor_mask = (obs_df["donor_id"] == donor).values
                donor_indices = np.where(donor_mask)[0]
                donor_n_cells = len(donor_indices)
                if donor_n_cells == 0:
                    continue

                if donor not in all_profiles_raw:
                    all_profiles_raw[donor] = np.zeros(n_atlas_features, dtype=np.float64)
                    donor_celltypes[donor] = {}
                    
                    sub_obs = obs_df[donor_mask].iloc[0]
                    orig_dis = str(sub_obs.get("disease", "unknown"))
                    mapped_cls = study_info["disease_mapping"].get(orig_dis, "Unknown")
                    donor_meta[donor] = {
                        "study_name": study_name,
                        "dataset_id": ds_id,
                        "donor_id": donor,
                        "mapped_class": mapped_cls,
                        "original_disease": orig_dis,
                        "unit": "donor",
                        "tissue": str(sub_obs.get("tissue", "alveolus of lung")),
                        "n_cells": 0,
                    }

                donor_meta[donor]["n_cells"] += donor_n_cells

                print(f"    Extracting {part_name} counts for donor {donor} ({donor_n_cells:,} cells)...")
                # Slice counts for this donor
                donor_chunk = raw_X[donor_indices, :][:, col_idxs]
                donor_sum_vector = np.array(donor_chunk.sum(axis=0)).flatten()
                np.add.at(all_profiles_raw[donor], atlas_idxs, donor_sum_vector)

                donor_cell_type_col = "cell_type" if "cell_type" in obs_df.columns else None
                if donor_cell_type_col:
                    ct_series = obs_df.loc[donor_mask, donor_cell_type_col]
                    for ct, count in ct_series.value_counts().items():
                        donor_celltypes[donor][ct] = donor_celltypes[donor].get(ct, 0) + int(count)

                        ct_indices = np.where((obs_df["donor_id"] == donor) & (obs_df[donor_cell_type_col] == ct))[0]
                        ct_chunk = raw_X[ct_indices, :][:, col_idxs]
                        ct_sum = np.array(ct_chunk.sum(axis=0)).flatten()
                        key = (donor, str(ct))
                        if key not in donor_celltype_pb:
                            donor_celltype_pb[key] = np.zeros(n_atlas_features, dtype=np.float64)
                        np.add.at(donor_celltype_pb[key], atlas_idxs, ct_sum)

            del adata
            h5ad_path.unlink()
            print(f"  Cleaned up temporary file {h5ad_path.name}.")
            print(f"  Current peak RAM: {get_peak_ram_mb():.1f} MB.")

    # 3. Apply CPM -> log1p normalization
    print("\n[Preprocessing] Applying exact CPM -> log1p normalization...")
    donor_list = sorted(all_profiles_raw.keys())
    n_donors = len(donor_list)
    raw_matrix = np.zeros((n_donors, n_atlas_features), dtype=np.float64)
    norm_matrix = np.zeros((n_donors, n_atlas_features), dtype=np.float32)
    library_sizes = []
    metadata_rows = []

    for row_idx, donor in enumerate(donor_list):
        raw_vec = all_profiles_raw[donor]
        raw_matrix[row_idx, :] = raw_vec
        lib_size = float(np.sum(raw_vec))
        library_sizes.append(lib_size)

        if lib_size <= 0:
            raise ValueError(f"Donor {donor} has zero total counts!")

        cpm = (raw_vec / lib_size) * 1e6
        norm_matrix[row_idx, :] = np.log1p(cpm).astype(np.float32)

        meta = donor_meta[donor]
        meta["library_size"] = lib_size
        metadata_rows.append({
            "profile_id": f"{meta['study_name']}_{donor}",
            "dataset_id": meta["dataset_id"],
            "donor_id": donor,
            "mapped_class": meta["mapped_class"],
            "original_disease": meta["original_disease"],
            "unit": "donor",
            "tissue": meta["tissue"],
            "n_cells": meta["n_cells"],
            "library_size": lib_size,
        })

    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(OUTPUT_METADATA_CSV, index=False)
    print(f"  Saved metadata to {OUTPUT_METADATA_CSV}")

    # 4. Save new profiles NPZ
    np.savez_compressed(
        OUTPUT_PROFILES_NPZ,
        expression=norm_matrix,
        counts=raw_matrix,
        studies=np.array([donor_meta[d]["study_name"] for d in donor_list], dtype=object),
        donors=np.array(donor_list, dtype=object),
        labels=np.array([donor_meta[d]["mapped_class"] for d in donor_list], dtype=object),
        feature_names=np.array(atlas_features, dtype=object),
        cell_counts=np.array([donor_meta[d]["n_cells"] for d in donor_list], dtype=np.int64),
        unit=np.array(["donor"] * n_donors, dtype=object),
        library_sizes=np.array(library_sizes, dtype=np.float64),
    )
    print(f"  Saved normalized profiles to {OUTPUT_PROFILES_NPZ} ({OUTPUT_PROFILES_NPZ.stat().st_size / (1024*1024):.2f} MB)")

    # 5. Cell type composition & Cell type pseudobulk
    comp_rows = []
    for donor in donor_list:
        total_cells = donor_meta[donor]["n_cells"]
        cts = donor_celltypes.get(donor, {})
        for ct, count in cts.items():
            comp_rows.append({
                "donor_id": donor,
                "cell_type": ct,
                "cell_count": count,
                "proportion": count / total_cells,
            })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUTPUT_CELLTYPE_COMP_CSV, index=False)
    print(f"  Saved cell-type composition to {OUTPUT_CELLTYPE_COMP_CSV}")

    ct_keys = sorted(donor_celltype_pb.keys())
    ct_matrix = np.array([donor_celltype_pb[k] for k in ct_keys], dtype=np.float32)
    np.savez_compressed(
        OUTPUT_CELLTYPE_PB_NPZ,
        counts=ct_matrix,
        donors=np.array([k[0] for k in ct_keys], dtype=object),
        cell_types=np.array([k[1] for k in ct_keys], dtype=object),
        feature_names=np.array(atlas_features, dtype=object),
    )
    print(f"  Saved cell-type pseudobulk to {OUTPUT_CELLTYPE_PB_NPZ}")

    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    peak_ram_mb = get_peak_ram_mb()
    final_free_gb = check_disk(3.0)
    print("\n" + "=" * 80)
    print("STEP 4b SUMMARY & QC")
    print("=" * 80)
    print(f"Total new profiles generated: {n_donors}")
    print(meta_df[["profile_id", "mapped_class", "n_cells", "library_size"]].to_string(index=False))
    print(f"\nGene overlap count: {len(gene_overlap_set):,} / {n_atlas_features:,} ({len(gene_overlap_set)/n_atlas_features:.1%})")
    print(f"Peak RAM: {peak_ram_mb:.1f} MB")
    print(f"Final Free Disk Space: {final_free_gb:.2f} GiB")
    print(f"Total time elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
