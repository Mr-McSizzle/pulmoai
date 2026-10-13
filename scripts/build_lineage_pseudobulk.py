#!/usr/bin/env python3
"""Build per-cell-type (11 lineages) pseudobulk profiles from atlas H5AD."""

from __future__ import annotations

import time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATLAS_H5AD = Path("/Users/nisarg/Downloads/atlas-001.h5ad")
ATLAS_PB_NPZ = PROJECT_ROOT / "data" / "processed" / "allcells_pseudobulk_counts.npz"
OUTPUT_NPZ = PROJECT_ROOT / "data" / "processed" / "atlas_lineage_pseudobulk_5k.npz"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "atlas_lineage_metadata.csv"

LINEAGE_MAP = {
    "AT1": "Alveolar_Epithelial",
    "AT2": "Alveolar_Epithelial",
    "Basal": "Airway_Epithelial",
    "Ciliated": "Airway_Epithelial",
    "Secretory": "Airway_Epithelial",
    "Endothelial": "Endothelial",
    "Fibroblast": "Fibroblast_Stromal",
    "Mesothelial": "Fibroblast_Stromal",
    "SMC_Pericytes": "Smooth_Muscle",
    "Macrophage": "Macrophage",
    "Monocyte": "Monocyte_DC",
    "DC": "Monocyte_DC",
    "CD4": "T_NK",
    "CD8": "T_NK",
    "NK": "T_NK",
    "B_Plasma": "B_Plasma",
    "Mast": "Mast",
    "Dividing": "Dividing",
}

LINEAGE_NAMES = sorted(list(set(LINEAGE_MAP.values())))

def main() -> None:
    t_start = time.time()
    print("=== Building Per-Lineage Pseudobulk Profiles ===")
    print(f"Broad lineages ({len(LINEAGE_NAMES)}): {LINEAGE_NAMES}")

    # 1. Select top 5,000 genes by prevalence in existing 372 profiles
    print("\n[Step 1] Selecting top 5,000 genes by expression prevalence in existing atlas...")
    with np.load(ATLAS_PB_NPZ, allow_pickle=True) as a:
        all_counts = a["counts"]
        all_features = a["feature_names"].astype(str)
        sample_ids_order = a["sample_ids"].astype(str).tolist()
        sample_studies_order = a["studies"].astype(str).tolist()
        sample_labels_order = a["labels"].astype(str).tolist()

    prevalence = (all_counts > 0).mean(axis=0)
    top_5000_col_indices = np.argsort(prevalence)[::-1][:5000]
    # Sort column indices for faster sparse slicing
    top_5000_col_indices = np.sort(top_5000_col_indices)
    selected_genes = all_features[top_5000_col_indices]
    print(f"Top 5,000 genes prevalence range: {prevalence[top_5000_col_indices[0]]:.4f} to {prevalence[top_5000_col_indices[-1]]:.4f}")

    # 2. Open H5AD in backed mode and build indexing
    print(f"\n[Step 2] Opening {ATLAS_H5AD} in backed mode...")
    adata = ad.read_h5ad(ATLAS_H5AD, backed="r")
    obs_df = adata.obs[["orig.ident", "gse_alias", "celltypeuse"]].copy()
    n_cells = len(obs_df)
    print(f"Atlas contains {n_cells:,} cells x {adata.n_vars:,} genes.")

    sample_id_to_idx = {sid: idx for idx, sid in enumerate(sample_ids_order)}
    lineage_to_idx = {lin: idx for idx, lin in enumerate(LINEAGE_NAMES)}

    # Map each cell to sample index and lineage index
    cell_sids = obs_df["gse_alias"].astype(str) + "::" + obs_df["orig.ident"].astype(str)
    cell_sample_codes = np.array([sample_id_to_idx[sid] for sid in cell_sids], dtype=np.int32)
    cell_lineages = obs_df["celltypeuse"].map(LINEAGE_MAP).fillna("Dividing").values
    cell_lineage_codes = np.array([lineage_to_idx[lin] for lin in cell_lineages], dtype=np.int32)

    n_samples = len(sample_ids_order)
    n_lineages = len(LINEAGE_NAMES)
    n_pairs = n_samples * n_lineages
    cell_pair_codes = cell_sample_codes.astype(np.int64) * n_lineages + cell_lineage_codes.astype(np.int64)

    # 3. Stream through H5AD in 50,000 cell chunks
    chunk_rows = 50000
    n_chunks = (n_cells + chunk_rows - 1) // chunk_rows
    counts_accumulator = np.zeros((n_pairs, 5000), dtype=np.float64)
    cell_counts_by_pair = np.bincount(cell_pair_codes, minlength=n_pairs)

    print(f"\n[Step 3] Streaming {n_chunks} chunks of {chunk_rows:,} cells...")
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_rows
        end = min(start + chunk_rows, n_cells)
        t_chunk0 = time.time()

        # Slice block
        block = adata.X[start:end]
        if not sparse.issparse(block):
            block = sparse.csr_matrix(block)
        # Keep top 5000 genes
        block_5k = block[:, top_5000_col_indices]

        # Aggregate block
        chunk_pair_codes = cell_pair_codes[start:end]
        unique_pairs, inverse = np.unique(chunk_pair_codes, return_inverse=True)

        aggregator = sparse.csr_matrix(
            (
                np.ones(end - start, dtype=np.float32),
                (inverse.astype(np.int32), np.arange(end - start, dtype=np.int32)),
            ),
            shape=(len(unique_pairs), end - start),
        )

        partial = aggregator @ block_5k
        counts_accumulator[unique_pairs] += partial.toarray()

        if (chunk_idx + 1) == 1 or (chunk_idx + 1) % 5 == 0 or (chunk_idx + 1) == n_chunks:
            print(f"  Processed chunk {chunk_idx+1:2d}/{n_chunks} ({end:9,d}/{n_cells:,} cells) in {time.time()-t_chunk0:.2f}s")

    adata.file.close()

    # 4. Filter combinations under 20 cells
    print("\n[Step 4] Filtering combinations with >= 20 cells...")
    pair_metadata = []
    for s_idx in range(n_samples):
        for l_idx in range(n_lineages):
            p_idx = s_idx * n_lineages + l_idx
            n_c = int(cell_counts_by_pair[p_idx])
            pair_metadata.append({
                "pair_index": p_idx,
                "sample_id": sample_ids_order[s_idx],
                "study": sample_studies_order[s_idx],
                "disease": sample_labels_order[s_idx],
                "lineage": LINEAGE_NAMES[l_idx],
                "cell_count": n_c,
                "valid": (n_c >= 20),
            })

    pair_meta_df = pd.DataFrame(pair_metadata)
    valid_mask = pair_meta_df["valid"].values
    n_valid = int(valid_mask.sum())
    print(f"Retained {n_valid:,} valid library x lineage profiles with >= 20 cells (out of {n_pairs:,} total pairs).")

    # Apply CPM -> log1p to valid combinations
    valid_counts = counts_accumulator[valid_mask, :]
    lib_sizes = valid_counts.sum(axis=1, keepdims=True)
    cpm = (valid_counts / np.maximum(lib_sizes, 1.0)) * 1e6
    valid_expr = np.log1p(cpm).astype(np.float32)

    valid_meta_df = pair_meta_df[valid_mask].copy().reset_index(drop=True)
    valid_meta_df["library_size"] = lib_sizes.flatten()
    valid_meta_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved metadata to {OUTPUT_CSV}")

    # Save to NPZ
    np.savez_compressed(
        OUTPUT_NPZ,
        counts=valid_counts,
        expression=valid_expr,
        sample_ids=valid_meta_df["sample_id"].values,
        studies=valid_meta_df["study"].values,
        disease=valid_meta_df["disease"].values,
        lineages=valid_meta_df["lineage"].values,
        cell_counts=valid_meta_df["cell_count"].values,
        feature_names=selected_genes,
    )
    print(f"Saved lineage pseudobulk to {OUTPUT_NPZ} ({OUTPUT_NPZ.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Total time elapsed: {time.time() - t_start:.1f}s")

if __name__ == "__main__":
    main()
