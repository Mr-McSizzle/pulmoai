#!/usr/bin/env python3
"""Compute MSigDB Hallmark pathway activities for Melms et al. using Decoupler."""

from __future__ import annotations

import time
from pathlib import Path

import decoupler as dc
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MELMS_NPZ = DATA_DIR / "melms_profiles.npz"
ATLAS_PW_CSV = DATA_DIR / "atlas_pathway_hallmark.csv"

OUTPUT_NPY = DATA_DIR / "melms_pathway_hallmark.npy"
OUTPUT_CSV = DATA_DIR / "melms_pathway_hallmark.csv"


def main() -> None:
    print("=" * 80)
    print("COMPUTING MSIGDB HALLMARK PATHWAY SCORES FOR MELMS (GSE171524)")
    print("=" * 80)

    if not MELMS_NPZ.exists():
        raise FileNotFoundError(f"Melms profiles not found at {MELMS_NPZ}")

    # 1. Load Melms profiles
    with np.load(MELMS_NPZ, allow_pickle=True) as m:
        expr = m["expression"].astype(np.float32)
        donors = m["donors"].astype(str)
        features = m["feature_names"].astype(str)

    print(f"Melms data: {len(donors)} donors x {len(features)} genes")
    expr_df = pd.DataFrame(expr, index=donors, columns=features)

    # 2. Get Hallmark network
    print("Loading Hallmark network via Decoupler...")
    hm = dc.op.hallmark(organism="human")
    print(f"Hallmark network: {len(hm)} interactions across {hm['source'].nunique()} pathways")

    # 3. Compute ULM activities
    t0 = time.time()
    acts, pvals = dc.mt.ulm(data=expr_df, net=hm)
    print(f"Computed ULM activities in {time.time() - t0:.2f}s! Raw shape: {acts.shape}")

    # 4. Align columns with atlas pathway CSV
    atlas_cols = pd.read_csv(ATLAS_PW_CSV, index_col=0).columns.tolist()
    print(f"Aligning with Atlas pathway column order ({len(atlas_cols)} pathways)...")

    missing_pathways = [col for col in atlas_cols if col not in acts.columns]
    if missing_pathways:
        raise ValueError(f"Missing pathways in Melms Decoupler output: {missing_pathways}")

    acts_aligned = acts[atlas_cols]
    print(f"Aligned shape: {acts_aligned.shape}")

    # 5. Save outputs
    np.save(OUTPUT_NPY, acts_aligned.values.astype(np.float32))
    acts_aligned.to_csv(OUTPUT_CSV)
    print(f"Saved {OUTPUT_NPY} ({OUTPUT_NPY.stat().st_size / 1024:.1f} KB)")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
