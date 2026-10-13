#!/usr/bin/env python3
"""Explore and summarize lung single-cell datasets in CELLxGENE Census in batches.

Queries CELLxGENE Census for primary human lung data, aggregates cell and donor
counts per (dataset_id, disease), cross-references against the 16 existing atlas
studies to identify overlaps, and saves reports/census_lung_summary.csv.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cellxgene_census
import pandas as pd
import pyarrow as pa

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_CSV = REPORTS_DIR / "census_lung_summary.csv"

# The 16 studies present in the existing lung atlas
ATLAS_STUDIES = [
    "ERP114453",
    "GSE122960",
    "GSE128033",
    "GSE128169",
    "GSE132771",
    "GSE135893",
    "GSE136831",
    "GSE150068",
    "GSE150247",
    "GSE158127",
    "GSE168191",
    "GSE171541",
    "GSE171668",
    "GSE173896",
    "GSE196341",
    "GSE227136",
]

# Known publications / DOIs / accessions associated with the atlas studies
# to ensure comprehensive overlap detection
ATLAS_PATTERNS = {
    "ERP114453": ["ERP114453", "PRJEB31843", "10.1183/13993003.01807-2019", "Madissoon"],
    "GSE122960": ["GSE122960", "10.1164/rccm.201812-2410OC", "Reyfman"],
    "GSE128033": ["GSE128033", "10.1038/s41591-020-0901-9", "Morse"],
    "GSE128169": ["GSE128169", "10.1126/sciadv.aba1983", "Peyser"],
    "GSE132771": ["GSE132771", "10.1016/j.cell.2020.05.010", "10.1038/s41586-020-2922-4", "Travaglini", "Krasnow Lab Human Lung Cell Atlas"],
    "GSE135893": ["GSE135893", "10.1126/sciadv.aba1983", "Habermann"],
    "GSE136831": ["GSE136831", "10.1126/sciadv.aba1983", "Adams"],
    "GSE150068": ["GSE150068", "10.1038/s41467-020-18974-4", "Valenzi"],
    "GSE150247": ["GSE150247", "10.1016/j.immuni.2020.06.001", "Delorey"],
    "GSE158127": ["GSE158127", "10.1016/j.cell.2021.04.027", "Melms"],
    "GSE168191": ["GSE168191", "10.1038/s41586-021-03569-1", "Sikkema"],
    "GSE171541": ["GSE171541", "10.1038/s41590-021-00938-9", "Chua"],
    "GSE171668": ["GSE171668", "10.1016/j.cell.2021.07.023", "Bharat"],
    "GSE173896": ["GSE173896", "10.1038/s41467-021-25601-6", "Sauler"],
    "GSE196341": ["GSE196341", "10.1183/13993003.02531-2021", "Deprez"],
    "GSE227136": ["GSE227136", "10.1038/s41591-023-02327-2", "Neumark"],
}


def check_disk_space(min_free_gb: float = 3.0) -> None:
    """Ensure at least min_free_gb is available before operations."""
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    free_gb = free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Disk space too low: {free_gb:.2f} GiB available, "
            f"requires at least {min_free_gb} GiB. Aborting."
        )
    print(f"[Resource Check] Free disk space: {free_gb:.2f} GiB (minimum required: {min_free_gb} GiB)")


def find_atlas_overlap(text_blob: str) -> tuple[str, str]:
    """Check if any text in titles/citations/DOIs matches existing atlas studies."""
    if not text_blob:
        return "NEW", ""
    for study_id, patterns in ATLAS_PATTERNS.items():
        for pat in patterns:
            # Case-insensitive search for accession/DOI/name
            if re.search(r"\b" + re.escape(pat) + r"\b", text_blob, re.IGNORECASE):
                return "OVERLAP", study_id
            if pat.lower() in text_blob.lower() and len(pat) > 5:
                return "OVERLAP", study_id
    return "NEW", ""


def main() -> None:
    check_disk_space(3.0)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Census Query] Connecting to CELLxGENE Census...")
    with cellxgene_census.open_soma() as census:
        # 1. Fetch dataset metadata
        print("[Census Query] Fetching dataset metadata table...")
        datasets_soma = census["census_info"]["datasets"].read().concat().to_pandas()
        datasets_map = {}
        for _, row in datasets_soma.iterrows():
            datasets_map[row["dataset_id"]] = {
                "dataset_title": str(row.get("dataset_title", "")),
                "collection_id": str(row.get("collection_id", "")),
                "collection_name": str(row.get("collection_name", "")),
                "collection_doi": str(row.get("collection_doi", "")),
                "citation": str(row.get("citation", "")),
            }
        print(f"[Census Query] Loaded metadata for {len(datasets_map)} datasets.")

        # 2. Query human lung primary obs in Arrow table batches
        print("[Census Query] Iterating over human lung obs (tissue_general == 'lung', is_primary_data == True)...")
        obs = census["census_data"]["homo_sapiens"].obs
        obs_reader = obs.read(
            value_filter="tissue_general == 'lung' and is_primary_data == True",
            column_names=["dataset_id", "disease", "donor_id"],
        )

        cell_counts: dict[tuple[str, str], int] = defaultdict(int)
        donor_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
        total_cells_processed = 0
        batch_idx = 0

        # Read tables and iterate in 100,000-cell RecordBatches to protect RAM
        for table in obs_reader:
            for batch in table.to_batches(max_chunksize=100_000):
                batch_idx += 1
                batch_df = batch.to_pandas()
                batch_cells = len(batch_df)
                total_cells_processed += batch_cells

                grouped = batch_df.groupby(["dataset_id", "disease"], observed=True)
                for (ds_id, dis), sub_df in grouped:
                    cell_counts[(ds_id, dis)] += len(sub_df)
                    donor_sets[(ds_id, dis)].update(sub_df["donor_id"].dropna().unique())

                if batch_idx % 10 == 0:
                    print(f"  Processed {batch_idx} batches ({total_cells_processed:,} cells)...")

        print(f"[Census Query] Finished: processed {batch_idx} batches, {total_cells_processed:,} total primary lung cells.")

    # 3. Build summary DataFrame
    records = []
    for (ds_id, disease), count in cell_counts.items():
        meta = datasets_map.get(ds_id, {})
        donors = donor_sets[(ds_id, disease)]
        
        # Build search text blob for overlap detection
        search_blob = " ".join([
            meta.get("dataset_title", ""),
            meta.get("collection_name", ""),
            meta.get("citation", ""),
            meta.get("collection_doi", ""),
        ])
        overlap_status, overlapping_study = find_atlas_overlap(search_blob)

        records.append({
            "dataset_id": ds_id,
            "dataset_title": meta.get("dataset_title", ""),
            "collection_name": meta.get("collection_name", ""),
            "collection_doi": meta.get("collection_doi", ""),
            "citation": meta.get("citation", ""),
            "disease": disease,
            "cell_count": count,
            "donor_count": len(donors),
            "overlap_status": overlap_status,
            "overlapping_atlas_study": overlapping_study,
        })

    df = pd.DataFrame(records)
    df.sort_values(by=["overlap_status", "disease", "cell_count"], ascending=[True, True, False], inplace=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Output] Wrote detailed summary to {OUTPUT_CSV}")

    # 4. Print Per-Disease Summary Table
    print("\n" + "=" * 95)
    print("CELLxGENE CENSUS HUMAN LUNG DATA: PER-DISEASE SUMMARY")
    print("=" * 95)
    disease_summary = df.groupby("disease", as_index=False).agg(
        dataset_count=("dataset_id", "nunique"),
        total_cells=("cell_count", "sum"),
        total_donors=("donor_count", "sum"),
    ).sort_values("total_cells", ascending=False)
    print(disease_summary.to_string(index=False))

    # 5. Print Overlap vs New Summary
    print("\n" + "=" * 95)
    print("ATLAS OVERLAP BREAKDOWN")
    print("=" * 95)
    overlap_summary = df.groupby(["overlap_status", "disease"], as_index=False).agg(
        dataset_count=("dataset_id", "nunique"),
        total_cells=("cell_count", "sum"),
        total_donors=("donor_count", "sum"),
    ).sort_values(["overlap_status", "total_cells"], ascending=[True, False])
    print(overlap_summary.to_string(index=False))

    # 6. Print Unique Datasets with Overlap Status
    print("\n" + "=" * 95)
    print("DATASETS OVERVIEW (NEW vs OVERLAP)")
    print("=" * 95)
    ds_agg = df.groupby(["dataset_id", "dataset_title", "collection_name", "overlap_status", "overlapping_atlas_study"], as_index=False).agg(
        total_cells=("cell_count", "sum"),
        diseases=("disease", lambda x: ", ".join(sorted(x.unique()))),
        total_donors=("donor_count", "sum"),
    ).sort_values(by=["overlap_status", "total_cells"], ascending=[True, False])
    for _, r in ds_agg.iterrows():
        overlap_str = f"[{r['overlap_status']}]" + (f" -> Overlaps with {r['overlapping_atlas_study']}" if r['overlapping_atlas_study'] else "")
        print(f"- {overlap_str}")
        print(f"  Dataset ID: {r['dataset_id']}")
        print(f"  Title: {r['dataset_title'][:80]}")
        print(f"  Collection: {r['collection_name'][:80]}")
        print(f"  Cells: {r['total_cells']:,} | Donors: {r['total_donors']} | Diseases: {r['diseases']}\n")


if __name__ == "__main__":
    main()
