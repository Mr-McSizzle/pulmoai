#!/usr/bin/env python3
"""Generate Single-Cell RNA sample input files for testing and frontend evaluation.

Generates into samples/:
1. single_cell_rna_sample.csv: Standard long-format CSV (gene,count) matching the 30,566 features.
2. single_cell_rna_sample_wide.csv: Wide-format CSV (sample_id,gene_1,gene_2,...) with 1 profile row.
3. scrna_sample_donor.csv: Representative healthy donor scRNA profile.
4. scrna_sample_copd.csv: Representative COPD scRNA profile.
5. scrna_sample_pf.csv: Representative Pulmonary Fibrosis scRNA profile.
6. scrna_sample_covid19.csv: Representative COVID-19 scRNA profile.
"""

from __future__ import annotations

import csv
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "samples"
DATA_INPUTS_DIR = PROJECT_ROOT / "data" / "sample_inputs"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_scrna_samples")


def ensure_samples_dir() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def generate_scrna_samples() -> None:
    ensure_samples_dir()

    # Ensure source sample files exist in samples/
    disease_files = [
        "scrna_sample_donor.csv",
        "scrna_sample_copd.csv",
        "scrna_sample_pf.csv",
        "scrna_sample_covid19.csv",
    ]

    for fname in disease_files:
        dest_path = SAMPLES_DIR / fname
        if not dest_path.exists():
            src_path = DATA_INPUTS_DIR / fname
            if src_path.exists():
                shutil.copy2(src_path, dest_path)
                logger.info("Copied %s -> %s", src_path, dest_path)

    # Use donor profile as the canonical single_cell_rna_sample
    donor_path = SAMPLES_DIR / "scrna_sample_donor.csv"
    if not donor_path.exists():
        raise FileNotFoundError(f"Source sample not found at {donor_path}")

    genes: list[str] = []
    counts: list[str] = []

    with donor_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row:
                genes.append(row[0].strip())
                counts.append(row[1].strip())

    logger.info("Loaded %d genes from %s", len(genes), donor_path.name)

    # 1. Generate canonical Long-Format CSV
    long_path = SAMPLES_DIR / "single_cell_rna_sample.csv"
    with long_path.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["gene", "count"])
        for g, c in zip(genes, counts):
            writer.writerow([g, c])
    logger.info("Generated canonical Long CSV: %s (%d genes)", long_path, len(genes))

    # 2. Generate canonical Wide-Format CSV
    wide_path = SAMPLES_DIR / "single_cell_rna_sample_wide.csv"
    with wide_path.open("w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["sample_id"] + genes)
        writer.writerow(["single_cell_donor_001"] + counts)
    logger.info("Generated canonical Wide CSV: %s (1 row, %d columns)", wide_path, len(genes) + 1)


def verify_samples() -> None:
    from pulmoai.service import PulmoAIService

    service = PulmoAIService.load()
    files_to_test = [
        "single_cell_rna_sample.csv",
        "single_cell_rna_sample_wide.csv",
        "scrna_sample_donor.csv",
        "scrna_sample_copd.csv",
        "scrna_sample_pf.csv",
        "scrna_sample_covid19.csv",
    ]

    logger.info("Verifying single-cell RNA samples with PulmoAIService...")
    for fname in files_to_test:
        fpath = SAMPLES_DIR / fname
        if not fpath.exists():
            logger.warning("File %s does not exist, skipping", fname)
            continue
        data = fpath.read_bytes()
        val = service.validate_upload(data)
        if not val["valid"]:
            raise ValueError(f"Validation failed for {fname}: {val}")
        pred = service.predict_upload(data)
        logger.info(
            "[PASSED] %s: Validated (%d genes) -> Predicted: %s (Probability: %.1f%%)",
            fname,
            val["recognized_gene_count"],
            pred["predicted_class"],
            pred["class_probabilities"][pred["predicted_class"]] * 100,
        )


def main() -> None:
    generate_scrna_samples()
    verify_samples()


if __name__ == "__main__":
    main()
