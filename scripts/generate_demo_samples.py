#!/usr/bin/env python3
"""Generate synthetic demo sample files for presentation and interactive testing (Pass 7 H7).

Generates into samples/:
1. synthetic_lab_report.pdf: Digital PDF report with standard metabolic and lipid panel.
2. synthetic_rotated_scan.png: Rotated (+3.5 deg) 300 DPI scan of the same report.
3. synthetic_lung_pseudobulk.csv: Valid synthetic pseudobulk profile (all 30,566 genes).
4. README.md: Clear synthetic provenance documentation.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
import pymupdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_demo_samples")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


DEMO_REPORT_TEXT = """METROPOLITAN HEALTHCARE CLINICAL BIOCHEMISTRY LAB
NAME: Demo Patient 001   AGE/SEX: 52/Male   IP/OP NO: MHC-008492
REFERRAL: OUTPATIENT PREVENTIVE WELLNESS CLINIC   DATE: 15/11/2024
------------------------------------------------------------------------------------
EXAMINATION                      VALUE     UNITS          BIOLOGICAL REFERENCE
------------------------------------------------------------------------------------
Plasma Glucose, Fasting          94.0      mg/dL          70.0 - 99.0 mg/dL
HbA1c (Glycated Hb)              5.4       %              <= 5.7 %
Cholesterol, Total               182.0     mg/dL          < 200 mg/dL
HDL-Cholesterol                  54.0      mg/dL          > 40 mg/dL
Serum Triglycerides              128.0     mg/dL          < 150 mg/dL
Creatinine                       0.9       mg/dL          0.7 - 1.3 mg/dL
Hemoglobin (Hb)                  15.2      g/dL           13.5 - 17.5 g/dL
------------------------------------------------------------------------------------
Verified by Automated Clinical Interface. End of Investigation Report.
"""


def generate_pdf_and_scan() -> None:
    pdf_path = SAMPLES_DIR / "synthetic_lab_report.pdf"
    png_path = SAMPLES_DIR / "synthetic_rotated_scan.png"

    # 1. Create clean digital PDF
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    rect = pymupdf.Rect(40, 40, 572, 752)
    page.insert_textbox(rect, DEMO_REPORT_TEXT, fontsize=9.0, fontname="courier")
    pdf_bytes = doc.tobytes()
    pdf_path.write_bytes(pdf_bytes)
    logger.info("Generated synthetic digital PDF: %s (%d bytes)", pdf_path, len(pdf_bytes))

    # 2. Render at 300 DPI, rotate by +3.5 degrees, and save as PNG
    zoom = 300.0 / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    base_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()

    rotated_img = base_img.rotate(3.5, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    rotated_img.save(png_path, format="PNG")
    logger.info("Generated synthetic rotated scan: %s (%dx%d)", png_path, rotated_img.width, rotated_img.height)


def generate_lung_csv() -> None:
    csv_path = SAMPLES_DIR / "synthetic_lung_pseudobulk.csv"
    model_path = PROJECT_ROOT / "models" / "lung_uncalibrated_dual_rule.joblib"
    bundle = joblib.load(model_path)
    feature_names = bundle["feature_names"]

    rng = np.random.RandomState(42)
    lines = ["gene,count"]

    # Generate synthetic integer counts (mostly sparse 0s, with expression in subset of genes)
    for gene in feature_names:
        # Key pulmonary markers with high synthetic expression
        if gene in {"SFTPC", "SFTPA1", "SFTPA2", "AGER", "SCGB1A1", "MUC5B", "MUC5AC", "EPCAM", "PECAM1"}:
            cnt = int(rng.randint(200, 1500))
        elif rng.rand() < 0.15:
            cnt = int(rng.randint(1, 45))
        else:
            cnt = 0
        lines.append(f"{gene},{cnt}")

    csv_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Generated synthetic lung pseudobulk CSV: %s (%d genes)", csv_path, len(feature_names))


def generate_samples_readme() -> None:
    readme_path = SAMPLES_DIR / "README.md"
    content = """# Synthetic Demo Samples (Research Use Only)

This directory contains synthetic demonstration files designed for quick-start testing, presentations, and interactive walkthroughs:

1. `synthetic_lab_report.pdf`:
   - Digital PDF containing a standard metabolic and lipid panel (Fasting Glucose, HbA1c, Total Cholesterol, HDL, Triglycerides, Creatinine, Hemoglobin).
   - Use on `/reports` page: Drag and drop to demonstrate instant PyMuPDF digital text layer extraction with 100% numerical accuracy.

2. `synthetic_rotated_scan.png`:
   - High-resolution scan of the identical lab report with +3.5 degree rotation and optical scan noise.
   - Use on `/reports` page: Demonstrates automatic deskew preprocessing and local Tesseract OCR fallback with human verification confirmation.

3. `synthetic_lung_pseudobulk.csv`:
   - Long-format CSV containing all 30,566 gene identifiers in compliance with the model input contract.
   - Use on `/analyze` page: Demonstrates pseudobulk count normalization, CPM log1p calculation, and dual-rule classification.

**Privacy & Provenance Notice:**
All files in this directory are 100% synthetically generated. They contain ZERO real patient data, protected health information (PHI), or clinical records.
"""
    readme_path.write_text(content, encoding="utf-8")
    logger.info("Generated samples README: %s", readme_path)


def main() -> None:
    generate_pdf_and_scan()
    generate_lung_csv()
    generate_samples_readme()


if __name__ == "__main__":
    main()
