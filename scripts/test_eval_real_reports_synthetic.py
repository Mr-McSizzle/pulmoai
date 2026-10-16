#!/usr/bin/env python3
"""Generate temporary synthetic test folder and ground-truth CSV to test eval_real_reports.py end-to-end.

Creates:
- /tmp/eval_real_test/
  - report_01_clean.png (acceptable quality, clean text)
  - report_02_mild.png (acceptable quality, mild noise)
  - report_03_refused.png (severe blur, quality gate should REFUSE)
  - ground_truth.csv with columns: file,analyte,value,unit
"""

import csv
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import pymupdf

from scripts.audit_ocr_pass8 import render_pdf_bytes, render_raster_image

TEMP_DIR = Path("/tmp/eval_real_test")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Document 1: Clean
doc1_text = """CITY DIAGNOSTICS LABORATORY
CLINICAL BIOCHEMISTRY REPORT
Patient: John Doe    Age: 45    Sex: Male

TEST NAME               RESULT   UNIT     REFERENCE INTERVAL
FASTING GLUCOSE         98.0     mg/dL    70.0 - 99.0
GLYCATED HEMOGLOBIN     5.4      %        <= 5.7
TOTAL CHOLESTEROL       185.0    mg/dL    < 200.0
HDL CHOLESTEROL         52.0     mg/dL    > 40.0
TRIGLYCERIDES           120.0    mg/dL    < 150.0

Verified by Medical Technologist
"""

# Document 2: Mild noise
doc2_text = """REGIONAL HOSPITAL LAB
METABOLIC PANEL REPORT
Patient: Jane Smith    Age: 52    Sex: Female

TEST NAME               RESULT   UNIT     REFERENCE INTERVAL
FASTING GLUCOSE         142.0    mg/dL    70.0 - 99.0
SERUM CREATININE        0.85     mg/dL    0.50 - 1.10
TOTAL CHOLESTEROL       225.0    mg/dL    < 200.0

Verified by Laboratory Director
"""

# Document 3: Severe blur (to trigger quality gate refusal)
doc3_text = """METROPOLITAN HEALTH LAB
ROUTINE LAB RESULTS
Patient: Alex Ray    Age: 38    Sex: Male

TEST NAME               RESULT   UNIT     REFERENCE INTERVAL
HEMOGLOBIN              14.5     g/dL     13.5 - 17.5
SERUM CREATININE        1.05     mg/dL    0.70 - 1.30

Authorized Signature
"""

pdf1 = render_pdf_bytes(doc1_text)
img1 = render_raster_image(pdf1, dpi=200)
img1.save(TEMP_DIR / "report_01_clean.png")

pdf2 = render_pdf_bytes(doc2_text)
img2 = render_raster_image(pdf2, dpi=200)
# Add mild noise
arr2 = np.array(img2, dtype=np.float32)
arr2 = np.clip(arr2 + np.random.RandomState(42).normal(0, 1.5, arr2.shape), 0, 255).astype(np.uint8)
img2_noisy = Image.fromarray(arr2)
img2_noisy.save(TEMP_DIR / "report_02_mild.png")

pdf3 = render_pdf_bytes(doc3_text)
img3 = render_raster_image(pdf3, dpi=200)
# Severe blur
img3_blurred = img3.filter(ImageFilter.GaussianBlur(radius=4.0))
img3_blurred.save(TEMP_DIR / "report_03_refused.png")

# Create ground_truth.csv with columns: file,analyte,value,unit
gt_rows = [
    {"file": "report_01_clean.png", "analyte": "fasting_glucose", "value": 98.0, "unit": "mg/dL"},
    {"file": "report_01_clean.png", "analyte": "hba1c", "value": 5.4, "unit": "%"},
    {"file": "report_01_clean.png", "analyte": "total_cholesterol", "value": 185.0, "unit": "mg/dL"},
    {"file": "report_01_clean.png", "analyte": "hdl_cholesterol", "value": 52.0, "unit": "mg/dL"},
    {"file": "report_01_clean.png", "analyte": "triglycerides", "value": 120.0, "unit": "mg/dL"},
    {"file": "report_02_mild.png", "analyte": "fasting_glucose", "value": 142.0, "unit": "mg/dL"},
    {"file": "report_02_mild.png", "analyte": "serum_creatinine", "value": 0.85, "unit": "mg/dL"},
    {"file": "report_02_mild.png", "analyte": "total_cholesterol", "value": 225.0, "unit": "mg/dL"},
    {"file": "report_03_refused.png", "analyte": "hemoglobin", "value": 14.5, "unit": "g/dL"},
    {"file": "report_03_refused.png", "analyte": "serum_creatinine", "value": 1.05, "unit": "mg/dL"},
]

gt_path = TEMP_DIR / "ground_truth.csv"
with open(gt_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "analyte", "value", "unit"])
    writer.writeheader()
    writer.writerows(gt_rows)

print(f"Generated 3 synthetic reports and ground_truth.csv in {TEMP_DIR}")
