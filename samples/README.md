# Synthetic Demo & Test Samples (Research Use Only)

This directory contains demonstration and testing sample files designed for quick-start validation, presentations, frontend testing, and interactive walkthroughs.

---

## 1. Single-Cell RNA Gene Expression Samples (Module 1 / Single-Cell RNA Page)

Compatible with the `/analyze` UI page and the `POST /api/predict` / `POST /api/validate` endpoints.
All files strictly contain the 30,566 gene identifiers matching the model input contract with raw integer-like pseudobulk counts.

### Primary Sample Files
- **`single_cell_rna_sample.csv`** (Format 02: Long CSV):
  - Primary single-cell RNA test input in standard long format (`gene,count`).
  - Contains 30,566 gene rows.
  - Predicted label: **Healthy Control (`Donor`)** (Probability ~90.6%).
  - Ideal for testing drag-and-drop on `/analyze`.

- **`single_cell_rna_sample_wide.csv`** (Format 01: Wide CSV):
  - Wide-format single-cell RNA profile (`sample_id` + 30,566 gene columns in 1 profile row).
  - Demonstrates the platform's support for horizontal/wide expression matrices.
  - Predicted label: **Healthy Control (`Donor`)** (Probability ~90.6%).

### Condition-Specific Single-Cell RNA Samples
- **`scrna_sample_donor.csv`**:
  - Healthy control donor profile (`gene,count`).
  - Predicted label: **`Donor`** (90.6% confidence).
- **`scrna_sample_copd.csv`**:
  - Chronic Obstructive Pulmonary Disease profile (`gene,count`).
  - Predicted label: **`COPD`** (77.1% confidence).
- **`scrna_sample_pf.csv`**:
  - Pulmonary Fibrosis profile (`gene,count`).
  - Predicted label: **`PF`** (91.1% confidence).
- **`scrna_sample_covid19.csv`**:
  - Severe COVID-19 respiratory profile (`gene,count`).
  - Predicted label: **`COVID19`** (71.2% confidence).
- **`synthetic_lung_pseudobulk.csv`**:
  - Synthetic pseudobulk count profile generated with elevated pulmonary marker genes (`SFTPC`, `SFTPA1`, `AGER`, `SCGB1A1`, etc.).

---

## 2. Clinical Lab Report Extraction Samples (Module 3 / Lab Extraction Page)

Compatible with the `/reports` UI page and the `POST /api/report/extract` endpoint:

- **`synthetic_lab_report.pdf`**:
  - Digital vector PDF containing a standard metabolic and lipid panel (Fasting Glucose, HbA1c, Total Cholesterol, HDL, Triglycerides, Creatinine, Hemoglobin).
  - Demonstrates instant PyMuPDF digital text layer extraction with 100% numerical accuracy.

- **`synthetic_rotated_scan.png`**:
  - High-resolution scan of the identical lab report with +3.5 degree rotation and optical scan noise.
  - Demonstrates deskew preprocessing and local Tesseract OCR fallback with human verification confirmation.

---

## 3. Medical Imaging Screening Samples (Module 5 / Imaging Page)

Located in `samples/images/`:
- Chest X-ray and CT screening image samples across normal and pathological findings.

---

## Quick Verification

You can verify all single-cell RNA sample inputs against the backend service with:
```bash
python scripts/generate_scrna_samples.py
```
Or test via HTTP API:
```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "file=@samples/single_cell_rna_sample.csv"
```

---

## Privacy & Provenance Notice
All files in this directory are demonstration and benchmark profiles (synthetic or public research atlas pseudobulks). They contain **ZERO** real patient Protected Health Information (PHI) or private clinical records.
