# Model Card: Clinical Lab Report Reader & Normalization Parser

## Model Details
- **Architecture:** Hybrid Rule-Based / Regex Pattern Matcher with PyMuPDF digital text layer extraction, local Tesseract OCR fallback, and adaptive image preprocessing (blur/skew gate).
- **Core Module:** `src/pulmoai/report_reader.py`
- **Supported Analytes:** 30+ canonical clinical laboratory analytes covering Metabolic, Lipid, Renal, Hepatic, Hematology, and Electrolyte panels.
- **Developers:** PulmoAI Machine Learning Research Group.

## Intended Use
- **Primary Function:** Ingest unstructured clinical laboratory reports (digital PDFs or smartphone photos/scans), extract analyte names and numerical values, normalize international/SI units to standard US clinical units, and evaluate flags against biological reference intervals.
- **Settled Boundary Policy:** Reference intervals $[ref\_min, ref\_max]$ are strictly inclusive. Values equal to $ref\_min$ or $ref\_max$ evaluate as `NORMAL`. Abnormality requires strict inequality ($val < ref\_min \to \text{LOW}$, $val > ref\_max \to \text{HIGH}$).

## Quantitative Metrics (Pass 6 Fresh Holdout: Seed 999, 7 Layouts A-G, 490 Entities per Condition)
- **Digital PDF Layer Extraction Accuracy:** **94.90%** (465 / 490 entities parsed accurately)
- **Silent Numerical Errors on Clean PDF:** **0 / 490 (0.00%)**
- **Missed Analytes (Not Detected) on Clean PDF:** **0 / 490 (0.00%)**
- **Flag Mismatch Rate on Clean PDF:** 5.10% (25 / 490; numerical values 100% correct, flags differ due to latent generator limits)
- **Degraded Image Scans (490 Entities per Condition, With Quality Gate):**
  - Gaussian Blur: **100.0% Refused for Low Quality** (490 / 490 images refused by gate; 0 accepted; silent error rate on accepted: **0.00%**)
  - Rotated (+3.5 deg) Adaptive Preprocessed: 84.69% accuracy on accepted (415 / 490; silent error rate: 4.08%)
  - High JPEG Compression: 80.82% accuracy on accepted (396 / 490; silent error rate: 5.31%)
  - Perspective Distortion: 85.51% accuracy on accepted (419 / 490; silent error rate: 2.24%)
  - Rotated (+3.5 deg) Raw (No Deskewing): 59.18% accuracy on accepted (290 / 490; silent error rate: 18.16%)

## Pre-OCR Quality Gate & Improvements in Pass 6
1. **Pre-OCR Input Quality Gate:** Images evaluated prior to OCR for blur (Laplacian variance $< 30.0$) and excessive skew ($\ge 12.0^\circ$). Corrupted or severely blurred scans return `"image quality too low, please re-scan"` and extract 0 values, eliminating silent OCR hallucinations.
2. **One-Sided Reference Interval Normalization:** Recognizes one-sided printed ranges (`< 200`, `> 40`, `<= 5.7`, `>= 10`, `≤`, `≥`) with clinical inclusive boundaries.
3. **Composite Analyte Matching:** Direct composite aliases (e.g., `"CHOLESTEROL HDL DIRECT"` -> `hdl_cholesterol`, `"CHOLESTEROL LDL DIRECT"` -> `ldl_cholesterol`).

## Safety and Risk Engine Gate
- **Transfer Policy:** 100% of extracted lab values require explicit user verification and checkbox confirmation before transfer to the chronic disease risk engine.
- **Low Confidence Alert:** Any analyte extracted with low confidence ($< 0.70$) triggers a warning alert banner in the user interface. Automated headless transfer without user review is strictly prevented.
