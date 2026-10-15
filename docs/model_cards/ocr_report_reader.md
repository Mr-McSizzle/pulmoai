# Model Card: Clinical Lab Report Reader & Normalization Parser

## Model Details
- **Architecture:** Hybrid Rule-Based / Regex Pattern Matcher with PyMuPDF digital text layer extraction, local Tesseract OCR fallback, and adaptive image preprocessing (blur/skew gate).
- **Core Module:** `src/pulmoai/report_reader.py`
- **Supported Analytes:** 30+ canonical clinical laboratory analytes covering Metabolic, Lipid, Renal, Hepatic, Hematology, and Electrolyte panels.
- **Developers:** PulmoAI Machine Learning Research Group.

## Intended Use
- **Primary Function:** Ingest unstructured clinical laboratory reports (digital PDFs or smartphone photos/scans), extract analyte names and numerical values, normalize international/SI units to standard US clinical units, and evaluate flags against biological reference intervals.
- **Settled Boundary Policy:** Reference intervals $[ref\_min, ref\_max]$ are strictly inclusive. Values equal to $ref\_min$ or $ref\_max$ evaluate as `NORMAL`. Abnormality requires strict inequality ($val < ref\_min \to \text{LOW}$, $val > ref\_max \to \text{HIGH}$).

## Quantitative Metrics (Fresh Holdout Evaluation: Seed 777, 6 Layouts A-F, 420 Entities per Condition)
- **Digital PDF Layer Extraction Accuracy:** **94.05%** (395 / 420 entities parsed accurately)
- **Silent Numerical Errors on Clean PDF:** **0 / 420 (0.00%)**
- **Missed Analytes (Not Detected) on Clean PDF:** **0 / 420 (0.00%)**
- **Flag Mismatch Only Rate (Clean PDF):** 5.95% (25 / 420; numerical values 100% correct)
- **Degraded Image Scans (420 Entities per Condition):**
  - Rotated (+3.5 deg) Raw Accuracy: 66.19% (278 / 420, silent errors: 15.48%)
  - Rotated (+3.5 deg) Adaptive Preprocessed Accuracy: 85.24% (358 / 420, silent errors: 2.62%; +19.05% accuracy improvement)
  - Gaussian Blur Accuracy: 54.52% (229 / 420, silent errors: 19.76%)
  - High JPEG Compression Accuracy: 85.95% (361 / 420, silent errors: 3.57%)
  - Perspective Skew Accuracy: 86.67% (364 / 420, silent errors: 4.29%)

## Parser Improvements in Pass 5
1. **One-Sided Reference Interval Normalization:** Fixed parser to recognize one-sided printed ranges (`< 200`, `> 40`, `<= 5.7`, `>= 10`, `≤`, `≥`) and apply consistent inclusive boundary comparison.
2. **Composite Analyte Matching:** Added direct composite aliases (e.g., `"CHOLESTEROL HDL DIRECT"` -> `hdl_cholesterol`, `"CHOLESTEROL LDL DIRECT"` -> `ldl_cholesterol`).
3. **Adaptive Image Preprocessing Per Image:** Applied blur and skew estimation to selectively trigger deskewing and adaptive binarization only when needed.

## Safety and Risk Engine Gate
- **Transfer Policy:** 100% of extracted lab values require explicit user verification and checkbox confirmation before transfer to the chronic disease risk engine.
- **Low Confidence Alert:** Any analyte extracted with low confidence ($< 0.70$) triggers a warning alert banner in the user interface. Automated headless transfer without user review is strictly prevented.
