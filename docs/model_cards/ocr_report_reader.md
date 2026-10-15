# Model Card: Clinical Lab Report Reader & Normalization Parser

## Model Details
- **Architecture:** Hybrid Rule-Based / Regex Pattern Matcher with PyMuPDF digital text layer extraction, local Tesseract OCR fallback, and adaptive image preprocessing (blur/skew gate).
- **Core Module:** `src/pulmoai/report_reader.py`
- **Supported Analytes:** 30+ canonical clinical laboratory analytes covering Metabolic, Lipid, Renal, Hepatic, Hematology, and Electrolyte panels.
- **Developers:** PulmoAI Machine Learning Research Group.

## Intended Use
- **Primary Function:** Ingest unstructured clinical laboratory reports (digital PDFs or smartphone photos/scans), extract analyte names and numerical values, normalize international/SI units to standard US clinical units, and evaluate flags against biological reference intervals.
- **Settled Boundary Policy:** Reference intervals $[ref\_min, ref\_max]$ are strictly inclusive. Values equal to $ref\_min$ or $ref\_max$ evaluate as `NORMAL`. Abnormality requires strict inequality ($val < ref\_min \to \text{LOW}$, $val > ref\_max \to \text{HIGH}$).

## Quantitative Metrics (Honest Holdout Evaluation on 4 Unseen Layouts, 350 Entities per Condition)
- **Digital PDF Layer Extraction Accuracy:** **92.57%** (324 / 350 entities parsed accurately)
- **Degraded Image Scans:**
  - Raw Rotated (+3.5 deg) Accuracy: 67.71% (237 / 350)
  - Adaptive Preprocessed Rotated Accuracy: 86.29% (302 / 350; +18.57% gain)
- **Silent Numerical Errors on Clean PDF:** **0 / 350 (0.0%)**

## Categorization of Clean PDF Failures (26 total failures on holdout)
1. **One-Sided Reference Interval Mismatches (13 cases, 50.0%):** Hospital templates printed one-sided biological reference ranges (`< 200 mg/dL` for Total Cholesterol, `> 40 mg/dL` for HDL). The numerical values were 100% accurately extracted, but the ground-truth evaluator assigned LOW/HIGH based on two-sided physiological limits, while the parser evaluated strictly against the printed one-sided limit.
2. **Unmatched Analyte Alias (13 cases, 50.0%):** Template A printed HDL as `"CHOLESTEROL HDL DIRECT"`. The frozen alias dictionary did not match this exact composite string, leading to analyte omission (not detected).

## Safety and Risk Engine Gate
- **Transfer Policy:** 100% of extracted lab values require explicit user verification and checkbox confirmation before transfer to the chronic disease risk engine.
- **Low Confidence Alert:** Any analyte extracted with low confidence ($< 0.70$) triggers a warning alert banner in the user interface. Automated headless transfer without user review is strictly prevented.
