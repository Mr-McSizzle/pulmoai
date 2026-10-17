# Model Card: Clinical Lab Report Reader & Normalization Parser

## Model Details
- **Architecture:** Hybrid Rule-Based / Regex Pattern Matcher with PyMuPDF digital text layer extraction, local Tesseract OCR fallback, and multi-factor image quality gate with adaptive deskewing and denoising preprocessing.
- **Core Module:** `src/pulmoai/report_reader.py`
- **Supported Analytes:** 30+ canonical clinical laboratory analytes covering Metabolic, Lipid, Renal, Hepatic, Hematology, and Electrolyte panels.
- **Developers:** PulmoAI Machine Learning Research Group.

## Intended Use
- **Primary Function:** Ingest unstructured clinical laboratory reports (digital PDFs or smartphone photos/scans), extract analyte names and numerical values, normalize international/SI units to standard US clinical units, and evaluate flags against biological reference intervals.
- **Fictional Layout Notice:** All synthetic laboratory report templates and headers used in evaluation and testing are fictional layouts and do not reproduce any real commercial or hospital laboratory's format. Real-world clinical performance on proprietary commercial formats remains unverified.
- **Settled Boundary Policy:** Reference intervals $[ref\_min, ref\_max]$ are strictly inclusive. Values equal to $ref\_min$ or $ref\_max$ evaluate as `NORMAL`. Abnormality requires strict inequality ($val < ref\_min \to \text{LOW}$, $val > ref\_max \to \text{HIGH}$).

## Multi-Factor Quality Gate Architecture (Pass 9b)
The image parser enforces a two-tier quality gate to avoid brittle binary failures and prevent OCR hallucination:

1. **Severe Degradation Gate (Hard Refusal, Zero Values Extracted):**
   - **Severe Defocus Blur:** Laplacian variance $\text{var} < 20.0$.
   - **Extreme Tilt / Skew:** Evaluated via horizontal projection variance sweep across $[-15^\circ, +15^\circ]$; $|\theta| \ge 15.0^\circ$ is rejected.
   - **Destructive Optical Noise:** Fast Immerkaer Laplacian noise standard deviation $\sigma \ge 12.0$.
   - **Severe Contrast Fading:** Dynamic range ($\max(I) - \min(I)$) $< 30.0$.
   - **Safety Enforcement:** Returns `"image quality too low, please re-scan"` and extracts 0 values, eliminating silent OCR hallucinations.

2. **Mild Degradation Zone (Accepted with Warning & Discounting):**
   - **Detection Criteria:** $\sigma \ge 1.35$, dynamic range $< 120.0$, Laplacian variance $< 80.0$, or $|\theta| \ge 2.5^\circ$.
   - **Adaptive Processing:** Applies Otsu thresholding, median denoising, and automated deskewing.
   - **Clinical Confidence Policy:** All extracted analytes are tagged with `confidence_category = "unconfirmed, low confidence"` and confidence scores are discounted by $0.60\times$, alerting downstream systems and requiring explicit user verification before any risk engine transfer.
   - **Development Validation:** On scanner sensor noise ($\sigma \approx 1.60$) and phone shadows ($\sigma \approx 1.72$), the multi-factor gate achieves 0.0% false refusal (66/66 accepted as unconfirmed), resolving the 100% false-refusal failure of older rigid gates [Tested - `reports/audit_ocr_realistic_dev.json:scanner_sensor_noise`].

## 300 DPI Resolution Diagnosis & Root Cause
In Pass 6 holdout testing, 34 entities failed detection at 300 DPI despite succeeding at lower resolutions.

- **Controlled Scale-Factor Experiment [Tested - `reports/diagnose_300dpi_undetected.json:scale_factor_experiment`]:**
  - Varying exclusively rasterization DPI across the exact 34 failure instances:
    - 150 DPI: **34 / 34 recovered (100.0%)**
    - 200 DPI: **34 / 34 recovered (100.0%)**
    - 250 DPI: 31 / 34 recovered (91.18%)
    - 300 DPI: 0 / 34 recovered (0.00%)
    - 350 DPI: 27 / 34 recovered (79.41%)
    - 400 DPI: 27 / 34 recovered (79.41%)
- **Root Cause Analysis:**
  - Tesseract OCR glyph segmentation exhibits severe resolution sensitivity when rendering thin monospace fonts (Courier 8.5pt at 300 DPI produces wide inter-character spacing and 1-pixel strokes).
  - Tesseract dropped the terminal letter "e" from `"Serum Creatinine"` (reading `"Serum Creatinin"`), which failed regex extraction because `"creatinin"` was omitted from legacy alias dictionaries.
  - Total hemoglobin rows were split across multiple lines (`"HEMOGLOBIN"` on one line, value on the next).
- **Remediations Deployed:**
  - Added `"creatinin"` alias to canonical serum creatinine mapping.
  - Normalized em-dash (`—`) and hyphen ligatures in reference range patterns.
  - Added multi-line analyte-to-value stitching.

## Quantitative Metrics: Pass 9c Holdout Benchmark (Independent Page Sets)
- **Holdout Suite:** 11 template families (Templates A-K), separate page sets per condition (seeds 4001-4013, 44 pages per condition, 308 ground-truth entities per condition; no pages reused).
- **Data Source:** `reports/audit_ocr_redesign_holdout.json`

| Condition | Pages Refused | Entities Accepted | Accuracy on Accepted | Silent Error Rate | Not Detected | Field Errors |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Digital PDF text layer (native)** (`clean_pdf`) | 0 / 44 (0.0%) | 304 | **98.70%** (304/308) | 0.00% | 4 | 0 |
| **Rasterized page at 150 DPI** (`clean_raster_150dpi`) | 0 / 44 (0.0%) | 281 | **90.58%** (279/308) | 0.65% | 27 | 2 |
| **Rasterized page at 200 DPI** (`clean_raster_200dpi`) | 0 / 44 (0.0%) | 231 | **74.68%** (230/308) | 0.32% | 77 | 1 |
| **Rasterized page at 300 DPI** (`clean_raster_300dpi`) | 0 / 44 (0.0%) | 228 | **74.03%** (228/308) | 0.00% | 80 | 0 |
| **JPEG compression quality Q=50** (`mild_jpeg_q50`) | 0 / 44 (0.0%) | 279 | **84.09%** (259/308) | 6.49% | 29 | 20 |
| **Additive fibrous paper texture** (`mild_paper_grain`) | 0 / 44 (0.0%) | 282 | **83.12%** (256/308) | 8.44% | 26 | 26 |
| **CCD scanner sensor shot noise & columnar streaks** (`mild_scanner_sensor`) | 0 / 44 (0.0%) | 276 | **82.47%** (254/308) | 7.14% | 32 | 22 |
| **Mobile phone camera illumination gradient & shadow** (`phone_lighting_shadow`) | 0 / 44 (0.0%) | 274 | **86.69%** (267/308) | 2.27% | 34 | 7 |
| **Low toner / faded contrast print** (`faded_print_low_contrast`) | 44 / 44 (100.0%) | 0 | N/A (Gate Refused) | 0.00% | 308 | 0 |
| **Slight 2.5-degree perspective tilt** (`perspective_tilt_2p5deg`) | 0 / 44 (0.0%) | 244 | **77.92%** (240/308) | 1.30% | 64 | 4 |
| **Severe defocus blur (radius=3.5)** (`severe_defocus_blur_r35`) | 44 / 44 (100.0%) | 0 | N/A (Gate Refused) | 0.00% | 308 | 0 |
| **Extreme 18-degree rotation** (`extreme_rotation_18deg`) | 22 / 44 (50.0%) | 26 | **7.79%** (24/308) | 0.00% | 282 | 2 |
| **Heavy sensor noise (sigma=15.0)** (`heavy_gaussian_noise_sigma15`) | 0 / 44 (0.0%) | 268 | **79.55%** (245/308) | 7.47% | 40 | 23 |

- **False Rejection Rate on Readable Conditions:** **0.0%** (0 / 396 pages across all 9 readable conditions).
- **Phone Illumination Improvement:** Background division normalization increased accuracy on phone lighting shadow from 7.36% (Pass 9b) to **86.69%** (Pass 9c) and reduced silent error rate from 31.17% to **2.27%**.

## Real-World Report Evaluation Protocol
- **Evaluation Tool:** `scripts/eval_real_reports.py`
- **Protocol Documentation:** `docs/REAL_EVAL.md`
- **Supported Inputs:** Digital PDFs, JPEG, PNG, TIFF lab report scans paired with ground-truth CSV (`file, analyte, value, unit, flag`).
- **Validation Test [Tested - `reports/eval_real_reports_test.json`]:** Evaluated on 3 synthetic test files (clean PDF, mild degradation, blurred) with ground truth:
  - Clean PDF: Gate ACCEPT, Accuracy 100.0% (5/5).
  - Mild Degradation: Gate ACCEPT, Accuracy 100.0% (3/3), tagged as unconfirmed.
  - Blurred Scan: Gate REFUSE, 0 values accepted, preventing hallucination.
  - Overall accuracy on test set: **80.00%** (8 / 10, 95% Wilson CI: [49.02%, 94.33%]).

## Safety and Risk Engine Gate
- **Transfer Policy:** 100% of extracted lab values require explicit user verification and checkbox confirmation before transfer to the chronic disease risk engine. Headless automated transfer is disabled.
- **Low Confidence Alert:** Any analyte extracted with low confidence ($< 0.70$) or marked as `unconfirmed, low confidence` triggers a warning alert banner in the user interface.
