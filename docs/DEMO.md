# PulmoAI 5-Minute Demonstration Walkthrough & Presentation Guide

This guide provides a structured, 5-minute interactive walkthrough of the PulmoAI Multimodal Health Platform, designed for live presentations, technical audits, and clinical AI evaluations.

---

## 1. Quick Start: Launching the Local Server

Run the unified single-port application from the repository root:

```bash
# Ensure virtual environment is active
source .venv/bin/activate

# Launch single-port full-stack server (FastAPI backend + React 19 SPA)
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open your browser to: **`http://127.0.0.1:8000`**

---

## 2. 5-Minute Page-by-Page Interactive Walkthrough

### Minute 1: System Overview & Data Integrity (`/`)
1. **Navigate to:** `Overview` dashboard.
2. **Observe:**
   - Multi-module health status across all 5 clinical AI modules.
   - Live **Integrity Verified** badge: every underlying model file is cryptographically checked against its SHA-256 hash at boot (`/api/integrity`).
   - Clear estimand clarification banner: cross-sectional prevalence vs prospective survival incidence.
3. **What to click:** Click on the **Integrity Manifest** modal to view the exact SHA-256 checksums matching the audit report.

### Minute 2: Clinical Lab Document Ingestion (`/reports`)
1. **Navigate to:** `Lab Reports` page.
2. **Action A (Digital PDF):** Drag and drop `samples/synthetic_lab_report.pdf`.
   - **What happens:** Extracted instantly via PyMuPDF native text layer in < 0.05 seconds.
   - **Observe:** 100% extraction accuracy across all 7 analytes (Glucose, HbA1c, Total Cholesterol, HDL, Triglycerides, Creatinine, Hemoglobin), units normalized, and biological reference intervals matched with settled inclusive limits.
3. **Action B (Corrupted / Rotated Scan):** Upload `samples/synthetic_rotated_scan.png`.
   - **What happens:** Adaptive Pre-OCR Quality Gate verifies blur and skew, automatically deskews (+3.5 deg correction), and runs local Tesseract OCR.
   - **Observe:** Yellow warning banner: *"scanned input: values may be wrong (synthetic set; real-world accuracy unknown)"*.
   - **Safety Feature:** Individual checkboxes are required per biomarker before the "Send to Chronic Risk Engine" button is enabled. Headless automatic risk transfer is strictly prohibited.

### Minute 3: Epidemiological Chronic Risk & Published Score Benchmarks (`/risk`)
1. **Navigate to:** `Chronic Risk` calculator.
2. **Observe 2 Distinct Sections:**
   - **Top Section (Epidemiological Prevalence):** Calibrated logistic regression models trained on CDC NHANES Cycles H & I (n=12,105) and tested on forward temporal Cycle J (n=5,856). Estimand: *probability of current self-reported doctor-diagnosed condition*. Notice that glycemic features (`fasting_glucose`, `hba1c`) are strictly excluded from the diabetes model to prevent circular diagnostic leakage.
   - **Bottom Section (Prospective Mortality Survival):** Cox Proportional Hazards models linked to the National Death Index (1999–2014, n=47,281; 8,828 deaths).
3. **PCE & IDRS Benchmarks:**
   - **Pooled Cohort Equations (ACC/AHA 2013):** Shows 10-year fatal CVD risk. Notice that when evaluated under guideline rules (complete cases or boundary clamping), there is NO statistically significant difference between ML Cox and published PCE (+0.0065 C-index gain, 95% CI [-0.0115, 0.0268]).
   - **Indian Diabetes Risk Score (IDRS, Mohan et al. 2005):** Strictly reproduces Table 1 scoring (Age, Waist: M <90/90-99/>=100 cm, Activity: 3 levels, Family History). Published IDRS achieves AUROC 0.7798 on US NHANES holdout.

### Minute 4: Biomedical Computer Vision Screening & Modality Gate (`/images`)
1. **Navigate to:** `Image Screening` page.
2. **Observe:** 4 supported clinical screening modalities:
   - Pediatric Chest X-Ray (Pneumonia)
   - Dermoscopy (Melanoma & Pigmented Lesions)
   - Fundus Photography (Diabetic Retinopathy)
   - Peripheral Blood Smear (Pathology)
3. **Demonstrate Front-Door Modality Gate:**
   - Select **Chest X-Ray** as the intended modality, but upload a dermoscopy image or random non-medical noise.
   - **What happens:** Front-door ResNet-18 Modality Gate detects the mismatch and **refuses to execute** the screening model, displaying a clear refusal message: *"Modality Gate rejected input image: detected modality contradicts selected modality"*.
   - **Observe:** Permanent warning banner: *"LOFO macro False Accept Rate is 19.03% (worst family: 100.0%). Screening models do not guarantee rejection of arbitrary photos."*

### Minute 5: Single-Cell Lung Disease Transcriptomics (`/analyze`)
1. **Navigate to:** `Single-Cell Atlas` classifier.
2. **Action:** Upload `samples/synthetic_lung_pseudobulk.csv`.
3. **What happens:** The server parses all 30,566 genes, verifies the strict input contract, calculates counts per million (CPM) log1p normalization, and applies the deployed **Balanced Dual-Rule Classifier**.
4. **Observe Output:**
   - Argmax prediction: Displays predicted disease category (Donor, COPD, Pulmonary Fibrosis, COVID-19) derived from uncalibrated balanced decision scores (preserving COVID recall at 21.1% vs 0.0% under naive calibration).
   - Probability distribution: Platt-calibrated cross-validated probabilities displayed alongside calibration caveats.

---

## 3. Critical Limitations to Mention When Presenting

When presenting PulmoAI to technical, regulatory, or clinical audiences, explicitly state these 7 measured boundaries:

1. **Not a Medical Device [Statutory Warning]:**
   - PulmoAI is an open-source technical benchmark and educational tool.
   - It is **NOT** approved by CDSCO (India), US FDA (21 CFR § 820), or European EMA (EU IVDR 2017/746). It cannot be used to guide patient care or clinical triage.

2. **PCE vs ML Cox Parity [Tested - `audit_pce_variants.json`]:**
   - The machine learning Cox model does **NOT** achieve a statistically significant discrimination gain over the published 2013 ACC/AHA Pooled Cohort Equations on identical primary-prevention inputs (paired delta C-index: +0.0065, 95% CI [-0.0115, 0.0268]). The apparent gain reported in earlier passes was an implementation artifact caused by assigning an ad-hoc 5.0% fallback to 303 high-risk participants with extreme biomarkers.

3. **Modality Gate Generalization Boundaries [Tested - `audit_modality_gate_v3.json`]:**
   - The Modality Gate achieves 99.50% Target True-Accept Rate, but its Leave-One-Family-Out (LOFO) macro False Accept Rate is **19.03%**, with 100% false acceptance on completely novel document layouts. It cannot guarantee rejection of arbitrary natural photos.

4. **Pediatric Restriction on Chest X-Rays [Tested]:**
   - Pneumonia models were trained exclusively on pediatric radiographs (children aged 1–5 from PneumoniaMNIST). Radiological characteristics differ fundamentally from adult radiographs; adult screening is unsupported.

5. **Diabetic Retinopathy Staging Disabled [Tested - `audit_ood_part_a.json`]:**
   - The 5-class ordinal staging model achieves only 52.8% top-1 accuracy on RetinaMNIST. Severity staging has been disabled to prevent clinical misclassification.

6. **Single-Cell Lung Atlas COVID Recall Limitation [Tested - `audit_lung_part_a.json`]:**
   - Whole-cell atlas models exhibit at best 21.05% recall (4/19 profiles) for COVID-19, and 0.0% recall on post-mortem autopsy cohorts (Melms et al., GSE171524), classifying nearly all severe cases as Pulmonary Fibrosis due to terminal fibrotic tissue remodeling.

7. **Synthetic OCR Accuracy Disclaimer [Tested - `audit_ocr_pass7.json`]:**
   - All document extraction benchmarks are evaluated on synthetic document templates with simulated optical degradations (*synthetic set; real-world clinical accuracy unknown*). 100% of parsed values require human verification before risk engine integration.
