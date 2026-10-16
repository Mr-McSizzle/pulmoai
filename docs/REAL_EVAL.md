# Real Clinical Laboratory Reports Evaluation Protocol

## Overview
This document specifies the evaluation protocol, directory structure, data formats, and metrics for validating the PulmoAI lab report parser (`pulmoai.report_reader`) against clinical laboratory documents.

Clinical laboratory reports exhibit significant layout diversity across institutional electronic health record (EHR) systems, reference laboratory vendors, and hospital networks. This evaluation harness assesses:
1. Optical character recognition (OCR) and text layer parsing accuracy.
2. Biomarker entity identification across canonical clinical analytes.
3. Numerical value extraction fidelity within 1% relative tolerance.
4. Reference range boundary parsing and clinical abnormal flag alignment (`NORMAL`, `HIGH`, `LOW`).
5. Pre-extraction automated document quality gate decisions (`ACCEPT` vs `REFUSE`).

---

## 1. Ground Truth CSV Specification

The evaluation script `scripts/eval_real_reports.py` expects a comma-separated values (CSV) file containing ground-truth annotations for all clinical documents under evaluation.

### Required Header Columns
```csv
file,analyte,value,unit
```

### Column Definitions
- `file` (string, required): Filename or report identifier matching the document filename in the reports directory (e.g., `report_001.pdf`, `scan_042.png`, or `PAT104`). Extension may be omitted if matching by stem.
- `analyte` (string, required): Canonical analyte identifier corresponding to the PulmoAI clinical laboratory catalog. Supported canonical keys include:
  - `fasting_glucose` (FPG, fasting blood sugar)
  - `hba1c` (Glycated hemoglobin)
  - `total_cholesterol` (Total serum cholesterol)
  - `hdl_cholesterol` (HDL direct cholesterol)
  - `ldl_cholesterol` (LDL direct or calculated)
  - `triglycerides` (Serum triglycerides)
  - `serum_creatinine` (Serum creatinine)
  - `hemoglobin` (Total blood hemoglobin)
  - `systolic_bp` / `diastolic_bp` (Vitals if present on report)
- `value` (float, required): Ground truth numerical biomarker value (e.g., `104.5`, `5.8`, `0.92`).
- `unit` (string, required): Ground truth measurement unit as reported (e.g., `mg/dL`, `%`, `mmol/L`, `g/dL`).
- `expected_flag` (string, optional): Clinical abnormality flag (`NORMAL`, `HIGH`, `LOW`, or `UNKNOWN`). If omitted, flag evaluation is computed from documented canonical reference boundaries.

### Example Ground Truth CSV
```csv
file,analyte,value,unit
synth_report_01.png,fasting_glucose,98.0,mg/dL
synth_report_01.png,hba1c,5.4,%
synth_report_01.png,total_cholesterol,185.0,mg/dL
synth_report_01.png,hdl_cholesterol,52.0,mg/dL
synth_report_01.png,triglycerides,120.0,mg/dL
synth_report_02.png,fasting_glucose,145.0,mg/dL
synth_report_02.png,hba1c,7.8,%
synth_report_02.png,total_cholesterol,240.0,mg/dL
synth_report_03.png,serum_creatinine,1.8,mg/dL
synth_report_03.png,hemoglobin,11.2,g/dL
```

---

## 2. Evaluation Execution

To evaluate a directory of laboratory documents against a ground-truth CSV:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/eval_real_reports.py \
    --reports-dir /path/to/reports/ \
    --ground-truth /path/to/ground_truth.csv \
    --output reports/real_reports_eval.json
```

### Command-line Arguments
- `--reports-dir`: Directory containing `.pdf`, `.png`, `.jpg`, `.jpeg`, or `.tiff` files.
- `--ground-truth`: Path to ground-truth CSV file.
- `--output`: Path to output JSON evaluation summary.
- `--no-adaptive-ocr`: Optional flag to disable adaptive preprocessing (deskewing, resolution normalization, Otsu binarization).

---

## 3. Metrics and Statistical Estimation

The evaluation harness computes:
1. **Document-Level Quality Gate Decisions**:
   - `decision`: `ACCEPT` vs `REFUSE`
   - Quality metrics: Laplacian variance (blur), projection skew angle, estimated noise sigma.
2. **Entity Detection Rate & Value Accuracy**:
   - Overall extraction accuracy across all expected entities.
   - Per-analyte extraction accuracy.
   - Strict relative error tolerance: $|V_{\text{extracted}} - V_{\text{expected}}| / \max(V_{\text{expected}}, 10^{-6}) \le 0.01$ (1%).
3. **Wilson Score Confidence Intervals**:
   - Two-sided 95% Wilson score confidence intervals computed for overall extraction rate and per-analyte rates:
   $$\tilde{p} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
4. **List of Wrong Values**:
   - Detailed itemization of any accepted entities where extracted value disagreed with ground truth, reporting document ID, analyte, extracted value, expected value, and relative error.
5. **Flag Concordance**:
   - Concordance between extracted clinical flags and ground truth flags.

---

## 4. Ethical, Privacy, and Institutional Governance

- **Zero Data Ingestion without IRB Approval**: The harness is validated in-repo using synthetic documents. No patient identifiable health information (PHI) or real clinical reports are stored in git or uploaded.
- **HIPAA Compliance**: Real clinical document evaluation must only be executed on de-identified documents conforming to HIPAA Safe Harbor de-identification standards (45 CFR § 164.514(b)) or under an approved Institutional Review Board (IRB) research protocol.
- **Local Offline Execution**: The evaluation script executes entirely offline using local PyMuPDF and Tesseract OCR engines. No network requests are made and no data is transmitted to third-party APIs unless `GEMINI_API_KEY` is explicitly provided.
