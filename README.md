# PulmoAI — Multimodal Precision Health & Lung Disease Platform

PulmoAI is a comprehensive, multi-module biomedical intelligence platform connecting single-cell transcriptomics, clinical laboratory OCR, epidemiological chronic disease risk engines, and biomedical computer vision screening.

All evaluations are conducted under strict, honest, out-of-sample protocols (no random-split leakage or diagnostic self-prediction).

---

## Multimodal Architecture & Phased Modules

| Module & Engine | Modality / Data Source | Cohort Scale | Key Performance (Held-Out Test) | Safeguards & Protocols |
| :--- | :--- | :--- | :--- | :--- |
| **Module 1: Lung Atlas Classifier** *(Phase 0 & 1)* | Single-Cell RNA-seq (30,566 genes) | 17 studies, 407 donors (1.5M cells) | **63.4% Accuracy / 56.1% Bal-Acc** (Platt-Calibrated SVM) | Honest Leave-One-Study-Out (LOGO) cross-validation across hospital batches. |
| **Module 2: Chronic Risk Engine** *(Phase 2)* | CDC NHANES Vitals & Labs (11 features) | 17,961 US Adults (Cycles 2011–2018) | **85.3% AUROC / 86.6% Accuracy** on Type 2 Diabetes; >75–84% across CVD, HTN, Stroke, COPD | Evaluated on forward temporal Cycle J. HbA1c and glucose withheld to prevent diagnostic tautology. |
| **Module 3: Lab Report Reader** *(Phase 3)* | Clinical PDFs & Scanned Photos | Lipid, Metabolic, CBC Panels | **100% extraction & normalization** accuracy on benchmark lab reports | Native PyMuPDF text layer + local Tesseract OCR. SI-to-US conversion & abnormal flags. |
| **Module 4: Medical Image Screening** *(Phase 4)* | MedMNIST v2 (64×64 images) | 5 clinical modalities (X-ray, Skin, Retina, Blood, Chest) | **97.6% AUC (X-Ray), 93.7% AUC (Skin), 99.8% AUC (Blood), 77.3% AUC (Chest 14-Cond)** | Pretrained ResNet-18 fine-tuning + Out-of-Distribution (OOD) guard rejecting <0.50 max-softmax. |
| **Module 5: Unified Health Dashboard** *(Phase 5)* | Full Stack Application | FastAPI + React 19 + Tailwind v4 | Unified localhost production app served at `http://127.0.0.1:8000` | Real-time cross-module piping: extract lab PDF -> auto-fill risk engine -> predict risk. |

---

## Quick Start (Run Locally)

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Integrated Server
From the project root:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in any browser:
- **Dashboard Interface**: Full React 19 SPA served directly from `frontend/dist`.
- **Interactive OpenAPI Documentation**: `http://127.0.0.1:8000/docs`.

### 2. Run Test Suite
Verify all 30 unit, integration, and API tests:

```bash
PYTHONPATH=src .venv/bin/pytest tests -v
```

### 3. Frontend Development (Optional)
To edit and hot-reload the React dashboard:

```bash
cd frontend
npm run dev
```

---

## Detailed Module Documentation

### Module 1: Single-Cell Lung Disease Classifier
- Compares healthy donors against COPD, Pulmonary Fibrosis (PF), and COVID-19.
- Evaluated with nested Leave-One-Study-Out CV to measure true generalizability when predicting on an unseen hospital cohort.
- Probabilities are calibrated using 5-fold cross-validated Platt scaling (`CalibratedClassifierCV`).

### Module 2: NHANES Chronic Disease Risk Engine
- Calibrated using CDC NHANES epidemiological survey data (17,961 participants).
- Evaluated on forward temporal cycle holdout (Cycle J: 2017–2018).
- Predicts 10-year risk for:
  - **Type 2 Diabetes**: 85.3% AUROC / 86.6% Accuracy (HbA1c/glucose excluded to avoid self-fulfilling tautology).
  - **Hypertension**: 83.8% AUROC (calibrated on triplicate averaged BP and age).
  - **Hypercholesterolemia**: 84.1% AUROC (lipid ratios without statin bias).
  - **Cardiovascular Disease**: 79.0% AUROC (Framingham proxy).
  - **Stroke**: 77.3% AUROC (cerebrovascular proxy).
  - **COPD / Chronic Bronchitis**: 74.8% AUROC (smoking pack-years and respiratory vitals).

### Module 3: Clinical Lab Report Reader & OCR
- Handles digital PDFs (native text extraction) and mobile photo scans (local Tesseract OCR).
- Standardizes international SI units to US clinical standards (e.g., cholesterol `mmol/L` $\to$ `mg/dL`, creatinine `µmol/L` $\to$ `mg/dL`).
- Flags abnormal and critical laboratory biomarkers.
- One-click pipeline integration sends extracted metabolic values directly into the Chronic Risk Engine.

### Module 4: Biomedical Image Screening Engine
- Evaluates 5 standardized MedMNIST v2 modalities with fine-tuned ResNet-18:
  - `PneumoniaMNIST`: Chest X-Ray pediatric pneumonia screening (88.1% Acc, 97.6% AUC).
  - `DermaMNIST`: Dermatoscopic skin lesion triage across 7 classes (78.8% Acc, 93.7% AUC).
  - `RetinaMNIST`: Retinal fundus diabetic retinopathy grading (51.3% Acc, 74.4% AUC).
  - `BloodMNIST`: Microscopic blood cell morphology across 8 types (95.8% Acc, 99.8% AUC).
  - `ChestMNIST`: 14 thoracic pathologies multi-label screening (94.7% Acc, 77.3% AUC).
- Built-in Out-of-Distribution (OOD) guard rejects images when maximum prediction confidence falls below the threshold (default: 0.50), preventing spurious classifications on arbitrary non-medical photos.

---

## Critical Medical Limitations & Disclaimer

> [!CAUTION]
> **Research Prototype Only**:
> 1. PulmoAI is designed for educational triage, research modeling, and exploratory analysis. It is **not** an FDA-cleared diagnostic device.
> 2. The 64×64 vision models are trained strictly on standardized benchmark datasets; they will **not** work reliably on uncalibrated phone photos or replace full-resolution PACS radiology scans.
> 3. Single-cell lung classifications are subject to inter-hospital batch effects and are intended for research benchmarking, not clinical diagnostics.
> 4. All outputs should be verified by certified medical professionals.
