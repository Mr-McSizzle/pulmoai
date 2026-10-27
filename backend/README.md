# PulmoAI Multimodal Health Platform API

The FastAPI service powers the unified PulmoAI multi-module platform across single-cell transcriptomics, laboratory OCR, epidemiological chronic risk prediction, and biomedical vision screening.

## Start the Platform Locally

From the project root:

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

1. **Unified Dashboard**: Visit `http://127.0.0.1:8000/` to use the interactive React 19 application (served directly from `frontend/dist`).
2. **Interactive API Docs**: Visit `http://127.0.0.1:8000/docs` for the OpenAPI / Swagger UI.

## Local Accounts and Saved Records

- `POST /api/auth/register`, `POST /api/auth/login`, `POST /api/auth/logout`, and `GET /api/auth/me` manage local accounts and an HttpOnly session cookie.
- Authenticated successful lung CSV, lab report, medical image, chronic-risk, and future-risk analyses are saved with their original file (when present), inputs/results, and a UTC upload timestamp.
- `GET /api/uploads` lists only the current account's records. `GET /api/uploads/{id}` returns the saved result, `GET /api/uploads/{id}/download` downloads its original file, and `DELETE /api/uploads/{id}` removes that record and file.
- By default the SQLite database is `data/user_storage/pulmoai.sqlite3`; original files are under `data/user_storage/uploads/`. Set `PULMOAI_USER_DATA_DIR` before starting the server to choose another local directory.
- Passwords are stored as salted PBKDF2 hashes; random session tokens are stored hashed. The database and files are not encrypted at rest. This localhost research prototype should use synthetic or de-identified inputs, not identifiable patient records.

---

## Multimodal Endpoints

### 1. Single-Cell Transcriptomics (Phases 0 & 1)
- `GET /api/health` — System status, loaded artifact IDs, and engine readiness.
- `GET /api/dashboard` — Multi-study lung atlas metadata (17 studies, 407 donors) and honest leave-one-study-out benchmark metrics.
- `GET /api/insights` — Model-derived feature coefficients (ANOVA top 500 features).
- `POST /api/validate` — Validates gene-expression CSV structure (30,566 genes).
- `POST /api/predict` — Predicts lung condition (Donor, COPD, Pulmonary Fibrosis, COVID-19) with 5-fold cross-validated Platt scaling probabilities.

### 2. Clinical Lab Report Reader & OCR (Phase 3)
- `POST /api/report/extract` — Upload clinical PDFs (digital text layer) or mobile scans (Tesseract OCR). Normalizes SI to standard US clinical units (mmol/L to mg/dL, umol/L to mg/dL) and auto-flags abnormal reference ranges. Returns mapped features ready for the Risk Engine.

### 3. NHANES Chronic Disease Risk Engine (Phase 2)
- `POST /api/risk/predict` — Calculates 10-year calibrated risk for 6 chronic conditions (Type 2 Diabetes: 85.3% AUROC, Hypertension: 83.8% AUROC, Hypercholesterolemia: 84.1% AUROC, CVD: 79.0% AUROC, Stroke: 77.3% AUROC, COPD: 74.8% AUROC). Evaluated on 17,961 CDC cohort participants with diagnostic tautology strictly prevented.

### 4. Biomedical Image Screening (Phase 4)
- `POST /api/image/predict` — ResNet-18 vision screening on 64×64 biomedical images across 5 clinical modalities (Chest X-ray Pneumonia, Dermatoscopy Skin Lesions, Retinal Fundus Diabetic Retinopathy, Blood Smear Microscopy, 14 Thoracic Conditions). Includes Out-of-Distribution (OOD) confidence gating (<0.50 threshold) to reject spurious inputs.

---

## Clinical Limitation Notice
PulmoAI is a research screening and educational triage prototype. It is **not** an FDA-cleared diagnostic device and must not replace professional clinical consultations, laboratory verifications, or radiological examinations.
