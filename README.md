# PulmoAI: Multimodal Precision Health & Biomedical Intelligence Platform

PulmoAI is a multimodal biomedical intelligence platform and reproducible technical benchmark spanning single-cell transcriptomics, clinical laboratory document parsing, epidemiological chronic disease risk engines, prospective mortality survival modeling, and biomedical computer vision screening.

Every model, metric, confidence interval, and sample size in this repository is derived under strict, out-of-sample evaluation protocols (Leave-One-Study-Out hospital cross-validation, temporal cycle holdouts, Leave-One-Family-Out negative testing, and double-blind synthetic OCR holdouts).

---

## 1. What This System Is and Is Not

### What It Is
- An open-source research and educational platform exploring multimodal biomedical machine learning architectures.
- A rigorously audited technical benchmark reconciling discrepancies between published clinical algorithms and machine learning models.
- A fully integrated, single-port full-stack web application (`FastAPI` backend + `React 19` SPA frontend) running 100% locally on consumer hardware (<= 8 GB RAM).

### What It Is NOT
- **NOT an FDA-cleared or CDSCO-registered medical device:** PulmoAI has NOT undergone clinical trials or regulatory approval under US FDA (21 CFR § 820), Indian CDSCO (Medical Device Rules 2017), or European EMA/EU IVDR (2017/746).
- **NOT for primary clinical diagnosis:** Models cannot be used to diagnose patients, prescribe medications, or replace certified radiologists, pathologists, or physicians.
- **NOT a general-purpose vision tool:** The 64x64 vision models are trained strictly on standardized benchmark datasets (MedMNIST v2); they do not process high-resolution DICOM PACS scans or arbitrary smartphone photos reliably.
- **NOT an acute COVID-19 diagnostic:** Single-cell lung models reflect end-stage fibrotic tissue organization (classifying acute autopsy and severe cases as Pulmonary Fibrosis).

---

## 2. Platform Architecture & Audited Results Table

All metrics below are dynamically verified from versioned JSON audit reports in `reports/audit_*.json`.

| Module / Engine | Modality / Data Source | Evaluated Cohort | Audited Performance | Baseline / Safeguards |
| :--- | :--- | :--- | :--- | :--- |
| **Module 1: Lung Atlas Classifier** | Single-Cell RNA-seq (30,566 genes, pseudobulk counts) | 16 GEO studies, 372 donor profiles (1.47M cells) | **64.78% Acc / 53.49% Bal-Acc / 0.5218 Macro-F1** (Dual-Rule); COVID recall: **21.05% (4/19)** | Leave-One-Study-Out CV across 16 hospital batches. Platt calibration provided as estimate. Majority class baseline: PF (48.12%). Melms post-mortem autopsy holdout (GSE171524): 25/26 predicted PF. |
| **Module 2: Chronic Disease Risk Engine** | CDC NHANES Vitals & Labs (43 features) | 17,961 US adults (Train: Cycles H & I, n=12,105; Test: Cycle J, n=5,856) | **Diabetes: 0.8525 AUROC / 86.6% Acc**; HTN: 0.8384; High Chol: 0.8413; CVD: 0.7904; Stroke: 0.7733; COPD: 0.7479 | Evaluated on forward temporal Cycle J (2017-2018). Estimand: current doctor-diagnosed prevalence, NOT future incidence. Glycemic markers excluded to prevent circularity. OOF-derived Low/Mod/High risk tiers (RR up to 9.25x). |
| **Module 3: Clinical Lab Report Reader** | Clinical PDF & Scanned Image Reports | Honest holdout: 7 hospital/clinic layouts (Templates A-G, n=490/cond) | **Clean PDF: 94.90% accurate, 0.00% silent errors**; Rotated scans: **84.69%** adaptive (+25.51% over raw 59.18%); **Blur: 100% refused by Quality Gate (0.00% silent error)** | Native PyMuPDF text layer prioritized. Pre-OCR blur/skew quality gate. Settled inclusive range limits [ref_min, ref_max]. 100% of extracted values blocked from automatic risk engine transfer without human confirmation. |
| **Module 4: Medical Image Screening & Gate** | MedMNIST v2 (64x64) + Negative Families | 5 medical modalities + 6 negative non-target families | **Modality Gate v3: 99.50% Target TAR** (X-ray 99.75%, Derma 99.00%, Fundus 99.25%, Smear 100.00%); **LOFO Macro FAR: 19.03%** | ResNet-18 modality classifier gate rejects cross-modality mismatches and non-medical photos. Pixel std pre-check (std < 0.040) eliminates blank/noise (FAR: 0.00%). Worst LOFO family: document layouts (100.0% FAR). Warning banner permanently active. Pneumonia restricted to pediatric ages 1-5; retina staging disabled. |
| **Module 5: Future Mortality Risk Engine** | CDC NHANES Linked Mortality Files (1999-2014) | 47,281 eligible adults (8,828 deaths). Temporal holdout: 2011-2014 (n=11,411) | **Primary Prevention (n=5,899, 110 fatal CVD events): Full Cox C-index 0.8297 [0.7885, 0.8663], 5-yr AUROC 0.8234** vs PCE-input Cox 0.7864 vs Published PCE 0.7391 | 5-Year CVD Decile Calibration: O/E ratio 1.032 [0.910, 1.170], Brier 0.0184, intercept 0.0356. Discrimination gain over PCE inputs is statistically significant (delta C: +0.0432 [0.0244, 0.0658]). Asian subgroup C-index 0.8618 [0.8032, 0.9140]. Mohan 2005 IDRS Table 1 verified (AUROC 0.7798 vs 4-feature ML 0.8303). |

---

## 3. Quick Start & Local Execution

PulmoAI is designed to run entirely locally on a single port with zero external network connectivity required at runtime.

### 1. Prerequisites & Virtual Environment
Requires Python 3.10+ and Node.js 18+ (Node.js is optional if using the pre-compiled `frontend/dist` bundle).

```bash
# Clone the repository
git clone https://github.com/your-org/pulmoai.git
cd pulmoai

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Launch the Integrated Full-Stack Server
Run FastAPI from the root directory. FastAPI serves both the REST API endpoints and the pre-built React 19 single-page application on port 8000:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open your browser to:
- **Application Dashboard:** `http://127.0.0.1:8000`
- **Interactive OpenAPI Documentation:** `http://127.0.0.1:8000/docs`
- **ReDoc Technical Specification:** `http://127.0.0.1:8000/redoc`

### 3. Rebuilding Frontend Assets (Optional)
The static distribution (`frontend/dist`) is pre-compiled and tracked for immediate deployment. To rebuild the frontend after modifications:

```bash
cd frontend
npm install
npm run build
cd ..
```

---

## 4. Automated Test Suite

Run all 54 unit, integration, estimand, and offline privacy tests:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

All 54 tests execute in under 12 seconds with 100% pass rate.

---

## 5. Offline Privacy & Security Guarantee

PulmoAI operates with a strict zero-telemetry, zero-cloud execution policy:
- **Zero Outbound Calls at Inference Time:** Confirmed by code audit and verified by `tests/test_privacy_offline.py` across all 5 clinical modules.
- **Local Tesseract OCR:** Document text extraction and image processing execute entirely via local CPU/MPS libraries (`pymupdf`, `pytesseract`, `PIL`).
- **No Remote Telemetry or Tracking:** No user data, uploaded images, single-cell counts, or lab values are transmitted over external networks.

---

## 6. Reproducing the Technical Audit Scripts

Every experimental finding, calibration curve, and comparison in `docs/final_report_v2.md` can be reproduced locally using the following audit scripts:

```bash
# Single-Cell Transcriptomic Atlas (LOGO-CV, tissue provenance, Melms scoring)
python3 scripts/audit_lung.py
python3 scripts/audit_lung_part_a.py

# NHANES Chronic Disease Risk (Cycle J holdout, Asian diabetes coincidence, OOF tiers)
python3 scripts/audit_nhanes.py
python3 scripts/audit_nhanes_part_a.py

# Future Mortality Risk & Baseline Comparisons (PCE, IDRS, CVD deciles, Asian subgroup)
python3 scripts/audit_future_risk_v3.py

# Modality Gate Classifier (Leave-One-Family-Out cross-validation, pixel std pre-check)
python3 scripts/audit_modality_gate_v3.py

# Clinical Lab OCR Honest Holdout (Templates A-D, silent errors, degraded scans)
python3 scripts/audit_ocr_holdout_v2.py

# Dynamic Report & Architecture Diagram Regeneration
python3 scripts/build_report.py
MPLCONFIGDIR="$PWD/.mplconfig" python3 scripts/generate_architecture_diagram.py
```

---

## 7. Dataset Provenance & Licenses

| Dataset / Resource | Source / Accession | Scale / Records | Primary License / Usage Terms |
| :--- | :--- | :--- | :--- |
| **Human Lung scRNA-seq Atlas** | 16 GEO Series (e.g. GSE158127, GSE171668, GSE135893, GSE136831) | 372 donor profiles (1,470,730 single cells) | Open Access / Public Domain (NCBI GEO Data Sharing Policy) |
| **Melms COVID-19 Autopsy Cohort** | NCBI GEO Series GSE171524 (Columbia University) | 20 COVID-19 autopsy, 7 control donors (snRNA-seq) | Open Access (NCBI GEO) |
| **CDC NHANES Survey Cycles** | CDC National Center for Health Statistics (Cycles 1999-2018) | 17,961 evaluated adults (Cycles H, I, J) | Public Domain (US Federal Government Work) |
| **NCHS Linked Mortality Files** | CDC NCHS National Death Index follow-up through 2015 | 47,281 eligible participants (8,828 deaths) | Public Domain (CDC Restricted-Use Data Perturbation Warning Applies) |
| **MedMNIST v2 Benchmarks** | Zenodo / Yang et al. (Pneumonia, Derma, Retina, Blood, Chest) | 64x64 standardized biomedical images | Creative Commons Attribution 4.0 International (CC-BY 4.0) |
| **Negative Modality Families** | CIFAR-100, BreastMNIST, OrganCMNIST, OrganSMNIST, Synthetic Blanks | 2,136 negative evaluation images | CC-BY 4.0 / Open Access / Internal Benchmark |
| **Synthetic Lab Report Benchmark** | Templates A-G styled after clinical pathology & urgent care layouts | 3,430 evaluated analyte entities across seeds 777 & 999 | Apache 2.0 (Synthetic Research Benchmark) |
| **HGNC Gene Nomenclature** | HUGO Gene Nomenclature Committee (EMBL-EBI / Wellcome Trust) | 43,000+ approved human gene symbols & aliases | Public Domain / Open Access (Free Academic & Commercial Use) |

---

## 8. Detailed Model Cards

Detailed model cards documenting training procedures, intended use, known limitations, and bias evaluations are available in `docs/model_cards/`:
- [Single-Cell Lung Disease Classifier](docs/model_cards/lung_classifier.md)
- [NHANES Chronic Disease Risk Engine](docs/model_cards/nhanes_chronic_risk.md)
- [Future Mortality Risk Engine](docs/model_cards/future_mortality_risk.md)
- [Modality Classifier & Gate](docs/model_cards/modality_classifier_gate.md)
- [Clinical Laboratory Report Reader](docs/model_cards/ocr_report_reader.md)

---

## 9. Statutory & Regulatory Disclaimers

### Research Use Only (RUO)
PulmoAI is developed strictly for research, educational, and benchmarking purposes. It is **NOT** intended for use in the diagnosis, cure, mitigation, treatment, or prevention of disease in humans or animals.

### Jurisdiction-Specific Statements
- **India (CDSCO):** Under the Medical Device Rules, 2017 (amended 2020), Software as a Medical Device (SaMD) intended for clinical diagnosis requires registration and licensing from the Central Drugs Standard Control Organisation (CDSCO). PulmoAI is **NOT registered** with CDSCO and cannot be used in Indian clinical practice.
- **United States (US FDA):** Under 21 CFR § 820 and Section 510(k) / De Novo pathways, clinical AI systems require premarket clearance or approval. PulmoAI is **NOT an FDA-cleared** medical device.
- **European Union (EMA / CE-IVD):** Under EU In Vitro Diagnostic Medical Devices Regulation (IVDR 2017/746) and the EU AI Act (2024), diagnostic AI systems require conformity assessment by an EU Notified Body. PulmoAI does **NOT bear a CE mark**.

---

## 10. License & Distribution

- **Software License:** PulmoAI codebase is released under the [MIT License](LICENSE).
- **Model Checkpoints:** Weight distribution, reproduction scripts, and cryptographic SHA-256 integrity checksums are documented in [docs/WEIGHTS.md](docs/WEIGHTS.md).

