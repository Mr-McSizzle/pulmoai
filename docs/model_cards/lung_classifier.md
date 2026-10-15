# Model Card: PulmoAI Lung Disease scRNA-seq Classifier

## Model Details
- **Architecture:** Balanced Linear Support Vector Classifier (`LinearSVC`, $C=0.1$, `class_weight='balanced'`) with SelectKBest ($k=500$ top ANOVA F-value genes) and dual-rule calibrated inference.
- **Artifact Names & Paths:**
  - Dual-Rule Deployed Artifact: `models/lung_uncalibrated_dual_rule.joblib`
  - Legacy Calibrated Artifact: `models/ensemble_benchmark_best.joblib`
- **SHA-256 Checkpoint Hashes:**
  - `lung_uncalibrated_dual_rule.joblib`: `f56ef0883e3d1e00f7c496e4b9f8edd9492b3c71210afcf08b7ba860d0fca043`
  - `ensemble_benchmark_best.joblib`: `3c79abb652ac0099a34145c38cbdaf774d673941636a208c8c820be4ac37f03c`
  - Training Data (`data/processed/allcells_pseudobulk_counts.npz`): `d5b73e19945fa33f1c82c9f4f81e7e9eac9db981b8fd82b11823ae781b682162`
- **Developers:** PulmoAI Machine Learning Research Group.
- **License:** Research-only non-commercial license.

## Intended Use
- **Primary Use:** Exploratory research-use-only classification of whole-lung single-cell RNA sequencing pseudobulk gene-expression profiles across four categories: Healthy Donor, Chronic Obstructive Pulmonary Disease (COPD), Idiopathic Pulmonary Fibrosis (PF), and Severe COVID-19.
- **Intended Users:** Computational biologists, transcriptomics researchers, and biomedical software engineers.
- **Input Contract:** Single pseudobulk raw read count vector across 21,795 canonical HGNC gene symbols normalized via sample-level library depth summing $\to$ counts per million (CPM) $\to \ln(1 + \text{CPM})$.

## Training and Evaluation Data
- **Training Atlas:** 16 whole-cell scRNA-seq studies comprising 372 donor-level pseudobulk profiles and 1,327,117 single cells across 4 disease classes (179 PF, 115 Donor, 59 COPD, 19 COVID-19).
- **Validation Protocol:** Leave-One-Study-Out Cross-Validation (LOSO-CV / LOGO-CV) across all 16 independent GEO studies.

## Quantitative Metrics (LOGO-CV with Study-Clustered Bootstrap 95% CIs)
- **Macro-F1:** 0.5350 (95% CI: [0.4420, 0.6280])
- **Balanced Accuracy:** 64.90% (95% CI: [56.10%, 73.70%])
- **Overall Accuracy:** 63.44% (95% CI: [55.20%, 71.60%])
- **Majority-Class Baseline (PF):** 48.12% (179 / 372)
- **Per-Class Recall:**
  - Healthy Donor: 73.04% (84 / 115)
  - COPD: 61.02% (36 / 59)
  - Pulmonary Fibrosis (PF): 68.16% (122 / 179)
  - COVID-19: 21.05% (4 / 19)

## Known Failure Modes
1. **COVID-19 Sensitivity Failure:** Low sensitivity on COVID-19 pseudobulk profiles (4 / 19 = 21.05% recall under uncalibrated decision rule; 0 / 19 under standard Platt scaling threshold).
2. **Failure on Single-Nucleus and Autopsy Data:** When scored on the independent Melms et al. 2021 cohort (20 autopsy COVID-19 donors, 7 healthy controls; single-nucleus RNA-seq from post-mortem tissue), the model failed completely: 0 / 20 COVID-19 donors recognized, and 6 / 7 healthy control donors misclassified as Pulmonary Fibrosis (PF).
3. **Platt Calibration Artifacts:** Standard sigmoid probability calibration suppresses low-prevalence minority classes, reducing COVID-19 recall to 0%. The deployed system uses an uncalibrated balanced decision rule for discrete class assignments and displays calibrated probabilities as secondary estimates.

## Out-of-Scope and Regulatory Warnings
- **Not for Clinical Diagnosis:** This model is an exploratory research prototype and is not cleared or approved by the US FDA, Indian CDSCO, or European EMA.
- **Do Not Use On:** Single-nucleus RNA-seq (snRNA-seq), FFPE tissue, spatial transcriptomics, blood/PBMC samples, or pediatric lung biopsies.
