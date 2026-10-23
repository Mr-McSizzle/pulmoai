# Data Inventory and Dataset Directory Structure

All project datasets are stored systematically within this `data/` directory.

---

## 1. Atlas & Single-Cell Transcriptomics (`data/atlas/`)

- `data/atlas/atlas-001.h5ad` (6,310,632,229 bytes):
  Full 16-study whole-cell single-cell RNA-seq source atlas containing 1,470,730 cells across Donor, COPD, Pulmonary Fibrosis, and COVID-19 cohorts.
- `data/atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv`:
  Curated sample metadata table containing study annotations, library IDs, and disease classifications.
- `data/atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad`:
  Pre-QC subset archive.

---

## 2. Processed Model Datasets (`data/processed/`)

- `data/processed/allcells_pseudobulk_counts.npz`:
  Aggregated raw pseudobulk expression matrix (372 profiles × 30,566 genes) used for model training and Leave-One-Study-Out evaluation.
- `data/processed/nhanes_merged.parquet`:
  Merged cross-sectional CDC NHANES examination dataset (n=47,281 adults) across 6 chronic condition targets.
- `data/processed/nhanes_mortality_linked.parquet`:
  Longitudinal prospective cohort (n=17,961 adults) with National Death Index (NDI) linked 5-year and 10-year follow-up for multivariable Cox PH and survival modeling.
- `data/processed/atlas_lineage_pseudobulk_5k.npz`:
  Lineage-resolved pseudobulk matrix across 5,000 highly variable genes.
- `data/processed/melms_profiles.npz`:
  External single-nucleus autopsy evaluation cohort (Melms et al. 2021).
- `data/processed/census_new_profiles.npz`:
  CZ CELLxGENE census external lung evaluation profiles.

---

## 3. Raw Cohorts & Study Archives (`data/raw/`)

- `data/raw/nhanes/`:
  10 complete CDC NHANES survey cycles (1999-2000 through 2017-2018) plus NCHS public-use Linked Mortality Files (LMF).
- `data/raw/gse136831/`:
  GEO GSE136831 raw count sparse matrix (`GSE136831_RawCounts_Sparse.mtx.gz`), cell barcodes, and sample metadata tables.
- `data/raw/gse248688/`:
  GEO GSE248688 raw scRNA-seq archives (`GSE248688_RAW.tar`).
- `data/raw/torchvision/`:
  CIFAR-10 test set archive used for empirical Out-Of-Distribution (OOD) benchmark evaluations.
- `data/raw/sample_images/`:
  Diagnostic imaging samples across modalities (Chest X-ray, Dermatoscopy, Retinal Fundus, Blood Smear).
- `data/raw/hgnc_complete_set.txt`:
  Complete HGNC approved human gene symbols and alias resolution mapping.

---

## 4. Sample Inputs for Testing (`data/sample_inputs/` & `samples/`)

- `scrna_sample_donor.csv`: Valid 30,566-gene single-cell pseudobulk profile (Healthy Donor).
- `scrna_sample_copd.csv`: Valid 30,566-gene single-cell pseudobulk profile (COPD).
- `scrna_sample_pf.csv`: Valid 30,566-gene single-cell pseudobulk profile (Pulmonary Fibrosis).
- `scrna_sample_covid19.csv`: Valid 30,566-gene single-cell pseudobulk profile (COVID-19).
- `synthetic_lab_report.pdf` & `synthetic_rotated_scan.png`: Digital and scanned lab reports for OCR testing.
