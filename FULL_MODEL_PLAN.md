# Full ImmuneAI Model Plan

> **Historical / superseded:** This is the earlier 19-disease plan for a 17.2 GB atlas and is not the active project. The current review project is PulmoAI on GSE136831; see `README.md`, `WALKTHROUGH.md`, and `FRONTEND_BACKEND_HANDOFF.md`. No files or prior work were deleted.

## Objective

Build a patient-level model that predicts the inflammatory condition from immune-cell expression data, with a complete, reproducible analysis and honest evaluation. The intended label set is the 19 diseases plus healthy controls described in the study; the exact class names and sample counts must be verified from the deposited metadata before fitting.

## Data source and current feasibility

The existing `GSE248688_RAW.tar` is useful for the eight-donor COVID-versus-healthy pilot, but it is not the full labeled atlas. The study authors deposited quality-controlled count matrices and patient metadata on Zenodo under record `14851902` (CC BY 4.0). The 149 kB metadata CSV is already in Downloads (two identical copies). It contains 1,047 sample rows spanning 19 diseases plus healthy controls. The complete expression matrices are not present locally; a 15 MB file named like the main matrix under `data/atlas/` is an incomplete transfer and must not be used.

- `INFLAMMATION_ATLAS_main_afterQC.h5ad` — 17,229,436,935 bytes; intended model-development split.
- `INFLAMMATION_ATLAS_validation_afterQC.h5ad` — 2,519,082,825 bytes; unseen-patient validation split.
- `INFLAMMATION_ATLAS_external_afterQC.h5ad` — 2,085,105,651 bytes; unseen-study evaluation split.
- `INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv` — patient and clinical metadata.
- `Supplementary_Table_1.xlsx` — published sample/split documentation.
- `scANVI_models.zip` — optional released cell-annotation models.

The three count matrices total 21,834,425,411 bytes (about 21.8 GB) before scratch space. The Mac currently reports about 30 GiB free. The downloader performs a live capacity check and preserves a safety buffer before fetching. The existing 3.6 GB GEO archive contains only 20 GSM records; most mixed-donor pools do not include per-cell donor assignments, so those cells cannot be safely labeled for patient-level training. Keep the original archive and pilot outputs unchanged.

Source: [Zenodo record 14851902](https://zenodo.org/records/14851902); [study article](https://www.nature.com/articles/s41591-025-04126-3); [authors' analysis code](https://github.com/Single-Cell-Genomics-Group-CNAG-CRG/Inflammation-PBMCs-Atlas).

## Analysis and model-selection design

1. Inspect the workbook and sample metadata first. Confirm disease labels, patient IDs, study IDs, split membership, duplicates, and per-class patient counts. Fail clearly if a label or patient mapping is ambiguous.
2. Use patient/sample as the prediction unit, never individual cells as independent train/test observations. Retain the authors' main, unseen-patient validation, and unseen-study external splits.
3. Compare a regularized linear baseline, linear and RBF support-vector classifiers, and XGBoost. Use cell-type-specific patient profiles or the released integrated embeddings if their contents and provenance support the analysis. Avoid treating millions of cells as millions of independent patients.
4. Tune only on the main split with patient-grouped, study-aware validation where class coverage permits. Select by macro-F1 and balanced accuracy, and report weighted F1 and Matthews correlation coefficient as complementary measures.
5. Freeze the selected pipeline, evaluate on the authors' validation split, then use the external unseen-study split once for the final generalization report. Report per-disease metrics, confusion matrices, class counts, calibration/uncertainty, and cohort-wise performance.
6. Add full analysis artifacts: label and cohort audit, QC and missingness summaries, class imbalance, exploratory plots, model comparison, feature importance/SHAP when supported, prediction tables, and a limitations section. No result is recorded until it is actually computed.

There is no universally best algorithm. XGBoost is a strong candidate and was used for cell-level disease/feature analysis in the paper; the paper's patient-level approach also compares SVM-family and nearest-neighbor classifiers on cell-type-specific patient embeddings. ImmuneAI will choose based on patient-grouped validation and preserve the external study split for an honest final test. The paper reports that unseen-study performance can be substantially worse than within-study cross-validation, so that external result is especially important.

## Next checkpoint

The full multi-disease model still requires the complete main H5AD. The user should download it directly if proceeding with that scope: [INFLAMMATION_ATLAS_main_afterQC.h5ad (17.2 GB)](https://zenodo.org/records/14851902/files/INFLAMMATION_ATLAS_main_afterQC.h5ad?download=1). After it is complete, verify the published MD5 checksum, inspect the AnnData schema and label counts, and then train with patient-level, study-aware validation. The current partial same-named file must not be treated as a usable download. Use the downloader's live space check before acquiring additional validation or external matrices.
