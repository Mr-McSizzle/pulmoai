# PulmoAI Project Walkthrough

## Current handoff state — local app running, 13 October 2026

- Workspace: `/Users/nisarg/Desktop/ml project/`. Existing scripts, models, and evaluation reports remain preserved. The local FastAPI backend and spatial React frontend are implemented in `backend/` and `frontend/` and connect to the saved model and verified reports.
- Full source dataset: `/Users/nisarg/Downloads/atlas-001.h5ad`, 6,310,632,229 bytes. It remains in Downloads and is not copied or modified.
- A 15,908,864-byte file at `data/atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad` has a similar name but is incomplete and was not used. The active processed data cache is `data/processed/allcells_pseudobulk_counts.npz` (372 × 30,566).
- Existing model artifacts include `models/ensemble_benchmark_best.joblib` and earlier preserved models. Current selected artifact is Linear SVM. The benchmark and reports are present under `reports/ensemble_benchmark/`.
- Environment: Python 3.12.4, Node 24.11.1, and npm 11.6.2. FastAPI/Uvicorn/multipart and React/Vite packages are now installed; the npm lockfile is present. Approximately 12 GiB was free at the earlier disk inspection.
- Added `data/README.md` with source/cache details and expanded `.gitignore` to exclude derived counts, virtual environments, and generated model binaries. No data or model was deleted.
- The expanded 9-candidate study-held-out result (63.4% accuracy, 52.2% macro-F1) was rerun with the same seed and reproduced exactly. Saved predictions, fold assignments, confusion matrix, and all candidate scores match. The separate 4-candidate procedure reports 65.1% accuracy / 56.0% macro-F1.
- **Current app status:** CP6–CP9 implement the backend, spatial frontend, build, and live launch. All 14 Python unit tests pass; TypeScript and Vite production build pass. Homepage, model health, dashboard, insights, CSV validation, and prediction were checked against the running server at `http://127.0.0.1:8000`.
- **Next scientific action:** resolve biological patient identity where source-study metadata permits, then assess independent external validation. The current app remains localhost research software, not a clinical tool.

## Implementation CP1 — Atlas metadata and model-contract validation

**Status:** Complete — metadata, grouping, and feature contract verified; patient identity unresolved  
**Date:** 13 October 2026

- Added `scripts/validate_atlas_metadata.py`, which reads the H5AD metadata in chunks and checks the source metadata against the processed profile cache and saved model without loading the full expression matrix into memory.
- Validation passed for 1,470,730 cells, 30,566 features, 16 studies, and 372 `gse_alias + orig.ident` groups. The four required label/group fields had no missing values; all groups had consistent labels; cell IDs were nonblank and unique within study; and source, cache, and model feature order matched exactly.
- Verified group counts are Donor 128, COPD 46, PF 179, and COVID19 19. These are sample/library groups, **not verified patient counts**. The source `.obs` has no explicit patient/subject ID field, so the patient count remains unknown.
- The final artifact is a Linear SVM bundle with 30,566 ordered raw-count features. Training normalizes each aggregated profile to counts per million and applies `log1p`; its fitted estimator then selects 500 features, scales them, and predicts. The SVM is not probability-calibrated.
- **Validation:** `scripts/validate_atlas_metadata.py` completed with `passed_with_identity_limitation`; Python compilation passed. Machine-readable evidence is in `reports/atlas_metadata_validation.json`.
- **Next action:** completed in CP2 below.

## Implementation CP2 — Strict single-profile preprocessing

**Status:** Complete — exact input contract implemented and tested  
**Date:** 13 October 2026

- Added `src/pulmoai/preprocessing.py` for one raw-count profile in either a one-row wide CSV or a long `gene_id,count` / `gene,count` CSV. Wide files may include one `sample_id` field. Any other orientation, extra/missing feature, duplicate identifier, multiple profile, non-finite/negative/fractional count, or all-zero input is rejected with a specific validation error.
- The parser aligns genes by the model's exact feature identifiers rather than trusting CSV order; it never fills missing genes with zero. It applies the verified training transform, `log1p(raw_counts / total_counts × 1,000,000)`, in the model's complete 30,566-feature order. Top-500 feature selection and scaling remain in the fitted model pipeline.
- Added nine unit tests in `tests/test_preprocessing.py`. All passed. A cached real profile matched the trainer's transformed vector exactly and produced a valid encoded class from the saved model. The smoke test only verifies the inference contract; it is not an independent accuracy result.
- The model emits encoded class IDs; `label_order` in its saved bundle maps them to Donor, COPD, PF, and COVID19. The future API must do this mapping and must not convert uncalibrated SVM outputs into probabilities.
- **Files:** `src/pulmoai/__init__.py`, `src/pulmoai/preprocessing.py`, `tests/test_preprocessing.py`, `README.md`, `data/README.md`, and this walkthrough.
- **Limitations:** CSV parsing/preprocessing exists, but no public prediction endpoint or frontend exists yet. It supports neither cell-level multi-sample matrices nor images or ordinary blood reports.
- **Next action:** completed in CP3–CP5 below; continue to CP6 only after the required backend packages are installed.

## Implementation CP3 — Baseline reproduction and metric audit

**Status:** Complete — expanded benchmark reproduced and score discrepancy reconciled  
**Date:** 13 October 2026

- Added `scripts/verify_benchmark_outputs.py`. It recomputes accuracy, balanced accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, and the confusion matrix from the saved out-of-fold table; it checks all nine candidate-model metric rows against the machine-readable summary.
- The verifier passed on 372 held-out sample/library profiles across 16 studies. Recomputed selected-procedure results: accuracy **0.6344**, balanced accuracy **0.5613**, macro-F1 **0.5219**, weighted-F1 **0.6666**. COVID19 recall is **4/19 (0.2105)**.
- Independently reran the full nine-candidate evaluation from `data/processed/allcells_pseudobulk_counts.npz` into `/private/tmp/`, preserving project reports and the final model. It took 487.24 seconds. All 372 selected predictions, fold selections, confusion matrix, and candidate metrics matched the saved run exactly; timing columns vary by run.
- The separate `reports/allcells_atlas/` result (accuracy 0.6505, macro-F1 0.5600) comes from a four-candidate model-selection menu. The expanded procedure adds Random Forest, AdaBoost, Histogram Gradient Boosting, MLP, and hard voting; changing the selectable candidate set changes nested predictions. Neither result is external validation.
- **Validation:** independent verifier passed; fresh benchmark reproduced identical out-of-fold predictions; all candidate metrics and the confusion matrix matched; nine preprocessing tests and Python compilation passed.
- **Evidence:** `reports/ensemble_benchmark/independent_verification.json`.

## Implementation CP4 — Candidate algorithm comparison

**Status:** Complete — eight classifiers plus hard voting independently checked  
**Date:** 13 October 2026

- Verified Logistic Regression, Linear SVM, RBF SVM, Random Forest, Extra Trees, AdaBoost, Histogram Gradient Boosting, MLP, and hard voting of Logistic Regression + Linear SVM + Extra Trees. XGBoost was unavailable and not run; no package was downloaded. Linear regression and reinforcement learning do not fit this labeled multiclass task.
- Independently verified each candidate's accuracy, balanced accuracy, macro-F1, and weighted-F1 from its held-out-study prediction column. Logistic Regression leads fixed-candidate macro-F1 (54.1%); Linear SVM leads balanced accuracy (57.9%) and accuracy (64.8%). Hard voting has 49.8% macro-F1. No one candidate dominates every metric.
- Model selection and feature selection remain fold-local. Inner-CV scores around 90% macro-F1 are development-selection values, not unseen-study performance.
- **Evidence:** `reports/ensemble_benchmark/analysis_summary.json`, `algorithm_comparison.csv`, and `independent_verification.json`.

## Implementation CP5 — Final model and evaluation review

**Status:** Complete — saved model contract and scientific limits verified  
**Date:** 13 October 2026

- Confirmed `models/ensemble_benchmark_best.joblib` is the selected Linear SVM fitted on all 372 sample/library profiles. It expects the exact 30,566 raw-count features; preprocessing parity and a model smoke test passed.
- The fitted pipeline emits encoded class IDs 0–3, mapped through `label_order`. It has no `predict_proba`; raw SVM decision values must not be shown as probabilities or clinical confidence.
- The 63.4% / 52.2% result estimates the nested model-selection procedure on held-out studies, not the all-data final artifact as an independent test. No external cohort is available here.
- COVID19 has 19 profiles from two studies; the nested procedure identifies 4/19. Disease labels are study-imbalanced, and biological patient IDs are unavailable. The model is exploratory, not diagnostic or clinical.
- At the time of this checkpoint the API and frontend had not yet been implemented; see CP6–CP8 below for the completed app source.

## Latest checkpoint — Multi-algorithm benchmark, 13 October 2026

The existing pseudobulk cache was reused; the 5.9 GB H5AD was not reopened. Eight classifiers and a hard-voting ensemble were evaluated with identical outer folds, each holding out one of the 16 source studies. Fold-local top-500 feature selection and the same CPM/log1p preprocessing were retained. The final development model was refit on all 372 sample/library groups.

### Results

The nested model-selection procedure produced unseen-study accuracy **63.4%**, balanced accuracy **56.1%**, macro-F1 **52.2%**, and weighted-F1 **66.7%**. Per-class recall was Donor **66%**, COPD **74%**, PF **63%**, and COVID19 **21%**.

| Candidate | Held-out-study accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 64.5% | 56.5% | **54.1%** | 67.6% |
| Linear SVM | **64.8%** | **57.9%** | 53.5% | **67.9%** |
| Hard Voting Ensemble | 63.7% | 52.5% | 49.8% | 66.5% |
| Neural Network (MLP) | 59.1% | 48.3% | 46.1% | 62.4% |
| RBF SVM | 59.9% | 46.1% | 43.5% | 62.0% |
| Histogram Gradient Boosting | 55.4% | 44.3% | 42.5% | 59.1% |
| AdaBoost | 57.0% | 45.7% | 42.2% | 59.1% |
| Random Forest | 56.7% | 44.2% | 42.0% | 59.9% |
| Extra Trees | 57.3% | 42.4% | 40.9% | 59.5% |

Linear SVM is the selected final algorithm because its mean inner-CV macro-F1 was highest (90.3% versus Logistic Regression 90.0%). That is a model-selection score, not generalization accuracy. In the fixed-candidate held-out comparison Logistic Regression has the highest macro-F1 and Linear SVM the highest balanced accuracy; therefore there is no decisive winner across metrics. The ensemble did not improve this dataset's result. XGBoost was not run because its package is not installed; nothing was downloaded.

The benchmark completed in **465 seconds (7 minutes 45 seconds)** on this machine. Most runtime came from Histogram Gradient Boosting (318 seconds) and AdaBoost (84 seconds). The saved model is `models/ensemble_benchmark_best.joblib`; readable results are in `reports/ensemble_benchmark/RESULTS.md`, and detailed metrics, runtimes, fold choices, and limitations are in `reports/ensemble_benchmark/analysis_summary.json`.

Run only the cached benchmark with:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/benchmark_models.py
```

The aggregate score is modest and COVID19 recall is low. More algorithms did not produce higher accuracy, and no metric should be presented as clinical performance.

## Data preparation checkpoint — All-cells lung atlas, 13 October 2026

The downloaded `/Users/nisarg/Downloads/atlas-001.h5ad` was inspected and used for a complete, reproducible multi-study training run. The original 5.9 GB file stayed in Downloads and was opened in backed, read-only mode. The trainer streamed all 1,470,730 cells in 2,048-row chunks; it did not load the full cell-by-gene matrix into RAM.

### Dataset and labels

- Matrix: **1,470,730 cells × 30,566 features**, from **16 source studies**. The official [Zenodo record](https://zenodo.org/records/21318856) describes this all-cells lung atlas.
- Local `.obs` columns used: `Disease`, `gse_alias`, `orig.ident`, and `celltypeuse`; there is no verified biological patient/donor identifier.
- Source-grouping key: `gse_alias + orig.ident`. This creates **372 study-scoped sample/library groups**, not 372 verified patients.
- Group counts: 128 Donor, 46 COPD, 179 PF, and 19 COVID19. Cell counts: 501,843 Donor, 254,763 COPD, 635,173 PF, and 78,951 COVID19.
- Important metadata caveat: the Zenodo description's sample count and the local number of `orig.ident` groups use different units. Do not claim patient-level counts until the source-study sample-to-patient mappings are checked.
- The matrix is backed CSR, float32. The full-matrix audit found no non-finite or negative stored values. A small number of values in ERP114453 are slightly off integers at strict absolute tolerance; the trainer confirmed all values are integer-like within float32 tolerance (`rtol=1e-6`, `atol=1e-5`) before rounding during aggregation.

### Training and validation

1. Sum cell counts into one pseudobulk vector per study-scoped `orig.ident` group.
2. Normalize each pseudobulk to counts per million, then apply `log1p`.
3. Use fold-local ANOVA feature selection (top 500) and compare Logistic Regression, Linear SVM, RBF SVM, and Extra Trees.
4. Primary evaluation: 16 outer folds, each withholding one entire source study. Within each outer training set, choose the algorithm using stratified three-fold sample-group CV. This tests cross-study transfer and avoids putting cells or groups from the held-out study in training.
5. Select the final algorithm by mean inner-CV macro-F1 across outer training folds. Linear SVM scored 0.903 and was selected; this is a model-selection score, not generalization performance. The saved model is for demonstration/inference; report only the held-out-study metrics below.

### Measured results

| Metric | Held-out-study result |
|---|---:|
| Accuracy | **0.651** (242/372 groups) |
| Balanced accuracy | **0.601** |
| Macro-F1 | **0.560** |
| Weighted-F1 | **0.683** |

Class recalls: Donor 0.664, COPD 0.783, PF 0.642, and COVID19 0.316. Inner sample-group CV gave Linear SVM mean macro-F1 0.903, while the nested unseen-study procedure scored 0.560 macro-F1. This gap is direct evidence of study/domain shift. The result is moderate and uneven—not a high-accuracy or clinical model. Show the class/study imbalance and low COVID19 recall. Candidate algorithms and fold-specific selection are in the reports.

### Outputs and reproduction

- `scripts/inspect_atlas_h5ad.py` — backed/read-only schema and metadata inspection.
- `scripts/audit_h5ad_expression_scale.py` — full-matrix streamed check of count-like values; results in `reports/atlas_expression_scale.json`.
- `scripts/train_allcells_atlas.py` — chunked aggregation, nested study-held-out evaluation, plots, feature report, and final fit.
- `data/processed/allcells_pseudobulk_counts.npz` — derived 372 × 30,566 pseudobulk count profiles plus labels and feature names.
- `models/allcells_lung_disease_classifier.joblib` — final Linear SVM model bundle and input metadata.
- `reports/allcells_atlas/analysis_summary.json` — authoritative metrics, cohort summary, environment, and limitations.
- Additional files: `study_heldout_predictions.csv`, `study_heldout_folds.csv`, `inner_model_selection.csv`, `algorithm_comparison.csv`, `classification_report.txt`, `sample_inventory.csv`, `celltype_composition.csv`, `top_features.csv`, and PNG plots.

Re-run from the project folder:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/inspect_atlas_h5ad.py \
  /Users/nisarg/Downloads/atlas-001.h5ad
MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/audit_h5ad_expression_scale.py \
  /Users/nisarg/Downloads/atlas-001.h5ad
MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/train_allcells_atlas.py \
  --h5ad /Users/nisarg/Downloads/atlas-001.h5ad
```

### Continue from here

At the time of this historical checkpoint, the model training was complete and the website was not yet built. The current website implementation is recorded in CP6–CP8. The remaining scientific task is to resolve the 372 `orig.ident` groups to source-study biological patients using official GEO/ENA metadata, then update the grouping key and rerun only if verified mapping changes it. Keep the whole-study outer holdout; do not describe the current groups as patients.

**Do not conflate datasets:** the older GSE136831 model and its 94.9% nested donor-CV score below are historical results from a separate 78-donor dataset. They are not results for `atlas-001.h5ad`.

## Historical checkpoint: GSE136831 Adams IPF atlas

**Project:** exploratory three-class classification of Control, COPD, and IPF lung samples from the Adams IPF Cell Atlas, GEO accession [GSE136831](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE136831).

The large sparse matrix was fully processed on the MacBook Air M1 using a streaming C++ parser, so the 8 GB machine did not load the full matrix into memory. The original data remains in Downloads.

### Dataset audit

- Expression matrix: **45,947 features × 312,928 cells**, with **692,789,348** declared nonzero entries.
- Donor labels verified against metadata and barcode order: **78 donors** — 32 IPF, 18 COPD, 28 Control.
- Cell counts: 147,169 IPF, 69,456 COPD, 96,303 Control.
- All donors had one consistent diagnosis. The matrix, metadata, and barcode files aligned exactly by row/order.
- The matrix row-to-gene-ID sidecar was not present. The training is valid, but the current feature-importance file uses row numbers rather than gene symbols/IDs.

### Method

1. Map every cell to its subject and diagnosis from the study metadata.
2. Stream each sparse matrix entry directly from the Matrix Market file and sum raw counts by donor. The C++ helper uses low memory; it checks all 692,789,348 entries against matrix dimensions.
3. Normalize each donor's summed counts to counts per million and apply `log1p`.
4. Within each training fold, remove constant features, select the top 500 features by ANOVA F-score, and scale them.
5. Compare logistic regression, linear SVM, RBF SVM, and Extra Trees.
6. Estimate performance using nested, stratified five-fold cross-validation. Each outer fold holds out donors; inner folds select the classifier using only the outer training donors. Every donor receives exactly one outer held-out prediction.
7. Select the final algorithm using five-fold donor-stratified development CV, then fit that selected pipeline on all 78 donors for the demo artifact.

### Measured results

The primary estimate is the **nested outer-fold out-of-fold result**, not the score of the final all-data fit:

| Metric | Nested donor-level result |
|---|---:|
| Accuracy | 0.949 (74/78 donors) |
| Balanced accuracy | 0.933 |
| Macro-F1 | 0.938 |
| Weighted-F1 | 0.948 |

Nested out-of-fold confusion matrix, with rows=true and columns=predicted, label order Control / COPD / IPF:

```text
             Control  COPD  IPF
Control          27     1    0
COPD              3    15    0
IPF               0     0   32
```

The inner-fold-selected candidate varied across outer folds (Linear SVM twice, Logistic Regression twice, RBF SVM once), so the report evaluates the **model-selection procedure**, not a single fixed classifier chosen with knowledge of held-out labels. On the full-dataset development comparison, Linear SVM and Logistic Regression tied on mean macro-F1 (0.972); Linear SVM had the higher mean balanced accuracy (0.979 vs 0.972) and was selected as the final algorithm. RBF SVM reached 0.957 mean macro-F1; Extra Trees reached 0.940. These small-cohort scores are not a universal ranking.

The class report shows Control recall 0.964, COPD recall 0.833, and IPF recall 1.000 on the nested predictions. In particular, the COPD class has only 18 donors and three of its held-out samples were predicted as Control.

### How to explain this at review

> “I used the Adams lung single-cell atlas and predicted Control, COPD, or IPF at the donor level. I summed cells into one pseudobulk profile per donor so the same donor could not leak into both sides of validation. Preprocessing and feature selection were repeated inside the folds, and nested cross-validation gave 93.3% balanced accuracy and 93.8% macro-F1 across 78 held-out donor predictions. This is promising exploratory performance, but it comes from one study and still needs independent, study-level validation. It is not a diagnostic tool.”

Avoid saying the model is clinically validated, that 312,928 cells are independent patients, that the result generalizes across hospitals, or that the top matrix rows are confirmed biomarkers.

### Generated outputs

- `scripts/train_lung_atlas.py` — reproducible Python workflow.
- `scripts/aggregate_matrix.cpp` — memory-safe streaming sparse-matrix aggregation helper, compiled when the trainer runs.
- `models/lung_disease_classifier.joblib` — final Linear SVM pipeline fitted on all 78 donors.
- `data/processed/pseudobulk_counts.npz` — derived donor-level count matrix, donor IDs, labels, and row identifiers.
- `reports/lung_atlas/analysis_summary.json` — complete machine-readable cohort, validation, model-comparison, and environment summary.
- `reports/lung_atlas/nested_oof_predictions.csv` — held-out donor predictions and the model selected for each outer fold.
- `reports/lung_atlas/nested_outer_folds.csv`, `classification_report.txt`, and `algorithm_comparison.csv` — fold metrics, per-class results, and candidate scores.
- `reports/lung_atlas/nested_confusion_matrix.png`, `algorithm_comparison.png`, `pseudobulk_pca.png`, and `cohort_overview.png` — figures for review/demo.
- `reports/lung_atlas/donor_summary.csv`, `celltype_composition.csv`, and `top_features.csv` — cohort and model detail.

### Re-run

```bash
cd ~/Desktop/"ml project"
MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/train_lung_atlas.py
```

The script's default inputs are under `/Users/nisarg/Downloads`. Override them with `--matrix`, `--metadata`, or `--barcodes` if needed. It reads the uncompressed `.mtx` file, writes derived reports under `reports/lung_atlas/`, the pseudobulk archive under `data/processed/`, and the model under `models/`.

### Remaining work

This is the historical GSE136831 model, not the active all-cells model served by the current PulmoAI app. If continuing this separate historical workflow, optionally download the small [NCBI GEO gene-ID sidecar](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE136nnn/GSE136831/suppl/GSE136831_AllCells.GeneIDs.txt.gz) and rerun it so feature rows can be named. Do not replace the active app artifact with this historical model. The active all-cells app's runtime-install and browser-integration steps are recorded at the top of this walkthrough and in CP6–CP8. Independent-cohort validation remains necessary before making generalization claims.

## Historical work

Earlier checkpoints in `CHECKPOINTS.md` record an eight-donor COVID-versus-healthy pilot using GSE248688 and feasibility research into a different, 19-disease atlas. Those efforts and the GSE136831 IPF model remain preserved in their original scripts/reports, but are historical; the active application serves the all-cells atlas model described at the top of this walkthrough. See `FULL_MODEL_PLAN.md` for the earlier scope and why it is not the current training target.
