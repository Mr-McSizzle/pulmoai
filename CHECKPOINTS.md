# Checkpoint Log

## Checkpoint 0 — Workspace setup and archive inspection

**Status:** Complete  
**Date:** 12 October 2026

- Created the `Desktop/ml project` workspace with `scripts/` and `reports/`.
- Located the original archive at `/Users/nisarg/Downloads/GSE248688_RAW.tar`.
- Confirmed it is a 3.6 GB POSIX tar archive.
- Listed all 68 archive entries: 18 matrices, 18 barcode tables, 18 feature tables, and 14 CMO/genotype/CMO-confidence metadata files across 20 GSM accessions.
- Added `scripts/inspect_archive.py` and saved its output to `reports/archive_inventory.txt`.
- No data was extracted, copied, or modified. No model was trained and no metrics were generated.
- Recorded the environment: Python 3.12.4, arm64, and about 11 GiB free disk space at inspection.

## Checkpoint 1 — Verify sample metadata and prediction labels

**Status:** Complete for an exploratory COVID-versus-healthy pilot  
**Date:** 12 October 2026

- GEO identifies GSE248688 as a super-series; the local archive contains 20 GSM records and is not the complete 1,047-patient, 19-disease atlas described in the associated paper.
- The paired CMO files for `GSM8335813` and `GSM8335815` name the same eight donors: four healthy, three `COVID_PNA`, and one `COVID_mild`.
- The pilot target combines the two COVID labels into COVID versus healthy.
- Aggregate both pools by donor and split by donor ID to prevent leakage. Other pooled cohorts need additional label mapping and are excluded from this first baseline.
- Evidence is in `reports/label_metadata_inventory.txt`; sources are linked in `WALKTHROUGH.md`.

## Checkpoint 2 — Validate selected input files

**Status:** Complete  
**Date:** 12 October 2026

- Selected matrices are 36,601 x 45,930 and 36,601 x 30,741, with 62,014,171 and 49,637,175 nonzero entries.
- Feature and barcode row counts match matrix headers; all 36,601 features are marked `Gene Expression`.
- The second pool's barcodes match its assignment table. The first pool has eight matrix barcodes without assignments and ten assignment-only rows; join by barcode and omit unmatched entries.
- No matrix was extracted. Details are in `reports/covid_subset_validation.txt`.

## Checkpoint 3 — Train a baseline

**Status:** Complete  
**Date:** 12 October 2026

- Created `.venv/` and installed NumPy, SciPy, scikit-learn, matplotlib, and joblib. Exact versions are recorded in `requirements-lock.txt`.
- Streamed the two selected sparse matrices directly from the archive and aggregated assigned counts by donor across both pools.
- Normalized per donor to counts per million, applied `log1p`, removed constant features, selected up to 100 features inside each fold, and fitted L2-regularized logistic regression.
- Four-fold stratified donor-level CV produced balanced accuracy **1.00**, weighted F1 **1.00**, and confusion matrix `[[4, 0], [0, 4]]`.
- Saved out-of-fold predictions, metrics, a confusion-matrix plot, and the final model. These scores are highly unstable at eight donors and do not establish clinical performance.

## Checkpoint 4 — Full-atlas source and feasibility

**Status:** Complete — source and local-data audit  
**Date:** 12 October 2026

- Verified the official processed-atlas deposit, Zenodo record `14851902`, licensed CC BY 4.0. It provides main, unseen-patient validation, and unseen-study external H5AD splits plus patient metadata.
- The 149 kB sample metadata CSV is already present in Downloads and contains 1,047 sample rows across 19 diseases plus healthy controls.
- The official main training matrix is 17.2 GB. The same-named file in `data/atlas/` is only 15 MB and is incomplete; it was not used. Current free disk space is about 30 GiB, but no large transfer was started because the complete analysis also needs validation data and processing space.
- The existing 3.6 GB GEO archive contains only 20 GSM records. Most mixed-donor pools do not include per-cell donor assignments, so they cannot safely be used for patient-level classification.
- No full-atlas data or full multi-disease metrics are available locally. `FULL_MODEL_PLAN.md` records the full-model requirements.

## Checkpoint 5 — Candidate-model comparison on the local pilot

**Status:** Complete — exploratory COVID-versus-healthy pilot only  
**Date:** 12 October 2026

- Compared regularized logistic regression, scaled logistic regression, linear SVM, RBF SVM, and random forest with leave-one-donor-out cross-validation. All preprocessing and feature selection were fit inside each training fold.
- Each model obtained balanced accuracy 1.00, macro-F1 1.00, and weighted-F1 1.00 on the same eight donors (four healthy, four COVID).
- Since all candidates tied, selected L2-regularized logistic regression for its simpler, more interpretable decision function. This is a tie-break on an eight-donor pilot, not evidence that it is globally the best algorithm.
- Saved the comparison report, donor predictions, comparison chart, selected-model confusion matrix, and final all-eight-donor fit under `reports/model_comparison/` and `models/pilot_best_logistic_regression.joblib`.
- The local archive cannot support a full multi-disease patient-level model because most pooled libraries lack barcode-to-donor assignments. The full official expression matrix remains necessary for that objective.

## Checkpoint 6 — Original 19-disease atlas direction

**Status:** Superseded by the smaller GSE136831 project below  
**Date:** 12 October 2026

- The earlier plan required a 17.2 GB main H5AD and additional validation data, which did not fit the user's download/time constraints.
- No 19-disease model was trained. The plan and downloader are retained as historical files; they are not the active review project.

## Checkpoint 7 — Train PulmoAI on the Adams IPF Cell Atlas

**Status:** Complete — exploratory donor-level model  
**Date:** 12 October 2026

- Used the local GSE136831 Matrix Market file and matching metadata/barcodes. The matrix declared 45,947 features, 312,928 cells, and 692,789,348 sparse entries; all entries were streamed and aggregated successfully.
- Verified barcode order, all cell labels, and consistent subject labels: 78 donors (28 Control, 18 COPD, 32 IPF). No cells were treated as independent patient samples.
- Created `scripts/aggregate_matrix.cpp` and `scripts/train_lung_atlas.py`. Counts were summed by donor, normalized to CPM and `log1p`, and evaluated with fold-local feature selection/scaling.
- Compared Logistic Regression, Linear SVM, RBF SVM, and Extra Trees. Nested stratified five-fold donor CV evaluated the whole algorithm-selection procedure; full-data development CV selected Linear SVM for the final all-donor fit.
- Nested out-of-fold metrics: accuracy **0.949**, balanced accuracy **0.933**, macro-F1 **0.938**; 74/78 donors correct. Confusion matrix, per-class report, out-of-fold predictions, comparison scores, PCA, and cohort plots are saved under `reports/lung_atlas/`.
- Saved final model at `models/lung_disease_classifier.joblib` and donor pseudobulk counts at `data/processed/pseudobulk_counts.npz`.
- The gene-ID sidecar was not present. Feature reports use matrix row numbers; gene-level names/biomarker claims are not available yet. Independent-study validation is also still required.
- Updated `README.md`, `WALKTHROUGH.md`, and `FRONTEND_BACKEND_HANDOFF.md` for the active project and actual measured results.

## Checkpoint 8 — New 5.9 GB H5AD received

**Status:** Complete — file located and reader installed  
**Date:** 13 October 2026

- Located `/Users/nisarg/Downloads/atlas-001.h5ad` (6,310,632,229 bytes); the download was not moved, copied, or modified.
- Installed `anndata` and `h5py` in the project `.venv` after the user asked Codex to proceed. The full environment is captured in `requirements-lock.txt`.
- Added `scripts/inspect_atlas_h5ad.py` for backed/read-only metadata inspection. The old GSE136831 model remains separate and its metrics do not apply to this file.

## Checkpoint 9 — Atlas schema and expression audit

**Status:** Complete — full matrix and labels inspected  
**Date:** 13 October 2026

- Inspected in backed mode: 1,470,730 cells × 30,566 features, 16 source studies, sparse CSR storage, 18 `celltypeuse` categories, and unique gene/feature names.
- Metadata fields used: `Disease`, `gse_alias`, `orig.ident`, and `celltypeuse`. Labels are Donor, COPD, PF, and COVID19. Disease cell counts are Donor 501,843; COPD 254,763; PF 635,173; COVID19 78,951.
- `orig.ident` combined with `gse_alias` yields 372 sample/library groups: Donor 128, COPD 46, PF 179, COVID19 19. There is no verified biological patient ID. These are not donor counts.
- The full streamed scale audit examined 2,778,291,635 stored values. No values were negative or non-finite; all were integer-like within float32 tolerance (`rtol=1e-6`, `atol=1e-5`). Strict absolute-tolerance deviations are confined to ERP114453 and consistent with float32 precision around integers.
- Saved schema details to `reports/atlas_h5ad_inspection.json` and the whole-matrix scale report to `reports/atlas_expression_scale.json`. See `scripts/audit_h5ad_expression_scale.py`.
- The public Zenodo record describes 16 studies and the same total cell count. Its sample description and the local 372 study-scoped group IDs are different units; source patient mapping remains unverified.

## Checkpoint 10 — Full all-cells atlas model

**Status:** Complete — nested study-held-out training, model comparison, and final fit  
**Date:** 13 October 2026

- Added `scripts/train_allcells_atlas.py`. It streams 2,048-cell chunks from the read-only H5AD, sums expression by `gse_alias + orig.ident`, normalizes pseudobulk counts to CPM, applies `log1p`, and performs fold-local ANOVA top-500 feature selection.
- Compared Logistic Regression, Linear SVM, RBF SVM, and Extra Trees. Outer evaluation holds out each of all 16 source studies. Inside each outer training fold, a stratified three-fold sample-group CV selects the candidate model; no cells/groups from the held-out study enter model selection or fitting.
- Nested held-out-study results across 372 source groups: accuracy **0.651** (242/372), balanced accuracy **0.601**, macro-F1 **0.560**, weighted-F1 **0.683**. Per-class recall: Donor 0.664, COPD 0.783, PF 0.642, COVID19 0.316. These are cross-study research results, not patient-level or clinical performance.
- Mean inner-CV macro-F1 selected Linear SVM for the final demo artifact (0.903 development-selection score); it was fitted on all 372 groups. The nested unseen-study procedure scored 0.560 macro-F1, exposing substantial study/domain shift. The SVM is not calibrated and provides no validated probabilities.
- Saved `models/allcells_lung_disease_classifier.joblib`, `data/processed/allcells_pseudobulk_counts.npz`, and analysis outputs under `reports/allcells_atlas/`; a one-row inference smoke test passed.
- Critical limitation: no patient identifier was available, and disease distribution is strongly study-dependent. The held-out-study design reduces source-study leakage but does not certify patient-level generalization. Keep this limitation beside all metrics.
- Next scientific step: resolve each source sample/library to biological patient IDs using official source-study metadata and rerun the pseudobulk grouping only if mapping changes it. The website remains unbuilt.

## Checkpoint 11 — Expanded classifier and ensemble comparison

**Status:** Complete — eight classifiers plus an ensemble evaluated on the cached profiles  
**Date:** 13 October 2026

- Added `scripts/benchmark_models.py`; it reuses `data/processed/allcells_pseudobulk_counts.npz` and does not reread, copy, or modify the original 5.9 GB H5AD.
- Compared Logistic Regression, Linear SVM, RBF SVM, Random Forest, Extra Trees, AdaBoost, Histogram Gradient Boosting, a small MLP neural network, and a hard-voting ensemble of Logistic Regression + Linear SVM + Extra Trees. XGBoost was not run because it is not installed; no download was made.
- Kept the 16-fold leave-one-study-out outer evaluation and fold-local feature selection. Model selection occurs inside each outer training split using stratified three-fold sample-group CV.
- Nested selection-procedure metrics: accuracy **0.634**, balanced accuracy **0.561**, macro-F1 **0.522**, weighted-F1 **0.667**. Recalls: Donor 0.66, COPD 0.74, PF 0.63, COVID19 0.21.
- Linear SVM was selected for the final all-profile fit by mean inner-CV macro-F1 (0.903); Logistic Regression had the highest fixed-candidate held-out macro-F1 (0.541), while Linear SVM had the highest held-out balanced accuracy (0.579). The hard-voting ensemble reached macro-F1 0.498 and did not improve the result. Do not claim a decisive universal winner.
- The run took **465 seconds (7 minutes 45 seconds)**. Most time was in Histogram Gradient Boosting (318 seconds) and AdaBoost (84 seconds). Saved the final model at `models/ensemble_benchmark_best.joblib` and reports under `reports/ensemble_benchmark/`.
- Updated `README.md`, `WALKTHROUGH.md`, and `FRONTEND_BACKEND_HANDOFF.md` to use the expanded benchmark and report the new metrics.
- This does not raise accuracy or validate clinical use. It makes the comparison and model selection more transparent; COVID19 performance remains weak and the 372 groups are not verified patients.

## Implementation CP0 — Workspace inspection for the end-to-end platform

**Status:** Complete — inventory verified; no source data modified  
**Date:** 13 October 2026

- Confirmed the workspace is `/Users/nisarg/Desktop/ml project/` and preserves 11 Python scripts, five saved model artifacts, existing metrics/plots, the virtual environment, and current project documentation.
- Confirmed the full source is `/Users/nisarg/Downloads/atlas-001.h5ad` (6,310,632,229 bytes). The similarly named `data/atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad` is only 15,908,864 bytes and is incomplete; it was not used. The original dataset was not moved, copied, or changed.
- Confirmed the derived pseudobulk cache is 372 × 30,566. It contains profiles aggregated by `gse_alias + orig.ident`; those groups are not verified patients.
- Confirmed Python 3.12.4 and the core ML/data dependencies are installed. Node 24.11.1 and npm 11.6.2 are present. FastAPI, Uvicorn, and XGBoost are not installed. No frontend/backend source or automated test suite exists yet.
- Checked disk space: approximately 12 GiB available at inspection. No additional packages or datasets were downloaded.
- Added `data/README.md`; updated `README.md` and `WALKTHROUGH.md`; expanded `.gitignore` to exclude the virtual environment, generated model binaries, processed data cache, and frontend build artifacts. No existing model or result was overwritten or deleted.
- **Checks:** file inventory, file sizes, package availability/versions, frontend toolchain, Git-ignore rules, and free space were inspected. The CP0 check commands were read-only; documentation and ignore rules were then updated.
- **Unresolved:** verify actual H5AD metadata fields and identifier integrity; determine whether any source mapping establishes patient identities; reproduce report metrics independently; implement supported inference preprocessing, API, frontend, and end-to-end tests.
- **Next action:** CP1 and CP2 are recorded below; proceed to CP3 metric reproduction.

## Implementation CP1 — Atlas metadata and model-contract validation

**Status:** Complete — metadata, grouping, and feature contract verified; patient identity unresolved  
**Date:** 13 October 2026

- Added `scripts/validate_atlas_metadata.py` to compare H5AD metadata with the profile cache and saved classifier, using bounded metadata chunks rather than loading the full expression matrix.
- Validation passed: 1,470,730 cells, 30,566 unique/nonblank features, 16 studies, and 372 `gse_alias + orig.ident` groups. Required metadata fields had zero missing values; cell IDs were unique within study; all grouped labels were consistent; source, cache, and model feature order matched exactly.
- Group counts: Donor 128, COPD 46, PF 179, COVID19 19. The H5AD has no explicit patient/subject ID field, so the number of biological patients is unknown and the 372 groups are not patient counts.
- Confirmed model contract: one raw pseudobulk count profile; normalize by total counts to CPM and apply `log1p`; the saved pipeline then performs top-500 feature selection, scaling, and Linear SVM prediction. It is not probability-calibrated.
- **Checks:** validator completed with `passed_with_identity_limitation`; `py_compile` passed. Evidence: `reports/atlas_metadata_validation.json`.
- **Files:** `scripts/validate_atlas_metadata.py`, `reports/atlas_metadata_validation.json`, `data/README.md`, `README.md`, and `WALKTHROUGH.md`.
- **Next action:** CP2 is recorded below; continue to CP3 metric reproduction.

## Implementation CP2 — Strict single-profile preprocessing

**Status:** Complete — exact input contract implemented and tested  
**Date:** 13 October 2026

- Added `src/pulmoai/preprocessing.py`: accepts one profile as either a one-row wide CSV (all model genes as columns, optional `sample_id`) or a long CSV with exactly `gene_id,count` / `gene,count` columns.
- Aligns identifiers to the saved model's exact 30,566-gene order. Rejects missing/extra or duplicate genes, multiple profiles, malformed files, negative/non-finite/fractional counts, and zero-total profiles. Does not impute missing genes.
- Applies the training transform, total-count CPM then `log1p`. Feature selection, scaling, and Linear SVM remain inside the saved model pipeline. The pipeline returns encoded class IDs; `label_order` maps these to class names.
- Added nine unit tests in `tests/test_preprocessing.py`; all passed. A cached real profile produced an exactly matching transformed vector and a valid encoded model output. This smoke check is not a performance estimate.
- **Files:** `src/pulmoai/__init__.py`, `src/pulmoai/preprocessing.py`, `tests/test_preprocessing.py`, `README.md`, `data/README.md`, and `WALKTHROUGH.md`.
- **Limitations:** no API or website yet; no calibrated probabilities; images, routine blood reports, multi-profile uploads, and normalized/log-transformed inputs are unsupported.
- **Next action:** completed in CP3 below.

## Implementation CP3 — Baseline reproduction and metric audit

**Status:** Complete — expanded benchmark reproduced and score discrepancy reconciled  
**Date:** 13 October 2026

- Added `scripts/verify_benchmark_outputs.py` to independently recompute overall metrics, per-class metrics, confusion matrix, and fixed-candidate results from saved out-of-fold predictions.
- The verifier passed for 372 study-scoped profiles across 16 studies. Selected-procedure metrics: accuracy **0.6344**, balanced accuracy **0.5613**, macro-F1 **0.5219**, weighted-F1 **0.6666**; COVID19 recall is **4/19 (0.2105)**.
- Reran the nine-candidate benchmark from the cached profile matrix to isolated `/private/tmp/` outputs (487.24 seconds). All 372 selected predictions, fold selections, confusion matrix, and candidate metrics matched the project report exactly. Existing model/report outputs were not overwritten.
- Explained the 65.1% / 56.0% values in `reports/allcells_atlas/`: they come from a separate four-candidate model-selection menu; the expanded nine-candidate procedure has different predictions. Neither result is external validation.
- **Checks:** verification script passed; all candidate and per-class metrics matched; full rerun reproduced predictions; nine CP2 unit tests and Python compilation passed.
- **Evidence:** `reports/ensemble_benchmark/independent_verification.json`.

## Implementation CP4 — Candidate algorithm comparison

**Status:** Complete — eight classifiers and hard voting independently checked  
**Date:** 13 October 2026

- Verified Logistic Regression, Linear SVM, RBF SVM, Random Forest, Extra Trees, AdaBoost, Histogram Gradient Boosting, MLP, and hard voting of Logistic Regression + Linear SVM + Extra Trees. XGBoost is not installed and was not run; no package was downloaded.
- Recomputed each candidate's accuracy, balanced accuracy, macro-F1, and weighted-F1 from its study-held-out predictions; all matched. Logistic Regression leads fixed-candidate macro-F1 (54.1%); Linear SVM leads balanced accuracy (57.9%) and accuracy (64.8%). Hard voting has 49.8% macro-F1. No one candidate dominates every metric.
- Model and feature selection remain fold-local. Inner-CV scores near 90% macro-F1 are selection values, not unseen-study performance.

## Implementation CP5 — Final model and evaluation review

**Status:** Complete — saved model contract and scientific limits verified  
**Date:** 13 October 2026

- Confirmed `models/ensemble_benchmark_best.joblib` is the selected Linear SVM fitted on all 372 sample/library profiles. It expects the exact 30,566 raw-count features; preprocessing parity and a model smoke test passed.
- The fitted pipeline emits encoded class IDs 0–3, mapped through `label_order`. It has no `predict_proba`; raw SVM decision values are not probabilities or clinical confidence.
- The 63.4% / 52.2% result estimates the nested selection procedure on held-out studies, not the all-data artifact as an independent test. No external cohort is available.
- COVID19 has 19 profiles from two studies; the nested procedure identifies 4/19. Disease labels are study-imbalanced, and biological patient IDs are unavailable. This remains exploratory, not diagnostic or clinical.
- At the time of CP5, the app source had not yet been implemented. CP6–CP8 below record the integrated app and current dependency gate. No packages have been downloaded by Codex.

## Implementation CP6 — FastAPI prediction and reporting backend

**Status:** Source complete — backend dependencies not installed, live server not yet run  
**Date:** 13 October 2026

- Added `src/pulmoai/service.py` to load the existing trusted Linear SVM artifact, calculate a short SHA-256 artifact ID, read only the independently verified report, validate strict CSV bytes using the shared preprocessing contract, return real predictions, and expose selected model coefficients as coefficients (not biomarkers).
- Added `backend/main.py` with startup model loading and `GET /api/health`, `GET /api/dashboard`, `GET /api/insights`, `POST /api/validate`, and `POST /api/predict`. FastAPI responses are typed in `backend/schemas.py`; uploads are capped at 8 MB and only localhost dev origins are allowed.
- The API does not return SVM margins as probabilities, persist uploaded content, process images/blood reports, or change model/training artifacts. If model startup fails, health reports unavailable instead of fabricating readiness.
- Added `backend/README.md` and `tests/test_inference_service.py`.
- **Limitations:** FastAPI, Uvicorn, and `python-multipart` are declared in `backend/requirements.txt`, but not installed. The live HTTP routes and upload transport remain untested until installation.

## Implementation CP7 — Spatial React frontend

**Status:** Source complete — frontend dependencies not installed, browser build not yet run  
**Date:** 13 October 2026

- Added a React + TypeScript + Vite app in `frontend/`, with a custom spatial/layered visual system, sidebar navigation, atlas overview, model performance, one-profile CSV analysis, Linear SVM coefficient explorer, methodology, and project status pages.
- Connected UI state to the real `/api` health, report, insight, validation, and prediction endpoints. Includes loading, API-offline, validation-error, and empty states; shows the returned artifact ID and does not invent probabilities or analysis history.
- Added responsive styles and reduced-motion support in `frontend/src/styles.css`; Vite proxies API requests to localhost port 8000. FastAPI can serve `frontend/dist` after production build.
- Added `frontend/README.md`, package manifest, strict TypeScript config, and ignored build/cache directories. No frontend packages or external assets were downloaded.

## Implementation CP8 — Integration contract and handoff

**Status:** Unit/smoke checks passed; live full-stack verification pending user-installed packages  
**Date:** 13 October 2026

- Updated `README.md`, `WALKTHROUGH.md`, and `FRONTEND_BACKEND_HANDOFF.md` with the implemented architecture, exact dependency/install commands, local run modes, input contract, results, warnings, and remaining work.
- **Checks passed:** 14 built-in Python unit tests (`PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v`); Python syntax compilation for backend, service, parser, and tests; saved-model smoke test on one cached raw-count profile; dashboard verification status `independently_verified`; 30 selected linear coefficients returned.
- The model smoke test only checks loading, feature alignment, and prediction wiring; it is not an evaluation or accuracy result. The sample was drawn from the training cache.
- **Not yet verified:** live FastAPI startup/HTTP upload, frontend TypeScript compilation, production bundle, browser rendering, and end-to-end upload in the UI. `pytest` is not installed; the built-in `unittest` runner was used.
- **User action to continue:** from project root, install Python requirements with `.venv/bin/python -m pip install -r backend/requirements.txt`, then install frontend packages with `cd frontend && npm install`. Start backend and Vite as documented in `README.md`, test the full flow, and append only observed results. No packages have been downloaded by Codex.
- **Scientific follow-up:** patient identity resolution and independent external/clinical validation remain outstanding. Current app is localhost research software only.

## Implementation CP9 — Install, build, and launch

**Status:** Complete — integrated app running locally  
**Date:** 13 October 2026

- Installed the declared Python API requirements and npm dependencies with the user's authorization. Used a temporary npm cache because the default npm cache in the home folder contains root-owned files; no ownership or permissions were changed there. `frontend/package-lock.json` was generated for reproducible installs.
- Installed backend packages include FastAPI 0.143.0, Uvicorn 0.54.0, and python-multipart 0.0.32. Frontend dependencies installed successfully (82 packages; npm audit reported zero vulnerabilities).
- `npm run build` passed TypeScript checking and Vite production build; assets were written to `frontend/dist/`.
- Restarted FastAPI after the build so it serves the production UI and API from one origin. The local server is running at `http://127.0.0.1:8000` with model artifact `b84ba2ac5b33`.
- **Live checks passed:** homepage HTTP 200; `/api/health` reports `model_loaded: true`; `/api/dashboard` reports `independently_verified`, 1,470,730 cells, 16 studies, 372 groups, and 30,566 features; `/api/insights` returns 30 coefficients; `/api/validate` accepts a complete 30,566-feature raw-count profile; `/api/predict` returns a real class and `probabilities_available: false`.
- The upload/prediction smoke test used a cached training profile and verifies transport/wiring only; it is not an accuracy, validation, or unseen-sample result. All 14 Python unit tests still pass after installation.
- **Keep running:** leave the server session active. To stop it, press Ctrl+C in its terminal. To restart, use `.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000` from the project root.
- **Remaining:** optional visual review at mobile/tablet sizes. Scientific patient identity mapping and independent external/clinical validation remain outstanding.
