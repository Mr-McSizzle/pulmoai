# PulmoAI — Multi-Study Lung Disease Analysis

PulmoAI is an exploratory research project that compares healthy-donor lung samples with COPD, pulmonary fibrosis (PF), and COVID-19 using the downloaded all-cells lung atlas. The active model was trained from `/Users/nisarg/Downloads/atlas-001.h5ad` on 13 October 2026. The original 5.9 GB H5AD remains in Downloads; the workflow reads it in backed, read-only chunks.

## Current implementation status

- Available: dataset inspection, count-scale audit, pseudobulk preprocessing, study-held-out model comparison, saved model artifacts, evaluation reports, and plots.
- Implemented: a FastAPI backend and responsive spatial React dashboard with live model status, real benchmark results, candidate comparison, SVM coefficient explorer, strict CSV validation, and real model predictions.
- Dependencies are installed and the production UI/API server is currently running at `http://127.0.0.1:8000`.
- The saved estimator is Linear SVM and has been smoke-tested on compatible processed profiles. This does not mean arbitrary CSV, cell-image, or routine blood-report inputs are supported.
- Environment: Python 3.12.4, Node 24.11.1, npm 11.6.2; FastAPI/Uvicorn/multipart and frontend packages are installed. XGBoost was unavailable during model benchmarking and was not used.
- Metadata integrity and model feature order passed validation; biological patient identity remains unresolved. The app is local research software, not a medical device.
- Check `WALKTHROUGH.md` and `CHECKPOINTS.md` for the active checkpoint, setup instructions, and remaining verification steps.

## Run the integrated app

The server is running now. To install dependencies in a fresh copy of the project, use:

```bash
cd ~/Desktop/"ml project"
.venv/bin/python -m pip install -r backend/requirements.txt
cd frontend && npm ci
```

To restart the integrated production server from the project root:

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. For frontend-only development, start `npm run dev` from `frontend/` in a second terminal; Vite proxies `/api` to the backend. More detail is in `backend/README.md`, `frontend/README.md`, and `WALKTHROUGH.md`.

Run the existing Python checks from the project root with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

`npm run build` succeeds, and live homepage, health, dashboard, insights, CSV validation, and prediction endpoints have been checked. The prediction smoke test used a profile from the training cache; it is not a generalization estimate.

## Current dataset and model

- Source: [Zenodo all-cells lung atlas](https://zenodo.org/records/21318856), comprising 16 public studies and 1,470,730 cells.
- Local matrix: **1,470,730 cells × 30,566 features**; the full matrix passed finite, nonnegative, float32-tolerant integer-count checks.
- Labels in the local metadata: `Donor`, `COPD`, `PF`, and `COVID19`; 372 study-scoped `orig.ident` groups were aggregated into one pseudobulk profile each.
- Group counts: 128 Donor, 46 COPD, 179 PF, and 19 COVID19. These are source sample/library groups, **not verified patient counts**; no biological patient ID is provided in the H5AD.
- Final benchmark artifact: `models/ensemble_benchmark_best.joblib` (Linear SVM refit on all 372 groups after model comparison). The earlier `models/allcells_lung_disease_classifier.joblib` is preserved.

## Evaluation results

The expanded benchmark holds out each entire source study in turn (16 outer folds). It compares eight classifiers plus a hard-voting ensemble, selecting a model inside each outer training fold. The main result below is for that model-selection procedure and is not a score on the final all-data demo fit.

| Metric | Held-out-study result |
|---|---:|
| Accuracy | **63.4%** |
| Balanced accuracy | **56.1%** |
| Macro-F1 | **52.2%** |
| Weighted-F1 | **66.7%** |

Per-class recall for the selected procedure is Donor **66%**, COPD **74%**, PF **63%**, and COVID19 **21%**. In the fixed-candidate held-out-study comparison, Logistic Regression had the highest macro-F1 (**54.1%**); Linear SVM had the highest balanced accuracy (**57.9%**). Linear SVM was selected for the final fit by mean inner-CV macro-F1 (**90.3%**, versus Logistic Regression **90.0%**). The high inner-CV scores do not represent unseen-study performance. The hard-voting ensemble scored **49.8% macro-F1**, so combining models did not help here. No algorithm is a decisive winner across every metric.

The expanded benchmark was rerun with the same seed on cached profiles and reproduced identical study-held-out predictions. Its metrics were independently recomputed; all nine candidate-model metric rows and the confusion matrix match. See `reports/ensemble_benchmark/independent_verification.json`. The earlier **65.1% accuracy / 56.0% macro-F1** report in `reports/allcells_atlas/` used a different four-model selection menu, so it is a different valid procedure. These checks reproduce computation on this dataset; they are not independent-cohort or clinical validation.

The benchmark used the cached 372 × 30,566 pseudobulk matrix and completed in **7 minutes 45 seconds** on this machine; it did not reread the 5.9 GB H5AD. XGBoost was not run because it is not installed. These results are exploratory cross-study research results, not clinical performance.

## Reproduce

```bash
cd ~/Desktop/"ml project"
MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/inspect_atlas_h5ad.py \
  /Users/nisarg/Downloads/atlas-001.h5ad
MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/audit_h5ad_expression_scale.py \
  /Users/nisarg/Downloads/atlas-001.h5ad
MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/train_allcells_atlas.py \
  --h5ad /Users/nisarg/Downloads/atlas-001.h5ad
MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/benchmark_models.py
```

The atlas trainer aggregates cells by `gse_alias + orig.ident`. The separate benchmark reuses those cached profiles and compares Logistic Regression, Linear/RBF SVM, Random Forest, Extra Trees, AdaBoost, Histogram Gradient Boosting, a small MLP neural network, and a hard-voting ensemble. Feature selection stays inside each fold; the outer test always holds out a complete source study. Linear regression and reinforcement learning do not fit this supervised multiclass target.

## Main outputs

- `scripts/train_allcells_atlas.py` — chunked pseudobulk aggregation, study-held-out validation, model comparison, plots, and final fit.
- `scripts/benchmark_models.py` — cached-profile comparison of eight classifiers plus hard voting; reports runtime and writes the selected final model without rereading the large H5AD.
- `scripts/verify_benchmark_outputs.py` — independently recomputes selected-procedure and fixed-candidate metrics from saved out-of-fold predictions and checks them against the report.
- `reports/ensemble_benchmark/RESULTS.md` — readable model comparison and headline results; `analysis_summary.json` contains detailed metrics and runtime.
- `models/ensemble_benchmark_best.joblib` — final Linear SVM selected by the documented inner-CV rule and refit on all sample groups.
- `scripts/audit_h5ad_expression_scale.py` — full-matrix read-only expression-scale audit.
- `scripts/validate_atlas_metadata.py` — read-only identifier, class-label, sample-group, feature-order, and model-contract validation; see `reports/atlas_metadata_validation.json`.
- `src/pulmoai/preprocessing.py` — strict one-profile CSV parsing and CPM/`log1p` preparation; supports a one-row wide CSV or a long `gene_id,count` CSV with the complete exact feature set.
- `src/pulmoai/service.py` — trusted model loading, model artifact fingerprint, strict upload handling, predictions, report data, and model-derived coefficient insights.
- `backend/main.py` and `backend/schemas.py` — local FastAPI health, dashboard, gene insights, validation, and prediction endpoints.
- `frontend/` — responsive spatial research workspace; its pages consume the local API and do not fabricate results.
- `tests/test_inference_service.py` — inference-service tests for valid inputs, class mapping, unsupported genes, and upload limits.
- `tests/test_preprocessing.py` — focused tests for alignment, normalization, and rejection of invalid or unsupported input.
- `reports/allcells_atlas/analysis_summary.json` — source of truth for cohort counts, metrics, model selection, environment, and limitations.
- `reports/allcells_atlas/study_heldout_predictions.csv`, `study_heldout_folds.csv`, `inner_model_selection.csv`, and `algorithm_comparison.csv` — reproducible evaluation details.
- `reports/allcells_atlas/study_heldout_confusion_matrix.png`, `algorithm_comparison.png`, `study_disease_coverage.png`, `sample_inventory.csv`, `celltype_composition.csv`, and `top_features.csv` — figures and analysis tables.
- `data/processed/allcells_pseudobulk_counts.npz` — derived 372 × 30,566 count matrix and feature/sample metadata.

## Important limitations

- The H5AD has `orig.ident`, disease, study, and cell-type metadata, but no verified biological patient identifier. The 372 groups must not be described as 372 patients.
- The Zenodo record's sample description and the local `orig.ident` group count are not the same unit; identity mapping should be verified against each source study before claiming patient-level evaluation.
- Holding out studies tests transfer across source studies, but the disease labels are unevenly distributed across studies. Study/batch effects remain substantial, as reflected in the modest metrics.
- Predictions are research demonstrations only. The Linear SVM is not probability-calibrated, the feature ranking is not a validated biomarker list, and the model is not for diagnosis or treatment.
- A larger model list or ensemble cannot guarantee better accuracy. The expanded benchmark confirms that Logistic Regression and Linear SVM are close, while the tested voting ensemble underperforms them; COVID19 recall remains low.

Read `WALKTHROUGH.md` for the full checkpoint narrative, `CHECKPOINTS.md` for the dated log, and `FRONTEND_BACKEND_HANDOFF.md` for the updated website/API brief. The older GSE136831 78-donor model remains archived in the project and its 94.9% result does **not** describe this atlas.
