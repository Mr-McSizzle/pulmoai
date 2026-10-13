# PulmoAI — Frontend and Backend Handoff

**Updated:** 13 October 2026  
**Status:** Dependencies installed, production build passed, and integrated local server running at `http://127.0.0.1:8000`.

This file is the continuation brief for a friend or Antigravity. Keep `README.md`, `WALKTHROUGH.md`, and `CHECKPOINTS.md` synchronized if implementation or validation changes.

## Project and model

PulmoAI is a local research interface for the downloaded all-cells lung atlas. It aggregates 1,470,730 cells into 372 study-scoped sample/library profiles across 16 studies, with 30,566 gene features and labels `Donor`, `COPD`, `PF`, and `COVID19`. These 372 groups are **not verified patient counts**: the source metadata does not establish biological patient IDs.

The active artifact is `models/ensemble_benchmark_best.joblib`, a Linear SVM fitted on all grouped profiles after candidate comparison. The saved inference contract is one complete raw-count expression profile with the exact 30,566 feature IDs. The API aligns the CSV headers, normalizes counts to CPM, applies `log1p`, and calls the saved model. The estimator has no calibrated probabilities; never show a probability or confidence percentage.

### Reported evaluation

The main result is nested model selection with 16 outer folds, each holding out one entire source study: accuracy **63.4%**, balanced accuracy **56.1%**, macro-F1 **52.2%**, weighted-F1 **66.7%**. COVID19 recall is **21% (4/19)**. Logistic Regression has the top fixed-candidate macro-F1 (54.1%); Linear SVM has the top fixed-candidate balanced accuracy (57.9%). The chosen final Linear SVM has the best mean inner-CV macro-F1 for model selection, not a proven win on every held-out metric. The hard-voting ensemble scored 49.8% macro-F1.

These are exploratory cross-study results, not external-cohort validation, patient-level evidence, or clinical performance. Do not imply the website improves model accuracy or discovers validated biomarkers.

## Implemented application

### Frontend — `frontend/`

React + TypeScript + Vite + Tailwind v4, with a custom spatial/layered research dashboard and responsive layouts.

- **Overview:** actual dataset dimensions, profile distribution, independently verified headline metrics, and current-session prediction history.
- **Analyze:** one CSV upload; backend validation must pass before the saved-model prediction runs. Results show the returned class and artifact ID, without confidence scores.
- **Model performance:** verified confusion matrix, per-class precision/recall/F1, and fixed-candidate comparison.
- **Gene insights:** top selected Linear SVM coefficients, explained as model weights—not biomarkers or statistical significance.
- **Methodology:** aggregation, CPM/log1p transform, fold-local feature selection, study-held-out evaluation, and key limitations.
- **About:** API/model status, supported file contract, local-only scope, and medical disclaimer.

Recent results live in React state only. The backend does not persist upload history.

### Backend — `backend/` and `src/pulmoai/service.py`

FastAPI loads the trusted local model artifact at startup. It exposes:

- `GET /api/health` — actual model readiness and artifact fingerprint.
- `GET /api/dashboard` — atlas metadata and metrics only when the independent verification report passes.
- `GET /api/insights` — coefficients derived from the fitted Linear SVM.
- `POST /api/validate` — checks the upload format and exact complete feature set.
- `POST /api/predict` — calls the saved classifier and maps its encoded output through the bundled label order.

Uploads are limited to one UTF-8 `.csv` file no larger than 8 MB. Accepted layouts: one wide row with all model genes as headers (optional `sample_id`), or a long CSV with exactly `gene_id,count` / `gene,count`. Counts must be finite, nonnegative, integer-like, and have a nonzero sum. Missing/extra genes, duplicate genes, multiple rows/profiles, normalized expression, images, and routine blood reports are unsupported.

The server binds to localhost by default in the documented commands. This project is not hardened for public deployment.

## Run locally

Dependencies are installed in this workspace. For a fresh environment, install them from the project root:

```bash
.venv/bin/python -m pip install -r backend/requirements.txt
cd frontend && npm ci
```

From the project root, run the backend in one terminal tab:

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In a second terminal tab:

```bash
cd ~/Desktop/"ml project"/frontend && npm run dev
```

Open `http://127.0.0.1:5173`; Vite forwards `/api` to the local API. For a single-origin build, run `npm run build` from `frontend/`, then start the backend and open `http://127.0.0.1:8000`; FastAPI serves `frontend/dist` when it exists.

## Verification and remaining work

- Python service/preprocessing unit tests: run `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` from the project root. Fourteen tests pass in the current environment.
- The saved artifact loaded successfully and returned a real class for a cached raw profile; API report verification returned `independently_verified`; 30 coefficient rows were derived. This smoke test checks the serving contract only and is not an accuracy estimate.
- `npm run build` passes TypeScript checking and creates `frontend/dist`; npm reports zero vulnerabilities. FastAPI serves the built UI and API from port 8000.
- Live checks pass: homepage HTTP 200, model health HTTP 200 with artifact `b84ba2ac5b33`, independently verified dashboard, 30 model-derived coefficients, and upload validation/prediction through HTTP. The upload smoke profile came from the training cache; it is not a generalization result.
- Fourteen Python unit tests pass. The browser has not been visually reviewed at multiple screen sizes; keep the research disclaimer and exact input restrictions visible if making UI changes.
- Do not change, move, or regenerate the source H5AD or model artifacts as part of UI validation.
- Do not change, move, or regenerate the source H5AD or model artifacts as part of UI validation.
- Scientific follow-up remains: resolve source sample groups to biological patients where source metadata permits; then evaluate only if verified grouping changes. Independent external validation and clinical validation are not available.

## Main files

- `frontend/src/App.tsx` — pages and API-driven interactions.
- `frontend/src/styles.css` — spatial theme, responsive layouts, and component styles.
- `frontend/src/api.ts` and `frontend/src/types.ts` — typed local API client.
- `backend/main.py` and `backend/schemas.py` — API routes and typed responses.
- `src/pulmoai/service.py` — saved model loading, report access, upload validation, prediction, and coefficient extraction.
- `src/pulmoai/preprocessing.py` — shared strict CSV parser and CPM/log1p transform.
- `reports/ensemble_benchmark/independent_verification.json` — verified metric source used by the dashboard.
- `WALKTHROUGH.md` and `CHECKPOINTS.md` — project history and next steps.
