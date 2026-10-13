# PulmoAI Spatial Frontend

Responsive React + TypeScript workspace for the local PulmoAI research API. The interface uses a spatial, layered visual language for the lung-atlas overview while keeping performance metrics, upload requirements, and research limitations explicit.

## Features

- Overview of the real atlas dimensions, grouped sample/library profiles, class distribution, and independently verified study-held-out metrics.
- CSV validation followed by a real prediction from the saved Linear SVM model.
- Held-out-study confusion matrix, per-class metrics, and fixed-candidate algorithm comparison.
- Searchable Linear SVM coefficient table, labeled as model weights rather than biomarkers.
- Methodology, input contract, model artifact status, and persistent research-only disclaimers.
- Recent predictions are held only in the current browser session; the API does not store upload history.

## Start in development

Install npm dependencies after the Python API requirements are installed. From the project root:

```bash
cd frontend
npm install
npm run dev
```

In a second terminal, start the backend from the project root:

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` requests to the local backend at port 8000.

## Build and serve from FastAPI

```bash
cd frontend
npm run build
cd ..
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. FastAPI serves `frontend/dist` when that production build exists.

## Input contract

The upload view accepts one UTF-8 CSV, up to 8 MB: either one wide row with all exact model gene identifiers as columns (optional `sample_id`), or a long table with exactly `gene_id,count` / `gene,count`. Counts must be complete, nonnegative, integer-like raw counts. Images, routine blood reports, normalized expression, multiple profiles, and missing-gene imputation are unsupported.

The frontend does not invent metrics, probabilities, confidence, sample histories, or biological claims. Prediction is a research demonstration only, not medical advice or a diagnosis.
