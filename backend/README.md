# PulmoAI API

The FastAPI service loads the trusted local Linear SVM artifact once at startup. It serves the actual verified reports, model-derived linear coefficients, pre-prediction validation, and live model predictions. It never returns simulated results or uncalibrated SVM margins as probabilities.

## Start locally

From the project root, install `backend/requirements.txt` into `.venv`, then run:

```bash
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the local API schema. The frontend development server proxies `/api` to this server. When a production frontend build exists at `frontend/dist`, FastAPI serves it from the same local origin.

## Endpoints

- `GET /api/health` — reports the real model-load state and artifact identifier.
- `GET /api/dashboard` — returns cohort metadata and only independently verified evaluation results.
- `GET /api/insights` — returns selected-feature Linear SVM coefficients, explicitly labeled as model weights rather than biomarkers.
- `POST /api/validate` — validates one supported CSV before analysis.
- `POST /api/predict` — predicts one compatible profile using the saved model.

## Input and safety

Upload a single UTF-8 `.csv` no larger than 8 MB. Accepted formats are a one-row wide CSV with all exact model feature identifiers as columns (optional `sample_id`), or a long CSV with exactly `gene_id,count` or `gene,count` columns. Counts must be complete, finite, nonnegative, integer-like, and nonzero in total. The API rejects missing/extra genes and multiple profiles; it never imputes missing features.

Images, routine blood reports, normalized/log-transformed expression, multiple profiles, and patient uploads are unsupported. The service is intended for localhost research demonstrations, not public deployment or medical use.
