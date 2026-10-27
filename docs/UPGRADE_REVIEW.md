# PulmoAI longitudinal workspace upgrade

This upgrade extends the existing React/FastAPI app rather than replacing its research services. The source baseline is `927c78d3342fb14a92295c7af7fff0eb64e2cfa8` from Nisarg10-leo/pulmoai. No upstream repository changes were published: the connected GitHub account has read-only access.

## What is implemented

- **Patient workspaces:** owner-scoped patient aliases and explicitly reported biological sex. Existing uploads remain intact after an additive SQLite migration. Earlier uploads can be assigned to a patient and observation date.
- **Batch ingestion:** up to 12 reports or images, 10 MB each and 60 MB total. Dates are entered per source document. All metadata is validated before processing; each file has its own saved/duplicate/error outcome. A failed file does not erase successful observations.
- **Lab evolution:** reviewed values are plotted against actual observation dates. Baseline/latest deltas require distinct, unambiguous dates and matching normalized units. Directional labels require the same parseable reference interval. Movement toward a reference interval is not described as overall health improvement.
- **Image evolution:** original images can be reviewed side by side. Model score series require reviewed accepted inputs, a shared modality and acquisition series, and identical classifier/gate/policy fingerprints. Rejected inputs, missing model identities and disabled retinal staging are excluded from score comparisons. Scores do not establish disease severity or treatment response.
- **Human review:** original extractions remain intact; corrected values are stored separately. Explicit confirmation is required. Reassigning the patient, date or acquisition series resets review. Single-report risk transfers also start unconfirmed.
- **Traceability:** timestamped account-scoped create/link/review/delete activity, source downloads, and authenticated JSON timeline export. The activity log is a local audit aid, not tamper-evident regulatory logging.
- **Reported-sex lung workflow:** the classifier accepts female/male/intersex/unknown metadata. It reports exactly which model scope was used. The released atlas has no verified sex labels, so the default is explicitly pooled. Separate research models can be trained and evaluated locally, then pinned by SHA-256 in an optional registry. No sex is inferred from expression and no confidence score is adjusted arbitrarily.
- **Operations:** bounded request bodies, document/image decode checks, PDF page limits, origin checks on mutations, login/register throttling, no-store API responses, request IDs, and fail-closed artifact checks. `/api/ready` reports unavailable modules. Docker and CI definitions are included.

## Critical issues found and corrected

| Original behavior | Resolution |
| --- | --- |
| Hash/perceptual-hash matched X-ray returned fixed 85% pneumonia output and bypassed pediatric confirmation | Removed sample recognition and all output overrides. Regression checks verify that output follows model logits. |
| API and frontend silently assumed pediatric age confirmation | Defaults are false; users must explicitly confirm the relevant 1–5-year cohort. |
| `GEMINI_API_KEY` silently enabled outbound lab-report transfer despite offline claims | Removed the cloud extraction helper from the runtime reader. Environment keys cannot authorize transfer. |
| Every lung result was described as high confidence | Replaced with an exploratory interpretation and model-scope disclosure. |
| OCR failure could decode binary images as text | Plain-text decoding is restricted to `.txt` inputs. |
| Artifact verification exceptions and omitted manifest entries could permit inference | Fail closed; readiness reflects unavailable model families. |
| Future-risk loader uses `future_risk_models.joblib`, while only the v3 audit artifact was pinned | Also pinned the actual legacy service artifact. This upgrade does not claim the service now implements the v3 estimand. |
| CSV temporary-file reopening failed on Windows | Use a closed file inside a temporary directory. |
| Saved sources lacked patient/observation chronology | Added patient/date/series context, review and comparison services. |
| Overview claimed 17 studies/407 donors | Uses actual dashboard study/profile counts and preserves the patient-identity limitation. |

## Run locally

Python 3.12 and Node.js 22 are suitable for the supplied workflows. Install Tesseract separately for scanned documents; digital PDF text and TXT reports do not require the executable.

```bash
python -m venv .venv
# Activate .venv using your shell's normal command.
python -m pip install -r requirements.txt -c requirements-runtime.txt
cd frontend
npm ci
npm run build
cd ..
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, create your own local account, and choose **Patient timelines**. The **Explore an example case** button creates three synthetic lab visits. Review their sources, confirm each, and switch to **Lab evolution**. Synthetic observations are never automatically reviewed.

The `frontend/dist` build is included. Node.js is needed only to rebuild it.

## Vision checkpoints

The upstream repository excludes all six required `.pt` vision checkpoints. Original images can now be uploaded with **Store originals · no AI screening**, reviewed and compared side by side without model weights. Automated model inference remains unavailable until the matching classifier and gate checkpoints are installed. See `docs/WEIGHTS.md`. Readiness correctly returns HTTP 503 in this clean checkout. Lab timelines, original-image archives and the shipped numerical research models remain available. See `docs/PRESENTATION_UPDATE.md` for source-linked evidence and persistent generated-example provenance.

Do not replace absent checkpoints with fabricated scores. Retraining creates new model versions; review their evidence and deliberately update trusted manifests rather than reusing old published metrics.

## Training sex-specific research candidates

The released inventory and metadata audit do not contain verified sex or patient IDs. This is a data limitation, not a missing selector. You must supply a metadata CSV with:

```csv
sample_id,patient_id,study,reported_sex,disease
sample_001,verified_patient_001,GSE_example,female,Donor
```

The count NPZ must contain `counts` (profiles × complete model gene set), `feature_names` in deployed order, and `sample_ids`. All IDs must match metadata. The trainer rejects duplicate patients, incomplete labels, missing training classes, normalized/nonfinite/negative counts, fewer than three source studies, and fewer than ten evaluated profiles per class in either sex cohort.

```bash
python scripts/train_sex_specific_lung.py --counts /path/counts.npz --metadata /path/metadata.csv --output work/sex_candidates
```

It writes separate fitted candidate artifacts, study-held-out out-of-fold predictions, metrics and a checksum registry. Feature selection and fitting run inside each held-out-study fold. This is a fixed-family retrospective research evaluation, not independent clinical validation or nested hyperparameter selection. Review performance, subgroup support, provenance and consent before deploying candidates. The minimum counts are software eligibility gates, not a statistical power calculation.

After review, an administrator may copy the candidate `.joblib` files and `registry.json` into `models/lung_sex_specific/`. The service checks evidence shape, support, feature/label contracts and checksums before loading them. Restart after registry changes. Unknown/intersex inputs use the pooled model until appropriate separately evaluated models exist. Retain the pooled fallback disclosure.

## Container and CI

```bash
docker compose up --build
```

Compose binds only to loopback, runs as a non-root user, uses a read-only container filesystem and persists local records in a named volume. Its strict readiness healthcheck stays unhealthy while required checkpoints are absent. One worker is intentional: SQLite and process-local throttling are suited to a local deployment. For a shared deployment, use a real reverse proxy, TLS, centralized rate limiting and reviewed deployment configuration.

`docker compose config --quiet` passes. The Docker daemon is unavailable in the authoring environment, so the image build and container runtime were not executed. The GitHub Actions workflow has been authored but has not run on GitHub.

## Verification

- Frontend: `npm run build` (TypeScript and production Vite build).
- Backend: `PYTHONPATH=src python -m unittest discover -s tests -v`.
- UI: browser check of local sign-in, synthetic batch ingestion, source review, persistence after reload, and chart dates/deltas at a narrow viewport.
- New regressions cover owner isolation, metadata validation, duplicate prevention, partial batch failures, corrections preserving originals, review reset after reassignment, unit/reference/date boundaries, image model/site separation, unavailable OCR, no cloud transfer with an environment key, optional sex-model routing, checksum rejection before deserialization, streamed body bounds, throttling and actual model-logit output.

Exact local verification results and remaining dependency-related skips are recorded in `VALIDATION.md`.

## Remaining work before clinical or shared production use

This is a substantial research application upgrade, not a clinically validated medical product. Patient data is still not encrypted at rest. There is no organizational RBAC, MFA, tamper-evident logging, validated backup/restore workflow, externally managed secrets or regulatory certification. Uploaded images may contain identifying metadata. Use synthetic/de-identified records until storage, consent, retention and deployment controls are reviewed.

The underlying lung model remains weak for COVID-19, and reference-range movement alone cannot establish treatment benefit. The current vision models have known modality-gate failure modes and do not support longitudinal disease severity. Independent clinical and demographic validation needs new, properly annotated data.

The code review covered backend routes/storage/schemas, all five service modules, frontend screens/data access, model and audit metadata, training entry points, dependency manifests and tests. Large historical audit/training scripts were inventoried and sampled; the full historical research literature and every reported metric were not re-audited in this implementation pass.

A design reference for transparent subgroup evidence and limitations is the [FDA's machine-learning transparency principles](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/transparency-machine-learning-enabled-medical-devices-guiding-principles). This project does not claim regulatory conformity from following them.
