# Local validation — 2026-10-27

Baseline: `927c78d3342fb14a92295c7af7fff0eb64e2cfa8`.

## Results

- **Backend: 102 tests run; 92 passed, 10 skipped; no failures.**
- Nine skips require the upstream-excluded vision checkpoints. One integration test requires a local Tesseract executable, which is absent here. Decoder, parser and image-model safety regressions using controlled inputs still run.
- **Frontend:** TypeScript check and Vite production build passed (1,588 modules; built assets included).
- **Diff hygiene:** `git diff --check` passed.
- **Compose:** `docker compose config --quiet` passed. A Docker daemon was not running, so no image build or container runtime is claimed.
- **Browser:** exercised isolated synthetic local sign-in, batch creation of three dated reports, explicit source review of each, persistence after reload, chronological charts, baseline/latest deltas, and narrow-viewport layout. The original timeline preview shows the generated cholesterol series (220 → 170 → 130 mg/dL). The evidence update also has browser coverage of the source-backed metric cards and processing workflow controls.
- **Model honesty:** the released sex-model registry is empty; female/male inputs use an explicitly disclosed pooled fallback. Tests verify routing to separate supplied models and reject checksum failures before deserialization. No newly trained sex-specific model is claimed.
- **Vision deployment:** missing `.pt` artifacts remain missing. Tests confirm the missing module fails closed; no substitute prediction is generated.
- **CI:** workflow authored but not dispatched to GitHub.
- **Evidence/provenance regressions:** confirm exact report values, no fabricated fallback on missing evidence, report-download whitelisting, correct lung procedure selection, persistent generated-example labels after review/export, and original-image archival without invoking a vision model, authenticated TIFF-to-PNG previews, and unchanged original downloads.

## Commands

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
npm --prefix frontend run build
git diff --check
docker compose config --quiet
```

On Windows, set `PYTHONPATH` in PowerShell instead of using the inline POSIX assignment. The authoring environment used workspace-local additional packages; normal installations should use a virtual environment and `requirements-runtime.txt` constraints.

## Runtime used

Python 3.12 on Windows; selected package versions:

- numpy 2.2.6
- scipy 1.16.0
- pandas 2.3.3
- scikit-learn 1.9.1
- scikit-survival 0.28.0
- joblib 1.6.0
- fastapi 0.116.2
- pydantic 2.11.9
- torch 2.12.0+cu130
- torchvision 0.27.0+cu130
- pymupdf 1.26.4
- lifelines 0.30.0

The runtime constraints pin the scikit-learn/scikit-survival family to match persisted model compatibility. They are not a complete deployment lockfile. Existing upstream dependency locks and model-card metric discrepancies need a separate reproducibility review before publishing a clinical or scientific claim.

## Limits

The local records database and uploaded files are not encrypted at rest. This upgrade does not establish clinical validation, demographic fairness, disease-severity progression, treatment response, regulatory clearance or suitability for identifiable patient data. See `docs/UPGRADE_REVIEW.md` for architecture changes, operation and remaining work.
