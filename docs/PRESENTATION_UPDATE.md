# Presentation and evidence update

The interface now uses clearer example-case language while retaining the fact that generated examples are not patient records. Original scientific benchmark labels, dataset perturbation notices, research limitations and measured results remain intact. No model was retrained or assigned a higher score in this update.

## Changes

- Added **Evidence you can inspect** to the overview and benchmark pages. Figures are read from named repository audit reports, with populations, evaluation scope, report downloads and report SHA-256 identifiers.
- Corrected the lung dashboard to use the dual-rule audit (64.78% accuracy, 57.92% balanced accuracy, 0.5349 macro-F1). It previously displayed an older model-selection procedure. Mismatched per-class results and confusion matrices are no longer displayed as if they belonged to the deployed procedure.
- Historical results are explicitly described as historical. The evidence panel shows both clean-document and scan performance, including errors; it does not imply that the current release was clinically evaluated.
- Generated-example provenance is stored with extraction/screening results and preserved in patient exports. Uploaded sources are marked unverified rather than asserted to be authentic patient records. Source review does not remove the origin label.
- Added **Store originals · no AI screening** for image batches. This path works without vision model weights, preserves dated originals for side-by-side review, and generates no prediction, confidence or progression score. AI screening still requires matching checkpoints and the appropriate cohort confirmation.
- Authenticated first-frame PNG previews support formats such as TIFF in the browser. Display previews are bounded and may be downscaled; original downloads remain unchanged.
- Generated image patterns now preview first rather than automatically submitting an inference request against potentially stale modality state. Model scores are labelled as scores rather than individual-case accuracy.

## Using the image archive

Open **Patient timelines**, select a patient alias, choose **Medical images**, and select **Store originals · no AI screening**. Choose files and enter their observation dates and acquisition series. After upload, the source records and side-by-side comparison remain usable without vision checkpoints. The archive does not verify that an image is medical or derive disease improvement/decline.

## Evidence sources

| Module | Named report |
| --- | --- |
| Dual-rule lung procedure | `reports/audit_lung_part_a.json` |
| Chronic risk temporal holdout | `reports/audit_nhanes.json` |
| Document extraction holdout | `reports/audit_ocr_pass9b.json` |
| Modality gate acceptance/rejection | `reports/audit_modality_gate_v3.json` |

These historical audit files are supplied by the upstream project. Their presence and checksums make the displayed source traceable; they do not independently validate the current model or create clinical approval.
