# PulmoAI presentation walkthrough

All records in this pack are generated examples, not records of a person receiving care. The dates are illustrative observation dates. Numeric risk and RNA outputs are calculated by the installed research models; no score is tuned to a desired outcome. Images are marked illustrations for archival comparison and are never passed off as clinical scans.

## Open the prepared timeline

Sign in, choose **Patient timelines**, then **Load presentation case**. This creates a separate example patient once per account. Repeating the action reopens the same case and preserves reviews/corrections. Four checked visits span approximately six months, ending two weeks before preparation. An extra report dated on the preparation day remains unreviewed for a live exercise. The core generated sources are checked by the preparation routine; this is distinct from source review by a clinician or user.

Use the guide cards or timeline tabs:

1. **Visit history:** show all modules grouped by observation date; open any source.
2. **Lab evolution:** choose Total Cholesterol (262 to 184 mg/dL), Hemoglobin (10.4 to 13.1 g/dL), then Serum Creatinine (0.7 to 1.4 mg/dL). The first two approach the supplied intervals while creatinine moves away. Inspect a point with mouse or keyboard, open all observations and compare dates/units. These examples demonstrate reference-range movement, not treatment efficacy.
3. **Risk evolution:** inspect chronic existing/reported-status estimates and each future mortality horizon separately. There are four dated calculations per available risk service using the same artifact. The displayed change is percentage points of a model estimate, not proof of disease improvement. Estimates may move in different directions as multiple inputs change.
4. **Image comparison:** select the first and latest marked illustration. The archive workflow, authentication, preview, original download and acquisition-series linkage work without vision checkpoints. No inference, probability, image registration or severity measurement is claimed. Screening real scans still requires installed matching model/gate weights.
5. **RNA history:** show four dates, the actual classification and sex-context/model scope. The same generated raw-count profile is repeated deliberately; it does not represent independent profiles or biological progression. Open/download one CSV and use Single-cell RNA to validate/classify it again.
6. **Overall assessment:** show latest confirmed measurements, all model findings, review priorities, provenance and coverage/processing gaps. It is a research summary, not a validated multimodal diagnosis.

## Live source review and OCR-to-risk transfer

Use **Start live source-review exercise**. Open the original text report and compare its values. In the review dialog, correct a field if demonstrating a correction, check the source-confirmation box and save. Reopen the report and choose **Send saved confirmed values to risk**. The source, patient and observation date carry across.

For the OCR screen itself, download `sources/<preparation-date>/generated-review-exercise.txt` from this pack. Select the example patient and the report date, go to **Lab report OCR**, upload the file and extract it. The report uses the actual text-extraction pipeline. Confirm only selected fields to demonstrate partial review, or confirm all. Change a value, observe that it needs confirmation again, then use **Send N confirmed values to risk**.

The example age is **58** and reported biological sex is **Female**. Enter the missing age explicitly in the risk form. Unknown answers can remain blank. The research models may impute missing features; PCE/IDRS published scores require their explicit inputs and display missing-input notices. If showing complete published scores, use these declared example answers: no current smoking, non-Black PCE category, on BP medication, no diagnosed diabetes, no prior cardiovascular disease, moderate activity, one parent with diabetes, and the waist measurement for that visit (103, 99, 95, 91 cm respectively).

Running a new calculation saves additional records. Same-day duplicate endpoints are intentionally not given an unambiguous directional delta. Use a new observation date for an actual additional visit rather than relabeling the same source as independent evidence.

## Batch, export, source access and account isolation

The pack includes the four original dated TXT reports, the additional review exercise, four generated CSV sources and four PNG illustrations when those modules are available. `manifest.json` records exact dates and dependencies. `timeline.json` contains example records and exact model inputs/outputs. None of these files establishes clinical provenance.

To demonstrate batch ingestion, create another de-identified example patient and upload the four visit TXT files together. Assign the report dates printed in the sources. Batch size/file/date checks still apply. Review the sources to include them in charts. Repeating the same batch for the same patient/date returns duplicate outcomes. A mixed batch can preserve successful files while showing per-file errors.

Use **Export timeline** to download the patient JSON, and **Download all example sources** to obtain this ZIP. Original downloads and PNG previews require sign-in and owner access. Use **Activity log** to inspect creation, linkage and review events; it excludes document contents and values.

## Availability checks before presenting

- Text report extraction, source review, batch ingestion, charts, export and image archives work locally.
- Actual RNA/chronic/future outputs require the matching checksum-verified model bundles and compatible dependencies. Missing services are reported in the case notices; substitute predictions are never manufactured.
- Scanned-photo OCR additionally needs local Tesseract. TXT reports avoid that requirement; they do not demonstrate camera OCR accuracy.
- Image AI screening needs the matching upstream-excluded `.pt` classifier/gate checkpoints. Archive comparison remains available. See `docs/WEIGHTS.md`.
- Sex-specific RNA weights require independently supplied, verified metadata and evaluated models. The released registry uses a disclosed pooled fallback.
- Model benchmark cards, gene coefficients, methodology, uploads and account settings are accessible from the sidebar. Benchmark numbers retain their source reports and limitations.

Use this prepared case for presentation. Do not describe it as a real patient's clinical course or the model outputs as validated diagnoses.
