# Connected patient workflow

Select an active patient and report/visit date at the top of any screen. This explicit selection follows navigation across RNA, OCR, risk and imaging. It resets on reload/sign-out so the next operator must deliberately select a patient. Dates are observation dates, not upload times.

| Module | Input | Saved timeline finding |
| --- | --- | --- |
| Single-cell RNA | Compatible raw-count CSV and reported biological sex | Expression class, reported sex/model scope, artifact identity and source file |
| Lab report OCR | A report plus sex context for reference intervals | Original extraction, explicit confirmed/corrected measurements, units and reference intervals |
| Chronic risk | Confirmed OCR values or manually supplied biomarkers/history | Condition-specific existing/reported diagnosis estimates and the exact normalized inputs |
| Future risk | Same biomarker form | Separate mortality endpoints and horizons; published equations remain separate |
| Medical imaging | Supported image, modality and named acquisition series | Accepted/rejected screening output or an archive-only original; compatible images can be reviewed side by side |

Every successful module run under this context is linked automatically. A failed module does not create a successful finding. Both risk engines run independently; one service failure preserves the other result. Original file hashes, model artifact identifiers and OCR source-record lineage are retained. Model output is not a clinical diagnosis.

## OCR to risk

Extract a report, correct values, then explicitly check the values you have verified against the source. Use **Send N confirmed values to risk**, above the results table, or the sidebar transfer button. The selected corrections are saved first. Only selected values are transferred; confirming part of a report does not confirm the rest. Review missing demographics/measurements in the risk form and explicitly choose **Calculate**. Both resulting records retain the patient/date, source record and generated-example provenance.

You can also open a reviewed lab report from **Source records** and use **Send saved confirmed values to risk**. Save any newly edited values before transfer. For partially reviewed reports, only the saved confirmed fields are eligible.

Unknown form fields start blank. The API translates visible form names into persisted-model features (including sex, creatinine, pulse, waist, smoking and activity). Published PCE/IDRS point scores display missing-input notices instead of filling answers with medians/defaults. PCE eligibility also requires an explicit prior-CVD answer.

## Timeline assessment and evolution

The **Overall research assessment** consolidates module coverage, latest reviewed laboratory values/flags, latest chronic and future estimates with their estimands/horizons, RNA history, imaging findings, pending source reviews and coverage gaps. It links each finding to its input/source review. Generated examples remain identified. Archived or rejected images do not generate a diagnosis or accepted class.

Lab charts use confirmed values only; partially reviewed reports contribute only confirmed fields. Numeric direction relative to the same reference interval is available across distinct dates and compatible units. Conflicting same-day endpoints do not get a directional delta.

Risk evolution compares only fully reviewed, dated records with the same artifact, target and horizon; changes are percentage points in model estimates. Image trends require matching modality, acquisition series and artifact with accepted output. RNA findings are a dated history, not a scalar improvement score. No validated multimodal fusion model is deployed, so no combined diagnostic probability is invented.

Earlier account-only uploads can be attached from **Source records → Link an earlier upload**, including RNA and both risk-record types. Select their real observation date and patient deliberately. Reassignment requires source review again; records are never assigned to an arbitrary patient automatically.

Legacy API calls without context retain account-only history for backward compatibility. The web interface requires patient/date context for new analyses.
