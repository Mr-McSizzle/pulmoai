"""Reproducible presentation records. Generated inputs, actual available numeric models."""
from __future__ import annotations

import hashlib
import io
import json
import threading
import zipfile
from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, Request
from PIL import Image, ImageDraw, ImageFont

from backend.analysis_records import normalize_risk_inputs
from backend.schemas import FutureRiskResponse, PredictionResponse, ReportExtractResponse, RiskPredictionResponse
from backend.workspace_storage import WorkspaceRepository
from pulmoai.report_reader import process_lab_report

VERSION = "presentation-journey-v1"
ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()
ORIGIN = "generated_example"
VISIT_VALUES = [
    dict(total_cholesterol=262, hdl_cholesterol=34, triglycerides=235, systolic_bp=156, diastolic_bp=94, hemoglobin=10.4, serum_creatinine=0.7, hba1c=7.1, bmi=32.5, waist_cm=103),
    dict(total_cholesterol=236, hdl_cholesterol=40, triglycerides=205, systolic_bp=148, diastolic_bp=89, hemoglobin=11.5, serum_creatinine=0.9, hba1c=6.8, bmi=31.0, waist_cm=99),
    dict(total_cholesterol=210, hdl_cholesterol=46, triglycerides=172, systolic_bp=136, diastolic_bp=84, hemoglobin=12.3, serum_creatinine=1.2, hba1c=6.4, bmi=29.2, waist_cm=95),
    dict(total_cholesterol=184, hdl_cholesterol=52, triglycerides=142, systolic_bp=128, diastolic_bp=79, hemoglobin=13.1, serum_creatinine=1.4, hba1c=6.0, bmi=27.8, waist_cm=91),
]
LAB_FIELDS = [
    ("total_cholesterol", "Total Cholesterol", "mg/dL", "100-200"),
    ("hdl_cholesterol", "HDL Cholesterol", "mg/dL", "40-80"),
    ("triglycerides", "Triglycerides", "mg/dL", "40-150"),
    ("systolic_bp", "Systolic BP", "mmHg", "90-120"),
    ("diastolic_bp", "Diastolic BP", "mmHg", "60-80"),
    ("hemoglobin", "Hemoglobin", "g/dL", "12-16"),
    ("serum_creatinine", "Serum Creatinine", "mg/dL", "0.5-1.1"),
    ("hba1c", "HbA1c", "%", "4.0-5.6"),
]


def report_text(observed_on: str, values: dict, title: str) -> bytes:
    lines = ["GENERATED EXAMPLE DATA - NOT A PATIENT RECORD", f"Presentation case: {title}",
             f"Report date: {observed_on}", "Reported biological sex: Female", ""]
    lines += [f"{label}: {values[key]} {unit} (Reference: {reference})" for key, label, unit, reference in LAB_FIELDS]
    return ("\n".join(lines) + "\n").encode()


def illustration(observed_on: str, index: int) -> bytes:
    """A visibly illustrated source for archival/comparison, never a model input."""
    image = Image.new("RGB", (640, 380), "#edf4f0")
    draw = ImageDraw.Draw(image); font = ImageFont.load_default(size=20)
    draw.text((24, 20), "GENERATED ILLUSTRATION - NOT A MEDICAL SCAN", fill="#173a30", font=font)
    draw.rounded_rectangle((32, 72, 608, 286), radius=14, fill="#d1b8a1", outline="#7c8b7f", width=2)
    radius = [72, 62, 52, 42][index]
    draw.ellipse((320-radius, 179-radius, 320+radius, 179+radius), fill="#83674f", outline="#554638", width=3)
    draw.text((24, 310), f"Visit {index+1} | {observed_on} | Same illustrated site", fill="#173a30", font=font)
    draw.text((24, 342), "Archive only. No screening or severity measurement.", fill="#173a30", font=font)
    buffer = io.BytesIO(); image.save(buffer, format="PNG"); return buffer.getvalue()


def metadata(repo: WorkspaceRepository, user_id: str, patient_id: str) -> dict | None:
    with repo.store._connection() as connection:
        row = connection.execute("SELECT manifest_json FROM example_cases WHERE user_id=? AND patient_id=?", (user_id, patient_id)).fetchone()
    return json.loads(row[0]) if row else None


def _optional_services(request: Request) -> tuple[dict, list[str]]:
    from backend.main import require_service, require_risk_service, require_future_risk_service
    services, notes = {}, []
    for kind, loader, label in (("rna", require_service, "RNA classification"), ("risk", require_risk_service, "Chronic risk"),
                                ("future", require_future_risk_service, "Future risk")):
        try: services[kind] = loader(request)
        except HTTPException: notes.append(f"{label} unavailable: install compatible, verified model artifacts and dependencies. No substitute result was created.")
    return services, notes


def prepare(request: Request, repo: WorkspaceRepository, user: dict) -> dict:
    """Create once per owner, then reopen the same case without changing reviews/edits."""
    with _LOCK:
        with repo.store._connection() as connection:
            row = connection.execute("SELECT patient_id, manifest_json FROM example_cases WHERE user_id=? AND scenario_version=?", (user["id"], VERSION)).fetchone()
        if row:
            return {"patient": repo.patient(user["id"], row[0]), "example_case": json.loads(row[1]), "reused": True}
        services, notes = _optional_services(request)
        from backend.main import get_artifact_verification
        verification = get_artifact_verification(request.app)
        artifacts = verification.get("artifacts", {})
        dates = [(date.today()-timedelta(days=days)).isoformat() for days in (180, 120, 60, 14)]
        today = date.today().isoformat()
        alias = "Presentation journey - generated records"
        if any(p["alias"] == alias for p in repo.patients(user["id"])): alias += " " + uuid4().hex[:6]
        patient = repo.create_patient(user["id"], alias, "female")
        created = []

        def save(kind, filename, content_type, result, summary, observed_on, contents=None, reviewed=True, series=""):
            result = dict(result)
            result.update(data_origin=ORIGIN, example_case_version=VERSION, example_validation="generated_fixture_check" if reviewed else "pending_user_review", analysis_context={"patient_id":patient["id"], "observed_on":observed_on, "series_label":series})
            if contents is not None: result["source_sha256"] = hashlib.sha256(contents).hexdigest()
            record = repo.store.save_record(user["id"], kind, filename, content_type, result, summary, contents)
            created.append(record["id"])
            repo.attach(user["id"], record["id"], patient["id"], observed_on, series)
            if reviewed:
                # Generated sources are checked by the preparation routine, not attributed to clinical review.
                with repo.store._connection() as connection:
                    connection.execute("UPDATE uploads SET review_status='reviewed' WHERE id=? AND user_id=?", (record["id"], user["id"]))
                    repo._audit(connection, user["id"], "example.record_checked", record["id"])
            return record["id"]

        try:
            rna_contents = (ROOT / "samples/synthetic_lung_pseudobulk.csv").read_bytes() if "rna" in services else None
            # Repeated generated profile demonstrates date linkage, not a biological response.
            rna_output = PredictionResponse(**services["rna"].predict_upload(rna_contents, reported_sex="female")).model_dump(mode="json") if rna_contents else None
            for index, (observed_on, values) in enumerate(zip(dates, VISIT_VALUES)):
                content = report_text(observed_on, values, f"Visit {index+1}")
                parsed = ReportExtractResponse(**process_lab_report(content, "generated-report.txt", "text/plain", sex="female")).model_dump(mode="json")
                extracted = {item["canonical_name"]: item["value"] for item in parsed["extracted_tests"]}
                for key, *_ in LAB_FIELDS:
                    if key not in extracted or abs(extracted[key]-values[key]) > 0.001:
                        raise ValueError(f"Example report could not be verified for {key}; no ready case was saved.")
                parsed["reviewed_values"] = extracted; parsed["reviewed_tests"] = sorted(extracted)
                source_id = save("lab_report", f"generated-visit-{index+1}-{observed_on}.txt", "text/plain", parsed,
                                 f"Generated visit {index+1}: checked laboratory values", observed_on, content)
                inputs = normalize_risk_inputs({**values, "age":58, "sex_male":0, "race_black":0, "current_smoker":0,
                    "bp_medication":1, "target_diabetes":0, "activity_level":1, "family_history_diabetes":1, "prior_cvd":0})
                for key, method, schema, kind, model in (("risk", "predict_risk", RiskPredictionResponse, "risk_assessment", "models/nhanes_risk_models.joblib"),
                        ("future", "predict_future_risk", FutureRiskResponse, "future_risk_assessment", "models/future_risk_models.joblib")):
                    if key not in services: continue
                    output = schema(**getattr(services[key], method)(inputs)).model_dump(mode="json")
                    output.update(data_origin=ORIGIN, artifact_id=artifacts.get(model, {}).get("actual_sha256"), source_record_id=source_id)
                    save(kind, f"Generated visit {index+1} - {key} risk", "application/json",
                         {"inputs":inputs, "result":output, "artifact_id":output["artifact_id"], "source_record_id":source_id},
                         f"Calculated from generated visit {index+1} inputs", observed_on)
                if rna_output:
                    output = dict(rna_output)
                    output["example_profile_notice"] = "Repeated generated raw-count profile. This demonstrates visit linkage, not independent samples or biological change."
                    save("scrna_csv", f"generated-rna-visit-{index+1}.csv", "text/csv", output,
                         "Research classification of repeated generated profile", observed_on, rna_contents)
                image = illustration(observed_on, index)
                save("medical_image", f"generated-illustration-visit-{index+1}.png", "image/png",
                    {"image_type":"skin_lesion", "modality_name":"Illustrated comparison source", "status":"not_screened", "recognized":False,
                     "predicted_class":None, "message":"Generated illustration archived for comparison. No automated screening or severity measurement.",
                     "clinical_notice":"Not a medical scan. Archive only.", "disclaimer":"No AI score is supplied."},
                    "Generated illustration - archive comparison only", observed_on, image, series="Illustrated site A - fixed view")
            exercise = report_text(today, {**VISIT_VALUES[-1], "total_cholesterol":188}, "Live source-review exercise")
            parsed = ReportExtractResponse(**process_lab_report(exercise, "generated-review-exercise.txt", "text/plain", sex="female")).model_dump(mode="json")
            practice_id = save("lab_report", "generated-review-exercise.txt", "text/plain", parsed,
                               "Live exercise: confirm fields, correct if needed, transfer to risk", today, exercise, reviewed=False)
            import shutil
            notes += ["Image sources are generated illustrations for archive comparison; they have no AI scores or clinical findings.",
                      "The repeated RNA profile demonstrates history and sex-context routing, not biological improvement."]
            if not verification.get("module_status", {}).get("image_service"):
                notes.append("Live image screening is unavailable until the matching vision checkpoints are installed.")
            if not shutil.which("tesseract"):
                notes.append("Scanned-photo OCR needs local Tesseract. The supplied text reports work without it.")
            manifest = {"version":VERSION, "title":"Six-month presentation journey", "data_origin":ORIGIN,
                "prepared_on":today, "visit_dates":dates, "visit_labels":["Baseline", "Visit 2", "Visit 3", "Latest checked visit"],
                "practice_record_id":practice_id, "practice_date":today, "record_count":len(created), "notices":notes,
                "story":"Cholesterol and hemoglobin move toward their reference intervals while creatinine moves away. Risk estimates are calculated by the installed models, not prescribed to follow a story.",
                "validation_notice":"Core generated records were checked by the example-preparation routine. This is not clinical source review. One extra report remains unreviewed for a live walkthrough."}
            with repo.store._connection() as connection:
                connection.execute("INSERT INTO example_cases VALUES (?, ?, ?, ?)", (user["id"], VERSION, patient["id"], json.dumps(manifest)))
                repo._audit(connection, user["id"], "example.case_prepared", patient["id"])
            return {"patient":patient, "example_case":manifest, "reused":False}
        except Exception:
            for rid in created: repo.store.delete_record(user["id"], rid)
            with repo.store._connection() as connection:
                connection.execute("DELETE FROM patients WHERE id=? AND user_id=?", (patient["id"], user["id"]))
            raise


def source_pack(repo: WorkspaceRepository, user_id: str, patient_id: str) -> bytes:
    manifest = metadata(repo, user_id, patient_id)
    if not manifest: raise HTTPException(404, "Example source pack not found.")
    guide = (ROOT / "docs/PRESENTATION_WALKTHROUGH.md").read_text(encoding="utf-8")
    records = [record for record in repo.records(user_id, patient_id) if record["result"].get("example_case_version") == VERSION]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", guide)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("timeline.json", json.dumps({"example_case":manifest, "records":records}, indent=2))
        for record in records:
            if record["result"].get("example_case_version") != VERSION: continue
            original = repo.store.get_record_file(user_id, record["id"])
            if original:
                archive.writestr(f"sources/{record['observed_on']}/{record['filename']}", original[0].read_bytes())
    return buffer.getvalue()
