"""Shared patient/visit context and lineage for every analysis module."""
from __future__ import annotations

import hashlib
from typing import Any

from fastapi import HTTPException
from backend.schemas import AnalysisContext
from backend.storage import UserDataStore
from backend.workspace_storage import WorkspaceRepository


def validate_context(store: UserDataStore, user: dict | None, context: AnalysisContext | dict | None) -> dict | None:
    if context is None:
        return None
    if user is None:
        raise HTTPException(401, "Sign in before linking an analysis to a patient.")
    context = context.model_dump() if isinstance(context, AnalysisContext) else dict(context)
    patient_id, observed_on = context.get("patient_id"), context.get("observed_on")
    if not patient_id or not observed_on:
        raise HTTPException(422, "Select a patient and observation date for this analysis.")
    repo = WorkspaceRepository(store)
    if not repo.patient(user["id"], patient_id):
        raise HTTPException(404, "Patient not found.")
    try:
        repo.validate_date(observed_on)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    context["series_label"] = context.get("series_label", "").strip()
    if len(context["series_label"]) > 80:
        raise HTTPException(422, "Acquisition series must be no longer than 80 characters.")
    return context


def form_context(store: UserDataStore, user: dict | None, patient_id: str | None, observed_on: str | None, series_label: str = "") -> dict | None:
    if not patient_id and not observed_on:
        return None
    return validate_context(store, user, {"patient_id": patient_id, "observed_on": observed_on, "series_label": series_label})


def validate_source(store: UserDataStore, user: dict | None, source_record_id: str | None, context: dict | None) -> dict | None:
    if not source_record_id:
        return None
    if user is None:
        raise HTTPException(401, "Sign in to use a saved source report.")
    source = store.get_record(user["id"], source_record_id)
    if not source or source["kind"] != "lab_report":
        raise HTTPException(404, "Source lab report not found.")
    if context and source.get("patient_id") != context["patient_id"]:
        raise HTTPException(409, "The source report belongs to a different patient. Select its patient before transferring.")
    if source.get("review_status") not in {"reviewed", "partially_reviewed"}:
        raise HTTPException(422, "Confirm source lab values before transferring them into risk assessment.")
    from pulmoai.provenance import data_origin
    source["result"].setdefault("data_origin", data_origin(source["result"].get("raw_text", "").encode(), source["filename"]))
    return source


def save_analysis(store: UserDataStore, user: dict | None, response: Any, kind: str, filename: str,
                  content_type: str, summary: str, context: dict | None, contents: bytes | None = None,
                  inputs: dict | None = None, source_record_id: str | None = None) -> dict | None:
    if user is None:
        return None
    if context:
        response.patient_id, response.observed_on = context["patient_id"], context["observed_on"]
    response.source_record_id = source_record_id
    result = response.model_dump(mode="json", exclude={"upload_id", "uploaded_at"})
    if inputs is not None:
        result = {"inputs": inputs, "result": result, "data_origin": response.data_origin,
                  "artifact_id": response.artifact_id, "source_record_id": source_record_id}
    result["analysis_context"] = context
    if contents is not None:
        result["source_sha256"] = hashlib.sha256(contents).hexdigest()
    record = store.save_record(user["id"], kind, filename, content_type, result, summary, contents)
    if context:
        try:
            # Inference history may contain multiple runs of the same source. Batch imports still deduplicate.
            WorkspaceRepository(store).attach(user["id"], record["id"], context["patient_id"], context["observed_on"], context.get("series_label", ""), deduplicate=False)
        except Exception:
            store.delete_record(user["id"], record["id"])
            raise
        response.patient_id = context["patient_id"]
        response.observed_on = context["observed_on"]
    response.upload_id, response.uploaded_at = record["id"], record["uploaded_at"]
    response.source_record_id = source_record_id
    return record

def normalize_risk_inputs(inputs: dict) -> dict:
    """Translate visible form names into the feature names the persisted models use."""
    values = {key: value for key, value in inputs.items() if value is not None}
    aliases = {"creatinine": "serum_creatinine", "pulse": "pulse_bpm", "waist_circumference_cm": "waist_cm",
               "physical_activity": "activity_level", "history_diabetes": "target_diabetes"}
    for alias, canonical in aliases.items():
        if alias in values:
            if canonical not in values or values[canonical] is None:
                values[canonical] = values[alias]
            values.pop(alias)
    gender = values.pop("gender", None)
    if values.get("sex_male") is None and gender in (1, 2):
        values["sex_male"] = 1 if gender == 1 else 0
    if values.get("physically_active") is None and values.get("activity_level") in (0, 1, 2):
        values["physically_active"] = 0 if values["activity_level"] == 2 else 1
    if values.get("activity_level") is None and values.get("physically_active") in (0, 1):
        values["activity_level"] = 2 if values["physically_active"] == 0 else 1
    if values.get("current_smoker") is None and values.get("smoking_status") in (0, 1, 2):
        values["current_smoker"] = int(values["smoking_status"] == 2)
    return values
