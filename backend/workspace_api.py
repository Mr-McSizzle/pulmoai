"""Authenticated patient workspace and bounded batch ingestion."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from backend.main import get_user_data_store, required_current_user, require_image_service, get_artifact_verification
from backend.schemas import ImagePredictionResponse, ReportExtractResponse
from backend.workspace_storage import WorkspaceRepository
from pulmoai.longitudinal import build_timeline
from pulmoai.report_reader import process_lab_report

router = APIRouter(prefix="/api", tags=["Patient workspace"])
logger = logging.getLogger("pulmoai.workspace")
User = Annotated[dict[str, str], Depends(required_current_user)]
MAX_FILES = 12
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_BATCH_BYTES = 60 * 1024 * 1024


class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alias: str = Field(min_length=1, max_length=80)
    reported_sex: Literal["female", "male", "intersex", "unknown"] = "unknown"


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observed_on: str
    series_label: str = Field(default="", max_length=80)
    data_origin: Literal["user_supplied_unverified", "generated_example"] = "user_supplied_unverified"


class RecordContext(Observation):
    patient_id: str


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmed: Literal[True]
    corrections: dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(default_factory=dict)


def repository(request: Request) -> WorkspaceRepository:
    return WorkspaceRepository(get_user_data_store(request))


def patient_or_404(repo: WorkspaceRepository, user: dict, patient_id: str) -> dict:
    patient = repo.patient(user["id"], patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found.")
    return patient


@router.get("/patients")
def patients(request: Request, user: User):
    return repository(request).patients(user["id"])


@router.post("/patients", status_code=201)
def create_patient(payload: PatientInput, request: Request, user: User):
    try:
        return repository(request).create_patient(user["id"], payload.alias, payload.reported_sex)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.get("/patients/{patient_id}/timeline")
def timeline(patient_id: str, request: Request, user: User):
    repo = repository(request)
    patient = patient_or_404(repo, user, patient_id)
    return build_timeline(patient, repo.records(user["id"], patient_id))


@router.get("/patients/{patient_id}/export")
def export_timeline(patient_id: str, request: Request, user: User):
    data = timeline(patient_id, request, user)
    return JSONResponse(data, headers={"Content-Disposition": f'attachment; filename="pulmoai-{patient_id}.json"'})


@router.patch("/uploads/{record_id}/context", status_code=204)
def link_record(record_id: str, payload: RecordContext, request: Request, user: User):
    try:
        repository(request).attach(user["id"], record_id, payload.patient_id, payload.observed_on, payload.series_label)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return Response(status_code=204)


@router.post("/uploads/{record_id}/review", status_code=204)
def review_record(record_id: str, payload: ReviewInput, request: Request, user: User):
    try:
        repository(request).review(user["id"], record_id, payload.corrections)
    except LookupError as error:
        raise HTTPException(404, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return Response(status_code=204)


@router.get("/workspace/audit")
def audit(request: Request, user: User):
    return repository(request).audit(user["id"])


@router.post("/patients/{patient_id}/batch")
async def ingest_batch(
    patient_id: str, request: Request, user: User,
    files: Annotated[list[UploadFile], File()],
    observations: Annotated[str, Form(description="JSON array in file order: observed_on and series_label")],
    kind: Annotated[Literal["lab_report", "medical_image"], Form()],
    image_type: Annotated[Literal["chest_xray", "skin_lesion", "retina", "blood_smear", "chest_multilabel"], Form()] = "chest_xray",
    patient_is_pediatric: Annotated[bool, Form()] = False,
    analysis_mode: Annotated[Literal["screen", "archive_only"], Form()] = "screen",
):
    repo = repository(request)
    patient = patient_or_404(repo, user, patient_id)
    try:
        if not 1 <= len(files) <= MAX_FILES:
            raise HTTPException(422, f"Upload between 1 and {MAX_FILES} files per batch.")
        try:
            raw = json.loads(observations)
            if not isinstance(raw, list) or len(raw) != len(files):
                raise ValueError("One observation is required for each file, in the same order.")
            metadata = [Observation.model_validate(item) for item in raw]
            for item in metadata:
                repo.validate_date(item.observed_on)
                if kind == "medical_image" and not item.series_label.strip():
                    raise ValueError("An acquisition series (same site/view) is required for image timelines.")
        except (ValueError, ValidationError, TypeError) as error:
            raise HTTPException(422, "Invalid observation metadata: " + str(error)) from error
        if kind == "medical_image" and analysis_mode == "screen" and image_type == "chest_xray" and not patient_is_pediatric:
            raise HTTPException(422, "Explicit confirmation of pediatric age 1-5 is required for this model.")
        payloads, total = [], 0
        for upload in files:
            data = await upload.read(MAX_FILE_BYTES + 1)
            total += len(data)
            if len(data) > MAX_FILE_BYTES or total > MAX_BATCH_BYTES:
                raise HTTPException(413, "Limit: 10 MB per file and 60 MB per batch.")
            payloads.append(data)
        service = require_image_service(request) if kind == "medical_image" and analysis_mode == "screen" else None
        artifact_id = image_artifact_id(request, image_type) if service else None
        outcomes = []
        for upload, content, observation in zip(files, payloads, metadata):
            filename = repo.store._safe_display_filename(upload.filename or "uploaded-file")
            digest = hashlib.sha256(content).hexdigest()
            duplicate_id = repo.duplicate(user["id"], patient_id, kind, observation.observed_on, digest)
            if duplicate_id:
                outcomes.append({"filename": filename, "status": "duplicate", "record_id": duplicate_id})
                continue
            record = None
            try:
                validate_file(filename, content, kind)
                if kind == "lab_report":
                    extracted = await run_in_threadpool(process_lab_report, content, filename, upload.content_type,
                        sex=patient["reported_sex"] if patient["reported_sex"] in {"male", "female"} else None)
                    result = ReportExtractResponse(**extracted).model_dump(mode="json", exclude={"upload_id", "uploaded_at"})
                    summary = f"Extracted {result['count']} lab values"
                elif analysis_mode == "archive_only":
                    from pulmoai.image_service import MODALITY_MAP
                    result = ImagePredictionResponse(image_type=image_type,
                        modality_name=MODALITY_MAP[image_type]["display_name"], recognized=False,
                        status="not_screened", message="Original image stored for visual comparison. No automated screening was performed.",
                        clinical_notice="Source archive only; no model output or clinical interpretation.",
                        disclaimer="No automated screening was performed.").model_dump(mode="json", exclude={"upload_id", "uploaded_at"})
                    summary = "Original image · no automated screening"
                else:
                    predicted = await run_in_threadpool(service.predict, content, image_type=image_type,
                        ood_threshold=0.5, patient_is_pediatric=patient_is_pediatric)
                    predicted["artifact_id"] = artifact_id
                    result = ImagePredictionResponse(**predicted).model_dump(mode="json", exclude={"upload_id", "uploaded_at"})
                    summary = f"{result['modality_name']}: {result.get('predicted_class') or result['status']}"
                from pulmoai.provenance import data_origin
                result["data_origin"] = data_origin(content, filename, observation.data_origin)
                record = repo.store.save_record(user["id"], kind, filename, upload.content_type or "application/octet-stream", result, summary, content)
                repo.attach(user["id"], record["id"], patient_id, observation.observed_on, observation.series_label)
                outcomes.append({"filename": filename, "status": "saved", "record_id": record["id"], "summary": summary})
            except Exception as error:
                if record:
                    repo.store.delete_record(user["id"], record["id"])
                duplicate_id = repo.duplicate(user["id"], patient_id, kind, observation.observed_on, digest)
                if duplicate_id:
                    outcomes.append({"filename": filename, "status": "duplicate", "record_id": duplicate_id})
                else:
                    logger.warning("Batch file failed (%s)", type(error).__name__)
                    message = str(error) if isinstance(error, ValueError) else "Analysis failed. Check file quality and local model availability."
                    outcomes.append({"filename": filename, "status": "error", "message": message})
        return {"patient_id": patient_id, "outcomes": outcomes}
    finally:
        for upload in files:
            await upload.close()


def validate_file(filename: str, content: bytes, kind: str) -> None:
    from pathlib import Path
    suffix = Path(filename).suffix.lower()
    images = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    allowed = images | {".pdf", ".txt"} if kind == "lab_report" else images
    if suffix not in allowed or not content:
        raise ValueError("Use a non-empty file in a supported format.")
    if suffix == ".pdf":
        if not content.lstrip().startswith(b"%PDF-"):
            raise ValueError("This file is not a PDF document.")
        import pymupdf
        try:
            with pymupdf.open(stream=content, filetype="pdf") as document:
                if document.needs_pass or not 1 <= len(document) <= 25:
                    raise ValueError("Use an unencrypted PDF with 1–25 pages.")
                if any(page.rect.width * page.rect.height > 2_000_000 for page in document):
                    raise ValueError("PDF page dimensions exceed the safe rendering limit.")
        except (RuntimeError, pymupdf.FileDataError) as error:
            raise ValueError("This PDF cannot be decoded safely.") from error
    if suffix in images:
        import io
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.width * image.height > 25_000_000:
                    raise ValueError("Image exceeds the 25 megapixel limit.")
                image.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError("The image cannot be decoded safely.") from error


def image_artifact_id(request: Request, image_type: str) -> str | None:
    from pulmoai.image_service import MODALITY_MAP
    artifacts = get_artifact_verification(request.app).get("artifacts", {})
    model_path = "models/medmnist/" + MODALITY_MAP[image_type]["model_file"]
    model = artifacts.get(model_path, {}).get("actual_sha256")
    gate = artifacts.get("models/modality_gate_resnet18_v3.pt", {}).get("actual_sha256")
    if not model or not gate:
        return None
    return hashlib.sha256((model + gate + ":screen-v2:threshold-0.5").encode()).hexdigest()[:16]


@router.get("/ready")
def readiness(request: Request):
    verification = get_artifact_verification(request.app)
    modules = verification.get("module_status", {})
    # Verify numerical services actually loaded, in addition to checking their bytes.
    modules = dict(modules)
    for key in ("pulmo_service", "risk_service", "future_risk_service"):
        modules[key] = bool(modules.get(key)) and getattr(request.app.state, key, None) is not None
    ready = bool(modules) and all(modules.values())
    return JSONResponse({"ready": ready, "modules": modules, "research_only": True}, status_code=200 if ready else 503)


@router.get("/evidence")
def evidence(request: Request):
    from pulmoai.evidence import benchmark_evidence
    return benchmark_evidence(get_artifact_verification(request.app).get("module_status", {}))


@router.get("/evidence/reports/{report_id}")
def evidence_report(report_id: str):
    from fastapi.responses import FileResponse
    from pulmoai.evidence import REPORTS, ROOT
    name = REPORTS.get(report_id)
    if not name or not (ROOT / "reports" / name).is_file():
        raise HTTPException(404, "Evidence report not found.")
    return FileResponse(ROOT / "reports" / name, media_type="application/json", filename=name)

@router.get("/uploads/{record_id}/preview")
def image_preview(record_id: str, request: Request, user: User):
    """Bounded first-frame PNG display preview; the original download stays unchanged."""
    import io
    from PIL import Image, UnidentifiedImageError
    store = get_user_data_store(request)
    record = store.get_record(user["id"], record_id)
    stored = store.get_record_file(user["id"], record_id)
    if not record or record["kind"] != "medical_image" or not stored:
        raise HTTPException(404, "Image source not found.")
    try:
        with Image.open(stored[0]) as original:
            if original.width * original.height > 25_000_000:
                raise ValueError("Preview dimensions exceed the limit.")
            preview = original.convert("RGB")
            preview.thumbnail((1600, 1600))
            output = io.BytesIO()
            preview.save(output, format="PNG")
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise HTTPException(415, "A display preview is unavailable. Download the original source.") from error
    return Response(output.getvalue(), media_type="image/png")
