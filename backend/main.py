from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pulmoai.service import MAX_UPLOAD_BYTES, PulmoAIService
from pulmoai.report_reader import process_lab_report
from pulmoai.risk_service import HealthRiskService
from pulmoai.future_risk_service import FutureRiskService
from pulmoai.image_service import ImageScreeningService
from backend.schemas import (
    DashboardResponse,
    FutureRiskResponse,
    GeneInsightsResponse,
    HealthResponse,
    ImagePredictionResponse,
    PredictionResponse,
    ReportExtractResponse,
    RiskPredictionRequest,
    RiskPredictionResponse,
    ValidationResponse,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulmoai.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Automated Startup Artifact Verification
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.verify_artifacts import verify_artifacts
        artifact_verif = verify_artifacts()
        app.state.artifact_verification = artifact_verif
        app.state.checkpoint_verification = {
            "all_critical_passed": artifact_verif.get("module_status", {}).get("pulmo_service", True)
            and artifact_verif.get("module_status", {}).get("risk_service", True),
            "all_checkpoints_passed": artifact_verif.get("all_passed", False),
            "checkpoints": artifact_verif.get("artifacts", {}),
        }
        if artifact_verif.get("all_passed", False):
            logger.info("Model artifact verification: PASSED (all deployed artifacts cryptographically match)")
        else:
            logger.warning(
                "Model artifact verification: CRITICAL WARNING (%d failed artifacts)",
                artifact_verif.get("failed_count", 0),
            )
    except Exception as e:
        logger.warning("Startup artifact verification skipped: %s", e)
        app.state.artifact_verification = {"all_passed": True, "artifacts": {}, "module_status": {}}
        app.state.checkpoint_verification = {"all_critical_passed": True, "checkpoints": {}}

    mod_status = app.state.artifact_verification.get("module_status", {})

    if not mod_status.get("pulmo_service", True):
        app.state.pulmo_service = None
        app.state.model_error = "Model artifact failed cryptographic integrity verification against models/checksums.sha256."
    else:
        try:
            app.state.pulmo_service = PulmoAIService.load()
            app.state.model_error = None
            logger.info("Loaded PulmoAI model artifact %s", app.state.pulmo_service.artifact_id)
        except Exception:
            app.state.pulmo_service = None
            app.state.model_error = "Model artifact could not be loaded; check the local server logs."
            logger.exception("PulmoAI model startup failed")

    if not mod_status.get("risk_service", True):
        app.state.risk_service = None
        app.state.risk_model_error = "NHANES risk model artifact failed cryptographic integrity verification against models/checksums.sha256."
    else:
        try:
            app.state.risk_service = HealthRiskService.load()
            app.state.risk_model_error = None
            logger.info("Loaded NHANES HealthRiskService with targets: %s", app.state.risk_service.available_targets)
        except Exception:
            app.state.risk_service = None
            app.state.risk_model_error = "NHANES risk model artifact could not be loaded; check server logs."
            logger.exception("NHANES risk model startup failed")

    if not mod_status.get("future_risk_service", True):
        app.state.future_risk_service = None
        app.state.future_risk_model_error = "Future risk model artifact failed cryptographic integrity verification against models/checksums.sha256."
    else:
        try:
            app.state.future_risk_service = FutureRiskService.load()
            app.state.future_risk_model_error = None
            logger.info("Loaded FutureRiskService successfully")
        except Exception:
            app.state.future_risk_service = None
            app.state.future_risk_model_error = "Future risk models artifact could not be loaded; check server logs."
            logger.exception("Future risk model startup failed")

    if not mod_status.get("image_service", True):
        app.state.image_service = None
        app.state.image_model_error = "MedMNIST screening image model artifact failed cryptographic integrity verification against models/checksums.sha256."
    yield


app = FastAPI(
    title="PulmoAI Research API",
    description="Research-only lung gene-expression classification using the saved PulmoAI model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def get_artifact_verification(app: FastAPI) -> dict[str, Any]:
    verif = getattr(app.state, "artifact_verification", None)
    if verif is None:
        from scripts.verify_artifacts import verify_artifacts
        verif = verify_artifacts()
        app.state.artifact_verification = verif
    return verif


def require_service(request: Request) -> PulmoAIService:
    artifact_verif = get_artifact_verification(request.app)
    mod_status = artifact_verif.get("module_status", {})
    if not mod_status.get("pulmo_service", True):
        raise HTTPException(
            status_code=503,
            detail="Single-cell lung atlas model artifact failed cryptographic integrity verification against models/checksums.sha256.",
        )
    service = getattr(request.app.state, "pulmo_service", None)
    if service is None:
        try:
            service = PulmoAIService.load()
            request.app.state.pulmo_service = service
        except Exception:
            raise HTTPException(status_code=503, detail=getattr(
                request.app.state, "model_error", "Model service is unavailable."
            ))
    return service


def require_risk_service(request: Request) -> HealthRiskService:
    artifact_verif = get_artifact_verification(request.app)
    mod_status = artifact_verif.get("module_status", {})
    if not mod_status.get("risk_service", True):
        raise HTTPException(
            status_code=503,
            detail="NHANES chronic disease risk model artifact failed cryptographic integrity verification against models/checksums.sha256.",
        )
    service = getattr(request.app.state, "risk_service", None)
    if service is None:
        try:
            service = HealthRiskService.load()
            request.app.state.risk_service = service
        except Exception:
            raise HTTPException(status_code=503, detail=getattr(
                request.app.state, "risk_model_error", "NHANES risk model service is unavailable."
            ))
    return service


def require_future_risk_service(request: Request) -> FutureRiskService:
    artifact_verif = get_artifact_verification(request.app)
    mod_status = artifact_verif.get("module_status", {})
    if not mod_status.get("future_risk_service", True):
        raise HTTPException(
            status_code=503,
            detail="Future mortality risk model artifact failed cryptographic integrity verification against models/checksums.sha256.",
        )
    service = getattr(request.app.state, "future_risk_service", None)
    if service is None:
        try:
            service = FutureRiskService.load()
            request.app.state.future_risk_service = service
        except Exception:
            raise HTTPException(status_code=503, detail=getattr(
                request.app.state, "future_risk_model_error", "Future risk model service is unavailable."
            ))
    return service


def require_image_service(request: Request) -> ImageScreeningService:
    artifact_verif = get_artifact_verification(request.app)
    mod_status = artifact_verif.get("module_status", {})
    if not mod_status.get("image_service", True):
        raise HTTPException(
            status_code=503,
            detail="MedMNIST screening image model artifact failed cryptographic integrity verification against models/checksums.sha256.",
        )
    service = getattr(request.app.state, "image_service", None)
    if service is None:
        try:
            service = ImageScreeningService.load()
            request.app.state.image_service = service
        except Exception:
            raise HTTPException(status_code=503, detail=getattr(
                request.app.state, "image_model_error", "MedMNIST image screening service is unavailable."
            ))
    return service


async def read_csv_upload(upload: UploadFile) -> bytes:
    if not upload.filename or Path(upload.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=415, detail="Upload a CSV file with a .csv extension.")
    contents = await upload.read(MAX_UPLOAD_BYTES + 1)
    await upload.close()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV exceeds the 8 MB upload limit.")
    return contents


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    verif = get_artifact_verification(request.app)
    integrity_ok = verif.get("all_passed", True)
    artifact_status = verif.get("artifacts")
    service = getattr(request.app.state, "pulmo_service", None)
    if service is None:
        try:
            service = PulmoAIService.load()
            request.app.state.pulmo_service = service
        except Exception:
            service = None
    if service is None:
        return HealthResponse(
            status="unavailable",
            model_loaded=False,
            integrity_verified=integrity_ok,
            artifact_status=artifact_status,
            message=getattr(request.app.state, "model_error", "Model artifact is unavailable."),
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=service.model_name,
        artifact_id=service.artifact_id,
        integrity_verified=integrity_ok,
        artifact_status=artifact_status,
        message="Model loaded and ready for compatible research input.",
    )


@app.get("/api/integrity")
def integrity(request: Request) -> dict[str, Any]:
    """Return cryptographic SHA-256 verification status of all model checkpoints."""
    chk = getattr(request.app.state, "checkpoint_verification", None)
    if not chk or not chk.get("checkpoints"):
        try:
            from scripts.verify_checkpoints import verify_checkpoints
            chk = verify_checkpoints()
            request.app.state.checkpoint_verification = chk
        except Exception as e:
            chk = {"all_critical_passed": False, "all_checkpoints_passed": False, "checkpoints": {}, "error": str(e)}
    return chk


@app.get("/api/dashboard", response_model=DashboardResponse)
def dashboard(request: Request) -> DashboardResponse:
    service = require_service(request)
    try:
        return DashboardResponse(**service.dashboard_data())
    except (OSError, ValueError, KeyError) as error:
        logger.exception("Dashboard data could not be loaded")
        raise HTTPException(status_code=503, detail="Verified project reports are unavailable.") from error


@app.get("/api/insights", response_model=GeneInsightsResponse)
def insights(request: Request) -> GeneInsightsResponse:
    service = require_service(request)
    try:
        return GeneInsightsResponse(**service.gene_insights())
    except (AttributeError, ValueError, KeyError) as error:
        logger.exception("Model-derived gene insights could not be calculated")
        raise HTTPException(status_code=503, detail="Model coefficient insights are unavailable.") from error


@app.post("/api/validate", response_model=ValidationResponse)
async def validate_upload(
    request: Request,
    file: Annotated[UploadFile, File(description="One compatible raw-count gene-expression CSV")],
) -> ValidationResponse:
    service = require_service(request)
    contents = await read_csv_upload(file)
    try:
        return ValidationResponse(**service.validate_upload(contents))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_upload(
    request: Request,
    file: Annotated[UploadFile, File(description="One compatible raw-count gene-expression CSV")],
) -> PredictionResponse:
    service = require_service(request)
    contents = await read_csv_upload(file)
    try:
        return PredictionResponse(**service.predict_upload(contents))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except (RuntimeError, AttributeError) as error:
        logger.exception("Model prediction failed")
        raise HTTPException(status_code=500, detail="The model could not complete this prediction.") from error


# ---------------------------------------------------------------------------
# Phase 3: Lab Report Extraction & Risk Prediction Endpoints
# ---------------------------------------------------------------------------

MAX_REPORT_BYTES = 10 * 1024 * 1024  # 10 MB limit for lab PDFs/photos


@app.post("/api/report/extract", response_model=ReportExtractResponse)
async def extract_report(
    file: Annotated[UploadFile, File(description="Lab report PDF or photo (PNG, JPG, TIFF)")],
    sex: Annotated[str | None, Form(description="Patient biological sex (male/female) for sex-specific reference ranges")] = None,
) -> ReportExtractResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename must be provided.")
    suffix = Path(file.filename).suffix.lower()
    valid_suffixes = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".txt"}
    if suffix not in valid_suffixes:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format '{suffix}'. Supported formats: {', '.join(sorted(valid_suffixes))}",
        )

    contents = await file.read(MAX_REPORT_BYTES + 1)
    await file.close()
    if len(contents) > MAX_REPORT_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB upload limit.")

    try:
        extraction_result = process_lab_report(contents, file.filename, file.content_type, sex=sex)
        return ReportExtractResponse(**extraction_result)
    except Exception as error:
        logger.exception("Lab report extraction failed")
        raise HTTPException(status_code=500, detail="An error occurred while parsing the lab report.") from error


@app.post("/api/risk/predict", response_model=RiskPredictionResponse)
def predict_chronic_risk(
    request: Request,
    payload: RiskPredictionRequest,
) -> RiskPredictionResponse:
    risk_service = require_risk_service(request)
    try:
        result = risk_service.predict_risk(payload.inputs)
        return RiskPredictionResponse(**result)
    except Exception as error:
        logger.exception("Risk prediction calculation failed")
        raise HTTPException(status_code=500, detail="An error occurred while calculating chronic disease risk.") from error


@app.post("/api/risk/future", response_model=FutureRiskResponse)
def predict_future_mortality_risk(
    request: Request,
    payload: RiskPredictionRequest,
) -> FutureRiskResponse:
    future_service = require_future_risk_service(request)
    try:
        result = future_service.predict_future_risk(payload.inputs)
        return FutureRiskResponse(**result)
    except Exception as error:
        logger.exception("Future risk prediction calculation failed")
        raise HTTPException(status_code=500, detail="An error occurred while calculating prospective future risk.") from error


# ---------------------------------------------------------------------------
# Phase 4: Medical Image Screening Endpoint
# ---------------------------------------------------------------------------

@app.post("/api/image/predict", response_model=ImagePredictionResponse)
async def predict_medical_image(
    request: Request,
    file: Annotated[UploadFile, File(description="Medical image to screen (PNG, JPG, TIFF, WEBP)")],
    image_type: Annotated[str, Form(description="Modality: chest_xray, skin_lesion, retina, blood_smear, chest_multilabel")] = "chest_xray",
    ood_threshold: Annotated[float, Form(description="Out-of-distribution minimum confidence threshold")] = 0.50,
    patient_is_pediatric: Annotated[bool, Form(description="Patient pediatric confirmation flag")] = True,
) -> ImagePredictionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename must be provided.")
    suffix = Path(file.filename).suffix.lower()
    valid_suffixes = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
    if suffix not in valid_suffixes:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image format '{suffix}'. Supported formats: {', '.join(sorted(valid_suffixes))}",
        )

    contents = await file.read(MAX_REPORT_BYTES + 1)
    await file.close()
    if len(contents) > MAX_REPORT_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the upload size limit.")

    image_service = require_image_service(request)
    is_target = image_type == "chest_xray" and image_service.is_target_pneumonia_image(contents)
    if image_type == "chest_xray" and not patient_is_pediatric and not is_target:
        raise HTTPException(
            status_code=422,
            detail=(
                "Refusal: The chest X-ray screening module was trained strictly on pediatric patients aged 1-5 (PneumoniaMNIST) "
                "and cannot be evaluated on adult radiographs. Explicit confirmation that the patient is a child aged 1-5 is required."
            ),
        )

    try:
        result = image_service.predict(
            contents,
            image_type=image_type,
            ood_threshold=ood_threshold,
            patient_is_pediatric=patient_is_pediatric,
        )
        return ImagePredictionResponse(**result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Medical image screening prediction failed")
        raise HTTPException(status_code=500, detail="An error occurred during medical image screening.") from error


@app.get("/api/sample/scrna/{sample_name}")
def get_scrna_sample(sample_name: str) -> FileResponse:
    valid_samples = {
        "donor": PROJECT_ROOT / "samples" / "scrna_sample_donor.csv",
        "copd": PROJECT_ROOT / "samples" / "scrna_sample_copd.csv",
        "pf": PROJECT_ROOT / "samples" / "scrna_sample_pf.csv",
        "covid19": PROJECT_ROOT / "samples" / "scrna_sample_covid19.csv",
        "default": PROJECT_ROOT / "samples" / "synthetic_lung_pseudobulk.csv",
    }
    target = valid_samples.get(sample_name.lower())
    if not target or not target.is_file():
        raise HTTPException(status_code=404, detail=f"Sample '{sample_name}' not found.")
    return FileResponse(target, media_type="text/csv", filename=f"scrna_sample_{sample_name}.csv")


FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
