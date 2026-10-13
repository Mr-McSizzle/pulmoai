from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pulmoai.service import MAX_UPLOAD_BYTES, PulmoAIService
from backend.schemas import (
    DashboardResponse,
    GeneInsightsResponse,
    HealthResponse,
    PredictionResponse,
    ValidationResponse,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pulmoai.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.pulmo_service = PulmoAIService.load()
        app.state.model_error = None
        logger.info("Loaded PulmoAI model artifact %s", app.state.pulmo_service.artifact_id)
    except Exception:
        app.state.pulmo_service = None
        app.state.model_error = "Model artifact could not be loaded; check the local server logs."
        logger.exception("PulmoAI model startup failed")
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


def require_service(request: Request) -> PulmoAIService:
    service = getattr(request.app.state, "pulmo_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail=getattr(
            request.app.state, "model_error", "Model service is unavailable."
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
    service = getattr(request.app.state, "pulmo_service", None)
    if service is None:
        return HealthResponse(
            status="unavailable",
            model_loaded=False,
            message=getattr(request.app.state, "model_error", "Model artifact is unavailable."),
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=service.model_name,
        artifact_id=service.artifact_id,
        message="Model loaded and ready for compatible research input.",
    )


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


FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
