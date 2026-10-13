from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_name: str | None = None
    artifact_id: str | None = None
    message: str


class ValidationResponse(BaseModel):
    valid: bool
    profile_count: int
    recognized_gene_count: int
    required_gene_count: int
    normalization: str
    message: str


class PredictionResponse(BaseModel):
    predicted_class: str
    model_name: str
    artifact_id: str
    input_profile_count: int
    input_gene_count: int
    probabilities_available: bool
    class_probabilities: dict[str, float] | None = None
    interpretation: str


class DashboardResponse(BaseModel):
    verification_status: str
    cells: int
    studies: int
    profiles: int
    features: int
    classes: list[str]
    class_distribution: dict[str, int]
    model_name: str
    artifact_id: str
    metrics: dict[str, float] | None = None
    per_class_metrics: dict[str, dict[str, Any]] | None = None
    confusion_matrix: list[list[int]] | None = None
    candidate_metrics: list[dict[str, Any]] = Field(default_factory=list)
    methodology: dict[str, str]
    limitations: list[str]


class GeneInsightsResponse(BaseModel):
    available: bool
    model_name: str | None = None
    method: str | None = None
    interpretation: str | None = None
    message: str | None = None
    features: list[dict[str, Any]] = Field(default_factory=list)
