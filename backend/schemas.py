from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_name: str | None = None
    artifact_id: str | None = None
    integrity_verified: bool = True
    artifact_status: dict[str, Any] | None = None
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
    sex_context: dict[str, Any] | None = None
    input_profile_count: int
    input_gene_count: int
    probabilities_available: bool
    class_probabilities: dict[str, float] | None = None
    interpretation: str
    decision_rule_note: str | None = None
    input_contract_notice: str | None = None
    upload_id: str | None = None
    uploaded_at: str | None = None


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


# ---------------------------------------------------------------------------
# Phase 3: Report Reader & Risk Prediction Schemas
# ---------------------------------------------------------------------------

class LabItemSchema(BaseModel):
    raw_name: str
    canonical_name: str
    display_name: str
    category: str
    value: float
    raw_value: str
    unit: str
    raw_unit: str
    reference_range: str
    flag: Literal["NORMAL", "HIGH", "LOW", "UNKNOWN"]
    confidence: float


class ReportExtractResponse(BaseModel):
    data_origin: Literal["user_supplied_unverified", "generated_example"] = "user_supplied_unverified"
    quality_tier: str | None = None
    extraction_method: str
    extracted_tests: list[LabItemSchema]
    raw_text: str
    unmatched_lines: list[str] = Field(default_factory=list)
    count: int
    upload_id: str | None = None
    uploaded_at: str | None = None


class RiskPredictionRequest(BaseModel):
    inputs: dict[str, Annotated[float, Field(allow_inf_nan=False)] | int | None] = Field(max_length=100)


class ConditionRiskResult(BaseModel):
    target_id: str
    display_name: str
    description: str
    estimand_type: Literal["current_prevalent_status", "survey_reported_history"] = "survey_reported_history"
    estimand_label: str
    estimand_description: str
    time_horizon: str
    calibrated_risk: float | None = None
    risk_category: str
    model_auroc: float | None = None
    model_brier_score: float | None = None
    test_prevalence: float | None = None


class RiskPredictionResponse(BaseModel):
    risk_predictions: dict[str, ConditionRiskResult]
    contributing_factors: list[str] = Field(default_factory=list)
    features_provided_count: int
    estimand_clarification: str
    disclaimer: str


# ---------------------------------------------------------------------------
# Phase 4: Medical Image Screening Schemas
# ---------------------------------------------------------------------------

class ImagePredictionResponse(BaseModel):
    data_origin: Literal["user_supplied_unverified", "generated_example"] = "user_supplied_unverified"
    artifact_id: str | None = None
    image_type: str
    modality_name: str
    recognized: bool
    status: Literal[
        "not_screened",
        "success",
        "low_confidence",
        "rejected_modality_mismatch",
        "rejected_non_medical_input",
        "rejected_pediatric_confirmation_required",
        "error",
    ]
    message: str | None = None
    predicted_class: str | None = None
    confidence: float | None = None
    max_confidence: float | None = None
    accuracy: float | None = None
    threshold_applied: float | None = None
    empirical_false_accept_rate: float | None = None
    cifar10_false_accept_rate: float | None = None
    guard_status: str | None = None
    ood_benchmark: dict[str, Any] | None = None
    ood_methodology_statement: str | None = None
    modality_gate: dict[str, Any] | None = None
    warning_banner: str | None = None
    class_probabilities: dict[str, float] | None = None
    multilabel: bool = False
    active_conditions: list[str] | None = None
    probabilities: dict[str, float] | None = None
    model_metrics: dict[str, float] | None = None
    staging_notice: str | None = None
    clinical_notice: str = "Benchmark model, not for clinical use."
    disclaimer: str
    upload_id: str | None = None
    uploaded_at: str | None = None


class RegisterRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class UserResponse(BaseModel):
    id: str
    full_name: str
    email: str
    created_at: str


class AuthResponse(BaseModel):
    authenticated: bool
    user: UserResponse | None = None


class StoredRecordResponse(BaseModel):
    id: str
    kind: str
    filename: str
    content_type: str
    size_bytes: int
    summary: str
    uploaded_at: str
    has_file: bool


class StoredRecordDetailResponse(StoredRecordResponse):
    result: dict[str, Any]


class IntegrityResponse(BaseModel):
    all_critical_passed: bool
    all_checkpoints_passed: bool
    verified_at: str
    checkpoints: dict[str, dict[str, Any]]


# ---------------------------------------------------------------------------
# Phase 5: Future Risk (Mortality-Linked) Schemas
# ---------------------------------------------------------------------------

class FutureRiskTimeHorizon(BaseModel):
    risk_probability: float
    risk_percent: float


class FutureRiskEndpoint(BaseModel):
    display_name: str
    estimand_type: str = "prospective_incidence"
    estimand_label: str
    estimand_description: str
    time_horizons: dict[str, FutureRiskTimeHorizon]
    risk_tier_5yr: str
    tier_cutoffs_5yr: dict[str, float]
    benchmark_metrics: dict[str, float]


class FutureRiskResponse(BaseModel):
    future_endpoints: dict[str, FutureRiskEndpoint]
    published_scores: dict[str, Any]
    features_provided_count: int
    nchs_privacy_perturbation_warning: str | None = None
    estimand_clarification: str
    disclaimer: str




