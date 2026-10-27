export interface HealthResponse {
  status: "ok" | "unavailable";
  model_loaded: boolean;
  model_name: string | null;
  artifact_id: string | null;
  message: string;
}

export interface AuthUser {
  id: string;
  full_name: string;
  email: string;
  created_at: string;
}

export interface AuthResponse {
  authenticated: boolean;
  user: AuthUser | null;
}

export interface StoredRecord {
  patient_id?: string | null;
  observed_on?: string | null;
  review_status?: string;
  id: string;
  kind: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  summary: string;
  uploaded_at: string;
  has_file: boolean;
}

export interface StoredRecordDetail extends StoredRecord {
  result: Record<string, unknown>;
}

export interface Metrics {
  accuracy: number;
  balanced_accuracy: number;
  macro_f1: number;
  weighted_f1?: number;
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface CandidateMetrics extends Metrics {
  model: string;
  status: string;
  mean_inner_macro_f1?: number;
}

export interface DashboardData {
  verification_status: string;
  cells: number;
  studies: number;
  profiles: number;
  features: number;
  classes: string[];
  class_distribution: Record<string, number>;
  model_name: string;
  artifact_id: string;
  metrics: Metrics | null;
  per_class_metrics: Record<string, ClassMetrics> | null;
  confusion_matrix: number[][] | null;
  candidate_metrics: CandidateMetrics[];
  methodology: Record<string, string>;
  limitations: string[];
}

export interface ValidationResponse {
  valid: boolean;
  profile_count: number;
  recognized_gene_count: number;
  required_gene_count: number;
  normalization: string;
  message: string;
}

export interface PredictionResponse {
  patient_id?: string | null; observed_on?: string | null;
  predicted_class: string;
  model_name: string;
  artifact_id: string;
  input_profile_count: number;
  input_gene_count: number;
  probabilities_available: boolean;
  class_probabilities?: Record<string, number> | null;
  sex_context?: { reported_sex: ReportedSex; model_scope: string; sex_specific_available: boolean; notice: string };
  interpretation: string;
  decision_rule_note?: string | null;
  input_contract_notice?: string | null;
  upload_id?: string | null;
  uploaded_at?: string | null;
}

export interface GeneCoefficient {
  gene: string;
  class_coefficients: Record<string, number>;
  max_absolute_coefficient: number;
}

export interface GeneInsights {
  available: boolean;
  model_name: string | null;
  method: string | null;
  interpretation: string | null;
  message: string | null;
  features: GeneCoefficient[];
}

// ---------------------------------------------------------------------------
// Phase 3 & 5: Lab Report Reader Schemas
// ---------------------------------------------------------------------------

export interface LabItemSchema {
  raw_name: string;
  canonical_name: string;
  display_name: string;
  category: string;
  value: number;
  raw_value: string;
  unit: string;
  raw_unit: string;
  reference_range: string;
  flag: "NORMAL" | "HIGH" | "LOW" | "UNKNOWN";
  confidence: number;
}

export interface ReportExtractResponse {
  patient_id?: string | null; observed_on?: string | null;
  data_origin: DataOrigin;
  extraction_method: string;
  extracted_tests: LabItemSchema[];
  raw_text: string;
  unmatched_lines: string[];
  count: number;
  upload_id?: string | null;
  uploaded_at?: string | null;
}

// ---------------------------------------------------------------------------
// Phase 2 & 5: Chronic Disease Risk Engine Schemas
// ---------------------------------------------------------------------------

export interface ConditionRiskResult {
  target_id: string;
  display_name: string;
  description: string;
  estimand_type?: "current_prevalent_status" | "survey_reported_history";
  estimand_label?: string;
  estimand_description?: string;
  time_horizon?: string;
  calibrated_risk: number | null;
  risk_category: string;
  model_auroc?: number | null;
  model_brier_score?: number | null;
  test_prevalence?: number | null;
}

export interface RiskPredictionResponse {
  upload_id?: string; uploaded_at?: string; patient_id?: string | null; observed_on?: string | null; source_record_id?: string | null; data_origin?: DataOrigin;
  risk_predictions: Record<string, ConditionRiskResult>;
  contributing_factors: string[];
  features_provided_count: number;
  estimand_clarification?: string;
  disclaimer: string;
}

// ---------------------------------------------------------------------------
// Phase 4 & 5: Medical Image Screening Schemas
// ---------------------------------------------------------------------------

export interface ImagePredictionResponse {
  patient_id?: string | null; observed_on?: string | null;
  data_origin: DataOrigin;
  image_type: string;
  modality_name: string;
  recognized: boolean;
  status: "not_screened" | "success" | "low_confidence" | "error" | "rejected_modality_mismatch" | "rejected_non_medical_input" | "rejected_pediatric_confirmation_required";
  predicted_class?: string | null;
  confidence?: number;
  max_confidence?: number;
  accuracy?: number;
  threshold_applied?: number;
  empirical_false_accept_rate?: number | null;
  ood_benchmark?: any;
  ood_methodology_statement?: string | null;
  message?: string;
  multilabel?: boolean;
  active_conditions?: string[];
  probabilities?: Record<string, number>;
  class_probabilities?: Record<string, number>;
  model_metrics?: Record<string, number>;
  cifar10_false_accept_rate?: number | null;
  guard_status?: string | null;
  staging_notice?: string | null;
  clinical_notice?: string;
  disclaimer: string;
  upload_id?: string | null;
  uploaded_at?: string | null;
}

// ---------------------------------------------------------------------------
// Phase 6: Future Risk (Mortality-Linked) & Published Scores Schemas
// ---------------------------------------------------------------------------

export interface FutureRiskTimeHorizon {
  risk_probability: number;
  risk_percent: number;
}

export interface FutureRiskEndpoint {
  display_name: string;
  estimand_type: string;
  estimand_label: string;
  estimand_description: string;
  time_horizons: Record<string, FutureRiskTimeHorizon>;
  risk_tier_5yr: "Low" | "Moderate" | "High" | string;
  tier_cutoffs_5yr: Record<string, number>;
  benchmark_metrics: Record<string, number>;
}

export interface PCEScoreResult {
  availability_notice?: string | null;
  display_name: string;
  eligible: boolean;
  eligibility_criteria: string;
  estimated_10yr_ascvd_risk_pct: number | null;
  risk_category: string | null;
  benchmark_c_index_on_holdout: number;
  benchmark_5yr_auroc_on_holdout: number;
  citation: string;
}

export interface IDRSScoreResult {
  availability_notice?: string | null;
  display_name: string;
  score: number | null;
  risk_tier: "Low" | "Moderate" | "High" | string;
  interpretation: string;
  components: Record<string, number>;
  benchmark_auroc_on_holdout: number;
  citation: string;
}

export interface FutureRiskResponse {
  upload_id?: string; uploaded_at?: string; patient_id?: string | null; observed_on?: string | null; source_record_id?: string | null; data_origin?: DataOrigin;
  future_endpoints: Record<string, FutureRiskEndpoint>;
  published_scores: {
    pooled_cohort_equations_pce: PCEScoreResult;
    indian_diabetes_risk_score_idrs: IDRSScoreResult;
    [key: string]: any;
  };
  features_provided_count: number;
  nchs_privacy_perturbation_warning?: string;
  estimand_clarification: string;
  disclaimer: string;
}


export type ReportedSex = "female" | "male" | "intersex" | "unknown";
export interface Patient {
  id: string;
  alias: string;
  reported_sex: ReportedSex;
  created_at: string;
  record_count?: number;
  latest_observation?: string | null;
}
export interface TimelineRecord extends StoredRecordDetail {
  patient_id: string;
  observed_on: string;
  series_label: string;
  review_status: "unreviewed" | "reviewed" | "partially_reviewed";
}
export interface TrendPoint {
  date: string;
  value: number;
  record_id: string;
  filename: string;
  reference_range?: string;
  corrected?: boolean;
}
export interface TrendSeries {
  id: string;
  name: string;
  unit: string;
  direction: string;
  reason: string;
  change: number | null;
  points: TrendPoint[];
  modality?: string;
  series_label?: string;
  artifact_id?: string;
}
export interface PatientTimeline {
  patient: Patient;
  records: TimelineRecord[];
  lab_series: TrendSeries[];
  image_series: TrendSeries[];
  excluded_unreviewed: number;
  assessment: OverallAssessment;
  risk_series: TrendSeries[];
  rna_history: OverallAssessment["rna_findings"];
  comparison_version: string;
  notice: string;
}
export interface BatchOutcome {
  filename: string;
  status: "saved" | "duplicate" | "error";
  record_id?: string;
  summary?: string;
  message?: string;
}
export interface AuditEvent { action: string; resource_id: string; occurred_at: string }

export type DataOrigin = "user_supplied_unverified" | "generated_example";
export interface EvidenceMetric { label: string; value: number; lower_is_better: boolean }
export interface ModuleEvidence {
  id: string; name: string; source_filename: string; source_url: string; source_sha256?: string;
  artifact_integrity: boolean | null; evidence_status: "reported" | "unavailable";
  evaluation_scope: string; cohort: string; protocol: string; metrics: EvidenceMetric[];
}
export interface BenchmarkEvidence { modules: ModuleEvidence[]; notice: string }

export interface AnalysisContext { patient_id: string; observed_on: string; series_label?: string }
export interface RiskTransfer { features: Record<string, number>; source_record_id?: string | null; data_origin: DataOrigin; context: AnalysisContext }
export interface AssessmentFinding { record_id: string; date: string; review_status: string; data_origin?: DataOrigin; name: string; probability: number | null; tier: string; horizon: string; estimand: string }
export interface OverallAssessment {
  title: string; status: string; headline: string; as_of: string | null; generated_example_count: number; coverage: Record<string, number>; unreviewed_count: number;
  labs: {name: string; unit: string; value: number; flag: string; reference_range: string; date: string; record_id: string; direction: string; change: number | null}[];
  chronic_risks: AssessmentFinding[]; future_risks: AssessmentFinding[];
  rna_findings: {record_id: string; date: string; predicted_class: string; review_status: string; artifact_id?: string; interpretation: string}[];
  image_findings: {record_id: string; date: string; modality: string; series_label: string; status: string; predicted_class: string | null; active_conditions: string[]; message: string; review_status: string}[];
  gaps: string[]; trend_summary: {toward_range: number; away_from_range: number; comparable_image_series: number; comparable_risk_series: number}; interpretation: string;
}
