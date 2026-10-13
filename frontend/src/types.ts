export interface HealthResponse {
  status: "ok" | "unavailable";
  model_loaded: boolean;
  model_name: string | null;
  artifact_id: string | null;
  message: string;
}

export interface Metrics {
  accuracy: number;
  balanced_accuracy: number;
  macro_f1: number;
  weighted_f1: number;
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
  predicted_class: string;
  model_name: string;
  artifact_id: string;
  input_profile_count: number;
  input_gene_count: number;
  probabilities_available: boolean;
  class_probabilities?: Record<string, number> | null;
  interpretation: string;
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
  extraction_method: string;
  extracted_tests: LabItemSchema[];
  raw_text: string;
  unmatched_lines: string[];
  count: number;
}

// ---------------------------------------------------------------------------
// Phase 2 & 5: Chronic Disease Risk Engine Schemas
// ---------------------------------------------------------------------------

export interface ConditionRiskResult {
  target_id: string;
  display_name: string;
  description: string;
  calibrated_risk: number | null;
  risk_category: string;
  model_auroc?: number | null;
  model_brier_score?: number | null;
  test_prevalence?: number | null;
}

export interface RiskPredictionResponse {
  risk_predictions: Record<string, ConditionRiskResult>;
  contributing_factors: string[];
  features_provided_count: number;
  disclaimer: string;
}

// ---------------------------------------------------------------------------
// Phase 4 & 5: Medical Image Screening Schemas
// ---------------------------------------------------------------------------

export interface ImagePredictionResponse {
  image_type: string;
  modality_name: string;
  recognized: boolean;
  status: "success" | "low_confidence";
  predicted_class?: string | null;
  confidence?: number;
  max_confidence?: number;
  threshold_applied?: number;
  message?: string;
  multilabel?: boolean;
  active_conditions?: string[];
  probabilities?: Record<string, number>;
  class_probabilities?: Record<string, number>;
  model_metrics?: Record<string, number>;
  disclaimer: string;
}
