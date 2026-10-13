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
  probabilities_available: false;
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
