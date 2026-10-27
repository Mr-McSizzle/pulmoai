import type { AnalysisContext } from "./types";
import type { BenchmarkEvidence, DataOrigin } from "./types";
import type { Patient, PatientTimeline, ReportedSex, BatchOutcome, AuditEvent } from "./types";
import type {
  DashboardData,
  AuthResponse,
  StoredRecord,
  StoredRecordDetail,
  GeneInsights,
  HealthResponse,
  FutureRiskResponse,
  ImagePredictionResponse,
  PredictionResponse,
  ReportExtractResponse,
  RiskPredictionResponse,
  ValidationResponse,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new Error("The local API is unreachable. Start the PulmoAI backend and try again.");
  }
  if (!response.ok) {
    let message = `The request failed (${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: string | { msg: string; loc?: string[] }[] };
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map(item => `${item.loc?.slice(1).join(".") || "Input"}: ${item.msg}`).join("; ");
    } catch {
      message = `The request failed (${response.status}).`;
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function upload(path: string, file: File): Promise<ValidationResponse | PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  return request(path, { method: "POST", body: form });
}

function appendContext(form: FormData, context?: AnalysisContext | null) {
  if (!context) return;
  form.append("patient_id", context.patient_id);
  form.append("observed_on", context.observed_on);
  if (context.series_label) form.append("series_label", context.series_label);
}

export const api = {
  authMe: () => request<AuthResponse>("/api/auth/me"),
  register: (fullName: string, email: string, password: string) => request<AuthResponse>("/api/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ full_name: fullName, email, password }),
  }),
  login: (email: string, password: string) => request<AuthResponse>("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  }),
  logout: () => request<void>("/api/auth/logout", { method: "POST" }),
  uploads: () => request<StoredRecord[]>("/api/uploads"),
  uploadDetail: (id: string) => request<StoredRecordDetail>(`/api/uploads/${encodeURIComponent(id)}`),
  deleteUpload: (id: string) => request<void>(`/api/uploads/${encodeURIComponent(id)}`, { method: "DELETE" }),
  health: () => request<HealthResponse>("/api/health"),
  dashboard: () => request<DashboardData>("/api/dashboard"),
  evidence: () => request<BenchmarkEvidence>("/api/evidence"),
  insights: () => request<GeneInsights>("/api/insights"),
  validate: (file: File) => upload("/api/validate", file) as Promise<ValidationResponse>,
  predict: (file: File, reportedSex: ReportedSex = "unknown", context?: AnalysisContext | null): Promise<PredictionResponse> => {
    const form = new FormData();
    form.append("file", file);
    form.append("reported_sex", reportedSex);
    appendContext(form, context);
    return request("/api/predict", { method: "POST", body: form });
  },
  patients: () => request<Patient[]>("/api/patients"),
  createPatient: (alias: string, reported_sex: ReportedSex) => request<Patient>("/api/patients", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ alias, reported_sex }),
  }),
  timeline: (id: string) => request<PatientTimeline>(`/api/patients/${encodeURIComponent(id)}/timeline`),
  audit: () => request<AuditEvent[]>("/api/workspace/audit"),
  review: (id: string, corrections: Record<string, number>, confirmedTests?: string[]) => request<void>(`/api/uploads/${encodeURIComponent(id)}/review`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirmed: true, corrections, confirmed_tests: confirmedTests }),
  }),
  linkRecord: (id: string, patient_id: string, observed_on: string, series_label: string) => request<void>(`/api/uploads/${encodeURIComponent(id)}/context`, {
    method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ patient_id, observed_on, series_label }),
  }),
  batch: (patientId: string, files: File[], observations: { observed_on: string; series_label: string; data_origin?: DataOrigin }[], kind: "lab_report" | "medical_image", imageType: string, pediatric: boolean, analysisMode: "screen" | "archive_only" = "screen") => {
    const form = new FormData();
    files.forEach(file => form.append("files", file));
    form.append("observations", JSON.stringify(observations));
    form.append("kind", kind);
    form.append("analysis_mode", analysisMode);
    form.append("image_type", imageType);
    form.append("patient_is_pediatric", String(pediatric));
    return request<{ outcomes: BatchOutcome[] }>(`/api/patients/${encodeURIComponent(patientId)}/batch`, { method: "POST", body: form });
  },

  // Phase 3: Lab Report Extraction
  extractReport: (file: File, sex?: string, dataOrigin: DataOrigin = "user_supplied_unverified", context?: AnalysisContext | null): Promise<ReportExtractResponse> => {
    const form = new FormData();
    form.append("file", file);
    if (sex) form.append("sex", sex);
    appendContext(form, context);
    form.append("data_origin", dataOrigin);
    return request<ReportExtractResponse>("/api/report/extract", { method: "POST", body: form });
  },

  // Phase 2: Chronic Disease Risk Engine
  predictRisk: (inputs: Record<string, number>, context?: AnalysisContext | null, source_record_id?: string | null, data_origin: DataOrigin = "user_supplied_unverified"): Promise<RiskPredictionResponse> => {
    return request<RiskPredictionResponse>("/api/risk/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs, context, source_record_id, data_origin }),
    });
  },

  // Phase 6: Future Risk (Mortality-Linked) & Published Scores
  predictFutureRisk: (inputs: Record<string, any>, context?: AnalysisContext | null, source_record_id?: string | null, data_origin: DataOrigin = "user_supplied_unverified"): Promise<FutureRiskResponse> => {
    return request<FutureRiskResponse>("/api/risk/future", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs, context, source_record_id, data_origin }),
    });
  },

  // Phase 4: Medical Image Screening
  predictImage: (
    file: File,
    imageType: string = "chest_xray",
    oodThreshold: number = 0.50,
    patientIsPediatric: boolean = false,
    dataOrigin: DataOrigin = "user_supplied_unverified",
    context?: AnalysisContext | null,
  ): Promise<ImagePredictionResponse> => {
    const form = new FormData();
    form.append("file", file);
    form.append("image_type", imageType);
    form.append("ood_threshold", oodThreshold.toString());
    appendContext(form, context);
    form.append("patient_is_pediatric", patientIsPediatric ? "true" : "false");
    form.append("data_origin", dataOrigin);
    return request<ImagePredictionResponse>("/api/image/predict", { method: "POST", body: form });
  },

  // scRNA Demo Samples
  fetchScRNASample: async (sampleName: string): Promise<File> => {
    const res = await fetch(`/api/sample/scrna/${sampleName}`);
    if (!res.ok) throw new Error(`Could not load ${sampleName} sample.`);
    const blob = await res.blob();
    return new File([blob], `scrna_sample_${sampleName}.csv`, { type: "text/csv" });
  },
};
