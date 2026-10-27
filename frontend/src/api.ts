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
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message = body.detail;
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
  insights: () => request<GeneInsights>("/api/insights"),
  validate: (file: File) => upload("/api/validate", file) as Promise<ValidationResponse>,
  predict: (file: File) => upload("/api/predict", file) as Promise<PredictionResponse>,

  // Phase 3: Lab Report Extraction
  extractReport: (file: File, sex?: string): Promise<ReportExtractResponse> => {
    const form = new FormData();
    form.append("file", file);
    if (sex) form.append("sex", sex);
    return request<ReportExtractResponse>("/api/report/extract", { method: "POST", body: form });
  },

  // Phase 2: Chronic Disease Risk Engine
  predictRisk: (inputs: Record<string, number>): Promise<RiskPredictionResponse> => {
    return request<RiskPredictionResponse>("/api/risk/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs }),
    });
  },

  // Phase 6: Future Risk (Mortality-Linked) & Published Scores
  predictFutureRisk: (inputs: Record<string, any>): Promise<FutureRiskResponse> => {
    return request<FutureRiskResponse>("/api/risk/future", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs }),
    });
  },

  // Phase 4: Medical Image Screening
  predictImage: (
    file: File,
    imageType: string = "chest_xray",
    oodThreshold: number = 0.50,
    patientIsPediatric: boolean = true,
  ): Promise<ImagePredictionResponse> => {
    const form = new FormData();
    form.append("file", file);
    form.append("image_type", imageType);
    form.append("ood_threshold", oodThreshold.toString());
    form.append("patient_is_pediatric", patientIsPediatric ? "true" : "false");
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
