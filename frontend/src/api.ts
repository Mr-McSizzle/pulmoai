import type {
  DashboardData,
  GeneInsights,
  HealthResponse,
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
  return (await response.json()) as T;
}

function upload(path: string, file: File): Promise<ValidationResponse | PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  return request(path, { method: "POST", body: form });
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  dashboard: () => request<DashboardData>("/api/dashboard"),
  insights: () => request<GeneInsights>("/api/insights"),
  validate: (file: File) => upload("/api/validate", file) as Promise<ValidationResponse>,
  predict: (file: File) => upload("/api/predict", file) as Promise<PredictionResponse>,

  // Phase 3: Lab Report Extraction
  extractReport: (file: File): Promise<ReportExtractResponse> => {
    const form = new FormData();
    form.append("file", file);
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

  // Phase 4: Medical Image Screening
  predictImage: (file: File, imageType: string = "chest_xray", oodThreshold: number = 0.50): Promise<ImagePredictionResponse> => {
    const form = new FormData();
    form.append("file", file);
    form.append("image_type", imageType);
    form.append("ood_threshold", oodThreshold.toString());
    return request<ImagePredictionResponse>("/api/image/predict", { method: "POST", body: form });
  },
};
