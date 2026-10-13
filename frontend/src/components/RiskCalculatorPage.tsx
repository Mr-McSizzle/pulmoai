import { useState, useEffect } from "react";
import {
  Activity,
  ShieldCheck,
  Sparkles,
  RefreshCw,
  Sliders,
  Zap,
} from "lucide-react";
import { api } from "../api";
import type { RiskPredictionResponse, ConditionRiskResult } from "../types";

interface RiskCalculatorPageProps {
  initialFeatures?: Record<string, number> | null;
}

export function RiskCalculatorPage({ initialFeatures }: RiskCalculatorPageProps) {
  const [inputs, setInputs] = useState<Record<string, number>>({
    age: 52,
    gender: 1, // 1: Male, 2: Female
    systolic_bp: 136,
    diastolic_bp: 86,
    bmi: 28.5,
    pulse: 74,
    total_cholesterol: 215,
    hdl_cholesterol: 44,
    triglycerides: 165,
    creatinine: 1.05,
    smoking_status: 0, // 0: Never, 1: Former, 2: Current
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RiskPredictionResponse | null>(null);

  // Sync initialFeatures if passed from Lab Report transfer
  useEffect(() => {
    if (initialFeatures && Object.keys(initialFeatures).length > 0) {
      setInputs((prev) => ({
        ...prev,
        ...initialFeatures,
      }));
      // Auto-run prediction with the transferred values
      runPrediction({ ...inputs, ...initialFeatures });
    } else {
      // Run initial prediction on default values
      runPrediction(inputs);
    }
  }, [initialFeatures]);

  const updateInput = (key: string, value: number) => {
    setInputs((prev) => ({ ...prev, [key]: value }));
  };

  const runPrediction = async (customInputs?: Record<string, number>) => {
    const payload = customInputs || inputs;
    setLoading(true);
    setError(null);
    try {
      const res = await api.predictRisk(payload);
      setResult(res);
    } catch (err: any) {
      setError(err?.message || "Failed to calculate chronic disease risk.");
    } finally {
      setLoading(false);
    }
  };

  const loadPreset = (preset: "healthy" | "metabolic" | "senior") => {
    let p: Record<string, number> = {};
    if (preset === "healthy") {
      p = {
        age: 32,
        gender: 2,
        systolic_bp: 115,
        diastolic_bp: 74,
        bmi: 22.4,
        pulse: 68,
        total_cholesterol: 175,
        hdl_cholesterol: 58,
        triglycerides: 110,
        creatinine: 0.85,
        smoking_status: 0,
      };
    } else if (preset === "metabolic") {
      p = {
        age: 58,
        gender: 1,
        systolic_bp: 148,
        diastolic_bp: 94,
        bmi: 33.2,
        pulse: 82,
        total_cholesterol: 242,
        hdl_cholesterol: 35,
        triglycerides: 230,
        creatinine: 1.25,
        smoking_status: 2,
      };
    } else {
      p = {
        age: 69,
        gender: 1,
        systolic_bp: 156,
        diastolic_bp: 84,
        bmi: 27.8,
        pulse: 76,
        total_cholesterol: 225,
        hdl_cholesterol: 42,
        triglycerides: 170,
        creatinine: 1.35,
        smoking_status: 1,
      };
    }
    setInputs(p);
    runPrediction(p);
  };

  const getTierColor = (tier: string) => {
    switch (tier) {
      case "High":
        return "bg-rose-100 text-rose-800 border-rose-200";
      case "Moderate":
        return "bg-amber-100 text-amber-800 border-amber-200";
      default:
        return "bg-emerald-100 text-emerald-800 border-emerald-200";
    }
  };

  const getProgressBarColor = (tier: string) => {
    switch (tier) {
      case "High":
        return "bg-rose-500";
      case "Moderate":
        return "bg-amber-500";
      default:
        return "bg-emerald-500";
    }
  };

  return (
    <div className="module-page space-y-6">
      <div className="page-heading">
        <div>
          <div className="eyebrow page-kicker">PHASE 2 · CHRONIC RISK ENGINE</div>
          <h1>Chronic Disease Risk Engine</h1>
          <p>
            Population-calibrated risk models trained on 17,961 NHANES adult participants.
            Achieves <strong>85.3% AUROC / 86.6% accuracy</strong> on out-of-sample forward evaluation cohorts.
          </p>
        </div>
      </div>

      {/* Preset Buttons */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold text-slate-500 flex items-center gap-1">
          <Zap size={14} className="text-amber-500" /> Quick Patient Presets:
        </span>
        <button
          className="button-secondary text-xs py-1.5 px-3"
          onClick={() => loadPreset("healthy")}
        >
          Healthy Young Adult (32yo)
        </button>
        <button
          className="button-secondary text-xs py-1.5 px-3"
          onClick={() => loadPreset("metabolic")}
        >
          Metabolic Syndrome Profile (58yo)
        </button>
        <button
          className="button-secondary text-xs py-1.5 px-3"
          onClick={() => loadPreset("senior")}
        >
          Senior Cardiovascular Risk (69yo)
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Input Parameters Panel */}
        <div className="lg:col-span-5 content-panel space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-200">
            <h2 className="text-sm font-semibold flex items-center gap-2 text-slate-800">
              <Sliders size={16} className="text-teal-600" /> Patient Biomarkers & Vitals
            </h2>
            <button
              className="button-primary text-xs py-1.5 px-3 flex items-center gap-1"
              disabled={loading}
              onClick={() => runPrediction()}
            >
              {loading ? <RefreshCw size={12} className="animate-spin" /> : <Sparkles size={12} />}
              Calculate
            </button>
          </div>

          <div className="grid grid-cols-2 gap-3.5 text-xs">
            {/* Age */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                Age: <strong className="text-slate-800">{inputs.age} yrs</strong>
              </label>
              <input
                type="range"
                min="18"
                max="85"
                value={inputs.age}
                onChange={(e) => updateInput("age", Number(e.target.value))}
                className="w-full accent-teal-600"
              />
            </div>

            {/* Gender */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">Biological Sex</label>
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  type="button"
                  className={`py-1 rounded text-center border text-[11px] font-medium ${
                    inputs.gender === 1
                      ? "bg-teal-50 border-teal-500 text-teal-800 font-bold"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                  onClick={() => updateInput("gender", 1)}
                >
                  Male
                </button>
                <button
                  type="button"
                  className={`py-1 rounded text-center border text-[11px] font-medium ${
                    inputs.gender === 2
                      ? "bg-teal-50 border-teal-500 text-teal-800 font-bold"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                  onClick={() => updateInput("gender", 2)}
                >
                  Female
                </button>
              </div>
            </div>

            {/* Systolic BP */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                Systolic BP: <strong className="text-slate-800">{inputs.systolic_bp} mmHg</strong>
              </label>
              <input
                type="range"
                min="90"
                max="200"
                value={inputs.systolic_bp}
                onChange={(e) => updateInput("systolic_bp", Number(e.target.value))}
                className="w-full accent-teal-600"
              />
            </div>

            {/* Diastolic BP */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                Diastolic BP: <strong className="text-slate-800">{inputs.diastolic_bp} mmHg</strong>
              </label>
              <input
                type="range"
                min="50"
                max="120"
                value={inputs.diastolic_bp}
                onChange={(e) => updateInput("diastolic_bp", Number(e.target.value))}
                className="w-full accent-teal-600"
              />
            </div>

            {/* BMI */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                BMI: <strong className="text-slate-800">{inputs.bmi} kg/m²</strong>
              </label>
              <input
                type="range"
                min="16"
                max="45"
                step="0.1"
                value={inputs.bmi}
                onChange={(e) => updateInput("bmi", Number(e.target.value))}
                className="w-full accent-teal-600"
              />
            </div>

            {/* Resting Pulse */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                Resting Pulse: <strong className="text-slate-800">{inputs.pulse} bpm</strong>
              </label>
              <input
                type="range"
                min="45"
                max="120"
                value={inputs.pulse}
                onChange={(e) => updateInput("pulse", Number(e.target.value))}
                className="w-full accent-teal-600"
              />
            </div>

            {/* Total Cholesterol */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                Total Cholesterol: <strong className="text-slate-800">{inputs.total_cholesterol} mg/dL</strong>
              </label>
              <input
                type="range"
                min="110"
                max="350"
                value={inputs.total_cholesterol}
                onChange={(e) => updateInput("total_cholesterol", Number(e.target.value))}
                className="w-full accent-teal-600"
              />
            </div>

            {/* HDL Cholesterol */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                HDL Cholesterol: <strong className="text-slate-800">{inputs.hdl_cholesterol} mg/dL</strong>
              </label>
              <input
                type="range"
                min="20"
                max="100"
                value={inputs.hdl_cholesterol}
                onChange={(e) => updateInput("hdl_cholesterol", Number(e.target.value))}
                className="w-full accent-teal-600"
              />
            </div>

            {/* Triglycerides */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                Triglycerides: <strong className="text-slate-800">{inputs.triglycerides} mg/dL</strong>
              </label>
              <input
                type="range"
                min="40"
                max="400"
                value={inputs.triglycerides}
                onChange={(e) => updateInput("triglycerides", Number(e.target.value))}
                className="w-full accent-teal-600"
              />
            </div>

            {/* Creatinine */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                Creatinine: <strong className="text-slate-800">{inputs.creatinine} mg/dL</strong>
              </label>
              <input
                type="range"
                min="0.4"
                max="3.0"
                step="0.05"
                value={inputs.creatinine}
                onChange={(e) => updateInput("creatinine", Number(e.target.value))}
                className="w-full accent-teal-600"
              />
            </div>
          </div>

          {/* Smoking Status */}
          <div className="pt-2 border-t border-slate-200">
            <label className="block text-[11px] font-medium text-slate-600 mb-1.5">Smoking History</label>
            <div className="grid grid-cols-3 gap-1.5 text-xs">
              {["Never", "Former", "Current"].map((label, idx) => (
                <button
                  key={label}
                  type="button"
                  className={`py-1.5 rounded text-center border text-[11px] font-medium ${
                    inputs.smoking_status === idx
                      ? "bg-teal-50 border-teal-500 text-teal-800 font-bold"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                  onClick={() => updateInput("smoking_status", idx)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Prediction Results Display */}
        <div className="lg:col-span-7 space-y-4">
          {error && (
            <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-xs">
              {error}
            </div>
          )}

          {result && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
                  <Activity size={14} className="text-teal-600" /> Multi-Condition Risk Assessment
                </h3>
                <span className="text-[10px] text-slate-500 font-medium">
                  Evaluation: NHANES Forward Cycle J
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                {Object.entries(result.risk_predictions).map(([condKey, condData]: [string, ConditionRiskResult]) => {
                  const prob = condData.calibrated_risk !== null ? condData.calibrated_risk : 0;
                  const percentStr = condData.calibrated_risk !== null ? `${(prob * 100).toFixed(1)}%` : "N/A";
                  return (
                    <div
                      key={condKey}
                      className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-shadow relative overflow-hidden"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div>
                          <div className="text-xs font-bold text-slate-900">{condData.display_name}</div>
                          <div className="text-[10px] text-slate-500 mt-0.5">
                            {condData.model_auroc ? `${(condData.model_auroc * 100).toFixed(1)}% AUROC · ` : ""}
                            {condData.description}
                          </div>
                        </div>
                        <span
                          className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${getTierColor(
                            condData.risk_category
                          )}`}
                        >
                          {condData.risk_category} Risk
                        </span>
                      </div>

                      <div className="mt-3.5 flex items-baseline justify-between">
                        <span className="text-2xl font-black text-slate-900">{percentStr}</span>
                        <span className="text-[10px] font-medium text-slate-500">10-yr predicted probability</span>
                      </div>

                      {/* Progress Bar */}
                      <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden mt-2">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${getProgressBarColor(
                            condData.risk_category
                          )}`}
                          style={{ width: `${Math.min(100, Math.max(4, prob * 100))}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Disclaimer Banner */}
              <div className="p-3.5 rounded-xl bg-slate-100 border border-slate-200 text-[11px] text-slate-600 leading-relaxed flex items-start gap-2.5">
                <ShieldCheck size={18} className="shrink-0 text-teal-600 mt-0.5" />
                <div>
                  <strong>Epidemiological Calibration:</strong> Probabilities are calibrated on CDC NHANES out-of-sample forward test cycles.
                  HbA1c and fasting glucose are strictly withheld to prevent tautological self-diagnosis.
                  This output provides personalized health risk screening, not a definitive clinical diagnosis.
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
