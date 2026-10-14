import { useState, useEffect } from "react";
import {
  Activity,
  ShieldCheck,
  Sparkles,
  RefreshCw,
  Sliders,
  Zap,
  Clock,
  HeartPulse,
  BookmarkCheck,
  Calendar,
} from "lucide-react";
import { api } from "../api";
import type {
  RiskPredictionResponse,
  ConditionRiskResult,
  FutureRiskResponse,
  FutureRiskEndpoint,
} from "../types";

interface RiskCalculatorPageProps {
  initialFeatures?: Record<string, number> | null;
}

export function RiskCalculatorPage({ initialFeatures }: RiskCalculatorPageProps) {
  const [inputs, setInputs] = useState<Record<string, any>>({
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
    waist_circumference_cm: 92,
    physical_activity: 1, // 0: Vigorous, 1: Moderate, 2: Sedentary
    family_history_diabetes: 0, // 0: None, 1: One Parent, 2: Both Parents
    bp_medication: 0, // 0: No, 1: Yes
    history_diabetes: 0, // 0: No, 1: Yes
    race_black: 0, // 0: Non-Black, 1: African American / Black
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RiskPredictionResponse | null>(null);
  const [futureResult, setFutureResult] = useState<FutureRiskResponse | null>(null);

  // Sync initialFeatures if passed from Lab Report transfer
  useEffect(() => {
    if (initialFeatures && Object.keys(initialFeatures).length > 0) {
      setInputs((prev) => ({
        ...prev,
        ...initialFeatures,
      }));
      runPrediction({ ...inputs, ...initialFeatures });
    } else {
      runPrediction(inputs);
    }
  }, [initialFeatures]);

  const updateInput = (key: string, value: any) => {
    setInputs((prev) => ({ ...prev, [key]: value }));
  };

  const runPrediction = async (customInputs?: Record<string, any>) => {
    const payload = customInputs || inputs;
    setLoading(true);
    setError(null);
    try {
      const [resRisk, resFuture] = await Promise.all([
        api.predictRisk(payload),
        api.predictFutureRisk(payload),
      ]);
      setResult(resRisk);
      setFutureResult(resFuture);
    } catch (err: any) {
      setError(err?.message || "Failed to calculate chronic disease and prospective survival risk.");
    } finally {
      setLoading(false);
    }
  };

  const loadPreset = (preset: "healthy" | "metabolic" | "senior") => {
    let p: Record<string, any> = {};
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
        waist_circumference_cm: 74,
        physical_activity: 0,
        family_history_diabetes: 0,
        bp_medication: 0,
        history_diabetes: 0,
        race_black: 0,
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
        waist_circumference_cm: 104,
        physical_activity: 2,
        family_history_diabetes: 1,
        bp_medication: 1,
        history_diabetes: 1,
        race_black: 0,
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
        waist_circumference_cm: 96,
        physical_activity: 1,
        family_history_diabetes: 0,
        bp_medication: 1,
        history_diabetes: 0,
        race_black: 0,
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
      case "Intermediate":
      case "Borderline":
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
      case "Intermediate":
      case "Borderline":
        return "bg-amber-500";
      default:
        return "bg-emerald-500";
    }
  };

  return (
    <div className="module-page space-y-6">
      <div className="page-heading">
        <div>
          <div className="eyebrow page-kicker">PHASE 2 & PART B · CHRONIC RISK & FUTURE MORTALITY ENGINE</div>
          <h1>Chronic Disease Risk & Future Mortality Engine</h1>
          <p>
            Population-calibrated clinical risk models trained on 17,961 NHANES adult participants (cross-sectional)
            and 47,281 adults linked to the CDC National Death Index across 16 years (future mortality risk).
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
                BMI: <strong className="text-slate-800">{inputs.bmi} kg/m2</strong>
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

            {/* Waist Circumference (IDRS predictor) */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                Waist Circ: <strong className="text-slate-800">{inputs.waist_circumference_cm} cm</strong>
              </label>
              <input
                type="range"
                min="60"
                max="140"
                value={inputs.waist_circumference_cm}
                onChange={(e) => updateInput("waist_circumference_cm", Number(e.target.value))}
                className="w-full accent-teal-600"
              />
            </div>

            {/* Race / Ethnicity (PCE comparator only) */}
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1">
                Race/Ethnicity <span className="text-[10px] text-slate-400 font-normal">(PCE only)</span>
              </label>
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  type="button"
                  className={`py-1 rounded text-center border text-[11px] font-medium ${
                    inputs.race_black === 0
                      ? "bg-teal-50 border-teal-500 text-teal-800 font-bold"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                  onClick={() => updateInput("race_black", 0)}
                >
                  Non-Black
                </button>
                <button
                  type="button"
                  className={`py-1 rounded text-center border text-[11px] font-medium ${
                    inputs.race_black === 1
                      ? "bg-teal-50 border-teal-500 text-teal-800 font-bold"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                  onClick={() => updateInput("race_black", 1)}
                >
                  Black / AA
                </button>
              </div>
              <p className="text-[10px] text-slate-400 mt-1 leading-tight">
                Not used in PulmoAI mortality models; collected solely for published 2013 ACC/AHA equations.
              </p>
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

          {/* Physical Activity & Family History (IDRS inputs) */}
          <div className="pt-2 border-t border-slate-200 grid grid-cols-2 gap-3 text-xs">
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1.5">Physical Activity (IDRS)</label>
              <div className="grid grid-cols-3 gap-1">
                {["Vigorous", "Moderate", "Sedentary"].map((label, idx) => (
                  <button
                    key={label}
                    type="button"
                    className={`py-1 rounded text-center border text-[10px] font-medium ${
                      inputs.physical_activity === idx
                        ? "bg-teal-50 border-teal-500 text-teal-800 font-bold"
                        : "border-slate-200 text-slate-600 hover:bg-slate-50"
                    }`}
                    onClick={() => updateInput("physical_activity", idx)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1.5">Family History Diabetes</label>
              <div className="grid grid-cols-3 gap-1">
                {["None", "1 Parent", "Both Parents"].map((label, idx) => (
                  <button
                    key={label}
                    type="button"
                    className={`py-1 rounded text-center border text-[10px] font-medium ${
                      inputs.family_history_diabetes === idx
                        ? "bg-teal-50 border-teal-500 text-teal-800 font-bold"
                        : "border-slate-200 text-slate-600 hover:bg-slate-50"
                    }`}
                    onClick={() => updateInput("family_history_diabetes", idx)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Clinical Treatment History (PCE inputs) */}
          <div className="pt-2 border-t border-slate-200 grid grid-cols-2 gap-3 text-xs">
            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1.5">On BP Medication?</label>
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  type="button"
                  className={`py-1 rounded text-center border text-[11px] font-medium ${
                    inputs.bp_medication === 0
                      ? "bg-teal-50 border-teal-500 text-teal-800 font-bold"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                  onClick={() => updateInput("bp_medication", 0)}
                >
                  No
                </button>
                <button
                  type="button"
                  className={`py-1 rounded text-center border text-[11px] font-medium ${
                    inputs.bp_medication === 1
                      ? "bg-teal-50 border-teal-500 text-teal-800 font-bold"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                  onClick={() => updateInput("bp_medication", 1)}
                >
                  Yes
                </button>
              </div>
            </div>

            <div>
              <label className="block text-[11px] font-medium text-slate-600 mb-1.5">Diagnosed Diabetes?</label>
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  type="button"
                  className={`py-1 rounded text-center border text-[11px] font-medium ${
                    inputs.history_diabetes === 0
                      ? "bg-teal-50 border-teal-500 text-teal-800 font-bold"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                  onClick={() => updateInput("history_diabetes", 0)}
                >
                  No
                </button>
                <button
                  type="button"
                  className={`py-1 rounded text-center border text-[11px] font-medium ${
                    inputs.history_diabetes === 1
                      ? "bg-teal-50 border-teal-500 text-teal-800 font-bold"
                      : "border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                  onClick={() => updateInput("history_diabetes", 1)}
                >
                  Yes
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Prediction Results Display */}
        <div className="lg:col-span-7 space-y-6">
          {error && (
            <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-xs">
              {error}
            </div>
          )}

          {/* ============================================================== */}
          {/* SECTION 1: FUTURE RISK (MORTALITY-LINKED SURVIVAL PROGNOSTICS) */}
          {/* ============================================================== */}
          {futureResult && (
            <div className="space-y-4 p-5 rounded-2xl bg-white border border-teal-200/80 shadow-sm">
              <div className="flex items-center justify-between border-b border-slate-200 pb-3">
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-wider text-teal-700 flex items-center gap-1.5">
                    <Clock size={13} className="text-teal-600" /> Future mortality risk (US survey data)
                  </div>
                  <h3 className="text-sm font-bold text-slate-900 mt-0.5">
                    Future Risk: 5-Year & 10-Year Mortality Endpoints
                  </h3>
                </div>
                <span className="text-[10px] bg-teal-50 text-teal-800 border border-teal-200 px-2 py-0.5 rounded font-medium">
                  CDC NHANES Linked Follow-up (1999-2019)
                </span>
              </div>

              {/* Prominent Estimand Distinction Banner */}
              <div className="p-3.5 rounded-xl bg-amber-50/80 border border-amber-200 text-[11px] text-amber-900 leading-relaxed flex items-start gap-2.5">
                <ShieldCheck size={18} className="shrink-0 text-amber-700 mt-0.5" />
                <div className="space-y-1">
                  <div className="font-semibold text-amber-950">
                    CRITICAL ESTIMAND DISTINCTION: Future mortality risk (US survey data)
                  </div>
                  <p className="text-[10px] text-amber-800 leading-normal">
                    The models in this section estimate <strong>future mortality risk (US survey data)</strong> over fixed
                    5-year (60 months) and 10-year (120 months) horizons derived from 47,281 civilian adults tracked
                    prospectively through the National Death Index on temporal holdout (2011-2014). This is mathematically and clinically distinct from
                    cross-sectional prevalence of current doctor diagnoses.
                  </p>
                </div>
              </div>

              {/* NCHS Data Perturbation Warning */}
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-[10px] text-slate-600 leading-relaxed">
                <strong>Data Privacy Notice:</strong> The public-use NHANES Linked Mortality Files (LMF) include synthetic data perturbation
                applied by NCHS to protect respondent confidentiality; subgroup counts and survival intervals may slightly diverge from unperturbed restricted-use files.
              </div>

              {/* Future Endpoints Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                {Object.entries(futureResult.future_endpoints).map(([endKey, endData]: [string, FutureRiskEndpoint]) => {
                  const risk5yr = endData.time_horizons["5_year"]?.risk_percent ?? 0;
                  const risk10yr = endData.time_horizons["10_year"]?.risk_percent ?? 0;
                  const tier = endData.risk_tier_5yr;

                  return (
                    <div
                      key={endKey}
                      className="p-4 rounded-xl bg-slate-50/60 border border-slate-200 hover:border-teal-300 transition-colors relative overflow-hidden flex flex-col justify-between"
                    >
                      <div>
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <div className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                              <HeartPulse size={14} className="text-rose-600" />
                              {endData.display_name}
                            </div>
                            <div className="text-[10px] text-slate-500 mt-0.5">
                              C-index {endData.benchmark_metrics?.c_index} · 5-yr AUROC {endData.benchmark_metrics?.holdout_5yr_auroc}
                            </div>
                          </div>
                          <span
                            className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${getTierColor(tier)}`}
                          >
                            {tier} Risk (5-Yr)
                          </span>
                        </div>

                        {/* Dual Horizon Display */}
                        <div className="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-slate-200">
                          <div>
                            <div className="text-[10px] font-semibold text-slate-500 flex items-center gap-1">
                              <Calendar size={11} className="text-teal-600" /> 5-Year Mortality Risk
                            </div>
                            <div className="text-xl font-black text-slate-900 mt-0.5">
                              {risk5yr.toFixed(2)}%
                            </div>
                          </div>
                          <div>
                            <div className="text-[10px] font-semibold text-slate-500 flex items-center gap-1">
                              <Calendar size={11} className="text-slate-400" /> 10-Year Mortality Risk
                            </div>
                            <div className="text-xl font-black text-slate-800 mt-0.5">
                              {risk10yr.toFixed(2)}%
                            </div>
                          </div>
                        </div>

                        {/* Progress Bar for 5-Year Risk */}
                        <div className="w-full h-1.5 rounded-full bg-slate-200 overflow-hidden mt-3">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${getProgressBarColor(tier)}`}
                            style={{ width: `${Math.min(100, Math.max(4, risk5yr * 2))}%` }}
                          />
                        </div>
                      </div>

                      {/* Estimand Note */}
                      <div className="mt-3 text-[9px] text-slate-500 bg-white border border-slate-200/80 rounded px-2 py-1 leading-snug">
                        {endData.estimand_description}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Published Clinical Scores Benchmark Comparison Cards */}
              <div className="pt-2 border-t border-slate-200">
                <div className="text-[11px] font-bold text-slate-700 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                  <BookmarkCheck size={14} className="text-teal-600" /> Head-to-Head Comparison with Published Clinical Scores
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  {/* ACC/AHA Pooled Cohort Equations */}
                  {futureResult.published_scores?.pooled_cohort_equations_pce && (
                    <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                      <div className="flex items-start justify-between gap-1">
                        <div>
                          <div className="font-bold text-slate-900">
                            {futureResult.published_scores.pooled_cohort_equations_pce.display_name}
                          </div>
                          <div className="text-[10px] text-slate-500">
                            Holdout C-index: {futureResult.published_scores.pooled_cohort_equations_pce.benchmark_c_index_on_holdout} · AUROC: {futureResult.published_scores.pooled_cohort_equations_pce.benchmark_5yr_auroc_on_holdout}
                          </div>
                        </div>
                        {futureResult.published_scores.pooled_cohort_equations_pce.risk_category && (
                          <span
                            className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${getTierColor(
                              futureResult.published_scores.pooled_cohort_equations_pce.risk_category
                            )}`}
                          >
                            {futureResult.published_scores.pooled_cohort_equations_pce.risk_category}
                          </span>
                        )}
                      </div>

                      <div className="mt-2 text-lg font-black text-slate-900">
                        {futureResult.published_scores.pooled_cohort_equations_pce.estimated_10yr_ascvd_risk_pct !== null
                          ? `${futureResult.published_scores.pooled_cohort_equations_pce.estimated_10yr_ascvd_risk_pct.toFixed(1)}% 10-Yr ASCVD Risk`
                          : "Ineligible (Age outside 40-79 or missing lipids)"}
                      </div>
                      <div className="text-[10px] text-slate-500 mt-1">
                        Primary prevention cohort (ages 40-79 without prior CVD). Evaluated for discrimination against fatal CVD.
                      </div>
                      <div className="text-[9px] text-slate-500 mt-1 italic">
                        {futureResult.published_scores.pooled_cohort_equations_pce.citation}
                      </div>
                    </div>
                  )}

                  {/* Indian Diabetes Risk Score (IDRS) */}
                  {futureResult.published_scores?.indian_diabetes_risk_score_idrs && (
                    <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200">
                      <div className="flex items-start justify-between gap-1">
                        <div>
                          <div className="font-bold text-slate-900">
                            {futureResult.published_scores.indian_diabetes_risk_score_idrs.display_name}
                          </div>
                          <div className="text-[10px] text-slate-500">
                            Holdout AUROC: {futureResult.published_scores.indian_diabetes_risk_score_idrs.benchmark_auroc_on_holdout}
                          </div>
                        </div>
                        <span
                          className={`text-[9px] font-bold px-1.5 py-0.5 rounded border ${getTierColor(
                            futureResult.published_scores.indian_diabetes_risk_score_idrs.risk_tier
                          )}`}
                        >
                          {futureResult.published_scores.indian_diabetes_risk_score_idrs.risk_tier} Risk
                        </span>
                      </div>

                      <div className="mt-2 text-lg font-black text-slate-900">
                        {futureResult.published_scores.indian_diabetes_risk_score_idrs.score} / 100 Points
                      </div>
                      <div className="text-[10px] text-slate-600 mt-0.5">
                        Age: {futureResult.published_scores.indian_diabetes_risk_score_idrs.components?.age_points} · Waist: {futureResult.published_scores.indian_diabetes_risk_score_idrs.components?.waist_points} · Activity: {futureResult.published_scores.indian_diabetes_risk_score_idrs.components?.activity_points} · Family: {futureResult.published_scores.indian_diabetes_risk_score_idrs.components?.family_history_points}
                      </div>
                      <div className="text-[10px] text-slate-500 mt-1">
                        * Physical activity and family history are approximate mappings from NHANES variables.
                      </div>
                      <div className="text-[9px] text-slate-500 mt-1 italic">
                        {futureResult.published_scores.indian_diabetes_risk_score_idrs.citation}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* ============================================================== */}
          {/* SECTION 2: CROSS-SECTIONAL CHRONIC CONDITION RISK (PREVALENCE) */}
          {/* ============================================================== */}
          {result && (
            <div className="space-y-3 pt-2">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600 flex items-center gap-1.5">
                    <Activity size={14} className="text-teal-600" /> Cross-Sectional Chronic Condition Risk (Current Prevalence)
                  </h3>
                  <div className="text-[10px] text-slate-500">
                    Estimates probability of current doctor-diagnosed condition among US adults (NHANES Cycles H-J)
                  </div>
                </div>
                <span className="text-[10px] text-slate-500 font-medium">
                  Evaluation: Holdout Cycle J (2017-2018)
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
                        <span className="text-[10px] font-medium text-slate-500">
                          {condData.time_horizon || "Prevalent condition / survey history"}
                        </span>
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

                      {/* Estimand Clarification Badge */}
                      <div className="mt-2.5 text-[10px] bg-slate-50 border border-slate-200/80 rounded px-2 py-1 text-slate-600">
                        <div className="font-semibold text-slate-700">
                          {condData.estimand_label || "Estimand: Cross-Sectional Survey"}
                        </div>
                        <div className="text-[9px] text-slate-500 mt-0.5 leading-snug">
                          {condData.estimand_description}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Disclaimer Banner */}
              <div className="p-3.5 rounded-xl bg-slate-100 border border-slate-200 text-[11px] text-slate-600 leading-relaxed flex items-start gap-2.5">
                <ShieldCheck size={18} className="shrink-0 text-teal-600 mt-0.5" />
                <div className="space-y-1">
                  <div>
                    <strong>Estimand Clarification:</strong>{" "}
                    {result.estimand_clarification ||
                      "All outputs in this section estimate the probability of a CURRENT self-reported diagnosis in civilian adults; they do NOT predict future disease incidence."}
                  </div>
                  <div className="text-[10px] text-slate-500">
                    <strong>Safeguards:</strong> Fasting glucose and HbA1c are strictly withheld from diabetes modeling to avoid diagnostic circularity. Screening triage only; not a clinical diagnosis.
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
