import { useState, type ChangeEvent, type DragEvent } from "react";
import {
  FileText,
  Upload,
  Sparkles,
  ArrowRight,
  Activity,
  Layers,
  FileCheck,
  Eye,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
} from "lucide-react";
import { api } from "../api";
import type { ReportExtractResponse, LabItemSchema } from "../types";

interface LabReportPageProps {
  onTransferToRisk: (features: Record<string, number>) => void;
}

export function LabReportPage({ onTransferToRisk }: LabReportPageProps) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ReportExtractResponse | null>(null);
  const [showRawText, setShowRawText] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [patientSex, setPatientSex] = useState<string>("unspecified");
  const [editedValues, setEditedValues] = useState<Record<string, number>>({});
  const [confirmedTests, setConfirmedTests] = useState<Record<string, boolean>>({});

  const handleFile = (selected: File) => {
    setFile(selected);
    setError(null);
    setResult(null);
    setEditedValues({});
    setConfirmedTests({});
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const onSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const runExtraction = async (targetFile?: File, sexOverride?: string) => {
    const f = targetFile || file;
    if (!f) return;
    setLoading(true);
    setError(null);
    const effSex = sexOverride !== undefined ? sexOverride : patientSex;
    try {
      const res = await api.extractReport(f, effSex !== "unspecified" ? effSex : undefined);
      setResult(res);
      setEditedValues({});
      setConfirmedTests({});
    } catch (err: any) {
      setError(err?.message || "Failed to extract lab report.");
    } finally {
      setLoading(false);
    }
  };

  const loadSample = (sampleType: "lipid" | "metabolic" | "cbc") => {
    let content = "";
    let name = "";
    if (sampleType === "lipid") {
      name = "sample_lipid_panel.txt";
      content = `CLINICAL BIOCHEMISTRY REPORT - PATIENT: DOE, JOHN
Cholesterol Total: 245 mg/dL (Reference: <200)
HDL Cholesterol: 38 mg/dL (Reference: >40)
Triglycerides: 210 mg/dL (Reference: <150)
Blood Pressure: 142/92 mmHg
Fasting Glucose: 118 mg/dL
HbA1c: 6.4 %
BMI: 29.4
Age: 56
Sex: Male`;
    } else if (sampleType === "metabolic") {
      name = "sample_metabolic_si_units.txt";
      content = `METABOLIC LAB PROFILE (SI UNITS)
Total Cholesterol: 6.2 mmol/L (Normal: <5.2)
HDL: 0.95 mmol/L (Normal: >1.0)
Triglycerides: 2.4 mmol/L (Normal: <1.7)
Serum Creatinine: 92 umol/L (Normal: 60-110)
Blood Pressure: 135/88 mmHg
BMI: 31.2
Age: 62
Sex: Female`;
    } else {
      name = "sample_cbc_panel.txt";
      content = `COMPLETE BLOOD COUNT (CBC)
White Blood Cells (WBC): 6.8 x10^9/L (4.0-10.0)
Hemoglobin: 14.2 g/dL (13.0-17.0)
Platelet Count: 245 x10^9/L (150-450)
Fasting Glucose: 98 mg/dL
Systolic BP: 122 mmHg
Diastolic BP: 78 mmHg
Age: 48
Sex: Male`;
    }
    const sampleBlob = new Blob([content], { type: "text/plain" });
    const sampleF = new File([sampleBlob], name, { type: "text/plain" });
    handleFile(sampleF);
    runExtraction(sampleF);
  };

  const handleEditValue = (canonicalName: string, val: number) => {
    setEditedValues((prev) => ({ ...prev, [canonicalName]: val }));
  };

  // Convert extracted tests to risk engine features
  const getRiskFeatures = (items: LabItemSchema[]): Record<string, number> => {
    const map: Record<string, number> = {};
    for (const it of items) {
      const val = editedValues[it.canonical_name] !== undefined ? editedValues[it.canonical_name] : it.value;
      if (it.canonical_name === "total_cholesterol") map.total_cholesterol = val;
      else if (it.canonical_name === "hdl_cholesterol") map.hdl_cholesterol = val;
      else if (it.canonical_name === "triglycerides") map.triglycerides = val;
      else if (it.canonical_name === "ldl_cholesterol") map.ldl_cholesterol = val;
      else if (it.canonical_name === "fasting_glucose") map.fasting_glucose = val;
      else if (it.canonical_name === "hba1c") map.hba1c = val;
      else if (it.canonical_name === "systolic_bp") map.systolic_bp = val;
      else if (it.canonical_name === "diastolic_bp") map.diastolic_bp = val;
      else if (it.canonical_name === "bmi") map.bmi = val;
      else if (it.canonical_name === "serum_creatinine" || it.canonical_name === "creatinine") map.serum_creatinine = val;
      else if (it.canonical_name === "bun") map.bun = val;
      else if (it.canonical_name === "uric_acid") map.uric_acid = val;
      else if (it.canonical_name === "albumin") map.albumin = val;
      else if (it.canonical_name === "total_protein") map.total_protein = val;
      else if (it.canonical_name === "alt") map.alt = val;
      else if (it.canonical_name === "ast") map.ast = val;
      else if (it.canonical_name === "alk_phosphatase") map.alk_phosphatase = val;
      else if (it.canonical_name === "sodium") map.sodium = val;
      else if (it.canonical_name === "potassium") map.potassium = val;
      else if (it.canonical_name === "calcium") map.calcium = val;
      else if (it.canonical_name === "wbc_count") map.wbc_count = val;
      else if (it.canonical_name === "rbc_count") map.rbc_count = val;
      else if (it.canonical_name === "hemoglobin") map.hemoglobin = val;
      else if (it.canonical_name === "hematocrit") map.hematocrit = val;
      else if (it.canonical_name === "platelets") map.platelets = val;
      else if (it.canonical_name === "pulse") map.pulse_bpm = val;
      else if (it.canonical_name === "age") map.age = val;
    }
    if (patientSex === "male") map.sex_male = 1.0;
    else if (patientSex === "female") map.sex_male = 0.0;
    return map;
  };

  const riskFeatures = result ? getRiskFeatures(result.extracted_tests) : {};
  const abnormalCount = result
    ? result.extracted_tests.filter((t) => t.flag !== "NORMAL" && t.flag !== "UNKNOWN").length
    : 0;
  const hasLowConfidence = result
    ? result.extracted_tests.some((t) => t.confidence < 0.70)
    : false;
  const isScannedOrOcr = result ? result.extraction_method !== "pdf_text_layer" : false;
  const confirmedCount = result
    ? result.extracted_tests.filter((t) => confirmedTests[t.canonical_name]).length
    : 0;
  const allTestsConfirmed = result
    ? result.extracted_tests.length > 0 &&
      result.extracted_tests.every((t) => confirmedTests[t.canonical_name])
    : false;

  return (
    <div className="module-page space-y-6">
      <div className="page-heading">
        <div>
          <div className="eyebrow page-kicker">PHASE 3 · LAB EXTRACTION</div>
          <h1>Clinical Lab Report Reader</h1>
          <p>
            Upload clinical PDFs with native text layers or smartphone photos using local OCR.
            Normalizes SI to US units and auto-flags abnormal metabolic/lipid ranges.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload & Controls */}
        <div className="lg:col-span-1 space-y-4">
          <div className="content-panel">
            <h2 className="text-base font-semibold mb-3 flex items-center gap-2">
              <Upload size={18} className="text-teal-600" /> Upload Document
            </h2>

            <div
              className={`drop-zone border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer ${
                dragOver ? "border-teal-500 bg-teal-50/50" : "border-slate-300 hover:border-teal-400 bg-slate-50/50"
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              onClick={() => document.getElementById("report-file-input")?.click()}
            >
              <input
                id="report-file-input"
                type="file"
                className="hidden"
                accept=".pdf,.png,.jpg,.jpeg,.tiff,.webp,.txt"
                onChange={onSelect}
              />
              <FileText size={36} className="mx-auto text-slate-400 mb-2" />
              <div className="text-xs font-semibold text-slate-700">
                {file ? file.name : "Drag & drop PDF or photo here"}
              </div>
              <div className="text-[10px] text-slate-500 mt-1">
                Supports PDF (digital & scanned), PNG, JPG, TIFF, TXT up to 10 MB
              </div>
            </div>

            {file && (
              <div className="mt-4 flex items-center justify-between p-2.5 rounded-lg bg-teal-50 border border-teal-200">
                <div className="truncate text-xs font-medium text-teal-900 pr-2">
                  {file.name} ({(file.size / 1024).toFixed(1)} KB)
                </div>
                <button
                  className="button-primary text-xs px-3 py-1.5 shrink-0"
                  disabled={loading}
                  onClick={() => runExtraction()}
                >
                  {loading ? (
                    <>
                      <RefreshCw size={14} className="animate-spin" /> Extracting...
                    </>
                  ) : (
                    <>
                      <Sparkles size={14} /> Extract Labs
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Patient Biological Sex Selection */}
            <div className="mt-3">
              <label className="text-[11px] font-semibold text-slate-700 block mb-1">
                Patient Biological Sex (for sex-specific ranges):
              </label>
              <select
                className="w-full text-xs p-2 rounded-lg border border-slate-300 bg-white text-slate-800"
                value={patientSex}
                onChange={(e) => setPatientSex(e.target.value)}
              >
                <option value="unspecified">Unspecified (Auto-detect from report)</option>
                <option value="male">Male (Sex-specific Hb, Cr, HDL)</option>
                <option value="female">Female (Sex-specific Hb, Cr, HDL)</option>
              </select>
            </div>

            <div className="mt-4 pt-4 border-t border-slate-200">
              <div className="text-[11px] font-semibold text-slate-600 mb-2">Or test with clinical samples:</div>
              <div className="flex flex-col gap-2">
                <button
                  className="button-secondary text-left text-xs py-1.5 px-2.5 justify-start"
                  onClick={() => loadSample("lipid")}
                >
                  <FileText size={14} className="text-amber-600" /> Sample Lipid Panel (US Units)
                </button>
                <button
                  className="button-secondary text-left text-xs py-1.5 px-2.5 justify-start"
                  onClick={() => loadSample("metabolic")}
                >
                  <FileText size={14} className="text-blue-600" /> Sample Metabolic Panel (SI Units)
                </button>
                <button
                  className="button-secondary text-left text-xs py-1.5 px-2.5 justify-start"
                  onClick={() => loadSample("cbc")}
                >
                  <FileText size={14} className="text-purple-600" /> Sample Complete Blood Count
                </button>
              </div>
            </div>
          </div>

          {result && Object.keys(riskFeatures).length > 0 && (
            <div className="content-panel bg-gradient-to-br from-teal-50 to-emerald-50 border-teal-200">
              <div className="flex items-center gap-2 text-teal-900 font-semibold text-sm mb-1">
                <Activity size={18} className="text-teal-600" />
                Pipeline Integration
              </div>
              <p className="text-xs text-teal-800 mb-3">
                {Object.keys(riskFeatures).length} clinical biomarkers ready for NHANES chronic disease risk prediction.
              </p>

              {hasLowConfidence && (
                <div className="p-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-900 text-xs mb-3 flex items-start gap-2">
                  <AlertTriangle size={16} className="text-amber-600 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold">Low confidence OCR:</span> Some analytes have extraction confidence &lt; 0.70. You must verify and confirm values before transfer.
                  </div>
                </div>
              )}

              <div className="mb-3 p-3 rounded-lg bg-white/90 border border-teal-200 text-xs space-y-2">
                <div className="flex items-center justify-between text-[11px] font-semibold text-slate-700">
                  <span>Biomarker Verification:</span>
                  <span className={allTestsConfirmed ? "text-emerald-700 font-bold" : "text-amber-700 font-bold"}>
                    {confirmedCount} / {result.extracted_tests.length} confirmed
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${allTestsConfirmed ? "bg-emerald-500" : "bg-amber-500"}`}
                    style={{ width: `${result.extracted_tests.length > 0 ? (confirmedCount / result.extracted_tests.length) * 100 : 0}%` }}
                  />
                </div>
                {!allTestsConfirmed && (
                  <div className="text-[10px] text-amber-800 leading-tight">
                    Individual confirmation is mandatory: check the box next to each biomarker in the table below to verify values against source.
                  </div>
                )}
                {allTestsConfirmed && (
                  <div className="text-[10px] text-emerald-800 font-medium">
                    All biomarkers confirmed by user. Ready for transfer.
                  </div>
                )}
              </div>

              {isScannedOrOcr && !allTestsConfirmed && (
                <div className="p-3.5 rounded-xl bg-amber-50 border-2 border-amber-300 text-amber-950 text-xs space-y-1.5 shadow-sm">
                  <div className="flex items-center gap-2 font-bold text-amber-900 text-xs">
                    <AlertTriangle size={16} className="text-amber-700 shrink-0" />
                    <span>One-Click Transfer Disabled Until Verified</span>
                  </div>
                  <p className="text-[11px] text-amber-800 leading-relaxed font-normal">
                    Due to potential optical character misread risk on scanned reports (silent error risk &gt; 5%), automated one-click transfer is disabled until all biomarkers are verified. Please check each biomarker box in the table below to verify values against your original report.
                  </p>
                </div>
              )}

              {isScannedOrOcr && allTestsConfirmed && (
                <div className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-300 text-emerald-950 text-xs flex items-center gap-2">
                  <CheckCircle2 size={16} className="text-emerald-600 shrink-0" />
                  <span className="text-[11px] font-medium text-emerald-800">
                    Source Verified: All scanned biomarkers confirmed against source report. Transfer enabled.
                  </span>
                </div>
              )}

              <button
                className={`button-primary w-full text-xs py-2.5 flex items-center justify-center gap-2 text-white shadow ${
                  allTestsConfirmed
                    ? "bg-teal-600 hover:bg-teal-700 cursor-pointer"
                    : "bg-slate-400 cursor-not-allowed opacity-60"
                }`}
                disabled={!allTestsConfirmed}
                onClick={() => onTransferToRisk(riskFeatures)}
              >
                <span>Send to Chronic Risk Engine</span>
                <ArrowRight size={14} />
              </button>
            </div>
          )}
        </div>

        {/* Results Display */}
        <div className="lg:col-span-2 space-y-4">
          {error && (
            <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-xs flex items-center gap-3">
              <AlertTriangle size={20} className="shrink-0 text-red-600" />
              <div>{error}</div>
            </div>
          )}

          {!result && !loading && !error && (
            <div className="content-panel text-center py-16 text-slate-500">
              <Layers size={40} className="mx-auto text-slate-300 mb-3" />
              <div className="text-sm font-medium text-slate-700">No report extracted yet</div>
              <div className="text-xs text-slate-500 max-w-sm mx-auto mt-1">
                Upload a medical lab report or pick a preset sample on the left to extract clinical tests and reference ranges.
              </div>
            </div>
          )}

          {loading && (
            <div className="content-panel text-center py-16 text-slate-500">
              <RefreshCw size={36} className="mx-auto text-teal-600 animate-spin mb-3" />
              <div className="text-sm font-semibold text-slate-800">Processing document...</div>
              <div className="text-xs text-slate-500 mt-1">
                Extracting digital text layer & running optical character recognition.
              </div>
            </div>
          )}

          {result && (
            <>
              {/* Persistent warning for scanned / OCR inputs (Pass 5 F5) */}
              {isScannedOrOcr && (
                <div className="p-3.5 rounded-xl bg-amber-50 border-2 border-amber-300 text-amber-950 text-xs font-semibold flex items-center gap-2.5">
                  <AlertTriangle size={18} className="text-amber-700 shrink-0" />
                  <div>
                    <div className="text-sm font-bold">scanned input: values may be wrong</div>
                    <div className="text-[11px] font-normal text-amber-800 mt-0.5">
                      Benchmark evaluation: <em>synthetic set; real-world accuracy unknown</em>. Mandatory side-by-side verification required for each biomarker before chronic risk prediction.
                    </div>
                  </div>
                </div>
              )}

              {/* Summary Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3.5 rounded-xl bg-white border border-slate-200 shadow-sm">
                  <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Method</div>
                  <div className="text-sm font-bold text-slate-800 mt-1 capitalize">
                    {result.extraction_method.replace(/_/g, " ")}
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-white border border-slate-200 shadow-sm">
                  <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Tests Parsed</div>
                  <div className="text-sm font-bold text-teal-700 mt-1">{result.count} biomarkers</div>
                </div>
                <div className="p-3.5 rounded-xl bg-white border border-slate-200 shadow-sm">
                  <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Abnormal</div>
                  <div className={`text-sm font-bold mt-1 ${abnormalCount > 0 ? "text-amber-600" : "text-emerald-600"}`}>
                    {abnormalCount} flagged
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-white border border-slate-200 shadow-sm">
                  <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Risk Features</div>
                  <div className="text-sm font-bold text-blue-700 mt-1">
                    {Object.keys(riskFeatures).length} mapped
                  </div>
                </div>
              </div>

              {/* Extracted Tests Table */}
              <div className="content-panel overflow-hidden p-0 border border-slate-200">
                <div className="p-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-2">
                    <FileCheck size={16} className="text-teal-600" />
                    {isScannedOrOcr ? "Extracted Biomarkers (Unconfirmed Suggestions - OCR)" : "Extracted Laboratory Biomarkers"}
                  </h3>
                  <button
                    className="text-xs text-slate-600 hover:text-teal-700 flex items-center gap-1 font-medium"
                    onClick={() => setShowRawText(!showRawText)}
                  >
                    <Eye size={14} /> {showRawText ? "Hide Raw Text" : "View Raw Text"}
                  </button>
                </div>

                {showRawText && result.raw_text && (
                  <div className="p-4 bg-slate-900 text-slate-200 text-xs font-mono max-h-48 overflow-y-auto border-b border-slate-200">
                    <pre className="whitespace-pre-wrap">{result.raw_text}</pre>
                  </div>
                )}

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50/70 text-slate-600">
                        <th className="py-2.5 px-3 font-semibold text-center w-12">
                          <input
                            type="checkbox"
                            title={allTestsConfirmed ? "Uncheck all" : "Verify and confirm all biomarkers"}
                            className="rounded border-slate-300 text-teal-600 focus:ring-teal-500 cursor-pointer h-4 w-4"
                            checked={allTestsConfirmed}
                            onChange={(e) => {
                              const checked = e.target.checked;
                              const updated: Record<string, boolean> = {};
                              result.extracted_tests.forEach((t) => {
                                updated[t.canonical_name] = checked;
                              });
                              setConfirmedTests(updated);
                            }}
                          />
                        </th>
                        <th className="py-2.5 px-4 font-semibold">Biomarker / Source Line</th>
                        <th className="py-2.5 px-4 font-semibold">Reported Raw</th>
                        <th className="py-2.5 px-4 font-semibold">Standardized (US)</th>
                        <th className="py-2.5 px-4 font-semibold">Reference Range</th>
                        <th className="py-2.5 px-4 font-semibold">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {result.extracted_tests.map((test: LabItemSchema, idx: number) => (
                        <tr key={`${test.canonical_name}-${idx}`} className="hover:bg-slate-50/60 transition-colors">
                          <td className="py-2.5 px-3 text-center">
                            <input
                              type="checkbox"
                              className="rounded border-slate-300 text-teal-600 focus:ring-teal-500 cursor-pointer h-4 w-4"
                              checked={!!confirmedTests[test.canonical_name]}
                              onChange={(e) =>
                                setConfirmedTests((prev) => ({
                                  ...prev,
                                  [test.canonical_name]: e.target.checked,
                                }))
                              }
                            />
                          </td>
                          <td className="py-2.5 px-4">
                            <div className="font-semibold text-slate-800">{test.display_name}</div>
                            <div className="text-[10px] text-slate-600 font-mono bg-slate-100 px-1.5 py-0.5 rounded mt-0.5 inline-block">
                              Raw: &quot;{test.raw_name}&quot;
                            </div>
                          </td>
                          <td className="py-2.5 px-4 text-slate-700">
                            {test.raw_value} <span className="text-[10px] text-slate-500">{test.raw_unit}</span>
                          </td>
                          <td className="py-2.5 px-4 font-medium text-slate-900">
                            <input
                              type="number"
                              step="any"
                              className="w-20 px-1.5 py-0.5 text-xs font-semibold rounded border border-slate-300 bg-white"
                              value={
                                editedValues[test.canonical_name] !== undefined
                                  ? editedValues[test.canonical_name]
                                  : test.value
                              }
                              onChange={(e) => {
                                const v = parseFloat(e.target.value);
                                if (!isNaN(v)) handleEditValue(test.canonical_name, v);
                              }}
                            />{" "}
                            <span className="text-[10px] text-slate-500">{test.unit}</span>
                          </td>
                          <td className="py-2.5 px-4 text-slate-500">
                            {test.reference_range} {test.unit}
                          </td>
                          <td className="py-2.5 px-4">
                            <span
                              className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold tracking-wide ${
                                test.flag === "NORMAL"
                                  ? "bg-emerald-100 text-emerald-800"
                                  : test.flag === "HIGH"
                                  ? "bg-amber-100 text-amber-800"
                                  : test.flag === "LOW"
                                  ? "bg-blue-100 text-blue-800"
                                  : "bg-slate-100 text-slate-700"
                              }`}
                            >
                              {test.flag}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
