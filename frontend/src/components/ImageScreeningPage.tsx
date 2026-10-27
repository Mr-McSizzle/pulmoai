import { SavedToTimeline, usePatientWorkspace } from "./PatientWorkspace";
import { useEffect, useState, type ChangeEvent, type DragEvent } from "react";
import {
  Image as ImageIcon,
  Upload,
  AlertTriangle,
  ShieldAlert,
  Sparkles,
  RefreshCw,
  Layers,
  CheckCircle,
} from "lucide-react";
import { api } from "../api";
import type { DataOrigin, ImagePredictionResponse } from "../types";

const MODALITIES = [
  {
    id: "chest_xray",
    name: "Chest Radiography (CXR)",
    caption: "Pneumonia Deep Screening",
    classes: "Normal vs Pneumonia (ResNet-18)",
  },
  {
    id: "skin_lesion",
    name: "Dermatoscopy",
    caption: "Skin Lesions (7 Classes)",
    classes: "Melanoma, Basal Cell, Keratosis...",
  },
  {
    id: "retina",
    name: "Retinal Fundus Photography",
    caption: "Diabetic Retinopathy Screening",
    classes: "Non-Referable vs Referable DR",
  },
  {
    id: "blood_smear",
    name: "Blood Smear",
    caption: "Hematopoietic Cells (8 Types)",
    classes: "Neutrophil, Basophil, Platelet...",
  },
  {
    id: "chest_multilabel",
    name: "Chest 14-Condition",
    caption: "Multi-Label Thoracic Pathology",
    classes: "Cardiomegaly, Effusion, Mass...",
  },
];

export function ImageScreeningPage({ onOpenTimeline }: { onOpenTimeline: () => void }) {
  const workspace = usePatientWorkspace();
  useEffect(() => { setResult(null); setImageFile(null); setImagePreview(null); }, [workspace.patientId, workspace.observedOn]);
  const [dataOrigin, setDataOrigin] = useState<DataOrigin>("user_supplied_unverified");
  const [pediatric, setPediatric] = useState(false);
  const [selectedModality, setSelectedModality] = useState("chest_xray");
  const oodThreshold = 0.50;
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ImagePredictionResponse | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleImage = (file: File, origin: DataOrigin = "user_supplied_unverified") => {
    if (loading) return;
    setDataOrigin(origin);
    setImageFile(file);
    setError(null);
    setResult(null);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImage(e.dataTransfer.files[0]);
    }
  };

  const onSelect = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleImage(e.target.files[0]);
    }
  };

  const runPrediction = async (customFile?: File, originOverride?: DataOrigin) => {
    const f = customFile || imageFile;
    if (!f) return;
    if (!workspace.context || !workspace.seriesLabel.trim()) { setError("Select a patient/date and enter the acquisition series before screening."); return; }
    const done = workspace.beginAnalysis();
    setLoading(true);
    setError(null);
    try {
      const res = await api.predictImage(f, selectedModality, oodThreshold, pediatric, originOverride ?? dataOrigin, workspace.context);
      setResult(res); workspace.analysisSaved();
    } catch (err: any) {
      setError(err?.message || "Medical image screening failed.");
    } finally {
      setLoading(false); done();
    }
  };

  // Generate labelled input-gate patterns for preview (canvas base64)
  const loadExampleImage = (type: "chest" | "skin" | "retina" | "noise") => {
    if (loading) return;
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (type === "chest") {
      // Simulate chest radiograph gradient
      const grad = ctx.createRadialGradient(32, 32, 5, 32, 32, 32);
      grad.addColorStop(0, "#222");
      grad.addColorStop(0.5, "#888");
      grad.addColorStop(1, "#111");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 64, 64);
    } else if (type === "skin") {
      // Skin tone with pigmented lesion center
      ctx.fillStyle = "#dfb49d";
      ctx.fillRect(0, 0, 64, 64);
      ctx.fillStyle = "#5c2b21";
      ctx.beginPath();
      ctx.arc(32, 32, 14, 0, Math.PI * 2);
      ctx.fill();
    } else if (type === "retina") {
      // Retinal fundus circle
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, 64, 64);
      ctx.fillStyle = "#bf4a24";
      ctx.beginPath();
      ctx.arc(32, 32, 28, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#e0a348";
      ctx.beginPath();
      ctx.arc(24, 32, 5, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // Random noisy pattern to test OOD rejection
      const imgData = ctx.createImageData(64, 64);
      for (let i = 0; i < imgData.data.length; i += 4) {
        const val = Math.floor(Math.random() * 256);
        imgData.data[i] = val;
        imgData.data[i + 1] = val;
        imgData.data[i + 2] = val;
        imgData.data[i + 3] = 255;
      }
      ctx.putImageData(imgData, 0, 0);
    }

    canvas.toBlob((blob) => {
      if (blob) {
        const f = new File([blob], `generated-example-${type}.png`, { type: "image/png" });
        handleImage(f, "generated_example");
        // Preview first; screening requires an explicit action after state updates.
      }
    });
  };

  return (
    <div className="module-page space-y-6">
      <div className="module-timeline-link"><div><strong>Compare imaging observations over time</strong><p>Batch uploads, original images and compatible model score trends in one patient workspace.</p></div><button className="button-secondary" onClick={onOpenTimeline}>Open patient timelines</button></div>
      <div className="page-heading">
        <div>
          <div className="eyebrow page-kicker">PHASE 4 · MEDICAL IMAGE SCREENING</div>
          <h1>Biomedical Image Screening Engine</h1>
          <p>
            Standardized 64×64 MedMNIST vision classification with ResNet-18 architectures and
            built-in <strong>Out-of-Distribution (OOD) confidence guard</strong>.
          </p>
        </div>
      </div>

      {/* Modality Chips */}
      <div className="flex flex-wrap gap-2.5">
        {MODALITIES.map((mod) => (
          <button
            key={mod.id}
            className={`p-3 rounded-xl border text-left transition-all ${
              selectedModality === mod.id
                ? "bg-teal-50 border-teal-500 shadow-sm"
                : "bg-white border-slate-200 hover:border-teal-300"
            }`}
            disabled={loading}
            onClick={() => {
              setSelectedModality(mod.id);
              setResult(null);
            }}
          >
            <div className="flex items-center justify-between gap-3">
              <span className="text-xs font-bold text-slate-900">{mod.name}</span>
              <span className="text-[10px] font-semibold text-teal-700 bg-teal-100/60 px-1.5 py-0.5 rounded">
                Research model
              </span>
            </div>
            <div className="text-[10px] text-slate-500 mt-1">{mod.caption}</div>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Upload & Controls */}
        <div className="lg:col-span-5 content-panel space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-200">
            <h2 className="text-sm font-semibold flex items-center gap-2 text-slate-800">
              <ImageIcon size={16} className="text-teal-600" /> Image Input & OOD Guard
            </h2>
            <button
              className="button-primary text-xs py-1.5 px-3 flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={loading || !imageFile || !workspace.context || !workspace.seriesLabel.trim() || (selectedModality === "chest_xray" && !pediatric)}
              onClick={() => runPrediction()}
            >
              {loading ? <RefreshCw size={12} className="animate-spin" /> : <Sparkles size={12} />}
              Screen Image
            </button>
          </div>

          {selectedModality === "chest_xray" && <label className="timeline-confirm"><input type="checkbox" checked={pediatric} onChange={e => setPediatric(e.target.checked)} /> I confirm this image is from a child aged 1–5, matching the training cohort.</label>}
          <label className="acquisition-field">Acquisition series (same site and view)<input aria-label="Acquisition series" maxLength={80} value={workspace.seriesLabel} disabled={loading} onChange={e => workspace.setSeriesLabel(e.target.value)} placeholder="e.g. left forearm · dermatoscopy" /></label>
          {/* Upload Drop Zone */}
          <div
            className={`drop-zone border-2 border-dashed rounded-xl p-5 text-center transition-all cursor-pointer ${
              dragOver ? "border-teal-500 bg-teal-50/50" : "border-slate-300 hover:border-teal-400 bg-slate-50/50"
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => document.getElementById("image-file-input")?.click()}
          >
            <input
              id="image-file-input"
              type="file"
              disabled={loading}
              className="hidden"
              accept=".png,.jpg,.jpeg,.tiff,.bmp,.webp"
              onChange={onSelect}
            />

            {imagePreview ? (
              <div className="space-y-2">
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="w-24 h-24 object-cover mx-auto rounded-lg border border-slate-300 shadow-sm"
                />
                <div className="text-xs font-medium text-slate-700 truncate">{imageFile?.name}</div>
                <div className="text-[10px] text-teal-600 font-semibold">Click to replace image</div>
              </div>
            ) : (
              <div>
                <Upload size={32} className="mx-auto text-slate-400 mb-2" />
                <div className="text-xs font-semibold text-slate-700">Drag & drop medical image here</div>
                <div className="text-[10px] text-slate-500 mt-1">PNG, JPG, TIFF, WEBP (resized to 64×64)</div>
              </div>
            )}
          </div>

          {/* Test Presets */}
          <div className="pt-2 border-t border-slate-200">
            <div className="text-[11px] font-semibold text-slate-600 mb-2">Generated patterns · input-gate examples:</div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <button
                className="button-secondary py-1.5 px-2 text-left justify-start text-[11px]"
                onClick={() => {
                  setSelectedModality("chest_xray");
                  loadExampleImage("chest");
                }}
              >
                Chest-like example
              </button>
              <button
                className="button-secondary py-1.5 px-2 text-left justify-start text-[11px]"
                onClick={() => {
                  setSelectedModality("skin_lesion");
                  loadExampleImage("skin");
                }}
              >
                Skin-like example
              </button>
              <button
                className="button-secondary py-1.5 px-2 text-left justify-start text-[11px]"
                onClick={() => {
                  setSelectedModality("retina");
                  loadExampleImage("retina");
                }}
              >
                Retina-like example
              </button>
              <button
                className="button-secondary py-1.5 px-2 text-left justify-start text-[11px] border-amber-300 bg-amber-50/50"
                onClick={() => loadExampleImage("noise")}
              >
                Random Noise Pattern (OOD Test)
              </button>
            </div>
          </div>
        </div>

        {/* Inference Results Display */}
        <div className="lg:col-span-7 space-y-4">
          {error && (
            <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-xs">
              {error}
            </div>
          )}

          {!result && !loading && !error && (
            <div className="content-panel text-center py-16 text-slate-500">
              <ImageIcon size={40} className="mx-auto text-slate-300 mb-3" />
              <div className="text-sm font-medium text-slate-700">Ready for image screening</div>
              <div className="text-xs text-slate-500 max-w-sm mx-auto mt-1">
                Select a modality above, upload an image or a clearly marked generated pattern, and test the OOD confidence guard.
              </div>
            </div>
          )}

          {loading && (
            <div className="content-panel text-center py-16 text-slate-500">
              <RefreshCw size={36} className="mx-auto text-teal-600 animate-spin mb-3" />
              <div className="text-sm font-semibold text-slate-800">Processing medical image...</div>
              <div className="text-xs text-slate-500 mt-1">
                Running ResNet-18 forward pass on Apple Silicon MPS / CPU.
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <SavedToTimeline patientId={result.patient_id} observedOn={result.observed_on} />
              <p className="data-origin-badge">{result.data_origin === "generated_example" ? "Generated pattern · not a medical scan" : "Uploaded source · authenticity unverified"}</p>
              {/* Benchmark model clinical notice banner */}
              <div className="p-3 rounded-xl bg-blue-50 border border-blue-200 text-blue-900 text-xs font-semibold flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <ShieldAlert size={15} className="text-blue-600 shrink-0" />
                  Benchmark model, not for clinical use.
                </span>
                <span className="text-[10px] font-mono bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
                  Research Evaluation Only
                </span>
              </div>

              {/* OOD Rejection Status Banner */}
              {!result.recognized ? (
                <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 space-y-2">
                  <div className="flex items-center gap-2 font-bold text-xs uppercase tracking-wider text-amber-800">
                    <AlertTriangle size={18} className="text-amber-600" />
                    Out-of-Distribution / Low Confidence Warning
                  </div>
                  <p className="text-xs leading-relaxed text-amber-800">
                    {result.message ||
                      "The image does not display expected anatomical features for this modality. Confidence did not meet the guard threshold."}
                  </p>
                  <div className="text-[11px] font-mono text-amber-700 bg-amber-100/60 p-2 rounded">
                    Max Softmax Confidence: {result.max_confidence ? `${(result.max_confidence * 100).toFixed(1)}%` : "N/A"} | Threshold Required: {((result.threshold_applied || oodThreshold) * 100).toFixed(0)}%
                  </div>
                </div>
              ) : (
                <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-900">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 font-bold text-xs uppercase tracking-wider text-emerald-800">
                      <CheckCircle size={18} className="text-emerald-600" />
                      Image Processed
                    </div>
                    {result.confidence && (
                      <span className="text-xs font-bold bg-emerald-200/70 text-emerald-900 px-2 py-0.5 rounded-full">
                        {(result.confidence * 100).toFixed(1)}% model score
                      </span>
                    )}
                  </div>

                  {/* Normal result with predicted class */}
                  {result.predicted_class && (
                    <div className="mt-2">
                      <div className="text-[11px] text-emerald-700 font-medium">Predicted Screening Finding:</div>
                      <div className="text-2xl font-black text-emerald-950 mt-0.5">
                        {result.predicted_class.toUpperCase()}
                      </div>
                    </div>
                  )}

                  {/* Retina experimental staging disabled banner */}
                  {(selectedModality === "retina" || result.image_type === "retina" || result.staging_notice) && (
                    <div className="mt-2 p-3 rounded-lg bg-amber-100/80 border border-amber-300 text-amber-950 text-xs">
                      <strong>Experimental Model (52.8% Accuracy):</strong> Severity staging is disabled because clinical staging accuracy is insufficient. Only class probabilities are shown below for benchmark evaluation.
                    </div>
                  )}

                  {/* Multi-label active conditions */}
                  {result.multilabel && result.active_conditions && (
                    <div className="mt-3">
                      <div className="text-[11px] font-semibold text-emerald-800 mb-1.5">
                        Labels above the model score threshold (≥ 50%):
                      </div>
                      {result.active_conditions.length > 0 ? (
                        <div className="flex flex-wrap gap-1.5">
                          {result.active_conditions.map((cond) => (
                            <span
                              key={cond}
                              className="text-xs font-bold bg-emerald-200 text-emerald-900 px-2.5 py-1 rounded"
                            >
                              {cond}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <div className="text-xs text-emerald-700 italic">No model label crossed the 50% threshold. This does not rule out abnormalities.</div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Class Probability Distribution */}
              {(result.class_probabilities || result.probabilities) && (
                <div className="content-panel border border-slate-200 space-y-3">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-600 flex items-center gap-1.5">
                    <Layers size={14} className="text-teal-600" /> Class Probability Distribution
                  </h3>
                  <div className="space-y-2">
                    {Object.entries(result.class_probabilities || result.probabilities || {}).map(
                      ([lbl, prob]: [string, number]) => (
                        <div key={lbl} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-medium text-slate-700">{lbl}</span>
                            <span className="font-mono text-slate-500">{(prob * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-teal-500 rounded-full transition-all duration-300"
                              style={{ width: `${Math.max(1, prob * 100)}%` }}
                            />
                          </div>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}

              {/* Empirical OOD Benchmark Information */}
              {result.ood_methodology_statement && (
                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
                  <div className="flex items-center justify-between text-xs font-bold text-slate-800">
                    <span className="flex items-center gap-1.5">
                      <ShieldAlert size={14} className="text-teal-600" /> Empirical OOD Rejection Profile
                    </span>
                    <div className="flex gap-1.5">
                      {result.empirical_false_accept_rate !== undefined && result.empirical_false_accept_rate !== null && (
                        <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-amber-100 text-amber-900 border border-amber-200">
                          Med FAR: {(result.empirical_false_accept_rate * 100).toFixed(1)}%
                        </span>
                      )}
                      {result.cifar10_false_accept_rate !== undefined && result.cifar10_false_accept_rate !== null && (
                        <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-red-100 text-red-900 border border-red-200">
                          Photo FAR: {(result.cifar10_false_accept_rate * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                  </div>
                  <p className="text-[11px] text-slate-600 leading-relaxed">
                    {result.ood_methodology_statement}
                  </p>
                  {result.ood_benchmark?.tested_against_cross_modalities && (
                    <div className="text-[10px] text-slate-500 pt-1 border-t border-slate-200">
                      Cross-modality test sets: {result.ood_benchmark.tested_against_cross_modalities.join(", ")}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
