import { useState, type ChangeEvent, type DragEvent } from "react";
import {
  Image as ImageIcon,
  Upload,
  AlertTriangle,
  ShieldAlert,
  Sparkles,
  RefreshCw,
  Layers,
  CheckCircle,
  Sliders,
} from "lucide-react";
import { api } from "../api";
import type { ImagePredictionResponse } from "../types";

const MODALITIES = [
  {
    id: "chest_xray",
    name: "Chest X-Ray",
    caption: "Pneumonia Screening (Binary)",
    classes: "Normal vs Pneumonia",
    testAuc: "97.57%",
  },
  {
    id: "skin_lesion",
    name: "Dermatoscopy",
    caption: "Skin Lesions (7 Classes)",
    classes: "Melanoma, Basal Cell, Keratosis...",
    testAuc: "93.70%",
  },
  {
    id: "retina",
    name: "Retinal Fundus",
    caption: "Diabetic Retinopathy (5 Grades)",
    classes: "Grades 0 to 4 (DR Severity)",
    testAuc: "74.38%",
  },
  {
    id: "blood_smear",
    name: "Blood Smear",
    caption: "Hematopoietic Cells (8 Types)",
    classes: "Neutrophil, Basophil, Platelet...",
    testAuc: "99.80%",
  },
  {
    id: "chest_multilabel",
    name: "Chest 14-Condition",
    caption: "Multi-Label Thoracic Pathology",
    classes: "Cardiomegaly, Effusion, Mass...",
    testAuc: "77.30%",
  },
];

export function ImageScreeningPage() {
  const [selectedModality, setSelectedModality] = useState("chest_xray");
  const [oodThreshold, setOodThreshold] = useState(0.50);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ImagePredictionResponse | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleImage = (file: File) => {
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

  const runPrediction = async (customFile?: File) => {
    const f = customFile || imageFile;
    if (!f) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.predictImage(f, selectedModality, oodThreshold);
      setResult(res);
    } catch (err: any) {
      setError(err?.message || "Medical image screening failed.");
    } finally {
      setLoading(false);
    }
  };

  // Generate synthetic clinical test images (canvas base64)
  const loadSyntheticImage = (type: "chest" | "skin" | "retina" | "noise") => {
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
        const f = new File([blob], `synthetic_${type}.png`, { type: "image/png" });
        handleImage(f);
        runPrediction(f);
      }
    });
  };

  return (
    <div className="module-page space-y-6">
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
            onClick={() => {
              setSelectedModality(mod.id);
              setResult(null);
            }}
          >
            <div className="flex items-center justify-between gap-3">
              <span className="text-xs font-bold text-slate-900">{mod.name}</span>
              <span className="text-[10px] font-semibold text-teal-700 bg-teal-100/60 px-1.5 py-0.5 rounded">
                AUC {mod.testAuc}
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
              className="button-primary text-xs py-1.5 px-3 flex items-center gap-1"
              disabled={loading || !imageFile}
              onClick={() => runPrediction()}
            >
              {loading ? <RefreshCw size={12} className="animate-spin" /> : <Sparkles size={12} />}
              Screen Image
            </button>
          </div>

          {/* OOD Threshold Slider */}
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
            <div className="flex items-center justify-between text-xs font-medium text-slate-700 mb-1">
              <span className="flex items-center gap-1">
                <Sliders size={12} className="text-slate-500" /> OOD Guard Threshold:
              </span>
              <strong className="text-teal-700">{(oodThreshold * 100).toFixed(0)}%</strong>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.95"
              step="0.05"
              value={oodThreshold}
              onChange={(e) => setOodThreshold(Number(e.target.value))}
              className="w-full accent-teal-600"
            />
            <div className="text-[10px] text-slate-500 mt-1">
              Rejects non-medical inputs when prediction confidence falls below this minimum threshold.
            </div>
          </div>

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
            <div className="text-[11px] font-semibold text-slate-600 mb-2">Or test with synthetic patterns:</div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <button
                className="button-secondary py-1.5 px-2 text-left justify-start text-[11px]"
                onClick={() => {
                  setSelectedModality("chest_xray");
                  loadSyntheticImage("chest");
                }}
              >
                Synthetic Chest X-Ray
              </button>
              <button
                className="button-secondary py-1.5 px-2 text-left justify-start text-[11px]"
                onClick={() => {
                  setSelectedModality("skin_lesion");
                  loadSyntheticImage("skin");
                }}
              >
                Synthetic Skin Lesion
              </button>
              <button
                className="button-secondary py-1.5 px-2 text-left justify-start text-[11px]"
                onClick={() => {
                  setSelectedModality("retina");
                  loadSyntheticImage("retina");
                }}
              >
                Synthetic Retina Scan
              </button>
              <button
                className="button-secondary py-1.5 px-2 text-left justify-start text-[11px] border-amber-300 bg-amber-50/50"
                onClick={() => loadSyntheticImage("noise")}
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
                Select a modality above, upload an image or synthetic preset, and test the OOD confidence guard.
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
                      Image Recognized & Classified
                    </div>
                    {result.confidence && (
                      <span className="text-xs font-bold bg-emerald-200/70 text-emerald-900 px-2 py-0.5 rounded-full">
                        {(result.confidence * 100).toFixed(1)}% Confidence
                      </span>
                    )}
                  </div>

                  {result.predicted_class && (
                    <div className="mt-2">
                      <div className="text-[11px] text-emerald-700 font-medium">Predicted Screening Finding:</div>
                      <div className="text-2xl font-black text-emerald-950 mt-0.5">
                        {result.predicted_class.toUpperCase()}
                      </div>
                    </div>
                  )}

                  {/* Multi-label active conditions */}
                  {result.multilabel && result.active_conditions && (
                    <div className="mt-3">
                      <div className="text-[11px] font-semibold text-emerald-800 mb-1.5">
                        Positive Findings (Probability &ge; 50%):
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
                        <div className="text-xs text-emerald-700 italic">No thoracic abnormalities detected above 50% probability.</div>
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

              {/* Disclaimer */}
              <div className="p-3.5 rounded-xl bg-slate-100 border border-slate-200 text-[11px] text-slate-600 leading-relaxed flex items-start gap-2.5">
                <ShieldAlert size={18} className="shrink-0 text-amber-600 mt-0.5" />
                <div>
                  <strong>Biomedical Vision Disclaimer:</strong> {result.disclaimer}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
