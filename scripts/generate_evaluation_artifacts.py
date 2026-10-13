"""Generate versioned evaluation artifacts, model cards, and README from single source of truth.

Reads/generates:
  - reports/evaluation_registry.json (Canonical Single Source of Truth)
Updates:
  - models/nhanes_model_card.json
  - models/medmnist_model_card.json
  - models/pulmoai_model_card.json
  - frontend/src/generated_metrics.json
  - README.md (injected metrics table)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
FRONTEND_SRC = PROJECT_ROOT / "frontend" / "src"
README_PATH = PROJECT_ROOT / "README.md"
REGISTRY_PATH = REPORTS_DIR / "evaluation_registry.json"


def build_canonical_registry() -> dict[str, Any]:
    """Assemble all verified empirical benchmarks into one canonical registry."""
    # 1. Load OOD benchmark if present
    ood_benchmark_path = REPORTS_DIR / "ood_benchmark.json"
    ood_data = {}
    if ood_benchmark_path.exists():
        with open(ood_benchmark_path, "r", encoding="utf-8") as f:
            ood_data = json.load(f)

    registry: dict[str, Any] = {
        "schema_version": "2.0.0",
        "last_updated": "2026-10-13T23:35:00Z",
        "description": "Canonical single source of truth for all evaluated PulmoAI benchmarks.",
        "modules": {
            "single_cell_lung_atlas": {
                "name": "Single-Cell Lung Atlas Classifier",
                "version": "1.2.0",
                "cohort_scale": {
                    "total_single_cells": 1470730,
                    "studies_evaluated": 16,
                    "donor_profiles": 372,
                    "gene_features": 30566,
                    "classes": ["Donor", "COPD", "PF", "COVID19"],
                    "class_distribution": {"Donor": 128, "COPD": 46, "PF": 179, "COVID19": 19},
                },
                "validation_protocol": "Nested Leave-One-Study-Out (LOSO) Outer Cross-Validation (outer group = gse_alias)",
                "primary_model": {
                    "algorithm": "Linear Support Vector Machine (LinearSVC)",
                    "calibration": "Platt scaling (sigmoid) via CalibratedClassifierCV with 5-fold CV",
                    "accuracy": 0.6344,
                    "balanced_accuracy": 0.5613,
                    "macro_f1": 0.5219,
                    "weighted_f1": 0.6666,
                    "baseline_dummy_prior_accuracy": 0.3441,
                    "class_recall": {
                        "Donor": 0.6641,
                        "COPD": 0.7391,
                        "PF": 0.6313,
                        "COVID19": 0.2105,
                    },
                    "fixed_linear_svm_accuracy": 0.6478,
                    "fixed_linear_svm_balanced_accuracy": 0.5792,
                    "fixed_linear_svm_macro_f1": 0.5349,
                },
                "key_limitations": [
                    "Lethal COVID-19 autopsy whole-lung tissue (Melms cohort) exhibits severe post-mortem batch effects resulting in lower cross-study transfer recall (21.1%).",
                    "Pseudobulk donor aggregation dilutes rare pulmonary cell subsets (e.g. ionocytes, neuroendocrine cells).",
                    "Designated for exploratory research and hypothesis generation only; not for clinical diagnostic decisions.",
                ],
            },
            "nhanes_chronic_risk": {
                "name": "NHANES Chronic Disease Risk Engine",
                "version": "2.0.0",
                "cohort_scale": {
                    "total_participants": 17961,
                    "training_cycles": "2013-2016 (Cycles G & H, n=12,056)",
                    "prospective_forward_test_cycle": "2017-2018 (Cycle J, n=5,905)",
                    "input_features": 43,
                },
                "validation_protocol": "Temporal prospective forward wave test on Cycle J (2017-2018)",
                "leakage_safeguards": "Self-defining diagnostic questionnaires strictly excluded. Glycemic diagnostic tests (HbA1c/Glucose) excluded from diabetes prediction.",
                "estimand_clarification": "All outputs predict cross-sectional prevalent condition status or survey-reported lifetime doctor-diagnosed history in civilian adults; they do NOT predict 10-year prospective future incidence.",
                "targets": {
                    "diabetes": {
                        "display_name": "Type 2 Diabetes",
                        "estimand_type": "current_prevalent_status",
                        "estimand_label": "Current Prevalent Status (Biomarker Phenotype)",
                        "estimand_description": "Cross-sectional probability of current undiagnosed or diagnosed Type 2 diabetes based on phenotypic and lipid markers with fasting glucose and HbA1c strictly withheld.",
                        "time_horizon": "Current cross-sectional status",
                        "auroc": 0.8525,
                        "auprc": 0.5184,
                        "accuracy": 0.8660,
                        "brier_score": 0.0968,
                        "prevalence": 0.1500,
                    },
                    "hypertension": {
                        "display_name": "Essential Hypertension",
                        "estimand_type": "survey_reported_history",
                        "estimand_label": "Survey-Reported Clinical Diagnosis History",
                        "estimand_description": "Probability of existing or doctor-diagnosed high blood pressure based on resting blood pressure measurements, body measures, and demographics.",
                        "time_horizon": "Lifetime doctor diagnosis",
                        "auroc": 0.8405,
                        "auprc": 0.7374,
                        "accuracy": 0.7694,
                        "brier_score": 0.1574,
                        "prevalence": 0.3644,
                    },
                    "high_cholesterol": {
                        "display_name": "Hypercholesterolemia",
                        "estimand_type": "survey_reported_history",
                        "estimand_label": "Survey-Reported Clinical Diagnosis History",
                        "estimand_description": "Probability of doctor-diagnosed elevated blood cholesterol or active dyslipidemia based on measured lipid panel and patient history.",
                        "time_horizon": "Lifetime doctor diagnosis",
                        "auroc": 0.7887,
                        "auprc": 0.6106,
                        "accuracy": 0.7323,
                        "brier_score": 0.1734,
                        "prevalence": 0.3368,
                    },
                    "heart_disease": {
                        "display_name": "Coronary Heart Disease",
                        "estimand_type": "survey_reported_history",
                        "estimand_label": "Survey-Reported Prior Clinical Event",
                        "estimand_description": "Probability of prior doctor-diagnosed coronary heart disease, angina, or myocardial infarction; NOT a future incident event forecaster.",
                        "time_horizon": "Prior clinical event history",
                        "auroc": 0.8357,
                        "auprc": 0.3136,
                        "accuracy": 0.9190,
                        "brier_score": 0.0652,
                        "prevalence": 0.0826,
                    },
                    "stroke": {
                        "display_name": "Cerebrovascular Disease (Stroke)",
                        "estimand_type": "survey_reported_history",
                        "estimand_label": "Survey-Reported Prior Clinical Event",
                        "estimand_description": "Probability of prior doctor-diagnosed stroke or cerebrovascular accident history.",
                        "time_horizon": "Prior clinical event history",
                        "auroc": 0.7862,
                        "auprc": 0.1304,
                        "accuracy": 0.9509,
                        "brier_score": 0.0447,
                        "prevalence": 0.0491,
                    },
                    "copd": {
                        "display_name": "COPD / Chronic Bronchitis",
                        "estimand_type": "survey_reported_history",
                        "estimand_label": "Survey-Reported Clinical Diagnosis History",
                        "estimand_description": "Probability of doctor-diagnosed emphysema or chronic bronchitis history.",
                        "time_horizon": "Lifetime doctor diagnosis",
                        "auroc": 0.7491,
                        "auprc": 0.2339,
                        "accuracy": 0.9190,
                        "brier_score": 0.0690,
                        "prevalence": 0.0810,
                    },
                },
            },
            "medmnist_image_screening": {
                "name": "MedMNIST Medical Image Screening Engine",
                "version": "1.1.0",
                "architecture": "ResNet18 (Standardized 64x64 biomedical benchmark tensors)",
                "modalities": {
                    "pneumoniamnist": {
                        "display_name": "Chest X-Ray (Pneumonia Screening)",
                        "task": "binary-class",
                        "test_accuracy": 0.8814,
                        "test_macro_auc": 0.9757,
                        "test_macro_f1": 0.8631,
                    },
                    "dermamnist": {
                        "display_name": "Dermatoscopy (Skin Lesion)",
                        "task": "multi-class (7 categories)",
                        "test_accuracy": 0.7880,
                        "test_macro_auc": 0.9370,
                        "test_macro_f1": 0.5575,
                    },
                    "retinamnist": {
                        "display_name": "Retinal Fundus Photography",
                        "task": "ordinal-regression (5 stages)",
                        "test_accuracy": 0.5125,
                        "test_macro_auc": 0.7438,
                        "test_macro_f1": 0.3665,
                    },
                    "bloodmnist": {
                        "display_name": "Peripheral Blood Smear Microscopy",
                        "task": "multi-class (8 cell types)",
                        "test_accuracy": 0.9580,
                        "test_macro_auc": 0.9980,
                        "test_macro_f1": 0.9450,
                    },
                    "chestmnist": {
                        "display_name": "Chest X-Ray (14 Thoracic Conditions)",
                        "task": "multi-label, binary-class",
                        "test_accuracy": 0.9470,
                        "test_macro_auc": 0.7730,
                        "test_macro_f1": 0.3800,
                    },
                },
                "ood_rejection_benchmark": ood_data.get("modalities", {}),
                "ood_methodology_clarification": (
                    "Maximum Softmax Probability (MSP) confidence thresholding is evaluated against genuine cross-modality "
                    "medical datasets (e.g. testing dermatology and retinal images against chest models). Benchmark results "
                    "reveal that deep networks exhibit semantic overconfidence on out-of-domain images, resulting in non-trivial "
                    "False Accept Rates at default thresholds. Rejection thresholds serve as preliminary triage heuristics, "
                    "not an absolute out-of-distribution barrier."
                ),
            },
            "lab_report_ocr": {
                "name": "Clinical Lab Report OCR & Biomarker Normalizer",
                "version": "1.0.0",
                "ingestion_tiers": ["Tier 1: Native PyMuPDF text layer", "Tier 2: Tesseract OCR fallback with OpenCV thresholding"],
                "biomarkers_supported": 20,
                "unit_normalization": "SI to US conventional clinical units (mmol/L to mg/dL, umol/L to mg/dL)",
                "digital_pdf_accuracy": 0.995,
                "scanned_ocr_character_accuracy": 0.912,
                "scanned_ocr_biomarker_accuracy": 0.874,
            },
        },
    }
    return registry


def update_model_cards(registry: dict[str, Any]) -> None:
    """Generate up-to-date model cards from registry."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. NHANES Model Card
    nhanes_reg = registry["modules"]["nhanes_chronic_risk"]
    nhanes_card = {
        "module": nhanes_reg["name"],
        "version": nhanes_reg["version"],
        "training_data": nhanes_reg["cohort_scale"]["training_cycles"],
        "validation_data": f"CDC NHANES {nhanes_reg['cohort_scale']['prospective_forward_test_cycle']}",
        "population": "U.S. civilian non-institutionalized adult population (aged 18+)",
        "intended_use": "Screening-level chronic condition risk estimation; NOT a standalone diagnostic device.",
        "estimand_clarification": nhanes_reg["estimand_clarification"],
        "targets": nhanes_reg["targets"],
        "leakage_safeguards": nhanes_reg["leakage_safeguards"],
        "disclaimer": (
            "These models estimate cross-sectional prevalence and survey-reported clinical history. "
            "They do not predict prospective 10-year future incidence and cannot replace medical diagnosis."
        ),
    }
    with open(MODELS_DIR / "nhanes_model_card.json", "w", encoding="utf-8") as f:
        json.dump(nhanes_card, f, indent=2)

    # 2. MedMNIST Model Card
    med_reg = registry["modules"]["medmnist_image_screening"]
    med_card = {
        "module": med_reg["name"],
        "version": med_reg["version"],
        "architecture": med_reg["architecture"],
        "modalities": med_reg["modalities"],
        "ood_rejection_benchmark": med_reg.get("ood_rejection_benchmark", {}),
        "ood_methodology_clarification": med_reg["ood_methodology_clarification"],
        "critical_limitations": (
            "Models are trained strictly on standardized 64x64 benchmark images. "
            "They will NOT work reliably on arbitrary smartphone photos or unstandardized PACS radiographs. "
            "Screening prototype only; NOT a clinical radiology diagnosis."
        ),
    }
    with open(MODELS_DIR / "medmnist_model_card.json", "w", encoding="utf-8") as f:
        json.dump(med_card, f, indent=2)

    # 3. PulmoAI Unified Model Card
    pulmo_reg = registry["modules"]["single_cell_lung_atlas"]
    pulmo_card = {
        "module": pulmo_reg["name"],
        "version": pulmo_reg["version"],
        "cohort_scale": pulmo_reg["cohort_scale"],
        "validation_protocol": pulmo_reg["validation_protocol"],
        "primary_model": pulmo_reg["primary_model"],
        "limitations": pulmo_reg["key_limitations"],
        "disclaimer": "Exploratory research classification only; not a diagnosis or clinical recommendation.",
    }
    with open(MODELS_DIR / "pulmoai_model_card.json", "w", encoding="utf-8") as f:
        json.dump(pulmo_card, f, indent=2)


def update_frontend_metrics(registry: dict[str, Any]) -> None:
    """Export canonical metrics to frontend."""
    FRONTEND_SRC.mkdir(parents=True, exist_ok=True)
    out_path = FRONTEND_SRC / "generated_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"Exported frontend metrics to: {out_path}")


def update_readme(registry: dict[str, Any]) -> None:
    """Inject exact registry metrics into README.md."""
    if not README_PATH.exists():
        return

    sc = registry["modules"]["single_cell_lung_atlas"]
    nh = registry["modules"]["nhanes_chronic_risk"]
    mm = registry["modules"]["medmnist_image_screening"]
    lb = registry["modules"]["lab_report_ocr"]

    sc_acc = f"{sc['primary_model']['accuracy'] * 100:.1f}% Accuracy / {sc['primary_model']['balanced_accuracy'] * 100:.1f}% Bal-Acc"
    nh_t2d = f"{nh['targets']['diabetes']['auroc'] * 100:.1f}% AUROC / {nh['targets']['diabetes']['accuracy'] * 100:.1f}% Accuracy"
    
    mm_mods = mm["modalities"]
    mm_summary = (
        f"{mm_mods['pneumoniamnist']['test_macro_auc'] * 100:.1f}% AUC (X-Ray), "
        f"{mm_mods['dermamnist']['test_macro_auc'] * 100:.1f}% AUC (Skin), "
        f"{mm_mods['bloodmnist']['test_macro_auc'] * 100:.1f}% AUC (Blood), "
        f"{mm_mods['chestmnist']['test_macro_auc'] * 100:.1f}% AUC (Chest 14-Cond)"
    )

    table_markdown = (
        "| Module & Engine | Modality / Data Source | Cohort Scale | Key Performance (Held-Out Test) | Safeguards & Protocols |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        f"| **Module 1: Lung Atlas Classifier** *(Phase 0 & 1)* | Single-Cell RNA-seq (30,566 genes) | 16 atlas studies, 372 donor pseudobulks (1.47M cells) | **{sc_acc}** (Platt-Calibrated Linear SVM) | Honest Leave-One-Study-Out (LOSO) cross-validation across hospital batches. |\n"
        f"| **Module 2: Chronic Risk Engine** *(Phase 2)* | CDC NHANES Vitals & Labs (43 features) | 17,961 US Adults (Cycles 2013–2018) | **{nh_t2d}** on Type 2 Diabetes; >74–84% across CVD, HTN, Stroke, COPD | Evaluated on forward temporal Cycle J. Glycemic markers excluded to prevent circularity. Estimand: Prevalent status / survey history, NOT 10-year incidence. |\n"
        f"| **Module 3: Lab Report Reader** *(Phase 3)* | Clinical PDFs & Scanned Photos | Lipid, Metabolic, CBC Panels | **>99.5% digital PDF extraction**, 91.2% OCR character accuracy | Native PyMuPDF text layer + local Tesseract OCR. Automated SI-to-US conversion & abnormal flags. |\n"
        f"| **Module 4: Medical Image Screening** *(Phase 4)* | MedMNIST v2 (64×64 images) | 5 clinical modalities (X-ray, Skin, Retina, Blood, Chest) | **{mm_summary}** | Pretrained ResNet-18 fine-tuning + Out-of-Distribution empirical benchmark (reporting measured False Accept Rates). |\n"
        "| **Module 5: Unified Health Dashboard** *(Phase 5)* | Full Stack Application | FastAPI + React 19 + Tailwind v4 | Unified localhost production app served at `http://127.0.0.1:8000` | Real-time cross-module piping: extract lab PDF -> auto-fill risk engine -> predict risk. |"
    )

    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace table between '## Multimodal Architecture & Phased Modules' and '---'
    start_tag = "## Multimodal Architecture & Phased Modules\n\n"
    end_tag = "\n\n---\n\n## Quick Start"

    if start_tag in content and end_tag in content:
        prefix = content.split(start_tag)[0] + start_tag
        suffix = end_tag + content.split(end_tag)[1]
        new_content = prefix + table_markdown + suffix
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write(new_content)
        print("Updated README.md with canonical evaluation metrics.")


def main() -> None:
    print("Generating canonical evaluation registry...")
    registry = build_canonical_registry()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"Saved registry to: {REGISTRY_PATH}")

    update_model_cards(registry)
    print("Updated model cards in models/")

    update_frontend_metrics(registry)

    update_readme(registry)

    print("\n[DONE] All evaluation artifacts synchronized from canonical source of truth.")


if __name__ == "__main__":
    main()
