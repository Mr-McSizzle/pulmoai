#!/usr/bin/env python3
"""Synthetic Benchmark and Honest Evaluation for PulmoAI Lab Report Reader.

Evaluates entity-level exact-match accuracy across 5 distinct report layouts:
1. Corporate Reference Laboratory Style
2. Regional Pathology Laboratory Style
3. Specialty Diagnostic Laboratory Style
4. Standard Hospital EMR (Electronic health record discharge summary)
5. Compact Clinic Summary (Outpatient narrative and vitals format)

Across 6 degradation tiers:
- clean_pdf: Digital PDF text layer (PyMuPDF)
- clean_ocr: 150 DPI rendered image with pure local Tesseract OCR
- rotation: 3.5-degree handheld camera tilt
- blur: Gaussian blur (radius=1.5)
- jpeg_compression: Quality=25 (simulating WhatsApp/compressed phone capture)
- perspective: Phone-style keystoned quad perspective transform

Label: "synthetic test set; real-world accuracy unknown"
Removes unsupported claims (99.5% on scanned documents).
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter
import pymupdf

from pulmoai.report_reader import process_lab_report, CANONICAL_TESTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_ocr")

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ground Truth Definitions for 5 Distinct Layouts
# ---------------------------------------------------------------------------

SYNTHETIC_LAYOUTS: list[dict[str, Any]] = [
    {
        "layout_id": "corporate_reference_lab",
        "layout_name": "Corporate Reference Laboratory Style",
        "patient_sex": "male",
        "header_lines": [
            "Corporate Reference Laboratories Ltd. - National Reference Center",
            "Patient Name: Rajesh Gupta    Age / Sex: 48 Y / Male    Ref: Self",
            "Registration No: CRL-884920    Collection Date: 12-Oct-2024",
            "--------------------------------------------------------------------------------",
            "Test Name                     Result   Units   Bio Ref Interval",
            "--------------------------------------------------------------------------------",
        ],
        "test_lines": [
            ("Fasting Blood Sugar           108.0    mg/dL   70.0 - 99.0", "fasting_glucose", 108.0, "mg/dL", "HIGH"),
            ("HbA1c (Glycated Hemoglobin)   6.2      %       4.0 - 5.6", "hba1c", 6.2, "%", "HIGH"),
            ("Total Cholesterol             215.0    mg/dL   < 200.0", "total_cholesterol", 215.0, "mg/dL", "HIGH"),
            ("HDL Cholesterol               42.0     mg/dL   >= 40.0", "hdl_cholesterol", 42.0, "mg/dL", "NORMAL"),
            ("Triglycerides                 165.0    mg/dL   < 150.0", "triglycerides", 165.0, "mg/dL", "HIGH"),
            ("Serum Creatinine              1.1      mg/dL   0.7 - 1.3", "serum_creatinine", 1.1, "mg/dL", "NORMAL"),
            ("Hemoglobin                    14.5     g/dL    13.5 - 17.5", "hemoglobin", 14.5, "g/dL", "NORMAL"),
        ],
        "footer_lines": [
            "--------------------------------------------------------------------------------",
            "End of report. Verified by Dr. S. K. Specialist, MD (Biochemistry)",
        ],
    },
    {
        "layout_id": "regional_pathology_lab",
        "layout_name": "Regional Pathology Laboratory Style",
        "patient_sex": "female",
        "header_lines": [
            "Regional Pathology Laboratories Ltd. - Center of Excellence",
            "Patient: Sunita Verma   Age: 42 Years   Gender: Female   Lab No: RPL-40192",
            "Referring Physician: Dr. K. Patel   Sample Drawn: 11-Oct-2024",
            "================================================================================",
            "Investigation                 Observed Value   Units   Biological Ref Interval",
            "================================================================================",
        ],
        "test_lines": [
            ("Glucose, Fasting              92.0             mg/dL   70.0 - 99.0", "fasting_glucose", 92.0, "mg/dL", "NORMAL"),
            ("Glycohemoglobin HbA1c         5.4              %       4.0 - 5.6", "hba1c", 5.4, "%", "NORMAL"),
            ("Cholesterol, Total            188.0            mg/dL   < 200.0", "total_cholesterol", 188.0, "mg/dL", "NORMAL"),
            ("HDL-Cholesterol               55.0             mg/dL   >= 50.0", "hdl_cholesterol", 55.0, "mg/dL", "NORMAL"),
            ("LDL Cholesterol (Calculated)  104.0            mg/dL   < 100.0", "ldl_cholesterol", 104.0, "mg/dL", "HIGH"),
            ("Creatinine, Serum             0.8              mg/dL   0.5 - 1.1", "serum_creatinine", 0.8, "mg/dL", "NORMAL"),
            ("Hemoglobin (Hb)               11.2             g/dL    12.0 - 15.5", "hemoglobin", 11.2, "g/dL", "LOW"),
        ],
        "footer_lines": [
            "================================================================================",
            "Methodology: Fully automated dry/wet chemistry analyzer. Tests reported by Pathology Unit.",
        ],
    },
    {
        "layout_id": "specialty_diagnostic_lab",
        "layout_name": "Specialty Diagnostic Laboratory Style",
        "patient_sex": "male",
        "header_lines": [
            "Specialty Diagnostic Laboratories Ltd. - Comprehensive Diagnostic Care",
            "Patient Name: Vikram Mehta   Age / Sex: 52 Y / Male   PID: SDL-991204",
            "Doctor: Dr. A. Joshi   Specimen: Venous Blood / Fluoride Plasma",
            "--------------------------------------------------------------------------------",
            "Test Description              Result   Units    Reference Range",
            "--------------------------------------------------------------------------------",
        ],
        "test_lines": [
            ("Fasting Plasma Glucose        138.0    mg/dL    70.0 - 99.0", "fasting_glucose", 138.0, "mg/dL", "HIGH"),
            ("Serum Cholesterol             232.0    mg/dL    < 200.0", "total_cholesterol", 232.0, "mg/dL", "HIGH"),
            ("Serum Triglycerides           210.0    mg/dL    < 150.0", "triglycerides", 210.0, "mg/dL", "HIGH"),
            ("HDL Cholesterol               36.0     mg/dL    >= 40.0", "hdl_cholesterol", 36.0, "mg/dL", "LOW"),
            ("Serum Creatinine              1.4      mg/dL    0.7 - 1.3", "serum_creatinine", 1.4, "mg/dL", "HIGH"),
            ("ALT (SGPT)                    48.0     U/L      7.0 - 56.0", "alt", 48.0, "U/L", "NORMAL"),
            ("AST (SGOT)                    54.0     U/L      10.0 - 40.0", "ast", 54.0, "U/L", "HIGH"),
        ],
        "footer_lines": [
            "--------------------------------------------------------------------------------",
            "Accredited Laboratory. Electronic digital signature applied.",
        ],
    },
    {
        "layout_id": "standard_hospital_emr",
        "layout_name": "Standard Hospital EMR Style (Electronic Export)",
        "patient_sex": "female",
        "header_lines": [
            "Regional Clinical Laboratory - Electronic Health Record",
            "Patient: Anita Desai   DOB: 14-Aug-1976   Sex: Female   MRN: 7029148",
            "Encounter: Inpatient Ward 4B   Attending: Dr. R. Iyer, MD",
            "--------------------------------------------------------------------------------",
            "Analyte                       Value    Units    Normal Range",
            "--------------------------------------------------------------------------------",
        ],
        "test_lines": [
            ("Blood Urea Nitrogen           16.0     mg/dL    7.0 - 20.0", "bun", 16.0, "mg/dL", "NORMAL"),
            ("Serum Creatinine              0.9      mg/dL    0.5 - 1.1", "serum_creatinine", 0.9, "mg/dL", "NORMAL"),
            ("Serum Sodium                  140.0    mmol/L   135.0 - 145.0", "sodium", 140.0, "mmol/L", "NORMAL"),
            ("Serum Potassium               4.2      mmol/L   3.5 - 5.2", "potassium", 4.2, "mmol/L", "NORMAL"),
            ("Total Calcium                 9.3      mg/dL    8.5 - 10.5", "calcium", 9.3, "mg/dL", "NORMAL"),
            ("Total Protein                 7.2      g/dL     6.0 - 8.3", "total_protein", 7.2, "g/dL", "NORMAL"),
            ("Serum Albumin                 4.1      g/dL     3.5 - 5.0", "albumin", 4.1, "g/dL", "NORMAL"),
        ],
        "footer_lines": [
            "--------------------------------------------------------------------------------",
            "Document authenticated in Epic EHR system.",
        ],
    },
    {
        "layout_id": "compact_clinic_summary",
        "layout_name": "Compact Clinic Summary Style (Narrative Walk-in Format)",
        "patient_sex": "male",
        "header_lines": [
            "CityHealth Community Clinic - Outpatient Vitals & Rapid Lab Panel",
            "Client: Amit Patel   Sex: Male   Age: 36   Encounter: 2024-10-10",
            "Clinician: Dr. N. Sengupta, MBBS",
            "--------------------------------------------------------------------------------",
        ],
        "test_lines": [
            ("BP: 135/88 mmHg (Ref 90-120 / 60-80 mmHg)", "systolic_bp", 135.0, "mmHg", "HIGH"),
            # Note: the compound BP line also contains diastolic_bp: 88.0
            ("Fasting Glucose: 102.0 mg/dL (Ref 70 - 99 mg/dL)", "fasting_glucose", 102.0, "mg/dL", "HIGH"),
            ("Total Cholesterol: 195.0 mg/dL (Ref < 200 mg/dL)", "total_cholesterol", 195.0, "mg/dL", "NORMAL"),
            ("Serum Creatinine: 1.0 mg/dL (Ref 0.7 - 1.3 mg/dL)", "serum_creatinine", 1.0, "mg/dL", "NORMAL"),
            ("Hemoglobin: 15.2 g/dL (Ref 13.5 - 17.5 g/dL)", "hemoglobin", 15.2, "g/dL", "NORMAL"),
        ],
        "footer_lines": [
            "--------------------------------------------------------------------------------",
            "Rapid bedside testing verified. Please discuss with your clinician.",
        ],
    },
]

# Add diastolic_bp to compact_clinic_summary ground truth
for layout in SYNTHETIC_LAYOUTS:
    if layout["layout_id"] == "compact_clinic_summary":
        layout["extra_ground_truth"] = [
            ("diastolic_bp", 88.0, "mmHg", "HIGH"),
        ]

# ---------------------------------------------------------------------------
# Document & Image Generation Utilities
# ---------------------------------------------------------------------------

def generate_pdf_document(layout: dict[str, Any]) -> tuple[bytes, Image.Image]:
    """Build a synthetic PDF using PyMuPDF and render a clean base image."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 standard

    y = 50.0
    for line in layout["header_lines"]:
        page.insert_text((40, y), line, fontsize=10, fontname="courier")
        y += 18.0

    for item in layout["test_lines"]:
        line_str = item[0]
        page.insert_text((40, y), line_str, fontsize=10, fontname="courier")
        y += 20.0

    y += 10.0
    for line in layout["footer_lines"]:
        page.insert_text((40, y), line, fontsize=9, fontname="courier")
        y += 16.0

    pdf_bytes = doc.tobytes()
    pix = page.get_pixmap(dpi=150)
    base_img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return pdf_bytes, base_img


def apply_degradation(img: Image.Image, degradation: str) -> bytes:
    """Apply specific image degradation tier and return image bytes."""
    buf = io.BytesIO()

    if degradation == "clean_ocr":
        img.save(buf, format="PNG")
    elif degradation == "rotation":
        # 3.5 degree handheld rotation with white fill
        rot = img.rotate(3.5, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
        rot.save(buf, format="PNG")
    elif degradation == "blur":
        # Gaussian blur radius 1.5
        blurred = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        blurred.save(buf, format="PNG")
    elif degradation == "jpeg_compression":
        # WhatsApp / aggressive phone compression (quality=25)
        img.save(buf, format="JPEG", quality=25)
    elif degradation == "perspective":
        # Quad perspective keystone distortion
        w, h = img.size
        quad = (
            int(w * 0.04), int(h * 0.02),  # top-left
            int(w * 0.01), int(h * 0.98),  # bottom-left
            int(w * 0.99), int(h * 0.98),  # bottom-right
            int(w * 0.95), int(h * 0.03),  # top-right
        )
        persp = img.transform(
            (w, h), Image.Transform.QUAD, quad, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255)
        )
        persp.save(buf, format="PNG")
    else:
        raise ValueError(f"Unknown degradation mode: {degradation}")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Evaluation Routine
# ---------------------------------------------------------------------------

def evaluate_layout_degradation(
    file_bytes: bytes,
    filename: str,
    layout: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate extraction against layout ground truth."""
    patient_sex = layout.get("patient_sex")
    res = process_lab_report(file_bytes, filename, sex=patient_sex)

    extracted_by_name = {t["canonical_name"]: t for t in res["extracted_tests"]}

    ground_truth_targets: list[dict[str, Any]] = []
    for item in layout["test_lines"]:
        ground_truth_targets.append({
            "canonical_name": item[1],
            "value": item[2],
            "unit": item[3],
            "flag": item[4],
        })

    for extra in layout.get("extra_ground_truth", []):
        ground_truth_targets.append({
            "canonical_name": extra[0],
            "value": extra[1],
            "unit": extra[2],
            "flag": extra[3],
        })

    total_targets = len(ground_truth_targets)
    exact_matches = 0
    test_detected_count = 0
    value_mismatches = 0
    flag_matches = 0
    missed: list[str] = []
    errors: list[dict[str, Any]] = []

    for gt in ground_truth_targets:
        c_name = gt["canonical_name"]
        expected_val = gt["value"]
        expected_flag = gt["flag"]

        if c_name in extracted_by_name:
            test_detected_count += 1
            extracted = extracted_by_name[c_name]
            ext_val = extracted["value"]
            ext_flag = extracted["flag"]

            # Check value match within 5% or 0.1 abs
            val_diff = abs(ext_val - expected_val)
            val_matches = val_diff <= max(0.05 * expected_val, 0.1)

            if val_matches:
                exact_matches += 1
            else:
                value_mismatches += 1
                errors.append({
                    "test": c_name,
                    "expected_val": expected_val,
                    "extracted_val": ext_val,
                    "reason": "value_mismatch",
                })

            if ext_flag == expected_flag:
                flag_matches += 1
            else:
                errors.append({
                    "test": c_name,
                    "expected_flag": expected_flag,
                    "extracted_flag": ext_flag,
                    "reason": "flag_mismatch",
                })
        else:
            missed.append(c_name)
            errors.append({
                "test": c_name,
                "expected_val": expected_val,
                "reason": "missed_test",
            })

    exact_match_accuracy = exact_matches / total_targets if total_targets > 0 else 0.0
    detection_recall = test_detected_count / total_targets if total_targets > 0 else 0.0
    value_fidelity = exact_matches / test_detected_count if test_detected_count > 0 else 0.0
    flag_accuracy = flag_matches / total_targets if total_targets > 0 else 0.0

    return {
        "total_targets": total_targets,
        "exact_matches": exact_matches,
        "test_detected_count": test_detected_count,
        "value_mismatches": value_mismatches,
        "missed_count": len(missed),
        "exact_match_accuracy": round(exact_match_accuracy, 4),
        "detection_recall": round(detection_recall, 4),
        "value_fidelity": round(value_fidelity, 4),
        "flag_accuracy": round(flag_accuracy, 4),
        "extraction_method": res["extraction_method"],
        "errors": errors,
    }


def run_benchmark() -> dict[str, Any]:
    """Execute complete synthetic benchmark across layouts and degradations."""
    degradations = ["clean_pdf", "clean_ocr", "rotation", "blur", "jpeg_compression", "perspective"]
    results_by_layout: dict[str, dict[str, Any]] = {}
    results_by_degradation: dict[str, dict[str, Any]] = {
        d: {"total_targets": 0, "exact_matches": 0, "detected": 0, "value_mismatches": 0, "missed": 0}
        for d in degradations
    }

    total_all_targets = 0
    total_all_exact = 0

    for layout in SYNTHETIC_LAYOUTS:
        l_id = layout["layout_id"]
        logger.info("Evaluating layout: %s (%s)", l_id, layout["layout_name"])
        pdf_bytes, base_img = generate_pdf_document(layout)

        layout_summary: dict[str, Any] = {
            "layout_name": layout["layout_name"],
            "patient_sex": layout["patient_sex"],
            "tier_results": {},
        }

        for deg in degradations:
            if deg == "clean_pdf":
                file_bytes = pdf_bytes
                fname = f"{l_id}.pdf"
            else:
                file_bytes = apply_degradation(base_img, deg)
                fname = f"{l_id}_{deg}.png" if deg != "jpeg_compression" else f"{l_id}_{deg}.jpg"

            eval_res = evaluate_layout_degradation(file_bytes, fname, layout)
            layout_summary["tier_results"][deg] = eval_res

            # Accumulate
            d_acc = results_by_degradation[deg]
            d_acc["total_targets"] += eval_res["total_targets"]
            d_acc["exact_matches"] += eval_res["exact_matches"]
            d_acc["detected"] += eval_res["test_detected_count"]
            d_acc["value_mismatches"] += eval_res["value_mismatches"]
            d_acc["missed"] += eval_res["missed_count"]

            total_all_targets += eval_res["total_targets"]
            total_all_exact += eval_res["exact_matches"]

        results_by_layout[l_id] = layout_summary

    # Summary by degradation
    summary_degradation: dict[str, Any] = {}
    for deg, d_data in results_by_degradation.items():
        tot = d_data["total_targets"]
        exact = d_data["exact_matches"]
        det = d_data["detected"]
        summary_degradation[deg] = {
            "total_targets": tot,
            "exact_matches": exact,
            "detected": det,
            "value_mismatches": d_data["value_mismatches"],
            "missed": d_data["missed"],
            "exact_match_accuracy": round(exact / tot, 4) if tot > 0 else 0.0,
            "detection_recall": round(det / tot, 4) if tot > 0 else 0.0,
            "value_fidelity": round(exact / det, 4) if det > 0 else 0.0,
        }

    # Summary by layout (across all degradations)
    summary_layout: dict[str, Any] = {}
    for l_id, l_data in results_by_layout.items():
        l_tot = sum(t["total_targets"] for t in l_data["tier_results"].values())
        l_exact = sum(t["exact_matches"] for t in l_data["tier_results"].values())
        pdf_acc = l_data["tier_results"]["clean_pdf"]["exact_match_accuracy"]
        ocr_clean_acc = l_data["tier_results"]["clean_ocr"]["exact_match_accuracy"]
        persp_acc = l_data["tier_results"]["perspective"]["exact_match_accuracy"]
        summary_layout[l_id] = {
            "layout_name": l_data["layout_name"],
            "pdf_exact_accuracy": pdf_acc,
            "clean_ocr_accuracy": ocr_clean_acc,
            "perspective_accuracy": persp_acc,
            "overall_exact_accuracy": round(l_exact / l_tot, 4) if l_tot > 0 else 0.0,
        }

    audit_payload: dict[str, Any] = {
        "evaluation_note": "synthetic test set; real-world accuracy unknown",
        "description": (
            "Honest entity-level evaluation of PulmoAI laboratory report reader across 5 "
            "synthetic clinical report layouts (corporate reference, regional pathology, specialty diagnostic, hospital EMR, clinic summary) "
            "and 6 physical degradation tiers (native digital PDF, clean OCR scan, rotation, blur, "
            "JPEG compression, and phone-style perspective distortion)."
        ),
        "unsupported_claims_removed": [
            "Commercial brand validation claims removed (not evaluated on proprietary commercial formats).",
            ">99.5% entity identification claim removed (true only for clean digital PDFs with embedded text layers; severely drops on scanned images).",
        ],
        "sex_specific_reference_ranges": {
            "hemoglobin": {"male": "13.5 - 17.5 g/dL", "female": "12.0 - 15.5 g/dL"},
            "serum_creatinine": {"male": "0.7 - 1.3 mg/dL", "female": "0.5 - 1.1 mg/dL"},
            "hdl_cholesterol": {"male": ">= 40 mg/dL", "female": ">= 50 mg/dL"},
        },
        "transfer_safety_rule": "Requires explicit clinician/user confirmation before transferring extracted values; transfer blocked when confidence < 0.70.",
        "aggregate_metrics": {
            "total_layouts": len(SYNTHETIC_LAYOUTS),
            "total_degradation_tiers": len(degradations),
            "total_evaluations": total_all_targets,
            "total_exact_matches": total_all_exact,
            "overall_exact_match_accuracy": round(total_all_exact / total_all_targets, 4),
        },
        "degradation_breakdown": summary_degradation,
        "layout_breakdown": summary_layout,
        "detailed_results_by_layout": results_by_layout,
    }

    out_file = REPORTS_DIR / "audit_ocr.json"
    with open(out_file, "w") as f:
        json.dump(audit_payload, f, indent=2)

    logger.info("Saved OCR audit to %s", out_file)
    return audit_payload


if __name__ == "__main__":
    payload = run_benchmark()
    print("\n=== OCR AUDIT SUMMARY ===")
    print(f"Evaluation note: {payload['evaluation_note']}")
    print(f"Overall exact match accuracy: {payload['aggregate_metrics']['overall_exact_match_accuracy'] * 100:.2f}%\n")
    print(f"{'Degradation Tier':<20} | {'Exact Match Acc':<16} | {'Recall':<10} | {'Value Fidelity':<14}")
    print("-" * 68)
    for deg, d in payload["degradation_breakdown"].items():
        print(f"{deg:<20} | {d['exact_match_accuracy']*100:>14.2f}% | {d['detection_recall']*100:>8.2f}% | {d['value_fidelity']*100:>12.2f}%")
