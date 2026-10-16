"""PulmoAI Pass 6 (G3) Fresh Holdout OCR Benchmark.

Evaluates an independent synthetic test cohort (seed=999) across 7 layout templates
(Templates A-F plus new Template G: Community Health Center EHR Export),
testing the input quality gate (blur/skew rejection) and the updated reference range parser.

Benchmark label: "synthetic set; real-world accuracy unknown"
"""

from __future__ import annotations

import io
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pulmoai.report_reader import (
    check_image_quality,
    parse_lab_text,
    preprocess_lab_image_adaptive,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_ocr_pass6")

OUTPUT_JSON = PROJECT_ROOT / "reports" / "audit_ocr_pass6.json"

TEST_SPECS = [
    {
        "name": "fasting_glucose",
        "unit": "mg/dL",
        "ref_male": (70.0, 99.0, "70.0 - 99.0 mg/dL"),
        "ref_female": (70.0, 99.0, "70.0 - 99.0 mg/dL"),
    },
    {
        "name": "hba1c",
        "unit": "%",
        "ref_male": (4.0, 5.6, "<= 5.7 %"),
        "ref_female": (4.0, 5.6, "<= 5.7 %"),
    },
    {
        "name": "total_cholesterol",
        "unit": "mg/dL",
        "ref_male": (125.0, 200.0, "< 200 mg/dL"),
        "ref_female": (125.0, 200.0, "< 200 mg/dL"),
    },
    {
        "name": "hdl_cholesterol",
        "unit": "mg/dL",
        "ref_male": (40.0, 60.0, "> 40 mg/dL"),
        "ref_female": (50.0, 70.0, "> 50 mg/dL"),
    },
    {
        "name": "triglycerides",
        "unit": "mg/dL",
        "ref_male": (50.0, 150.0, "< 150 mg/dL"),
        "ref_female": (50.0, 150.0, "< 150 mg/dL"),
    },
    {
        "name": "serum_creatinine",
        "unit": "mg/dL",
        "ref_male": (0.7, 1.3, "0.7 - 1.3 mg/dL"),
        "ref_female": (0.5, 1.1, "0.5 - 1.1 mg/dL"),
    },
    {
        "name": "hemoglobin",
        "unit": "g/dL",
        "ref_male": (13.5, 17.5, "13.5 - 17.5 g/dL"),
        "ref_female": (12.0, 15.5, "12.0 - 15.5 g/dL"),
    },
]

# 7 Layout Templates (A-F plus new Template G)
PASS6_LAYOUT_TEMPLATES = [
    {
        "layout_id": "template_a",
        "layout_name": "Template A (Hospital pathology tabular format)",
        "default_sex": "male",
        "header": "REGIONAL CLINICAL LABORATORY - DEPARTMENT OF PATHOLOGY\nPATIENT NAME: Subject_{idx}   AGE: {age} YRS   GENDER: {sex_cap}   UHID: RCL-{idx:05d}\nREG DATE: 15-Nov-2024   LOCATION: METRO CLINIC   STATUS: FINAL\n====================================================================================\nTEST INVESTIGATION               OBSERVED RESULT   UNIT       REFERENCE INTERVAL\n====================================================================================\n",
        "footer": "\n====================================================================================\nValidated by: Automated Laboratory Information System. Senior Consultant Review.",
        "fmt": "{test_name:<30}   {val:<15.1f}   {unit:<8}   {ref}",
    },
    {
        "layout_id": "template_b",
        "layout_name": "Template B (Cardio-metabolic panel format)",
        "default_sex": "female",
        "header": "REGIONAL HEALTHCARE CLINICAL BIOCHEMISTRY LAB\nNAME: Subject_{idx}   AGE/SEX: {age}/{sex_cap}   IP/OP NO: RHC-{idx:06d}\nREFERRAL: OUTPATIENT CARDIOLOGY CLINIC   DATE: 14/11/2024\n------------------------------------------------------------------------------------\nEXAMINATION                      VALUE     UNITS          BIOLOGICAL REFERENCE\n------------------------------------------------------------------------------------\n",
        "footer": "\n------------------------------------------------------------------------------------\nVerified by Automated Clinical Interface. End of Investigation Report.",
        "fmt": "{test_name:<30}   {val:<7.1f}   {unit:<12}   {ref}",
    },
    {
        "layout_id": "template_c",
        "layout_name": "Template C (High-throughput diagnostic network format)",
        "default_sex": "male",
        "header": "CENTRAL DIAGNOSTIC NETWORK - AUTOMATED PROCESSING UNIT\nPATIENT: Subject_{idx}   AGE: {age}   SEX: {sex_cap}   BARCODE: CDN-{idx:07d}\nSAMPLE COLLECTED AT: REGIONAL CENTER   REPORT GENERATED: 16-NOV-2024\n************************************************************************************\nANALYTE NAME                     VALUE     UNIT           REFERENCE RANGE\n************************************************************************************\n",
        "footer": "\n************************************************************************************\nAccreditation: Standard Clinical Laboratory Quality System. Barcode Verified.",
        "fmt": "{test_name:<30}   {val:<7.1f}   {unit:<12}   {ref}",
    },
    {
        "layout_id": "template_d",
        "layout_name": "Template D (Outpatient summary bulleted format)",
        "default_sex": "female",
        "header": "OUTPATIENT CLINICAL LABORATORY SERVICE - SUMMARY REPORT\nPatient Name: Subject_{idx} ({sex_cap}, {age} Years)   File No: OCS-{idx:04d}\nReporting Service: Clinical Pathology   Reporting Date: 17-Nov-2024\n------------------------------------------------------------------------------------\nLABORATORY REPORT SUMMARY:\n",
        "footer": "\n------------------------------------------------------------------------------------\nKindly correlate clinically. Report reviewed and released electronically.",
        "fmt": "* {test_name}: {val:.1f} {unit} [Reference: {ref}]",
    },
    {
        "layout_id": "template_e",
        "layout_name": "Template E (Academic Health Science Center multi-column tabular)",
        "default_sex": "male",
        "header": "UNIVERSITY HEALTH SCIENCES - CENTRAL CLINICAL LABORATORIES\nDEPARTMENT OF PATHOLOGY & LABORATORY MEDICINE\nPATIENT: Subject_{idx}   DOB/AGE: {age}Y   SEX: {sex_cap}   ACCESSION: UHC-{idx:06d}\nCOLLECTED: 08-Oct-2024   REPORTED: 08-Oct-2024   STATUS: FINAL REPORT\n------------------------------------------------------------------------------------\nTEST PROCEDURE                 FLAG   VALUE     UNIT           REFERENCE RANGE\n------------------------------------------------------------------------------------\n",
        "footer": "\n------------------------------------------------------------------------------------\nMedical Director Review Completed. Electronic Authentication Signature Applied.",
        "fmt": "{test_name:<30}        {val:<7.1f}   {unit:<12}   {ref}",
    },
    {
        "layout_id": "template_f",
        "layout_name": "Template F (Ambulatory Point-of-Care Rapid Panel format)",
        "default_sex": "female",
        "header": "AMBULATORY CARE PARTNERS - POINT OF CARE TESTING NETWORK\nPatient Name: Subject_{idx}   Age: {age}   Sex: {sex_cap}   Visit ID: ACP-{idx:05d}\nOrder Source: Rapid Health Clinic   Provider: Clinical Staff\n====================================================================================\nRAPID DIAGNOSTIC LAB PANEL:\n",
        "footer": "\n====================================================================================\nPoint of care testing device quality control verified.",
        "fmt": "  * {test_name}: {val:.1f} {unit} (Ref Interval: {ref})",
    },
    {
        "layout_id": "template_g",
        "layout_name": "Template G (Community Health Center EHR electronic chart export)",
        "default_sex": "male",
        "header": "COMMUNITY HEALTH EHR - ELECTRONIC LAB RESULTS DISCLOSURE\nPATIENT: Subject_{idx}   CHART#: CHC-{idx:05d}   SEX: {sex_cap}   AGE: {age}\nCLINIC: DOWNTOWN FAMILY HEALTH   DATE/TIME: 20-Oct-2024 09:30\n------------------------------------------------------------------------------------\nANALYTE                          RESULT    UNITS       REFERENCE SPECIFICATION\n------------------------------------------------------------------------------------\n",
        "footer": "\n------------------------------------------------------------------------------------\nElectronically signed by Attending Physician. Clinical correlation required.",
        "fmt": "{test_name:<30}   {val:<8.1f}  {unit:<10}  {ref}",
    },
]

PASS6_TEST_VARIANTS = {
    "template_a": {
        "fasting_glucose": "GLUCOSE, FASTING (HEXOKINASE)",
        "hba1c": "GLYCOSYLATED HEMOGLOBIN (HBA1C HPLC)",
        "total_cholesterol": "CHOLESTEROL TOTAL",
        "hdl_cholesterol": "CHOLESTEROL HDL DIRECT",
        "triglycerides": "TRIGLYCERIDES ENZYMATIC",
        "serum_creatinine": "CREATININE JAFFE KINETIC",
        "hemoglobin": "HEMOGLOBIN CYANMETHEMOGLOBIN",
    },
    "template_b": {
        "fasting_glucose": "Plasma Glucose Fasting",
        "hba1c": "Glycated Hemoglobin (HbA1c)",
        "total_cholesterol": "Total Serum Cholesterol",
        "hdl_cholesterol": "High Density Lipoprotein (HDL)",
        "triglycerides": "Serum Triglycerides",
        "serum_creatinine": "Creatinine, Serum",
        "hemoglobin": "Blood Hemoglobin",
    },
    "template_c": {
        "fasting_glucose": "FASTING BLOOD GLUCOSE",
        "hba1c": "HBA1C - TOTAL GLYCOHEMOGLOBIN",
        "total_cholesterol": "TOTAL CHOLESTEROL",
        "hdl_cholesterol": "HDL CHOLESTEROL",
        "triglycerides": "TRIGLYCERIDES",
        "serum_creatinine": "SERUM CREATININE",
        "hemoglobin": "HEMOGLOBIN ESTIMATION",
    },
    "template_d": {
        "fasting_glucose": "Fasting Blood Sugar (FBS)",
        "hba1c": "HbA1c",
        "total_cholesterol": "Cholesterol Total",
        "hdl_cholesterol": "HDL Cholesterol",
        "triglycerides": "Triglycerides (Fasting)",
        "serum_creatinine": "Creatinine",
        "hemoglobin": "Total Hemoglobin",
    },
    "template_e": {
        "fasting_glucose": "GLUCOSE FASTING REF",
        "hba1c": "HEMOGLOBIN A1C FRACTION",
        "total_cholesterol": "CHOLESTEROL TOTAL",
        "hdl_cholesterol": "CHOLESTEROL HDL DIRECT",
        "triglycerides": "TRIGLYCERIDES SERUM",
        "serum_creatinine": "CREATININE QUANT",
        "hemoglobin": "HEMOGLOBIN LEVEL",
    },
    "template_f": {
        "fasting_glucose": "Fasting Glucose",
        "hba1c": "HbA1c Glycohemoglobin",
        "total_cholesterol": "Total Cholesterol",
        "hdl_cholesterol": "HDL Cholesterol Direct",
        "triglycerides": "Serum Triglycerides",
        "serum_creatinine": "Creatinine",
        "hemoglobin": "Blood Hemoglobin",
    },
    "template_g": {
        "fasting_glucose": "Glucose, Fasting",
        "hba1c": "Hemoglobin A1c",
        "total_cholesterol": "Cholesterol, Total",
        "hdl_cholesterol": "HDL Cholesterol",
        "triglycerides": "Triglycerides",
        "serum_creatinine": "Creatinine",
        "hemoglobin": "Hemoglobin",
    },
}


def build_pass6_holdout_dataset(n_per_template: int = 10, seed: int = 999) -> list[dict[str, Any]]:
    dataset = []
    doc_id = 3000
    rng = np.random.RandomState(seed)

    for layout in PASS6_LAYOUT_TEMPLATES:
        lid = layout["layout_id"]
        lname = layout["layout_name"]
        d_sex = layout["default_sex"]

        for var_idx in range(n_per_template):
            doc_id += 1
            sex = d_sex if var_idx % 2 == 0 else ("female" if d_sex == "male" else "male")
            age = int(rng.randint(25, 80))
            sex_cap = sex.capitalize()

            hdr = layout["header"].format(idx=doc_id, age=age, sex_cap=sex_cap)
            lines = []
            entities = []

            for spec in TEST_SPECS:
                canon = spec["name"]
                unit = spec["unit"]
                disp_name = PASS6_TEST_VARIANTS[lid][canon]

                ref_tuple = spec["ref_male"] if sex == "male" else spec["ref_female"]
                rmin, rmax, rstr = ref_tuple

                mode = rng.choice(["norm", "high", "low"])
                if mode == "norm":
                    val = round(float(rng.uniform(rmin, rmax if rmax is not None else rmin + 40)), 1)
                    if rmin is not None and val < rmin:
                        val = rmin
                    if rmax is not None and val > rmax:
                        val = rmax
                    flag = "NORMAL"
                elif mode == "high":
                    base_max = rmax if rmax is not None else 150.0
                    val = round(float(rng.uniform(base_max + 1.0, base_max * 1.5)), 1)
                    if val <= base_max:
                        val = base_max + 1.0
                    flag = "HIGH"
                else:  # low
                    base_min = rmin if rmin is not None else 50.0
                    val = round(float(rng.uniform(base_min * 0.4, base_min - 0.5)), 1)
                    if val >= base_min:
                        val = base_min - 0.5
                    flag = "LOW"

                line_str = layout["fmt"].format(test_name=disp_name, val=val, unit=unit, ref=rstr)
                lines.append(line_str)
                entities.append({
                    "canonical_name": canon,
                    "expected_value": val,
                    "expected_unit": unit,
                    "expected_flag": flag,
                    "reference_range": rstr,
                })

            full_text = hdr + "\n".join(lines) + layout["footer"]
            dataset.append({
                "doc_id": doc_id,
                "layout_id": lid,
                "layout_name": lname,
                "patient_sex": sex,
                "patient_age": age,
                "text": full_text,
                "entities": entities,
            })

    return dataset


def render_pdf_bytes(text_content: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 standard
    rect = pymupdf.Rect(40, 40, 555, 802)
    page.insert_textbox(rect, text_content, fontsize=9.5, fontname="Courier")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def render_base_image(pdf_bytes: bytes, dpi: int = 150) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[0].get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return img


def apply_degradation(base_img: Image.Image, degradation: str) -> Image.Image:
    if degradation == "clean_pdf":
        return base_img
    if degradation == "rotated_raw":
        return base_img.rotate(3.5, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    if degradation == "blur":
        return base_img.filter(ImageFilter.GaussianBlur(radius=1.2))
    if degradation == "jpeg":
        buf = io.BytesIO()
        base_img.save(buf, format="JPEG", quality=18)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if degradation == "perspective":
        w, h = base_img.size
        coeffs = [1.02, 0.03, -15, 0.01, 0.98, -10, 0.0001, 0.00005]
        return base_img.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    return base_img


def run_ocr(pil_img: Image.Image) -> str:
    import pytesseract
    try:
        return pytesseract.image_to_string(pil_img)
    except Exception as e:
        logger.warning("Local Tesseract OCR execution failed: %s", e)
        return ""


def evaluate_extracted_triad(
    extracted_items: list[Any],
    ground_truth_entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ext_dict = {item.canonical_name: item for item in extracted_items}
    results = []

    for gt in ground_truth_entities:
        canon = gt["canonical_name"]
        exp_val = gt["expected_value"]
        exp_flag = gt["expected_flag"]

        if canon not in ext_dict:
            results.append({
                "canonical_name": canon,
                "status": "not_detected",
                "parsed_value": None,
                "expected_value": exp_val,
                "parsed_flag": None,
                "expected_flag": exp_flag,
                "confidence": 0.0,
            })
        else:
            it = ext_dict[canon]
            val_ok = abs(it.value - exp_val) <= 0.15
            flag_ok = (it.flag == exp_flag)
            conf = float(getattr(it, "confidence", 0.95))

            if val_ok and flag_ok:
                results.append({
                    "canonical_name": canon,
                    "status": "correct",
                    "parsed_value": it.value,
                    "expected_value": exp_val,
                    "parsed_flag": it.flag,
                    "expected_flag": exp_flag,
                    "confidence": conf,
                })
            elif val_ok and not flag_ok:
                results.append({
                    "canonical_name": canon,
                    "status": "flag_mismatch_only",
                    "parsed_value": it.value,
                    "expected_value": exp_val,
                    "parsed_flag": it.flag,
                    "expected_flag": exp_flag,
                    "confidence": conf,
                })
            else:
                results.append({
                    "canonical_name": canon,
                    "status": "wrong_value_accepted",
                    "parsed_value": it.value,
                    "expected_value": exp_val,
                    "parsed_flag": it.flag,
                    "expected_flag": exp_flag,
                    "confidence": conf,
                })

    return results


def main() -> None:
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("PASS 6 (G3): FRESH HOLDOUT OCR BENCHMARK WITH QUALITY GATE")
    logger.info("=" * 80)

    dataset = build_pass6_holdout_dataset(n_per_template=10, seed=999)
    total_docs = len(dataset)
    entities_per_doc = len(dataset[0]["entities"])
    total_entities_per_cond = total_docs * entities_per_doc

    logger.info(
        "Generated %d synthetic documents across %d layout templates (%d entities per condition).",
        total_docs,
        len(PASS6_LAYOUT_TEMPLATES),
        total_entities_per_cond,
    )

    doc_renderings = []
    for d in dataset:
        pdf_b = render_pdf_bytes(d["text"])
        base_img = render_base_image(pdf_b)
        doc_renderings.append({"pdf_bytes": pdf_b, "base_img": base_img, "doc": d})

    eval_conditions = [
        "clean_pdf",
        "rotated_raw",
        "rotated_adaptive",
        "blur",
        "jpeg",
        "perspective",
    ]

    condition_summaries = {}
    all_wrong_value_cases = []

    for cond in eval_conditions:
        logger.info("Evaluating condition: %s...", cond)
        cond_t0 = time.time()
        counts = {
            "correct": 0,
            "wrong_value_accepted": 0,
            "not_detected": 0,
            "flag_mismatch_only": 0,
            "refused_for_quality": 0,
        }
        refused_docs = 0

        for rend in doc_renderings:
            d = rend["doc"]
            gt_entities = d["entities"]
            sex = d["patient_sex"]

            if cond == "clean_pdf":
                doc_pdf = pymupdf.open(stream=rend["pdf_bytes"], filetype="pdf")
                raw_text = doc_pdf[0].get_text()
                doc_pdf.close()
                extracted_items, _ = parse_lab_text(raw_text, sex=sex)
                triad_results = evaluate_extracted_triad(extracted_items, gt_entities)
                for tr in triad_results:
                    counts[tr["status"]] += 1
            else:
                deg_img = apply_degradation(rend["base_img"], "rotated_raw" if cond == "rotated_adaptive" else cond)
                is_ok, q_msg, q_metrics = check_image_quality(deg_img)

                if not is_ok:
                    refused_docs += 1
                    counts["refused_for_quality"] += len(gt_entities)
                else:
                    if cond == "rotated_adaptive":
                        prep_img, _ = preprocess_lab_image_adaptive(deg_img)
                    else:
                        prep_img = deg_img
                    raw_text = run_ocr(prep_img)
                    extracted_items, _ = parse_lab_text(raw_text, sex=sex)
                    triad_results = evaluate_extracted_triad(extracted_items, gt_entities)
                    for tr in triad_results:
                        st = tr["status"]
                        counts[st] += 1
                        if st == "wrong_value_accepted":
                            all_wrong_value_cases.append({
                                "condition": cond,
                                "layout_id": d["layout_id"],
                                "doc_id": d["doc_id"],
                                "canonical_name": tr["canonical_name"],
                                "parsed_value": tr["parsed_value"],
                                "expected_value": tr["expected_value"],
                            })

        accepted_entities = total_entities_per_cond - counts["refused_for_quality"]
        silent_err_rate_accepted = (
            counts["wrong_value_accepted"] / accepted_entities if accepted_entities > 0 else 0.0
        )
        accuracy_accepted = (
            counts["correct"] / accepted_entities if accepted_entities > 0 else 0.0
        )

        condition_summaries[cond] = {
            "condition": cond,
            "total_entities": total_entities_per_cond,
            "refused_for_quality_count": counts["refused_for_quality"],
            "refused_for_quality_rate": round(counts["refused_for_quality"] / total_entities_per_cond, 4),
            "accepted_entities_count": accepted_entities,
            "correct_count": counts["correct"],
            "accuracy_on_accepted": round(accuracy_accepted, 4),
            "wrong_value_accepted_count": counts["wrong_value_accepted"],
            "silent_error_rate_on_accepted": round(silent_err_rate_accepted, 4),
            "flag_mismatch_only_count": counts["flag_mismatch_only"],
            "flag_mismatch_rate_on_accepted": round(counts["flag_mismatch_only"] / accepted_entities if accepted_entities > 0 else 0.0, 4),
            "not_detected_count": counts["not_detected"],
            "not_detected_rate_on_accepted": round(counts["not_detected"] / accepted_entities if accepted_entities > 0 else 0.0, 4),
            "runtime_seconds": round(time.time() - cond_t0, 2),
        }

        logger.info(
            "%-18s: Total: %d | Refused: %3d (%.1f%%) | Accepted: %3d | Correct: %3d (%.1f%%) | Silent Err: %3d (%.2f%%)",
            cond,
            total_entities_per_cond,
            counts["refused_for_quality"],
            counts["refused_for_quality"] / total_entities_per_cond * 100.0,
            accepted_entities,
            counts["correct"],
            accuracy_accepted * 100.0,
            counts["wrong_value_accepted"],
            silent_err_rate_accepted * 100.0,
        )

    out_data = {
        "metadata": {
            "title": "PulmoAI Pass 6 (G3) Fresh Holdout OCR Evaluation with Quality Gate",
            "benchmark_label": "synthetic set; real-world accuracy unknown",
            "seed": 999,
            "total_synthetic_documents": total_docs,
            "total_entities_per_condition": total_entities_per_cond,
            "templates_evaluated": [l["layout_id"] for l in PASS6_LAYOUT_TEMPLATES],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_runtime_seconds": round(time.time() - t0, 2),
        },
        "condition_summaries": condition_summaries,
        "quality_gate_policy": "If image is severely blurred (lap_var < 30.0) or excessively skewed (best_angle >= 12.0 deg), OCR execution is refused and 0 values are extracted.",
        "silent_error_examples": all_wrong_value_cases[:10],
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    logger.info("Saved fresh Pass 6 OCR benchmark to %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
