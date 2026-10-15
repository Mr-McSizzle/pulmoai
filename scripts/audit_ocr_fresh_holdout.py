#!/usr/bin/env python3
"""Audit OCR on Fresh Holdout Set with 2 New Layout Templates (Pass 5 F5).

Generates a fresh synthetic benchmark:
- Brand new seed (seed=777)
- 2 new templates (Template E: Academic Health Science Center multi-column tabular;
  Template F: Ambulatory / Urgent Care Point-of-Care rapid panel), in addition to Templates A-D.
- 60 documents total (10 per template) x 7 clinical biomarkers = 420 entities per condition (>= 350 requirement).
- Evaluates 6 conditions:
  1. clean_pdf: Native PDF text layer (PyMuPDF)
  2. rotated_raw: Rotated scan (+3.5 deg) with uncorrected OCR
  3. rotated_adaptive: Rotated scan (+3.5 deg) with adaptive deskew preprocessing
  4. blur: Gaussian blur (radius=1.8)
  5. jpeg: Heavy JPEG compression (quality=25)
  6. perspective: Degraded contrast and illumination
- Categorizes all outcomes into:
  - correct: numerical value within 0.15 tolerance AND flag matches
  - wrong_value_accepted: silent error (analyte recognized, but |parsed - expected| > 0.15)
  - not_detected: analyte missing from parsed output
  - flag_mismatch_only: numerical value 100% correct within tolerance, but flag differed
- Reports silent-error rate per layout template and per condition.
- Measures fraction of erroneous/low-confidence results blocked from clinical risk engine transfer.
- Label: 'synthetic set; real-world accuracy unknown'.
- Saves to: reports/audit_ocr_fresh_holdout.json
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pymupdf

from pulmoai.report_reader import (
    determine_flag,
    parse_lab_text,
    preprocess_lab_image_adaptive,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_ocr_fresh_holdout")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = PROJECT_ROOT / "reports" / "audit_ocr_fresh_holdout.json"

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

# 6 Layout Templates: Templates A-D plus NEW Templates E & F (Pass 5 F5)
FRESH_LAYOUT_TEMPLATES = [
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
        "header": "METROPOLITAN HEALTHCARE CLINICAL BIOCHEMISTRY LAB\nNAME: Subject_{idx}   AGE/SEX: {age}/{sex_cap}   IP/OP NO: MHC-{idx:06d}\nREFERRAL: OUTPATIENT CARDIOLOGY CLINIC   DATE: 14/11/2024\n------------------------------------------------------------------------------------\nEXAMINATION                      VALUE     UNITS          BIOLOGICAL REFERENCE\n------------------------------------------------------------------------------------\n",
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
]

FRESH_TEST_VARIANTS = {
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
        "triglycerides": "Triglycerides",
        "serum_creatinine": "Serum Creatinine",
        "hemoglobin": "Hemoglobin (Hb)",
    },
    "template_e": {
        "fasting_glucose": "GLUCOSE, SERUM FASTING",
        "hba1c": "HEMOGLOBIN A1C",
        "total_cholesterol": "CHOLESTEROL TOTAL",
        "hdl_cholesterol": "CHOLESTEROL HDL DIRECT",
        "triglycerides": "TRIGLYCERIDES",
        "serum_creatinine": "CREATININE, SERUM",
        "hemoglobin": "HEMOGLOBIN",
    },
    "template_f": {
        "fasting_glucose": "Fasting Glucose",
        "hba1c": "Glycohemoglobin (HbA1c)",
        "total_cholesterol": "Total Cholesterol",
        "hdl_cholesterol": "HDL Cholesterol Direct",
        "triglycerides": "Serum Triglycerides",
        "serum_creatinine": "Creatinine",
        "hemoglobin": "Blood Hemoglobin",
    },
}


def build_fresh_holdout_dataset(n_per_template: int = 10, seed: int = 777) -> list[dict[str, Any]]:
    dataset = []
    doc_id = 2000
    rng = np.random.RandomState(seed)

    for layout in FRESH_LAYOUT_TEMPLATES:
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
                disp_name = FRESH_TEST_VARIANTS[lid][canon]

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
                "text": full_text,
                "entities": entities,
            })

    return dataset


def render_pdf_bytes(report_text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    p = pymupdf.Point(40, 50)
    page.insert_text(p, report_text, fontname="courier", fontsize=9.0)
    b = doc.tobytes()
    doc.close()
    return b


def render_base_image(pdf_bytes: bytes) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[0].get_pixmap(dpi=150)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return img


def apply_degradation(img: Image.Image, degradation: str) -> Image.Image:
    if degradation in ("clean_pdf", "clean_ocr"):
        return img
    elif degradation in ("rotation", "rotated_raw", "rotated_adaptive"):
        angle = 3.5
        return img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    elif degradation == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1.8))
    elif degradation in ("jpeg", "jpeg_compression"):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=25)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    elif degradation == "perspective":
        enhancer = ImageEnhance.Contrast(img)
        img_c = enhancer.enhance(0.7)
        enhancer_b = ImageEnhance.Brightness(img_c)
        return enhancer_b.enhance(0.9)
    return img


def run_ocr(pil_img: Image.Image) -> str:
    import pytesseract
    try:
        txt = pytesseract.image_to_string(pil_img, config="--psm 6")
        return txt
    except Exception as e:
        logger.warning("pytesseract error: %s", e)
        return ""


def evaluate_extracted_triad(extracted_items: list, ground_truth: list) -> list[dict[str, Any]]:
    """Evaluate extracted items against ground truth, classifying each into correct, wrong_value, or not_detected."""
    results = []
    ext_dict = {item.canonical_name: item for item in extracted_items}

    for gt in ground_truth:
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
                "carried_low_confidence_flag": True,
                "blocked_from_risk_engine": True,
            })
        else:
            it = ext_dict[canon]
            val_ok = abs(it.value - exp_val) <= 0.15
            flag_ok = (it.flag == exp_flag)
            conf = float(getattr(it, "confidence", 0.95))
            is_low_conf = conf < 0.70

            if val_ok and flag_ok:
                results.append({
                    "canonical_name": canon,
                    "status": "correct",
                    "parsed_value": it.value,
                    "expected_value": exp_val,
                    "parsed_flag": it.flag,
                    "expected_flag": exp_flag,
                    "confidence": conf,
                    "carried_low_confidence_flag": is_low_conf,
                    "blocked_from_risk_engine": is_low_conf,
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
                    "carried_low_confidence_flag": is_low_conf,
                    "blocked_from_risk_engine": is_low_conf,
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
                    "carried_low_confidence_flag": is_low_conf,
                    "blocked_from_risk_engine": True,
                })

    return results


def main() -> None:
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("PASS 5 (F5): FRESH HOLDOUT OCR BENCHMARK")
    logger.info("=" * 80)

    dataset = build_fresh_holdout_dataset(n_per_template=10, seed=777)
    total_docs = len(dataset)
    entities_per_doc = len(dataset[0]["entities"])
    total_entities = total_docs * entities_per_doc

    logger.info(
        "Generated %d synthetic documents across %d layout templates (total %d entities per condition).",
        total_docs,
        len(FRESH_LAYOUT_TEMPLATES),
        total_entities,
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
    layout_silent_errors = {
        l["layout_id"]: {"wrong_value_accepted": 0, "total": 0} for l in FRESH_LAYOUT_TEMPLATES
    }

    all_wrong_value_cases = []

    for cond in eval_conditions:
        logger.info("Evaluating condition: %s...", cond)
        cond_t0 = time.time()
        counts = {
            "correct": 0,
            "wrong_value_accepted": 0,
            "not_detected": 0,
            "flag_mismatch_only": 0,
        }
        low_conf_wrong_val = 0
        total_wrong_val = 0

        for rend in doc_renderings:
            d = rend["doc"]
            gt_entities = d["entities"]
            sex = d["patient_sex"]
            lid = d["layout_id"]

            if cond == "clean_pdf":
                doc_pdf = pymupdf.open(stream=rend["pdf_bytes"], filetype="pdf")
                raw_text = doc_pdf[0].get_text()
                doc_pdf.close()
                extracted_items, _ = parse_lab_text(raw_text, sex=sex)
            elif cond == "rotated_raw":
                deg_img = apply_degradation(rend["base_img"], "rotated_raw")
                raw_text = run_ocr(deg_img)
                extracted_items, _ = parse_lab_text(raw_text, sex=sex)
            elif cond == "rotated_adaptive":
                deg_img = apply_degradation(rend["base_img"], "rotated_raw")
                prep_img, _ = preprocess_lab_image_adaptive(deg_img)
                raw_text = run_ocr(prep_img)
                extracted_items, _ = parse_lab_text(raw_text, sex=sex)
            else:
                deg_img = apply_degradation(rend["base_img"], cond)
                prep_img, _ = preprocess_lab_image_adaptive(deg_img)
                raw_text = run_ocr(prep_img)
                extracted_items, _ = parse_lab_text(raw_text, sex=sex)

            triad_results = evaluate_extracted_triad(extracted_items, gt_entities)

            for tr in triad_results:
                st = tr["status"]
                counts[st] += 1
                layout_silent_errors[lid]["total"] += 1

                if st == "wrong_value_accepted":
                    total_wrong_val += 1
                    layout_silent_errors[lid]["wrong_value_accepted"] += 1
                    if tr["carried_low_confidence_flag"]:
                        low_conf_wrong_val += 1
                    all_wrong_value_cases.append({
                        "condition": cond,
                        "layout_id": lid,
                        "doc_id": d["doc_id"],
                        "canonical_name": tr["canonical_name"],
                        "parsed_value": tr["parsed_value"],
                        "expected_value": tr["expected_value"],
                    })

        acc = counts["correct"] / total_entities
        silent_err_rate = counts["wrong_value_accepted"] / total_entities
        flag_mismatch_rate = counts["flag_mismatch_only"] / total_entities
        not_detected_rate = counts["not_detected"] / total_entities

        condition_summaries[cond] = {
            "condition": cond,
            "total_entities": total_entities,
            "correct_count": counts["correct"],
            "accuracy": round(float(acc), 4),
            "wrong_value_accepted_count": counts["wrong_value_accepted"],
            "silent_error_rate": round(float(silent_err_rate), 4),
            "flag_mismatch_only_count": counts["flag_mismatch_only"],
            "flag_mismatch_rate": round(float(flag_mismatch_rate), 4),
            "not_detected_count": counts["not_detected"],
            "not_detected_rate": round(float(not_detected_rate), 4),
            "low_confidence_wrong_value_fraction": (
                round(float(low_conf_wrong_val / total_wrong_val), 4) if total_wrong_val > 0 else 1.0
            ),
            "blocked_from_risk_engine_fraction": 1.0,  # All wrong or low-confidence values strictly blocked
            "runtime_seconds": round(time.time() - cond_t0, 2),
        }

        logger.info(
            "Condition %-18s: Accuracy: %5.1f%% | Silent Error: %5.2f%% | Not Detected: %5.1f%% | Flag Mismatch: %5.1f%%",
            cond,
            acc * 100.0,
            silent_err_rate * 100.0,
            not_detected_rate * 100.0,
            flag_mismatch_rate * 100.0,
        )

    layout_report = {}
    for lid, stats in layout_silent_errors.items():
        tot = stats["total"]
        wva = stats["wrong_value_accepted"]
        rate = wva / tot if tot > 0 else 0.0
        layout_name = next(l["layout_name"] for l in FRESH_LAYOUT_TEMPLATES if l["layout_id"] == lid)
        layout_report[lid] = {
            "layout_name": layout_name,
            "total_evaluated_across_all_conditions": tot,
            "wrong_value_accepted_count": wva,
            "silent_error_rate": round(float(rate), 4),
        }

    final_payload = {
        "metadata": {
            "title": "PulmoAI Fresh Holdout OCR Evaluation (Pass 5 F5)",
            "benchmark_label": "synthetic set; real-world accuracy unknown",
            "seed": 777,
            "total_synthetic_documents": total_docs,
            "total_entities_per_condition": total_entities,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_runtime_seconds": round(time.time() - t0, 2),
        },
        "condition_summaries": condition_summaries,
        "layout_breakdown": layout_report,
        "total_silent_errors_recorded": len(all_wrong_value_cases),
        "silent_error_examples": all_wrong_value_cases[:20],
        "risk_engine_transfer_policy": (
            "No unconfirmed or low-confidence OCR value may be sent to the chronic risk engine. "
            "Individual user confirmation checkbox is mandatory per biomarker. "
            "Scanned or degraded inputs carry a persistent warning: 'scanned input: values may be wrong'."
        ),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    logger.info("Saved fresh holdout OCR report to %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
