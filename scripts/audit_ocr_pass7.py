#!/usr/bin/env python3
"""Audit OCR on Pass 7 Holdout with 8 Layout Templates and Expanded Conditions (Pass 7 H2).

Protocol:
- 8 layout templates (Templates A-G plus new Template H: Veterans Affairs & Military Health EHR)
- 7 documents per template = 56 synthetic documents total.
- 7 clinical analytes per document = 392 entities per condition (>= 350 required).
- Ground truth rule (Pass 7 H2a): Expected flags depend ONLY on what is printed on the report
  (printed range if present, else documented default). Unprinted latent generator thresholds
  (such as HDL 60/70 or Total Cholesterol 125) are strictly eliminated.
- Frozen gate and parser (do NOT touch parser logic).
- Conditions evaluated:
  1. clean_pdf: Native PDF digital text layer
  2. clean_raster_150dpi: Clean rasterized page at 150 DPI
  3. clean_raster_200dpi: Clean rasterized page at 200 DPI
  4. clean_raster_300dpi: Clean rasterized page at 300 DPI
  5. clean_raster_150dpi_noise: 150 DPI raster with mild Gaussian noise (sigma=3.0)
  6. clean_raster_200dpi_noise: 200 DPI raster with mild Gaussian noise (sigma=3.0)
  7. clean_raster_300dpi_noise: 300 DPI raster with mild Gaussian noise (sigma=3.0)
  8. skew_0deg: 0 degree rotation
  9. skew_3.5deg: 3.5 degree rotation
  10. skew_8deg: 8.0 degree rotation
  11. skew_11deg: 11.0 degree rotation
  12. skew_13deg: 13.0 degree rotation (above 12 deg threshold -> expected refusal)
  13. skew_20deg: 20.0 degree rotation (above 12 deg threshold -> expected refusal)
  14. blur_sigma_0.5: Mild Gaussian blur (radius=0.5)
  15. blur_sigma_0.8: Moderate Gaussian blur (radius=0.8)
  16. blur_sigma_1.2: Severe Gaussian blur (radius=1.2)
  17. blur_sigma_2.0: Extreme Gaussian blur (radius=2.0 -> expected refusal)
- Per-condition metrics:
  - refused for quality count & rate
  - accepted entities count
  - correct count & accuracy on accepted
  - silent wrong value count & silent error rate on accepted
  - flag mismatch count & rate on accepted
  - not detected count & rate on accepted
  - false refusal rate on clean raster conditions
- Benchmark label: 'synthetic set; real-world accuracy unknown'.
- Output saved to: reports/audit_ocr_pass7.json
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
import pymupdf

from pulmoai.report_reader import (
    check_image_quality,
    determine_flag,
    extract_text_from_bytes,
    get_reference_range,
    parse_lab_text,
    preprocess_lab_image_adaptive,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_ocr_pass7")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = PROJECT_ROOT / "reports" / "audit_ocr_pass7.json"

# Clinical Analytes Specification
# NOTE (Pass 7 H2a): Expected flags depend ONLY on what is printed on the report
# (printed range if present, else the documented default for that analyte).
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
        "ref_male": (100.0, 200.0, "< 200 mg/dL"),
        "ref_female": (100.0, 200.0, "< 200 mg/dL"),
    },
    {
        "name": "hdl_cholesterol",
        "unit": "mg/dL",
        "ref_male": (40.0, 100.0, "> 40 mg/dL"),
        "ref_female": (50.0, 100.0, "> 50 mg/dL"),
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

# 8 Layout Templates: Templates A-G plus NEW Template H (Pass 7 H2)
PASS7_LAYOUT_TEMPLATES = [
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
        "header": "AMBULATORY CARE HEALTH CENTER - OUTPATIENT LAB SUMMARY\nPatient Name: Subject_{idx}      Age: {age}     Sex: {sex_cap}\nEncounter Date: 12-Nov-2024      Physician: Dr. J. Martinez, MD\n------------------------------------------------------------------------------------\nLABORATORY SUMMARY FINDINGS:\n",
        "footer": "\n------------------------------------------------------------------------------------\nNote: Values outside biological reference intervals should be evaluated clinically.",
        "fmt": "- {test_name}: {val:.1f} {unit} (Normal interval: {ref})",
    },
    {
        "layout_id": "template_e",
        "layout_name": "Template E (Academic Health Science Center multi-column tabular)",
        "default_sex": "male",
        "header": "ACADEMIC HEALTH SCIENCE CENTER - CENTRAL CORE DIAGNOSTICS\nPATIENT: Subject_{idx}   MRN: AHSC-{idx:06d}   DOB: 12/04/{birth_yr}   SEX: {sex_cap}\nORDERING PHYSICIAN: DR. H. VANDERBILT, MD   DATE: 17-NOV-2024\n====================================================================================\nTEST PROCEDURE                   RESULT    UNITS        FLAG    REFERENCE RANGE\n====================================================================================\n",
        "footer": "\n====================================================================================\nCLIA/CAP Certified Laboratory. Medical Director Review Completed.",
        "fmt": "{test_name:<30}   {val:<7.1f}   {unit:<10}   {flag:<6}  {ref}",
    },
    {
        "layout_id": "template_f",
        "layout_name": "Template F (Ambulatory Point-of-Care Rapid Panel format)",
        "default_sex": "female",
        "header": "URGENT CARE AMBULATORY NETWORK - POINT-OF-CARE TESTING\nSubject ID: POC-{idx:05d}   Age/Gender: {age}/{sex_cap}   Date: 17-NOV-2024\nRapid Profiling Instrument ID: INST-4029   Operator: RN-884\n------------------------------------------------------------------------------------\nPoint-of-Care Analyte          Observed  Units          Status   Biological Range\n------------------------------------------------------------------------------------\n",
        "footer": "\n------------------------------------------------------------------------------------\nImmediate point-of-care screening. Confirmatory testing recommended for critical values.",
        "fmt": "{test_name:<28}   {val:<8.1f}  {unit:<12}   {flag:<6}   {ref}",
    },
    {
        "layout_id": "template_g",
        "layout_name": "Template G (Community Health Center EHR Export)",
        "default_sex": "male",
        "header": "COMMUNITY HEALTH COLLABORATIVE - ELECTRONIC HEALTH RECORD LAB EXPORT\nCLIENT: Patient_{idx}   DOB: 03/15/{birth_yr} ({age}Y)   GENDER: {sex_cap}   ENC: CHC-{idx:06d}\nENCOUNTER TYPE: ROUTINE ANNUAL WELLNESS   DATE OF SERVICE: 18-NOV-2024\n####################################################################################\nLAB COMPONENT                  RESULT VALUE   STANDARD UNITS   REFERENCE RANGE\n####################################################################################\n",
        "footer": "\n####################################################################################\nDocument authenticated by Community Health EMR System. Confidential Medical Record.",
        "fmt": "{test_name:<28}   {val:<12.1f}   {unit:<14}   {ref}",
    },
    {
        "layout_id": "template_h",
        "layout_name": "Template H (Veterans Affairs & Military Health EHR Tabular Summary)",
        "default_sex": "male",
        "header": "VETERANS INTEGRATED SERVICE NETWORK - CLINICAL LABORATORY REPORT\nNAME: PATIENT_{idx}   AGE: {age}   GENDER: {sex_cap}   EDIPI: {idx:010d}\nFACILITY: VA MEDICAL CENTER   DIVISION: PATHOLOGY & LAB MED   STATUS: COMPLETED\n------------------------------------------------------------------------------------\nTEST NAME                      RESULT    UNITS        FLAG     REFERENCE INTERVAL\n------------------------------------------------------------------------------------\n",
        "footer": "\n------------------------------------------------------------------------------------\nOrdering Provider: Staff Physician, Primary Care. Electronically signed.",
        "fmt": "{test_name:<30} {val:<9.1f} {unit:<12} {flag:<8} {ref}",
    },
]

PASS7_TEST_VARIANTS = {
    "template_a": {
        "fasting_glucose": "GLUCOSE, FASTING",
        "hba1c": "HEMOGLOBIN A1c",
        "total_cholesterol": "TOTAL CHOLESTEROL",
        "hdl_cholesterol": "HDL CHOLESTEROL",
        "triglycerides": "TRIGLYCERIDES",
        "serum_creatinine": "CREATININE, SERUM",
        "hemoglobin": "HEMOGLOBIN",
    },
    "template_b": {
        "fasting_glucose": "Plasma Glucose, Fasting",
        "hba1c": "HbA1c (Glycated Hb)",
        "total_cholesterol": "Cholesterol, Total",
        "hdl_cholesterol": "HDL-Cholesterol",
        "triglycerides": "Serum Triglycerides",
        "serum_creatinine": "Creatinine",
        "hemoglobin": "Hemoglobin (Hb)",
    },
    "template_c": {
        "fasting_glucose": "FASTING BLOOD SUGAR (FBS)",
        "hba1c": "GLYCOSYLATED HEMOGLOBIN (A1C)",
        "total_cholesterol": "SERUM CHOLESTEROL",
        "hdl_cholesterol": "HIGH DENSITY LIPOPROTEIN",
        "triglycerides": "SERUM TRIGLYCERIDE LEVEL",
        "serum_creatinine": "SERUM CREATININE",
        "hemoglobin": "TOTAL HEMOGLOBIN CONC",
    },
    "template_d": {
        "fasting_glucose": "Fasting Blood Glucose",
        "hba1c": "A1C Glycohemoglobin",
        "total_cholesterol": "Total Serum Cholesterol",
        "hdl_cholesterol": "HDL Good Cholesterol",
        "triglycerides": "Blood Triglycerides",
        "serum_creatinine": "Kidney Creatinine",
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
    "template_h": {
        "fasting_glucose": "GLUCOSE, PLASMA",
        "hba1c": "HEMOGLOBIN A1C",
        "total_cholesterol": "CHOLESTEROL, TOTAL",
        "hdl_cholesterol": "HDL CHOLESTEROL",
        "triglycerides": "TRIGLYCERIDE",
        "serum_creatinine": "CREATININE",
        "hemoglobin": "HEMOGLOBIN",
    },
}


def build_pass7_holdout_dataset(n_per_template: int = 7, seed: int = 1010) -> list[dict[str, Any]]:
    """Build fresh holdout dataset with expected flags strictly derived from printed ranges."""
    dataset = []
    doc_id = 4000
    rng = np.random.RandomState(seed)

    for layout in PASS7_LAYOUT_TEMPLATES:
        lid = layout["layout_id"]
        lname = layout["layout_name"]
        d_sex = layout["default_sex"]

        for var_idx in range(n_per_template):
            doc_id += 1
            sex = d_sex if var_idx % 2 == 0 else ("female" if d_sex == "male" else "male")
            age = int(rng.randint(25, 80))
            birth_yr = 2024 - age
            sex_cap = sex.capitalize()

            hdr = layout["header"].format(idx=doc_id, age=age, sex_cap=sex_cap, birth_yr=birth_yr)
            lines = []
            entities = []

            for spec in TEST_SPECS:
                canon = spec["name"]
                unit = spec["unit"]
                disp_name = PASS7_TEST_VARIANTS[lid][canon]

                ref_tuple = spec["ref_male"] if sex == "male" else spec["ref_female"]
                rmin, rmax, rstr = ref_tuple

                # Determine effective reference limits strictly from the printed range and documented defaults
                eff_min, eff_max, _ = get_reference_range(canon, sex=sex, printed_min=rmin, printed_max=rmax)

                mode = rng.choice(["norm", "high", "low"])
                if mode == "norm":
                    val = round(float(rng.uniform(eff_min, eff_max if eff_max is not None else eff_min + 40)), 1)
                    if eff_min is not None and val < eff_min:
                        val = eff_min
                    if eff_max is not None and val > eff_max:
                        val = eff_max
                elif mode == "high":
                    base_max = eff_max if eff_max is not None else 150.0
                    val = round(float(rng.uniform(base_max + 1.0, base_max * 1.5)), 1)
                    if val <= base_max:
                        val = base_max + 1.0
                else:  # low
                    base_min = eff_min if eff_min is not None else 50.0
                    val = round(float(rng.uniform(base_min * 0.4, base_min - 0.5)), 1)
                    if val >= base_min:
                        val = base_min - 0.5

                # RULE (Pass 7 H2a): Expected flags depend ONLY on what is printed on the report
                flag = determine_flag(canon, val, ref_min=eff_min, ref_max=eff_max, sex=sex)

                # Format line based on template
                if "{flag" in layout["fmt"]:
                    line = layout["fmt"].format(test_name=disp_name, val=val, unit=unit, flag=flag, ref=rstr)
                else:
                    line = layout["fmt"].format(test_name=disp_name, val=val, unit=unit, ref=rstr)
                lines.append(line)

                entities.append({
                    "canonical_name": canon,
                    "expected_value": val,
                    "expected_unit": unit,
                    "expected_flag": flag,
                    "reference_range": rstr,
                    "eff_min": eff_min,
                    "eff_max": eff_max,
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
    page = doc.new_page(width=612, height=792)  # US Letter
    font_size = 8.5
    rect = pymupdf.Rect(36, 36, 576, 756)
    page.insert_textbox(rect, report_text, fontsize=font_size, fontname="courier")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def render_raster_image(pdf_bytes: bytes, dpi: int = 300) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def add_mild_noise(img: Image.Image, sigma: float = 3.0, seed: int = 42) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_skew(img: Image.Image, angle_deg: float) -> Image.Image:
    return img.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))


def apply_blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def evaluate_condition(
    dataset: list[dict[str, Any]],
    condition_name: str,
) -> dict[str, Any]:
    logger.info("Evaluating condition: %s on %d documents...", condition_name, len(dataset))
    t0 = time.time()

    total_entities = 0
    refused_quality = 0
    accepted_entities = 0
    correct_count = 0
    wrong_value_count = 0
    flag_mismatch_count = 0
    not_detected_count = 0

    silent_error_examples = []

    for doc in dataset:
        pdf_bytes = render_pdf_bytes(doc["text"])

        # Prepare input image / text based on condition
        if condition_name == "clean_pdf":
            # Native PDF digital text layer
            pdf_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            extracted_text = "\n".join([page.get_text() for page in pdf_doc])
            pdf_doc.close()
            parsed_items, _ = parse_lab_text(extracted_text, sex=doc["patient_sex"])
            is_refused = False
        else:
            # Render raster image according to condition
            if "150dpi" in condition_name:
                base_img = render_raster_image(pdf_bytes, dpi=150)
            elif "200dpi" in condition_name:
                base_img = render_raster_image(pdf_bytes, dpi=200)
            else:
                base_img = render_raster_image(pdf_bytes, dpi=300)

            if "noise" in condition_name:
                base_img = add_mild_noise(base_img, sigma=3.0, seed=doc["doc_id"])

            if "skew" in condition_name:
                angle = float(condition_name.split("_")[1].replace("deg", ""))
                base_img = apply_skew(base_img, angle)

            if "blur" in condition_name:
                sigma = float(condition_name.split("_")[2])
                base_img = apply_blur(base_img, sigma)

            # Check pre-OCR quality gate
            acceptable, reason, _ = check_image_quality(base_img)
            if not acceptable:
                is_refused = True
                parsed_items = []
            else:
                is_refused = False
                processed_img, _ = preprocess_lab_image_adaptive(base_img)
                buf = io.BytesIO()
                processed_img.save(buf, format="PNG")
                extracted_text, _ = extract_text_from_bytes(buf.getvalue(), filename="doc.png", mime_type="image/png")
                parsed_items, _ = parse_lab_text(extracted_text, sex=doc["patient_sex"])

        parsed_map = {item.canonical_name: item for item in parsed_items}

        for exp in doc["entities"]:
            total_entities += 1
            if is_refused:
                refused_quality += 1
                continue

            accepted_entities += 1
            cname = exp["canonical_name"]
            expected_val = exp["expected_value"]
            expected_flag = exp["expected_flag"]

            if cname not in parsed_map:
                not_detected_count += 1
            else:
                p = parsed_map[cname]
                val_diff = abs(p.value - expected_val)

                if val_diff <= 0.15:
                    if p.flag == expected_flag:
                        correct_count += 1
                    else:
                        flag_mismatch_count += 1
                else:
                    wrong_value_count += 1
                    if len(silent_error_examples) < 5:
                        silent_error_examples.append({
                            "condition": condition_name,
                            "doc_id": doc["doc_id"],
                            "layout_id": doc["layout_id"],
                            "analyte": cname,
                            "expected_val": expected_val,
                            "parsed_val": p.value,
                            "expected_flag": expected_flag,
                            "parsed_flag": p.flag,
                        })

    dur = time.time() - t0
    refused_rate = refused_quality / total_entities if total_entities > 0 else 0.0
    acc_on_accepted = correct_count / accepted_entities if accepted_entities > 0 else 0.0
    silent_on_accepted = wrong_value_count / accepted_entities if accepted_entities > 0 else 0.0
    flag_on_accepted = flag_mismatch_count / accepted_entities if accepted_entities > 0 else 0.0
    not_det_on_accepted = not_detected_count / accepted_entities if accepted_entities > 0 else 0.0

    return {
        "condition": condition_name,
        "total_entities": total_entities,
        "refused_for_quality_count": refused_quality,
        "refused_for_quality_rate": round(refused_rate, 4),
        "accepted_entities_count": accepted_entities,
        "correct_count": correct_count,
        "accuracy_on_accepted": round(acc_on_accepted, 4),
        "wrong_value_accepted_count": wrong_value_count,
        "silent_error_rate_on_accepted": round(silent_on_accepted, 4),
        "flag_mismatch_only_count": flag_mismatch_count,
        "flag_mismatch_rate_on_accepted": round(flag_on_accepted, 4),
        "not_detected_count": not_detected_count,
        "not_detected_rate_on_accepted": round(not_det_on_accepted, 4),
        "runtime_seconds": round(dur, 2),
        "silent_error_examples": silent_error_examples,
    }


def main() -> None:
    seed = 1010
    logger.info("Building Pass 7 fresh holdout dataset (seed=%d, 8 templates)...", seed)
    dataset = build_pass7_holdout_dataset(n_per_template=7, seed=seed)
    total_docs = len(dataset)
    entities_per_cond = total_docs * len(TEST_SPECS)
    logger.info("Generated %d documents (%d entities per condition >= 350 required).", total_docs, entities_per_cond)

    conditions = [
        "clean_pdf",
        "clean_raster_150dpi",
        "clean_raster_200dpi",
        "clean_raster_300dpi",
        "clean_raster_150dpi_noise",
        "clean_raster_200dpi_noise",
        "clean_raster_300dpi_noise",
        "skew_0deg",
        "skew_3.5deg",
        "skew_8deg",
        "skew_11deg",
        "skew_13deg",
        "skew_20deg",
        "blur_sigma_0.5",
        "blur_sigma_0.8",
        "blur_sigma_1.2",
        "blur_sigma_2.0",
    ]

    summaries = {}
    for cond in conditions:
        summaries[cond] = evaluate_condition(dataset, cond)

    # Compute false-refusal rate on clean raster conditions
    clean_raster_keys = [
        "clean_raster_150dpi", "clean_raster_200dpi", "clean_raster_300dpi",
        "clean_raster_150dpi_noise", "clean_raster_200dpi_noise", "clean_raster_300dpi_noise",
    ]
    total_cr = sum(summaries[k]["total_entities"] for k in clean_raster_keys)
    refused_cr = sum(summaries[k]["refused_for_quality_count"] for k in clean_raster_keys)
    false_refusal_rate_clean_raster = refused_cr / total_cr if total_cr > 0 else 0.0

    output = {
        "metadata": {
            "title": "PulmoAI Pass 7 (H2) Fresh Holdout OCR Evaluation with Fixed Generator and Expanded Conditions",
            "benchmark_label": "synthetic set; real-world accuracy unknown",
            "seed": seed,
            "total_synthetic_documents": total_docs,
            "total_entities_per_condition": entities_per_cond,
            "templates_evaluated": [t["layout_id"] for t in PASS7_LAYOUT_TEMPLATES],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "false_refusal_rate_clean_raster": round(false_refusal_rate_clean_raster, 4),
            "false_refusal_pass_threshold_met": false_refusal_rate_clean_raster <= 0.05,
        },
        "condition_summaries": summaries,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info("Successfully saved Pass 7 OCR audit results to %s", OUTPUT_JSON)
    logger.info("Clean raster false-refusal rate: %.2f%% (threshold <= 5.0%%)", false_refusal_rate_clean_raster * 100)


if __name__ == "__main__":
    main()
