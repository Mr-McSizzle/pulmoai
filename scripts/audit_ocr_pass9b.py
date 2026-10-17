#!/usr/bin/env python3
"""Audit OCR on Pass 9b Fresh Holdout with 11 Layout Templates and Realistic Degradations (Pass 9b L3).

Protocol:
- 11 layout templates:
  - 8 development templates with generic naming (Templates A-H)
  - 3 brand new template families not used during development:
    - Template I: Tabular Bordered Grid (template_i_bordered_table)
    - Template J: Two-Column Clinical Summary (template_j_two_column)
    - Template K: Compact Outpatient EHR Ledger (template_k_compact_ledger)
- 6 documents per template = 66 synthetic documents total.
- 7 clinical analytes per document = 462 entities per condition (>= 350 required).
- Ground truth rule: Expected flags depend ONLY on what is printed on the report
  (printed range if present, else documented default). Unprinted latent generator thresholds
  are strictly eliminated.
- Single-pass production multi-factor preprocessing via check_image_quality and extract_text_from_bytes.
- Conditions evaluated:
  1. clean_pdf: Native PDF digital text layer
  2. clean_raster_150dpi: Clean rasterized page at 150 DPI
  3. clean_raster_200dpi: Clean rasterized page at 200 DPI
  4. clean_raster_300dpi: Clean rasterized page at 300 DPI
  5. mild_jpeg_q50: Realistic JPEG compression at Quality 50
  6. mild_paper_grain: Additive fibrous paper texture
  7. mild_scanner_sensor: Flatbed scanner CCD sensor noise + columnar streaks
  8. mild_phone_lighting: Mobile phone camera overhead lighting gradient + shadow
  9. mild_faded_contrast: Low toner / faded print contrast reduction
  10. mild_perspective_tilt: Slight 2.5-degree perspective tilt
  11. catastrophic_severe_blur: Severe defocus blur (radius=3.5 -> expected refusal)
  12. catastrophic_extreme_skew: 18.0-degree rotation (above 15 deg threshold -> expected refusal)
  13. catastrophic_noise_flood: Destructive sensor noise flood (sigma=15.0 -> expected refusal)
- Per-condition metrics:
  - pages evaluated, refused, refusal rate
  - accepted entities, correct, accuracy on accepted
  - not detected, silent wrong values, flag mismatches
  - false refusal rate on mild-degradation pages with 95% Wilson score confidence intervals over pages
- Benchmark label: 'synthetic set; real-world accuracy unknown'.
- Output saved to: reports/audit_ocr_pass9b.json
"""

from __future__ import annotations

import io
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pymupdf

from pulmoai.report_reader import (
    check_image_quality,
    determine_flag,
    extract_text_from_bytes,
    get_reference_range,
    parse_lab_text,
)
from scripts.audit_ocr_pass8 import (
    LAYOUT_TEMPLATES as BASE_TEMPLATES,
    TEST_SPECS,
    TEST_VARIANTS as BASE_VARIANTS,
    render_pdf_bytes,
    render_raster_image,
)
from scripts.audit_ocr_realistic_dev import (
    apply_jpeg_compression,
    apply_paper_grain,
    apply_scanner_sensor_noise,
    apply_phone_camera_lighting_shadow,
    apply_low_contrast,
    apply_slight_perspective,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_ocr_pass9b")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = PROJECT_ROOT / "reports" / "audit_ocr_pass9b.json"


def wilson_score_interval(successes: int, total: int, z: float = 1.95996) -> tuple[float, float]:
    """Compute two-sided 95% Wilson score confidence interval."""
    if total <= 0:
        return (0.0, 0.0)
    p_hat = float(successes) / float(total)
    denominator = 1.0 + (z ** 2) / float(total)
    center = (p_hat + (z ** 2) / (2.0 * float(total))) / denominator
    margin = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) / float(total)) + ((z ** 2) / (4.0 * (float(total) ** 2))))
        / denominator
    )
    ci_lower = max(0.0, float(round(center - margin, 4)))
    ci_upper = min(1.0, float(round(center + margin, 4)))
    return (ci_lower, ci_upper)


# ---------------------------------------------------------------------------
# 3 Brand New Template Families Not Used During Development
# ---------------------------------------------------------------------------

NEW_LAYOUT_TEMPLATES: list[dict[str, Any]] = [
    # Family 1: Tabular Bordered Grid
    {
        "layout_id": "template_i",
        "layout_name": "Regional Health Clinical Laboratory - Tabular Bordered Grid",
        "default_sex": "male",
        "header": (
            "+-----------------------------------------------------------------------------+\n"
            "|                     REGIONAL HEALTH CLINICAL LABORATORY                     |\n"
            "|                      PATIENT BIOCHEMISTRY REPORT                            |\n"
            "+-----------------------------------------------------------------------------+\n"
            "| Patient ID: PAT_{idx:<6}     Age: {age:<3}     Sex: {sex_cap:<6}     Year: {birth_yr}         |\n"
            "+--------------------+------------+------------+---------------+--------------+\n"
            "| TEST DESCRIPTION   | RESULT     | UNITS      | REFERENCE     | STATUS       |\n"
            "+--------------------+------------+------------+---------------+--------------+"
        ),
        "fmt": "| {test_name:<18} | {val:>10.1f} | {unit:<10} | {ref:<13} | {flag:<12} |",
        "footer": (
            "+--------------------+------------+------------+---------------+--------------+\n"
            "| End of Report - Electronically Verified by Automated Clinical Chemistry Sys |\n"
            "+-----------------------------------------------------------------------------+"
        ),
    },
    # Family 2: Two-Column Clinical Summary
    {
        "layout_id": "template_j",
        "layout_name": "Urban Academic Medical Center - Two-Column Pathology Summary",
        "default_sex": "female",
        "header": (
            "================================================================================\n"
            "                       URBAN ACADEMIC MEDICAL CENTER\n"
            "                       DIVISION OF CLINICAL PATHOLOGY\n"
            "================================================================================\n"
            "PATIENT IDENTIFIER: MED-{idx}          AGE: {age}      GENDER: {sex_cap}\n"
            "COLLECTION: 2024-03-15 08:30          LOCATION: OUTPATIENT CLINIC\n"
            "--------------------------------------------------------------------------------\n"
            "ANALYTE / PANEL                VALUE    UNITS        INTERVAL           FLAG\n"
            "--------------------------------------------------------------------------------"
        ),
        "fmt": ":: {test_name:<26} {val:>7.1f}  {unit:<10}   [{ref}]        {flag}",
        "footer": (
            "--------------------------------------------------------------------------------\n"
            "CLINICAL INTERPRETATION NOTE: Review abnormal findings with treating physician.\n"
            "================================================================================"
        ),
    },
    # Family 3: Compact Outpatient EHR Ledger
    {
        "layout_id": "template_k",
        "layout_name": "Ambulatory Health Services - Compact EHR Laboratory Ledger",
        "default_sex": "male",
        "header": (
            "[CLINICAL LAB LEDGER - AMBULATORY SERVICE]\n"
            "ID: AMB-{idx} | AGE: {age} | SEX: {sex_cap} | DOB: {birth_yr}\n"
            "----------------------------------------------------------------------"
        ),
        "fmt": "{test_name:<22} : {val:>6.1f} {unit:<8} (Ref: {ref:<14}) [{flag}]",
        "footer": (
            "----------------------------------------------------------------------\n"
            "[REPORT COMPLETED - LABORATORY INFORMATION SYSTEM ARCHIVE]"
        ),
    },
]

ALL_TEMPLATES = BASE_TEMPLATES + NEW_LAYOUT_TEMPLATES

ALL_VARIANTS = dict(BASE_VARIANTS)
ALL_VARIANTS["template_i"] = {
    "fasting_glucose": "Glucose Fasting",
    "hba1c": "Hemoglobin A1c",
    "total_cholesterol": "Cholesterol Total",
    "hdl_cholesterol": "HDL Cholesterol",
    "triglycerides": "Triglycerides",
    "serum_creatinine": "Creatinine Serum",
    "hemoglobin": "Hemoglobin",
}
ALL_VARIANTS["template_j"] = {
    "fasting_glucose": "Fasting Blood Sugar",
    "hba1c": "Glycohemoglobin A1c",
    "total_cholesterol": "Serum Cholesterol",
    "hdl_cholesterol": "Direct HDL Cholesterol",
    "triglycerides": "Serum Triglycerides",
    "serum_creatinine": "Serum Creatinine",
    "hemoglobin": "Total Hemoglobin",
}
ALL_VARIANTS["template_k"] = {
    "fasting_glucose": "FBS (Glucose)",
    "hba1c": "HbA1c (Glycated)",
    "total_cholesterol": "Total Chol",
    "hdl_cholesterol": "HDL-C",
    "triglycerides": "Triglyceride",
    "serum_creatinine": "Creatinine",
    "hemoglobin": "Hemoglobin (Hb)",
}


def build_pass9b_holdout(n_per_template: int = 6, seed: int = 92026) -> list[dict[str, Any]]:
    """Build fresh holdout dataset across 11 templates with strict printed range ground truth."""
    dataset = []
    doc_id = 9000
    rng = np.random.RandomState(seed)

    for layout in ALL_TEMPLATES:
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
                disp_name = ALL_VARIANTS[lid][canon]

                ref_tuple = spec["ref_male"] if sex == "male" else spec["ref_female"]
                rmin, rmax, rstr = ref_tuple

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

                flag = determine_flag(canon, val, ref_min=eff_min, ref_max=eff_max, sex=sex)

                if "{flag" in layout["fmt"]:
                    line = layout["fmt"].format(test_name=disp_name, val=val, unit=unit, flag=flag, ref=rstr)
                else:
                    line = layout["fmt"].format(test_name=disp_name, val=val, unit=unit, ref=rstr)
                lines.append(line)

                entities.append({
                    "canonical_name": canon,
                    "expected_value": val,
                    "expected_flag": flag,
                    "expected_unit": unit,
                    "printed_range": rstr,
                    "ref_min": eff_min,
                    "ref_max": eff_max,
                })

            full_text = f"{hdr}\n" + "\n".join(lines) + f"\n{layout['footer']}"
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


def evaluate_condition_pass9b(
    dataset: list[dict[str, Any]],
    condition_name: str,
) -> dict[str, Any]:
    logger.info("Evaluating condition: %s on %d documents...", condition_name, len(dataset))
    t0 = time.time()

    total_pages = len(dataset)
    pages_refused = 0
    total_entities = 0
    accepted_entities = 0
    correct_count = 0
    wrong_value_count = 0
    flag_mismatch_count = 0
    not_detected_count = 0

    silent_error_examples = []

    for doc in dataset:
        pdf_bytes = render_pdf_bytes(doc["text"])

        if condition_name == "clean_pdf":
            pdf_doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            extracted_text = "\n".join([page.get_text() for page in pdf_doc])
            pdf_doc.close()
            parsed_items, _ = parse_lab_text(extracted_text, sex=doc["patient_sex"])
            is_refused = False
        else:
            if "150dpi" in condition_name:
                base_img = render_raster_image(pdf_bytes, dpi=150)
            elif "300dpi" in condition_name:
                base_img = render_raster_image(pdf_bytes, dpi=300)
            else:
                base_img = render_raster_image(pdf_bytes, dpi=200)

            # Apply specific degradation
            if condition_name == "mild_jpeg_q50":
                degraded = apply_jpeg_compression(base_img, quality=50)
            elif condition_name == "mild_paper_grain":
                degraded = apply_paper_grain(base_img, sigma=3.5, seed=doc["doc_id"])
            elif condition_name == "mild_scanner_sensor":
                degraded = apply_scanner_sensor_noise(base_img, seed=doc["doc_id"])
            elif condition_name == "mild_phone_lighting":
                degraded = apply_phone_camera_lighting_shadow(base_img, seed=doc["doc_id"])
            elif condition_name == "mild_faded_contrast":
                degraded = apply_low_contrast(base_img, alpha=0.60)
            elif condition_name == "mild_perspective_tilt":
                degraded = apply_slight_perspective(base_img, seed=doc["doc_id"])
            elif condition_name == "catastrophic_severe_blur":
                degraded = base_img.filter(ImageFilter.GaussianBlur(radius=3.5))
            elif condition_name == "catastrophic_extreme_skew":
                degraded = base_img.rotate(18.0, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
            elif condition_name == "catastrophic_noise_flood":
                arr = np.array(base_img, dtype=np.float32)
                rng = np.random.RandomState(doc["doc_id"])
                arr = np.clip(arr + rng.normal(0, 15.0, arr.shape), 0, 255).astype(np.uint8)
                degraded = Image.fromarray(arr)
            else:
                degraded = base_img

            # Quality gate check
            acceptable, reason, q_metrics = check_image_quality(degraded)
            if not acceptable:
                is_refused = True
                pages_refused += 1
                parsed_items = []
            else:
                is_refused = False
                buf = io.BytesIO()
                degraded.save(buf, format="PNG")
                extracted_text, method = extract_text_from_bytes(buf.getvalue(), filename="doc.png", mime_type="image/png")
                is_mild = (method == "tesseract_ocr_mild_degradation") or q_metrics.get("is_mildly_degraded", False)
                parsed_items, _ = parse_lab_text(extracted_text, sex=doc["patient_sex"], is_mildly_degraded=is_mild)

        parsed_map = {item.canonical_name: item for item in parsed_items}

        for exp in doc["entities"]:
            total_entities += 1
            if is_refused:
                continue

            accepted_entities += 1
            cname = exp["canonical_name"]
            expected_val = exp["expected_value"]
            expected_flag = exp["expected_flag"]

            if cname not in parsed_map:
                not_detected_count += 1
            else:
                item = parsed_map[cname]
                parsed_val = item.value
                parsed_flag = item.flag

                rel_err = abs(parsed_val - expected_val) / max(expected_val, 1e-6)
                if rel_err <= 0.01:
                    if parsed_flag == expected_flag:
                        correct_count += 1
                    else:
                        flag_mismatch_count += 1
                else:
                    wrong_value_count += 1
                    if len(silent_error_examples) < 5:
                        silent_error_examples.append({
                            "doc_id": doc["doc_id"],
                            "template": doc["layout_name"],
                            "analyte": cname,
                            "expected_value": expected_val,
                            "extracted_value": parsed_val,
                            "relative_error": round(rel_err, 4),
                            "expected_flag": expected_flag,
                            "extracted_flag": parsed_flag,
                        })

    dt = time.time() - t0
    refusal_rate = pages_refused / max(total_pages, 1)
    acc_on_accepted = correct_count / max(accepted_entities, 1)
    silent_rate = wrong_value_count / max(accepted_entities, 1)
    flag_rate = flag_mismatch_count / max(accepted_entities, 1)
    not_det_rate = not_detected_count / max(accepted_entities, 1)

    ci_refusal = wilson_score_interval(pages_refused, total_pages)
    ci_acc = wilson_score_interval(correct_count, accepted_entities)

    return {
        "condition": condition_name,
        "total_pages": total_pages,
        "pages_refused": pages_refused,
        "page_refusal_rate": round(refusal_rate, 4),
        "page_refusal_95ci": list(ci_refusal),
        "total_entities": total_entities,
        "accepted_entities": accepted_entities,
        "correct_count": correct_count,
        "accuracy_on_accepted": round(acc_on_accepted, 4),
        "accuracy_95ci": list(ci_acc),
        "wrong_value_count": wrong_value_count,
        "silent_error_rate_on_accepted": round(silent_rate, 4),
        "flag_mismatch_count": flag_mismatch_count,
        "flag_mismatch_rate_on_accepted": round(flag_rate, 4),
        "not_detected_count": not_detected_count,
        "not_detected_rate_on_accepted": round(not_det_rate, 4),
        "runtime_seconds": round(dt, 2),
        "silent_error_examples": silent_error_examples,
    }


def run_benchmark_pass9b() -> dict[str, Any]:
    dataset = build_pass9b_holdout(n_per_template=6, seed=92026)
    logger.info(
        "Built Pass 9b Fresh Holdout: %d documents across %d templates (%d entities per condition).",
        len(dataset),
        len(ALL_TEMPLATES),
        len(dataset) * 7,
    )

    conditions = [
        "clean_pdf",
        "clean_raster_150dpi",
        "clean_raster_200dpi",
        "clean_raster_300dpi",
        "mild_jpeg_q50",
        "mild_paper_grain",
        "mild_scanner_sensor",
        "mild_phone_lighting",
        "mild_faded_contrast",
        "mild_perspective_tilt",
        "catastrophic_severe_blur",
        "catastrophic_extreme_skew",
        "catastrophic_noise_flood",
    ]

    summaries = {}
    for cond in conditions:
        summaries[cond] = evaluate_condition_pass9b(dataset, cond)

    # Compute aggregate false refusal rate on mild degradation conditions
    mild_conditions = [
        "clean_raster_150dpi",
        "clean_raster_200dpi",
        "clean_raster_300dpi",
        "mild_jpeg_q50",
        "mild_paper_grain",
        "mild_scanner_sensor",
        "mild_phone_lighting",
        "mild_faded_contrast",
        "mild_perspective_tilt",
    ]
    mild_total_pages = sum(summaries[c]["total_pages"] for c in mild_conditions)
    mild_refused_pages = sum(summaries[c]["pages_refused"] for c in mild_conditions)
    mild_false_refusal_rate = mild_refused_pages / max(mild_total_pages, 1)
    mild_false_refusal_ci = wilson_score_interval(mild_refused_pages, mild_total_pages)

    out_obj = {
        "metadata": {
            "title": "Pass 9b Fresh Holdout OCR Evaluation with Realistic Degradations and Multi-Factor Gate",
            "benchmark_label": "synthetic set; real-world accuracy unknown",
            "seed": 92026,
            "total_synthetic_documents": len(dataset),
            "total_entities_per_condition": len(dataset) * 7,
            "templates_evaluated": [t["layout_id"] for t in ALL_TEMPLATES],
            "new_template_families": [
                "template_i_bordered_table (Tabular Bordered Grid)",
                "template_j_two_column (Two-Column Pathology Summary)",
                "template_k_compact_ledger (Compact Outpatient EHR Ledger)",
            ],
            "timestamp": "2026-10-16",
            "mild_conditions_evaluated": mild_conditions,
            "mild_pages_total": mild_total_pages,
            "mild_pages_refused": mild_refused_pages,
            "false_refusal_rate_mild_pages": round(mild_false_refusal_rate, 4),
            "false_refusal_rate_mild_pages_95ci": list(mild_false_refusal_ci),
        },
        "condition_summaries": summaries,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out_obj, f, indent=2)

    logger.info("Pass 9b OCR holdout benchmark completed and saved to %s", OUTPUT_JSON)
    return out_obj


if __name__ == "__main__":
    run_benchmark_pass9b()
