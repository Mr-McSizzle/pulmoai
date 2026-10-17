#!/usr/bin/env python3
"""OCR Gate Redesign Holdout Audit across 11 Templates with Independent Page Sets (Pass 9c M5).

Protocol:
- Evaluates 11 clinical report layout templates (Templates A-K).
- Uses SEPARATE page sets per condition (no pages reused across conditions).
- Objective condition naming (replaces generic 'catastrophic' label).
- Reports: detection rate, field error rate, silent error rate, rejection rate.
- Saves output to reports/audit_ocr_redesign_holdout.json.
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

from pulmoai.report_reader import (
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
from scripts.audit_ocr_pass9b import NEW_LAYOUT_TEMPLATES
from scripts.audit_ocr_realistic_dev import (
    apply_jpeg_compression,
    apply_paper_grain,
    apply_scanner_sensor_noise,
    apply_phone_camera_lighting_shadow,
    apply_low_contrast,
    apply_slight_perspective,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ocr_holdout_redesign")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = PROJECT_ROOT / "reports" / "audit_ocr_redesign_holdout.json"

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


def build_condition_dataset(n_per_template: int = 4, seed: int = 4000) -> list[dict[str, Any]]:
    dataset = []
    doc_id = seed * 100
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
                else:
                    base_min = eff_min if eff_min is not None else 50.0
                    val = round(float(rng.uniform(base_min * 0.6, base_min - 0.5)), 1)
                    if val >= base_min:
                        val = base_min - 0.5

                flag = determine_flag(canon, val, ref_min=eff_min, ref_max=eff_max)
                line_str = layout["fmt"].format(
                    test_name=disp_name,
                    val=val,
                    unit=unit,
                    flag=flag,
                    ref=rstr,
                )
                lines.append(line_str)
                entities.append({
                    "canonical_name": canon,
                    "expected_value": val,
                    "expected_unit": unit,
                    "expected_flag": flag,
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


CONDITIONS = [
    ("clean_pdf", 4001, "Digital PDF text layer (native)", "clean_digital"),
    ("clean_raster_150dpi", 4002, "Rasterized page at 150 DPI", "clean_raster"),
    ("clean_raster_200dpi", 4003, "Rasterized page at 200 DPI", "clean_raster"),
    ("clean_raster_300dpi", 4004, "Rasterized page at 300 DPI", "clean_raster"),
    ("mild_jpeg_q50", 4005, "JPEG compression quality Q=50", "mild_optical"),
    ("mild_paper_grain", 4006, "Additive fibrous paper texture", "mild_optical"),
    ("mild_scanner_sensor", 4007, "CCD scanner sensor shot noise & columnar streaks", "mild_optical"),
    ("phone_lighting_shadow", 4008, "Mobile phone camera illumination gradient & shadow", "mild_optical"),
    ("faded_print_low_contrast", 4009, "Low toner / faded contrast print", "mild_optical"),
    ("perspective_tilt_2p5deg", 4010, "Slight 2.5-degree perspective tilt", "mild_optical"),
    ("severe_defocus_blur_r35", 4011, "Severe defocus blur (radius=3.5)", "severe_degradation"),
    ("extreme_rotation_18deg", 4012, "Extreme 18-degree rotation", "severe_degradation"),
    ("heavy_gaussian_noise_sigma15", 4013, "Heavy sensor noise (sigma=15.0)", "heavy_noise"),
]


def render_condition_file_bytes(doc: dict[str, Any], cond_id: str, seed: int) -> bytes:
    pdf_bytes = render_pdf_bytes(doc["text"])
    if cond_id == "clean_pdf":
        return pdf_bytes

    if cond_id == "clean_raster_150dpi":
        img = render_raster_image(pdf_bytes, dpi=150)
    elif cond_id == "clean_raster_200dpi":
        img = render_raster_image(pdf_bytes, dpi=200)
    elif cond_id == "clean_raster_300dpi":
        img = render_raster_image(pdf_bytes, dpi=300)
    elif cond_id == "mild_jpeg_q50":
        base = render_raster_image(pdf_bytes, dpi=150)
        img = apply_jpeg_compression(base, quality=50)
    elif cond_id == "mild_paper_grain":
        base = render_raster_image(pdf_bytes, dpi=150)
        img = apply_paper_grain(base, sigma=3.5, seed=seed)
    elif cond_id == "mild_scanner_sensor":
        base = render_raster_image(pdf_bytes, dpi=150)
        img = apply_scanner_sensor_noise(base, seed=seed)
    elif cond_id == "phone_lighting_shadow":
        base = render_raster_image(pdf_bytes, dpi=150)
        img = apply_phone_camera_lighting_shadow(base, seed=seed)
    elif cond_id == "faded_print_low_contrast":
        base = render_raster_image(pdf_bytes, dpi=150)
        img = apply_low_contrast(base, alpha=0.55)
    elif cond_id == "perspective_tilt_2p5deg":
        base = render_raster_image(pdf_bytes, dpi=150)
        img = apply_slight_perspective(base, seed=seed)
    elif cond_id == "severe_defocus_blur_r35":
        base = render_raster_image(pdf_bytes, dpi=150)
        img = base.filter(ImageFilter.GaussianBlur(radius=3.5))
    elif cond_id == "extreme_rotation_18deg":
        base = render_raster_image(pdf_bytes, dpi=150)
        arr = np.array(base)
        h, w = arr.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 18.0, 1.0)
        arr_rot = cv2.warpAffine(arr, M, (w, h), borderValue=(255, 255, 255))
        img = Image.fromarray(arr_rot)
    elif cond_id == "heavy_gaussian_noise_sigma15":
        base = render_raster_image(pdf_bytes, dpi=150)
        arr = np.array(base, dtype=np.float32)
        noise = np.random.RandomState(seed).normal(0, 15.0, arr.shape)
        arr_n = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr_n)
    else:
        img = render_raster_image(pdf_bytes, dpi=150)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def evaluate_condition(cond_id: str, seed: int, label: str, tier: str) -> dict[str, Any]:
    logger.info("Evaluating condition '%s' (seed=%d, separate page set)...", cond_id, seed)
    dataset = build_condition_dataset(n_per_template=4, seed=seed)
    total_pages = len(dataset)
    refused_pages = 0

    total_gt = sum(len(d["entities"]) for d in dataset)
    detected = 0
    field_errors = 0
    silent_errors = 0
    flag_mismatches = 0
    not_detected = 0

    t_start = time.perf_counter()

    for doc in dataset:
        f_bytes = render_condition_file_bytes(doc, cond_id, seed)
        ext = "pdf" if cond_id == "clean_pdf" else "png"
        mime = "application/pdf" if cond_id == "clean_pdf" else "image/png"
        text, method = extract_text_from_bytes(f_bytes, f"doc_{doc['doc_id']}.{ext}", mime)

        if method == "refused_low_quality" or "image quality too low" in text:
            refused_pages += 1
            not_detected += len(doc["entities"])
            continue

        parsed_items, _ = parse_lab_text(text, sex=doc["patient_sex"])
        parsed_dict = {item.canonical_name: item for item in parsed_items}

        for gt in doc["entities"]:
            cname = gt["canonical_name"]
            if cname not in parsed_dict:
                not_detected += 1
                continue

            detected += 1
            pred = parsed_dict[cname]
            val_diff = abs(pred.value - gt["expected_value"])

            if val_diff > 0.15:
                field_errors += 1
                silent_errors += 1
            elif pred.flag != gt["expected_flag"]:
                field_errors += 1
                flag_mismatches += 1

    runtime = round(time.perf_counter() - t_start, 2)
    rejection_rate = round(refused_pages / total_pages, 4)
    detection_rate = round(detected / total_gt, 4)
    field_error_rate = round(field_errors / max(detected, 1), 4)
    silent_error_rate = round(silent_errors / total_gt, 4)

    return {
        "condition_id": cond_id,
        "description": label,
        "tier": tier,
        "seed": seed,
        "total_pages": total_pages,
        "refused_pages": refused_pages,
        "rejection_rate": rejection_rate,
        "rejection_rate_pct": round(rejection_rate * 100, 2),
        "total_ground_truth_entities": total_gt,
        "detected_entities": detected,
        "detection_rate": detection_rate,
        "detection_rate_pct": round(detection_rate * 100, 2),
        "field_errors": field_errors,
        "field_error_rate": field_error_rate,
        "field_error_rate_pct": round(field_error_rate * 100, 2),
        "silent_errors": silent_errors,
        "silent_error_rate": silent_error_rate,
        "silent_error_rate_pct": round(silent_error_rate * 100, 2),
        "not_detected": not_detected,
        "runtime_sec": runtime,
    }


def main() -> None:
    results: dict[str, Any] = {
        "metadata": {
            "title": "OCR Gate Redesign Clean Holdout Audit",
            "date": "2026-10-17",
            "page_set_policy": "Separate page sets per condition (no pages reused across conditions)",
            "total_conditions": len(CONDITIONS),
            "templates_count": len(ALL_TEMPLATES),
            "templates": [t["layout_id"] for t in ALL_TEMPLATES],
        },
        "conditions": {},
    }

    print("=== RUNNING OCR GATE REDESIGN CLEAN HOLDOUT AUDIT ===")
    for cond_id, seed, label, tier in CONDITIONS:
        cond_res = evaluate_condition(cond_id, seed, label, tier)
        results["conditions"][cond_id] = cond_res
        print(
            f"{cond_id:<30} | Rejection: {cond_res['rejection_rate_pct']:>5.1f}% | "
            f"Detect: {cond_res['detection_rate_pct']:>5.1f}% | "
            f"FieldErr: {cond_res['field_error_rate_pct']:>5.1f}% | "
            f"SilentErr: {cond_res['silent_error_rate_pct']:>5.2f}%"
        )

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Wrote clean holdout results to %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
