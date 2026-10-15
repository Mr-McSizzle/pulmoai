#!/usr/bin/env python3
"""Audit Part C: Lab OCR Robustness Evaluation with Adaptive Per-Image Preprocessing.

Evaluates entity-level exact match across 5 report layouts and 6 degradation tiers.
Evaluates n = 350 entities per condition (50 documents x 7 tests).
Compares:
  1. Baseline Tesseract (raw image, no preprocessing)
  2. Fixed Enhanced Preprocessed Tesseract (deskew, rescale >=300 DPI, median denoising, Otsu binarization)
  3. Adaptive Preprocessed Tesseract (per-image selection based on blur and skew estimates)

Reports per layout and per degradation condition.
Labels strictly: "synthetic set; real-world accuracy unknown".
Outputs to reports/audit_ocr_part_c.json.
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

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageFilter
import pytesseract

from pulmoai.report_reader import (
    CANONICAL_TESTS,
    NAME_TO_DEF,
    extract_text_from_bytes,
    parse_lab_text,
    preprocess_lab_image,
    preprocess_lab_image_adaptive,
)
from scripts.audit_ocr_part_a import (
    LAYOUT_TEMPLATES,
    TEST_NAME_VARIANTS,
    TEST_SPECS,
    apply_degradation,
    build_synthetic_dataset,
    evaluate_extracted_entities,
    render_base_image,
    render_pdf_bytes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_ocr_part_c")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = PROJECT_ROOT / "reports" / "audit_ocr_part_c.json"

DEGRADATION_TIERS = [
    "clean_pdf",
    "clean_ocr",
    "rotation",
    "blur",
    "jpeg_compression",
    "perspective",
]


def main() -> None:
    t0 = time.time()
    logger.info("Starting Lab OCR Robustness Audit Part C (Adaptive Preprocessing)...")

    dataset = build_synthetic_dataset(n_variations_per_layout=10)
    total_docs = len(dataset)
    entities_per_doc = len(dataset[0]["entities"])
    total_entities_per_condition = total_docs * entities_per_doc

    logger.info(
        "Built synthetic dataset: %d reports across 5 layouts = %d entities per degradation condition",
        total_docs,
        total_entities_per_condition,
    )

    results_by_condition = {}
    results_by_layout = {l["layout_id"]: {} for l in LAYOUT_TEMPLATES}

    # Pre-render base PDFs and base images
    doc_renderings = []
    for d in dataset:
        pdf_b = render_pdf_bytes(d["text"])
        base_img = render_base_image(pdf_b)
        doc_renderings.append({"pdf_bytes": pdf_b, "base_img": base_img, "doc": d})

    # Diagnostic check on clean_pdf failure categorization
    clean_pdf_failures = []
    clean_pdf_by_layout = {l["layout_id"]: {"matched": 0, "total": 0} for l in LAYOUT_TEMPLATES}

    for item in doc_renderings:
        d = item["doc"]
        lid = d["layout_id"]
        pdf_b = item["pdf_bytes"]
        sex = d["patient_sex"]

        doc_pdf = pymupdf.open(stream=pdf_b, filetype="pdf")
        txt = doc_pdf[0].get_text()
        doc_pdf.close()
        extracted, _ = parse_lab_text(txt, sex=sex)
        m, tot, details = evaluate_extracted_entities(extracted, d["entities"])
        clean_pdf_by_layout[lid]["matched"] += m
        clean_pdf_by_layout[lid]["total"] += tot
        for det in details:
            if not det["is_match"]:
                clean_pdf_failures.append({
                    "doc_id": d["doc_id"],
                    "layout_id": lid,
                    "test": det["test"],
                    "expected": det["expected"],
                    "extracted": det["extracted"],
                    "category": "rounding_boundary" if det["extracted"] and abs(det["expected"]["val"] - det["extracted"]["val"]) < 0.05 else "parsing_error",
                })

    logger.info(
        "Clean PDF diagnostic: %d failures out of %d entities",
        len(clean_pdf_failures),
        total_entities_per_condition,
    )

    for deg in DEGRADATION_TIERS:
        logger.info("Evaluating degradation tier: %s...", deg)
        deg_start = time.time()

        base_matched = 0
        fixed_prep_matched = 0
        adaptive_prep_matched = 0
        total_deg = 0

        layout_stats = {
            l["layout_id"]: {
                "base_matched": 0,
                "fixed_prep_matched": 0,
                "adaptive_prep_matched": 0,
                "total": 0,
            }
            for l in LAYOUT_TEMPLATES
        }
        adaptive_decisions = {}

        for item in doc_renderings:
            d = item["doc"]
            lid = d["layout_id"]
            pdf_b = item["pdf_bytes"]
            base_img = item["base_img"]
            sex = d["patient_sex"]

            # Degrade
            deg_img = apply_degradation(base_img, deg)

            if deg == "clean_pdf":
                doc_pdf = pymupdf.open(stream=pdf_b, filetype="pdf")
                txt = doc_pdf[0].get_text()
                doc_pdf.close()
                extracted_base, _ = parse_lab_text(txt, sex=sex)
                extracted_fixed = extracted_base
                extracted_adaptive = extracted_base
                decision = "digital_pdf_text_layer"
            else:
                # 1. Baseline: Raw image into Tesseract
                raw_ocr_text = pytesseract.image_to_string(deg_img)
                extracted_base, _ = parse_lab_text(raw_ocr_text, sex=sex)

                # 2. Fixed Enhanced Preprocessing: Deskew + Otsu always
                fixed_img = preprocess_lab_image(deg_img)
                fixed_ocr_text = pytesseract.image_to_string(fixed_img)
                extracted_fixed, _ = parse_lab_text(fixed_ocr_text, sex=sex)

                # 3. Adaptive Preprocessing: Per-image blur/skew gating
                adaptive_img, decision = preprocess_lab_image_adaptive(deg_img)
                adaptive_ocr_text = pytesseract.image_to_string(adaptive_img)
                extracted_adaptive, _ = parse_lab_text(adaptive_ocr_text, sex=sex)

            adaptive_decisions[decision] = adaptive_decisions.get(decision, 0) + 1

            m_base, tot, _ = evaluate_extracted_entities(extracted_base, d["entities"])
            m_fixed, _, _ = evaluate_extracted_entities(extracted_fixed, d["entities"])
            m_adaptive, _, _ = evaluate_extracted_entities(extracted_adaptive, d["entities"])

            base_matched += m_base
            fixed_prep_matched += m_fixed
            adaptive_prep_matched += m_adaptive
            total_deg += tot

            layout_stats[lid]["base_matched"] += m_base
            layout_stats[lid]["fixed_prep_matched"] += m_fixed
            layout_stats[lid]["adaptive_prep_matched"] += m_adaptive
            layout_stats[lid]["total"] += tot

        acc_base = round(base_matched / total_deg, 4)
        acc_fixed = round(fixed_prep_matched / total_deg, 4)
        acc_adaptive = round(adaptive_prep_matched / total_deg, 4)

        results_by_condition[deg] = {
            "degradation_tier": deg,
            "total_entities_evaluated": total_deg,
            "baseline_raw_accuracy": acc_base,
            "baseline_raw_matched": base_matched,
            "fixed_enhanced_accuracy": acc_fixed,
            "fixed_enhanced_matched": fixed_prep_matched,
            "adaptive_preprocessed_accuracy": acc_adaptive,
            "adaptive_preprocessed_matched": adaptive_prep_matched,
            "adaptive_vs_raw_delta": round(acc_adaptive - acc_base, 4),
            "adaptive_vs_fixed_delta": round(acc_adaptive - acc_fixed, 4),
            "adaptive_decisions_distribution": adaptive_decisions,
            "per_layout": {
                lid: {
                    "baseline_raw_accuracy": round(stats["base_matched"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
                    "fixed_enhanced_accuracy": round(stats["fixed_prep_matched"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
                    "adaptive_preprocessed_accuracy": round(stats["adaptive_prep_matched"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
                    "entities_evaluated": stats["total"],
                }
                for lid, stats in layout_stats.items()
            },
            "runtime_seconds": round(time.time() - deg_start, 2),
        }

        for lid, stats in layout_stats.items():
            results_by_layout[lid][deg] = {
                "baseline_accuracy": round(stats["base_matched"] / stats["total"], 4),
                "fixed_enhanced_accuracy": round(stats["fixed_prep_matched"] / stats["total"], 4),
                "adaptive_accuracy": round(stats["adaptive_prep_matched"] / stats["total"], 4),
                "n_entities": stats["total"],
            }

        logger.info(
            "Degradation %s complete: Baseline=%.2f%% -> Fixed=%.2f%% -> Adaptive=%.2f%% (n=%d)",
            deg,
            acc_base * 100,
            acc_fixed * 100,
            acc_adaptive * 100,
            total_deg,
        )

    output_payload = {
        "metadata": {
            "audit_step": "Section C1 Fix: Lab Parser on Native PDFs and Adaptive Per-Image Preprocessing",
            "benchmark_dataset_label": "synthetic set; real-world accuracy unknown",
            "total_reports_evaluated": total_docs,
            "entities_per_condition": total_entities_per_condition,
            "total_evaluations": total_entities_per_condition * len(DEGRADATION_TIERS),
            "layouts_evaluated": [l["layout_id"] for l in LAYOUT_TEMPLATES],
            "degradations_evaluated": DEGRADATION_TIERS,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_runtime_seconds": round(time.time() - t0, 2),
        },
        "clean_pdf_audit": {
            "overall_accuracy": round(sum(s["matched"] for s in clean_pdf_by_layout.values()) / total_entities_per_condition, 4),
            "matched_entities": sum(s["matched"] for s in clean_pdf_by_layout.values()),
            "total_entities": total_entities_per_condition,
            "meets_target_98pct": all(s["matched"] / s["total"] >= 0.98 for s in clean_pdf_by_layout.values()),
            "per_layout_accuracy": {
                lid: {
                    "accuracy": round(s["matched"] / s["total"], 4),
                    "matched": s["matched"],
                    "total": s["total"],
                }
                for lid, s in clean_pdf_by_layout.items()
            },
            "failures": clean_pdf_failures,
        },
        "results_by_condition": results_by_condition,
        "results_by_layout": results_by_layout,
        "summary": {
            "clean_pdf_accuracy": results_by_condition["clean_pdf"]["adaptive_preprocessed_accuracy"],
            "clean_ocr_accuracy": results_by_condition["clean_ocr"]["adaptive_preprocessed_accuracy"],
            "rotation_accuracy_adaptive": results_by_condition["rotation"]["adaptive_preprocessed_accuracy"],
            "rotation_accuracy_raw": results_by_condition["rotation"]["baseline_raw_accuracy"],
            "blur_accuracy_adaptive": results_by_condition["blur"]["adaptive_preprocessed_accuracy"],
            "blur_accuracy_fixed_otsu": results_by_condition["blur"]["fixed_enhanced_accuracy"],
            "jpeg_accuracy_adaptive": results_by_condition["jpeg_compression"]["adaptive_preprocessed_accuracy"],
            "perspective_accuracy_adaptive": results_by_condition["perspective"]["adaptive_preprocessed_accuracy"],
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_payload, f, indent=2)

    logger.info("Saved Lab OCR Part C audit results to %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
