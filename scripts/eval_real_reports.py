#!/usr/bin/env python3
"""Evaluation Harness for Real Clinical Laboratory Reports.

This script evaluates the PulmoAI lab parser against real-world de-identified
clinical laboratory reports (PDFs, TIFFs, or scanned images).

Usage:
    python scripts/eval_real_reports.py \\
        --reports-dir data/real_reports/ \\
        --ground-truth data/real_reports_ground_truth.csv \\
        --output reports/real_reports_eval.json

Expected Ground Truth CSV Format:
    report_id,analyte,ground_truth_value,unit,expected_flag
    PAT001,fasting_glucose,105.0,mg/dL,HIGH
    PAT001,serum_creatinine,0.9,mg/dL,NORMAL
    ...

Boundary Semantics:
    Reference intervals [ref_min, ref_max] are strictly inclusive:
    val < ref_min  => LOW
    ref_min <= val <= ref_max => NORMAL
    val > ref_max  => HIGH
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from pulmoai.report_reader import (
    determine_flag,
    parse_lab_text,
    preprocess_lab_image_adaptive,
    read_pdf_text_and_images,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_real_reports")

CANONICAL_ANALYTES = {
    "fasting_glucose",
    "hba1c",
    "total_cholesterol",
    "hdl_cholesterol",
    "triglycerides",
    "serum_creatinine",
    "hemoglobin",
}


def load_ground_truth(gt_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load ground-truth annotations keyed by report_id and analyte."""
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    records: dict[str, dict[str, dict[str, Any]]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rep_id = row["report_id"].strip()
            analyte = row["analyte"].strip().lower()
            val = float(row["ground_truth_value"])
            unit = row.get("unit", "").strip()
            flag = row.get("expected_flag", "").strip().upper()

            if rep_id not in records:
                records[rep_id] = {}
            records[rep_id][analyte] = {
                "value": val,
                "unit": unit,
                "flag": flag,
            }
    return records


def evaluate_single_report(
    file_path: Path,
    expected_entities: dict[str, dict[str, Any]],
    adaptive_ocr: bool = True,
) -> dict[str, Any]:
    """Evaluate extraction on a single real report."""
    is_pdf = file_path.suffix.lower() == ".pdf"
    text_content = ""

    if is_pdf:
        pdf_pages = read_pdf_text_and_images(file_path)
        # Check if native text is available
        native_text = "\n".join(p["native_text"] for p in pdf_pages if p.get("native_text"))
        if len(native_text.strip()) > 50:
            text_content = native_text
        else:
            # Fall back to OCR on rendered pages
            import pytesseract
            ocr_pages = []
            for p in pdf_pages:
                img = p.get("image")
                if img is not None:
                    if adaptive_ocr:
                        processed_img, _ = preprocess_lab_image_adaptive(img)
                    else:
                        processed_img = img
                    ocr_pages.append(pytesseract.image_to_string(processed_img))
            text_content = "\n".join(ocr_pages)
    else:
        # Image file (PNG, JPG, TIFF)
        from PIL import Image
        import pytesseract

        with Image.open(file_path) as img:
            rgb_img = img.convert("RGB")
            if adaptive_ocr:
                processed_img, _ = preprocess_lab_image_adaptive(rgb_img)
            else:
                processed_img = rgb_img
            text_content = pytesseract.image_to_string(processed_img)

    extracted = parse_lab_text(text_content)

    entity_results = {}
    for analyte, target in expected_entities.items():
        if analyte not in extracted:
            entity_results[analyte] = {
                "status": "MISSING",
                "expected": target["value"],
                "extracted": None,
                "rel_error": 1.0,
                "flag_match": False,
            }
        else:
            ext = extracted[analyte]
            ext_val = ext["value"]
            exp_val = target["value"]
            rel_err = abs(ext_val - exp_val) / max(exp_val, 1e-6)
            is_match = rel_err <= 0.01  # within 1%

            ext_flag = ext.get("flag", "")
            exp_flag = target.get("flag", "")
            flag_match = (ext_flag == exp_flag) if exp_flag else True

            entity_results[analyte] = {
                "status": "CORRECT" if is_match else "INCORRECT_VALUE",
                "expected": exp_val,
                "extracted": ext_val,
                "rel_error": rel_err,
                "flag_match": flag_match,
            }

    # Also detect false positive detections (analytes reported but not present in ground truth)
    false_positives = [a for a in extracted if a not in expected_entities and a in CANONICAL_ANALYTES]

    return {
        "file_name": file_path.name,
        "entity_results": entity_results,
        "false_positives": false_positives,
    }


def run_evaluation(
    reports_dir: Path,
    ground_truth_path: Path,
    output_path: Path,
    adaptive_ocr: bool = True,
) -> dict[str, Any]:
    """Run full evaluation suite across all real reports."""
    gt = load_ground_truth(ground_truth_path)
    logger.info("Loaded ground truth for %d reports.", len(gt))

    report_files = sorted(
        [p for p in reports_dir.iterdir() if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}]
    )

    if not report_files:
        logger.warning("No supported report files found in %s", reports_dir)
        return {"error": "No report files found", "reports_dir": str(reports_dir)}

    results = []
    total_entities = 0
    correct_entities = 0
    flag_correct_entities = 0
    total_false_positives = 0

    per_analyte_stats: dict[str, dict[str, int]] = {
        a: {"expected": 0, "correct": 0, "flag_correct": 0, "missing": 0} for a in CANONICAL_ANALYTES
    }

    for fpath in report_files:
        stem = fpath.stem
        if stem not in gt:
            logger.warning("Report file %s has no matching ground-truth ID; skipping.", fpath.name)
            continue

        exp = gt[stem]
        res = evaluate_single_report(fpath, exp, adaptive_ocr=adaptive_ocr)
        results.append(res)

        for analyte, r in res["entity_results"].items():
            if analyte not in per_analyte_stats:
                per_analyte_stats[analyte] = {"expected": 0, "correct": 0, "flag_correct": 0, "missing": 0}
            per_analyte_stats[analyte]["expected"] += 1
            total_entities += 1

            if r["status"] == "CORRECT":
                correct_entities += 1
                per_analyte_stats[analyte]["correct"] += 1
            elif r["status"] == "MISSING":
                per_analyte_stats[analyte]["missing"] += 1

            if r["flag_match"]:
                flag_correct_entities += 1
                per_analyte_stats[analyte]["flag_correct"] += 1

        total_false_positives += len(res["false_positives"])

    acc = correct_entities / max(total_entities, 1)
    flag_acc = flag_correct_entities / max(total_entities, 1)

    summary = {
        "total_reports": len(results),
        "total_expected_entities": total_entities,
        "correctly_extracted_entities": correct_entities,
        "extraction_accuracy": round(acc, 4),
        "flag_accuracy": round(flag_acc, 4),
        "total_false_positives": total_false_positives,
        "per_analyte": {
            a: {
                "expected": per_analyte_stats[a]["expected"],
                "accuracy": round(
                    per_analyte_stats[a]["correct"] / max(per_analyte_stats[a]["expected"], 1), 4
                ),
                "flag_accuracy": round(
                    per_analyte_stats[a]["flag_correct"] / max(per_analyte_stats[a]["expected"], 1), 4
                ),
            }
            for a in per_analyte_stats
            if per_analyte_stats[a]["expected"] > 0
        },
        "per_report_details": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Evaluation complete. Accuracy: %.2f%% (%d/%d). Saved to %s", acc * 100, correct_entities, total_entities, output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PulmoAI lab parser on real clinical reports.")
    parser.add_argument("--reports-dir", type=Path, default=Path("data/real_reports"), help="Directory containing report files.")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/real_reports_ground_truth.csv"), help="CSV ground truth file.")
    parser.add_argument("--output", type=Path, default=Path("reports/real_reports_eval.json"), help="Output JSON path.")
    parser.add_argument("--no-adaptive-ocr", action="store_true", help="Disable adaptive image preprocessing.")
    args = parser.parse_args()

    if not args.reports_dir.exists() or not args.ground_truth.exists():
        print("Real reports evaluation script ready.")
        print(f"Reports directory exists: {args.reports_dir.exists()} ({args.reports_dir})")
        print(f"Ground truth exists: {args.ground_truth.exists()} ({args.ground_truth})")
        print("To run when real clinical reports are acquired under IRB approval:")
        print("  python scripts/eval_real_reports.py --reports-dir <dir> --ground-truth <gt.csv> --output <out.json>")
        return

    run_evaluation(
        reports_dir=args.reports_dir,
        ground_truth_path=args.ground_truth,
        output_path=args.output,
        adaptive_ocr=not args.no_adaptive_ocr,
    )


if __name__ == "__main__":
    main()
