#!/usr/bin/env python3
"""Evaluation Harness for Clinical Laboratory Reports.

This script evaluates the PulmoAI lab parser against clinical laboratory
reports (PDFs, TIFFs, PNGs, JPEGs) given a ground-truth CSV.

Expected Ground Truth CSV Format (flexible column names):
    file,analyte,value,unit[,expected_flag]
    synth_01.png,fasting_glucose,95.0,mg/dL,NORMAL
    synth_01.png,serum_creatinine,0.9,mg/dL,NORMAL
    ...

Features:
- Automated pre-extraction quality gate evaluation (ACCEPT vs REFUSE) per file.
- Per-analyte and overall extraction accuracy with 95% Wilson score confidence intervals.
- Detailed itemization of any incorrect values (wrong values list).
- Abnormality flag concordance verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pymupdf

from pulmoai.report_reader import (
    check_image_quality,
    determine_flag,
    extract_text_from_bytes,
    parse_lab_text,
    preprocess_lab_image_adaptive,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_real_reports")

CANONICAL_ANALYTES = {
    "fasting_glucose",
    "hba1c",
    "total_cholesterol",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "triglycerides",
    "serum_creatinine",
    "hemoglobin",
}


def wilson_score_interval(successes: int, total: int, z: float = 1.95996) -> tuple[float, float]:
    """Compute two-sided Wilson score confidence interval for a binomial proportion."""
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


def load_ground_truth(gt_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load ground-truth annotations keyed by file/report_id and analyte."""
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    records: dict[str, dict[str, dict[str, Any]]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return records

        # Auto-detect column headers
        col_id = next((c for c in ["file", "report_id", "filename", "id"] if c in reader.fieldnames), reader.fieldnames[0])
        col_analyte = next((c for c in ["analyte", "test_name", "biomarker"] if c in reader.fieldnames), "analyte")
        col_val = next((c for c in ["value", "ground_truth_value", "target_value", "val"] if c in reader.fieldnames), "value")
        col_unit = next((c for c in ["unit", "units"] if c in reader.fieldnames), "unit")
        col_flag = next((c for c in ["expected_flag", "flag"] if c in reader.fieldnames), None)

        for row in reader:
            rep_id = row[col_id].strip()
            analyte = row[col_analyte].strip().lower()
            try:
                val = float(row[col_val])
            except (ValueError, KeyError):
                continue
            unit = row.get(col_unit, "").strip() if col_unit in row else ""
            flag = row.get(col_flag, "").strip().upper() if col_flag and col_flag in row else ""

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
    """Evaluate quality gate and extraction on a single report."""
    is_pdf = file_path.suffix.lower() == ".pdf"
    quality_decision = "ACCEPT"
    quality_message = "acceptable"
    quality_metrics: dict[str, float] = {}
    text_content = ""

    if is_pdf:
        # Check quality of rasterized first page
        try:
            doc = pymupdf.open(file_path)
            if len(doc) > 0:
                pix = doc[0].get_pixmap(dpi=150)
                page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                is_ok, quality_message, quality_metrics = check_image_quality(page_img)
                quality_decision = "ACCEPT" if is_ok else "REFUSE"
            doc.close()
        except Exception as e:
            logger.warning("PDF quality check error: %s", e)

        pdf_pages = read_pdf_text_and_images(file_path)
        native_text = "\n".join(p["native_text"] for p in pdf_pages if p.get("native_text"))
        if len(native_text.strip()) > 50:
            text_content = native_text
        else:
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
        # Image file
        try:
            with Image.open(file_path) as img:
                rgb_img = img.convert("RGB")
                is_ok, quality_message, quality_metrics = check_image_quality(rgb_img)
                quality_decision = "ACCEPT" if is_ok else "REFUSE"

                if is_ok:
                    import pytesseract
                    if adaptive_ocr:
                        processed_img, _ = preprocess_lab_image_adaptive(rgb_img)
                    else:
                        processed_img = rgb_img
                    text_content = pytesseract.image_to_string(processed_img)
                else:
                    text_content = ""
        except Exception as e:
            logger.warning("Image processing error for %s: %s", file_path.name, e)
            quality_decision = "REFUSE"
            quality_message = f"error: {e}"

    if quality_decision == "REFUSE":
        entity_results = {}
        for analyte, target in expected_entities.items():
            entity_results[analyte] = {
                "status": "REFUSED_BY_GATE",
                "expected": target["value"],
                "extracted": None,
                "rel_error": 1.0,
                "flag_match": False,
            }
        return {
            "file_name": file_path.name,
            "quality_gate": {
                "decision": quality_decision,
                "message": quality_message,
                "metrics": quality_metrics,
            },
            "entity_results": entity_results,
            "wrong_values": [],
            "false_positives": [],
        }

    parsed_items, _ = parse_lab_text(text_content)
    extracted = {item.canonical_name: item for item in parsed_items}

    entity_results = {}
    wrong_values = []

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
            item = extracted[analyte]
            ext_val = float(item.value)
            exp_val = float(target["value"])
            rel_err = abs(ext_val - exp_val) / max(exp_val, 1e-6)
            is_match = rel_err <= 0.01

            ext_flag = item.flag
            exp_flag = target.get("flag", "")
            flag_match = (ext_flag == exp_flag) if exp_flag else True

            status = "CORRECT" if is_match else "INCORRECT_VALUE"
            entity_results[analyte] = {
                "status": status,
                "expected": exp_val,
                "extracted": ext_val,
                "rel_error": round(rel_err, 4),
                "flag_match": flag_match,
            }
            if not is_match:
                wrong_values.append({
                    "file": file_path.name,
                    "analyte": analyte,
                    "extracted": ext_val,
                    "expected": exp_val,
                    "rel_error": round(rel_err, 4),
                })

    false_positives = [a for a in extracted if a not in expected_entities and a in CANONICAL_ANALYTES]

    return {
        "file_name": file_path.name,
        "quality_gate": {
            "decision": quality_decision,
            "message": quality_message,
            "metrics": quality_metrics,
        },
        "entity_results": entity_results,
        "wrong_values": wrong_values,
        "false_positives": false_positives,
    }


def run_evaluation(
    reports_dir: Path,
    ground_truth_path: Path,
    output_path: Path,
    adaptive_ocr: bool = True,
) -> dict[str, Any]:
    """Run full evaluation suite across clinical reports."""
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
    all_wrong_values: list[dict[str, Any]] = []

    per_analyte_stats: dict[str, dict[str, int]] = {
        a: {"expected": 0, "correct": 0, "flag_correct": 0, "missing": 0, "wrong": 0} for a in CANONICAL_ANALYTES
    }

    quality_gate_summary = {
        "total_files": len(report_files),
        "accepted": 0,
        "refused": 0,
        "decisions": {},
    }

    for fpath in report_files:
        stem = fpath.stem
        fname = fpath.name

        # Match either exact filename or stem in ground truth
        matched_id = None
        if fname in gt:
            matched_id = fname
        elif stem in gt:
            matched_id = stem

        if not matched_id:
            logger.warning("Report file %s has no matching ground-truth entry; skipping.", fpath.name)
            continue

        exp = gt[matched_id]
        res = evaluate_single_report(fpath, exp, adaptive_ocr=adaptive_ocr)
        results.append(res)

        decision = res["quality_gate"]["decision"]
        quality_gate_summary["decisions"][fname] = {
            "decision": decision,
            "message": res["quality_gate"]["message"],
        }
        if decision == "ACCEPT":
            quality_gate_summary["accepted"] += 1
        else:
            quality_gate_summary["refused"] += 1

        all_wrong_values.extend(res["wrong_values"])

        for analyte, r in res["entity_results"].items():
            if analyte not in per_analyte_stats:
                per_analyte_stats[analyte] = {"expected": 0, "correct": 0, "flag_correct": 0, "missing": 0, "wrong": 0}
            per_analyte_stats[analyte]["expected"] += 1
            total_entities += 1

            if r["status"] == "CORRECT":
                correct_entities += 1
                per_analyte_stats[analyte]["correct"] += 1
            elif r["status"] == "MISSING":
                per_analyte_stats[analyte]["missing"] += 1
            elif r["status"] == "INCORRECT_VALUE":
                per_analyte_stats[analyte]["wrong"] += 1

            if r["flag_match"]:
                flag_correct_entities += 1
                per_analyte_stats[analyte]["flag_correct"] += 1

        total_false_positives += len(res["false_positives"])

    acc = correct_entities / max(total_entities, 1)
    acc_ci = wilson_score_interval(correct_entities, total_entities)
    flag_acc = flag_correct_entities / max(total_entities, 1)
    flag_acc_ci = wilson_score_interval(flag_correct_entities, total_entities)

    per_analyte_output = {}
    for a in sorted(per_analyte_stats.keys()):
        stats = per_analyte_stats[a]
        if stats["expected"] > 0:
            a_acc = stats["correct"] / stats["expected"]
            a_ci = wilson_score_interval(stats["correct"], stats["expected"])
            per_analyte_output[a] = {
                "expected": stats["expected"],
                "correct": stats["correct"],
                "wrong": stats["wrong"],
                "missing": stats["missing"],
                "accuracy": round(a_acc, 4),
                "accuracy_95ci": list(a_ci),
                "flag_accuracy": round(stats["flag_correct"] / stats["expected"], 4),
            }

    summary = {
        "total_reports": len(results),
        "quality_gate_summary": quality_gate_summary,
        "total_expected_entities": total_entities,
        "correctly_extracted_entities": correct_entities,
        "overall_accuracy": round(acc, 4),
        "overall_accuracy_95ci": list(acc_ci),
        "flag_accuracy": round(flag_acc, 4),
        "flag_accuracy_95ci": list(flag_acc_ci),
        "total_wrong_values_count": len(all_wrong_values),
        "wrong_values": all_wrong_values,
        "total_false_positives": total_false_positives,
        "per_analyte": per_analyte_output,
        "per_report_details": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Evaluation complete. Overall accuracy: %.2f%% (95%% CI: [%.4f, %.4f]). Wrong values: %d. Saved to %s",
        acc * 100,
        acc_ci[0],
        acc_ci[1],
        len(all_wrong_values),
        output_path,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PulmoAI lab parser on clinical reports.")
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
