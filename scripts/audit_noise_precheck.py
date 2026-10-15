#!/usr/bin/env python3
"""Audit Noise Pre-Check and Modality Gate on High-Variance Inputs (Pass 5 F6).

Tests the extended pre-check and ResNet-18 Modality Gate pipeline on high-variance inputs:
1. Gaussian noise (std 0.1, 0.2, 0.3)
2. Uniform random noise
3. Checkerboards (2px, 4px, 8px, 16px)
4. Gradients (horizontal, vertical, diagonal, radial)
5. Rendered text pages (medical / lab text layouts)
6. Application screenshots & UI cards

Computes:
- Total trials per category (N=100 each)
- Rejection count (correctly rejected as 'other' or pre-check failure)
- False acceptance count (accepted into chest_xray, dermoscopy, fundus, or blood_smear)
- False Accept Rate (FAR) with Wilson 95% confidence intervals
- Rejection Rate with Wilson 95% confidence intervals
- Overall false accept rate across all non-medical high-variance inputs
- Saves results to: reports/audit_noise_precheck.json
"""

import io
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pulmoai.image_service import ImageScreeningService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_noise_precheck")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = PROJECT_ROOT / "reports" / "audit_noise_precheck.json"


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, list[float]]:
    """Compute point estimate and Wilson score 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, [0.0, 0.0]
    p_hat = k / n
    z = 1.959963984540054  # 95% normal quantile
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2) / (4 * (n**2)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(float(p_hat), 4), [round(float(lower), 4), round(float(upper), 4)]


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_gaussian_noise(std: float, count: int = 100, seed: int = 42) -> list[Image.Image]:
    rng = np.random.RandomState(seed)
    images = []
    for _ in range(count):
        arr = np.clip(0.5 + rng.normal(0, std, (64, 64, 3)), 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8))
        images.append(img)
    return images


def generate_uniform_noise(count: int = 100, seed: int = 42) -> list[Image.Image]:
    rng = np.random.RandomState(seed)
    images = []
    for _ in range(count):
        arr = rng.uniform(0.0, 1.0, (64, 64, 3))
        img = Image.fromarray((arr * 255).astype(np.uint8))
        images.append(img)
    return images


def generate_checkerboards(count: int = 100, seed: int = 42) -> list[Image.Image]:
    rng = np.random.RandomState(seed)
    images = []
    block_sizes = [2, 4, 8, 16]
    for i in range(count):
        bs = block_sizes[i % len(block_sizes)]
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        color1 = rng.randint(0, 50, size=3)
        color2 = rng.randint(200, 255, size=3)
        for y in range(64):
            for x in range(64):
                if ((y // bs) + (x // bs)) % 2 == 0:
                    arr[y, x] = color1
                else:
                    arr[y, x] = color2
        images.append(Image.fromarray(arr))
    return images


def generate_gradients(count: int = 100, seed: int = 42) -> list[Image.Image]:
    rng = np.random.RandomState(seed)
    images = []
    grad_types = ["horizontal", "vertical", "diagonal", "radial"]
    for i in range(count):
        gtype = grad_types[i % len(grad_types)]
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        start_val = rng.randint(0, 80)
        end_val = rng.randint(180, 255)
        if gtype == "horizontal":
            line = np.linspace(start_val, end_val, 64)
            arr = np.tile(line[None, :, None], (64, 1, 3)).astype(np.uint8)
        elif gtype == "vertical":
            line = np.linspace(start_val, end_val, 64)
            arr = np.tile(line[:, None, None], (1, 64, 3)).astype(np.uint8)
        elif gtype == "diagonal":
            for y in range(64):
                for x in range(64):
                    val = start_val + (end_val - start_val) * (x + y) / 126.0
                    arr[y, x] = int(val)
        else:  # radial
            cy, cx = 31.5, 31.5
            max_r = math.sqrt(32**2 + 32**2)
            for y in range(64):
                for x in range(64):
                    r = math.sqrt((y - cy)**2 + (x - cx)**2)
                    val = start_val + (end_val - start_val) * (r / max_r)
                    arr[y, x] = int(np.clip(val, 0, 255))
        images.append(Image.fromarray(arr))
    return images


def generate_rendered_text(count: int = 100, seed: int = 42) -> list[Image.Image]:
    rng = np.random.RandomState(seed)
    images = []
    snippets = [
        "Patient: John Doe\nResult: 142.5 mg/dL\nRef: 70-99",
        "CLINICAL PATHOLOGY\nGlucose: 95.0\nStatus: NORMAL",
        "HbA1c: 6.8%\nHigh Risk Flag\nVerified: ALIS",
        "Cholesterol: 240\nHDL: 42 mg/dL\nTRIG: 195",
        "Hemoglobin: 13.2\nPlatelets: 210\nCBC Final",
    ]
    for i in range(count):
        bg = rng.randint(240, 255)
        img = Image.new("RGB", (64, 64), color=(bg, bg, bg))
        draw = ImageDraw.Draw(img)
        txt = snippets[i % len(snippets)]
        draw.text((3, 3), txt, fill=(0, 0, 0))
        images.append(img)
    return images


def generate_screenshots(count: int = 100, seed: int = 42) -> list[Image.Image]:
    rng = np.random.RandomState(seed)
    images = []
    for _ in range(count):
        bg = rng.randint(230, 250)
        img = Image.new("RGB", (64, 64), color=(bg, bg, bg))
        draw = ImageDraw.Draw(img)
        # Draw mock UI window / buttons / banner
        hdr_col = tuple(rng.randint(20, 80, size=3).tolist())
        btn_col = tuple(rng.randint(100, 200, size=3).tolist())
        draw.rectangle([4, 4, 60, 16], fill=hdr_col)
        draw.rectangle([8, 22, 56, 40], fill=(255, 255, 255), outline=(180, 180, 180))
        draw.rectangle([12, 46, 52, 58], fill=btn_col)
        images.append(img)
    return images


def main() -> None:
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("PASS 5 (F6): NOISE PRE-CHECK & MODALITY GATE AUDIT")
    logger.info("=" * 80)

    service = ImageScreeningService.load()
    logger.info("ImageScreeningService initialized with device: %s", service.device)

    test_categories = {
        "gaussian_noise_std_0.1": generate_gaussian_noise(std=0.1, count=100, seed=101),
        "gaussian_noise_std_0.2": generate_gaussian_noise(std=0.2, count=100, seed=102),
        "gaussian_noise_std_0.3": generate_gaussian_noise(std=0.3, count=100, seed=103),
        "uniform_noise": generate_uniform_noise(count=100, seed=104),
        "checkerboards": generate_checkerboards(count=100, seed=105),
        "smooth_gradients": generate_gradients(count=100, seed=106),
        "rendered_text_pages": generate_rendered_text(count=100, seed=107),
        "screenshots_ui_cards": generate_screenshots(count=100, seed=108),
    }

    results = {}
    total_non_medical = 0
    total_false_accepts = 0

    for cat_name, img_list in test_categories.items():
        n = len(img_list)
        rejections = 0
        false_accepts = 0
        false_accept_details = []

        logger.info("Evaluating category: %s (N=%d)...", cat_name, n)
        for i, img in enumerate(img_list):
            raw_b = pil_to_bytes(img)
            # Request chest_xray; gate should reject
            resp = service.predict(raw_b, image_type="chest_xray", enforce_modality_gate=True)
            gate = resp.get("modality_gate", {})
            pred_mod = gate.get("predicted_modality", "unknown")
            is_rejected = not resp.get("recognized", True)

            if is_rejected or pred_mod == "other":
                rejections += 1
            else:
                false_accepts += 1
                false_accept_details.append({
                    "sample_idx": i,
                    "predicted_modality": pred_mod,
                    "confidence": gate.get("confidence"),
                })

        far_point, far_ci = wilson_score_interval(false_accepts, n)
        rej_point, rej_ci = wilson_score_interval(rejections, n)

        total_non_medical += n
        total_false_accepts += false_accepts

        results[cat_name] = {
            "total_tested": n,
            "rejections": rejections,
            "rejection_rate": rej_point,
            "rejection_rate_95ci_wilson": rej_ci,
            "false_accepts": false_accepts,
            "false_accept_rate": far_point,
            "false_accept_rate_95ci_wilson": far_ci,
            "false_accept_details": false_accept_details[:5],
        }

        logger.info(
            "%-25s: Rejected: %3d/%3d (%5.1f%%, 95%% CI %s) | FAR: %5.2f%% (95%% CI %s)",
            cat_name,
            rejections,
            n,
            rej_point * 100.0,
            rej_ci,
            far_point * 100.0,
            far_ci,
        )

    overall_far_pt, overall_far_ci = wilson_score_interval(total_false_accepts, total_non_medical)
    overall_rej_pt, overall_rej_ci = wilson_score_interval(
        total_non_medical - total_false_accepts, total_non_medical
    )

    logger.info("=" * 80)
    logger.info(
        "OVERALL HIGH-VARIANCE NON-MEDICAL INPUTS (N=%d): Rejection Rate: %.2f%% (95%% CI %s), FAR: %.2f%% (95%% CI %s)",
        total_non_medical,
        overall_rej_pt * 100.0,
        overall_rej_ci,
        overall_far_pt * 100.0,
        overall_far_ci,
    )
    logger.info("=" * 80)

    final_record = {
        "metadata": {
            "title": "Noise Pre-Check and Modality Gate Audit on High-Variance Inputs (Pass 5 F6)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_seconds": round(time.time() - t0, 2),
            "total_samples_evaluated": total_non_medical,
            "total_false_accepts": total_false_accepts,
            "overall_rejection_rate": overall_rej_pt,
            "overall_rejection_rate_95ci_wilson": overall_rej_ci,
            "overall_false_accept_rate": overall_far_pt,
            "overall_false_accept_rate_95ci_wilson": overall_far_ci,
        },
        "precheck_architecture": {
            "low_entropy_std_threshold": 0.040,
            "synthetic_gradient_laplacian_threshold": 5.0,
            "resnet18_modality_classes": ["chest_xray", "dermoscopy", "fundus", "blood_smear", "other"],
            "lofo_macro_far": 0.1903,
            "worst_left_out_family": "synthetic_lab_reports (100.0%)",
            "domain_boundaries_warning": (
                "The Modality Gate recognizes only the specific image distributions it was trained on "
                "(pediatric chest x-ray, dermoscopy, fundus, blood smear, and specific training negatives). "
                "It cannot guarantee rejection of unseen out-of-domain photographic, histological, or synthetic formats."
            ),
        },
        "categories": results,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_record, f, indent=2)
    logger.info("Saved noise pre-check audit to %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
