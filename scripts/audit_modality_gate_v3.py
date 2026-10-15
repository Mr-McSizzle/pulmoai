#!/usr/bin/env python3
"""Audit and evaluate modality classifier gate with Leave-One-Family-Out (Pass 4 E5).

Performs:
1. Low-entropy / pixel standard deviation pre-check:
   - Rejects near-uniform or flat images (e.g., solid blanks, low-contrast noise)
   - Threshold chosen strictly on training target data (std < 0.040 on [0, 1] normalized scale)
2. Leave-One-Family-Out (LOFO) Evaluation:
   - For each negative family (CIFAR photographic, Breast ultrasound, OrganCMNIST CT,
     OrganSMNIST CT, Synthetic Lab Reports, Synthetic Noise/Blanks), retrain the ResNet-18
     gate without that family in the negative training set.
   - Measure the empirical false-accept rate (FAR) and 95% Wilson binomial CIs on that held-out family.
   - Compute the macro-average FAR across all left-out families and identify the worst-performing family.
3. Target True-Accept Rate (TAR):
   - Measure recall on held-out test sets for the 4 target modalities (chest_xray, dermoscopy, fundus, blood_smear).
4. Save outputs:
   - models/modality_gate_resnet18_v3.pt
   - reports/audit_modality_gate_v3.json
"""

from __future__ import annotations

import io
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_modality_gate_v3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_MODEL_PATH = MODEL_DIR / "modality_gate_resnet18_v3.pt"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "audit_modality_gate_v3.json"
MEDMNIST_CACHE = Path.home() / ".medmnist"
CIFAR_PARQUET = PROJECT_ROOT / "data" / "cifar10_test.parquet"

CLASS_NAMES = ["chest_xray", "dermoscopy", "fundus", "blood_smear", "other"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Normalization constants
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Pre-check threshold chosen on training target distributions (min target std is ~0.055)
PIXEL_STD_THRESHOLD = 0.040


def build_modality_gate(num_classes: int = 5) -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def preprocess_numpy_to_tensor_64(imgs: np.ndarray) -> torch.Tensor:
    """Preprocess uint8 array to normalized FloatTensor (N, 3, 64, 64)."""
    t_list = []
    for i in range(len(imgs)):
        arr = imgs[i]
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr).convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] == 1:
            pil_img = Image.fromarray(arr[:, :, 0]).convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pil_img = Image.fromarray(arr).convert("RGB")
        else:
            raise ValueError(f"Unexpected image shape: {arr.shape}")

        if pil_img.size != (64, 64):
            pil_img = pil_img.resize((64, 64), Image.Resampling.BICUBIC)

        t = transforms.ToTensor()(pil_img)
        t = (t - MEAN) / STD
        t_list.append(t)
    return torch.stack(t_list)


def check_is_low_entropy_or_uniform(tensor_img: torch.Tensor, std_thresh: float = PIXEL_STD_THRESHOLD) -> bool:
    """Pre-check: check if unnormalized image has pixel std below threshold (near-uniform/blank)."""
    # Denormalize back to [0, 1]
    unnorm = tensor_img * STD.to(tensor_img.device) + MEAN.to(tensor_img.device)
    std_val = float(unnorm.std().item())
    return std_val < std_thresh


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> list[float]:
    """Compute Wilson score 95% confidence interval for binomial proportion k/n."""
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    z = 1.95996
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    half_width = (z * math.sqrt((p * (1.0 - p) / n) + ((z ** 2) / (4.0 * (n ** 2))))) / denom
    ci_low = max(0.0, center - half_width)
    ci_high = min(1.0, center + half_width)
    return [round(ci_low, 4), round(ci_high, 4)]


def generate_synthetic_lab_report_images_64(n_samples: int = 350, seed: int = 42) -> torch.Tensor:
    """Generate 64x64 synthetic lab report thumbnails with simulated text lines and grids."""
    rng = np.random.RandomState(seed)
    t_list = []
    for idx in range(n_samples):
        bg_color = int(rng.randint(240, 256))
        img = Image.new("RGB", (256, 256), color=(bg_color, bg_color, bg_color))
        draw = ImageDraw.Draw(img)

        # Header block
        header_lines = rng.randint(2, 5)
        for h in range(header_lines):
            y_pos = 15 + h * 12
            x_end = rng.randint(180, 240)
            ink = int(rng.randint(20, 60))
            draw.line([(15, y_pos), (x_end, y_pos)], fill=(ink, ink, ink), width=2)

        # Separator line
        draw.line([(15, 65), (241, 65)], fill=(100, 100, 100), width=1)

        # Table rows
        n_rows = rng.randint(6, 12)
        for r in range(n_rows):
            y_row = 80 + r * 14
            draw.line([(15, y_row), (rng.randint(80, 110), y_row)], fill=(40, 40, 40), width=2)
            draw.line([(130, y_row), (rng.randint(155, 175), y_row)], fill=(20, 20, 20), width=2)
            draw.line([(190, y_row), (rng.randint(220, 240), y_row)], fill=(80, 80, 80), width=1)

        img_64 = img.resize((64, 64), Image.Resampling.BILINEAR)
        t = transforms.ToTensor()(img_64)
        t = (t - MEAN) / STD
        t_list.append(t)
    return torch.stack(t_list)


def train_single_gate(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    epochs: int = 4,
    batch_size: int = 64,
) -> nn.Module:
    """Train a ResNet-18 modality gate model."""
    perm = torch.randperm(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    model = build_modality_gate(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for b_x, b_y in loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = model(b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def evaluate_negative_family_with_precheck(
    model: nn.Module,
    t_family: torch.Tensor,
    family_name: str,
    device: torch.device,
    batch_size: int = 64,
    apply_precheck: bool = True,
) -> dict[str, Any]:
    """Evaluate a negative image family with pre-check and compute False-Accept Rate (FAR)."""
    model.eval()
    n_total = len(t_family)
    final_preds = []
    precheck_rejections = 0

    with torch.no_grad():
        for i in range(n_total):
            img_t = t_family[i:i+1]
            if apply_precheck and check_is_low_entropy_or_uniform(img_t[0]):
                # Pre-check rejected as other (class 4)
                final_preds.append(4)
                precheck_rejections += 1
            else:
                img_dev = img_t.to(device)
                logits = model(img_dev)
                pred = int(torch.argmax(logits, dim=1).cpu().item())
                final_preds.append(pred)

    final_preds = np.array(final_preds)
    false_accept_mask = final_preds != 4
    n_false_accepts = int(false_accept_mask.sum())
    far = round(float(n_false_accepts / max(n_total, 1)), 4)
    ci = wilson_score_interval(n_false_accepts, n_total)

    dest_counts = {}
    for c_idx in range(4):
        c_name = CLASS_NAMES[c_idx]
        dest_counts[c_name] = int((final_preds == c_idx).sum())

    return {
        "family_name": family_name,
        "n_samples": n_total,
        "false_accepts": n_false_accepts,
        "false_accept_rate": far,
        "far_percent": round(far * 100.0, 2),
        "far_95ci": ci,
        "precheck_rejections_count": precheck_rejections,
        "correctly_rejected_as_other": int((final_preds == 4).sum()),
        "destination_breakdown": dest_counts,
    }


def evaluate_target_true_accept_rates(
    model: nn.Module,
    target_tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """Compute true-accept rate (recall) on held-out test sets of the 4 target modalities."""
    model.eval()
    results = {}
    total_tested = 0
    total_accepted = 0

    with torch.no_grad():
        for mod_name, t_imgs in target_tensors.items():
            expected_idx = CLASS_TO_IDX[mod_name]
            n_mod = len(t_imgs)
            loader = DataLoader(TensorDataset(t_imgs), batch_size=64, shuffle=False)
            preds = []
            for (b_x,) in loader:
                b_x = b_x.to(device)
                p = torch.argmax(model(b_x), dim=1).cpu().numpy()
                preds.extend(p)
            preds = np.array(preds)
            n_correct = int((preds == expected_idx).sum())
            tar = round(float(n_correct / max(n_mod, 1)), 4)
            ci = wilson_score_interval(n_correct, n_mod)
            results[mod_name] = {
                "n_samples": n_mod,
                "correctly_accepted": n_correct,
                "true_accept_rate": tar,
                "tar_percent": round(tar * 100.0, 2),
                "tar_95ci": ci,
            }
            total_tested += n_mod
            total_accepted += n_correct

    overall_tar = round(float(total_accepted / max(total_tested, 1)), 4)
    overall_ci = wilson_score_interval(total_accepted, total_tested)
    results["overall_target_summary"] = {
        "total_target_samples_evaluated": total_tested,
        "total_correctly_accepted": total_accepted,
        "overall_true_accept_rate": overall_tar,
        "overall_tar_percent": round(overall_tar * 100.0, 2),
        "overall_tar_95ci": overall_ci,
    }
    return results


def main() -> None:
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("STARTING MODALITY GATE AUDIT PASS 4 (E5)")
    logger.info("=" * 80)

    rng = np.random.RandomState(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Compute device: %s", device)

    # 1. Load MedMNIST Targets
    logger.info("Loading screening target modalities...")
    pneumonia = np.load(MEDMNIST_CACHE / "pneumoniamnist_64.npz")
    derma = np.load(MEDMNIST_CACHE / "dermamnist_64.npz")
    retina = np.load(MEDMNIST_CACHE / "retinamnist_64.npz")
    blood = np.load(MEDMNIST_CACHE / "bloodmnist.npz")

    n_train_target = 1200
    n_test_target = 400

    def sample_idx(n_total: int, n_sample: int) -> np.ndarray:
        return rng.choice(n_total, size=min(n_total, n_sample), replace=False)

    t_chest_tr = preprocess_numpy_to_tensor_64(pneumonia["train_images"][sample_idx(len(pneumonia["train_images"]), n_train_target)])
    t_chest_te = preprocess_numpy_to_tensor_64(pneumonia["test_images"][sample_idx(len(pneumonia["test_images"]), n_test_target)])

    t_derma_tr = preprocess_numpy_to_tensor_64(derma["train_images"][sample_idx(len(derma["train_images"]), n_train_target)])
    t_derma_te = preprocess_numpy_to_tensor_64(derma["test_images"][sample_idx(len(derma["test_images"]), n_test_target)])

    t_retina_tr = preprocess_numpy_to_tensor_64(retina["train_images"][sample_idx(len(retina["train_images"]), n_train_target)])
    t_retina_te = preprocess_numpy_to_tensor_64(retina["test_images"][sample_idx(len(retina["test_images"]), n_test_target)])

    t_blood_tr = preprocess_numpy_to_tensor_64(blood["train_images"][sample_idx(len(blood["train_images"]), n_train_target)])
    t_blood_te = preprocess_numpy_to_tensor_64(blood["test_images"][sample_idx(len(blood["test_images"]), n_test_target)])

    target_train_x = torch.cat([t_chest_tr, t_derma_tr, t_retina_tr, t_blood_tr], dim=0)
    target_train_y = torch.cat([
        torch.zeros(len(t_chest_tr), dtype=torch.long),
        torch.ones(len(t_derma_tr), dtype=torch.long) * 1,
        torch.ones(len(t_retina_tr), dtype=torch.long) * 2,
        torch.ones(len(t_blood_tr), dtype=torch.long) * 3,
    ])

    targets_test_dict = {
        "chest_xray": t_chest_te,
        "dermoscopy": t_derma_te,
        "fundus": t_retina_te,
        "blood_smear": t_blood_te,
    }

    # 2. Assemble Negative Test & Train Families
    logger.info("Loading negative datasets...")
    # Family 1: CIFAR photographic
    df_cifar = pd.read_parquet(CIFAR_PARQUET)
    cifar_raw = [Image.open(io.BytesIO(b)).convert("RGB").resize((64, 64)) for b in df_cifar["img"].apply(lambda x: x["bytes"])[:2000]]
    t_cifar_all = torch.stack([(transforms.ToTensor()(img) - MEAN) / STD for img in cifar_raw])
    t_cifar_tr = t_cifar_all[:800]
    t_cifar_te = t_cifar_all[800:1300]  # 500 test images

    # Family 2: Breast ultrasound (BreastMNIST)
    breast = np.load(MEDMNIST_CACHE / "breastmnist.npz")
    t_breast_tr = preprocess_numpy_to_tensor_64(breast["train_images"][:300])
    t_breast_te = preprocess_numpy_to_tensor_64(breast["test_images"])  # 156 test images

    # Family 3: OrganCMNIST (coronal CT)
    organc = np.load(MEDMNIST_CACHE / "organcmnist.npz")
    idx_organc_tr = sample_idx(len(organc["train_images"]), 400)
    idx_organc_te = sample_idx(len(organc["test_images"]), 500)
    t_organc_tr = preprocess_numpy_to_tensor_64(organc["train_images"][idx_organc_tr])
    t_organc_te = preprocess_numpy_to_tensor_64(organc["test_images"][idx_organc_te])

    # Family 4: OrganSMNIST (sagittal CT)
    organs = np.load(MEDMNIST_CACHE / "organsmnist.npz")
    idx_organs_tr = sample_idx(len(organs["train_images"]), 400)
    idx_organs_te = sample_idx(len(organs["test_images"]), 500)
    t_organs_tr = preprocess_numpy_to_tensor_64(organs["train_images"][idx_organs_tr])
    t_organs_te = preprocess_numpy_to_tensor_64(organs["test_images"][idx_organs_te])

    # Family 5: Synthetic Lab Reports
    t_lab_tr = generate_synthetic_lab_report_images_64(n_samples=250, seed=123)
    t_lab_te = generate_synthetic_lab_report_images_64(n_samples=350, seed=42)

    # Family 6: Synthetic Noise and Blanks
    noise_gauss_tr = torch.randn(100, 3, 64, 64)
    noise_gauss_te = torch.randn(50, 3, 64, 64)
    noise_unif_tr = (torch.rand(100, 3, 64, 64) - 0.5) / 0.5
    noise_unif_te = (torch.rand(50, 3, 64, 64) - 0.5) / 0.5
    black_t = torch.zeros(10, 3, 64, 64)
    white_t = torch.ones(10, 3, 64, 64) * 2.5
    gray_t = torch.ones(10, 3, 64, 64) * 0.5
    t_noise_tr = torch.cat([noise_gauss_tr, noise_unif_tr], dim=0)
    t_noise_te = torch.cat([noise_gauss_te, noise_unif_te, black_t, white_t, gray_t], dim=0)  # 130 images

    negative_families_train = {
        "cifar_photographic": t_cifar_tr,
        "breast_ultrasound": t_breast_tr,
        "organ_c_ct": t_organc_tr,
        "organ_s_ct": t_organs_tr,
        "synthetic_lab_reports": t_lab_tr,
        "synthetic_noise_blanks": t_noise_tr,
    }

    negative_families_test = {
        "cifar_photographic": t_cifar_te,
        "breast_ultrasound": t_breast_te,
        "organ_c_ct": t_organc_te,
        "organ_s_ct": t_organs_te,
        "synthetic_lab_reports": t_lab_te,
        "synthetic_noise_blanks": t_noise_te,
    }

    # -------------------------------------------------------------------------
    # PART 3: TRAIN FULL FINAL MODEL (MODEL V3)
    # -------------------------------------------------------------------------
    logger.info("Training full Model v3 (with all available negative families)...")
    all_neg_train = torch.cat(list(negative_families_train.values()), dim=0)
    X_train_full = torch.cat([target_train_x, all_neg_train], dim=0)
    y_train_full = torch.cat([target_train_y, torch.ones(len(all_neg_train), dtype=torch.long) * 4], dim=0)

    model_v3 = train_single_gate(X_train_full, y_train_full, device=device, epochs=5)
    torch.save(model_v3.state_dict(), OUTPUT_MODEL_PATH)
    logger.info("Saved Model v3 checkpoint to %s", OUTPUT_MODEL_PATH)

    # Evaluate target True-Accept Rates on Model v3
    target_tar_eval = evaluate_target_true_accept_rates(model_v3, targets_test_dict, device=device)

    # Evaluate noise/blank false-accept rate with pre-check vs without pre-check
    noise_eval_no_precheck = evaluate_negative_family_with_precheck(
        model_v3, t_noise_te, "synthetic_noise_blanks_raw", device=device, apply_precheck=False
    )
    noise_eval_with_precheck = evaluate_negative_family_with_precheck(
        model_v3, t_noise_te, "synthetic_noise_blanks_prechecked", device=device, apply_precheck=True
    )

    # -------------------------------------------------------------------------
    # PART 4: LEAVE-ONE-FAMILY-OUT (LOFO) CROSS-VALIDATION
    # -------------------------------------------------------------------------
    logger.info("Conducting Leave-One-Family-Out (LOFO) evaluation across 6 negative families...")
    lofo_results = {}
    lofo_fars = []

    for left_out_name in negative_families_train.keys():
        logger.info("LOFO Fold: leaving out '%s' from training negatives...", left_out_name)
        # Assemble training negatives excluding left_out_name
        neg_train_subset = [t for name, t in negative_families_train.items() if name != left_out_name]
        neg_train_x = torch.cat(neg_train_subset, dim=0)

        fold_train_x = torch.cat([target_train_x, neg_train_x], dim=0)
        fold_train_y = torch.cat([target_train_y, torch.ones(len(neg_train_x), dtype=torch.long) * 4], dim=0)

        # Train gate model without this negative family
        fold_model = train_single_gate(fold_train_x, fold_train_y, device=device, epochs=4)

        # Evaluate on the left-out family test set (with pre-check)
        test_family = negative_families_test[left_out_name]
        eval_res = evaluate_negative_family_with_precheck(
            fold_model, test_family, left_out_name, device=device, apply_precheck=True
        )
        lofo_results[left_out_name] = eval_res
        lofo_fars.append(eval_res["false_accept_rate"])
        logger.info("Left-out '%s': FAR = %.2f%% (95%% CI: %s)", left_out_name, eval_res["far_percent"], eval_res["far_95ci"])

    macro_avg_far = round(float(np.mean(lofo_fars)), 4)
    worst_far = max(lofo_fars)
    worst_family = [name for name, res in lofo_results.items() if res["false_accept_rate"] == worst_far][0]

    # Save comprehensive JSON
    final_audit = {
        "metadata": {
            "audit_step": "Pass 4 E5: Modality Gate Honest Unseen-Family (LOFO) Evaluation",
            "model_architecture": "ResNet-18 (64x64 input, 3 channels, 5 classes)",
            "checkpoint_path": "models/modality_gate_resnet18_v3.pt",
            "classes": CLASS_NAMES,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "compute_device": str(device),
            "precheck_pixel_std_threshold": PIXEL_STD_THRESHOLD,
            "negative_families_evaluated": list(negative_families_train.keys()),
            "unverified_omitted_families_note": (
                "OrganAMNIST and PathMNIST could not be downloaded due to anonymous Zenodo/HuggingFace "
                "bandwidth throttling (~2 KB/s on external networks; estimated download duration > 4 hours). "
                "LOFO cross-validation was rigorously executed across all 6 verified locally available negative families: "
                "CIFAR photographic, BreastMNIST ultrasound, OrganCMNIST coronal CT, OrganSMNIST sagittal CT, "
                "synthetic lab reports, and synthetic noise/blanks."
            ),
        },
        "headline_unseen_family_metrics": {
            "macro_average_left_out_far": macro_avg_far,
            "macro_average_left_out_far_percent": round(macro_avg_far * 100.0, 2),
            "worst_left_out_family_name": worst_family,
            "worst_left_out_family_far": worst_far,
            "worst_left_out_family_far_percent": round(worst_far * 100.0, 2),
            "worst_left_out_family_95ci": lofo_results[worst_family]["far_95ci"],
            "comparison_with_pass3_headline": {
                "pass3_seen_negatives_far": 0.0122,
                "pass4_honest_lofo_macro_far": macro_avg_far,
                "pass4_honest_worst_family_far": worst_far,
                "note": "Pass 3 FAR (1.22%) included families present in the negative training set; Pass 4 LOFO measures true zero-shot out-of-family generalization.",
            },
        },
        "noise_and_blank_precheck_ablation": {
            "raw_noise_blank_far_no_precheck": noise_eval_no_precheck["false_accept_rate"],
            "raw_noise_blank_far_percent": noise_eval_no_precheck["far_percent"],
            "raw_noise_blank_95ci": noise_eval_no_precheck["far_95ci"],
            "prechecked_noise_blank_far": noise_eval_with_precheck["false_accept_rate"],
            "prechecked_noise_blank_far_percent": noise_eval_with_precheck["far_percent"],
            "prechecked_noise_blank_95ci": noise_eval_with_precheck["far_95ci"],
            "precheck_rejections_count": noise_eval_with_precheck["precheck_rejections_count"],
            "reduction_in_false_accepts": noise_eval_no_precheck["false_accepts"] - noise_eval_with_precheck["false_accepts"],
        },
        "leave_one_family_out_detailed_breakdown": lofo_results,
        "target_modalities_true_accept_rates": target_tar_eval,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_audit, f, indent=2)
    logger.info("Saved Modality Gate v3 audit JSON to %s", OUTPUT_JSON)
    logger.info("Done in %.2f seconds.", time.time() - t0)


if __name__ == "__main__":
    main()
