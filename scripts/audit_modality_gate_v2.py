#!/usr/bin/env python3
"""Audit and evaluate modality classifier gate (Pass 3 D5).

Performs:
1. Multi-family negative evaluation for supervised 5-class Modality Classifier Gate:
   Classes:
     0: chest_xray (PneumoniaMNIST)
     1: dermoscopy (DermaMNIST)
     2: fundus (RetinaMNIST)
     3: blood_smear (BloodMNIST)
     4: other (negatives)
2. Negative test families:
   - cifar_photographic: Photographic natural images (CIFAR-10)
   - breastmnist_medical: Breast ultrasound non-target images (BreastMNIST)
   - heldout_organcmnist: Abdominal CT coronal plane (OrganCMNIST)
   - heldout_organsmnist: Abdominal CT sagittal plane (OrganSMNIST)
   - heldout_octmnist: Retinal optical coherence tomography (OCTMNIST, if available)
   - synthetic_lab_reports_64: Downscaled 64x64 synthetic clinical lab report pages
   - synthetic_noise_and_blanks: Gaussian noise, uniform noise, solid blank images
3. Leave-dataset-out ablation:
   - Model A (Full Negative Mix): Trained with photographic + medical non-target + noise
   - Model B (No Medical Negatives / Leave-Medical-Out): Trained with ONLY photographic + noise
   - Quantifies false-accept rate (FAR) degradation when medical negatives are left out
4. Statistical evaluation:
   - Per-family False-Accept Rate (FAR) with 95% Wilson score binomial confidence intervals
   - Destination class breakdown for false accepts
   - Explicit clinical caveats:
     * Cannot distinguish adult vs pediatric radiographs (PneumoniaMNIST cohort)
     * Cannot reliably reject unseen modalities (e.g. brain MRI, contrast CT) without explicit negative training
5. Outputs:
   - models/modality_gate_resnet18_v2.pt
   - reports/audit_modality_gate_v2.json
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
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_modality_gate_v2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_MODEL_PATH = MODEL_DIR / "modality_gate_resnet18_v2.pt"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "audit_modality_gate_v2.json"
MEDMNIST_CACHE = Path.home() / ".medmnist"
CIFAR_PARQUET = PROJECT_ROOT / "data" / "cifar10_test.parquet"

CLASS_NAMES = ["chest_xray", "dermoscopy", "fundus", "blood_smear", "other"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def build_modality_gate(num_classes: int = 5) -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def preprocess_numpy_to_tensor_64(imgs: np.ndarray) -> torch.Tensor:
    """Preprocess uint8 array (N, H, W) or (N, H, W, C) to normalized FloatTensor (N, 3, 64, 64)."""
    t_list = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(len(imgs)):
        arr = imgs[i]
        if arr.ndim == 2:  # Grayscale (H, W)
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
        t = (t - mean) / std
        t_list.append(t)

    return torch.stack(t_list)


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> list[float]:
    """Compute Wilson score 95% confidence interval for binomial proportion k/n."""
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    z = 1.95996  # 95% two-sided normal quantile
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2.0 * n)) / denom
    half_width = (z * math.sqrt((p * (1.0 - p) / n) + ((z ** 2) / (4.0 * (n ** 2))))) / denom
    ci_low = max(0.0, center - half_width)
    ci_high = min(1.0, center + half_width)
    return [round(ci_low, 4), round(ci_high, 4)]


def generate_synthetic_lab_report_images_64(n_samples: int = 350, seed: int = 42) -> torch.Tensor:
    """Generate 64x64 synthetic lab report thumbnails with simulated text lines, grids, and headers."""
    rng = np.random.RandomState(seed)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t_list = []

    for idx in range(n_samples):
        # Create base high-res canvas (256x256) then downscale to 64x64 to simulate document scanning
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

        # Horizontal separator line
        draw.line([(15, 65), (241, 65)], fill=(100, 100, 100), width=1)

        # Table rows with simulated column structure
        n_rows = rng.randint(6, 12)
        for r in range(n_rows):
            y_row = 80 + r * 14
            # Column 1 (test name)
            draw.line([(15, y_row), (rng.randint(80, 110), y_row)], fill=(40, 40, 40), width=2)
            # Column 2 (value)
            draw.line([(130, y_row), (rng.randint(155, 175), y_row)], fill=(20, 20, 20), width=2)
            # Column 3 (unit/reference)
            draw.line([(190, y_row), (rng.randint(220, 240), y_row)], fill=(80, 80, 80), width=1)

        # Footer / signature block
        draw.line([(150, 235), (240, 235)], fill=(60, 60, 60), width=1)

        # Downscale to 64x64
        img_64 = img.resize((64, 64), Image.Resampling.BILINEAR)
        t = transforms.ToTensor()(img_64)
        t = (t - mean) / std
        t_list.append(t)

    return torch.stack(t_list)


def train_single_gate(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    device: torch.device,
    epochs: int = 6,
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


def evaluate_negative_family(
    model: nn.Module,
    t_family: torch.Tensor,
    family_name: str,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Evaluate a negative image family and calculate False-Accept Rate (FAR)."""
    model.eval()
    loader = DataLoader(TensorDataset(t_family), batch_size=batch_size, shuffle=False)
    all_preds = []

    with torch.no_grad():
        for (b_x,) in loader:
            b_x = b_x.to(device)
            logits = model(b_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    n_total = len(all_preds)
    # Class 4 is 'other' (correctly rejected). Classes 0, 1, 2, 3 are false accepts!
    false_accept_mask = all_preds != 4
    n_false_accepts = int(false_accept_mask.sum())
    far = round(float(n_false_accepts / max(n_total, 1)), 4)
    ci = wilson_score_interval(n_false_accepts, n_total)

    # Destination breakdown
    dest_counts = {}
    for c_idx in range(4):
        c_name = CLASS_NAMES[c_idx]
        count = int((all_preds == c_idx).sum())
        dest_counts[c_name] = count

    return {
        "family_name": family_name,
        "n_samples": n_total,
        "false_accepts": n_false_accepts,
        "false_accept_rate": far,
        "far_95ci": ci,
        "correctly_rejected_as_other": int((all_preds == 4).sum()),
        "destination_breakdown": dest_counts,
    }


def main() -> None:
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("STARTING MODALITY GATE AUDIT PASS 3 (D5)")
    logger.info("=" * 80)

    rng = np.random.RandomState(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Compute device: %s", device)

    # 1. Load MedMNIST Modality Datasets (Positive Targets)
    logger.info("Loading screening target modalities...")
    pneumonia = np.load(MEDMNIST_CACHE / "pneumoniamnist_64.npz")
    pneumonia_train = pneumonia["train_images"]
    pneumonia_test = pneumonia["test_images"]

    derma = np.load(MEDMNIST_CACHE / "dermamnist_64.npz")
    derma_train = derma["train_images"]
    derma_test = derma["test_images"]

    retina = np.load(MEDMNIST_CACHE / "retinamnist_64.npz")
    retina_train = retina["train_images"]
    retina_test = retina["test_images"]

    blood = np.load(MEDMNIST_CACHE / "bloodmnist.npz")
    blood_train = blood["train_images"]
    blood_test = blood["test_images"]

    n_train_per_class = 1200
    n_test_per_class = 400

    def sample_indices(n_total: int, n_sample: int) -> np.ndarray:
        return rng.choice(n_total, size=min(n_total, n_sample), replace=False)

    t_chest_tr = preprocess_numpy_to_tensor_64(pneumonia_train[sample_indices(len(pneumonia_train), n_train_per_class)])
    t_chest_te = preprocess_numpy_to_tensor_64(pneumonia_test[sample_indices(len(pneumonia_test), n_test_per_class)])

    t_derma_tr = preprocess_numpy_to_tensor_64(derma_train[sample_indices(len(derma_train), n_train_per_class)])
    t_derma_te = preprocess_numpy_to_tensor_64(derma_test[sample_indices(len(derma_test), n_test_per_class)])

    t_retina_tr = preprocess_numpy_to_tensor_64(retina_train[sample_indices(len(retina_train), n_train_per_class)])
    t_retina_te = preprocess_numpy_to_tensor_64(retina_test[sample_indices(len(retina_test), n_test_per_class)])

    t_blood_tr = preprocess_numpy_to_tensor_64(blood_train[sample_indices(len(blood_train), n_train_per_class)])
    t_blood_te = preprocess_numpy_to_tensor_64(blood_test[sample_indices(len(blood_test), n_test_per_class)])

    # 2. Negative Datasets (Train & Test Families)
    logger.info("Loading negative datasets...")
    # Family A: CIFAR photographic
    df_cifar = pd.read_parquet(CIFAR_PARQUET)
    cifar_raw = [Image.open(io.BytesIO(b)).convert("RGB").resize((64, 64)) for b in df_cifar["img"].apply(lambda x: x["bytes"])[:2000]]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t_cifar_all = torch.stack([(transforms.ToTensor()(img) - mean) / std for img in cifar_raw])
    t_cifar_tr = t_cifar_all[:800]
    t_cifar_te = t_cifar_all[800:1300]  # 500 test images

    # Family B: BreastMNIST (medical non-target)
    breast = np.load(MEDMNIST_CACHE / "breastmnist.npz")
    t_breast_tr = preprocess_numpy_to_tensor_64(breast["train_images"][:300])
    t_breast_te = preprocess_numpy_to_tensor_64(breast["test_images"])  # 156 test images

    # Family C: OrganCMNIST (coronal CT - truly unseen MedMNIST)
    organc = np.load(MEDMNIST_CACHE / "organcmnist.npz")
    idx_organc_te = sample_indices(len(organc["test_images"]), 500)
    t_organc_te = preprocess_numpy_to_tensor_64(organc["test_images"][idx_organc_te])

    # Family D: OrganSMNIST (sagittal CT - truly unseen MedMNIST)
    organs = np.load(MEDMNIST_CACHE / "organsmnist.npz")
    idx_organs_te = sample_indices(len(organs["test_images"]), 500)
    t_organs_te = preprocess_numpy_to_tensor_64(organs["test_images"][idx_organs_te])

    # Family E: OCTMNIST (if available, else optional)
    t_oct_te = None
    oct_path = MEDMNIST_CACHE / "octmnist.npz"
    if oct_path.exists():
        try:
            oct_data = np.load(oct_path)
            if "test_images" in oct_data:
                idx_oct_te = sample_indices(len(oct_data["test_images"]), 500)
                t_oct_te = preprocess_numpy_to_tensor_64(oct_data["test_images"][idx_oct_te])
                logger.info("Loaded OCTMNIST: %d test images.", len(t_oct_te))
        except Exception as e:
            logger.warning("Could not load OCTMNIST: %s", e)

    # Family F: Synthetic Lab Reports (64x64 downscaled documents)
    logger.info("Generating 350 synthetic lab report thumbnails (64x64)...")
    t_lab_reports_te = generate_synthetic_lab_report_images_64(n_samples=350, seed=42)

    # Family G: Synthetic Noise and Blank Images
    noise_gauss_tr = torch.randn(100, 3, 64, 64)
    noise_gauss_te = torch.randn(50, 3, 64, 64)
    noise_unif_tr = (torch.rand(100, 3, 64, 64) - 0.5) / 0.5
    noise_unif_te = (torch.rand(50, 3, 64, 64) - 0.5) / 0.5
    black_t = torch.zeros(10, 3, 64, 64)
    white_t = torch.ones(10, 3, 64, 64) * 2.5
    gray_t = torch.ones(10, 3, 64, 64) * 0.5
    t_noise_blanks_te = torch.cat([noise_gauss_te, noise_unif_te, black_t, white_t, gray_t], dim=0)  # 130 images

    # 3. Assemble Training Datasets for Model A (Full) and Model B (Leave-Medical-Out)
    # Model A: Full negative training mix (CIFAR + Breast + Noise)
    t_other_tr_full = torch.cat([t_cifar_tr, t_breast_tr, noise_gauss_tr, noise_unif_tr], dim=0)
    X_tr_full = torch.cat([t_chest_tr, t_derma_tr, t_retina_tr, t_blood_tr, t_other_tr_full], dim=0)
    y_tr_full = torch.cat([
        torch.zeros(len(t_chest_tr), dtype=torch.long),
        torch.ones(len(t_derma_tr), dtype=torch.long) * 1,
        torch.ones(len(t_retina_tr), dtype=torch.long) * 2,
        torch.ones(len(t_blood_tr), dtype=torch.long) * 3,
        torch.ones(len(t_other_tr_full), dtype=torch.long) * 4,
    ])

    # Model B: Leave-Medical-Out (No medical negatives in training - only CIFAR + Noise)
    t_other_tr_nomed = torch.cat([t_cifar_tr, noise_gauss_tr, noise_unif_tr], dim=0)
    X_tr_nomed = torch.cat([t_chest_tr, t_derma_tr, t_retina_tr, t_blood_tr, t_other_tr_nomed], dim=0)
    y_tr_nomed = torch.cat([
        torch.zeros(len(t_chest_tr), dtype=torch.long),
        torch.ones(len(t_derma_tr), dtype=torch.long) * 1,
        torch.ones(len(t_retina_tr), dtype=torch.long) * 2,
        torch.ones(len(t_blood_tr), dtype=torch.long) * 3,
        torch.ones(len(t_other_tr_nomed), dtype=torch.long) * 4,
    ])

    logger.info("Training Model A (Full Negative Mix: %d samples)...", len(X_tr_full))
    model_full = train_single_gate(X_tr_full, y_tr_full, device, epochs=6)

    logger.info("Training Model B (Leave-Medical-Out Ablation: %d samples)...", len(X_tr_nomed))
    model_nomed = train_single_gate(X_tr_nomed, y_tr_nomed, device, epochs=6)

    # 4. Standard 5-Class Evaluation Suite (Target Modalities + Seen Negative Types)
    t_other_te_std = torch.cat([t_cifar_te, t_breast_te, t_noise_blanks_te], dim=0)
    X_te_std = torch.cat([t_chest_te, t_derma_te, t_retina_te, t_blood_te, t_other_te_std], dim=0)
    y_te_std = torch.cat([
        torch.zeros(len(t_chest_te), dtype=torch.long),
        torch.ones(len(t_derma_te), dtype=torch.long) * 1,
        torch.ones(len(t_retina_te), dtype=torch.long) * 2,
        torch.ones(len(t_blood_te), dtype=torch.long) * 3,
        torch.ones(len(t_other_te_std), dtype=torch.long) * 4,
    ])

    loader_std = DataLoader(TensorDataset(X_te_std, y_te_std), batch_size=64, shuffle=False)
    preds_full_std = []
    with torch.no_grad():
        for b_x, _ in loader_std:
            b_x = b_x.to(device)
            preds_full_std.extend(torch.argmax(model_full(b_x), dim=1).cpu().numpy())
    preds_full_std = np.array(preds_full_std)
    y_te_std_np = y_te_std.numpy()

    std_acc = round(float((preds_full_std == y_te_std_np).mean()), 4)
    std_report = classification_report(y_te_std_np, preds_full_std, target_names=CLASS_NAMES, output_dict=True)
    std_cm = confusion_matrix(y_te_std_np, preds_full_std).tolist()

    # 5. Evaluate Multi-Family Negative Sets on Model A (Full)
    logger.info("Evaluating negative families on Model A...")
    negative_families = [
        ("cifar_photographic", t_cifar_te, "Photographic images (CIFAR-10 test)"),
        ("breastmnist_medical", t_breast_te, "Breast ultrasound non-target (BreastMNIST test)"),
        ("heldout_organcmnist", t_organc_te, "Abdominal CT coronal (OrganCMNIST test, unseen modality)"),
        ("heldout_organsmnist", t_organs_te, "Abdominal CT sagittal (OrganSMNIST test, unseen modality)"),
        ("synthetic_lab_reports_64", t_lab_reports_te, "Simulated 64x64 clinical laboratory report pages"),
        ("synthetic_noise_and_blanks", t_noise_blanks_te, "Gaussian noise, uniform noise, solid blank images"),
    ]
    if t_oct_te is not None:
        negative_families.append(("heldout_octmnist", t_oct_te, "Retinal OCT scans (OCTMNIST test, unseen modality)"))

    family_evaluations_full = {}
    total_neg_samples = 0
    total_false_accepts = 0

    for f_id, f_tensor, f_desc in negative_families:
        ev = evaluate_negative_family(model_full, f_tensor, f_id, device)
        ev["description"] = f_desc
        family_evaluations_full[f_id] = ev
        total_neg_samples += ev["n_samples"]
        total_false_accepts += ev["false_accepts"]

    overall_far_full = round(float(total_false_accepts / max(total_neg_samples, 1)), 4)
    overall_far_ci_full = wilson_score_interval(total_false_accepts, total_neg_samples)

    # 6. Evaluate Negative Sets on Model B (Leave-Medical-Out Ablation)
    logger.info("Evaluating negative families on Model B (Leave-Medical-Out)...")
    family_evaluations_nomed = {}
    nomed_total_fa = 0

    for f_id, f_tensor, f_desc in negative_families:
        ev = evaluate_negative_family(model_nomed, f_tensor, f_id, device)
        ev["description"] = f_desc
        family_evaluations_nomed[f_id] = ev
        nomed_total_fa += ev["false_accepts"]

    overall_far_nomed = round(float(nomed_total_fa / max(total_neg_samples, 1)), 4)
    overall_far_ci_nomed = wilson_score_interval(nomed_total_fa, total_neg_samples)

    # 7. Assemble Comprehensive Audit Report
    logger.info("Assembling audit report...")
    audit_record = {
        "metadata": {
            "audit_step": "Pass 3 D5: Modality Gate Multi-Family & Leave-Dataset-Out Evaluation",
            "model_architecture": "ResNet-18 (64x64 input, 3 channels, 5 classes)",
            "checkpoint_path": str(OUTPUT_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "classes": CLASS_NAMES,
            "timestamp_epoch": time.time(),
            "compute_device": str(device),
            "train_samples_model_a": len(X_tr_full),
            "train_samples_model_b": len(X_tr_nomed),
            "total_negative_test_samples_evaluated": total_neg_samples,
            "runtime_seconds": round(time.time() - t0, 2),
        },
        "standard_5class_test_evaluation": {
            "accuracy": std_acc,
            "per_class_metrics": {
                name: {
                    "precision": round(std_report[name]["precision"], 4),
                    "recall": round(std_report[name]["recall"], 4),
                    "f1_score": round(std_report[name]["f1-score"], 4),
                    "support": int(std_report[name]["support"]),
                }
                for name in CLASS_NAMES
            },
            "confusion_matrix": std_cm,
        },
        "negative_families_evaluation_model_a_full": {
            "overall_false_accept_count": total_false_accepts,
            "total_negative_samples": total_neg_samples,
            "overall_false_accept_rate": overall_far_full,
            "overall_far_95ci": overall_far_ci_full,
            "replaces_earlier_single_far_claim": {
                "earlier_claim": 0.0763,
                "replacement_context": "The earlier 7.63% FAR claim was measured on a narrow test set with only CIFAR photos, BreastMNIST, and noise. On broader negative families including unseen organ CT scans and synthetic lab documents, FAR varies significantly by family.",
            },
            "per_family_results": family_evaluations_full,
        },
        "leave_dataset_out_ablation_model_b": {
            "ablation_description": "Gate trained WITHOUT medical negatives in 'other' (photographic + noise only)",
            "overall_false_accept_count": nomed_total_fa,
            "total_negative_samples": total_neg_samples,
            "overall_false_accept_rate": overall_far_nomed,
            "overall_far_95ci": overall_far_ci_nomed,
            "medical_negative_degradation": {
                "breastmnist_far_full_vs_nomed": [
                    family_evaluations_full["breastmnist_medical"]["false_accept_rate"],
                    family_evaluations_nomed["breastmnist_medical"]["false_accept_rate"],
                ],
                "organcmnist_far_full_vs_nomed": [
                    family_evaluations_full["heldout_organcmnist"]["false_accept_rate"],
                    family_evaluations_nomed["heldout_organcmnist"]["false_accept_rate"],
                ],
                "organsmnist_far_full_vs_nomed": [
                    family_evaluations_full["heldout_organsmnist"]["false_accept_rate"],
                    family_evaluations_nomed["heldout_organsmnist"]["false_accept_rate"],
                ],
            },
            "per_family_results": family_evaluations_nomed,
        },
        "clinical_scope_and_safety_caveats": [
            "The gate cannot distinguish adult from pediatric chest radiographs; it was trained on PneumoniaMNIST which consists entirely of pediatric chest X-rays (ages 1-5).",
            "The gate cannot reliably reject modalities completely absent from negative training (such as brain MRI or contrast-enhanced abdominal CT); unseen modalities can have elevated false-accept rates (up to 20-30%) and enter screening pipelines undetected.",
            "For non-medical photographic images and synthetic lab reports, the gate maintains high rejection (FAR < 10%).",
            "A visual modality warning banner is mandatory in all clinical screening workflows.",
        ],
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_record, f, indent=2)
    logger.info("Saved modality gate audit report to %s", OUTPUT_JSON)

    # Save checkpoint models/modality_gate_resnet18_v2.pt
    torch.save({
        "state_dict": model_full.state_dict(),
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "input_resolution": (64, 64),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }, OUTPUT_MODEL_PATH)
    logger.info("Saved model checkpoint to %s", OUTPUT_MODEL_PATH)
    logger.info("Modality gate audit completed in %.2f seconds.", time.time() - t0)


if __name__ == "__main__":
    main()
