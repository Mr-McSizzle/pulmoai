"""Benchmark Out-of-Distribution (OOD) Rejection Behavior.

Evaluates trained MedMNIST models on genuine cross-modality clinical OOD datasets
and non-medical semantic anomalies to report empirical False Accept Rates (FAR),
True Accept Rates (TAR), and OOD Detection AUROCs.
"""

from __future__ import annotations

import json
import logging
import ssl
from pathlib import Path
from typing import Any

import certifi
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import medmnist
from medmnist import (
    BloodMNIST,
    ChestMNIST,
    DermaMNIST,
    PneumoniaMNIST,
    RetinaMNIST,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ood_benchmark")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "medmnist"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "ood_benchmark.json"

MODALITY_CONFIGS = {
    "pneumoniamnist": {
        "cls": PneumoniaMNIST,
        "display_name": "Chest X-Ray (Pneumonia Screening)",
        "model_file": "pneumoniamnist_resnet18.pt",
        "channels": 1,
        "ood_candidates": ["dermamnist", "retinamnist"],
    },
    "dermamnist": {
        "cls": DermaMNIST,
        "display_name": "Dermatoscopy (Skin Lesions)",
        "model_file": "dermamnist_resnet18.pt",
        "channels": 3,
        "ood_candidates": ["pneumoniamnist", "retinamnist"],
    },
    "retinamnist": {
        "cls": RetinaMNIST,
        "display_name": "Retinal Fundus Photography",
        "model_file": "retinamnist_resnet18.pt",
        "channels": 3,
        "ood_candidates": ["pneumoniamnist", "dermamnist"],
    },
}

DATASET_CLASSES = {
    "pneumoniamnist": PneumoniaMNIST,
    "dermamnist": DermaMNIST,
    "retinamnist": RetinaMNIST,
}


def build_model(n_channels: int, n_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    if n_channels == 1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def get_dataset_tensors(dataset_id: str, split: str = "test", size: int = 64, target_channels: int = 1) -> torch.Tensor:
    cls = DATASET_CLASSES[dataset_id]
    ds = cls(split=split, download=False, size=size)
    images = ds.imgs  # numpy uint8 array (N, H, W) or (N, H, W, C)
    if images.ndim == 3:  # (N, H, W)
        images = images[:, np.newaxis, :, :]  # (N, 1, H, W)
        if target_channels == 3:
            images = np.repeat(images, 3, axis=1)
    elif images.ndim == 4:  # (N, H, W, C)
        images = np.transpose(images, (0, 3, 1, 2))  # (N, C, H, W)
        if target_channels == 1:
            # Grayscale conversion
            images = (0.2989 * images[:, 0:1, :, :] + 0.5870 * images[:, 1:2, :, :] + 0.1140 * images[:, 2:3, :, :]).astype(images.dtype)

    # Normalize to [0, 1] float tensor with standard ImageNet or simple scale
    tensor = torch.from_numpy(images).float() / 255.0
    # Standardize
    if target_channels == 1:
        mean, std = 0.5, 0.5
        tensor = (tensor - mean) / std
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    return tensor


def compute_max_softmax_probabilities(model: nn.Module, tensors: torch.Tensor, device: torch.device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    max_probs = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i : i + batch_size].to(device)
            logits = model(batch)
            if logits.shape[1] == 1:
                # Binary classification
                probs = torch.sigmoid(logits)
                p_max = torch.max(probs, 1.0 - probs).squeeze(-1)
            else:
                probs = torch.softmax(logits, dim=1)
                p_max = torch.max(probs, dim=1).values
            max_probs.append(p_max.cpu().numpy())
    return np.concatenate(max_probs)


def evaluate_ood_for_modality(modality_id: str, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    ckpt_path = MODEL_DIR / config["model_file"]
    if not ckpt_path.exists():
        logger.warning("Checkpoint for %s not found at %s; skipping.", modality_id, ckpt_path)
        return {}

    logger.info("Evaluating OOD behavior for %s...", modality_id)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_ch = checkpoint["n_channels"]
    n_cls = checkpoint["n_classes"]

    model = build_model(n_ch, n_cls)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 1. In-Distribution (ID) Test Split
    id_tensors = get_dataset_tensors(modality_id, split="test", size=64, target_channels=n_ch)
    id_msp = compute_max_softmax_probabilities(model, id_tensors, device)

    # 2. Genuine Cross-Modality OOD Datasets
    ood_results = {}
    combined_ood_msp = []

    for ood_id in config["ood_candidates"]:
        ood_tensors = get_dataset_tensors(ood_id, split="test", size=64, target_channels=n_ch)
        ood_msp = compute_max_softmax_probabilities(model, ood_tensors, device)
        combined_ood_msp.append(ood_msp)

        # Compute AUROC for distinguishing ID vs this specific OOD
        y_true = np.concatenate([np.ones(len(id_msp)), np.zeros(len(ood_msp))])
        scores = np.concatenate([id_msp, ood_msp])
        ood_auroc = float(roc_auc_score(y_true, scores))

        ood_results[ood_id] = {
            "sample_count": len(ood_msp),
            "mean_confidence": float(np.mean(ood_msp)),
            "ood_auroc": round(ood_auroc, 4),
            "false_accept_rate_at_050": round(float(np.mean(ood_msp >= 0.50)), 4),
            "false_accept_rate_at_070": round(float(np.mean(ood_msp >= 0.70)), 4),
            "false_accept_rate_at_085": round(float(np.mean(ood_msp >= 0.85)), 4),
            "false_accept_rate_at_095": round(float(np.mean(ood_msp >= 0.95)), 4),
        }

    # Combined Cross-Modality Clinical OOD
    combined_ood = np.concatenate(combined_ood_msp)
    y_true_comb = np.concatenate([np.ones(len(id_msp)), np.zeros(len(combined_ood))])
    scores_comb = np.concatenate([id_msp, combined_ood])
    overall_ood_auroc = float(roc_auc_score(y_true_comb, scores_comb))

    # FAR at 95% True Accept Rate (TAR)
    # Find threshold where 95% of ID samples are accepted
    p95_threshold = float(np.percentile(id_msp, 5))
    far_at_95_tar = float(np.mean(combined_ood >= p95_threshold))

    # 3. Far-OOD / Noise Baseline
    noise_tensors = torch.randn_like(id_tensors[:500])
    noise_msp = compute_max_softmax_probabilities(model, noise_tensors, device)
    noise_far_050 = float(np.mean(noise_msp >= 0.50))

    threshold_profiles = {
        "0.50": {
            "true_accept_rate": round(float(np.mean(id_msp >= 0.50)), 4),
            "false_accept_rate": round(float(np.mean(combined_ood >= 0.50)), 4),
        },
        "0.70": {
            "true_accept_rate": round(float(np.mean(id_msp >= 0.70)), 4),
            "false_accept_rate": round(float(np.mean(combined_ood >= 0.70)), 4),
        },
        "0.85": {
            "true_accept_rate": round(float(np.mean(id_msp >= 0.85)), 4),
            "false_accept_rate": round(float(np.mean(combined_ood >= 0.85)), 4),
        },
        "0.95": {
            "true_accept_rate": round(float(np.mean(id_msp >= 0.95)), 4),
            "false_accept_rate": round(float(np.mean(combined_ood >= 0.95)), 4),
        },
    }

    return {
        "modality_id": modality_id,
        "display_name": config["display_name"],
        "in_distribution_samples": len(id_msp),
        "mean_in_distribution_confidence": round(float(np.mean(id_msp)), 4),
        "total_ood_samples": len(combined_ood),
        "mean_cross_modality_ood_confidence": round(float(np.mean(combined_ood)), 4),
        "overall_ood_auroc": round(overall_ood_auroc, 4),
        "far_at_95_tar": round(far_at_95_tar, 4),
        "tar_95_threshold": round(p95_threshold, 4),
        "noise_far_at_050": round(noise_far_050, 4),
        "threshold_profiles": threshold_profiles,
        "cross_modality_breakdown": ood_results,
        "scientific_limitation_statement": (
            f"Confidence-based rejection (MSP) on {config['display_name']} achieves an OOD detection AUROC of "
            f"{overall_ood_auroc:.3f}. At the default 0.50 threshold, the empirical False Accept Rate across genuinely "
            f"out-of-distribution medical datasets is {threshold_profiles['0.50']['false_accept_rate'] * 100:.1f}%. "
            f"Increasing the threshold to 0.85 reduces the False Accept Rate to "
            f"{threshold_profiles['0.85']['false_accept_rate'] * 100:.1f}%, at a True Accept Rate cost of "
            f"{threshold_profiles['0.85']['true_accept_rate'] * 100:.1f}%. MSP confidence thresholding does NOT guarantee "
            "complete OOD separation; high-confidence errors can still occur on out-of-domain images."
        ),
    }


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Running OOD benchmark on device: %s", device)

    benchmark_output: dict[str, Any] = {
        "benchmark_title": "MedMNIST Cross-Modality Out-of-Distribution Empirical Evaluation",
        "methodology": "Maximum Softmax Probability (MSP) evaluated against genuine medical cross-modality datasets",
        "evaluation_timestamp": "2026-10-13T23:30:00Z",
        "modalities": {},
    }

    for modality_id, config in MODALITY_CONFIGS.items():
        res = evaluate_ood_for_modality(modality_id, config, device)
        if res:
            benchmark_output["modalities"][modality_id] = res

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_output, f, indent=2)

    logger.info("Saved OOD rejection benchmark to: %s", OUTPUT_PATH)
    print(f"\n[DONE] OOD Benchmark complete. Generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
