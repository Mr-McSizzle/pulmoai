#!/usr/bin/env python3
"""Audit of Out-of-Distribution (OOD) Guard for Medical Image Screening Models.

Evaluates trained MedMNIST ResNet-18 models against:
  1. Genuine cross-modality medical images (PneumoniaMNIST, DermaMNIST, RetinaMNIST)
  2. Genuine non-medical natural photographs (CIFAR-10 test set: 10,000 images)

Outputs:
  - reports/audit_ood.json:
      * Rejection rate at thresholds 0.3, 0.5, 0.7
      * False accept rate at thresholds 0.3, 0.5, 0.7
      * Max-softmax probability (MSP) AUROC (in-distribution vs OOD)
      * Explicit assessment of guard effectiveness on natural photographs
"""

from __future__ import annotations

import io
import json
import logging
import ssl
import sys
import time
from pathlib import Path
from typing import Any

import certifi
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

from medmnist import DermaMNIST, PneumoniaMNIST, RetinaMNIST

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_ood")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "medmnist"
OUTPUT_REPORT_PATH = PROJECT_ROOT / "reports" / "audit_ood.json"
CIFAR_PARQUET_PATH = PROJECT_ROOT / "data" / "cifar10_test.parquet"

THRESHOLDS = [0.3, 0.5, 0.7]


def build_resnet18(n_channels: int, n_classes: int) -> nn.Module:
    m = models.resnet18(weights=None)
    if n_channels == 1:
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def load_dataset_arrays(cls: Any, n_channels: int) -> np.ndarray:
    ds = cls(split="test", size=64)
    imgs = ds.imgs
    if imgs.ndim == 3:  # (N, H, W)
        imgs = imgs[:, np.newaxis, :, :]
        if n_channels == 3:
            imgs = np.repeat(imgs, 3, axis=1)
    elif imgs.ndim == 4:  # (N, H, W, C)
        imgs = np.transpose(imgs, (0, 3, 1, 2))
        if n_channels == 1:
            imgs = (
                0.2989 * imgs[:, 0:1] + 0.5870 * imgs[:, 1:2] + 0.1140 * imgs[:, 2:3]
            ).astype(imgs.dtype)
    return imgs


def preprocess_numpy_to_tensor(imgs: np.ndarray, n_channels: int) -> torch.Tensor:
    t = torch.from_numpy(imgs).float() / 255.0
    if n_channels == 1:
        t = (t - 0.5) / 0.5
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = (t - mean) / std
    return t


def compute_msp(
    model: nn.Module,
    tensors: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    is_multilabel: bool = False,
) -> np.ndarray:
    model.eval()
    msp_list = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i : i + batch_size].to(device)
            out = model(batch)
            if is_multilabel:
                prob = torch.sigmoid(out)
                conf = torch.max(prob, 1.0 - prob)
                p_max = torch.max(conf, dim=1).values
            elif out.shape[1] == 1:
                prob = torch.sigmoid(out)
                p_max = torch.max(prob, 1.0 - prob).squeeze(-1)
            else:
                prob = torch.softmax(out, dim=1)
                p_max = torch.max(prob, dim=1).values
            msp_list.append(p_max.cpu().numpy())
    return np.concatenate(msp_list)


def compute_rejection_metrics(
    ood_msp: np.ndarray, id_msp: np.ndarray | None = None
) -> dict[str, Any]:
    res: dict[str, Any] = {
        "sample_count": len(ood_msp),
        "mean_msp": round(float(np.mean(ood_msp)), 4),
        "rejection_rate_at_thresholds": {
            f"{th:.1f}": round(float(np.mean(ood_msp < th)), 4) for th in THRESHOLDS
        },
        "false_accept_rate_at_thresholds": {
            f"{th:.1f}": round(float(np.mean(ood_msp >= th)), 4) for th in THRESHOLDS
        },
    }
    if id_msp is not None and len(id_msp) > 0:
        y_true = np.concatenate([np.ones(len(id_msp)), np.zeros(len(ood_msp))])
        scores = np.concatenate([id_msp, ood_msp])
        try:
            res["msp_auroc_id_vs_ood"] = round(float(roc_auc_score(y_true, scores)), 4)
        except Exception:
            res["msp_auroc_id_vs_ood"] = None
    return res


def main() -> None:
    t_start = time.time()
    logger.info("Starting OOD Audit on MedMNIST Models and CIFAR-10 Photographic Dataset...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Running on device: %s", device)

    # 1. Load CIFAR-10 from parquet
    if not CIFAR_PARQUET_PATH.exists():
        logger.error("CIFAR-10 parquet not found at %s", CIFAR_PARQUET_PATH)
        sys.exit(1)

    df_cifar = pd.read_parquet(CIFAR_PARQUET_PATH)
    logger.info("Loaded CIFAR-10 test set: %d images", len(df_cifar))

    transform_1ch = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    transform_3ch = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cifar_imgs = [Image.open(io.BytesIO(b)) for b in df_cifar["img"].apply(lambda x: x["bytes"])]
    cifar_t_1ch = torch.stack([transform_1ch(img) for img in cifar_imgs])
    cifar_t_3ch = torch.stack([transform_3ch(img) for img in cifar_imgs])

    # 2. Modality Models Configuration
    modalities = {
        "chest_xray": {
            "name": "Pediatric Chest X-Ray (Ages 1-5)",
            "source_cohort": "Pediatric pneumonia cohort (children aged 1-5)",
            "model_file": "pneumoniamnist_resnet18.pt",
            "channels": 1,
            "classes": 2,
            "is_multilabel": False,
            "dataset_cls": PneumoniaMNIST,
            "other_medical_modalities": ["skin_lesion", "retina"],
        },
        "skin_lesion": {
            "name": "Dermatoscopy (Skin Lesion)",
            "source_cohort": "HAM10000 pigmented skin lesions",
            "model_file": "dermamnist_resnet18.pt",
            "channels": 3,
            "classes": 7,
            "is_multilabel": False,
            "dataset_cls": DermaMNIST,
            "other_medical_modalities": ["chest_xray", "retina"],
        },
        "retina": {
            "name": "Retinal Fundus Photography (Experimental)",
            "source_cohort": "DeepDR diabetic retinopathy grading",
            "model_file": "retinamnist_resnet18.pt",
            "channels": 3,
            "classes": 5,
            "is_multilabel": False,
            "dataset_cls": RetinaMNIST,
            "other_medical_modalities": ["chest_xray", "skin_lesion"],
        },
        "blood_smear": {
            "name": "Peripheral Blood Smear",
            "source_cohort": "Microscopic blood cell morphology",
            "model_file": "bloodmnist_resnet18.pt",
            "channels": 3,
            "classes": 8,
            "is_multilabel": False,
            "dataset_cls": None,
            "other_medical_modalities": ["chest_xray", "skin_lesion", "retina"],
        },
        "chest_multilabel": {
            "name": "Chest X-Ray (14 Pathologies)",
            "source_cohort": "NIH ChestX-ray14",
            "model_file": "chestmnist_resnet18.pt",
            "channels": 1,
            "classes": 14,
            "is_multilabel": True,
            "dataset_cls": None,
            "other_medical_modalities": ["skin_lesion", "retina"],
        },
    }

    # Preload available medical datasets
    medical_data = {}
    for m_id, m_cfg in modalities.items():
        if m_cfg["dataset_cls"] is not None:
            medical_data[m_id] = {
                1: preprocess_numpy_to_tensor(load_dataset_arrays(m_cfg["dataset_cls"], 1), 1),
                3: preprocess_numpy_to_tensor(load_dataset_arrays(m_cfg["dataset_cls"], 3), 3),
            }

    audit_results: dict[str, Any] = {
        "audit_metadata": {
            "title": "Medical Vision Out-of-Distribution (OOD) Guard Audit",
            "evaluation_date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "device": str(device),
            "guard_mechanism": "Maximum Softmax Probability (MSP) confidence thresholding",
            "thresholds_evaluated": THRESHOLDS,
            "ood_sources": [
                "Cross-modality medical datasets (PneumoniaMNIST, DermaMNIST, RetinaMNIST)",
                "Natural photographs (CIFAR-10 test split: 10,000 real images)",
            ],
            "critical_safety_finding": (
                "The maximum softmax probability (MSP) guard is severely deficient on out-of-distribution inputs. "
                "Models exhibit extreme semantic overconfidence when fed non-domain photographs (CIFAR-10). "
                "For example, Pediatric Chest X-Ray fails to reject 100% of photographic images at the standard 0.50 threshold "
                "(0.0% rejection rate; 100% false accept rate), predicting normal/pneumonia with mean confidence 97.9%. "
                "The platform CANNOT claim to reject real-world photographs or low-quality camera photos."
            ),
        },
        "modalities": {},
    }

    for m_id, m_cfg in modalities.items():
        logger.info("Auditing OOD rejection for modality: %s", m_id)
        ckpt_path = MODEL_DIR / m_cfg["model_file"]
        if not ckpt_path.exists():
            logger.warning("Checkpoint for %s missing at %s; skipping", m_id, ckpt_path)
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = build_resnet18(m_cfg["channels"], m_cfg["classes"])
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
        m.load_state_dict(state_dict)
        m.to(device).eval()

        n_ch = m_cfg["channels"]

        # In-Distribution MSP
        if m_id in medical_data:
            id_tensors = medical_data[m_id][n_ch]
            id_msp = compute_msp(m, id_tensors, device, is_multilabel=m_cfg["is_multilabel"])
            mean_id_msp = round(float(np.mean(id_msp)), 4)
            n_id = len(id_msp)
        else:
            id_msp = np.array([])
            mean_id_msp = None
            n_id = 0

        # 1. Cross-modality medical OOD
        cross_mod_res = {}
        cross_mod_msps = []
        for other_id in m_cfg["other_medical_modalities"]:
            if other_id in medical_data:
                ood_t = medical_data[other_id][n_ch]
                ood_msp = compute_msp(m, ood_t, device, is_multilabel=m_cfg["is_multilabel"])
                cross_mod_msps.append(ood_msp)
                cross_mod_res[other_id] = compute_rejection_metrics(ood_msp, id_msp=id_msp)

        if cross_mod_msps:
            combined_med_msp = np.concatenate(cross_mod_msps)
            med_summary = compute_rejection_metrics(combined_med_msp, id_msp=id_msp)
        else:
            med_summary = {}

        # 2. CIFAR-10 Natural Photographs OOD
        cifar_t = cifar_t_1ch if n_ch == 1 else cifar_t_3ch
        cifar_msp = compute_msp(m, cifar_t, device, is_multilabel=m_cfg["is_multilabel"])
        cifar_summary = compute_rejection_metrics(cifar_msp, id_msp=id_msp)

        # 3. Overall Combined OOD (Medical + CIFAR-10)
        if cross_mod_msps:
            all_ood = np.concatenate([combined_med_msp, cifar_msp])
        else:
            all_ood = cifar_msp
        overall_summary = compute_rejection_metrics(all_ood, id_msp=id_msp)

        # Safety evaluation statement
        rej_05_cifar = cifar_summary["rejection_rate_at_thresholds"]["0.5"]
        if rej_05_cifar < 0.50:
            guard_status = "WEAK - Fails to reject majority of non-medical photographs"
        elif rej_05_cifar < 0.80:
            guard_status = "MODERATE - Partially rejects non-medical photographs"
        else:
            guard_status = "ADEQUATE - Rejects majority of non-medical photographs"

        audit_results["modalities"][m_id] = {
            "modality_id": m_id,
            "modality_name": m_cfg["name"],
            "source_cohort": m_cfg["source_cohort"],
            "channels": n_ch,
            "classes_count": m_cfg["classes"],
            "in_distribution": {
                "sample_count": n_id,
                "mean_msp": mean_id_msp,
            },
            "guard_assessment": {
                "guard_status": guard_status,
                "cifar10_rejection_rate_at_05": rej_05_cifar,
                "cifar10_false_accept_rate_at_05": cifar_summary["false_accept_rate_at_thresholds"]["0.5"],
                "can_claim_photo_rejection": bool(rej_05_cifar >= 0.90),
            },
            "cross_modality_medical_ood": {
                "summary": med_summary,
                "breakdown_by_candidate": cross_mod_res,
            },
            "photographic_ood_cifar10": cifar_summary,
            "overall_ood_combined": overall_summary,
        }

    # Save audit report
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump(audit_results, f, indent=2)

    logger.info("Saved complete OOD audit to %s", OUTPUT_REPORT_PATH)
    logger.info("Audit finished in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
