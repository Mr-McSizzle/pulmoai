#!/usr/bin/env python3
"""Train and Evaluate Modality Classifier Gate (Section C2).

Replaces ineffective penultimate OOD scoring with a supervised 5-class gate:
  Classes:
    0: chest_xray (PneumoniaMNIST)
    1: dermoscopy (DermaMNIST)
    2: fundus (RetinaMNIST)
    3: blood_smear (BloodMNIST)
    4: other (CIFAR photographic images, non-target MedMNIST like BreastMNIST, noise, blank images)

Evaluates on strictly held-out test data:
  - Held-out photographic images (CIFAR)
  - Held-out medical non-target images (BreastMNIST test)
  - Held-out synthetic noise and blank images
  - Held-out modality test sets (PneumoniaMNIST test, DermaMNIST test, RetinaMNIST test, BloodMNIST test)

Reports:
  - 5-class test accuracy
  - Per-class precision, recall, F1, confusion matrix
  - False-Accept Rate (FAR) for 'other' (rate at which non-modality images are misclassified as medical)

Saves:
  - Checkpoint: models/modality_gate_resnet18.pt
  - Metrics: reports/audit_modality_gate.json
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_modality_gate")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_MODEL_PATH = MODEL_DIR / "modality_gate_resnet18.pt"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "audit_modality_gate.json"
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
    # ImageNet normalization
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

        t = transforms.ToTensor()(pil_img)  # (3, 64, 64) in [0, 1]
        t = (t - mean) / std
        t_list.append(t)

    return torch.stack(t_list)


def main() -> None:
    t0 = time.time()
    logger.info("Building dataset for 5-class Modality Classifier Gate...")
    rng = np.random.RandomState(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using training compute device: %s", device)

    # 1. Load MedMNIST Modality Datasets
    logger.info("Loading chest_xray (PneumoniaMNIST)...")
    pneumonia = np.load(MEDMNIST_CACHE / "pneumoniamnist_64.npz")
    pneumonia_train = pneumonia["train_images"]
    pneumonia_test = pneumonia["test_images"]

    logger.info("Loading dermoscopy (DermaMNIST)...")
    derma = np.load(MEDMNIST_CACHE / "dermamnist_64.npz")
    derma_train = derma["train_images"]
    derma_test = derma["test_images"]

    logger.info("Loading fundus (RetinaMNIST)...")
    retina = np.load(MEDMNIST_CACHE / "retinamnist_64.npz")
    retina_train = retina["train_images"]
    retina_test = retina["test_images"]

    logger.info("Loading blood_smear (BloodMNIST)...")
    blood = np.load(MEDMNIST_CACHE / "bloodmnist.npz")
    blood_train = blood["train_images"]
    blood_test = blood["test_images"]

    # Sample balanced train and test subsets per modality (max 1500 train, max 500 test)
    n_train_per_class = 1200
    n_test_per_class = 400

    def sample_indices(n_total: int, n_sample: int) -> np.ndarray:
        return rng.choice(n_total, size=min(n_total, n_sample), replace=False)

    # Modality 0: Chest X-Ray
    idx_pneu_tr = sample_indices(len(pneumonia_train), n_train_per_class)
    idx_pneu_te = sample_indices(len(pneumonia_test), n_test_per_class)
    t_chest_tr = preprocess_numpy_to_tensor_64(pneumonia_train[idx_pneu_tr])
    t_chest_te = preprocess_numpy_to_tensor_64(pneumonia_test[idx_pneu_te])

    # Modality 1: Dermoscopy
    idx_derma_tr = sample_indices(len(derma_train), n_train_per_class)
    idx_derma_te = sample_indices(len(derma_test), n_test_per_class)
    t_derma_tr = preprocess_numpy_to_tensor_64(derma_train[idx_derma_tr])
    t_derma_te = preprocess_numpy_to_tensor_64(derma_test[idx_derma_te])

    # Modality 2: Fundus
    idx_retina_tr = sample_indices(len(retina_train), n_train_per_class)
    idx_retina_te = sample_indices(len(retina_test), n_test_per_class)
    t_retina_tr = preprocess_numpy_to_tensor_64(retina_train[idx_retina_tr])
    t_retina_te = preprocess_numpy_to_tensor_64(retina_test[idx_retina_te])

    # Modality 3: Blood Smear
    idx_blood_tr = sample_indices(len(blood_train), n_train_per_class)
    idx_blood_te = sample_indices(len(blood_test), n_test_per_class)
    t_blood_tr = preprocess_numpy_to_tensor_64(blood_train[idx_blood_tr])
    t_blood_te = preprocess_numpy_to_tensor_64(blood_test[idx_blood_te])

    # Modality 4: Other (Negatives)
    # Source A: CIFAR photographic images
    logger.info("Loading CIFAR photographic images for 'other' negative class...")
    df_cifar = pd.read_parquet(CIFAR_PARQUET)
    cifar_raw = [Image.open(io.BytesIO(b)).convert("RGB").resize((64, 64)) for b in df_cifar["img"].apply(lambda x: x["bytes"])[:2000]]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t_cifar_all = torch.stack([(transforms.ToTensor()(img) - mean) / std for img in cifar_raw])
    t_cifar_tr = t_cifar_all[:800]
    t_cifar_te = t_cifar_all[800:1300]

    # Source B: Non-target MedMNIST (BreastMNIST)
    breast = np.load(MEDMNIST_CACHE / "breastmnist.npz")
    t_breast_tr = preprocess_numpy_to_tensor_64(breast["train_images"][:300])
    t_breast_te = preprocess_numpy_to_tensor_64(breast["test_images"])

    # Source C: Synthetic Noise and Blank Images
    # Gaussian noise
    noise_gauss_tr = torch.randn(100, 3, 64, 64)
    noise_gauss_te = torch.randn(50, 3, 64, 64)
    # Uniform noise
    noise_unif_tr = (torch.rand(100, 3, 64, 64) - 0.5) / 0.5
    noise_unif_te = (torch.rand(50, 3, 64, 64) - 0.5) / 0.5
    # Blank images (black, white, gray)
    black_t = torch.zeros(10, 3, 64, 64)
    white_t = torch.ones(10, 3, 64, 64) * 2.5
    gray_t = torch.ones(10, 3, 64, 64) * 0.5
    blanks_te = torch.cat([black_t, white_t, gray_t], dim=0)

    t_other_tr = torch.cat([t_cifar_tr, t_breast_tr, noise_gauss_tr, noise_unif_tr], dim=0)
    t_other_te = torch.cat([t_cifar_te, t_breast_te, noise_gauss_te, noise_unif_te, blanks_te], dim=0)

    logger.info(
        "Built classes: chest_xray=(%d tr, %d te), dermoscopy=(%d tr, %d te), fundus=(%d tr, %d te), blood=(%d tr, %d te), other=(%d tr, %d te)",
        len(t_chest_tr), len(t_chest_te),
        len(t_derma_tr), len(t_derma_te),
        len(t_retina_tr), len(t_retina_te),
        len(t_blood_tr), len(t_blood_te),
        len(t_other_tr), len(t_other_te),
    )

    # Assemble Train Tensors
    X_train = torch.cat([t_chest_tr, t_derma_tr, t_retina_tr, t_blood_tr, t_other_tr], dim=0)
    y_train = torch.cat([
        torch.zeros(len(t_chest_tr), dtype=torch.long),
        torch.ones(len(t_derma_tr), dtype=torch.long) * 1,
        torch.ones(len(t_retina_tr), dtype=torch.long) * 2,
        torch.ones(len(t_blood_tr), dtype=torch.long) * 3,
        torch.ones(len(t_other_tr), dtype=torch.long) * 4,
    ])

    # Assemble Test Tensors
    X_test = torch.cat([t_chest_te, t_derma_te, t_retina_te, t_blood_te, t_other_te], dim=0)
    y_test = torch.cat([
        torch.zeros(len(t_chest_te), dtype=torch.long),
        torch.ones(len(t_derma_te), dtype=torch.long) * 1,
        torch.ones(len(t_retina_te), dtype=torch.long) * 2,
        torch.ones(len(t_blood_te), dtype=torch.long) * 3,
        torch.ones(len(t_other_te), dtype=torch.long) * 4,
    ])

    logger.info("Total Train Samples: %d | Total Test Samples: %d", len(X_train), len(X_test))

    # Shuffle training set
    perm = torch.randperm(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    # Initialize model
    model = build_modality_gate(num_classes=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train model
    epochs = 6
    logger.info("Training Modality Gate ResNet-18 for %d epochs...", epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            optimizer.zero_grad()
            logits = model(b_x)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(b_y)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == b_y).sum().item())
            total += len(b_y)

        train_acc = correct / total
        avg_loss = total_loss / total
        logger.info("Epoch %d/%d - Loss: %.4f - Accuracy: %.2f%%", epoch, epochs, avg_loss, train_acc * 100)

    # Evaluate on held-out test data
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for b_x, b_y in test_loader:
            b_x = b_x.to(device)
            logits = model(b_x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(b_y.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    all_probs_arr = np.array(all_probs)

    overall_accuracy = round(float(np.mean(all_preds_arr == all_targets_arr)), 4)
    cm = confusion_matrix(all_targets_arr, all_preds_arr).tolist()
    rep = classification_report(all_targets_arr, all_preds_arr, target_names=CLASS_NAMES, output_dict=True)

    # Compute False-Accept Rate (FAR) for 'other' class
    # Definition: What fraction of 'other' (class 4) images were classified as medical (classes 0, 1, 2, or 3)?
    other_mask = all_targets_arr == 4
    other_total = int(other_mask.sum())
    other_accepted_as_medical = int(np.sum(all_preds_arr[other_mask] < 4))
    far_other = round(other_accepted_as_medical / other_total, 4) if other_total > 0 else 0.0

    # Specific break-down for 'other' sub-datasets
    # CIFAR held-out photographic:
    cifar_te_len = len(t_cifar_te)
    # The first cifar_te_len items of 'other' are CIFAR
    cifar_preds = all_preds_arr[all_targets_arr == 4][:cifar_te_len]
    cifar_far = round(float(np.mean(cifar_preds < 4)), 4)

    logger.info("=" * 80)
    logger.info("MODALITY GATE EVALUATION RESULTS:")
    logger.info("  Overall 5-Class Test Accuracy: %.2f%%", overall_accuracy * 100)
    logger.info("  False Accept Rate for 'other': %.2f%% (%d/%d false accepts)", far_other * 100, other_accepted_as_medical, other_total)
    logger.info("  False Accept Rate on held-out CIFAR photos: %.2f%%", cifar_far * 100)
    for c_name in CLASS_NAMES:
        c_p = rep[c_name]["precision"]
        c_r = rep[c_name]["recall"]
        c_f1 = rep[c_name]["f1-score"]
        logger.info("  Class %-12s: Prec=%.2f%%, Rec=%.2f%%, F1=%.4f", c_name, c_p * 100, c_r * 100, c_f1)
    logger.info("=" * 80)

    # Save model checkpoint
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "input_resolution": (64, 64),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "overall_test_accuracy": overall_accuracy,
        "far_for_other": far_other,
    }, OUTPUT_MODEL_PATH)
    logger.info("Saved modality gate model to %s", OUTPUT_MODEL_PATH)

    # Save benchmark JSON
    payload = {
        "metadata": {
            "audit_step": "Section C2: Modality Classifier Gate",
            "model_architecture": "ResNet-18 (64x64 input, 3 channels, 5 classes)",
            "classes": CLASS_NAMES,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "compute_device": str(device),
            "train_samples_total": len(X_train),
            "test_samples_total": len(X_test),
            "runtime_seconds": round(time.time() - t0, 2),
        },
        "evaluation_metrics": {
            "overall_accuracy": overall_accuracy,
            "false_accept_rate_for_other": far_other,
            "false_accept_rate_on_cifar_photos": cifar_far,
            "meets_below_10pct_target": bool(far_other < 0.10),
            "per_class_metrics": {
                c_name: {
                    "precision": round(rep[c_name]["precision"], 4),
                    "recall": round(rep[c_name]["recall"], 4),
                    "f1_score": round(rep[c_name]["f1-score"], 4),
                    "support": int(rep[c_name]["support"]),
                }
                for c_name in CLASS_NAMES
            },
            "confusion_matrix": cm,
            "confusion_matrix_labels": CLASS_NAMES,
        },
        "integration_policy": {
            "mismatch_behavior": "If gate predicted modality != user-selected modality, reject request and refuse to execute screening model.",
            "warning_banner_status": "Warning banner maintained because model is trained on curated datasets (MedMNIST, CIFAR) and does not guarantee detection of arbitrary arbitrary real-world photos.",
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved modality gate audit results to %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
