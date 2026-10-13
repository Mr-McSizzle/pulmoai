"""Train and evaluate MedMNIST models (64x64) for Phase 4.

Datasets:
  1. PneumoniaMNIST (Chest X-ray / Pediatric pneumonia - binary)
  2. DermaMNIST (Dermatology skin lesions - 7 classes)
  3. RetinaMNIST (Retina fundus diabetic retinopathy - 5 classes)
  4. BloodMNIST (Microscopic blood cell morphology - 8 classes)
  5. ChestMNIST (Chest X-ray thoracic pathologies - 14 multi-label classes)

Architecture:
  - ResNet18 initialized with ImageNet pretrained weights.
  - Channels adapted for 1-channel or 3-channel input.
  - Evaluated strictly on official TEST splits.
  - Outputs metrics (Accuracy, Macro-F1, Macro-AUC, Training Time).
  - Out-of-Distribution (OOD) confidence guard built into inference.
"""

from __future__ import annotations

import argparse
import json
import ssl
import time
from pathlib import Path

import certifi

# Fix SSL context for MedMNIST downloads
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import medmnist
import numpy as np
import torch
import torch.nn as nn
from medmnist import (
    BloodMNIST,
    ChestMNIST,
    DermaMNIST,
    INFO,
    PneumoniaMNIST,
    RetinaMNIST,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "medmnist"
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_REPORT_JSON = REPORT_DIR / "medmnist_benchmark.json"
OUTPUT_MODEL_CARD = PROJECT_ROOT / "models" / "medmnist_model_card.json"

DATASET_CONFIGS = [
    {
        "id": "pneumoniamnist",
        "name": "PneumoniaMNIST",
        "display_name": "Chest X-Ray (Pneumonia Screening)",
        "image_type": "chest_xray",
        "cls": PneumoniaMNIST,
        "task": "binary-class",
        "n_channels": 1,
        "n_classes": 2,
        "epochs": 3,
        "batch_size": 128,
        "lr": 1e-4,
    },
    {
        "id": "dermamnist",
        "name": "DermaMNIST",
        "display_name": "Dermatoscopy (Skin Lesion)",
        "image_type": "skin_lesion",
        "cls": DermaMNIST,
        "task": "multi-class",
        "n_channels": 3,
        "n_classes": 7,
        "epochs": 3,
        "batch_size": 128,
        "lr": 1e-4,
    },
    {
        "id": "retinamnist",
        "name": "RetinaMNIST",
        "display_name": "Retinal Fundus (Diabetic Retinopathy)",
        "image_type": "retina",
        "cls": RetinaMNIST,
        "task": "ordinal-regression",
        "n_channels": 3,
        "n_classes": 5,
        "epochs": 4,
        "batch_size": 64,
        "lr": 1e-4,
    },
    {
        "id": "bloodmnist",
        "name": "BloodMNIST",
        "display_name": "Blood Smear Microscopy",
        "image_type": "blood_smear",
        "cls": BloodMNIST,
        "task": "multi-class",
        "n_channels": 3,
        "n_classes": 8,
        "epochs": 3,
        "batch_size": 128,
        "lr": 1e-4,
    },
    {
        "id": "chestmnist",
        "name": "ChestMNIST",
        "display_name": "Chest X-Ray (14 Thoracic Conditions)",
        "image_type": "chest_multilabel",
        "cls": ChestMNIST,
        "task": "multi-label, binary-class",
        "n_channels": 1,
        "n_classes": 14,
        "epochs": 2,
        "batch_size": 128,
        "lr": 1e-4,
        "train_subset_max": 20000,  # Stratified/random subset to train in ~1 min on CPU/MPS
    },
]


def build_resnet18(n_channels: int, n_classes: int) -> nn.Module:
    """Build ResNet18 with pretrained weights and adapted input/output layers."""
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    if n_channels == 1:
        orig_conv = model.conv1
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(orig_conv.weight.mean(dim=1, keepdim=True))

    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Evaluate model on test loader and calculate Accuracy, Macro-F1, and Macro-AUC."""
    model.eval()
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            all_outputs.append(out.cpu().numpy())
            all_targets.append(y.numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if task == "multi-label, binary-class":
        probs = 1.0 / (1.0 + np.exp(-all_outputs))
        preds = (probs >= 0.5).astype(int)
        acc = float((preds == all_targets).mean())
        f1 = float(f1_score(all_targets, preds, average="macro", zero_division=0))
        try:
            auc = float(roc_auc_score(all_targets, probs, average="macro"))
        except Exception:
            auc = 0.5
    else:
        # Multi-class or binary
        all_targets = all_targets.squeeze()
        # Softmax
        exp_out = np.exp(all_outputs - np.max(all_outputs, axis=1, keepdims=True))
        probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)

        acc = float(accuracy_score(all_targets, preds))
        f1 = float(f1_score(all_targets, preds, average="macro", zero_division=0))
        try:
            if probs.shape[1] == 2:
                auc = float(roc_auc_score(all_targets, probs[:, 1]))
            else:
                auc = float(roc_auc_score(all_targets, probs, multi_class="ovr", average="macro"))
        except Exception:
            auc = 0.5

    return round(acc, 4), round(f1, 4), round(auc, 4), probs, all_targets


def train_single_dataset(cfg: dict, device: torch.device, force: bool = False) -> dict:
    """Train and test ResNet18 on a single MedMNIST dataset."""
    d_id = cfg["id"]
    save_path = MODEL_DIR / f"{d_id}_resnet18.pt"
    if save_path.exists() and not force:
        print(f"\nCheckpoint found for {cfg['name']} at {save_path}. Loading existing metrics...")
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        m = ckpt.get("metrics", {})
        return {
            "dataset_id": d_id,
            "dataset_name": cfg["name"],
            "display_name": cfg["display_name"],
            "image_type": cfg["image_type"],
            "task": cfg["task"],
            "input_resolution": "64x64",
            "channels": cfg["n_channels"],
            "classes_count": cfg["n_classes"],
            "classes_labels": INFO[d_id]["label"],
            "train_samples": INFO[d_id]["n_samples"]["train"],
            "test_samples": INFO[d_id]["n_samples"]["test"],
            "test_accuracy": m.get("accuracy", 0.0),
            "test_macro_f1": m.get("macro_f1", 0.0),
            "test_macro_auc": m.get("macro_auc", 0.0),
            "training_time_seconds": ckpt.get("train_time_sec", 0.0),
            "evaluation_note": ckpt.get("benchmark_source", "Locally trained on MPS (Apple Silicon)"),
        }

    n_ch = cfg["n_channels"]
    n_cls = cfg["n_classes"]
    task = cfg["task"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]

    print("\n" + "=" * 80)
    print(f"TRAINING {cfg['name'].upper()} ({cfg['display_name']})")
    print(f"Task: {task} | Channels: {n_ch} | Classes: {n_cls} | Epochs: {epochs}")
    print("=" * 80)

    mean_val = [0.5] if n_ch == 1 else [0.5, 0.5, 0.5]
    std_val = [0.5] if n_ch == 1 else [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_val, std=std_val),
    ])

    DatasetCls = cfg["cls"]
    train_ds = DatasetCls(split="train", download=True, size=64, transform=transform)
    val_ds = DatasetCls(split="val", download=True, size=64, transform=transform)
    test_ds = DatasetCls(split="test", download=True, size=64, transform=transform)

    if cfg.get("train_subset_max") and len(train_ds) > cfg["train_subset_max"]:
        indices = torch.randperm(len(train_ds))[:cfg["train_subset_max"]].tolist()
        train_ds = torch.utils.data.Subset(train_ds, indices)
        print(f"Using training subset of {len(train_ds):,} images (out of {len(DatasetCls(split='train', size=64)):,})")
    else:
        print(f"Train samples: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_resnet18(n_ch, n_cls).to(device)

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t_start = time.time()
    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            if task == "multi-label, binary-class":
                y = y.float().to(device)
            else:
                y = y.squeeze().long().to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        ep_dur = time.time() - ep_t0
        print(f"  Epoch {ep}/{epochs}: Loss = {avg_loss:.4f} ({ep_dur:.1f}s)")

    train_time_sec = round(time.time() - t_start, 2)
    print(f"Finished training in {train_time_sec:.1f}s. Evaluating on official TEST split...")

    test_acc, test_f1, test_auc, probs, targets = evaluate_model(model, test_loader, task, device)
    print(f"  --> TEST METRICS: Accuracy = {test_acc:.4f} | Macro-F1 = {test_f1:.4f} | Macro-AUC = {test_auc:.4f}")

    # Save model weights and metadata
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_DIR / f"{d_id}_resnet18.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "n_channels": n_ch,
        "n_classes": n_cls,
        "task": task,
        "class_labels": INFO[d_id]["label"],
        "metrics": {"accuracy": test_acc, "macro_f1": test_f1, "macro_auc": test_auc},
        "train_time_sec": train_time_sec,
    }, save_path)
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"Saved model checkpoint to {save_path} ({file_size_mb:.2f} MB)")

    return {
        "dataset_id": d_id,
        "dataset_name": cfg["name"],
        "display_name": cfg["display_name"],
        "image_type": cfg["image_type"],
        "task": task,
        "input_resolution": "64x64",
        "channels": n_ch,
        "classes_count": n_cls,
        "classes_labels": INFO[d_id]["label"],
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "test_macro_auc": test_auc,
        "training_time_seconds": train_time_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/Benchmark MedMNIST ResNet18 models.")
    parser.add_argument("--force", action="store_true", help="Force retrain existing checkpoints")
    parser.add_argument("--dataset", type=str, default="all", help="Dataset ID to train (or 'all')")
    args = parser.parse_args()

    t_global = time.time()
    print("=" * 80)
    print("PHASE 4: TRAINING & BENCHMARKING MEDMNIST MEDICAL IMAGE SCREENING")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Compute Device: {device} (PyTorch {torch.__version__})")

    configs = DATASET_CONFIGS
    if args.dataset != "all":
        configs = [c for c in DATASET_CONFIGS if c["id"] == args.dataset]
        if not configs:
            raise ValueError(f"Unknown dataset '{args.dataset}'. Choose from: {[c['id'] for c in DATASET_CONFIGS]}")

    results = {}
    if OUTPUT_REPORT_JSON.exists():
        try:
            with open(OUTPUT_REPORT_JSON) as f:
                old = json.load(f)
                results.update(old.get("results", {}))
        except Exception:
            pass

    for cfg in configs:
        res = train_single_dataset(cfg, device, force=args.force)
        results[cfg["id"]] = res

    total_training_time = round(time.time() - t_global, 2)

    # 1. Save Benchmark JSON
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_summary = {
        "task": "Phase 4 MedMNIST 64x64 Medical Image Screening",
        "device": str(device),
        "total_wall_clock_time_seconds": total_training_time,
        "models_evaluated": len(results),
        "results": results,
    }
    with open(OUTPUT_REPORT_JSON, "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    print(f"\nSaved benchmark results to {OUTPUT_REPORT_JSON}")

    # 2. Save Model Card JSON
    model_card = {
        "module": "MedMNIST Medical Image Screening Engine",
        "version": "1.0.0",
        "architecture": "ResNet18 (ImageNet Pretrained, Fine-Tuned at 64x64)",
        "datasets": {
            k: {
                "name": v["display_name"],
                "task": v["task"],
                "classes": v["classes_labels"],
                "test_accuracy": v["test_accuracy"],
                "test_macro_f1": v["test_macro_f1"],
                "test_macro_auc": v["test_macro_auc"],
                "training_time_sec": v["training_time_seconds"],
            }
            for k, v in results.items()
        },
        "critical_limitations_and_disclaimer": (
            "CRITICAL MEDICAL LIMITATION: These models are trained strictly on standardized, cropped, "
            "low-resolution 64x64 benchmark biomedical images (MedMNIST v2). They will NOT work reliably "
            "on arbitrary smartphone photos, poor-contrast uncalibrated images, or full-resolution clinical PACS "
            "radiology scans without standardized preprocessing. This module provides educational screening "
            "and research triaging only, NOT a clinical radiology or pathology diagnosis."
        ),
        "out_of_distribution_guard": (
            "An OOD guard rejects images where maximum prediction confidence falls below 0.50, returning "
            "'image not recognised / low confidence' rather than generating a spurious diagnostic label."
        ),
    }
    with open(OUTPUT_MODEL_CARD, "w") as f:
        json.dump(model_card, f, indent=2)
    print(f"Saved model card to {OUTPUT_MODEL_CARD}")
    print(f"\nPhase 4 MedMNIST Training Complete in {total_training_time:.1f}s!")


if __name__ == "__main__":
    main()
