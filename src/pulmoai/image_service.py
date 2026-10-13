"""Medical Image Screening Inference Service (MedMNIST 64x64).

Serves fine-tuned ResNet18 models for:
  - chest_xray: PneumoniaMNIST (Normal vs Pneumonia)
  - skin_lesion: DermaMNIST (7 dermatology classes)
  - retina: RetinaMNIST (5 diabetic retinopathy stages)
  - blood_smear: BloodMNIST (8 hematopoietic cell types)
  - chest_multilabel: ChestMNIST (14 thoracic conditions)

Includes an Out-of-Distribution (OOD) confidence guard to reject
non-medical or low-confidence images rather than predicting spurious labels.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger("pulmoai.image_service")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "medmnist"

MODALITY_MAP = {
    "chest_xray": {
        "model_file": "pneumoniamnist_resnet18.pt",
        "display_name": "Chest X-Ray (Pneumonia Screening)",
        "channels": 1,
        "task": "binary-class",
    },
    "skin_lesion": {
        "model_file": "dermamnist_resnet18.pt",
        "display_name": "Dermatoscopy (Skin Lesion)",
        "channels": 3,
        "task": "multi-class",
    },
    "retina": {
        "model_file": "retinamnist_resnet18.pt",
        "display_name": "Retinal Fundus Photography",
        "channels": 3,
        "task": "multi-class",
    },
    "blood_smear": {
        "model_file": "bloodmnist_resnet18.pt",
        "display_name": "Peripheral Blood Smear",
        "channels": 3,
        "task": "multi-class",
    },
    "chest_multilabel": {
        "model_file": "chestmnist_resnet18.pt",
        "display_name": "Chest X-Ray (14 Pathologies)",
        "channels": 1,
        "task": "multi-label, binary-class",
    },
}

IMAGE_DISCLAIMER = (
    "RESEARCH & SCREENING PROTOTYPE ONLY: These models are trained strictly on low-resolution 64x64 "
    "benchmark images (MedMNIST v2). They will NOT work reliably on arbitrary smartphone photos, "
    "uncalibrated ambient lighting, or full-resolution clinical PACS radiology exams without standardized "
    "preprocessing. This output is NOT a medical or radiological diagnosis."
)


def _build_model(n_channels: int, n_classes: int) -> nn.Module:
    """Instantiate ResNet18 architecture with adapted input/output layers."""
    model = models.resnet18(weights=None)
    if n_channels == 1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


class ImageScreeningService:
    """Inference service for MedMNIST image classification with OOD guard."""

    def __init__(self, model_dir: Path | None = None, device: torch.device | None = None):
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self._loaded_models: dict[str, dict[str, Any]] = {}
        logger.info("ImageScreeningService initialized with device: %s", self.device)

    @classmethod
    def load(cls, model_dir: Path | None = None) -> ImageScreeningService:
        return cls(model_dir=model_dir)

    def _get_model_bundle(self, image_type: str) -> dict[str, Any]:
        """Lazy-load and cache the model for the requested image modality."""
        if image_type not in MODALITY_MAP:
            raise ValueError(
                f"Invalid image type '{image_type}'. Supported types: {', '.join(sorted(MODALITY_MAP.keys()))}"
            )

        if image_type in self._loaded_models:
            return self._loaded_models[image_type]

        meta = MODALITY_MAP[image_type]
        ckpt_path = self.model_dir / meta["model_file"]
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint for '{image_type}' not found at {ckpt_path}. "
                "Ensure scripts/train_medmnist_models.py has completed."
            )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        n_ch = checkpoint["n_channels"]
        n_cls = checkpoint["n_classes"]
        model = _build_model(n_ch, n_cls)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()

        bundle = {
            "model": model,
            "labels": checkpoint.get("class_labels", {}),
            "n_channels": n_ch,
            "task": checkpoint.get("task", meta["task"]),
            "metrics": checkpoint.get("metrics", {}),
            "display_name": meta["display_name"],
        }
        self._loaded_models[image_type] = bundle
        return bundle

    def predict(
        self,
        image_bytes: bytes,
        image_type: str,
        ood_threshold: float = 0.50,
    ) -> dict[str, Any]:
        """Classify a medical image with an Out-of-Distribution (OOD) confidence guard."""
        bundle = self._get_model_bundle(image_type)
        model = bundle["model"]
        labels = bundle["labels"]
        n_ch = bundle["n_channels"]
        task = bundle["task"]

        # 1. Preprocess image: load -> convert channels -> resize 64x64 -> normalize
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if n_ch == 1:
                img = img.convert("L")
                mean, std = [0.5], [0.5]
            else:
                img = img.convert("RGB")
                mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            x_tensor = transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Unable to decode or process image: {e}") from e

        # 2. Forward pass
        with torch.no_grad():
            logits = model(x_tensor)

        # 3. Handle task-specific output & OOD Guard
        if task == "multi-label, binary-class":
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            label_probs = {}
            active_conditions = []
            for idx, p in enumerate(probs):
                lbl_name = labels.get(str(idx), labels.get(idx, f"Condition_{idx}"))
                p_val = round(float(p), 4)
                label_probs[lbl_name] = p_val
                if p_val >= 0.50:
                    active_conditions.append(lbl_name)

            return {
                "image_type": image_type,
                "modality_name": bundle["display_name"],
                "recognized": True,
                "status": "success",
                "multilabel": True,
                "active_conditions": active_conditions,
                "probabilities": label_probs,
                "model_metrics": bundle["metrics"],
                "disclaimer": IMAGE_DISCLAIMER,
            }

        # Multi-class and binary
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        max_conf = float(np.max(probs))
        pred_idx = int(np.argmax(probs))

        # Format label dictionary
        class_probabilities = {}
        for idx, p in enumerate(probs):
            lbl_name = labels.get(str(idx), labels.get(idx, f"Class_{idx}"))
            class_probabilities[lbl_name] = round(float(p), 4)

        pred_label = labels.get(str(pred_idx), labels.get(pred_idx, f"Class_{pred_idx}"))

        # Out-of-Distribution Guard: check if max confidence is below threshold
        if max_conf < ood_threshold:
            return {
                "image_type": image_type,
                "modality_name": bundle["display_name"],
                "recognized": False,
                "status": "low_confidence",
                "message": (
                    "Image not recognised / low confidence. The input image does not display the expected "
                    f"pathological features for {bundle['display_name']}."
                ),
                "predicted_class": None,
                "max_confidence": round(max_conf, 4),
                "threshold_applied": ood_threshold,
                "class_probabilities": class_probabilities,
                "model_metrics": bundle["metrics"],
                "disclaimer": IMAGE_DISCLAIMER,
            }

        return {
            "image_type": image_type,
            "modality_name": bundle["display_name"],
            "recognized": True,
            "status": "success",
            "predicted_class": pred_label,
            "confidence": round(max_conf, 4),
            "class_probabilities": class_probabilities,
            "model_metrics": bundle["metrics"],
            "disclaimer": IMAGE_DISCLAIMER,
        }
