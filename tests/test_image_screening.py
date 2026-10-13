"""Tests for MedMNIST Image Screening Service and FastAPI Endpoint (Phase 4)."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from pulmoai.image_service import (
    IMAGE_DISCLAIMER,
    MODALITY_MAP,
    ImageScreeningService,
)


class ImageScreeningUnitTest(unittest.TestCase):
    """Unit tests for MedMNIST inference and OOD guard."""

    def setUp(self) -> None:
        self.service = ImageScreeningService.load()

    def test_modality_validation(self) -> None:
        """Verify invalid image_type raises ValueError."""
        dummy_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buf = io.BytesIO()
        dummy_img.save(buf, format="PNG")
        with self.assertRaises(ValueError):
            self.service.predict(buf.getvalue(), "unsupported_type")

    def test_pneumonia_prediction_with_synthetic_image(self) -> None:
        """Verify chest_xray predicts binary label and returns required metadata."""
        # Create a synthetic 64x64 grayscale image
        arr = np.random.randint(50, 200, size=(64, 64), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = self.service.predict(buf.getvalue(), "chest_xray", ood_threshold=0.30)
        self.assertEqual(result["image_type"], "chest_xray")
        self.assertIn("modality_name", result)
        self.assertIn("class_probabilities", result)
        self.assertIn("disclaimer", result)
        self.assertIn("model_metrics", result)

    def test_ood_guard_triggers_on_low_confidence(self) -> None:
        """Verify OOD guard rejects image when max confidence is below threshold."""
        arr = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        # Set threshold to 1.0 so any prediction triggers OOD guard
        result = self.service.predict(buf.getvalue(), "chest_xray", ood_threshold=1.0)
        self.assertFalse(result["recognized"])
        self.assertEqual(result["status"], "low_confidence")
        self.assertIn("not recognised", result["message"].lower())
        self.assertIsNone(result["predicted_class"])


class ImageScreeningAPITest(unittest.TestCase):
    """Integration tests for POST /api/image/predict endpoint."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_api_predict_chest_xray(self) -> None:
        """Test POST /api/image/predict endpoint with PNG image."""
        arr = np.random.randint(60, 180, size=(64, 64), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        files = {"file": ("chest.png", buf.getvalue(), "image/png")}
        data = {"image_type": "chest_xray", "ood_threshold": "0.30"}

        resp = self.client.post("/api/image/predict", files=files, data=data)
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["image_type"], "chest_xray")
        self.assertIn("disclaimer", payload)
        self.assertIn("status", payload)

    def test_api_predict_unsupported_format(self) -> None:
        """Verify 415 Unsupported Media Type for non-image files."""
        files = {"file": ("document.pdf", b"fake pdf", "application/pdf")}
        data = {"image_type": "chest_xray"}
        resp = self.client.post("/api/image/predict", files=files, data=data)
        self.assertEqual(resp.status_code, 415)

    def test_api_predict_invalid_modality(self) -> None:
        """Verify 400 Bad Request for unknown image_type."""
        arr = np.zeros((64, 64), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        files = {"file": ("scan.png", buf.getvalue(), "image/png")}
        data = {"image_type": "mri_brain"}
        resp = self.client.post("/api/image/predict", files=files, data=data)
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
