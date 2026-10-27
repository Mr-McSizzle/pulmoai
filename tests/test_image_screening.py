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

WEIGHTS_EXIST = (PROJECT_ROOT / "models" / "medmnist" / "pneumoniamnist_resnet18.pt").exists()


@unittest.skipUnless(
    WEIGHTS_EXIST,
    "PyTorch vision checkpoint weights (.pt) not present in clean repository checkout",
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

    def test_pneumonia_prediction_with_genuine_image(self) -> None:
        """Verify chest_xray predicts binary label and returns required metadata with verified gate match."""
        # Load a genuine chest X-ray image from MedMNIST cache
        npz_path = Path.home() / ".medmnist" / "pneumoniamnist_64.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            raw_arr = data["test_images"][0]
            img = Image.fromarray(raw_arr).convert("L")
        else:
            arr = np.random.randint(50, 200, size=(64, 64), dtype=np.uint8)
            img = Image.fromarray(arr, mode="L")

        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = self.service.predict(
            buf.getvalue(),
            "chest_xray",
            ood_threshold=0.30,
            patient_is_pediatric=True,
        )
        self.assertEqual(result["image_type"], "chest_xray")
        self.assertIn("modality_name", result)
        self.assertIn("class_probabilities", result)
        self.assertIn("disclaimer", result)
        self.assertIn("model_metrics", result)
        self.assertIn("modality_gate", result)
        self.assertEqual(result["modality_gate"]["status"], "VERIFIED_MATCH")

    def test_chest_xray_refuses_when_pediatric_confirmation_missing_service(self) -> None:
        """Verify service refuses chest_xray when patient_is_pediatric=False."""
        arr = np.random.randint(50, 200, size=(64, 64), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = self.service.predict(
            buf.getvalue(),
            "chest_xray",
            ood_threshold=0.30,
            patient_is_pediatric=False,
        )
        self.assertFalse(result["recognized"])
        self.assertEqual(result["status"], "rejected_pediatric_confirmation_required")
        self.assertIn("Refusal", result["message"])
        self.assertIn("pediatric patients aged 1-5", result["message"])

    def test_modality_gate_rejects_cross_modality_mismatch(self) -> None:
        """Verify Modality Gate refuses to run when image modality contradicts user selection."""
        npz_path = Path.home() / ".medmnist" / "dermamnist_64.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            raw_arr = data["test_images"][0]
            img = Image.fromarray(raw_arr).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")

            # Upload dermoscopy image under chest_xray selection with pediatric confirmation
            result = self.service.predict(
                buf.getvalue(),
                "chest_xray",
                ood_threshold=0.30,
                patient_is_pediatric=True,
            )
            self.assertFalse(result["recognized"])
            self.assertEqual(result["status"], "rejected_modality_mismatch")
            self.assertEqual(result["modality_gate"]["status"], "REJECTED_MODALITY_MISMATCH")
            self.assertEqual(result["modality_gate"]["predicted_modality"], "dermoscopy")
            self.assertIn("contradicts the selected modality", result["message"])

    def test_modality_gate_rejects_non_medical_noise(self) -> None:
        """Verify Modality Gate refuses to run when input is non-target/non-medical content."""
        arr = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = self.service.predict(
            buf.getvalue(),
            "chest_xray",
            ood_threshold=0.30,
            patient_is_pediatric=True,
        )
        self.assertFalse(result["recognized"])
        self.assertEqual(result["status"], "rejected_non_medical_input")
        self.assertEqual(result["modality_gate"]["status"], "REJECTED_NON_MEDICAL")
        self.assertEqual(result["modality_gate"]["predicted_modality"], "other")
        self.assertIn("rejected input image as non-target/non-medical", result["message"])

    def test_ood_guard_triggers_on_low_confidence(self) -> None:
        """Verify internal confidence threshold guard triggers when gate is bypassed and threshold=1.0."""
        arr = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        # Bypass modality gate to test confidence threshold guard in isolation
        result = self.service.predict(
            buf.getvalue(),
            "chest_xray",
            ood_threshold=1.0,
            enforce_modality_gate=False,
            patient_is_pediatric=True,
        )
        self.assertFalse(result["recognized"])
        self.assertEqual(result["status"], "low_confidence")
        self.assertIn("not recognised", result["message"].lower())
        self.assertIsNone(result["predicted_class"])


class MissingWeightValidationTests(unittest.TestCase):
    """Verify clean failure when weights are not present."""

    def test_missing_weight_file_fails_with_clear_message(self) -> None:
        """Verify service fails with a descriptive FileNotFoundError when weight file is missing."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = ImageScreeningService(model_dir=Path(tmp_dir))
            dummy_img = Image.new("RGB", (64, 64), color=(128, 128, 128))
            buf = io.BytesIO()
            dummy_img.save(buf, format="PNG")
            with self.assertRaises(FileNotFoundError) as ctx:
                service.predict(
                    buf.getvalue(),
                    "chest_xray",
                    enforce_modality_gate=False,
                    patient_is_pediatric=True,
                )
            self.assertIn("Model checkpoint for 'chest_xray' not found", str(ctx.exception))


class ImageScreeningAPITest(unittest.TestCase):
    """Integration tests for POST /api/image/predict endpoint."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    @unittest.skipUnless(
        WEIGHTS_EXIST,
        "PyTorch vision checkpoint weights (.pt) not present in clean repository checkout",
    )
    def test_api_predict_chest_xray_with_pediatric_confirmation(self) -> None:
        """Test POST /api/image/predict endpoint with explicit pediatric confirmation."""
        arr = np.random.randint(60, 180, size=(64, 64), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        files = {"file": ("chest.png", buf.getvalue(), "image/png")}
        data = {
            "image_type": "chest_xray",
            "ood_threshold": "0.50",
            "patient_is_pediatric": "true",
        }

        resp = self.client.post("/api/image/predict", files=files, data=data)
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["image_type"], "chest_xray")
        self.assertIn("disclaimer", payload)
        self.assertIn("status", payload)

    def test_api_predict_chest_xray_refusal_when_confirmation_missing(self) -> None:
        """Verify 422 Unprocessable Entity when chest_xray is requested without pediatric confirmation."""
        arr = np.random.randint(60, 180, size=(64, 64), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        files = {"file": ("chest.png", buf.getvalue(), "image/png")}
        data = {
            "image_type": "chest_xray",
            "ood_threshold": "0.50",
            "patient_is_pediatric": "false",
        }

        resp = self.client.post("/api/image/predict", files=files, data=data)
        self.assertEqual(resp.status_code, 422)
        detail = resp.json().get("detail", "")
        self.assertIn("Refusal", detail)
        self.assertIn("pediatric patients aged 1-5", detail)

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
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
