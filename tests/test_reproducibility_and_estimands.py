"""Tests for reproducibility, checkpoint verification, empirical OOD benchmarks, and risk estimands."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi.testclient import TestClient
from backend.main import app
from scripts.verify_checkpoints import verify_checkpoints


class ReproducibilityAndVerificationTests(unittest.TestCase):
    """Test checkpoint verification and integrity endpoints."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_checkpoint_manifest_and_sha256_verification(self) -> None:
        """Verify that all critical model files on disk match their SHA-256 integrity hashes."""
        res = verify_checkpoints()
        self.assertTrue(res["all_critical_passed"], f"Critical checkpoints failed verification: {res}")
        self.assertIn("single_cell_lung_atlas", res["checkpoints"])
        self.assertIn("nhanes_chronic_risk", res["checkpoints"])
        self.assertTrue(res["checkpoints"]["single_cell_lung_atlas"]["passed"])
        self.assertTrue(res["checkpoints"]["nhanes_chronic_risk"]["passed"])

    def test_api_integrity_endpoint(self) -> None:
        """Verify GET /api/integrity returns verified status and individual checkpoint records."""
        resp = self.client.get("/api/integrity")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["all_critical_passed"])
        self.assertIn("checkpoints", data)
        self.assertIn("single_cell_lung_atlas", data["checkpoints"])

    def test_api_health_includes_integrity_flag(self) -> None:
        """Verify GET /api/health includes integrity_verified flag."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("integrity_verified", data)
        self.assertTrue(data["integrity_verified"])


class RiskEstimandValidationTests(unittest.TestCase):
    """Verify that all risk engine outputs explicitly specify their estimand without 10-year incidence claims."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_estimands_in_risk_prediction(self) -> None:
        """Verify that condition predictions define estimand_type, label, and time_horizon."""
        payload = {
            "inputs": {
                "age": 55,
                "gender": 1,
                "systolic_bp": 135,
                "diastolic_bp": 85,
                "bmi": 28.0,
                "total_cholesterol": 210,
                "hdl_cholesterol": 45,
                "triglycerides": 160,
                "creatinine": 1.0,
                "smoking_status": 0,
            }
        }
        resp = self.client.post("/api/risk/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("estimand_clarification", data)
        self.assertIn("prevalence", data["estimand_clarification"].lower())
        self.assertNotIn("10-year future incidence", data["estimand_clarification"].lower())

        preds = data["risk_predictions"]
        self.assertIn("diabetes", preds)
        self.assertEqual(preds["diabetes"]["estimand_type"], "current_prevalent_status")
        self.assertEqual(preds["diabetes"]["time_horizon"], "Current cross-sectional status")
        self.assertIn("Current Prevalent Status", preds["diabetes"]["estimand_label"])

        for cond_key in ["hypertension", "high_cholesterol", "heart_disease", "stroke", "copd"]:
            self.assertIn(cond_key, preds)
            self.assertEqual(preds[cond_key]["estimand_type"], "survey_reported_history")
            self.assertIn("Survey-Reported", preds[cond_key]["estimand_label"])


class OODRejectionBenchmarkTests(unittest.TestCase):
    """Verify empirical cross-modality False Accept Rate reporting."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_image_predict_returns_empirical_far_and_methodology(self) -> None:
        """Verify image screening response includes empirical False Accept Rate and methodology statement."""
        import numpy as np
        from PIL import Image
        import io

        arr = np.random.randint(60, 180, size=(64, 64), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        files = {"file": ("test.png", buf.getvalue(), "image/png")}
        data = {"image_type": "chest_xray", "ood_threshold": "0.50"}

        resp = self.client.post("/api/image/predict", files=files, data=data)
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()

        self.assertIn("empirical_false_accept_rate", payload)
        self.assertIsNotNone(payload["empirical_false_accept_rate"])
        self.assertGreaterEqual(payload["empirical_false_accept_rate"], 0.0)
        self.assertLessEqual(payload["empirical_false_accept_rate"], 1.0)

        self.assertIn("ood_benchmark", payload)
        self.assertIn("tested_against_cross_modalities", payload["ood_benchmark"])
        self.assertIn("ood_methodology_statement", payload)
        self.assertIn("overconfident", payload["ood_methodology_statement"].lower())


class CanonicalEvaluationRegistryTests(unittest.TestCase):
    """Verify evaluation registry consistency across files."""

    def test_registry_file_structure(self) -> None:
        """Verify evaluation_registry.json contains all required modules and metrics."""
        reg_path = PROJECT_ROOT / "reports" / "evaluation_registry.json"
        self.assertTrue(reg_path.exists(), "evaluation_registry.json does not exist")

        with open(reg_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        self.assertIn("modules", registry)
        self.assertIn("single_cell_lung_atlas", registry["modules"])
        self.assertIn("nhanes_chronic_risk", registry["modules"])
        self.assertIn("medmnist_image_screening", registry["modules"])
        self.assertIn("lab_report_ocr", registry["modules"])

        # Check NHANES estimands in registry
        nhanes = registry["modules"]["nhanes_chronic_risk"]
        self.assertIn("estimand_clarification", nhanes)
        self.assertIn("current_prevalent_status", nhanes["targets"]["diabetes"]["estimand_type"])

        # Check OOD benchmark in registry
        medmnist = registry["modules"]["medmnist_image_screening"]
        self.assertIn("ood_rejection_benchmark", medmnist)
        self.assertIn("pneumoniamnist", medmnist["ood_rejection_benchmark"])


if __name__ == "__main__":
    unittest.main()
