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

WEIGHTS_EXIST = (PROJECT_ROOT / "models" / "medmnist" / "pneumoniamnist_resnet18.pt").exists()


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

    @unittest.skipUnless(
        WEIGHTS_EXIST,
        "PyTorch vision checkpoint weights (.pt) not present in clean repository checkout",
    )
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


class FutureRiskEstimandTests(unittest.TestCase):
    """Verify that prospective future risk predictions return correct time horizons, estimands, and published benchmarks."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_future_risk_endpoints_and_published_scores(self) -> None:
        payload = {
            "inputs": {
                "age": 58,
                "sex_male": 1,
                "race_black": 0,
                "bmi": 28.5,
                "waist_cm": 98,
                "systolic_bp": 138,
                "diastolic_bp": 84,
                "total_cholesterol": 220,
                "hdl_cholesterol": 45,
                "current_smoker": 1,
                "bp_medication": 1,
                "target_diabetes": 1,
                "physically_active": 0,
                "family_history_diabetes": 1,
            }
        }
        resp = self.client.post("/api/risk/future", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Check estimand distinction
        self.assertIn("estimand_clarification", data)
        self.assertIn("future mortality risk", data["estimand_clarification"].lower())
        self.assertIn("5-year", data["estimand_clarification"])
        self.assertIn("10-year", data["estimand_clarification"])

        # Check future mortality endpoints
        endpoints = data["future_endpoints"]
        self.assertIn("all_cause_mortality", endpoints)
        self.assertIn("cardiovascular_mortality", endpoints)

        all_cause = endpoints["all_cause_mortality"]
        self.assertEqual(all_cause["estimand_type"], "prospective_incidence")
        self.assertIn("5_year", all_cause["time_horizons"])
        self.assertIn("10_year", all_cause["time_horizons"])
        self.assertGreater(all_cause["time_horizons"]["10_year"]["risk_probability"], all_cause["time_horizons"]["5_year"]["risk_probability"])

        cvd = endpoints["cardiovascular_mortality"]
        self.assertEqual(cvd["estimand_type"], "prospective_incidence")
        self.assertIn("5_year", cvd["time_horizons"])
        self.assertIn("10_year", cvd["time_horizons"])

        # Check published scores
        scores = data["published_scores"]
        self.assertIn("pooled_cohort_equations_pce", scores)
        self.assertIn("indian_diabetes_risk_score_idrs", scores)

        pce = scores["pooled_cohort_equations_pce"]
        self.assertTrue(pce["eligible"])
        self.assertGreater(pce["estimated_10yr_ascvd_risk_pct"], 0.0)

        idrs = scores["indian_diabetes_risk_score_idrs"]
        self.assertEqual(idrs["score"], 80)
        self.assertEqual(idrs["risk_tier"], "High")

    def test_pce_unit_test_worked_examples(self) -> None:
        """Verify PCE calculation matches 2013 ACC/AHA published worked examples."""
        from pulmoai.future_risk_service import compute_pce_10yr
        # 55yo, TC 210, HDL 50, SBP 140 treated, non-smoker, non-diabetic
        w_f = compute_pce_10yr(False, False, False, True, False, 55, 140, 210, 50)
        b_f = compute_pce_10yr(False, True, False, True, False, 55, 140, 210, 50)
        w_m = compute_pce_10yr(True, False, False, True, False, 55, 140, 210, 50)
        b_m = compute_pce_10yr(True, True, False, True, False, 55, 140, 210, 50)
        self.assertEqual(round(w_f, 1), 3.7)
        self.assertEqual(round(b_f, 1), 7.6)
        self.assertEqual(round(w_m, 1), 8.1)
        self.assertEqual(round(b_m, 1), 13.0)

    def test_idrs_unit_test_worked_examples(self) -> None:
        """Verify IDRS scoring matches Mohan et al. 2005 published worked examples."""
        from pulmoai.future_risk_service import compute_idrs
        tc1 = compute_idrs(52, 102, True, 2, 1)
        self.assertEqual(tc1["score"], 90)
        self.assertEqual(tc1["tier"], "High")

        tc2 = compute_idrs(28, 75, False, 0, 0)
        self.assertEqual(tc2["score"], 0)
        self.assertEqual(tc2["tier"], "Low")

        tc3 = compute_idrs(42, 85, False, 1, 2)
        self.assertEqual(tc3["score"], 70)
        self.assertEqual(tc3["tier"], "High")

    def test_glycemic_features_excluded_from_diabetes_model(self) -> None:
        """Verify glycemic features are excluded from diabetes model inputs to prevent diagnostic leakage."""
        import joblib
        bundle_path = PROJECT_ROOT / "models" / "nhanes_risk_models.joblib"
        self.assertTrue(bundle_path.exists())
        bundle = joblib.load(bundle_path)
        self.assertIn("diabetes", bundle)
        diabetes_feats = bundle["diabetes"]["feature_names"]
        glycemic_markers = ["fasting_glucose", "hba1c", "LBXGLU", "LBXGH"]
        for marker in glycemic_markers:
            self.assertNotIn(marker, diabetes_feats, f"Glycemic marker {marker} found in diabetes features!")

    def test_non_hispanic_asian_subgroup_reproducibility(self) -> None:
        """Verify Non-Hispanic Asian subgroup mortality metrics in audit report match evaluated holdout."""
        rep_path = PROJECT_ROOT / "reports" / "audit_future_risk_v3.json"
        self.assertTrue(rep_path.exists())
        with open(rep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        asian = data["subgroup_metrics_with_bootstrap_95ci_e4"]["non_hispanic_asian"]
        self.assertEqual(asian["n_subjects"], 1529)
        self.assertEqual(asian["deaths_observed"], 53)
        self.assertAlmostEqual(asian["c_index"], 0.8618, places=4)
        self.assertAlmostEqual(asian["auroc_5yr"], 0.8834, places=4)
        self.assertEqual(asian["c_index_95ci"], [0.8032, 0.914])
        self.assertEqual(asian["auroc_5yr_95ci"], [0.8146, 0.9465])


if __name__ == "__main__":
    unittest.main()


