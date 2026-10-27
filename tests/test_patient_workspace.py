"""Regression coverage for patient boundaries and conservative longitudinal claims."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import Request
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.storage import UserDataStore
from backend.workspace_storage import WorkspaceRepository
from pulmoai.longitudinal import build_timeline
from pulmoai.report_reader import process_lab_report
from pulmoai.service import PulmoAIService
from pulmoai.sex_models import read_registry


def lab_record(rid="one", date="2025-01-01", value=240, unit="mg/dL", reference="100-200", reviewed=True):
    return {"id": rid, "filename": f"{rid}.txt", "observed_on": date, "review_status": "reviewed" if reviewed else "unreviewed",
            "kind": "lab_report", "result": {"extracted_tests": [{"canonical_name": "total_cholesterol",
            "display_name": "Total cholesterol", "value": value, "unit": unit, "reference_range": reference}]}}


class LongitudinalTests(unittest.TestCase):
    def series(self, records):
        return build_timeline({"alias": "Synthetic case"}, records)["lab_series"]

    def test_dates_sort_and_toward_range_is_not_health_claim(self):
        records = [lab_record("later", "2025-04-01", 190), lab_record()]
        series = self.series(records)[0]
        self.assertEqual(series["direction"], "toward_range")
        self.assertEqual(series["change"], -50)
        self.assertEqual(series["points"][0]["date"], "2025-01-01")
        self.assertNotIn("improved", json.dumps(series))

    def test_away_from_range_and_low_value_recovery(self):
        self.assertEqual(self.series([lab_record(value=150), lab_record("two", "2025-02-01", 250)])[0]["direction"], "away_from_range")
        self.assertEqual(self.series([lab_record(value=50), lab_record("two", "2025-02-01", 100)])[0]["direction"], "toward_range")

    def test_units_and_reference_intervals_do_not_mix(self):
        self.assertEqual(len(self.series([lab_record(), lab_record("two", "2025-02-01", 5, unit="mmol/L")])), 2)
        series = self.series([lab_record(), lab_record("two", "2025-02-01", 180, reference="80-150")])[0]
        self.assertEqual(series["direction"], "numeric_change_only")

    def test_same_day_observations_do_not_imply_sequence(self):
        series = self.series([lab_record(), lab_record("two", "2025-01-01", 180), lab_record("three", "2025-02-01", 170)])[0]
        self.assertEqual(series["direction"], "not_comparable")
        self.assertIsNone(series["change"])

    def test_unreviewed_and_unknown_units_excluded(self):
        result = build_timeline({}, [lab_record(reviewed=False), lab_record("two", unit="")])
        self.assertEqual(result["lab_series"], [])
        self.assertEqual(result["excluded_unreviewed"], 1)

    def test_corrected_values_override_original_without_erasing_it(self):
        record = lab_record()
        record["result"]["reviewed_values"] = {"total_cholesterol": 140}
        point = self.series([record])[0]["points"][0]
        self.assertEqual(point["value"], 140)
        self.assertTrue(point["corrected"])
        self.assertEqual(record["result"]["extracted_tests"][0]["value"], 240)

    def test_image_scores_separate_models_sites_and_rejections(self):
        def image(rid, model="modelA", site="left arm", status="success", date="2025-01-01"):
            return {"id": rid, "filename": f"{rid}.png", "observed_on": date, "review_status": "reviewed", "kind": "medical_image",
                    "series_label": site, "result": {"status": status, "recognized": status == "success", "artifact_id": model,
                    "image_type": "skin_lesion", "class_probabilities": {"nevus": 0.8}}}
        records = [image("a"), image("b", date="2025-02-01"), image("c", model="modelB"), image("d", site="right arm"), image("e", status="low_confidence"), image("f", model=None)]
        result = build_timeline({}, records)
        self.assertEqual(len(result["image_series"]), 3)
        self.assertTrue(all(s["direction"] == "score_change_only" for s in result["image_series"]))
        self.assertEqual(sum(len(s["points"]) for s in result["image_series"]), 4)


class WorkspaceApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = getattr(app.state, "user_data_store", None)
        self.store = UserDataStore(self.tmp.name)
        self.store.initialize()
        self.user = self.store.register("Synthetic owner", "journey@example.test", "strong-password-123")
        self.repo = WorkspaceRepository(self.store)
        app.state.user_data_store = self.store
        self.client = TestClient(app)
        token, _ = self.store.create_session(self.user["id"])
        self.client.cookies.set("pulmoai_session", token)
        self.patient = self.repo.create_patient(self.user["id"], "Synthetic patient", "female")

    def tearDown(self):
        self.client.close()
        app.state.user_data_store = self.previous
        self.tmp.cleanup()

    def batch(self, files=None, dates=None, **kwargs):
        files = files or [("files", ("visit.txt", b"LAB REPORT\nTotal Cholesterol: 240 mg/dL (Reference: 100-200)\n", "text/plain"))]
        return self.client.post(f"/api/patients/{self.patient['id']}/batch", files=files,
            data={"kind": "lab_report", "observations": json.dumps(dates or [{"observed_on": "2025-01-01"}]), **kwargs})

    def test_batch_review_correction_export_and_duplicate(self):
        result = self.batch()
        self.assertEqual(result.status_code, 200, result.text)
        rid = result.json()["outcomes"][0]["record_id"]
        self.assertEqual(self.batch().json()["outcomes"][0]["status"], "duplicate")
        path = f"/api/patients/{self.patient['id']}/timeline"
        self.assertEqual(self.client.get(path).json()["lab_series"], [])
        reviewed = self.client.post(f"/api/uploads/{rid}/review", json={"confirmed": True, "corrections": {"total_cholesterol": 180}})
        self.assertEqual(reviewed.status_code, 204, reviewed.text)
        timeline = self.client.get(path).json()
        self.assertEqual(timeline["lab_series"][0]["points"][0]["value"], 180)
        self.assertEqual(timeline["records"][0]["result"]["extracted_tests"][0]["value"], 240)
        export = self.client.get(f"/api/patients/{self.patient['id']}/export")
        self.assertIn("attachment", export.headers["content-disposition"])
        self.assertEqual(export.headers["cache-control"], "no-store")
        self.assertTrue(export.headers["x-request-id"])
        self.assertIn("record.reviewed", [event["action"] for event in self.client.get("/api/workspace/audit").json()])

    def test_other_owner_cannot_read_link_review_export_or_batch(self):
        other = self.store.register("Other", "other-journey@example.test", "other-password-123")
        patient = self.repo.create_patient(other["id"], "Other patient", "unknown")
        record = self.store.save_record(other["id"], "lab_report", "secret.txt", "text/plain", {"count": 0}, "secret", b"secret")
        self.assertEqual(self.client.get(f"/api/patients/{patient['id']}/timeline").status_code, 404)
        self.assertEqual(self.client.get(f"/api/patients/{patient['id']}/export").status_code, 404)
        self.assertEqual(self.client.post(f"/api/uploads/{record['id']}/review", json={"confirmed": True}).status_code, 404)
        response = self.client.patch(f"/api/uploads/{record['id']}/context", json={"patient_id": self.patient["id"], "observed_on": "2025-01-01"})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.get("/api/workspace/audit").json()[0]["resource_id"], self.patient["id"])
        self.client.cookies.clear()
        self.assertEqual(self.client.get("/api/patients").status_code, 401)

    def test_invalid_metadata_rejects_entire_batch_before_storage(self):
        self.assertEqual(self.batch(dates=[{"observed_on": "2999-01-01"}]).status_code, 422)
        self.assertEqual(self.batch(dates=[{"observed_on": "2025-01-01"}, {"observed_on": "2025-01-02"}]).status_code, 422)
        self.assertEqual(self.repo.records(self.user["id"], self.patient["id"]), [])

    def test_partial_failure_keeps_successful_file(self):
        files = [("files", ("good.txt", b"Total Cholesterol: 240 mg/dL (100-200)", "text/plain")),
                 ("files", ("bad.exe", b"invalid", "application/octet-stream"))]
        response = self.batch(files, [{"observed_on": "2025-01-01"}, {"observed_on": "2025-02-01"}])
        self.assertEqual([x["status"] for x in response.json()["outcomes"]], ["saved", "error"])
        self.assertEqual(len(self.repo.records(self.user["id"], self.patient["id"])), 1)

    def test_review_requires_confirmation_and_finite_values(self):
        rid = self.batch().json()["outcomes"][0]["record_id"]
        self.assertEqual(self.client.post(f"/api/uploads/{rid}/review", json={"confirmed": False}).status_code, 422)
        self.assertEqual(self.client.post(f"/api/uploads/{rid}/review", json={"confirmed": True, "corrections": {"missing": 1}}).status_code, 422)
        self.assertEqual(self.client.post(f"/api/uploads/{rid}/review", json={"confirmed": True, "corrections": {"total_cholesterol": -1}}).status_code, 422)

    def test_changing_assignment_requires_review_again(self):
        rid = self.batch().json()["outcomes"][0]["record_id"]
        self.repo.review(self.user["id"], rid, {})
        self.repo.attach(self.user["id"], rid, self.patient["id"], "2025-02-01")
        self.assertEqual(self.repo.records(self.user["id"], self.patient["id"])[0]["review_status"], "unreviewed")

    def test_migration_is_repeatable_and_preserves_existing_uploads(self):
        record = self.store.save_record(self.user["id"], "lab_report", "old.txt", "text/plain", {"count": 0}, "Legacy", b"legacy")
        self.store.initialize()
        self.store.initialize()
        self.assertIsNotNone(self.store.get_record(self.user["id"], record["id"]))

    def test_untrusted_origin_and_large_request_blocked(self):
        response = self.client.post("/api/patients", json={"alias": "Cross origin"}, headers={"origin": "https://untrusted.example"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.post("/api/patients", content=b"", headers={"content-length": str(70 * 1024 * 1024)}).status_code, 413)


class InferenceSafetyTests(unittest.TestCase):
    def test_cloud_environment_key_never_sends_report(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "not-authorization"}), patch("socket.socket.connect", side_effect=AssertionError("Unexpected network")):
            result = process_lab_report(b"LAB REPORT\nTotal Cholesterol: 210 mg/dL (100-200)", "visit.txt", "text/plain")
        self.assertEqual(result["extraction_method"], "text_input")
        self.assertEqual(result["count"], 1)

    def test_binary_image_is_never_decoded_as_lab_text_when_ocr_unavailable(self):
        image = Image.new("RGB", (256, 256), "white")
        image.putpixel((20, 20), (0, 0, 0))
        buffer = io.BytesIO(); image.save(buffer, format="PNG")
        with patch("pytesseract.image_to_string", side_effect=RuntimeError("Missing local engine")):
            result = process_lab_report(buffer.getvalue(), "scan.png", "image/png")
        self.assertNotEqual(result["extraction_method"], "text_input")
        self.assertEqual(result["count"], 0)

    def test_reported_sex_uses_explicit_pooled_fallback(self):
        service = PulmoAIService.load()
        contents = (Path(__file__).resolve().parents[1] / "samples/scrna_sample_donor.csv").read_bytes()
        male = service.predict_upload(contents, "male")
        female = service.predict_upload(contents, "female")
        self.assertEqual(male["predicted_class"], female["predicted_class"])
        self.assertEqual(male["sex_context"]["model_scope"], "pooled")
        self.assertFalse(female["sex_context"]["sex_specific_available"])
        self.assertNotIn("High-confidence", male["interpretation"])
        with self.assertRaises(ValueError):
            service.predict_upload(contents, "inferred")

    def test_registry_rejects_unverified_model_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "candidate.joblib"; model.write_bytes(b"never deserialize")
            (root / "registry.json").write_text(json.dumps({"models": {"female": {"filename": model.name, "sha256": "bad",
                "evaluation": {"protocol": "leave_one_study_out", "source_studies": 3,
                "class_support": {label: 10 for label in ["Donor", "PF", "COVID19", "COPD"]}}}}}), encoding="utf-8")
            with patch("joblib.load", side_effect=AssertionError("must not load")):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    class Pooled:
                        label_order = ["Donor", "PF", "COVID19", "COPD"]
                    read_registry(Pooled(), root)

    def test_image_model_output_is_never_replaced_with_sample_answer(self):
        import numpy as np
        import torch
        from pulmoai.image_service import ImageScreeningService
        service = ImageScreeningService()
        class Model:
            def __call__(self, x):
                return torch.tensor([[3.0, 0.0]])
        bundle = {"model": Model(), "labels": {"0": "normal", "1": "pneumonia"}, "n_channels": 1,
                  "task": "binary-class", "display_name": "Synthetic test model", "metrics": {"accuracy": 0.61}}
        with patch.object(service, "_get_model_bundle", return_value=bundle), patch.object(service, "_get_modality_gate", return_value=None):
            data = io.BytesIO(); Image.fromarray(np.arange(4096, dtype=np.uint8).reshape(64, 64)).save(data, format="PNG")
            result = service.predict(data.getvalue(), "chest_xray", enforce_modality_gate=False, patient_is_pediatric=True)
            refused = service.predict(data.getvalue(), "chest_xray", enforce_modality_gate=False, patient_is_pediatric=False)
        self.assertEqual(result["predicted_class"], "normal")
        self.assertAlmostEqual(result["class_probabilities"]["normal"], 0.9526, places=4)
        self.assertEqual(result["accuracy"], 0.61)
        self.assertEqual(refused["status"], "rejected_pediatric_confirmation_required")

class AdditionalWorkspaceSafetyTests(unittest.TestCase):
    def test_sex_specific_route_uses_separate_model_when_explicitly_available(self):
        import numpy as np
        class Classifier:
            classes_ = np.array([0, 1])
            def __init__(self, label): self.label = label
            def predict(self, x): return np.array([self.label])
        def model(code):
            return PulmoAIService({"pipeline": Classifier(code), "feature_names": ["g1", "g2"],
                                   "label_order": ["Donor", "PF"], "model_name": "Test model"}, f"test-{code}")
        pooled, female = model(0), model(1)
        pooled._sex_registry = {"female": (female, {"protocol": "test-only"})}
        contents = b"gene,count\ng1,10\ng2,20\n"
        result = pooled.predict_upload(contents, "female")
        self.assertEqual(result["predicted_class"], "PF")
        self.assertEqual(result["artifact_id"], "test-1")
        self.assertEqual(result["sex_context"]["model_scope"], "sex_specific_research")
        self.assertEqual(pooled.predict_upload(contents, "unknown")["predicted_class"], "Donor")

    def test_rate_limit_and_streamed_body_limit(self):
        from fastapi import FastAPI, Request
        from backend.security import ApiSecurityMiddleware
        local = FastAPI()
        local.add_middleware(ApiSecurityMiddleware)
        @local.post("/api/auth/login")
        async def login(): return {"ok": True}
        @local.post("/api/consume")
        async def consume(request: Request):
            await request.body()
            return {"ok": True}
        client = TestClient(local)
        for _ in range(10): self.assertEqual(client.post("/api/auth/login").status_code, 200)
        self.assertEqual(client.post("/api/auth/login").status_code, 429)
        with patch("backend.security.MAX_REQUEST_BYTES", 10):
            response = client.post("/api/consume", content=iter([b"123456", b"123456"]))
        self.assertEqual(response.status_code, 413)

class DeploymentSafetyTests(unittest.TestCase):
    def test_health_does_not_deserialize_failed_integrity_artifact(self):
        previous = getattr(app.state, "artifact_verification", None)
        app.state.artifact_verification = {"all_passed": False, "artifacts": {}, "module_status": {"pulmo_service": False}}
        try:
            with patch("backend.main.PulmoAIService.load", side_effect=AssertionError("Unsafe deserialize")):
                response = TestClient(app).get("/api/health")
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()["model_loaded"])
            self.assertEqual(response.json()["status"], "unavailable")
        finally:
            app.state.artifact_verification = previous

    def test_risk_schema_rejects_nonfinite_input(self):
        from backend.schemas import RiskPredictionRequest
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            RiskPredictionRequest(inputs={"age": float("inf")})

    def test_text_encoding_error_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            process_lab_report(b"\xff\xfe\x00\x00bad report", "scan.txt", "text/plain")
