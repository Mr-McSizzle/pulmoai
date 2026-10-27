"""Checks for source-linked evidence and persistent example-data provenance."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend.main import app
from backend.storage import UserDataStore
from backend.workspace_storage import WorkspaceRepository
from pulmoai.evidence import REPORTS, ROOT, _metric, benchmark_evidence
from pulmoai.provenance import data_origin
from pulmoai.service import PulmoAIService


class EvidenceTests(unittest.TestCase):
    def test_metrics_exactly_match_named_reports(self):
        evidence = benchmark_evidence({"pulmo_service": True, "image_service": False})
        modules = {item["id"]: item for item in evidence["modules"]}
        audit = json.loads((ROOT / "reports" / REPORTS["lung"]).read_text())
        self.assertEqual(modules["lung"]["metrics"][0]["value"], audit["configuration_comparison"]["model_c_dual_rule"]["accuracy"])
        self.assertEqual(modules["lab"]["metrics"][0]["value"], 0.987)
        self.assertEqual(modules["lab"]["metrics"][1]["value"], 0.8398)
        self.assertEqual(modules["lab"]["metrics"][2]["value"], 0.0498)
        self.assertIn("Generated", modules["lab"]["cohort"])
        self.assertFalse(modules["vision"]["artifact_integrity"])
        self.assertEqual(modules["vision"]["metrics"][1]["value"], 0.1903)
        self.assertTrue(all(len(item["source_sha256"]) == 64 for item in modules.values()))

    def test_missing_evidence_does_not_generate_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = benchmark_evidence({}, Path(tmp))
        self.assertTrue(all(item["evidence_status"] == "unavailable" and item["metrics"] == [] for item in result["modules"]))

    def test_invalid_metric_is_rejected(self):
        for value in [1.1, -0.1, float("nan"), float("inf"), True, "0.99"]:
            with self.assertRaises(ValueError): _metric("Invalid", value)

    def test_report_download_is_whitelisted(self):
        client = TestClient(app)
        response = client.get("/api/evidence/reports/lab")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["metadata"]["total_synthetic_documents"], 66)
        self.assertEqual(client.get("/api/evidence/reports/not-a-report").status_code, 404)
        response = client.get("/api/evidence")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_dashboard_uses_current_procedure_and_drops_mismatched_matrix(self):
        result = PulmoAIService.load().dashboard_data()
        self.assertEqual(result["verification_status"], "reported_research_audit")
        self.assertEqual(result["metrics"]["accuracy"], 0.6478)
        self.assertEqual(result["metrics"]["macro_f1"], 0.5349)
        self.assertIsNone(result["per_class_metrics"])
        self.assertIsNone(result["confusion_matrix"])

    def test_dashboard_does_not_substitute_older_scores_when_report_is_missing(self):
        with patch("pulmoai.service.DEFAULT_DUAL_RULE_AUDIT_PATH", Path("does-not-exist.json")):
            result = PulmoAIService.load().dashboard_data()
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["verification_status"], "not_verified")


class ProvenanceTests(unittest.TestCase):
    def test_uploads_are_never_automatically_called_real(self):
        self.assertEqual(data_origin(b"Hospital report", "report.txt"), "user_supplied_unverified")
        self.assertEqual(data_origin(b"GENERATED EXAMPLE DATA", "report.txt"), "generated_example")
        self.assertEqual(data_origin(b"SYNTHETIC DEMO", "old-example.txt"), "generated_example")
        self.assertEqual(data_origin(b"pixels", "synthetic_chest.png"), "generated_example")
        self.assertEqual(data_origin(b"pixels", "example.png", "generated_example"), "generated_example")

    def test_origin_survives_review_and_export(self):
        prior = getattr(app.state, "user_data_store", None)
        with tempfile.TemporaryDirectory() as tmp:
            store = UserDataStore(tmp); store.initialize()
            app.state.user_data_store = store
            try:
                user = store.register("Example QA", "origin@example.test", "test-password-123")
                token, _ = store.create_session(user["id"])
                patient = WorkspaceRepository(store).create_patient(user["id"], "Generated case", "unknown")
                client = TestClient(app); client.cookies.set("pulmoai_session", token)
                response = client.post(f"/api/patients/{patient['id']}/batch", files=[("files", ("example.txt", b"GENERATED EXAMPLE DATA\nTotal Cholesterol: 180 mg/dL (100-200)", "text/plain"))],
                    data={"kind": "lab_report", "observations": json.dumps([{"observed_on": "2025-01-01", "data_origin": "generated_example"}])})
                self.assertEqual(response.status_code, 200, response.text)
                rid = response.json()["outcomes"][0]["record_id"]
                self.assertEqual(client.post(f"/api/uploads/{rid}/review", json={"confirmed": True}).status_code, 204)
                export = client.get(f"/api/patients/{patient['id']}/export").json()
                self.assertEqual(export["records"][0]["result"]["data_origin"], "generated_example")
                self.assertEqual(export["records"][0]["review_status"], "reviewed")
                self.assertEqual(client.get(f"/api/uploads/{rid}").json()["result"]["data_origin"], "generated_example")
                upload = client.post("/api/report/extract", files={"file": ("plain.txt", b"Total Cholesterol: 180 mg/dL (100-200)", "text/plain")})
                self.assertEqual(upload.json()["data_origin"], "user_supplied_unverified")
                declared = client.post("/api/report/extract", files={"file": ("example.txt", b"Total Cholesterol: 180 mg/dL (100-200)", "text/plain")}, data={"data_origin": "generated_example"})
                self.assertEqual(declared.json()["data_origin"], "generated_example")
            finally:
                app.state.user_data_store = prior

class ImageArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.prior = getattr(app.state, "user_data_store", None)
        self.store = UserDataStore(self.tmp.name); self.store.initialize()
        app.state.user_data_store = self.store
        self.user = self.store.register("Archive QA", "archive@example.test", "test-password-123")
        self.patient = WorkspaceRepository(self.store).create_patient(self.user["id"], "Archive case", "unknown")
        token, _ = self.store.create_session(self.user["id"])
        self.client = TestClient(app); self.client.cookies.set("pulmoai_session", token)

    def tearDown(self):
        self.client.close()
        app.state.user_data_store = self.prior
        self.tmp.cleanup()

    def upload(self, mode="archive_only", contents=None):
        import io
        from PIL import Image
        if contents is None:
            output = io.BytesIO(); Image.new("RGB", (64, 64), "gray").save(output, format="PNG")
            contents = output.getvalue()
        return self.client.post(f"/api/patients/{self.patient['id']}/batch", files=[("files", ("example-source.png", contents, "image/png"))],
            data={"kind": "medical_image", "image_type": "chest_xray", "analysis_mode": mode,
                "observations": json.dumps([{"observed_on": "2025-01-01", "series_label": "same view", "data_origin": "generated_example"}])})

    def test_archive_works_without_vision_weights_or_age_assumptions(self):
        with patch("backend.workspace_api.require_image_service", side_effect=AssertionError("Do not load model")):
            response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        rid = response.json()["outcomes"][0]["record_id"]
        result = self.client.get(f"/api/uploads/{rid}").json()["result"]
        self.assertEqual(result["status"], "not_screened")
        self.assertFalse(result["recognized"])
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["artifact_id"])
        self.assertEqual(result["data_origin"], "generated_example")
        self.assertEqual(self.client.get(f"/api/uploads/{rid}/download").status_code, 200)
        self.assertEqual(self.client.post(f"/api/uploads/{rid}/review", json={"confirmed": True}).status_code, 204)
        timeline = self.client.get(f"/api/patients/{self.patient['id']}/timeline").json()
        self.assertEqual(timeline["image_series"], [])
        self.assertEqual(timeline["records"][0]["result"]["status"], "not_screened")

    def test_screening_still_requires_pediatric_confirmation(self):
        response = self.upload("screen")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.store.list_records(self.user["id"]), [])

    def test_archive_does_not_accept_undecodable_images(self):
        response = self.upload(contents=b"not an image")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["outcomes"][0]["status"], "error")
        self.assertEqual(self.store.list_records(self.user["id"]), [])

class ImagePreviewTests(unittest.TestCase):
    def test_tiff_preview_is_authenticated_png_and_original_is_unchanged(self):
        import io
        from PIL import Image
        prior = getattr(app.state, "user_data_store", None)
        with tempfile.TemporaryDirectory() as tmp:
            store = UserDataStore(tmp); store.initialize(); app.state.user_data_store = store
            try:
                user = store.register("Preview QA", "preview@example.test", "test-password-123")
                source = io.BytesIO(); Image.new("RGB", (60, 40), (20, 80, 100)).save(source, format="TIFF")
                record = store.save_record(user["id"], "medical_image", "source.tiff", "image/tiff", {"status": "not_screened"}, "Original", source.getvalue())
                client = TestClient(app)
                self.assertEqual(client.get(f"/api/uploads/{record['id']}/preview").status_code, 401)
                token, _ = store.create_session(user["id"]); client.cookies.set("pulmoai_session", token)
                response = client.get(f"/api/uploads/{record['id']}/preview")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "image/png")
                self.assertEqual(response.headers["cache-control"], "no-store")
                with Image.open(io.BytesIO(response.content)) as image:
                    self.assertEqual(image.size, (60, 40))
                    self.assertEqual(image.getpixel((0, 0)), (20, 80, 100))
                self.assertEqual(client.get(f"/api/uploads/{record['id']}/download").content, source.getvalue())
                other = store.register("Other preview", "other-preview@example.test", "test-password-123")
                token, _ = store.create_session(other["id"]); client.cookies.set("pulmoai_session", token)
                self.assertEqual(client.get(f"/api/uploads/{record['id']}/preview").status_code, 404)
            finally:
                app.state.user_data_store = prior
