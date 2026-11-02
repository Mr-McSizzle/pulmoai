"""A real-model presentation case, repeatability, source packs and owner boundaries."""
from __future__ import annotations
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app
from backend.storage import UserDataStore
from backend.workspace_storage import WorkspaceRepository
from pulmoai.service import PulmoAIService
from pulmoai.risk_service import HealthRiskService
from pulmoai.future_risk_service import FutureRiskService

ROOT = Path(__file__).resolve().parents[1]

class PresentationCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.services = {"rna":PulmoAIService.load(), "risk":HealthRiskService.load(), "future":FutureRiskService.load()}
        cls.verification = {"module_status":{"image_service":False}, "artifacts":{name:{"actual_sha256":hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}
            for name in ("models/nhanes_risk_models.joblib", "models/future_risk_models.joblib")}}

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.previous=getattr(app.state,"user_data_store",None)
        self.store=UserDataStore(self.tmp.name);self.store.initialize();app.state.user_data_store=self.store
        self.repo=WorkspaceRepository(self.store)
        self.user=self.store.register("Presentation QA","presentation@example.test","test-password-123")
        self.client=TestClient(app)
        token,_=self.store.create_session(self.user["id"]);self.client.cookies.set("pulmoai_session",token)
        self.optional=patch("backend.example_cases._optional_services",return_value=(self.services,[]));self.optional.start()
        self.verifier=patch("backend.main.get_artifact_verification",return_value=self.verification);self.verifier.start()

    def tearDown(self):
        self.optional.stop();self.verifier.stop();self.client.close();app.state.user_data_store=self.previous;self.tmp.cleanup()

    def prepare(self):
        response=self.client.post("/api/workspace/presentation-case")
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

    def test_complete_case_has_all_modules_dates_and_both_lab_directions(self):
        case=self.prepare();pid=case["patient"]["id"]
        self.assertFalse(case["reused"])
        timeline=self.client.get(f"/api/patients/{pid}/timeline").json()
        self.assertEqual(timeline["assessment"]["coverage"],{"scrna_csv":4,"lab_report":5,"risk_assessment":4,"future_risk_assessment":4,"medical_image":4})
        self.assertEqual(len(timeline["records"]),21)
        self.assertEqual(timeline["assessment"]["generated_example_count"],21)
        self.assertEqual(timeline["assessment"]["unreviewed_count"],1)
        series={s["id"].split(":")[0]:s for s in timeline["lab_series"]}
        self.assertEqual([p["value"] for p in series["total_cholesterol"]["points"]],[262,236,210,184])
        self.assertEqual(series["total_cholesterol"]["direction"],"toward_range")
        self.assertEqual(series["hemoglobin"]["direction"],"toward_range")
        self.assertEqual(series["serum_creatinine"]["direction"],"away_from_range")
        self.assertEqual(len({r["observed_on"] for r in timeline["records"]}),5)
        self.assertTrue(all(len(s["points"])==4 for s in timeline["risk_series"]))
        self.assertEqual(len(timeline["risk_series"]),10)
        self.assertTrue(any(abs(s["change"] or 0)>0.01 for s in timeline["risk_series"]))
        self.assertEqual(len(timeline["rna_history"]),4)
        self.assertEqual(timeline["image_series"],[])
        images=[r for r in timeline["records"] if r["kind"]=="medical_image"]
        self.assertTrue(all(r["result"]["status"]=="not_screened" and r["result"]["predicted_class"] is None for r in images))
        self.assertTrue(all(r["result"]["data_origin"]=="generated_example" for r in timeline["records"]))
        self.assertTrue(all(r["result"].get("example_validation")=="generated_fixture_check" for r in timeline["records"] if r["review_status"]=="reviewed"))
        self.assertTrue(any("tesseract" in note.lower() for note in case["example_case"]["notices"]) or __import__("shutil").which("tesseract"))

    def test_reopening_preserves_reviews_and_does_not_duplicate(self):
        case=self.prepare();pid=case["patient"]["id"];rid=case["example_case"]["practice_record_id"]
        response=self.client.post(f"/api/uploads/{rid}/review",json={"confirmed":True,"confirmed_tests":["total_cholesterol"],"corrections":{"total_cholesterol":177}})
        self.assertEqual(response.status_code,204,response.text)
        second=self.prepare();self.assertTrue(second["reused"]);self.assertEqual(second["patient"]["id"],pid)
        records=self.repo.records(self.user["id"],pid);self.assertEqual(len(records),21)
        record=next(r for r in records if r["id"]==rid)
        self.assertEqual(record["review_status"],"partially_reviewed")
        self.assertEqual(record["result"]["reviewed_values"]["total_cholesterol"],177)
        self.assertEqual(record["result"]["example_validation"],"user_reviewed")

    def test_source_pack_contains_dated_originals_and_actual_inputs(self):
        case=self.prepare();pid=case["patient"]["id"]
        response=self.client.get(f"/api/patients/{pid}/example-pack")
        self.assertEqual(response.status_code,200,response.text[:100])
        self.assertEqual(response.headers["content-type"],"application/zip")
        self.assertEqual(response.headers["cache-control"],"no-store")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertIsNone(archive.testzip());names=archive.namelist()
            self.assertEqual(len([name for name in names if name.endswith('.txt')]),5)
            self.assertEqual(len([name for name in names if name.endswith('.csv')]),4)
            self.assertEqual(len([name for name in names if name.endswith('.png')]),4)
            manifest=json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["visit_dates"],case["example_case"]["visit_dates"])
            risk=next(r for r in json.loads(archive.read("timeline.json"))["records"] if r["kind"]=="risk_assessment")
            self.assertEqual(risk["result"]["inputs"]["age"],58)
            self.assertIn("source_record_id",risk["result"])
            self.assertIn("generated",archive.read("README.md").decode().lower())

    def test_source_pack_excludes_later_non_example_records(self):
        case=self.prepare();pid=case["patient"]["id"]
        record=self.store.save_record(self.user["id"],"lab_report","private-manual.txt","text/plain",{},"Manual",b"private")
        self.repo.attach(self.user["id"],record["id"],pid,"2025-01-01")
        response=self.client.get(f"/api/patients/{pid}/example-pack")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertNotIn("private-manual.txt",str(archive.namelist()))
            self.assertNotIn("private-manual.txt",archive.read("timeline.json").decode())

    def test_pack_and_timeline_require_the_owner(self):
        case=self.prepare();pid=case["patient"]["id"]
        other=self.store.register("Other","other-presentation@example.test","test-password-123")
        token,_=self.store.create_session(other["id"]);self.client.cookies.set("pulmoai_session",token)
        self.assertEqual(self.client.get(f"/api/patients/{pid}/example-pack").status_code,404)
        self.assertEqual(self.client.get(f"/api/patients/{pid}/timeline").status_code,404)
        other_case=self.prepare();self.assertNotEqual(other_case["patient"]["id"],pid)
        self.client.cookies.clear();self.assertEqual(self.client.post("/api/workspace/presentation-case").status_code,401)

    def test_missing_models_create_sources_and_explicit_gaps_without_predictions(self):
        with patch("backend.example_cases._optional_services",return_value=({},["Numeric services unavailable; no substitute results."])):
            case=self.prepare()
        timeline=self.client.get(f"/api/patients/{case['patient']['id']}/timeline").json()
        self.assertEqual(len(timeline["records"]),9)
        self.assertEqual(timeline["risk_series"],[])
        self.assertEqual(timeline["rna_history"],[])
        self.assertEqual(timeline["image_series"],[])
        self.assertIn("Numeric services unavailable",timeline["example_case"]["notices"][0])

    def test_generation_failure_removes_only_new_case_and_files(self):
        existing=self.repo.create_patient(self.user["id"],"Existing patient","unknown")
        with patch("backend.example_cases.illustration",side_effect=ValueError("Fixture failed")):
            response=self.client.post("/api/workspace/presentation-case")
        self.assertEqual(response.status_code,422)
        self.assertEqual(self.store.list_records(self.user["id"]),[])
        self.assertEqual([p["id"] for p in self.repo.patients(self.user["id"])],[existing["id"]])
        self.assertEqual([p for p in self.store.files_dir.rglob('*') if p.is_file()],[])
        with self.store._connection() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM example_cases").fetchone()[0],0)

if __name__=="__main__":unittest.main()
