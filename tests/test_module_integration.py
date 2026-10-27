"""Patient-context integration, OCR lineage and multimodule timeline regressions."""
from __future__ import annotations
import io
import tempfile
import unittest
from unittest.mock import Mock, patch
from PIL import Image
from fastapi.testclient import TestClient
from backend.main import app
from backend.storage import UserDataStore
from backend.workspace_storage import WorkspaceRepository
from backend.analysis_records import normalize_risk_inputs
from pulmoai.longitudinal import build_timeline

RNA = dict(predicted_class="Donor", model_name="fixture", artifact_id="rna-v1", input_profile_count=1,
           input_gene_count=3, probabilities_available=False, interpretation="Fixture research classification.")
RISK = dict(risk_predictions={"diabetes": dict(target_id="diabetes", display_name="Diabetes", description="Fixture",
    estimand_label="Reported status", estimand_description="Fixture", time_horizon="Existing status",
    calibrated_risk=0.7, risk_category="High")}, features_provided_count=3, estimand_clarification="Fixture", disclaimer="Fixture")
FUTURE = dict(future_endpoints={"mortality": dict(display_name="All-cause mortality", estimand_label="Future mortality",
    estimand_description="Fixture", time_horizons={"5_year":dict(risk_probability=0.1, risk_percent=10),
    "10_year":dict(risk_probability=0.2, risk_percent=20)}, risk_tier_5yr="High", tier_cutoffs_5yr={}, benchmark_metrics={})},
    published_scores={}, features_provided_count=3, estimand_clarification="Fixture", disclaimer="Fixture")
IMAGE = dict(image_type="skin_lesion", modality_name="Dermatoscopy", recognized=True, status="success",
             predicted_class="nevus", class_probabilities={"nevus":0.8}, disclaimer="Fixture")
LAB = b"GENERATED EXAMPLE DATA\nAge: 56\nTotal Cholesterol: 245 mg/dL (100-200)\nHDL Cholesterol: 38 mg/dL (40-80)"

class ModuleIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = getattr(app.state, "user_data_store", None)
        self.store = UserDataStore(self.tmp.name); self.store.initialize()
        app.state.user_data_store = self.store
        self.user = self.store.register("Integration QA", "modules@example.test", "test-password-123")
        self.repo = WorkspaceRepository(self.store)
        self.patient = self.repo.create_patient(self.user["id"], "Generated integration case", "female")
        self.context = {"patient_id":self.patient["id"], "observed_on":"2025-01-01"}
        self.client = TestClient(app)
        token, _ = self.store.create_session(self.user["id"]); self.client.cookies.set("pulmoai_session", token)
        self.rna, self.risk, self.future, self.image = [Mock() for _ in range(4)]
        self.rna.predict_upload.return_value = RNA
        self.risk.predict_risk.return_value = RISK
        self.future.predict_future_risk.return_value = FUTURE
        self.image.predict.return_value = IMAGE
        self.patches = [patch("backend.main.require_service", return_value=self.rna),
            patch("backend.main.require_risk_service", return_value=self.risk),
            patch("backend.main.require_future_risk_service", return_value=self.future),
            patch("backend.main.require_image_service", return_value=self.image),
            patch("backend.main.get_artifact_verification", return_value={"artifacts":{
                "models/nhanes_risk_models.joblib":{"actual_sha256":"risk-v1"},
                "models/future_risk_models.joblib":{"actual_sha256":"future-v1"}}}),
            patch("backend.workspace_api.image_artifact_id", return_value="image-v1")]
        for patcher in self.patches: patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches): patcher.stop()
        self.client.close(); app.state.user_data_store = self.previous; self.tmp.cleanup()

    def extract(self):
        response = self.client.post("/api/report/extract", data=self.context, files={"file":("visit.txt",LAB,"text/plain")})
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

    def review(self, rid, names=None, corrections=None):
        body={"confirmed":True,"corrections":corrections or {}}
        if names is not None: body["confirmed_tests"]=names
        return self.client.post(f"/api/uploads/{rid}/review",json=body)

    def timeline(self):
        return self.client.get(f"/api/patients/{self.patient['id']}/timeline").json()

    def test_every_module_automatically_links_and_contributes_to_assessment(self):
        lab=self.extract()
        self.assertEqual(self.review(lab["upload_id"], ["total_cholesterol"], {"total_cholesterol":190}).status_code,204)
        rna=self.client.post("/api/predict",data={**self.context,"reported_sex":"female"},files={"file":("counts.csv",b"gene,count\nA,1","text/csv")})
        self.assertEqual(rna.status_code,200,rna.text)
        payload={"context":self.context,"source_record_id":lab["upload_id"],"inputs":{"age":56,"gender":2,"total_cholesterol":190}}
        for route in ("/api/risk/predict","/api/risk/future"):
            response=self.client.post(route,json=payload)
            self.assertEqual(response.status_code,200,response.text)
            self.assertEqual(response.json()["data_origin"],"generated_example")
            self.assertEqual(response.json()["source_record_id"],lab["upload_id"])
            self.assertEqual(response.json()["patient_id"],self.patient["id"])
        buf=io.BytesIO(); Image.new("RGB",(32,32),"white").save(buf,format="PNG")
        image=self.client.post("/api/image/predict",data={**self.context,"image_type":"skin_lesion","series_label":"left forearm"},
            files={"file":("skin.png",buf.getvalue(),"image/png")})
        self.assertEqual(image.status_code,200,image.text)
        timeline=self.timeline(); assessment=timeline["assessment"]
        self.assertTrue(all(count==1 for count in assessment["coverage"].values()))
        self.assertEqual(len(timeline["records"]),5)
        self.assertEqual(assessment["labs"][0]["value"],190)
        self.assertEqual(len(assessment["chronic_risks"]),1)
        self.assertEqual(len(assessment["future_risks"]),2)
        self.assertEqual(assessment["rna_findings"][0]["predicted_class"],"Donor")
        self.assertEqual(assessment["image_findings"][0]["series_label"],"left forearm")
        self.assertEqual(assessment["generated_example_count"],3)
        self.assertEqual(assessment["status"],"review_required")
        for record in timeline["records"]:
            self.assertEqual(record["observed_on"],self.context["observed_on"])
            self.assertEqual(record["result"]["analysis_context"]["patient_id"],self.patient["id"])
        self.assertEqual(self.risk.predict_risk.call_args.args[0]["sex_male"],0)
        self.assertIn("not a medical diagnosis",assessment["interpretation"])

    def test_partial_review_does_not_promote_other_extracted_values(self):
        lab=self.extract(); rid=lab["upload_id"]
        self.assertEqual(self.review(rid,["total_cholesterol"],{"total_cholesterol":180}).status_code,204)
        timeline=self.timeline()
        self.assertEqual(timeline["records"][0]["review_status"],"partially_reviewed")
        self.assertEqual([s["name"] for s in timeline["lab_series"]],["Total Cholesterol"])
        self.assertEqual(timeline["records"][0]["result"]["reviewed_tests"],["total_cholesterol"])
        self.assertEqual(self.review(rid,["hdl_cholesterol"],{"hdl_cholesterol":55}).status_code,204)
        self.assertEqual(len(self.timeline()["lab_series"]),2)
        self.assertEqual(self.review(rid).status_code,204)
        self.assertEqual(self.timeline()["records"][0]["review_status"],"reviewed")

    def test_review_rejects_unselected_corrections_and_empty_confirmations(self):
        rid=self.extract()["upload_id"]
        self.assertEqual(self.review(rid,[]).status_code,422)
        self.assertEqual(self.review(rid,["total_cholesterol"],{"hdl_cholesterol":55}).status_code,422)
        self.assertEqual(self.review(rid,["total_cholesterol","total_cholesterol"]).status_code,422)
        self.assertEqual(self.timeline()["records"][0]["review_status"],"unreviewed")

    def test_reassignment_requires_fresh_field_confirmations(self):
        rid=self.extract()["upload_id"]; self.review(rid)
        self.repo.attach(self.user["id"],rid,self.patient["id"],"2025-02-01")
        self.review(rid,["total_cholesterol"])
        result=self.timeline()
        self.assertEqual(len(result["lab_series"]),1)
        self.assertEqual(result["records"][0]["result"]["reviewed_tests"],["total_cholesterol"])

    def test_source_must_be_owned_reviewed_and_for_same_patient(self):
        rid=self.extract()["upload_id"]
        payload={"context":self.context,"source_record_id":rid,"inputs":{"age":56,"bmi":25}}
        self.assertEqual(self.client.post("/api/risk/predict",json=payload).status_code,422)
        self.review(rid)
        other=self.repo.create_patient(self.user["id"],"Other generated case","unknown")
        payload["context"]={**self.context,"patient_id":other["id"]}
        self.assertEqual(self.client.post("/api/risk/future",json=payload).status_code,409)
        payload["context"]=self.context; payload["source_record_id"]="not-owned"
        self.assertEqual(self.client.post("/api/risk/predict",json=payload).status_code,404)
        self.risk.predict_risk.assert_not_called(); self.future.predict_future_risk.assert_not_called()

    def test_invalid_patient_or_date_rejected_before_inference_or_storage(self):
        other=self.store.register("Other","other-module@example.test","test-password-123")
        patient=self.repo.create_patient(other["id"],"Other","unknown")
        for context, code in (({**self.context,"patient_id":patient["id"]},404),({**self.context,"observed_on":"2999-01-01"},422)):
            response=self.client.post("/api/risk/predict",json={"context":context,"inputs":{"age":56}})
            self.assertEqual(response.status_code,code,response.text)
            response=self.client.post("/api/predict",data=context,files={"file":("counts.csv",b"A,1","text/csv")})
            self.assertEqual(response.status_code,code,response.text)
        self.rna.predict_upload.assert_not_called(); self.risk.predict_risk.assert_not_called()
        self.assertEqual(self.store.list_records(self.user["id"]),[])

    def test_repeated_direct_analysis_saves_history_without_batch_dedup_collision(self):
        first,second=self.extract(),self.extract()
        self.assertNotEqual(first["upload_id"],second["upload_id"])
        self.assertEqual(len(self.timeline()["records"]),2)
        self.assertEqual(self.client.get("/api/uploads").json()[0]["patient_id"],self.patient["id"])

    def test_legacy_no_context_api_keeps_account_only_history(self):
        response=self.client.post("/api/risk/predict",json={"inputs":{"age":56,"bmi":25}})
        self.assertEqual(response.status_code,200,response.text)
        self.assertIsNone(response.json()["patient_id"])
        self.assertEqual(len(self.store.list_records(self.user["id"])),1)
        self.assertEqual(self.timeline()["records"],[])

class AssessmentComparisonTests(unittest.TestCase):
    def risk(self,rid,date,probability=0.5,artifact="model-a",reviewed=True):
        result={**RISK,"risk_predictions":{"diabetes":{**RISK["risk_predictions"]["diabetes"],"calibrated_risk":probability}}}
        return {"id":rid,"filename":rid,"observed_on":date,"review_status":"reviewed" if reviewed else "unreviewed",
            "kind":"risk_assessment","result":{"result":result,"artifact_id":artifact}}

    def test_risk_trends_separate_models_dates_and_pending_reviews(self):
        records=[self.risk("a","2025-01-01",0.7),self.risk("b","2025-02-01",0.4),
                 self.risk("c","2025-03-01",0.2,artifact="model-b"),self.risk("d","2025-04-01",0.9,reviewed=False)]
        result=build_timeline({},records)
        self.assertEqual(len(result["risk_series"]),2)
        series=next(s for s in result["risk_series"] if s["artifact_id"]=="model-a")
        self.assertEqual(series["change"],-30)
        self.assertEqual(series["direction"],"estimate_change_only")
        self.assertEqual(result["assessment"]["chronic_risks"][0]["record_id"],"d")
        self.assertEqual(result["assessment"]["trend_summary"]["comparable_risk_series"],1)

    def test_same_day_risk_does_not_imply_chronology(self):
        result=build_timeline({},[self.risk("a","2025-01-01"),self.risk("b","2025-01-01",0.2)])
        self.assertIsNone(result["risk_series"][0]["change"])

    def test_archive_or_rejected_image_does_not_generate_diagnosis(self):
        for status in ("not_screened","low_confidence","rejected_non_medical_input"):
            record={"id":"image","kind":"medical_image","observed_on":"2025-01-01","result":{
                **IMAGE,"status":status,"predicted_class":"must not appear"}}
            result=build_timeline({},[record])
            self.assertEqual(result["assessment"]["status"],"insufficient_evidence")
            self.assertIsNone(result["assessment"]["image_findings"][0]["predicted_class"])

    def test_form_aliases_reach_models_and_explicit_canonical_wins(self):
        result=normalize_risk_inputs({"gender":2,"creatinine":1.2,"pulse":75,"waist_circumference_cm":90,
            "physical_activity":2,"history_diabetes":1,"smoking_status":2})
        self.assertEqual(result["sex_male"],0); self.assertEqual(result["serum_creatinine"],1.2)
        self.assertEqual(result["pulse_bpm"],75); self.assertEqual(result["waist_cm"],90)
        self.assertEqual(result["target_diabetes"],1); self.assertEqual(result["physically_active"],0)
        self.assertEqual(result["current_smoker"],1); self.assertNotIn("gender",result)
        self.assertEqual(normalize_risk_inputs({"gender":1,"sex_male":0,"creatinine":9,"serum_creatinine":1})["serum_creatinine"],1)
        self.assertEqual(normalize_risk_inputs({"gender":1,"sex_male":0})["sex_male"],0)

if __name__ == "__main__": unittest.main()
