"""Tests for Lab Report Extraction, Unit Normalization, and Chronic Risk Prediction (Phase 3)."""

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
import pymupdf
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from backend.main import app
from pulmoai.report_reader import (
    determine_flag,
    normalize_unit_and_value,
    parse_lab_text,
    process_lab_report,
)
from pulmoai.risk_service import HealthRiskService

# ---------------------------------------------------------------------------
# 3 Realistic Synthetic Dummy Lab Reports (No real patient data)
# ---------------------------------------------------------------------------

DUMMY_REPORT_1_US_METABOLIC = """
======================================================================
ACME DIAGNOSTICS - CLINICAL PATHOLOGY REPORT
Patient: Sample Patient A     DOB: 05/12/1972     Sex: Male
Physician: Dr. J. Clinical   Collection Date: 2026-03-10
======================================================================
TEST NAME                       RESULT   FLAG   REFERENCE INTERVAL   UNITS
----------------------------------------------------------------------
GLUCOSE, FASTING                 118      H       70 - 99            mg/dL
BLOOD UREA NITROGEN (BUN)         17              7 - 20             mg/dL
CREATININE, SERUM                1.05             0.60 - 1.30        mg/dL
SODIUM                           140             135 - 145           mmol/L
POTASSIUM                        4.4             3.5 - 5.2           mmol/L
CHLORIDE                         102              96 - 106           mmol/L
CALCIUM                          9.5             8.5 - 10.5          mg/dL
TOTAL PROTEIN                    7.2             6.0 - 8.3           g/dL
ALBUMIN                          4.3             3.5 - 5.0           g/dL
TOTAL BILIRUBIN                  0.7             0.2 - 1.2           mg/dL
ALT (SGPT)                        36              7 - 56             U/L
AST (SGOT)                        31             10 - 40             U/L

LIPID PANEL
CHOLESTEROL, TOTAL               235      H       < 200              mg/dL
HDL CHOLESTEROL                   36      L       >= 40              mg/dL
TRIGLYCERIDES                    210      H       < 150              mg/dL
LDL CHOLESTEROL (CALC)           157      H       < 100              mg/dL

HEMOGLOBIN A1c (HbA1c)           6.3      H       4.0 - 5.6          %

CLINICAL VITALS
Blood Pressure: 142/88 mmHg
Heart Rate: 74 bpm
"""

DUMMY_REPORT_2_UK_SI_UNITS = """
======================================================================
NATIONAL HEALTH SERVICE - BIOCHEMISTRY INVESTIGATION
Patient: Dummy Patient B      DOB: 11/08/1980     Sex: Female
Hospital: St. Jude Medical    Specimen ID: LAB-2026-9912
======================================================================
INVESTIGATION                   RESULT           REFERENCE RANGE     UNITS
----------------------------------------------------------------------
Fasting Blood Sugar: 6.5 mmol/L (3.9 - 5.5)
Serum Cholesterol: 5.9 mmol/L (< 5.2)
Serum Triglycerides: 2.1 mmol/L (< 1.7)
HDL-C: 0.98 mmol/L (> 1.0)
Creatinine: 92 umol/L (53 - 115)
HbA1c (IFCC): 45 mmol/mol (20 - 38)
Total Calcium: 2.38 mmol/L (2.15 - 2.55)
Serum Albumin: 44 g/L (35 - 50)
Serum Potassium: 4.1 mmol/L (3.5 - 5.2)
Serum Sodium: 139 mmol/L (135 - 145)

Blood Pressure: 126/82 mmHg
"""

DUMMY_REPORT_3_HEMATOLOGY_CBC = """
======================================================================
METRO HEALTH LABS - COMPLETE BLOOD COUNT & HEMATOLOGY
Patient: Dummy Patient C      DOB: 02/20/1965     Sex: Male
======================================================================
TEST                            RESULT   FLAG   REFERENCE RANGE     UNITS
----------------------------------------------------------------------
White Blood Cell Count (WBC)     7.4             4.5 - 11.0         10^3/uL
Red Blood Cell Count (RBC)       4.85            4.20 - 5.90        10^6/uL
Hemoglobin                      15.2            12.0 - 17.5         g/dL
Hematocrit                      44.8            36.0 - 50.0         %
Platelet Count                   265             150 - 450          10^3/uL
Mean Corpuscular Volume (MCV)   89.5            80.0 - 100.0        fL
Mean Corpuscular Hemoglobin     30.2            27.0 - 33.0         pg
MCHC                            33.8            32.0 - 36.0         g/dL
RDW                             12.8            11.5 - 14.5         %

Fasting Glucose                  92              70 - 99            mg/dL
Total Cholesterol               185             < 200               mg/dL
Resting Pulse                    68              60 - 100           bpm
"""


class ReportReaderUnitTest(unittest.TestCase):
    """Unit tests for report parsing, unit conversion, and risk modeling."""

    def test_parse_dummy_report_1_us_units(self) -> None:
        """Verify standard US unit report extracts all biomarkers with correct flags."""
        items, unmatched = parse_lab_text(DUMMY_REPORT_1_US_METABOLIC)
        extracted_map = {it.canonical_name: it for it in items}

        self.assertIn("fasting_glucose", extracted_map)
        self.assertEqual(extracted_map["fasting_glucose"].value, 118.0)
        self.assertEqual(extracted_map["fasting_glucose"].unit, "mg/dL")
        self.assertEqual(extracted_map["fasting_glucose"].flag, "HIGH")

        self.assertIn("total_cholesterol", extracted_map)
        self.assertEqual(extracted_map["total_cholesterol"].value, 235.0)
        self.assertEqual(extracted_map["total_cholesterol"].flag, "HIGH")

        self.assertIn("hdl_cholesterol", extracted_map)
        self.assertEqual(extracted_map["hdl_cholesterol"].value, 36.0)
        self.assertEqual(extracted_map["hdl_cholesterol"].flag, "LOW")

        self.assertIn("systolic_bp", extracted_map)
        self.assertEqual(extracted_map["systolic_bp"].value, 142.0)
        self.assertEqual(extracted_map["systolic_bp"].flag, "HIGH")

        self.assertIn("diastolic_bp", extracted_map)
        self.assertEqual(extracted_map["diastolic_bp"].value, 88.0)
        self.assertEqual(extracted_map["diastolic_bp"].flag, "HIGH")

        self.assertIn("hba1c", extracted_map)
        self.assertEqual(extracted_map["hba1c"].value, 6.3)
        self.assertEqual(extracted_map["hba1c"].flag, "HIGH")

        # Confidence checks: all parsed with high confidence
        for it in items:
            self.assertGreaterEqual(it.confidence, 0.90)

    def test_parse_dummy_report_2_si_units(self) -> None:
        """Verify UK/SI units are automatically converted to standard NHANES units."""
        items, _ = parse_lab_text(DUMMY_REPORT_2_UK_SI_UNITS)
        extracted_map = {it.canonical_name: it for it in items}

        # Glucose: 6.5 mmol/L * 18.0182 = 117.1 mg/dL
        self.assertIn("fasting_glucose", extracted_map)
        self.assertAlmostEqual(extracted_map["fasting_glucose"].value, 117.1, places=1)
        self.assertEqual(extracted_map["fasting_glucose"].unit, "mg/dL")
        self.assertEqual(extracted_map["fasting_glucose"].flag, "HIGH")

        # Cholesterol: 5.9 mmol/L * 38.67 = 228.2 mg/dL
        self.assertIn("total_cholesterol", extracted_map)
        self.assertAlmostEqual(extracted_map["total_cholesterol"].value, 228.2, places=1)
        self.assertEqual(extracted_map["total_cholesterol"].unit, "mg/dL")
        self.assertEqual(extracted_map["total_cholesterol"].flag, "HIGH")

        # Creatinine: 92 umol/L / 88.42 = 1.04 mg/dL
        self.assertIn("serum_creatinine", extracted_map)
        self.assertAlmostEqual(extracted_map["serum_creatinine"].value, 1.04, places=1)
        self.assertEqual(extracted_map["serum_creatinine"].unit, "mg/dL")

        # HbA1c: 45 mmol/mol -> (45 * 0.09148) + 2.152 = 6.27%
        self.assertIn("hba1c", extracted_map)
        self.assertAlmostEqual(extracted_map["hba1c"].value, 6.27, places=1)
        self.assertEqual(extracted_map["hba1c"].unit, "%")
        self.assertEqual(extracted_map["hba1c"].flag, "HIGH")

        # Albumin: 44 g/L / 10 = 4.4 g/dL
        self.assertIn("albumin", extracted_map)
        self.assertAlmostEqual(extracted_map["albumin"].value, 4.4, places=1)
        self.assertEqual(extracted_map["albumin"].unit, "g/dL")

    def test_parse_dummy_report_3_cbc(self) -> None:
        """Verify CBC and normal hematology values are parsed correctly."""
        items, _ = parse_lab_text(DUMMY_REPORT_3_HEMATOLOGY_CBC)
        extracted_map = {it.canonical_name: it for it in items}

        self.assertIn("wbc_count", extracted_map)
        self.assertEqual(extracted_map["wbc_count"].value, 7.4)
        self.assertEqual(extracted_map["wbc_count"].flag, "NORMAL")

        self.assertIn("hemoglobin", extracted_map)
        self.assertEqual(extracted_map["hemoglobin"].value, 15.2)
        self.assertEqual(extracted_map["hemoglobin"].flag, "NORMAL")

        self.assertIn("platelets", extracted_map)
        self.assertEqual(extracted_map["platelets"].value, 265.0)
        self.assertEqual(extracted_map["platelets"].flag, "NORMAL")

    def test_digital_pdf_layer_extraction(self) -> None:
        """Create a real in-memory PDF with text layer and verify extraction."""
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text(
            (50, 72),
            "CLINICAL LABORATORY REPORT\n"
            "Fasting Glucose: 104 mg/dL\n"
            "Total Cholesterol: 215 mg/dL\n"
            "Blood Pressure: 134/86 mmHg\n",
            fontsize=12,
        )
        pdf_bytes = doc.tobytes()
        doc.close()

        res = process_lab_report(pdf_bytes, "test_report.pdf", "application/pdf")
        self.assertEqual(res["extraction_method"], "pdf_text_layer")
        test_names = {t["canonical_name"]: t["value"] for t in res["extracted_tests"]}
        self.assertIn("fasting_glucose", test_names)
        self.assertEqual(test_names["fasting_glucose"], 104.0)
        self.assertIn("total_cholesterol", test_names)
        self.assertEqual(test_names["total_cholesterol"], 215.0)
        self.assertIn("systolic_bp", test_names)
        self.assertEqual(test_names["systolic_bp"], 134.0)

    def test_image_tesseract_ocr_extraction(self) -> None:
        """Create an in-memory image and verify local Tesseract OCR extracts lab text."""
        img = Image.new("RGB", (500, 150), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 30), "Fasting Glucose: 110 mg/dL", fill=(0, 0, 0))
        draw.text((20, 70), "Total Cholesterol: 220 mg/dL", fill=(0, 0, 0))

        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")
        png_bytes = img_buf.getvalue()

        res = process_lab_report(png_bytes, "scan.png", "image/png")
        self.assertEqual(res["extraction_method"], "tesseract_ocr")
        test_names = {t["canonical_name"]: t["value"] for t in res["extracted_tests"]}
        self.assertIn("fasting_glucose", test_names)

    def test_health_risk_service_predict(self) -> None:
        """Verify HealthRiskService loads Phase 2 models and produces calibrated risks."""
        service = HealthRiskService.load()
        inputs = {
            "age": 58,
            "sex_male": 1,
            "bmi": 32.1,
            "systolic_bp": 145,
            "diastolic_bp": 90,
            "total_cholesterol": 240,
            "hdl_cholesterol": 35,
            "triglycerides": 210,
            "smoking_status": 2,
            "physically_active": 0,
        }
        pred = service.predict_risk(inputs)

        # Must evaluate all 6 conditions
        expected_targets = {
            "diabetes", "hypertension", "high_cholesterol",
            "heart_disease", "stroke", "copd",
        }
        self.assertEqual(set(pred["risk_predictions"].keys()), expected_targets)

        for t_id, item in pred["risk_predictions"].items():
            prob = item["calibrated_risk"]
            self.assertIsNotNone(prob)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertIn(item["risk_category"], {"Low", "Moderate", "Elevated", "High"})

        self.assertIn("disclaimer", pred)
        self.assertTrue(len(pred["contributing_factors"]) > 0)


class ReportReaderAPITests(unittest.TestCase):
    """Integration tests for FastAPI endpoints."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_api_report_extract_txt(self) -> None:
        """Test POST /api/report/extract with a text report."""
        files = {
            "file": ("lab_report.txt", DUMMY_REPORT_1_US_METABOLIC.encode("utf-8"), "text/plain")
        }
        resp = self.client.post("/api/report/extract", files=files)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("extracted_tests", data)
        self.assertGreater(data["count"], 5)
        names = [t["canonical_name"] for t in data["extracted_tests"]]
        self.assertIn("fasting_glucose", names)
        self.assertIn("total_cholesterol", names)

    def test_api_report_extract_unsupported_format(self) -> None:
        """Verify 415 on unsupported file extensions."""
        files = {"file": ("report.exe", b"binary content", "application/octet-stream")}
        resp = self.client.post("/api/report/extract", files=files)
        self.assertEqual(resp.status_code, 415)

    def test_api_risk_predict_endpoint(self) -> None:
        """Test POST /api/risk/predict endpoint returns calibrated predictions and disclaimer."""
        payload = {
            "inputs": {
                "age": 62,
                "sex_male": 1,
                "bmi": 29.5,
                "systolic_bp": 138,
                "diastolic_bp": 85,
                "total_cholesterol": 225,
                "hdl_cholesterol": 38,
                "triglycerides": 180,
                "smoking_status": 1,
                "physically_active": 1,
            }
        }
        resp = self.client.post("/api/risk/predict", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("risk_predictions", data)
        self.assertIn("diabetes", data["risk_predictions"])
        self.assertIn("hypertension", data["risk_predictions"])
        self.assertIn("disclaimer", data)
        self.assertIn("contributing_factors", data)


if __name__ == "__main__":
    unittest.main()
