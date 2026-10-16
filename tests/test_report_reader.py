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


class ReportReaderRegressionTests(unittest.TestCase):
    """Regression tests for clean PDF extraction, alias priorities, and adaptive preprocessing."""

    def test_clean_pdf_five_layouts_accuracy(self) -> None:
        """Verify native PDF text layer extraction achieves >= 98% accuracy across all 5 layouts."""
        from scripts.audit_ocr_part_a import build_synthetic_dataset, render_pdf_bytes, evaluate_extracted_entities

        dataset = build_synthetic_dataset(n_variations_per_layout=10)
        by_layout: dict[str, list[int]] = {}

        for d in dataset:
            lid = d["layout_id"]
            if lid not in by_layout:
                by_layout[lid] = [0, 0]
            pdf_b = render_pdf_bytes(d["text"])
            doc = pymupdf.open(stream=pdf_b, filetype="pdf")
            txt = doc[0].get_text()
            doc.close()
            extracted, _ = parse_lab_text(txt, sex=d["patient_sex"])
            m, tot, _ = evaluate_extracted_entities(extracted, d["entities"])
            by_layout[lid][0] += m
            by_layout[lid][1] += tot

        for lid, (m, tot) in by_layout.items():
            acc = m / tot
            self.assertGreaterEqual(
                acc,
                0.98,
                f"Layout {lid} clean PDF accuracy {acc:.2%} ({m}/{tot}) below 98.0% target",
            )

    def test_hdl_direct_not_overwritten_by_total_cholesterol(self) -> None:
        """Verify HDL direct assay is not captured by total cholesterol alias."""
        text = (
            "REGIONAL CLINICAL PATHOLOGY LABORATORY\n"
            "TEST DESCRIPTION               FINDING   STANDARD UNIT   REFERENCE RANGE\n"
            "CHOLESTEROL, TOTAL              215.0     mg/dL           < 200.0\n"
            "HDL CHOLESTEROL - DIRECT        48.0      mg/dL           >= 40.0\n"
            "TRIGLYCERIDES                   160.0     mg/dL           < 150.0\n"
        )
        extracted, _ = parse_lab_text(text, sex="male")
        extracted_map = {it.canonical_name: it for it in extracted}

        self.assertIn("total_cholesterol", extracted_map)
        self.assertEqual(extracted_map["total_cholesterol"].value, 215.0)
        self.assertEqual(extracted_map["total_cholesterol"].flag, "HIGH")

        self.assertIn("hdl_cholesterol", extracted_map)
        self.assertEqual(extracted_map["hdl_cholesterol"].value, 48.0)
        self.assertEqual(extracted_map["hdl_cholesterol"].flag, "NORMAL")

        self.assertIn("triglycerides", extracted_map)
        self.assertEqual(extracted_map["triglycerides"].value, 160.0)
        self.assertEqual(extracted_map["triglycerides"].flag, "HIGH")

    def test_one_sided_reference_range_flagging(self) -> None:
        """Verify one-sided printed ranges fall back to physiological minima for deficit detection."""
        # Total cholesterol: printed < 200 mg/dL, physiological min 100 mg/dL
        flag_low = determine_flag("total_cholesterol", 80.0, ref_max=199.0)
        self.assertEqual(flag_low, "LOW")

        flag_norm = determine_flag("total_cholesterol", 150.0, ref_max=199.0)
        self.assertEqual(flag_norm, "NORMAL")

        # Triglycerides: printed < 150 mg/dL, physiological min 50 mg/dL
        flag_tg_low = determine_flag("triglycerides", 35.0, ref_max=149.0)
        self.assertEqual(flag_tg_low, "LOW")

    def test_bulleted_and_prefixed_lines(self) -> None:
        """Verify leading bullet points and hyphens are stripped without corrupting test names."""
        text = (
            "HOSPITAL DISCHARGE SUMMARY\n"
            "- Fasting Plasma Glucose: 110.0 mg/dL (Ref: 70.0 - 99.0)\n"
            "• Creatinine: 0.9 mg/dL (Ref: 0.7 - 1.3)\n"
            "* Hemoglobin: 14.5 g/dL (Ref: 13.5 - 17.5)\n"
        )
        extracted, _ = parse_lab_text(text, sex="male")
        extracted_map = {it.canonical_name: it for it in extracted}

        self.assertIn("fasting_glucose", extracted_map)
        self.assertEqual(extracted_map["fasting_glucose"].value, 110.0)
        self.assertEqual(extracted_map["fasting_glucose"].flag, "HIGH")

        self.assertIn("serum_creatinine", extracted_map)
        self.assertEqual(extracted_map["serum_creatinine"].value, 0.9)
        self.assertEqual(extracted_map["serum_creatinine"].flag, "NORMAL")

        self.assertIn("hemoglobin", extracted_map)
        self.assertEqual(extracted_map["hemoglobin"].value, 14.5)
        self.assertEqual(extracted_map["hemoglobin"].flag, "NORMAL")

    def test_adaptive_preprocessing_decision(self) -> None:
        """Verify adaptive preprocessing gates blur and skew accurately."""
        from pulmoai.report_reader import preprocess_lab_image_adaptive
        from PIL import ImageFilter

        # Base clear image
        base = Image.new("RGB", (800, 600), color=(255, 255, 255))
        draw = ImageDraw.Draw(base)
        for y in range(50, 550, 40):
            draw.text((50, y), f"Sample Laboratory Text Line {y} Result 120.0 mg/dL", fill=(0, 0, 0))

        # Blurred image
        blurred = base.filter(ImageFilter.GaussianBlur(radius=3.0))
        _, rationale_blur = preprocess_lab_image_adaptive(blurred)
        self.assertIn("blur_defocus_detected", rationale_blur)

        # Rotated image
        rotated = base.rotate(3.5, expand=True, fillcolor=(255, 255, 255))
        _, rationale_rot = preprocess_lab_image_adaptive(rotated)
        self.assertIn("deskewed_and_enhanced", rationale_rot)

    def test_boundary_semantics_inclusive_reference_limits(self) -> None:
        """Verify settled boundary semantics: reference intervals are strictly inclusive [min, max]."""
        # Exactly on the lower bound -> NORMAL
        self.assertEqual(determine_flag("fasting_glucose", 70.0, ref_min=70.0, ref_max=99.0), "NORMAL")
        # Exactly on the upper bound -> NORMAL
        self.assertEqual(determine_flag("fasting_glucose", 99.0, ref_min=70.0, ref_max=99.0), "NORMAL")
        # Strictly below lower bound -> LOW
        self.assertEqual(determine_flag("fasting_glucose", 69.9, ref_min=70.0, ref_max=99.0), "LOW")
        # Strictly above upper bound -> HIGH
        self.assertEqual(determine_flag("fasting_glucose", 99.1, ref_min=70.0, ref_max=99.0), "HIGH")
        # Female creatinine exact lower cutoff 0.5 mg/dL -> NORMAL
        # Female creatinine below lower cutoff 0.49 mg/dL -> LOW
        self.assertEqual(determine_flag("serum_creatinine", 0.49, ref_min=0.5, ref_max=1.1), "LOW")

    def test_pass5_one_sided_ranges_and_composite_hdl(self) -> None:
        """Verify Pass 5 fixes for one-sided ranges (<=5.7, >=10, <200, >40) and composite HDL names."""
        text = (
            "CHOLESTEROL HDL DIRECT   45.0   mg/dL   > 40 mg/dL\n"
            "HBA1C   5.7   %   <= 5.7\n"
            "BUN   9.0   mg/dL   >=10\n"
            "CHOLESTEROL TOTAL   210.0   mg/dL   <200\n"
        )
        extracted, _ = parse_lab_text(text)
        res_map = {it.canonical_name: it for it in extracted}

        self.assertIn("hdl_cholesterol", res_map)
        self.assertEqual(res_map["hdl_cholesterol"].value, 45.0)
        self.assertEqual(res_map["hdl_cholesterol"].flag, "NORMAL")

        self.assertIn("hba1c", res_map)
        self.assertEqual(res_map["hba1c"].value, 5.7)
        self.assertEqual(res_map["hba1c"].flag, "NORMAL")

        self.assertIn("bun", res_map)
        self.assertEqual(res_map["bun"].value, 9.0)
        self.assertEqual(res_map["bun"].flag, "LOW")

        self.assertIn("total_cholesterol", res_map)
        self.assertEqual(res_map["total_cholesterol"].value, 210.0)
        self.assertEqual(res_map["total_cholesterol"].flag, "HIGH")

    def test_hba1c_alias_safety_and_explicit_patterns(self) -> None:
        """Verify bare ALC (CBC) and Alcohol do not produce HbA1c, while explicit patterns do."""
        # CBC line with ALC must not produce HbA1c
        extracted_alc, unmatched_alc = parse_lab_text("ALC  2.1  x10^3/uL")
        self.assertEqual(len(extracted_alc), 0)
        self.assertIn("ALC  2.1  x10^3/uL", unmatched_alc)

        # Alcohol must not match anything
        extracted_alc2, unmatched_alc2 = parse_lab_text("Alcohol  0.0  g/dL")
        self.assertEqual(len(extracted_alc2), 0)
        self.assertIn("Alcohol  0.0  g/dL", unmatched_alc2)

        # HbAlc 6.1 % and Hb A1c 6.1 % must produce HbA1c 6.1
        extracted_hbalc, _ = parse_lab_text("HbAlc 6.1 %")
        self.assertEqual(len(extracted_hbalc), 1)
        self.assertEqual(extracted_hbalc[0].canonical_name, "hba1c")
        self.assertEqual(extracted_hbalc[0].value, 6.1)

        extracted_hb_a1c, _ = parse_lab_text("Hb A1c 6.1 %")
        self.assertEqual(len(extracted_hb_a1c), 1)
        self.assertEqual(extracted_hb_a1c[0].canonical_name, "hba1c")
        self.assertEqual(extracted_hb_a1c[0].value, 6.1)


if __name__ == "__main__":
    unittest.main()


