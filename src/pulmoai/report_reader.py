"""Lab Report Document Extraction and Normalization Engine.

Handles text extraction from digital PDFs (via PyMuPDF/pypdf) and scanned images
(via local Tesseract OCR or optional Gemini API).
Parses unstructured laboratory text into a canonical dictionary of normalized
biomarkers, maps units, flags abnormal values against standard reference ranges,
and outputs confidence scores so the user can review and edit values.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pymupdf
from PIL import Image

logger = logging.getLogger("pulmoai.report_reader")

# ---------------------------------------------------------------------------
# Canonical Laboratory Test Catalog & Reference Ranges
# ---------------------------------------------------------------------------

@dataclass
class CanonicalTestDef:
    canonical_name: str
    display_name: str
    category: str
    target_unit: str
    aliases: list[str]
    ref_min: float | None = None
    ref_max: float | None = None
    ref_str: str = ""


CANONICAL_TESTS: list[CanonicalTestDef] = [
    # Glycemic
    CanonicalTestDef(
        canonical_name="fasting_glucose",
        display_name="Fasting Plasma Glucose",
        category="Glycemic",
        target_unit="mg/dL",
        aliases=[
            "glucose, fasting", "fasting glucose", "glucose", "fbs", "fpg",
            "blood glucose", "glucose serum", "serum glucose", "plasma glucose",
            "fasting blood sugar", "blood sugar",
        ],
        ref_min=70.0,
        ref_max=99.0,
        ref_str="70 - 99 mg/dL",
    ),
    CanonicalTestDef(
        canonical_name="hba1c",
        display_name="Glycated Hemoglobin (HbA1c)",
        category="Glycemic",
        target_unit="%",
        aliases=[
            "hba1c", "hemoglobin a1c", "glycohemoglobin", "glycated hemoglobin",
            "a1c", "hba1c (ngsp)", "glycated hb",
        ],
        ref_min=4.0,
        ref_max=5.6,
        ref_str="4.0 - 5.6 %",
    ),
    # Lipids
    CanonicalTestDef(
        canonical_name="total_cholesterol",
        display_name="Total Cholesterol",
        category="Lipid Panel",
        target_unit="mg/dL",
        aliases=[
            "total cholesterol", "cholesterol, total", "cholesterol total",
            "serum cholesterol", "cholesterol",
        ],
        ref_min=100.0,
        ref_max=199.0,
        ref_str="< 200 mg/dL",
    ),
    CanonicalTestDef(
        canonical_name="hdl_cholesterol",
        display_name="HDL Cholesterol",
        category="Lipid Panel",
        target_unit="mg/dL",
        aliases=[
            "hdl cholesterol", "hdl-cholesterol", "hdl-c", "hdl",
            "high density lipoprotein", "cholesterol, hdl",
        ],
        ref_min=40.0,
        ref_max=100.0,
        ref_str=">= 40 mg/dL",
    ),
    CanonicalTestDef(
        canonical_name="triglycerides",
        display_name="Triglycerides",
        category="Lipid Panel",
        target_unit="mg/dL",
        aliases=[
            "triglycerides", "triglyceride", "serum triglycerides",
            "fasting triglycerides", "tg",
        ],
        ref_min=30.0,
        ref_max=149.0,
        ref_str="< 150 mg/dL",
    ),
    CanonicalTestDef(
        canonical_name="ldl_cholesterol",
        display_name="LDL Cholesterol (Calculated)",
        category="Lipid Panel",
        target_unit="mg/dL",
        aliases=[
            "ldl cholesterol", "ldl-cholesterol", "ldl-c", "ldl",
            "low density lipoprotein", "cholesterol, ldl", "calculated ldl",
        ],
        ref_min=0.0,
        ref_max=99.0,
        ref_str="< 100 mg/dL",
    ),
    # Renal & Metabolic
    CanonicalTestDef(
        canonical_name="serum_creatinine",
        display_name="Serum Creatinine",
        category="Renal Function",
        target_unit="mg/dL",
        aliases=[
            "creatinine", "serum creatinine", "creatinine, serum",
            "creat", "cr", "s. creatinine",
        ],
        ref_min=0.6,
        ref_max=1.3,
        ref_str="0.6 - 1.3 mg/dL",
    ),
    CanonicalTestDef(
        canonical_name="bun",
        display_name="Blood Urea Nitrogen (BUN)",
        category="Renal Function",
        target_unit="mg/dL",
        aliases=[
            "bun", "blood urea nitrogen", "urea nitrogen", "urea", "serum urea",
        ],
        ref_min=7.0,
        ref_max=20.0,
        ref_str="7 - 20 mg/dL",
    ),
    CanonicalTestDef(
        canonical_name="uric_acid",
        display_name="Serum Uric Acid",
        category="Renal Function",
        target_unit="mg/dL",
        aliases=["uric acid", "serum uric acid", "urate"],
        ref_min=3.5,
        ref_max=7.2,
        ref_str="3.5 - 7.2 mg/dL",
    ),
    # Hepatic Panel
    CanonicalTestDef(
        canonical_name="albumin",
        display_name="Serum Albumin",
        category="Hepatic Function",
        target_unit="g/dL",
        aliases=["albumin", "serum albumin", "alb"],
        ref_min=3.5,
        ref_max=5.0,
        ref_str="3.5 - 5.0 g/dL",
    ),
    CanonicalTestDef(
        canonical_name="total_protein",
        display_name="Total Protein",
        category="Hepatic Function",
        target_unit="g/dL",
        aliases=["total protein", "serum total protein", "protein, total", "tp"],
        ref_min=6.0,
        ref_max=8.3,
        ref_str="6.0 - 8.3 g/dL",
    ),
    CanonicalTestDef(
        canonical_name="total_bilirubin",
        display_name="Total Bilirubin",
        category="Hepatic Function",
        target_unit="mg/dL",
        aliases=[
            "total bilirubin", "bilirubin, total", "bilirubin total",
            "t. bili", "t-bili",
        ],
        ref_min=0.2,
        ref_max=1.2,
        ref_str="0.2 - 1.2 mg/dL",
    ),
    CanonicalTestDef(
        canonical_name="alt",
        display_name="ALT (SGPT)",
        category="Hepatic Function",
        target_unit="U/L",
        aliases=[
            "alt", "alanine aminotransferase", "sgpt", "alanine transaminase",
            "alt (sgpt)",
        ],
        ref_min=7.0,
        ref_max=56.0,
        ref_str="7 - 56 U/L",
    ),
    CanonicalTestDef(
        canonical_name="ast",
        display_name="AST (SGOT)",
        category="Hepatic Function",
        target_unit="U/L",
        aliases=[
            "ast", "aspartate aminotransferase", "sgot", "aspartate transaminase",
            "ast (sgot)",
        ],
        ref_min=10.0,
        ref_max=40.0,
        ref_str="10 - 40 U/L",
    ),
    CanonicalTestDef(
        canonical_name="alk_phosphatase",
        display_name="Alkaline Phosphatase (ALP)",
        category="Hepatic Function",
        target_unit="U/L",
        aliases=[
            "alkaline phosphatase", "alk phos", "alp", "alkaline phos",
        ],
        ref_min=44.0,
        ref_max=147.0,
        ref_str="44 - 147 U/L",
    ),
    # Electrolytes & Minerals
    CanonicalTestDef(
        canonical_name="sodium",
        display_name="Serum Sodium",
        category="Electrolytes",
        target_unit="mmol/L",
        aliases=["sodium", "serum sodium", "na", "na+"],
        ref_min=135.0,
        ref_max=145.0,
        ref_str="135 - 145 mmol/L",
    ),
    CanonicalTestDef(
        canonical_name="potassium",
        display_name="Serum Potassium",
        category="Electrolytes",
        target_unit="mmol/L",
        aliases=["potassium", "serum potassium", "k", "k+"],
        ref_min=3.5,
        ref_max=5.2,
        ref_str="3.5 - 5.2 mmol/L",
    ),
    CanonicalTestDef(
        canonical_name="chloride",
        display_name="Serum Chloride",
        category="Electrolytes",
        target_unit="mmol/L",
        aliases=["chloride", "serum chloride", "cl", "cl-"],
        ref_min=96.0,
        ref_max=106.0,
        ref_str="96 - 106 mmol/L",
    ),
    CanonicalTestDef(
        canonical_name="bicarbonate",
        display_name="Bicarbonate (CO2)",
        category="Electrolytes",
        target_unit="mmol/L",
        aliases=[
            "bicarbonate", "carbon dioxide", "co2", "total co2",
            "serum bicarbonate", "hco3",
        ],
        ref_min=22.0,
        ref_max=29.0,
        ref_str="22 - 29 mmol/L",
    ),
    CanonicalTestDef(
        canonical_name="calcium",
        display_name="Total Calcium",
        category="Electrolytes",
        target_unit="mg/dL",
        aliases=["calcium", "serum calcium", "total calcium", "ca"],
        ref_min=8.5,
        ref_max=10.5,
        ref_str="8.5 - 10.5 mg/dL",
    ),
    CanonicalTestDef(
        canonical_name="phosphorus",
        display_name="Serum Phosphorus",
        category="Electrolytes",
        target_unit="mg/dL",
        aliases=["phosphorus", "serum phosphorus", "phosphate", "po4"],
        ref_min=2.5,
        ref_max=4.5,
        ref_str="2.5 - 4.5 mg/dL",
    ),
    # Hematology (CBC)
    CanonicalTestDef(
        canonical_name="wbc_count",
        display_name="White Blood Cell Count (WBC)",
        category="Hematology",
        target_unit="10^3/uL",
        aliases=[
            "wbc", "white blood cell count", "white blood cells", "leukocytes",
            "wbc count",
        ],
        ref_min=4.5,
        ref_max=11.0,
        ref_str="4.5 - 11.0 10^3/uL",
    ),
    CanonicalTestDef(
        canonical_name="rbc_count",
        display_name="Red Blood Cell Count (RBC)",
        category="Hematology",
        target_unit="10^6/uL",
        aliases=[
            "rbc", "red blood cell count", "red blood cells", "erythrocytes",
            "rbc count",
        ],
        ref_min=4.2,
        ref_max=5.9,
        ref_str="4.2 - 5.9 10^6/uL",
    ),
    CanonicalTestDef(
        canonical_name="hemoglobin",
        display_name="Hemoglobin",
        category="Hematology",
        target_unit="g/dL",
        aliases=["hemoglobin", "hgb", "hb"],
        ref_min=12.0,
        ref_max=17.5,
        ref_str="12.0 - 17.5 g/dL",
    ),
    CanonicalTestDef(
        canonical_name="hematocrit",
        display_name="Hematocrit",
        category="Hematology",
        target_unit="%",
        aliases=["hematocrit", "hct", "pcv"],
        ref_min=36.0,
        ref_max=50.0,
        ref_str="36.0 - 50.0 %",
    ),
    CanonicalTestDef(
        canonical_name="platelets",
        display_name="Platelet Count",
        category="Hematology",
        target_unit="10^3/uL",
        aliases=[
            "platelets", "platelet count", "plt", "thrombocytes",
        ],
        ref_min=150.0,
        ref_max=450.0,
        ref_str="150 - 450 10^3/uL",
    ),
    CanonicalTestDef(
        canonical_name="mcv",
        display_name="Mean Corpuscular Volume (MCV)",
        category="Hematology",
        target_unit="fL",
        aliases=["mcv", "mean corpuscular volume"],
        ref_min=80.0,
        ref_max=100.0,
        ref_str="80 - 100 fL",
    ),
    CanonicalTestDef(
        canonical_name="mch",
        display_name="Mean Corpuscular Hemoglobin (MCH)",
        category="Hematology",
        target_unit="pg",
        aliases=["mch", "mean corpuscular hemoglobin"],
        ref_min=27.0,
        ref_max=33.0,
        ref_str="27 - 33 pg",
    ),
    CanonicalTestDef(
        canonical_name="mchc",
        display_name="Mean Corpuscular Hemoglobin Conc. (MCHC)",
        category="Hematology",
        target_unit="g/dL",
        aliases=["mchc", "mean corpuscular hemoglobin conc"],
        ref_min=32.0,
        ref_max=36.0,
        ref_str="32 - 36 g/dL",
    ),
    CanonicalTestDef(
        canonical_name="rdw",
        display_name="Red Cell Distribution Width (RDW)",
        category="Hematology",
        target_unit="%",
        aliases=["rdw", "red cell distribution width", "rdw-cv"],
        ref_min=11.5,
        ref_max=14.5,
        ref_str="11.5 - 14.5 %",
    ),
    # Vitals
    CanonicalTestDef(
        canonical_name="systolic_bp",
        display_name="Systolic Blood Pressure",
        category="Vitals",
        target_unit="mmHg",
        aliases=["systolic", "systolic bp", "systolic blood pressure", "sys bp"],
        ref_min=90.0,
        ref_max=120.0,
        ref_str="90 - 120 mmHg",
    ),
    CanonicalTestDef(
        canonical_name="diastolic_bp",
        display_name="Diastolic Blood Pressure",
        category="Vitals",
        target_unit="mmHg",
        aliases=["diastolic", "diastolic bp", "diastolic blood pressure", "dia bp"],
        ref_min=60.0,
        ref_max=80.0,
        ref_str="60 - 80 mmHg",
    ),
    CanonicalTestDef(
        canonical_name="pulse_bpm",
        display_name="Resting Heart Rate (Pulse)",
        category="Vitals",
        target_unit="bpm",
        aliases=["pulse", "heart rate", "resting heart rate", "pulse rate", "hr"],
        ref_min=60.0,
        ref_max=100.0,
        ref_str="60 - 100 bpm",
    ),
]

NAME_TO_DEF: dict[str, CanonicalTestDef] = {d.canonical_name: d for d in CANONICAL_TESTS}


@dataclass
class ExtractedTestItem:
    raw_name: str
    canonical_name: str
    display_name: str
    category: str
    value: float
    raw_value: str
    unit: str
    raw_unit: str
    reference_range: str
    flag: Literal["NORMAL", "HIGH", "LOW", "UNKNOWN"]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Unit Normalization Engine
# ---------------------------------------------------------------------------

def normalize_unit_and_value(
    canonical_name: str,
    raw_val: float,
    raw_unit_str: str,
) -> tuple[float, str, float]:
    """Convert incoming lab test units into standard NHANES target units.

    Returns:
        (converted_value, target_unit, unit_confidence_multiplier)
    """
    clean_unit = raw_unit_str.strip().lower()
    t_def = NAME_TO_DEF.get(canonical_name)
    target_unit = t_def.target_unit if t_def else ""

    # 1. Glucose: mmol/L -> mg/dL (1 mmol/L = 18.0182 mg/dL)
    if canonical_name == "fasting_glucose":
        if "mmol" in clean_unit:
            return round(raw_val * 18.0182, 1), "mg/dL", 0.95
        return round(raw_val, 1), "mg/dL", 0.98

    # 2. HbA1c: mmol/mol (IFCC) -> % (NGSP)
    # Equation: NGSP% = (0.09148 * IFCC mmol/mol) + 2.152
    if canonical_name == "hba1c":
        if "mmol/mol" in clean_unit or (raw_val > 20.0 and "%" not in clean_unit):
            ngsp_pct = (raw_val * 0.09148) + 2.152
            return round(ngsp_pct, 2), "%", 0.95
        return round(raw_val, 2), "%", 0.98

    # 3. Lipids (Cholesterol, HDL, LDL): mmol/L -> mg/dL (factor 38.67)
    if canonical_name in {"total_cholesterol", "hdl_cholesterol", "ldl_cholesterol"}:
        if "mmol" in clean_unit:
            return round(raw_val * 38.67, 1), "mg/dL", 0.95
        return round(raw_val, 1), "mg/dL", 0.98

    # 4. Triglycerides: mmol/L -> mg/dL (factor 88.57)
    if canonical_name == "triglycerides":
        if "mmol" in clean_unit:
            return round(raw_val * 88.57, 1), "mg/dL", 0.95
        return round(raw_val, 1), "mg/dL", 0.98

    # 5. Creatinine: umol/L -> mg/dL (divide by 88.42)
    if canonical_name == "serum_creatinine":
        if "umol" in clean_unit or "µmol" in clean_unit or "micromol" in clean_unit or raw_val > 30.0:
            return round(raw_val / 88.42, 2), "mg/dL", 0.95
        return round(raw_val, 2), "mg/dL", 0.98

    # 6. Total Bilirubin: umol/L -> mg/dL (divide by 17.1)
    if canonical_name == "total_bilirubin":
        if "umol" in clean_unit or "µmol" in clean_unit or raw_val > 5.0:
            return round(raw_val / 17.1, 2), "mg/dL", 0.95
        return round(raw_val, 2), "mg/dL", 0.98

    # 7. Calcium: mmol/L -> mg/dL (multiply by 4.008)
    if canonical_name == "calcium":
        if "mmol" in clean_unit or (raw_val < 4.0 and raw_val > 1.0):
            return round(raw_val * 4.008, 1), "mg/dL", 0.95
        return round(raw_val, 1), "mg/dL", 0.98

    # 8. Albumin & Total Protein: g/L -> g/dL (divide by 10)
    if canonical_name in {"albumin", "total_protein"}:
        if "g/l" in clean_unit or raw_val > 20.0:
            return round(raw_val / 10.0, 1), "g/dL", 0.95
        return round(raw_val, 1), "g/dL", 0.98

    # 9. Complete Blood Count
    if canonical_name in {"wbc_count", "platelets"}:
        if "/ul" in clean_unit and raw_val > 1000.0 and canonical_name == "wbc_count":
            return round(raw_val / 1000.0, 1), "10^3/uL", 0.95
        return round(raw_val, 1), "10^3/uL", 0.98

    return round(raw_val, 2), target_unit, 0.95


def determine_flag(
    canonical_name: str,
    val: float,
    explicit_flag_text: str = "",
) -> Literal["NORMAL", "HIGH", "LOW", "UNKNOWN"]:
    """Determine whether an extracted lab result is high, low, or normal."""
    explicit_clean = explicit_flag_text.strip().upper()
    if explicit_clean in {"H", "HIGH", "*H*", "CRITICAL HIGH"}:
        return "HIGH"
    if explicit_clean in {"L", "LOW", "*L*", "CRITICAL LOW"}:
        return "LOW"

    t_def = NAME_TO_DEF.get(canonical_name)
    if not t_def:
        return "UNKNOWN"

    if t_def.ref_max is not None and val > t_def.ref_max:
        return "HIGH"
    if t_def.ref_min is not None and val < t_def.ref_min:
        return "LOW"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Text Extraction: Digital PDF, Tesseract OCR, and Gemini
# ---------------------------------------------------------------------------

def extract_text_from_bytes(
    file_bytes: bytes,
    filename: str,
    mime_type: str | None = None,
) -> tuple[str, str]:
    """Extract raw text from PDF or image upload.

    Returns:
        (extracted_text, method_used)
    """
    ext = Path(filename).suffix.lower()
    is_pdf = ext == ".pdf" or (mime_type and "pdf" in mime_type.lower())

    if is_pdf:
        # Step 1: Try digital text layer via PyMuPDF
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            full_text = []
            for page in doc:
                txt = page.get_text()
                if txt:
                    full_text.append(txt)
            combined_text = "\n".join(full_text).strip()
            if len(combined_text) >= 50:
                logger.info("Successfully extracted text layer from PDF (%d chars)", len(combined_text))
                return combined_text, "pdf_text_layer"

            # If digital layer is empty/sparse, render pages and run local Tesseract OCR
            logger.info("PDF has minimal text layer; falling back to rendering page for OCR")
            ocr_text_list = []
            import pytesseract
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                page_ocr = pytesseract.image_to_string(img)
                if page_ocr:
                    ocr_text_list.append(page_ocr)
            ocr_combined = "\n".join(ocr_text_list).strip()
            if ocr_combined:
                return ocr_combined, "tesseract_ocr"
        except Exception as e:
            logger.warning("PDF extraction error: %s; falling back to OCR", e)

    # Step 2: Image or fallback OCR
    try:
        import pytesseract
        img = Image.open(io.BytesIO(file_bytes))
        ocr_text = pytesseract.image_to_string(img)
        if ocr_text.strip():
            return ocr_text.strip(), "tesseract_ocr"
    except Exception as e:
        logger.warning("Local Tesseract OCR error: %s", e)

    # If both failed or empty, try plain text decode if user uploaded .txt
    try:
        raw_decoded = file_bytes.decode("utf-8", errors="ignore").strip()
        if len(raw_decoded) > 10:
            return raw_decoded, "text_input"
    except Exception:
        pass

    return "", "unrecognized"


def extract_with_gemini(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    api_key: str | None = None,
) -> list[ExtractedTestItem] | None:
    """Optional extraction path using Gemini API if GEMINI_API_KEY is configured."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None

    try:
        import base64
        import requests

        b64_data = base64.b64encode(file_bytes).decode("utf-8")
        prompt = (
            "You are a medical laboratory report extractor. Extract all lab test results "
            "from this document into structured JSON. Format must be a JSON array of objects:\n"
            '[{"test_name": str, "value": float, "unit": str, "reference_range": str, "flag": "NORMAL"|"HIGH"|"LOW"}]\n'
            "Only output valid JSON, no markdown formatting."
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type or "application/pdf",
                                "data": b64_data,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0, "response_mime_type": "application/json"},
        }

        resp = requests.post(url, json=payload, timeout=25)
        if resp.status_code == 200:
            res_data = resp.json()
            text_resp = res_data["candidates"][0]["content"]["parts"][0]["text"]
            parsed_json = json.loads(text_resp)
            items = []
            for entry in parsed_json:
                raw_name = str(entry.get("test_name", ""))
                try:
                    val = float(entry.get("value", 0))
                except (ValueError, TypeError):
                    continue
                unit = str(entry.get("unit", ""))
                ref = str(entry.get("reference_range", ""))
                flag_str = str(entry.get("flag", "UNKNOWN"))

                # Match with canonical catalog
                canon_def = match_test_name(raw_name)
                if not canon_def:
                    continue
                norm_val, norm_unit, _ = normalize_unit_and_value(canon_def.canonical_name, val, unit)
                flag = determine_flag(canon_def.canonical_name, norm_val, flag_str)

                items.append(
                    ExtractedTestItem(
                        raw_name=raw_name,
                        canonical_name=canon_def.canonical_name,
                        display_name=canon_def.display_name,
                        category=canon_def.category,
                        value=norm_val,
                        raw_value=str(val),
                        unit=norm_unit,
                        raw_unit=unit,
                        reference_range=ref or canon_def.ref_str,
                        flag=flag,
                        confidence=0.96,
                    )
                )
            return items
    except Exception as e:
        logger.warning("Gemini extraction failed: %s; using local parsing", e)
    return None


# ---------------------------------------------------------------------------
# Rule-Based Structured Table & Line Parser
# ---------------------------------------------------------------------------

def match_test_name(name_candidate: str) -> CanonicalTestDef | None:
    """Fuzzy and alias match a candidate line against the canonical test catalog."""
    clean = re.sub(r"[^a-zA-Z0-9\s/+-]", " ", name_candidate.lower())
    clean = re.sub(r"\s+", " ", clean).strip()

    # Exact alias match
    for t_def in CANONICAL_TESTS:
        for alias in t_def.aliases:
            if clean == alias:
                return t_def

    # Starts with or contains alias as isolated token
    for t_def in CANONICAL_TESTS:
        for alias in t_def.aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", clean):
                return t_def

    return None


def parse_lab_text(text: str) -> tuple[list[ExtractedTestItem], list[str]]:
    """Parse unstructured report text into standardized lab entries.

    Returns:
        (extracted_items, unmatched_lines)
    """
    extracted: dict[str, ExtractedTestItem] = {}
    unmatched: list[str] = []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # 1. First check for Blood Pressure compound lines (e.g., "BP: 130/85 mmHg" or "130/85")
    bp_pattern = re.compile(
        r"(?:bp|blood\s*pressure)?\s*:?\s*(\d{2,3})\s*[/]\s*(\d{2,3})\s*(?:mm\s*hg)?",
        re.IGNORECASE,
    )

    for line in lines:
        bp_match = bp_pattern.search(line)
        if bp_match and ("bp" in line.lower() or "pressure" in line.lower() or "/" in line):
            sys_val = float(bp_match.group(1))
            dia_val = float(bp_match.group(2))
            if 70 <= sys_val <= 260 and 40 <= dia_val <= 150:
                if "systolic_bp" not in extracted:
                    extracted["systolic_bp"] = ExtractedTestItem(
                        raw_name="Systolic BP",
                        canonical_name="systolic_bp",
                        display_name="Systolic Blood Pressure",
                        category="Vitals",
                        value=sys_val,
                        raw_value=str(sys_val),
                        unit="mmHg",
                        raw_unit="mmHg",
                        reference_range="90 - 120 mmHg",
                        flag=determine_flag("systolic_bp", sys_val),
                        confidence=0.95,
                    )
                if "diastolic_bp" not in extracted:
                    extracted["diastolic_bp"] = ExtractedTestItem(
                        raw_name="Diastolic BP",
                        canonical_name="diastolic_bp",
                        display_name="Diastolic Blood Pressure",
                        category="Vitals",
                        value=dia_val,
                        raw_value=str(dia_val),
                        unit="mmHg",
                        raw_unit="mmHg",
                        reference_range="60 - 80 mmHg",
                        flag=determine_flag("diastolic_bp", dia_val),
                        confidence=0.95,
                    )

    # 2. General line parser: [Test Name] ... [Numeric Result] ... [Flag] ... [Units] ... [Reference]
    # Common format: "GLUCOSE, FASTING      104  H   70-99 mg/dL"
    # Or format: "Fasting Blood Sugar: 5.8 mmol/L (3.9 - 5.5)"
    # Or format: "Hemoglobin A1c (HbA1c)    6.2 %    High   < 5.7"
    for line in lines:
        # Skip header/footer noise
        if any(skip in line.lower() for skip in ["patient name", "date of birth", "physician", "specimen id", "accession", "page "]):
            continue

        # Look for numbers (int or float)
        # Match test name before the number
        match = re.search(
            r"^([A-Za-z0-9\s,/()\-+]+?)[\s:=-]+([><]?\s*\d+(?:\.\d+)?)\s*([A-Za-z*%]*)\s*([A-Za-z0-9/%^µμ\s\-.]+)?",
            line,
        )

        if not match:
            # Try delimiter splitting (tab, comma, multiple spaces)
            parts = re.split(r"\s{2,}|\t|,", line)
            if len(parts) >= 2:
                name_cand = parts[0].strip()
                t_def = match_test_name(name_cand)
                if t_def and t_def.canonical_name not in extracted:
                    # Look for first numeric part
                    for p in parts[1:]:
                        num_m = re.search(r"(\d+(?:\.\d+)?)", p)
                        if num_m:
                            try:
                                raw_val = float(num_m.group(1))
                                unit_cand = parts[-1] if len(parts) > 2 else ""
                                norm_val, norm_unit, conf = normalize_unit_and_value(t_def.canonical_name, raw_val, unit_cand)
                                flag = determine_flag(t_def.canonical_name, norm_val)
                                extracted[t_def.canonical_name] = ExtractedTestItem(
                                    raw_name=name_cand,
                                    canonical_name=t_def.canonical_name,
                                    display_name=t_def.display_name,
                                    category=t_def.category,
                                    value=norm_val,
                                    raw_value=str(raw_val),
                                    unit=norm_unit,
                                    raw_unit=unit_cand.strip(),
                                    reference_range=t_def.ref_str,
                                    flag=flag,
                                    confidence=conf,
                                )
                                break
                            except ValueError:
                                pass
            continue

        name_part = match.group(1).strip()
        val_str = match.group(2).replace(">", "").replace("<", "").strip()
        after_val = line[match.end(2):].strip()

        try:
            raw_val = float(val_str)
        except ValueError:
            unmatched.append(line)
            continue

        t_def = match_test_name(name_part)
        if not t_def:
            unmatched.append(line)
            continue

        # Extract unit candidate from after_val
        unit_match = re.search(
            r"\b(mg/dl|mmol/l|mmol/mol|g/dl|g/l|umol/l|µmol/l|u/l|iu/l|%|fl|pg|bpm|mmhg|10\^3/ul|k/ul)\b",
            after_val,
            re.IGNORECASE,
        )
        unit_cand = unit_match.group(1) if unit_match else ""

        # Extract explicit abnormal flag (H, HIGH, L, LOW) without matching /L or units
        flag_match = re.search(r"(?<![/\w])([HL]|HIGH|LOW|CRITICAL)(?![/\w])", after_val, re.IGNORECASE)
        flag_cand = flag_match.group(1) if flag_match else ""

        norm_val, norm_unit, conf = normalize_unit_and_value(t_def.canonical_name, raw_val, unit_cand)
        flag = determine_flag(t_def.canonical_name, norm_val, flag_cand)

        # Do not overwrite if already extracted with higher confidence
        if t_def.canonical_name not in extracted or conf > extracted[t_def.canonical_name].confidence:
            extracted[t_def.canonical_name] = ExtractedTestItem(
                raw_name=name_part,
                canonical_name=t_def.canonical_name,
                display_name=t_def.display_name,
                category=t_def.category,
                value=norm_val,
                raw_value=str(raw_val),
                unit=norm_unit,
                raw_unit=unit_cand,
                reference_range=t_def.ref_str,
                flag=flag,
                confidence=conf,
            )

    return list(extracted.values()), unmatched[:15]


def process_lab_report(
    file_bytes: bytes,
    filename: str,
    mime_type: str | None = None,
    gemini_key: str | None = None,
) -> dict[str, Any]:
    """End-to-end pipeline to extract and normalize laboratory report."""
    # 1. Optional Gemini Vision path if key is provided or set in env
    key = gemini_key or os.environ.get("GEMINI_API_KEY")
    if key:
        gemini_items = extract_with_gemini(file_bytes, filename, mime_type or "application/pdf", key)
        if gemini_items:
            return {
                "extraction_method": "gemini_vision",
                "extracted_tests": [item.to_dict() for item in gemini_items],
                "raw_text": f"[Extracted via Gemini Vision multimodal model with key from env]",
                "unmatched_lines": [],
                "count": len(gemini_items),
            }

    # 2. Local extraction (PyMuPDF text layer or local Tesseract OCR)
    raw_text, method = extract_text_from_bytes(file_bytes, filename, mime_type)
    extracted_items, unmatched = parse_lab_text(raw_text)

    return {
        "extraction_method": method,
        "extracted_tests": [item.to_dict() for item in extracted_items],
        "raw_text": raw_text[:2000] if raw_text else "",
        "unmatched_lines": unmatched,
        "count": len(extracted_items),
    }
