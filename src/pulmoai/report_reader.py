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
            "fasting blood sugar", "blood sugar", "fasting plasma glucose",
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
            "a1c", "hba1c (ngsp)", "glycated hb", "glycosylated hemoglobin",
            "glycosylated hemoglobin (hba1c)", "glycosylated hemoglobin hba1c",
            "glycohemoglobin hba1c",
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
            "hdl cholesterol - direct", "hdl cholesterol direct",
            "hdl - direct", "hdl direct", "hdl-cholesterol - direct",
            "hdl - cholesterol",
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
            "fasting triglycerides", "tg", "tgl",
        ],
        ref_min=50.0,
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


def get_reference_range(
    canonical_name: str,
    sex: str | None = None,
    printed_min: float | None = None,
    printed_max: float | None = None,
    printed_str: str | None = None,
) -> tuple[float | None, float | None, str]:
    """Get clinical reference limits, prioritizing printed ranges and applying sex-specific defaults."""
    s = (sex or "").lower().strip()
    is_male = s in {"male", "m", "1", "man"}
    is_female = s in {"female", "f", "0", "woman"}

    # Base sex-specific defaults
    def_min: float | None = None
    def_max: float | None = None
    def_str: str = ""

    if canonical_name == "hemoglobin":
        def_min, def_max = (13.5, 17.5) if is_male else ((12.0, 15.5) if is_female else (12.0, 17.5))
        def_str = f"{def_min} - {def_max} g/dL"
    elif canonical_name == "serum_creatinine":
        def_min, def_max = (0.7, 1.3) if is_male else ((0.5, 1.1) if is_female else (0.6, 1.3))
        def_str = f"{def_min} - {def_max} mg/dL"
    elif canonical_name == "hdl_cholesterol":
        def_min = 40.0 if is_male else (50.0 if is_female else 40.0)
        def_max = 100.0
        def_str = f">= {int(def_min)} mg/dL"
    elif canonical_name == "total_cholesterol":
        def_min, def_max = 100.0, 199.0
        def_str = "< 200 mg/dL"
    elif canonical_name == "triglycerides":
        def_min, def_max = 50.0, 149.0
        def_str = "< 150 mg/dL"
    else:
        t_def = NAME_TO_DEF.get(canonical_name)
        if t_def:
            def_min, def_max, def_str = t_def.ref_min, t_def.ref_max, t_def.ref_str

    if printed_str:
        # Complement one-sided printed ranges with physiological boundaries
        eff_min = printed_min if printed_min is not None else def_min
        eff_max = printed_max if printed_max is not None else def_max
        return eff_min, eff_max, printed_str

    return def_min, def_max, def_str


def determine_flag(
    canonical_name: str,
    val: float,
    explicit_flag_text: str = "",
    ref_min: float | None = None,
    ref_max: float | None = None,
    sex: str | None = None,
) -> Literal["NORMAL", "HIGH", "LOW", "UNKNOWN"]:
    """Determine whether an extracted lab result is high, low, or normal."""
    explicit_clean = explicit_flag_text.strip().upper()
    if explicit_clean in {"H", "HIGH", "*H*", "CRITICAL HIGH"}:
        return "HIGH"
    if explicit_clean in {"L", "LOW", "*L*", "CRITICAL LOW"}:
        return "LOW"

    if ref_min is None or ref_max is None:
        eff_min, eff_max, _ = get_reference_range(
            canonical_name, sex=sex, printed_min=ref_min, printed_max=ref_max
        )
        if ref_min is None:
            ref_min = eff_min
        if ref_max is None:
            ref_max = eff_max

    if ref_max is not None and val > ref_max:
        return "HIGH"
    if ref_min is not None and val < ref_min:
        return "LOW"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Text Extraction: Digital PDF, Tesseract OCR, and Gemini
# ---------------------------------------------------------------------------

def preprocess_lab_image_adaptive(pil_img: Image.Image) -> tuple[Image.Image, str]:
    """Adaptive per-image preprocessing: selects raw vs enhanced processing based on blur and skew.

    Returns:
        (processed_image, decision_rationale)
    """
    import cv2
    import numpy as np

    arr = np.array(pil_img.convert("L"))
    lap_var = float(cv2.Laplacian(arr, cv2.CV_64F).var())

    # 1. Blur detection: if heavily blurred (defocused), Otsu binarization
    # thins characters into broken strokes. Preserve raw image.
    if lap_var < 50.0:
        return pil_img, f"blur_defocus_detected (var={lap_var:.1f}); raw image preserved"

    # 2. Skew detection: sweep angles [-5, +5] degrees via horizontal projection variance
    h, w = arr.shape
    center = (w // 2, h // 2)
    best_score, best_angle = -1.0, 0.0
    for angle in np.arange(-5.0, 5.5, 0.5):
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(
            arr, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255
        )
        score = float(np.var(np.sum(rotated < 128, axis=1)))
        if score > best_score:
            best_score, best_angle = score, float(angle)

    # If significant tilt (>= 1.0 degree), deskew and enhance
    if abs(best_angle) >= 1.0:
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        deskewed = cv2.warpAffine(
            arr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255
        )
        up_w, up_h = int(w * (300 / 150)), int(h * (300 / 150))
        upscaled = cv2.resize(deskewed, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
        denoised = cv2.medianBlur(upscaled, 3)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(thresh), f"deskewed_and_enhanced (angle={best_angle:+.1f} deg)"

    # If upright and unblurred (e.g. clean OCR or JPEG compression), preserve raw to avoid DCT artifacts
    return pil_img, "upright_unblurred; raw preserved"


# ---------------------------------------------------------------------------
# Text Extraction: Digital PDF, Tesseract OCR, and Gemini
# ---------------------------------------------------------------------------

def preprocess_lab_image(pil_img: Image.Image) -> Image.Image:
    """Robust document image preprocessing pipeline:
    1. Rescale to >= 300 DPI equivalent (width >= 2400) via Bicubic resampling
    2. Deskew using horizontal projection profile variance optimization
    3. Median blur denoising to suppress noise and artifact speckles
    4. Adaptive Otsu thresholding / binarization for crisp character boundaries
    """
    try:
        import cv2
        import numpy as np

        # 1. Rescale to >= 300 DPI equivalent
        w, h = pil_img.size
        if w < 2200:
            scale = 2400.0 / float(w)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), resample=Image.Resampling.BICUBIC)

        gray = np.array(pil_img.convert("L"))

        # 2. Deskew via projection profile variance
        h_arr, w_arr = gray.shape
        thumb_h = 600
        thumb_w = int(w_arr * (thumb_h / float(h_arr)))
        thumb = cv2.resize(gray, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        best_var = -1.0
        best_angle = 0.0
        # Search coarse angles from -15 to +15 in 0.5 degree steps
        for a in np.arange(-15.0, 15.5, 0.5):
            M = cv2.getRotationMatrix2D((thumb_w / 2.0, thumb_h / 2.0), a, 1.0)
            rot = cv2.warpAffine(thumb, M, (thumb_w, thumb_h), flags=cv2.INTER_LINEAR, borderValue=255)
            inv = 255 - rot
            proj = np.sum(inv, axis=1)
            v = float(np.var(proj))
            if v > best_var:
                best_var = v
                best_angle = a

        # Fine tune within +/- 0.5 degrees in 0.1 degree steps
        fine_best_angle = best_angle
        fine_best_var = best_var
        for a in np.arange(best_angle - 0.5, best_angle + 0.6, 0.1):
            M = cv2.getRotationMatrix2D((thumb_w / 2.0, thumb_h / 2.0), a, 1.0)
            rot = cv2.warpAffine(thumb, M, (thumb_w, thumb_h), flags=cv2.INTER_LINEAR, borderValue=255)
            inv = 255 - rot
            proj = np.sum(inv, axis=1)
            v = float(np.var(proj))
            if v > fine_best_var:
                fine_best_var = v
                fine_best_angle = a

        if abs(fine_best_angle) >= 0.2:
            M_full = cv2.getRotationMatrix2D((w_arr / 2.0, h_arr / 2.0), fine_best_angle, 1.0)
            gray = cv2.warpAffine(gray, M_full, (w_arr, h_arr), flags=cv2.INTER_CUBIC, borderValue=255)

        # 3. Denoising
        denoised = cv2.medianBlur(gray, 3)

        # 4. Adaptive Otsu thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return Image.fromarray(thresh)
    except Exception as e:
        logger.warning("Image preprocessing fallback: %s", e)
        return pil_img


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

            # If digital layer is empty/sparse, render pages at 300 DPI and run local Tesseract OCR with preprocessing
            logger.info("PDF has minimal text layer; falling back to rendering page for OCR")
            ocr_text_list = []
            import pytesseract
            for page in doc:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                preprocessed_img, _ = preprocess_lab_image_adaptive(img)
                page_ocr = pytesseract.image_to_string(preprocessed_img)
                if page_ocr:
                    ocr_text_list.append(page_ocr)
            ocr_combined = "\n".join(ocr_text_list).strip()
            if ocr_combined:
                return ocr_combined, "tesseract_ocr"
        except Exception as e:
            logger.warning("PDF extraction error: %s; falling back to OCR", e)

    # Step 2: Image or fallback OCR with robust adaptive preprocessing
    try:
        import pytesseract
        raw_img = Image.open(io.BytesIO(file_bytes))
        preprocessed_img, _ = preprocess_lab_image_adaptive(raw_img)
        ocr_text = pytesseract.image_to_string(preprocessed_img)
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
    clean = re.sub(r"[^a-zA-Z0-9\s/]", " ", name_candidate.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    clean_no_hyphen = re.sub(r"^[\+\-\*•\s]+", "", clean).strip()

    # Pre-sort all aliases by descending character length so specific multi-word phrases match first
    sorted_pairs: list[tuple[str, CanonicalTestDef]] = []
    for t_def in CANONICAL_TESTS:
        for alias in t_def.aliases:
            sorted_pairs.append((alias.lower().strip(), t_def))
    sorted_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    # 1. Exact alias match
    for alias, t_def in sorted_pairs:
        if clean == alias or clean_no_hyphen == alias:
            return t_def

    # 2. Starts with or contains alias as isolated token (longer aliases first)
    for alias, t_def in sorted_pairs:
        if re.search(r"\b" + re.escape(alias) + r"\b", clean) or re.search(
            r"\b" + re.escape(alias) + r"\b", clean_no_hyphen
        ):
            return t_def

    return None


def parse_lab_text(text: str, sex: str | None = None) -> tuple[list[ExtractedTestItem], list[str]]:
    """Parse unstructured report text into standardized lab entries.

    Prioritizes printed reference intervals from the report when present,
    and applies sex-specific clinical reference ranges for Hemoglobin, Creatinine, and HDL.

    Returns:
        (extracted_items, unmatched_lines)
    """
    extracted: dict[str, ExtractedTestItem] = {}
    unmatched: list[str] = []

    # Detect patient biological sex if not explicitly provided
    if sex is None:
        sex_m = re.search(r"\b(?:sex|gender)\s*[:=\s]\s*(male|female|m|f)\b", text, re.IGNORECASE)
        if sex_m:
            s_raw = sex_m.group(1).lower()
            sex = "male" if s_raw in {"male", "m"} else "female"

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
                        flag=determine_flag("systolic_bp", sys_val, ref_min=90.0, ref_max=120.0),
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
                        flag=determine_flag("diastolic_bp", dia_val, ref_min=60.0, ref_max=80.0),
                        confidence=0.95,
                    )

    # 2. General line parser: [Test Name] ... [Numeric Result] ... [Flag] ... [Units] ... [Reference]
    for line in lines:
        # Skip header/footer noise
        if any(skip in line.lower() for skip in ["patient name", "date of birth", "physician", "specimen id", "accession", "page "]):
            continue

        clean_line = re.sub(r"^[•\-\*]\s*", "", line).strip()

        match = re.search(
            r"^([A-Za-z0-9\s,/()\-+]+?)[\s:=-]+([><]?\s*\d+(?:\.\d+)?)\s*([A-Za-z*%]*)\s*([A-Za-z0-9/%^µμ\s\-.]+)?",
            clean_line,
        )

        if not match:
            # Try delimiter splitting (tab, comma, multiple spaces)
            parts = re.split(r"\s{2,}|\t|,", clean_line)
            if len(parts) >= 2:
                name_cand = parts[0].strip()
                t_def = match_test_name(name_cand)
                if t_def and t_def.canonical_name not in extracted:
                    for p_idx, p in enumerate(parts[1:], 1):
                        num_m = re.search(r"(\d+(?:\.\d+)?)", p)
                        if num_m:
                            try:
                                raw_val = float(num_m.group(1))
                                unit_cand = parts[p_idx + 1] if p_idx + 1 < len(parts) else ""
                                norm_val, norm_unit, conf = normalize_unit_and_value(t_def.canonical_name, raw_val, unit_cand)

                                # Look for printed reference range in subsequent columns
                                remaining_text = " ".join(parts[p_idx + 1:])
                                ref_match = re.search(
                                    r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*([A-Za-z/%^µμ\s\-.]+)?",
                                    remaining_text,
                                )
                                ref_less = re.search(r"<\s*(\d+(?:\.\d+)?)", remaining_text)
                                ref_greater = re.search(r">\s*(\d+(?:\.\d+)?)", remaining_text)
                                p_min, p_max, p_str = None, None, None
                                p_min_raw, p_max_raw = None, None
                                if ref_match:
                                    p_min_raw = float(ref_match.group(1))
                                    p_max_raw = float(ref_match.group(2))
                                    p_str = f"{p_min_raw} - {p_max_raw}"
                                elif ref_less:
                                    p_max_raw = float(ref_less.group(1))
                                    p_str = f"< {p_max_raw}"
                                elif ref_greater:
                                    p_min_raw = float(ref_greater.group(1))
                                    p_str = f"> {p_min_raw}"

                                if p_min_raw is not None:
                                    p_min = normalize_unit_and_value(t_def.canonical_name, p_min_raw, unit_cand)[0] if unit_cand else p_min_raw
                                if p_max_raw is not None:
                                    p_max = normalize_unit_and_value(t_def.canonical_name, p_max_raw, unit_cand)[0] if unit_cand else p_max_raw

                                eff_min, eff_max, eff_ref_str = get_reference_range(
                                    t_def.canonical_name,
                                    sex=sex,
                                    printed_min=p_min,
                                    printed_max=p_max,
                                    printed_str=p_str,
                                )
                                flag = determine_flag(
                                    t_def.canonical_name,
                                    norm_val,
                                    ref_min=eff_min,
                                    ref_max=eff_max,
                                    sex=sex,
                                )
                                extracted[t_def.canonical_name] = ExtractedTestItem(
                                    raw_name=name_cand,
                                    canonical_name=t_def.canonical_name,
                                    display_name=t_def.display_name,
                                    category=t_def.category,
                                    value=norm_val,
                                    raw_value=str(raw_val),
                                    unit=norm_unit,
                                    raw_unit=unit_cand.strip(),
                                    reference_range=eff_ref_str,
                                    flag=flag,
                                    confidence=conf,
                                )
                                break
                            except ValueError:
                                pass
            continue

        name_part = match.group(1).strip()
        val_str = match.group(2).replace(">", "").replace("<", "").strip()
        after_val = clean_line[match.end(2):].strip()

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

        # Extract explicit abnormal flag (H, HIGH, L, LOW)
        flag_match = re.search(r"(?<![/\w])([HL]|HIGH|LOW|CRITICAL)(?![/\w])", after_val, re.IGNORECASE)
        flag_cand = flag_match.group(1) if flag_match else ""

        # Extract printed reference range from after_val if present
        ref_match = re.search(
            r"(?:(?:ref(?:\.|erence)?\s*(?:range|interval|limits)?[:\s]*)|\(|\b)(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*([A-Za-z/%^µμ\s\-.]+)?(?:\)|\b)",
            after_val,
            re.IGNORECASE,
        )
        ref_less_match = re.search(r"<\s*(\d+(?:\.\d+)?)", after_val)
        ref_greater_match = re.search(r">\s*(\d+(?:\.\d+)?)", after_val)

        p_min, p_max, p_str = None, None, None
        p_min_raw, p_max_raw = None, None
        if ref_match:
            p_min_raw = float(ref_match.group(1))
            p_max_raw = float(ref_match.group(2))
            u_part = (ref_match.group(3) or "").strip()
            p_str = f"{p_min_raw} - {p_max_raw}" + (f" {u_part}" if u_part else "")
        elif ref_less_match:
            p_max_raw = float(ref_less_match.group(1))
            p_str = f"< {p_max_raw}"
        elif ref_greater_match:
            p_min_raw = float(ref_greater_match.group(1))
            p_str = f"> {p_min_raw}"

        norm_val, norm_unit, conf = normalize_unit_and_value(t_def.canonical_name, raw_val, unit_cand)

        if p_min_raw is not None:
            p_min = normalize_unit_and_value(t_def.canonical_name, p_min_raw, unit_cand)[0] if unit_cand else p_min_raw
        if p_max_raw is not None:
            p_max = normalize_unit_and_value(t_def.canonical_name, p_max_raw, unit_cand)[0] if unit_cand else p_max_raw

        eff_min, eff_max, eff_ref_str = get_reference_range(
            t_def.canonical_name,
            sex=sex,
            printed_min=p_min,
            printed_max=p_max,
            printed_str=p_str,
        )

        flag = determine_flag(
            t_def.canonical_name,
            norm_val,
            explicit_flag_text=flag_cand,
            ref_min=eff_min,
            ref_max=eff_max,
            sex=sex,
        )

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
                reference_range=eff_ref_str,
                flag=flag,
                confidence=conf,
            )

    return list(extracted.values()), unmatched[:15]


def process_lab_report(
    file_bytes: bytes,
    filename: str,
    mime_type: str | None = None,
    gemini_key: str | None = None,
    sex: str | None = None,
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
    extracted_items, unmatched = parse_lab_text(raw_text, sex=sex)

    return {
        "extraction_method": method,
        "extracted_tests": [item.to_dict() for item in extracted_items],
        "raw_text": raw_text[:2000] if raw_text else "",
        "unmatched_lines": unmatched,
        "count": len(extracted_items),
    }
