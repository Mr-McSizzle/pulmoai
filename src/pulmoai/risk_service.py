"""NHANES Chronic Condition Risk Inference Service.

Loads calibrated models from models/nhanes_risk_models.joblib and computes
individualized disease risk probabilities from user inputs and confirmed lab results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

logger = logging.getLogger("pulmoai.risk_service")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "nhanes_risk_models.joblib"

CONDITION_METADATA = {
    "diabetes": {
        "display_name": "Type 2 Diabetes",
        "description": "Risk of diagnosed Type 2 diabetes (phenotypic prediction with HbA1c/glucose excluded to prevent circularity).",
        "icon": "activity",
    },
    "hypertension": {
        "display_name": "Essential Hypertension",
        "description": "Risk of persistent high blood pressure / chronic hypertension.",
        "icon": "heart",
    },
    "high_cholesterol": {
        "display_name": "Hypercholesterolemia",
        "description": "Risk of dyslipidemia and high blood cholesterol.",
        "icon": "droplet",
    },
    "heart_disease": {
        "display_name": "Cardiovascular Disease",
        "description": "Risk of coronary heart disease, angina, or myocardial infarction.",
        "icon": "shield-alert",
    },
    "stroke": {
        "display_name": "Cerebrovascular Disease (Stroke)",
        "description": "Risk of cerebrovascular event / ischemic stroke.",
        "icon": "alert-circle",
    },
    "copd": {
        "display_name": "Chronic Obstructive Pulmonary Disease (COPD)",
        "description": "Risk of chronic bronchitis, emphysema, or irreversible airflow obstruction.",
        "icon": "wind",
    },
}


class HealthRiskService:
    """Inference engine for calibrated NHANES chronic condition risk prediction."""

    def __init__(self, model_bundle: dict[str, Any], model_path: Path):
        self.model_bundle = model_bundle
        self.model_path = model_path
        self.available_targets = list(model_bundle.keys())
        logger.info(
            "Initialized HealthRiskService with targets: %s",
            ", ".join(self.available_targets),
        )

    @classmethod
    def load(cls, path: Path | None = None) -> HealthRiskService:
        target_path = path or DEFAULT_MODEL_PATH
        if not target_path.exists():
            raise FileNotFoundError(
                f"Trained NHANES risk models bundle not found at {target_path}. "
                "Ensure scripts/train_nhanes_risk_models.py has completed."
            )
        bundle = joblib.load(target_path)
        return cls(bundle, target_path)

    def predict_risk(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Generate calibrated probability risk estimates across all 6 chronic conditions.

        Inputs dict can contain any subset of:
          - Demographics: age, sex_male (1/0), race_ethnicity (1-7), education (1-5), poverty_ratio (0-5)
          - Anthropometrics: weight_kg, height_cm, bmi, waist_cm
          - Vitals: systolic_bp, diastolic_bp, pulse_bpm
          - Labs: total_cholesterol, hdl_cholesterol, triglycerides, ldl_cholesterol,
                  serum_creatinine, bun, uric_acid, albumin, total_protein,
                  alt, ast, alk_phosphatase, calcium, phosphorus, sodium,
                  potassium, chloride, bicarbonate, wbc_count, rbc_count,
                  hemoglobin, hematocrit, platelets, mcv, mch, mchc, rdw,
                  hba1c, fasting_glucose
          - Lifestyle: smoking_status (0=never, 1=former, 2=current), drinks_per_day, physically_active (1/0)
        """
        # Auto-compute BMI if height and weight are provided but bmi is not
        user_inputs = dict(inputs)
        if "bmi" not in user_inputs or user_inputs["bmi"] is None:
            w = user_inputs.get("weight_kg")
            h = user_inputs.get("height_cm")
            if w is not None and h is not None and float(h) > 0:
                h_m = float(h) / 100.0
                user_inputs["bmi"] = round(float(w) / (h_m * h_m), 1)

        results: dict[str, Any] = {}
        contributing_factors = self._identify_contributing_factors(user_inputs)

        for target_id, target_data in self.model_bundle.items():
            feature_names: list[str] = target_data["feature_names"]
            model = target_data["clinical_model"]
            metrics = target_data.get("metrics", {})

            # Build 1-sample 2D feature array with np.nan for unmeasured features
            feat_vector = np.full((1, len(feature_names)), np.nan, dtype=np.float64)
            for col_idx, f_name in enumerate(feature_names):
                val = user_inputs.get(f_name)
                if val is not None:
                    try:
                        feat_vector[0, col_idx] = float(val)
                    except (ValueError, TypeError):
                        pass

            # Predict calibrated probability
            try:
                proba = model.predict_proba(feat_vector)[0, 1]
                calibrated_risk = round(float(proba), 4)
            except Exception as e:
                logger.warning("Prediction error for %s: %s; falling back to NaN", target_id, e)
                calibrated_risk = None

            # Categorize clinical risk level
            if calibrated_risk is not None:
                if calibrated_risk < 0.15:
                    risk_category = "Low"
                elif calibrated_risk < 0.35:
                    risk_category = "Moderate"
                elif calibrated_risk < 0.60:
                    risk_category = "Elevated"
                else:
                    risk_category = "High"
            else:
                risk_category = "Indeterminate"

            meta = CONDITION_METADATA.get(target_id, {})
            results[target_id] = {
                "target_id": target_id,
                "display_name": meta.get("display_name", target_id.capitalize()),
                "description": meta.get("description", ""),
                "calibrated_risk": calibrated_risk,
                "risk_category": risk_category,
                "model_auroc": metrics.get("auroc"),
                "model_brier_score": metrics.get("brier_score"),
                "test_prevalence": metrics.get("prevalence"),
            }

        # Count how many features were supplied
        non_null_count = sum(1 for v in user_inputs.values() if v is not None)

        return {
            "risk_predictions": results,
            "contributing_factors": contributing_factors,
            "features_provided_count": non_null_count,
            "disclaimer": (
                "Exploratory statistical health screening only. This model estimates population-level "
                "risk calibrated from CDC NHANES cohorts. It is NOT a medical diagnosis, clinical prognosis, "
                "or treatment plan. Always consult a licensed healthcare professional for medical advice."
            ),
        }

    def _identify_contributing_factors(self, inputs: dict[str, Any]) -> list[str]:
        """Detect abnormal clinical or lifestyle drivers provided by the user."""
        factors: list[str] = []

        # Age
        age = inputs.get("age")
        if age is not None and float(age) >= 60:
            factors.append(f"Age {int(age)} years (advanced demographic risk band)")

        # BMI
        bmi = inputs.get("bmi")
        if bmi is not None:
            bmi_f = float(bmi)
            if bmi_f >= 30.0:
                factors.append(f"Elevated BMI ({bmi_f:.1f} kg/m² - Obese category)")
            elif bmi_f >= 25.0:
                factors.append(f"Overweight BMI ({bmi_f:.1f} kg/m²)")

        # Blood Pressure
        sys_bp = inputs.get("systolic_bp")
        dia_bp = inputs.get("diastolic_bp")
        if sys_bp is not None and float(sys_bp) >= 130:
            factors.append(f"Elevated systolic blood pressure ({float(sys_bp):.0f} mmHg)")
        if dia_bp is not None and float(dia_bp) >= 80:
            factors.append(f"Elevated diastolic blood pressure ({float(dia_bp):.0f} mmHg)")

        # Lipids
        tc = inputs.get("total_cholesterol")
        if tc is not None and float(tc) >= 200:
            factors.append(f"High total cholesterol ({float(tc):.0f} mg/dL)")

        hdl = inputs.get("hdl_cholesterol")
        if hdl is not None and float(hdl) < 40:
            factors.append(f"Low protective HDL cholesterol ({float(hdl):.0f} mg/dL)")

        tg = inputs.get("triglycerides")
        if tg is not None and float(tg) >= 150:
            factors.append(f"Elevated fasting triglycerides ({float(tg):.0f} mg/dL)")

        # Glycemic
        fpg = inputs.get("fasting_glucose")
        if fpg is not None and float(fpg) >= 100:
            factors.append(f"Impaired fasting glucose ({float(fpg):.0f} mg/dL)")

        a1c = inputs.get("hba1c")
        if a1c is not None and float(a1c) >= 5.7:
            factors.append(f"Elevated HbA1c ({float(a1c):.1f}%)")

        # Smoking
        smk = inputs.get("smoking_status")
        if smk is not None:
            if float(smk) == 2.0:
                factors.append("Current tobacco smoker (active respiratory & cardiovascular risk)")
            elif float(smk) == 1.0:
                factors.append("Former tobacco smoker")

        # Physical activity
        act = inputs.get("physically_active")
        if act is not None and float(act) == 0.0:
            factors.append("Sedentary lifestyle (reports lack of moderate/vigorous activity)")

        return factors
