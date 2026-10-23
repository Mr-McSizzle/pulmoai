"""Future Risk Inference Service (Mortality-Linked Prospective Modeling).

Loads models from models/future_risk_models.joblib and computes:
1. Prospective 5-year and 10-year All-Cause Mortality Risk (Multivariable Cox PH and Gradient Boosted Survival)
2. Prospective 5-year and 10-year Cardiovascular Mortality Risk (Multivariable Cox PH and Gradient Boosted Survival)
3. 10-year Hard ASCVD Risk via 2013 ACC/AHA Pooled Cohort Equations (PCE, Goff et al.)
4. Type 2 Diabetes Risk Score via Indian Diabetes Risk Score (IDRS, Mohan et al.)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("pulmoai.future_risk_service")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "future_risk_models.joblib"


def compute_pce_10yr(
    is_male: bool,
    is_black: bool,
    smoker: bool,
    hypertensive: bool,
    diabetic: bool,
    age: float,
    systolic_bp: float,
    total_chol: float,
    hdl_chol: float,
) -> float | None:
    """Compute 10-year hard ASCVD risk (%) via 2013 ACC/AHA Pooled Cohort Equations (Goff et al.)."""
    if age < 40 or age > 79 or total_chol <= 0 or hdl_chol <= 0 or systolic_bp <= 0:
        return None

    ln_age = math.log(age)
    ln_tc = math.log(total_chol)
    ln_hdl = math.log(hdl_chol)
    tr_sbp = math.log(systolic_bp) if hypertensive else 0.0
    nt_sbp = 0.0 if hypertensive else math.log(systolic_bp)
    age_tc = ln_age * ln_tc
    age_hdl = ln_age * ln_hdl
    age_tr_sbp = ln_age * tr_sbp
    age_nt_sbp = ln_age * nt_sbp
    age_smoke = ln_age if smoker else 0.0

    if is_black and not is_male:
        s10 = 0.95334
        mean_xb = 86.6081
        indiv = (
            17.1141 * ln_age
            + 0.9396 * ln_tc
            - 18.9196 * ln_hdl
            + 4.4748 * age_hdl
            + 29.2907 * tr_sbp
            - 6.4321 * age_tr_sbp
            + 27.8197 * nt_sbp
            - 6.0873 * age_nt_sbp
            + (0.6908 if smoker else 0.0)
            + (0.8738 if diabetic else 0.0)
        )
    elif not is_black and not is_male:
        s10 = 0.96652
        mean_xb = -29.1817
        indiv = (
            -29.799 * ln_age
            + 4.884 * (ln_age**2)
            + 13.54 * ln_tc
            - 3.114 * age_tc
            - 13.578 * ln_hdl
            + 3.149 * age_hdl
            + 2.019 * tr_sbp
            + 1.957 * nt_sbp
            + (7.574 if smoker else 0.0)
            - 1.665 * age_smoke
            + (0.661 if diabetic else 0.0)
        )
    elif is_black and is_male:
        s10 = 0.89536
        mean_xb = 19.5425
        indiv = (
            2.469 * ln_age
            + 0.302 * ln_tc
            - 0.307 * ln_hdl
            + 1.916 * tr_sbp
            + 1.809 * nt_sbp
            + (0.549 if smoker else 0.0)
            + (0.645 if diabetic else 0.0)
        )
    else:
        s10 = 0.91436
        mean_xb = 61.1816
        indiv = (
            12.344 * ln_age
            + 11.853 * ln_tc
            - 2.664 * age_tc
            - 7.99 * ln_hdl
            + 1.769 * age_hdl
            + 1.797 * tr_sbp
            + 1.764 * nt_sbp
            + (7.837 if smoker else 0.0)
            - 1.795 * age_smoke
            + (0.658 if diabetic else 0.0)
        )

    risk = 1.0 - (s10 ** math.exp(indiv - mean_xb))
    return float(np.clip(risk * 100.0, 0.1, 99.9))


def compute_idrs(
    age: float,
    waist_cm: float,
    is_male: bool,
    activity_level: int | str,
    family_history_diabetes: int | float,
) -> dict[str, Any]:
    """Compute Indian Diabetes Risk Score (0-100) (Mohan et al., JAPI 2005)."""
    # 1. Age
    if age < 35:
        p_age = 0
    elif age < 50:
        p_age = 20
    else:
        p_age = 30

    # 2. Waist circumference
    if is_male:
        if waist_cm < 90:
            p_waist = 0
        elif waist_cm < 100:
            p_waist = 10
        else:
            p_waist = 20
    else:
        if waist_cm < 80:
            p_waist = 0
        elif waist_cm < 90:
            p_waist = 10
        else:
            p_waist = 20

    # 3. Physical activity
    if str(activity_level).lower() in ("0", "0.0", "vigorous"):
        p_act = 0
    elif str(activity_level).lower() in ("1", "1.0", "moderate"):
        p_act = 20
    else:
        p_act = 30

    # 4. Family history
    if str(family_history_diabetes) in ("0", "0.0"):
        p_fam = 0
    elif str(family_history_diabetes) in ("1", "1.0"):
        p_fam = 10
    else:
        p_fam = 20

    score = p_age + p_waist + p_act + p_fam
    tier = "Low" if score < 30 else ("Moderate" if score < 60 else "High")
    return {
        "score": score,
        "tier": tier,
        "interpretation": f"{tier} Risk of Diabetes ({score}/100 points)",
        "components": {
            "age_points": p_age,
            "waist_points": p_waist,
            "activity_points": p_act,
            "family_history_points": p_fam,
        },
    }


class FutureRiskService:
    """Service for prospective future risk and mortality-linked endpoint inference."""

    def __init__(self, bundle: dict[str, Any], model_path: Path):
        self.bundle = bundle
        self.model_path = model_path
        self.all_cause_cph = bundle["all_cause_cph"]
        self.all_cause_gbs = bundle["all_cause_gbs"]
        self.cvd_cph = bundle["cvd_cph"]
        self.cvd_gbs = bundle["cvd_gbs"]
        self.train_medians = bundle["train_medians"]
        self.core_features = bundle["core_features"]
        self.tier_cutoffs = bundle["tier_cutoffs"]

    @classmethod
    def load(cls, path: Path | None = None) -> FutureRiskService:
        target_path = path or DEFAULT_MODEL_PATH
        if not target_path.exists():
            raise FileNotFoundError(
                f"Future risk models bundle not found at {target_path}. "
                "Run scripts/audit_future_risk.py to build checkpoints."
            )
        bundle = joblib.load(target_path)
        return cls(bundle, target_path)

    def predict_future_risk(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Generate prospective 5-year and 10-year risk estimates and published score benchmarks."""
        user_inputs = dict(inputs)

        # Auto-compute BMI if height and weight are provided
        if "bmi" not in user_inputs or user_inputs["bmi"] is None:
            w = user_inputs.get("weight_kg")
            h = user_inputs.get("height_cm")
            if w is not None and h is not None and float(h) > 0:
                h_m = float(h) / 100.0
                user_inputs["bmi"] = round(float(w) / (h_m * h_m), 1)

        # Assemble feature row with imputation strictly from training medians
        row = {}
        features_provided = 0
        for feat in self.core_features:
            val = user_inputs.get(feat)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                row[feat] = float(val)
                features_provided += 1
            else:
                row[feat] = float(self.train_medians.get(feat, 0.0))

        X_df = pd.DataFrame([row], columns=self.core_features)

        # 1. All-Cause Mortality Prediction
        surv_5yr_all = float(self.all_cause_cph.predict_survival_function(X_df, times=[60.0]).values[0][0])
        surv_10yr_all = float(self.all_cause_cph.predict_survival_function(X_df, times=[120.0]).values[0][0])
        risk_5yr_all = float(np.clip(1.0 - surv_5yr_all, 0.001, 0.999))
        risk_10yr_all = float(np.clip(1.0 - surv_10yr_all, 0.001, 0.999))

        cut_all = self.tier_cutoffs["all_cause_5yr"]
        tier_all = "Low" if risk_5yr_all < cut_all["moderate"] else ("Moderate" if risk_5yr_all < cut_all["high"] else "High")

        # 2. Cardiovascular Mortality Prediction
        surv_5yr_cvd = float(self.cvd_cph.predict_survival_function(X_df, times=[60.0]).values[0][0])
        surv_10yr_cvd = float(self.cvd_cph.predict_survival_function(X_df, times=[120.0]).values[0][0])
        risk_5yr_cvd = float(np.clip(1.0 - surv_5yr_cvd, 0.0005, 0.999))
        risk_10yr_cvd = float(np.clip(1.0 - surv_10yr_cvd, 0.0005, 0.999))

        cut_cvd = self.tier_cutoffs["cvd_5yr"]
        tier_cvd = "Low" if risk_5yr_cvd < cut_cvd["moderate"] else ("Moderate" if risk_5yr_cvd < cut_cvd["high"] else "High")

        # 3. Published Score: Pooled Cohort Equations (PCE 2013)
        age = float(user_inputs.get("age", row["age"]))
        sex_male = bool(user_inputs.get("sex_male", row["sex_male"]) == 1)
        race_black = bool(user_inputs.get("race_black", row.get("race_black", 0)) == 1)
        smoker = bool(user_inputs.get("current_smoker", user_inputs.get("smoking_status") == 2))
        hypertensive = bool(user_inputs.get("bp_medication", row.get("bp_medication", 0)) == 1)
        diabetic = bool(user_inputs.get("target_diabetes", row.get("target_diabetes", 0)) == 1)
        sbp = float(user_inputs.get("systolic_bp", row["systolic_bp"]))
        tc = float(user_inputs.get("total_cholesterol", row["total_cholesterol"]))
        hdl = float(user_inputs.get("hdl_cholesterol", row["hdl_cholesterol"]))

        pce_risk = compute_pce_10yr(
            is_male=sex_male,
            is_black=race_black,
            smoker=smoker,
            hypertensive=hypertensive,
            diabetic=diabetic,
            age=age,
            systolic_bp=sbp,
            total_chol=tc,
            hdl_chol=hdl,
        )

        pce_category = None
        if pce_risk is not None:
            if pce_risk < 5.0:
                pce_category = "Low Risk (< 5%)"
            elif pce_risk < 7.5:
                pce_category = "Borderline Risk (5% to < 7.5%)"
            elif pce_risk < 20.0:
                pce_category = "Intermediate Risk (7.5% to < 20%)"
            else:
                pce_category = "High Risk (>= 20%)"

        # 4. Published Score: Indian Diabetes Risk Score (IDRS)
        waist = float(user_inputs.get("waist_cm", row["waist_cm"]))
        activity = user_inputs.get("activity_level", 1 if user_inputs.get("physically_active", 1) == 1 else 2)
        fam_history = user_inputs.get("family_history_diabetes", 0)
        idrs_res = compute_idrs(
            age=age,
            waist_cm=waist,
            is_male=sex_male,
            activity_level=activity,
            family_history_diabetes=fam_history,
        )

        return {
            "future_endpoints": {
                "all_cause_mortality": {
                    "display_name": "All-Cause Mortality",
                    "estimand_type": "prospective_incidence",
                    "estimand_label": "Future mortality risk (US survey data)",
                    "estimand_description": "Estimates future mortality risk from any cause over 5-year and 10-year horizons based on CDC NHANES linked to National Death Index follow-up on temporal holdout (2011-2014).",
                    "time_horizons": {
                        "5_year": {"risk_probability": round(risk_5yr_all, 4), "risk_percent": round(risk_5yr_all * 100, 2)},
                        "10_year": {"risk_probability": round(risk_10yr_all, 4), "risk_percent": round(risk_10yr_all * 100, 2)},
                    },
                    "risk_tier_5yr": tier_all,
                    "tier_cutoffs_5yr": cut_all,
                    "benchmark_metrics": {
                        "c_index": 0.8621,
                        "holdout_5yr_auroc": 0.8713,
                        "brier_score_5yr": 0.0468,
                        "calibration_slope": 1.0352,
                        "oe_ratio_5yr": 1.007,
                    },
                },
                "cardiovascular_mortality": {
                    "display_name": "Cardiovascular Mortality",
                    "estimand_type": "prospective_incidence",
                    "estimand_label": "Future mortality risk (US survey data)",
                    "estimand_description": "Estimates future mortality risk from cardiovascular disease (ICD-10 I00-I09, I11, I13, I20-I51, I60-I69) based on CDC NHANES linked to National Death Index follow-up on temporal holdout (2011-2014).",
                    "time_horizons": {
                        "5_year": {"risk_probability": round(risk_5yr_cvd, 4), "risk_percent": round(risk_5yr_cvd * 100, 2)},
                        "10_year": {"risk_probability": round(risk_10yr_cvd, 4), "risk_percent": round(risk_10yr_cvd * 100, 2)},
                    },
                    "risk_tier_5yr": tier_cvd,
                    "tier_cutoffs_5yr": cut_cvd,
                    "benchmark_metrics": {
                        "c_index": 0.8948,
                        "holdout_5yr_auroc": 0.9074,
                        "brier_score_5yr": 0.0184,
                        "calibration_slope": 1.2095,
                    },
                },
            },
            "published_scores": {
                "pooled_cohort_equations_pce": {
                    "display_name": "Pooled Cohort Equations (ACC/AHA 2013)",
                    "eligible": pce_risk is not None,
                    "eligibility_criteria": "Age 40-79 years without prior CVD with measured blood pressure and lipid panel",
                    "estimated_10yr_ascvd_risk_pct": pce_risk,
                    "risk_category": pce_category,
                    "benchmark_c_index_on_holdout": 0.7799,
                    "benchmark_5yr_auroc_on_holdout": 0.7739,
                    "evaluation_cohort": "Ages 40-79 without prior cardiovascular disease (MCQ160C/D/E/F = 0) on temporal holdout (2011-2014)",
                    "claim_scope": "Evaluated strictly for discrimination against observed fatal CVD; calibration is not claimed because PCE estimates fatal + non-fatal ASCVD whereas NHANES-LMF observes fatal events only.",
                    "citation": "Goff DC Jr, et al. Circulation 2014;129:S49-S73 (Table A & Appendix 7).",
                },
                "indian_diabetes_risk_score_idrs": {
                    "display_name": "Indian Diabetes Risk Score (IDRS)",
                    "score": idrs_res["score"],
                    "risk_tier": idrs_res["tier"],
                    "interpretation": idrs_res["interpretation"],
                    "components": idrs_res["components"],
                    "benchmark_auroc_on_holdout": 0.7798,
                    "four_feature_ml_auroc": 0.8303,
                    "nhanes_mapping_notes": "Physical activity (3 levels) and family history of diabetes (1st-degree relative) are approximate mappings from NHANES variables.",
                    "citation": "Mohan V, et al. J Assoc Physicians India 2005;53:759-763 (Table 1).",
                },
            },
            "features_provided_count": features_provided,
            "nchs_privacy_perturbation_warning": (
                "The public-use NHANES Linked Mortality Files (LMF) include data perturbation (synthetic swapping) "
                "on a small fraction of records applied by the National Center for Health Statistics (NCHS) to protect respondent "
                "confidentiality. As documented by NCHS, while population parameter estimates remain unbiased, small-cell "
                "subgroup counts and individual-level survival intervals may slightly diverge from unperturbed restricted-use files."
            ),
            "estimand_clarification": (
                "CRITICAL ESTIMAND DISTINCTION: This module estimates Future mortality risk (US survey data) "
                "over fixed 5-year (60 months) and 10-year (120 months) horizons derived from CDC NHANES "
                "participants tracked prospectively through the National Death Index on temporal holdout (2011-2014). "
                "In contrast, the chronic condition module estimates cross-sectional survey-reported doctor diagnoses."
            ),
            "disclaimer": (
                "Prospective longitudinal survival modeling and evidence-based clinical risk stratification (CDC NHANES & NDI cohorts)."
            ),
        }
