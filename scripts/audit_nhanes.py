#!/usr/bin/env python3
"""Audit and Re-evaluation of NHANES Chronic Condition Risk Engine.

Evaluates the deployed models on the later-cycle temporal holdout (Cycle J: 2017-2018).
Computes unweighted and survey-weighted metrics:
  - AUROC, AUPRC, Prevalence, Accuracy of always predicting negative
  - Brier score, Brier skill score vs prevalence-only model
  - Calibration slope and intercept
  - Diagnostic metrics (sensitivity, specificity, PPV) at threshold derived on training data only
  - Subgroup AUROC across Sex, Age Band, and Race/Ethnicity (including Non-Hispanic Asian)
  - Ablations:
      * Hypertension without measured BP
      * High cholesterol without lipid values
      * Lifestyle and body size only version of each target

Saves results to reports/audit_nhanes.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_nhanes")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nhanes_merged.parquet"
DEMO_J_PATH = PROJECT_ROOT / "data" / "raw" / "nhanes" / "2017-2018" / "DEMO_J.XPT"
DEMO_H_PATH = PROJECT_ROOT / "data" / "raw" / "nhanes" / "2013-2014" / "DEMO_H.XPT"
DEMO_I_PATH = PROJECT_ROOT / "data" / "raw" / "nhanes" / "2015-2016" / "DEMO_I.XPT"
MODEL_PATH = PROJECT_ROOT / "models" / "nhanes_risk_models.joblib"
OUTPUT_REPORT_PATH = PROJECT_ROOT / "reports" / "audit_nhanes.json"

TARGET_SPECS = [
    {
        "id": "diabetes",
        "col": "target_diabetes",
        "name": "Diabetes (Self-reported)",
        "questionnaire_variable": "DIQ010",
        "question_text": "Other than during pregnancy, have you ever been told by a doctor or health professional that you have diabetes or sugar diabetes?",
        "is_self_reported_ever_told": True,
        "distinguishes_type2": False,
        "leakage_exclude": ["hba1c", "fasting_glucose"],
    },
    {
        "id": "hypertension",
        "col": "target_hypertension",
        "name": "Essential Hypertension (Self-reported)",
        "questionnaire_variable": "BPQ020",
        "question_text": "Have you ever been told by a doctor or other health professional that you had hypertension, also called high blood pressure?",
        "is_self_reported_ever_told": True,
        "distinguishes_type2": False,
        "leakage_exclude": [],
    },
    {
        "id": "high_cholesterol",
        "col": "target_high_cholesterol",
        "name": "Hypercholesterolemia (Self-reported)",
        "questionnaire_variable": "BPQ080",
        "question_text": "Have you ever been told by a doctor or other health professional that your blood cholesterol level was high?",
        "is_self_reported_ever_told": True,
        "distinguishes_type2": False,
        "leakage_exclude": [],
    },
    {
        "id": "heart_disease",
        "col": "target_heart_disease",
        "name": "Cardiovascular Disease (Self-reported)",
        "questionnaire_variable": "MCQ160C, MCQ160D, MCQ160E",
        "question_text": "Has a doctor or other health professional ever told you that you had coronary heart disease, angina, or a heart attack?",
        "is_self_reported_ever_told": True,
        "distinguishes_type2": False,
        "leakage_exclude": [],
    },
    {
        "id": "stroke",
        "col": "target_stroke",
        "name": "Cerebrovascular Disease / Stroke (Self-reported)",
        "questionnaire_variable": "MCQ160F",
        "question_text": "Has a doctor or other health professional ever told you that you had a stroke?",
        "is_self_reported_ever_told": True,
        "distinguishes_type2": False,
        "leakage_exclude": [],
    },
    {
        "id": "copd",
        "col": "target_copd",
        "name": "Chronic Respiratory Disease / COPD (Self-reported)",
        "questionnaire_variable": "MCQ160G, MCQ160K",
        "question_text": "Has a doctor or other health professional ever told you that you had emphysema or chronic bronchitis?",
        "is_self_reported_ever_told": True,
        "distinguishes_type2": False,
        "leakage_exclude": [],
    },
]

LIFESTYLE_BODY_FEATURES = [
    # Demographics
    "age", "sex_male", "race_ethnicity", "education", "poverty_ratio",
    # Body Measures
    "weight_kg", "height_cm", "bmi", "waist_cm",
    # Lifestyle
    "smoking_status", "drinks_per_day", "physically_active",
]

RACE_ETHNICITY_MAP = {
    1.0: "Mexican American",
    2.0: "Other Hispanic",
    3.0: "Non-Hispanic White",
    4.0: "Non-Hispanic Black",
    6.0: "Non-Hispanic Asian",
    7.0: "Other Race / Multi-Racial",
}


def fit_calibration_slope_intercept(
    y_true: np.ndarray, y_prob: np.ndarray, sample_weight: np.ndarray | None = None
) -> tuple[float, float]:
    """Fit logistic calibration: logit(p) -> y."""
    eps = 1e-6
    p_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=1000)
    try:
        lr.fit(logit_p, y_true, sample_weight=sample_weight)
        slope = float(lr.coef_[0, 0])
        intercept = float(lr.intercept_[0])
    except Exception as exc:
        logger.warning("Calibration slope/intercept fit failed: %s", exc)
        slope, intercept = 1.0, 0.0
    return round(slope, 4), round(intercept, 4)


def compute_diagnostic_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute sensitivity, specificity, PPV, and accuracy at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    if sample_weight is None:
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        tn = float(np.sum((y_true == 0) & (y_pred == 0)))
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        total = float(len(y_true))
    else:
        w = sample_weight
        tp = float(np.sum(w * ((y_true == 1) & (y_pred == 1))))
        fp = float(np.sum(w * ((y_true == 0) & (y_pred == 1))))
        tn = float(np.sum(w * ((y_true == 0) & (y_pred == 0))))
        fn = float(np.sum(w * ((y_true == 1) & (y_pred == 0))))
        total = float(np.sum(w))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0
    bal_acc = 0.5 * (sens + spec)

    return {
        "threshold": round(threshold, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "ppv": round(ppv, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
    }


def find_optimal_training_threshold(y_train: np.ndarray, p_train: np.ndarray) -> float:
    """Find threshold maximizing Youden's J statistic (TPR - FPR) on training data."""
    fpr, tpr, thresholds = roc_curve(y_train, p_train)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    best_thresh = float(thresholds[best_idx])
    return best_thresh


def evaluate_cohort(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    train_prev: float,
    train_threshold: float,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate binary predictions with full metrics suite."""
    auroc = float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight))
    auprc = float(average_precision_score(y_true, y_prob, sample_weight=sample_weight))
    brier = float(brier_score_loss(y_true, y_prob, sample_weight=sample_weight))

    if sample_weight is None:
        prev = float(np.mean(y_true))
        brier_ref = float(np.mean((train_prev - y_true) ** 2))
    else:
        prev = float(np.average(y_true, weights=sample_weight))
        brier_ref = float(np.average((train_prev - y_true) ** 2, weights=sample_weight))

    brier_skill = 1.0 - (brier / brier_ref) if brier_ref > 0 else 0.0
    acc_always_negative = 1.0 - prev

    slope, intercept = fit_calibration_slope_intercept(y_true, y_prob, sample_weight=sample_weight)
    diag_metrics = compute_diagnostic_metrics(
        y_true, y_prob, threshold=train_threshold, sample_weight=sample_weight
    )

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "prevalence": round(prev, 4),
        "accuracy_always_negative": round(acc_always_negative, 4),
        "brier_score": round(brier, 4),
        "brier_reference_prevalence_only": round(brier_ref, 4),
        "brier_skill_score": round(brier_skill, 4),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "threshold_training_optimal": round(train_threshold, 4),
        "at_training_threshold": diag_metrics,
    }


def train_ablation_model(
    X_train: np.ndarray, y_train: np.ndarray
) -> CalibratedClassifierCV:
    """Train HistGradientBoostingClassifier wrapped in CalibratedClassifierCV."""
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(estimator=hgb, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def main() -> None:
    t_start = time.time()
    logger.info("Starting NHANES Audit Pass on Later-Cycle Temporal Holdout (Cycle J: 2017-2018)...")

    # 1. Load Data
    df = pd.read_parquet(DATA_PATH)
    logger.info("Loaded parquet dataset: %d total respondents", len(df))

    # Load DEMO_J for examination survey weights (WTMEC2YR)
    demo_j = pd.read_sas(DEMO_J_PATH, format="xport")[["SEQN", "WTMEC2YR"]]
    demo_j["SEQN"] = demo_j["SEQN"].astype(int)

    # Load DEMO_H and DEMO_I for training survey weights
    demo_h = pd.read_sas(DEMO_H_PATH, format="xport")[["SEQN", "WTMEC2YR"]]
    demo_h["SEQN"] = demo_h["SEQN"].astype(int)
    demo_i = pd.read_sas(DEMO_I_PATH, format="xport")[["SEQN", "WTMEC2YR"]]
    demo_i["SEQN"] = demo_i["SEQN"].astype(int)
    demo_weights_all = pd.concat([demo_h, demo_i, demo_j], ignore_index=True)

    df = df.merge(demo_weights_all, on="SEQN", how="left")
    logger.info("Merged WTMEC2YR survey weights: %d with valid weights", (df["WTMEC2YR"] > 0).sum())

    # Adults filter
    df_adults = df[df["age"] >= 18].copy()
    train_mask = df_adults["cycle"].isin(["2013-2014", "2015-2016"])
    test_mask = df_adults["cycle"] == "2017-2018"

    logger.info("Adults train (Cycles H+I): %d", train_mask.sum())
    logger.info("Adults test (Cycle J holdout): %d", test_mask.sum())

    # 2. Load Deployed Models
    if not MODEL_PATH.exists():
        logger.error("Deployed model bundle not found at %s", MODEL_PATH)
        sys.exit(1)

    model_bundle = joblib.load(MODEL_PATH)
    logger.info("Loaded deployed model bundle containing %d targets", len(model_bundle))

    audit_output: dict[str, Any] = {
        "audit_metadata": {
            "module_name": "NHANES Chronic Condition Risk Engine",
            "evaluation_date": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "temporal_split_type": "later_cycle_temporal_holdout",
            "training_cycles": ["2013-2014 (Cycle H)", "2015-2016 (Cycle I)"],
            "holdout_cycle": "2017-2018 (Cycle J)",
            "train_sample_adults": int(train_mask.sum()),
            "holdout_sample_adults": int(test_mask.sum()),
            "estimand_clarification": (
                "All outputs estimate cross-sectional prevalence or survey-reported lifetime doctor diagnoses "
                "in adult examination cohorts; they do NOT forecast 10-year prospective future incidence."
            ),
            "type_2_distinction_note": (
                "CDC NHANES question DIQ010 asks whether a doctor has ever diagnosed diabetes; it does not "
                "differentiate Type 1 from Type 2 diabetes. The target is accurately designated 'Diagnosed Diabetes'."
            ),
        },
        "targets": {},
        "summary_table": [],
    }

    for spec in TARGET_SPECS:
        t_id = spec["id"]
        t_col = spec["col"]
        t_name = spec["name"]
        logger.info("Evaluating target: %s (%s)", t_id, t_name)

        target_model_info = model_bundle[t_id]
        feature_names: list[str] = target_model_info["feature_names"]
        clinical_model = target_model_info["clinical_model"]

        # Valid train/test masks for this specific target
        valid_train = train_mask & df_adults[t_col].notna()
        valid_test = test_mask & df_adults[t_col].notna()

        train_data = df_adults[valid_train]
        test_data = df_adults[valid_test]

        y_train = train_data[t_col].values.astype(int)
        y_test = test_data[t_col].values.astype(int)
        X_train = train_data[feature_names].values
        X_test = test_data[feature_names].values

        w_train = train_data["WTMEC2YR"].values.astype(float)
        w_test = test_data["WTMEC2YR"].values.astype(float)

        train_prev_unw = float(np.mean(y_train))
        train_prev_w = float(np.average(y_train, weights=w_train))

        # Predict on train to derive optimal threshold (Youden's J)
        probs_train = clinical_model.predict_proba(X_train)[:, 1]
        opt_thresh = find_optimal_training_threshold(y_train, probs_train)

        # Predict on Cycle J holdout
        probs_test = clinical_model.predict_proba(X_test)[:, 1]

        # Evaluate unweighted and survey-weighted
        metrics_unweighted = evaluate_cohort(
            y_test, probs_test, train_prev=train_prev_unw, train_threshold=opt_thresh
        )
        metrics_weighted = evaluate_cohort(
            y_test, probs_test, train_prev=train_prev_w, train_threshold=opt_thresh, sample_weight=w_test
        )

        # ---------------------------------------------------------------------
        # Subgroup Fairness Analysis on Cycle J Holdout
        # ---------------------------------------------------------------------
        subgroups: dict[str, Any] = {}

        # By Biological Sex
        for s_code, s_name in [(1.0, "male"), (0.0, "female")]:
            m_s = (test_data["sex_male"] == s_code).values
            if m_s.sum() >= 20 and len(np.unique(y_test[m_s])) == 2:
                y_sub = y_test[m_s]
                p_sub = probs_test[m_s]
                w_sub = w_test[m_s]
                subgroups[f"sex_{s_name}"] = {
                    "subgroup_name": f"Sex: {s_name.capitalize()}",
                    "n_test": int(m_s.sum()),
                    "n_positive": int(y_sub.sum()),
                    "prevalence_unweighted": round(float(np.mean(y_sub)), 4),
                    "prevalence_weighted": round(float(np.average(y_sub, weights=w_sub)), 4),
                    "auroc_unweighted": round(float(roc_auc_score(y_sub, p_sub)), 4),
                    "auroc_weighted": round(float(roc_auc_score(y_sub, p_sub, sample_weight=w_sub)), 4),
                }

        # By Age Band
        age_bands = [
            ("age_18_39", "Age 18-39", (test_data["age"] >= 18) & (test_data["age"] < 40)),
            ("age_40_64", "Age 40-64", (test_data["age"] >= 40) & (test_data["age"] < 65)),
            ("age_65_plus", "Age 65+", test_data["age"] >= 65),
        ]
        for b_key, b_label, b_mask in age_bands:
            m_b = b_mask.values
            if m_b.sum() >= 20 and len(np.unique(y_test[m_b])) == 2:
                y_sub = y_test[m_b]
                p_sub = probs_test[m_b]
                w_sub = w_test[m_b]
                subgroups[b_key] = {
                    "subgroup_name": b_label,
                    "n_test": int(m_b.sum()),
                    "n_positive": int(y_sub.sum()),
                    "prevalence_unweighted": round(float(np.mean(y_sub)), 4),
                    "prevalence_weighted": round(float(np.average(y_sub, weights=w_sub)), 4),
                    "auroc_unweighted": round(float(roc_auc_score(y_sub, p_sub)), 4),
                    "auroc_weighted": round(float(roc_auc_score(y_sub, p_sub, sample_weight=w_sub)), 4),
                }

        # By Race/Ethnicity (including Non-Hispanic Asian)
        for r_code, r_label in RACE_ETHNICITY_MAP.items():
            m_r = (test_data["race_ethnicity"] == r_code).values
            if m_r.sum() >= 20 and len(np.unique(y_test[m_r])) == 2:
                y_sub = y_test[m_r]
                p_sub = probs_test[m_r]
                w_sub = w_test[m_r]
                subgroups[f"race_{r_label.lower().replace(' ', '_').replace('/', '_')}"] = {
                    "subgroup_name": f"Race/Ethnicity: {r_label}",
                    "race_code": int(r_code),
                    "n_test": int(m_r.sum()),
                    "n_positive": int(y_sub.sum()),
                    "prevalence_unweighted": round(float(np.mean(y_sub)), 4),
                    "prevalence_weighted": round(float(np.average(y_sub, weights=w_sub)), 4),
                    "auroc_unweighted": round(float(roc_auc_score(y_sub, p_sub)), 4),
                    "auroc_weighted": round(float(roc_auc_score(y_sub, p_sub, sample_weight=w_sub)), 4),
                }

        # ---------------------------------------------------------------------
        # Ablations
        # ---------------------------------------------------------------------
        ablations: dict[str, Any] = {}

        # 1. Lifestyle & body size only ablation for this target
        lifestyle_feats = [f for f in LIFESTYLE_BODY_FEATURES if f in train_data.columns]
        X_train_life = train_data[lifestyle_feats].values
        X_test_life = test_data[lifestyle_feats].values

        cal_life = train_ablation_model(X_train_life, y_train)
        probs_train_life = cal_life.predict_proba(X_train_life)[:, 1]
        thresh_life = find_optimal_training_threshold(y_train, probs_train_life)
        probs_test_life = cal_life.predict_proba(X_test_life)[:, 1]

        ablations["lifestyle_and_body_size_only"] = {
            "description": "Restricted to body measures, demographics, and lifestyle factors (no lab tests, no vitals)",
            "feature_count": len(lifestyle_feats),
            "features": lifestyle_feats,
            "unweighted": evaluate_cohort(
                y_test, probs_test_life, train_prev=train_prev_unw, train_threshold=thresh_life
            ),
            "survey_weighted": evaluate_cohort(
                y_test, probs_test_life, train_prev=train_prev_w, train_threshold=thresh_life, sample_weight=w_test
            ),
        }

        # 2. Specific ablation: Hypertension without measured blood pressure
        if t_id == "hypertension":
            no_bp_feats = [f for f in feature_names if f not in ["systolic_bp", "diastolic_bp", "pulse_bpm"]]
            X_train_nobp = train_data[no_bp_feats].values
            X_test_nobp = test_data[no_bp_feats].values

            cal_nobp = train_ablation_model(X_train_nobp, y_train)
            probs_train_nobp = cal_nobp.predict_proba(X_train_nobp)[:, 1]
            thresh_nobp = find_optimal_training_threshold(y_train, probs_train_nobp)
            probs_test_nobp = cal_nobp.predict_proba(X_test_nobp)[:, 1]

            ablations["hypertension_without_measured_bp"] = {
                "description": "All clinical features except measured blood pressure exam readings (systolic_bp, diastolic_bp, pulse_bpm)",
                "feature_count": len(no_bp_feats),
                "excluded_features": ["systolic_bp", "diastolic_bp", "pulse_bpm"],
                "unweighted": evaluate_cohort(
                    y_test, probs_test_nobp, train_prev=train_prev_unw, train_threshold=thresh_nobp
                ),
                "survey_weighted": evaluate_cohort(
                    y_test, probs_test_nobp, train_prev=train_prev_w, train_threshold=thresh_nobp, sample_weight=w_test
                ),
            }

        # 3. Specific ablation: High cholesterol without lipid panel
        if t_id == "high_cholesterol":
            no_lip_feats = [
                f for f in feature_names
                if f not in ["total_cholesterol", "hdl_cholesterol", "triglycerides", "ldl_cholesterol"]
            ]
            X_train_nolip = train_data[no_lip_feats].values
            X_test_nolip = test_data[no_lip_feats].values

            cal_nolip = train_ablation_model(X_train_nolip, y_train)
            probs_train_nolip = cal_nolip.predict_proba(X_train_nolip)[:, 1]
            thresh_nolip = find_optimal_training_threshold(y_train, probs_train_nolip)
            probs_test_nolip = cal_nolip.predict_proba(X_test_nolip)[:, 1]

            ablations["high_cholesterol_without_lipid_values"] = {
                "description": "All clinical features except measured lipid panel biomarkers (total_cholesterol, hdl_cholesterol, triglycerides, ldl_cholesterol)",
                "feature_count": len(no_lip_feats),
                "excluded_features": ["total_cholesterol", "hdl_cholesterol", "triglycerides", "ldl_cholesterol"],
                "unweighted": evaluate_cohort(
                    y_test, probs_test_nolip, train_prev=train_prev_unw, train_threshold=thresh_nolip
                ),
                "survey_weighted": evaluate_cohort(
                    y_test, probs_test_nolip, train_prev=train_prev_w, train_threshold=thresh_nolip, sample_weight=w_test
                ),
            }

        # Assemble target payload
        target_payload = {
            "target_id": t_id,
            "target_name": t_name,
            "target_column": t_col,
            "questionnaire_variable": spec["questionnaire_variable"],
            "question_text": spec["question_text"],
            "is_self_reported_ever_told_diagnosis": spec["is_self_reported_ever_told"],
            "distinguishes_type_2": spec["distinguishes_type2"],
            "sample_counts": {
                "train_n": len(y_train),
                "train_positive": int(y_train.sum()),
                "train_prevalence_unweighted": round(train_prev_unw, 4),
                "train_prevalence_weighted": round(train_prev_w, 4),
                "holdout_n": len(y_test),
                "holdout_positive": int(y_test.sum()),
            },
            "features_used_count": len(feature_names),
            "features_used": feature_names,
            "unweighted_metrics": metrics_unweighted,
            "survey_weighted_metrics": metrics_weighted,
            "subgroups": subgroups,
            "ablations": ablations,
        }

        audit_output["targets"][t_id] = target_payload

        # Summary row for table
        audit_output["summary_table"].append({
            "target_id": t_id,
            "target_name": t_name,
            "unweighted_auroc": metrics_unweighted["auroc"],
            "survey_weighted_auroc": metrics_weighted["auroc"],
            "unweighted_auprc": metrics_unweighted["auprc"],
            "survey_weighted_auprc": metrics_weighted["auprc"],
            "unweighted_prevalence": metrics_unweighted["prevalence"],
            "always_negative_baseline": metrics_unweighted["accuracy_always_negative"],
            "brier_score": metrics_unweighted["brier_score"],
            "brier_skill_score": metrics_unweighted["brier_skill_score"],
            "calibration_slope": metrics_unweighted["calibration_slope"],
            "calibration_intercept": metrics_unweighted["calibration_intercept"],
            "training_threshold": metrics_unweighted["threshold_training_optimal"],
            "sensitivity": metrics_unweighted["at_training_threshold"]["sensitivity"],
            "specificity": metrics_unweighted["at_training_threshold"]["specificity"],
            "ppv": metrics_unweighted["at_training_threshold"]["ppv"],
        })

    # Save output json
    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w") as f:
        json.dump(audit_output, f, indent=2)

    logger.info("Saved complete NHANES audit to %s", OUTPUT_REPORT_PATH)
    logger.info("Completed in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
