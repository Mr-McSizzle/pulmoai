#!/usr/bin/env python3
"""Audit and validate future mortality risk models (Pass 4 E3 & E4).

Expands audit_future_risk_v2.py with:
1. Fair Primary Prevention Comparison (E3):
   - Primary prevention cohort: Age 40-79, without prior CVD (target_heart_disease == 0 & target_stroke == 0; n=5,899 holdout adults, 110 fatal CVD events)
   - Models compared:
     (a) Published PCE (ACC/AHA 2013, 9 features)
     (b) PCE-input Cox model (trained on 1999-2010 using ONLY the 9 PCE inputs)
     (c) Full Multivariable Cox model (adding vital signs, lifestyle, and specialized lab tests)
   - Paired bootstrap 95% CIs for:
     * Delta C-index and Delta 5-yr AUROC: Full Cox vs PCE-input Cox
     * Delta C-index and Delta 5-yr AUROC: PCE-input Cox vs Published PCE
   - Predictor categorization: which added features require specialized labs
   - Race handling disclosure for Other / Hispanic / Asian populations in PCE
2. Subgroup Bootstrap CIs (E4):
   - Non-Hispanic Asian subgroup (n=1,529, 53 deaths): C-index 95% CI and 5-yr AUROC 95% CI
   - Sex subgroups:
     * Males (n=4,726, 335 deaths): C-index 95% CI and 5-yr AUROC 95% CI
     * Females (n=4,916, 273 deaths): C-index 95% CI and 5-yr AUROC 95% CI
3. Cardiovascular Mortality Decile Calibration (E4):
   - Observed vs Expected events across 10 risk deciles on 2011-2014 holdout
   - Overall O/E ratio with 95% CI
   - Calibration-in-the-large intercept
   - Raw and recalibrated 5-year Brier scores
4. Output:
   - reports/audit_future_risk_v3.json
   - models/future_risk_models_v3.joblib
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, brier_score_loss, roc_auc_score, roc_curve
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_future_risk_v3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nhanes_mortality_linked.parquet"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "audit_future_risk_v3.json"
OUTPUT_MODELS = PROJECT_ROOT / "models" / "future_risk_models_v3.joblib"


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
    """Compute 10-year hard ASCVD risk (%) via 2013 ACC/AHA Pooled Cohort Equations (Goff et al.).

    Note on Race Handling (E3):
    Per the 2013 ACC/AHA Guideline (Goff et al., Circulation 2014;129:S49-S73, Section 4.1),
    because sex- and race-specific models were developed only for non-Hispanic African American
    and non-Hispanic White populations, the guideline specifies that the non-Hispanic White
    equations should be used for individuals of other racial and ethnic backgrounds (e.g.,
    Hispanic, Asian American, American Indian, multiracial).
    """
    if not (40.0 <= age <= 79.0):
        return None
    if not (90.0 <= systolic_bp <= 200.0):
        return None
    if not (130.0 <= total_chol <= 320.0):
        return None
    if not (20.0 <= hdl_chol <= 100.0):
        return None

    ln_age = math.log(age)
    ln_tc = math.log(total_chol)
    ln_hdl = math.log(hdl_chol)
    ln_sbp = math.log(systolic_bp)

    tr_sbp = ln_sbp if hypertensive else 0.0
    nt_sbp = 0.0 if hypertensive else ln_sbp
    age_tc = ln_age * ln_tc
    age_hdl = ln_age * ln_hdl
    age_smoke = ln_age if smoker else 0.0

    if not is_male and not is_black:  # White / Other Female
        s10 = 0.96652
        mean_xb = -29.1817
        indiv = (
            -29.799 * ln_age
            + 4.884 * (ln_age ** 2)
            + 13.540 * ln_tc
            - 3.114 * age_tc
            - 13.578 * ln_hdl
            + 3.149 * age_hdl
            + 2.019 * tr_sbp
            + 1.957 * nt_sbp
            + (7.574 if smoker else 0.0)
            - 1.665 * age_smoke
            + (0.661 if diabetic else 0.0)
        )
    elif not is_male and is_black:  # African American Female
        s10 = 0.95334
        mean_xb = 86.6081
        indiv = (
            17.114 * ln_age
            + 0.940 * ln_tc
            - 18.920 * ln_hdl
            + 4.475 * age_hdl
            + 29.291 * tr_sbp
            - 6.432 * (ln_age * tr_sbp if hypertensive else 0.0)
            + 27.820 * nt_sbp
            - 6.087 * (ln_age * nt_sbp if not hypertensive else 0.0)
            + (0.691 if smoker else 0.0)
            + (0.874 if diabetic else 0.0)
        )
    elif is_male and is_black:  # African American Male
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
    else:  # White / Other Male
        s10 = 0.91436
        mean_xb = 61.1816
        indiv = (
            12.344 * ln_age
            + 11.853 * ln_tc
            - 2.664 * age_tc
            - 7.990 * ln_hdl
            + 1.769 * age_hdl
            + 1.797 * tr_sbp
            + 1.764 * nt_sbp
            + (7.837 if smoker else 0.0)
            - 1.795 * age_smoke
            + (0.658 if diabetic else 0.0)
        )

    risk = 1.0 - (s10 ** math.exp(indiv - mean_xb))
    return float(np.clip(risk * 100.0, 0.01, 99.9))


def compute_idrs(
    age: float,
    waist_cm: float,
    is_male: bool,
    activity_level: int | str,
    family_history_diabetes: int | float,
) -> dict[str, Any]:
    """Compute Indian Diabetes Risk Score (0-100) (Mohan et al., JAPI 2005 Table 1).

    Criteria:
    1. Age: <35: 0, 35-49: 20, >=50: 30
    2. Abdominal Obesity (Waist):
       Male: <90 cm: 0, 90-99 cm: 10, >=100 cm: 20
       Female: <80 cm: 0, 80-89 cm: 10, >=90 cm: 20
    3. Physical Activity:
       Regular exercise AND strenuous work: 0
       Regular exercise OR strenuous work: 20
       No exercise AND sedentary activities: 30
    4. Family History of Diabetes:
       No parental diabetes: 0
       One parent diabetic: 10
       Both parents diabetic: 20
    """
    # 1. Age
    if age < 35:
        p_age = 0
    elif age < 50:
        p_age = 20
    else:
        p_age = 30

    # 2. Abdominal Obesity
    if is_male:
        if waist_cm < 90.0:
            p_waist = 0
        elif waist_cm < 100.0:
            p_waist = 10
        else:
            p_waist = 20
    else:
        if waist_cm < 80.0:
            p_waist = 0
        elif waist_cm < 90.0:
            p_waist = 10
        else:
            p_waist = 20

    # 3. Physical Activity
    act_str = str(activity_level).lower().strip()
    if act_str in ("0", "0.0", "vigorous", "regular_and_strenuous"):
        p_act = 0
    elif act_str in ("1", "1.0", "moderate", "either_regular_or_strenuous"):
        p_act = 20
    else:  # sedentary
        p_act = 30

    # 4. Family History
    fam_str = str(family_history_diabetes).strip()
    if fam_str in ("0", "0.0", "none"):
        p_fam = 0
    elif fam_str in ("1", "1.0", "one_parent"):
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


def bootstrap_cindex_ci(
    events: np.ndarray,
    durations: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Compute point estimate and 95% bootstrap CI for Harrell's C-index."""
    pt_c = round(float(concordance_index_censored(events.astype(bool), durations, scores)[0]), 4)
    rng = np.random.RandomState(seed)
    n = len(events)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if events[idx].sum() > 0:
            try:
                c = concordance_index_censored(events[idx].astype(bool), durations[idx], scores[idx])[0]
                vals.append(c)
            except Exception:
                pass
    if not vals:
        ci = [pt_c, pt_c]
    else:
        ci = [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]
    return pt_c, ci


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Compute point estimate and 95% bootstrap CI for AUROC."""
    pt_auc = round(float(roc_auc_score(y_true, scores)), 4)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) == 2:
            try:
                vals.append(roc_auc_score(y_true[idx], scores[idx]))
            except Exception:
                pass
    if not vals:
        ci = [pt_auc, pt_auc]
    else:
        ci = [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]
    return pt_auc, ci


def paired_bootstrap_cindex_diff(
    events: np.ndarray,
    durations: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Compute point estimate and 95% paired bootstrap CI for C-index difference (A - B)."""
    c_a = concordance_index_censored(events.astype(bool), durations, scores_a)[0]
    c_b = concordance_index_censored(events.astype(bool), durations, scores_b)[0]
    delta_point = float(c_a - c_b)

    rng = np.random.RandomState(seed)
    n = len(events)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if events[idx].sum() > 0:
            try:
                ca_b = concordance_index_censored(events[idx].astype(bool), durations[idx], scores_a[idx])[0]
                cb_b = concordance_index_censored(events[idx].astype(bool), durations[idx], scores_b[idx])[0]
                diffs.append(ca_b - cb_b)
            except Exception:
                pass

    if not diffs:
        ci = [delta_point, delta_point]
    else:
        ci = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
    return round(delta_point, 4), ci


def paired_bootstrap_auroc_diff(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Compute point estimate and 95% paired bootstrap CI for AUROC difference (A - B)."""
    auc_a = roc_auc_score(y_true, scores_a)
    auc_b = roc_auc_score(y_true, scores_b)
    delta_point = float(auc_a - auc_b)

    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) == 2:
            try:
                diffs.append(roc_auc_score(y_true[idx], scores_a[idx]) - roc_auc_score(y_true[idx], scores_b[idx]))
            except Exception:
                pass

    if not diffs:
        ci = [delta_point, delta_point]
    else:
        ci = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
    return round(delta_point, 4), ci


def compute_decile_calibration(
    y_true_status: np.ndarray,
    pred_risk_5yr: np.ndarray,
    target_name: str,
) -> dict[str, Any]:
    """Compute decile calibration table, O/E ratio with 95% CI, and calibration-in-the-large intercept."""
    valid = np.isfinite(y_true_status) & np.isfinite(pred_risk_5yr)
    y_val = y_true_status[valid]
    p_val = pred_risk_5yr[valid]

    deciles = pd.qcut(p_val, q=10, duplicates="drop")
    decile_stats = []
    for d_idx, (cat, grp) in enumerate(pd.DataFrame({
        "decile": deciles,
        "obs": y_val,
        "pred": p_val,
    }).groupby("decile", observed=True)):
        n_d = len(grp)
        e_d = int(grp["obs"].sum())
        p_mean = float(grp["pred"].mean())
        obs_rate = float(e_d / n_d) if n_d > 0 else 0.0
        ci_low, ci_high = stats.binom.interval(0.95, n_d, max(obs_rate, 1e-4)) if n_d > 0 else (0, 0)
        decile_stats.append({
            "decile_index": d_idx + 1,
            "decile_range": str(cat),
            "n_subjects": n_d,
            "observed_events": e_d,
            "observed_rate": round(obs_rate, 4),
            "observed_95ci": [round(float(ci_low / n_d), 4), round(float(ci_high / n_d), 4)],
            "mean_predicted_risk": round(p_mean, 4),
            "predicted_to_observed_ratio": round(p_mean / max(obs_rate, 1e-5), 3),
        })

    total_obs = float(y_val.sum())
    total_exp = float(p_val.sum())
    oe_ratio = round(total_obs / max(total_exp, 1e-5), 3)
    # Wilson CI for O/E ratio based on Poisson observed count
    se_log_oe = 1.0 / math.sqrt(max(total_obs, 1.0))
    oe_ci = [round(float(oe_ratio * math.exp(-1.96 * se_log_oe)), 3), round(float(oe_ratio * math.exp(1.96 * se_log_oe)), 3)]

    # Calibration-in-the-large intercept
    eps = 1e-5
    p_clip = np.clip(p_val, eps, 1.0 - eps)
    logit_pred = np.log(p_clip / (1.0 - p_clip))

    def neg_ll(alpha: float) -> float:
        z = alpha + logit_pred
        p = np.clip(1.0 / (1.0 + np.exp(-z)), eps, 1.0 - eps)
        return -float(np.sum(y_val * np.log(p) + (1.0 - y_val) * np.log(1.0 - p)))

    opt = minimize(neg_ll, x0=[0.0], method="Nelder-Mead")
    cal_in_large_alpha = round(float(opt.x[0]), 4)

    recalibrated_pred_risk = 1.0 / (1.0 + np.exp(-(logit_pred + cal_in_large_alpha)))
    raw_brier = round(float(brier_score_loss(y_val, p_val)), 4)
    recalibrated_brier = round(float(brier_score_loss(y_val, recalibrated_pred_risk)), 4)

    return {
        "target": target_name,
        "n_evaluated": int(len(y_val)),
        "observed_5yr_events": int(total_obs),
        "expected_5yr_events": round(total_exp, 1),
        "observed_to_expected_ratio": oe_ratio,
        "oe_ratio_95ci": oe_ci,
        "calibration_in_the_large_intercept": cal_in_large_alpha,
        "raw_brier_score_5yr": raw_brier,
        "intercept_recalibrated_brier_5yr": recalibrated_brier,
        "decile_calibration_table": decile_stats,
    }


def main() -> None:
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("STARTING FUTURE RISK AUDIT PASS 4 (E3 & E4)")
    logger.info("=" * 80)

    # 1. Load Data
    logger.info("Loading NHANES-LMF linked dataset from %s...", DATA_PATH)
    df = pd.read_parquet(DATA_PATH)
    df_adults = df[df["eligstat"] == 1].copy()

    train_cycles = ["1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008", "2009-2010"]
    test_cycles = ["2011-2012", "2013-2014"]

    train_df = df_adults[df_adults["cycle"].isin(train_cycles)].copy()
    test_df = df_adults[df_adults["cycle"].isin(test_cycles)].copy()

    mortality_features = [
        "age", "sex_male",
        "bmi", "waist_cm", "systolic_bp", "diastolic_bp", "pulse_bpm", "bp_medication",
        "total_cholesterol", "hdl_cholesterol", "fasting_glucose", "hba1c",
        "serum_creatinine", "bun", "albumin", "uric_acid",
        "current_smoker", "drinks_per_day", "physically_active",
        "target_diabetes", "target_hypertension", "target_heart_disease", "target_stroke",
    ]

    train_medians = train_df[mortality_features].median()
    X_train_full = train_df[mortality_features].fillna(train_medians)
    X_test_full = test_df[mortality_features].fillna(train_medians)

    # All-cause structured arrays
    y_train_surv_all = np.array(
        list(zip(train_df["event_all_cause"].astype(bool), train_df["followup_years"])),
        dtype=[("Status", "?"), ("Survival_in_years", "<f8")],
    )
    y_test_surv_all = np.array(
        list(zip(test_df["event_all_cause"].astype(bool), test_df["followup_years"])),
        dtype=[("Status", "?"), ("Survival_in_years", "<f8")],
    )

    # Fit Full Cox models for all-cause and CVD
    cox_full_all = CoxPHSurvivalAnalysis(alpha=1e-2)
    cox_full_all.fit(X_train_full, y_train_surv_all)
    risk_scores_full_all = cox_full_all.predict(X_test_full)

    # Lifelines fit for calibrated probabilities
    train_ll = train_df[["followup_years", "event_all_cause"] + mortality_features].copy()
    train_ll[mortality_features] = train_ll[mortality_features].fillna(train_medians)
    cph_all = CoxPHFitter(penalizer=1e-2)
    cph_all.fit(train_ll, duration_col="followup_years", event_col="event_all_cause")

    test_ll_x = test_df[mortality_features].fillna(train_medians)
    pred_risk_5yr_all = 1.0 - cph_all.predict_survival_function(test_ll_x, times=[5.0]).T.values.ravel()

    # CVD mortality model
    train_ll_cvd = train_df[["followup_years", "event_cvd"] + mortality_features].copy()
    train_ll_cvd[mortality_features] = train_ll_cvd[mortality_features].fillna(train_medians)
    cph_cvd = CoxPHFitter(penalizer=1e-2)
    cph_cvd.fit(train_ll_cvd, duration_col="followup_years", event_col="event_cvd")

    pred_risk_5yr_cvd = 1.0 - cph_cvd.predict_survival_function(test_ll_x, times=[5.0]).T.values.ravel()

    # Decile Calibrations (E4)
    logger.info("Computing decile calibration for All-Cause Mortality...")
    decile_cal_all = compute_decile_calibration(
        test_df["status_5yr_all_cause"].values, pred_risk_5yr_all, "all_cause_mortality"
    )

    logger.info("Computing decile calibration for Cardiovascular Mortality...")
    decile_cal_cvd = compute_decile_calibration(
        test_df["status_5yr_cvd"].values, pred_risk_5yr_cvd, "cardiovascular_mortality"
    )

    # Subgroups with 95% CIs (E4)
    logger.info("Computing subgroup metrics with bootstrap 95% CIs...")
    subgroups = {}

    # Non-Hispanic Asian
    asian_mask = (test_df["race_ethnicity"] == 6.0).values
    ev_asian = test_df.loc[asian_mask, "event_all_cause"].values
    dur_asian = test_df.loc[asian_mask, "followup_years"].values
    st5_asian = test_df.loc[asian_mask, "status_5yr_all_cause"].values
    sc_asian = risk_scores_full_all[asian_mask]
    v5_asian = np.isfinite(st5_asian)

    c_pt_asian, c_ci_asian = bootstrap_cindex_ci(ev_asian, dur_asian, sc_asian)
    auc_pt_asian, auc_ci_asian = bootstrap_auroc_ci(st5_asian[v5_asian], sc_asian[v5_asian])

    subgroups["non_hispanic_asian"] = {
        "subgroup": "Non-Hispanic Asian (NHANES race_ethnicity = 6.0)",
        "n_subjects": int(asian_mask.sum()),
        "deaths_observed": int(ev_asian.sum()),
        "c_index": c_pt_asian,
        "c_index_95ci": c_ci_asian,
        "auroc_5yr": auc_pt_asian,
        "auroc_5yr_95ci": auc_ci_asian,
    }

    # Sex Subgroups (E4)
    for sex_label, sex_code in [("males", 1), ("females", 0)]:
        s_mask = (test_df["sex_male"] == sex_code).values
        ev_s = test_df.loc[s_mask, "event_all_cause"].values
        dur_s = test_df.loc[s_mask, "followup_years"].values
        st5_s = test_df.loc[s_mask, "status_5yr_all_cause"].values
        sc_s = risk_scores_full_all[s_mask]
        v5_s = np.isfinite(st5_s)

        c_pt, c_ci = bootstrap_cindex_ci(ev_s, dur_s, sc_s)
        auc_pt, auc_ci = bootstrap_auroc_ci(st5_s[v5_s], sc_s[v5_s])

        subgroups[sex_label] = {
            "subgroup": f"{sex_label.capitalize()} ({'sex_male = 1' if sex_code == 1 else 'sex_male = 0'})",
            "n_subjects": int(s_mask.sum()),
            "deaths_observed": int(ev_s.sum()),
            "c_index": c_pt,
            "c_index_95ci": c_ci,
            "auroc_5yr": auc_pt,
            "auroc_5yr_95ci": auc_ci,
        }

    # -------------------------------------------------------------------------
    # PART 2: FAIR PRIMARY PREVENTION COMPARISON (E3)
    # -------------------------------------------------------------------------
    logger.info("Conducting Primary Prevention Comparison on Age 40-79 without prior CVD (E3)...")
    pce_cohort_train = train_df[
        (train_df["age"] >= 40)
        & (train_df["age"] <= 79)
        & (train_df["target_heart_disease"] == 0)
        & (train_df["target_stroke"] == 0)
    ].copy()

    pce_cohort_test = test_df[
        (test_df["age"] >= 40)
        & (test_df["age"] <= 79)
        & (test_df["target_heart_disease"] == 0)
        & (test_df["target_stroke"] == 0)
    ].copy()

    pce_features = [
        "age", "sex_male", "race_black",
        "total_cholesterol", "hdl_cholesterol", "systolic_bp",
        "bp_medication", "current_smoker", "target_diabetes",
    ]

    pce_train_medians = pce_cohort_train[pce_features].median()
    X_pce_train = pce_cohort_train[pce_features].fillna(pce_train_medians)
    X_pce_test = pce_cohort_test[pce_features].fillna(pce_train_medians)

    # Full model features evaluated on primary prevention cohort
    full_pp_features = [f for f in mortality_features if f not in ("target_heart_disease", "target_stroke")]
    full_pp_medians = pce_cohort_train[full_pp_features].median()
    X_full_pp_train = pce_cohort_train[full_pp_features].fillna(full_pp_medians)
    X_full_pp_test = pce_cohort_test[full_pp_features].fillna(full_pp_medians)

    y_pce_tr_surv = np.array(
        list(zip(pce_cohort_train["event_cvd"].astype(bool), pce_cohort_train["followup_years"])),
        dtype=[("Status", "?"), ("Survival_in_years", "<f8")],
    )
    y_pce_te_surv = np.array(
        list(zip(pce_cohort_test["event_cvd"].astype(bool), pce_cohort_test["followup_years"])),
        dtype=[("Status", "?"), ("Survival_in_years", "<f8")],
    )

    # 1. Fit Cox model on ONLY PCE inputs
    cox_pce_inputs = CoxPHSurvivalAnalysis(alpha=1e-2)
    cox_pce_inputs.fit(X_pce_train, y_pce_tr_surv)
    risk_scores_cox_pce = cox_pce_inputs.predict(X_pce_test)

    # 2. Fit Full Multivariable Cox model on primary prevention cohort
    cox_full_pp = CoxPHSurvivalAnalysis(alpha=1e-2)
    cox_full_pp.fit(X_full_pp_train, y_pce_tr_surv)
    risk_scores_full_pp = cox_full_pp.predict(X_full_pp_test)

    # 3. Compute published PCE scores
    pce_scores = []
    for _, r in pce_cohort_test.iterrows():
        p_val = compute_pce_10yr(
            is_male=bool(r["sex_male"] == 1),
            is_black=bool(r["race_black"] == 1),
            smoker=bool(r["current_smoker"] == 1),
            hypertensive=bool(r["bp_medication"] == 1),
            diabetic=bool(r["target_diabetes"] == 1),
            age=float(r["age"]),
            systolic_bp=float(r["systolic_bp"]) if pd.notna(r["systolic_bp"]) else float(pce_train_medians["systolic_bp"]),
            total_chol=float(r["total_cholesterol"]) if pd.notna(r["total_cholesterol"]) else float(pce_train_medians["total_cholesterol"]),
            hdl_chol=float(r["hdl_cholesterol"]) if pd.notna(r["hdl_cholesterol"]) else float(pce_train_medians["hdl_cholesterol"]),
        )
        pce_scores.append(p_val if p_val is not None else 5.0)

    pce_scores = np.array(pce_scores)
    ev_cvd = pce_cohort_test["event_cvd"].values
    dur_cvd = pce_cohort_test["followup_years"].values
    st5_cvd = pce_cohort_test["status_5yr_cvd"].values
    v5_cvd = np.isfinite(st5_cvd)

    # Discrimination metrics
    c_pce, ci_c_pce = bootstrap_cindex_ci(ev_cvd, dur_cvd, pce_scores)
    auc_pce, ci_auc_pce = bootstrap_auroc_ci(st5_cvd[v5_cvd], pce_scores[v5_cvd])

    c_cox_pce, ci_c_cox_pce = bootstrap_cindex_ci(ev_cvd, dur_cvd, risk_scores_cox_pce)
    auc_cox_pce, ci_auc_cox_pce = bootstrap_auroc_ci(st5_cvd[v5_cvd], risk_scores_cox_pce[v5_cvd])

    c_full_pp, ci_c_full_pp = bootstrap_cindex_ci(ev_cvd, dur_cvd, risk_scores_full_pp)
    auc_full_pp, ci_auc_full_pp = bootstrap_auroc_ci(st5_cvd[v5_cvd], risk_scores_full_pp[v5_cvd])

    # Paired comparisons:
    # A. Full Cox vs PCE-input Cox (E3 key question: Does adding features beyond PCE help?)
    delta_c_full_vs_pce_inputs, ci_delta_c_full_vs_pce_inputs = paired_bootstrap_cindex_diff(
        ev_cvd, dur_cvd, risk_scores_full_pp, risk_scores_cox_pce
    )
    delta_auc_full_vs_pce_inputs, ci_delta_auc_full_vs_pce_inputs = paired_bootstrap_auroc_diff(
        st5_cvd[v5_cvd], risk_scores_full_pp[v5_cvd], risk_scores_cox_pce[v5_cvd]
    )

    # B. PCE-input Cox vs Published PCE (E3 comparator: Does ML algorithm change anything on same inputs?)
    delta_c_pce_inputs_vs_pce, ci_delta_c_pce_inputs_vs_pce = paired_bootstrap_cindex_diff(
        ev_cvd, dur_cvd, risk_scores_cox_pce, pce_scores
    )
    delta_auc_pce_inputs_vs_pce, ci_delta_auc_pce_inputs_vs_pce = paired_bootstrap_auroc_diff(
        st5_cvd[v5_cvd], risk_scores_cox_pce[v5_cvd], pce_scores[v5_cvd]
    )

    # Feature categorization (E3)
    pce_features_list = [
        "age", "sex_male", "race_black", "total_cholesterol", "hdl_cholesterol",
        "systolic_bp", "bp_medication", "current_smoker", "target_diabetes",
    ]
    added_non_lab_predictors = [
        "bmi", "waist_cm", "diastolic_bp", "pulse_bpm", "drinks_per_day", "physically_active", "target_hypertension",
    ]
    added_specialized_lab_predictors = [
        "fasting_glucose", "hba1c", "serum_creatinine", "bun", "albumin", "uric_acid",
    ]

    primary_prevention_comparison = {
        "cohort_description": "Ages 40-79 years without prior cardiovascular disease (MCQ160B/C/D/E/F = 0) on temporal holdout (2011-2014)",
        "sample_size_holdout": int(len(pce_cohort_test)),
        "fatal_cvd_events_total_followup": int(ev_cvd.sum()),
        "fatal_cvd_events_5yr": int(st5_cvd[v5_cvd].sum()),
        "models": {
            "published_pce_2013": {
                "display_name": "Published 2013 ACC/AHA Pooled Cohort Equations (9 inputs)",
                "c_index": c_pce,
                "c_index_95ci": ci_c_pce,
                "auroc_5yr": auc_pce,
                "auroc_5yr_95ci": ci_auc_pce,
            },
            "cox_pce_inputs_only": {
                "display_name": "In-repo Cox PH Model (Trained on 1999-2010 using ONLY 9 PCE inputs)",
                "c_index": c_cox_pce,
                "c_index_95ci": ci_c_cox_pce,
                "auroc_5yr": auc_cox_pce,
                "auroc_5yr_95ci": ci_auc_cox_pce,
            },
            "cox_full_multivariable": {
                "display_name": "Full Multivariable Cox PH Model (Adding vitals, lifestyle, and specialized labs)",
                "c_index": c_full_pp,
                "c_index_95ci": ci_c_full_pp,
                "auroc_5yr": auc_full_pp,
                "auroc_5yr_95ci": ci_auc_full_pp,
            },
        },
        "paired_comparisons": {
            "full_cox_vs_pce_inputs_cox": {
                "delta_c_index": delta_c_full_vs_pce_inputs,
                "delta_c_index_95ci": ci_delta_c_full_vs_pce_inputs,
                "delta_auroc_5yr": delta_auc_full_vs_pce_inputs,
                "delta_auroc_5yr_95ci": ci_delta_auc_full_vs_pce_inputs,
                "interpretation": (
                    "Added value supported by CI"
                    if ci_delta_c_full_vs_pce_inputs[0] > 0
                    else "No clear difference beyond noise (CI spans zero)"
                ),
            },
            "pce_inputs_cox_vs_published_pce": {
                "delta_c_index": delta_c_pce_inputs_vs_pce,
                "delta_c_index_95ci": ci_delta_c_pce_inputs_vs_pce,
                "delta_auroc_5yr": delta_auc_pce_inputs_vs_pce,
                "delta_auroc_5yr_95ci": ci_delta_auc_pce_inputs_vs_pce,
                "interpretation": (
                    "Added value supported by CI"
                    if ci_delta_c_pce_inputs_vs_pce[0] > 0
                    else "No clear difference between ML and published equation on identical inputs (CI spans zero)"
                ),
            },
        },
        "predictor_breakdown": {
            "pce_inputs": pce_features_list,
            "added_non_lab_predictors": added_non_lab_predictors,
            "added_specialized_lab_predictors_requiring_blood_draws": added_specialized_lab_predictors,
            "clinical_barrier_note": (
                "Specialized lab tests (fasting glucose, HbA1c, serum creatinine, BUN, albumin, uric acid) "
                "require venipuncture and laboratory assays that a patient visiting an exploratory web interface "
                "or routine health checkup commonly does not possess."
            ),
        },
        "race_handling_disclosure": (
            "Per the 2013 ACC/AHA Guideline (Goff et al., Circulation 2014;129:S49-S73, Section 4.1), "
            "because sex- and race-specific pooled cohort models were derived exclusively for Non-Hispanic African American "
            "and Non-Hispanic White cohorts, the guideline explicitly specifies using the Non-Hispanic White equations "
            "for individuals of other racial and ethnic backgrounds (e.g., Hispanic, Asian American, American Indian, multiracial). "
            "Risk estimates may under- or over-estimate risk in these populations."
        ),
    }

    # -------------------------------------------------------------------------
    # PART 3: IDRS vs 4-FEATURE ML AUDIT (E1)
    # -------------------------------------------------------------------------
    logger.info("Conducting IDRS vs 4-feature ML audit on diabetes...")
    idrs_features = ["age", "waist_cm", "activity_level", "family_history_diabetes"]
    idrs_tr_df = train_df[idrs_features + ["target_diabetes"]].dropna().copy()
    idrs_te_df = test_df[idrs_features + ["target_diabetes"]].dropna().copy()

    lr_idrs = LogisticRegression(solver="lbfgs")
    lr_idrs.fit(idrs_tr_df[idrs_features], idrs_tr_df["target_diabetes"])
    ml_idrs_scores = lr_idrs.predict_proba(idrs_te_df[idrs_features])[:, 1]

    idrs_pts = [
        compute_idrs(
            age=r["age"],
            waist_cm=r["waist_cm"],
            is_male=bool(test_df.loc[idx, "sex_male"] == 1),
            activity_level=r["activity_level"],
            family_history_diabetes=r["family_history_diabetes"],
        )["score"]
        for idx, r in idrs_te_df.iterrows()
    ]
    idrs_pts = np.array(idrs_pts)
    y_diab_test = idrs_te_df["target_diabetes"].values

    auc_idrs, ci_auc_idrs = bootstrap_auroc_ci(y_diab_test, idrs_pts)
    auc_ml_4feat, ci_auc_ml = bootstrap_auroc_ci(y_diab_test, ml_idrs_scores)
    delta_auc_idrs, ci_delta_auc_idrs = paired_bootstrap_auroc_diff(
        y_diab_test, ml_idrs_scores, idrs_pts
    )

    idrs_comparison = {
        "sample_size_holdout": int(len(y_diab_test)),
        "diabetes_cases": int(y_diab_test.sum()),
        "prevalence_pct": round(float(y_diab_test.mean() * 100), 2),
        "published_idrs_auroc": auc_idrs,
        "published_idrs_auroc_95ci": ci_auc_idrs,
        "ml_4feature_auroc": auc_ml_4feat,
        "ml_4feature_auroc_95ci": ci_auc_ml,
        "delta_auroc_point": delta_auc_idrs,
        "delta_auroc_95ci": ci_delta_auc_idrs,
        "spectrum_shift_note": (
            "IDRS was developed by Mohan et al. (JAPI 2005) in Chennai, India (CURES cohort) "
            "specifically to screen for UNDIAGNOSED type 2 diabetes in an Asian Indian population. "
            "Evaluating IDRS on US NHANES against self-reported DIAGNOSED diabetes evaluates a different "
            "clinical phenotype (known diagnosis rather than undetected asymptomatic hyperglycemia) "
            "in a multi-ethnic US population."
        ),
    }

    # Assemble Full JSON
    final_audit_record = {
        "metadata": {
            "title": "PulmoAI Future Risk Audit Pass 4 (E3 & E4)",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "evaluation_design": "Temporal holdout (Train: 1999-2010, Test: 2011-2014) linked to CDC Public-Use Linked Mortality Files (Dec 31, 2019 cutoff)",
            "training_cohort_size": int(len(train_df)),
            "holdout_cohort_size": int(len(test_df)),
            "holdout_all_cause_deaths": int(test_df["event_all_cause"].sum()),
            "holdout_cvd_deaths": int(test_df["event_cvd"].sum()),
        },
        "primary_prevention_cvd_comparison_e3": primary_prevention_comparison,
        "subgroup_metrics_with_bootstrap_95ci_e4": subgroups,
        "decile_calibration_all_cause": decile_cal_all,
        "decile_calibration_cardiovascular_e4": decile_cal_cvd,
        "idrs_diabetes_comparison_e1": idrs_comparison,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_audit_record, f, indent=2)
    logger.info("Saved Pass 4 audit JSON to %s", OUTPUT_JSON)

    # Save models
    joblib.dump({
        "cox_full_all": cox_full_all,
        "cph_all": cph_all,
        "cph_cvd": cph_cvd,
        "cox_pce_inputs": cox_pce_inputs,
        "cox_full_pp": cox_full_pp,
        "train_medians": train_medians,
        "mortality_features": mortality_features,
        "pce_features": pce_features,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, OUTPUT_MODELS, compress=3)
    logger.info("Saved model bundle to %s", OUTPUT_MODELS)
    logger.info("Done in %.2f seconds.", time.time() - t0)


if __name__ == "__main__":
    main()
