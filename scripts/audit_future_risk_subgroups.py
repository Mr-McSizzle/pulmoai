#!/usr/bin/env python3
"""Audit Subgroup Mortality Risk Metrics for NHANES 2011-2014 Holdout.

Generates reports/audit_future_risk_subgroups.json with separate entries for
All-Cause Mortality and Cardiovascular Mortality.
Handles zero-event subgroups by setting metrics to null with explicit reasons.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_future_risk_subgroups")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nhanes_mortality_linked.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "future_risk_models_v3.joblib"
OUTPUT_JSON = PROJECT_ROOT / "reports" / "audit_future_risk_subgroups.json"

BOOTSTRAP_SEED = 42
N_BOOTSTRAP = 500
RESAMPLING_UNIT = "evaluable_individual_with_replacement"


def compute_subgroup_outcome(
    events: np.ndarray,
    durations: np.ndarray,
    st5: np.ndarray,
    scores: np.ndarray,
    outcome_name: str,
    subgroup_name: str,
    seed: int = BOOTSTRAP_SEED,
    n_boot: int = N_BOOTSTRAP,
) -> dict[str, Any]:
    n_subjects = int(len(events))
    n_events = int(events.sum())
    v5 = np.isfinite(st5)
    n_evaluable_5yr = int(v5.sum())
    n_5yr_events = int(st5[v5].sum()) if n_evaluable_5yr > 0 else 0

    if n_events == 0:
        return {
            "outcome": outcome_name,
            "subgroup": subgroup_name,
            "n_subjects": n_subjects,
            "n_events": 0,
            "n_evaluable_5yr": n_evaluable_5yr,
            "n_5yr_events": 0,
            "c_index": None,
            "c_index_95ci": None,
            "auroc_5yr": None,
            "auroc_5yr_95ci": None,
            "bootstrap_seed": seed,
            "n_boot": n_boot,
            "resampling_unit": RESAMPLING_UNIT,
            "null_reason": "zero_events_in_subgroup",
        }

    # C-index
    pt_c = round(float(concordance_index_censored(events.astype(bool), durations, scores)[0]), 4)
    rng_c = np.random.RandomState(seed)
    c_vals = []
    for _ in range(n_boot):
        idx = rng_c.choice(n_subjects, size=n_subjects, replace=True)
        if events[idx].sum() > 0:
            try:
                c = concordance_index_censored(events[idx].astype(bool), durations[idx], scores[idx])[0]
                c_vals.append(c)
            except Exception:
                pass
    ci_c = [round(float(np.percentile(c_vals, 2.5)), 4), round(float(np.percentile(c_vals, 97.5)), 4)] if c_vals else [pt_c, pt_c]

    # AUROC 5-year
    y_v5 = st5[v5]
    sc_v5 = scores[v5]
    if len(np.unique(y_v5)) < 2:
        pt_auc = None
        ci_auc = None
        auc_reason = "single_class_in_5yr_followup"
    else:
        pt_auc = round(float(roc_auc_score(y_v5, sc_v5)), 4)
        rng_auc = np.random.RandomState(seed)
        n_v5 = len(y_v5)
        auc_vals = []
        for _ in range(n_boot):
            idx = rng_auc.choice(n_v5, size=n_v5, replace=True)
            if len(np.unique(y_v5[idx])) == 2:
                try:
                    auc_vals.append(roc_auc_score(y_v5[idx], sc_v5[idx]))
                except Exception:
                    pass
        ci_auc = [round(float(np.percentile(auc_vals, 2.5)), 4), round(float(np.percentile(auc_vals, 97.5)), 4)] if auc_vals else [pt_auc, pt_auc]
        auc_reason = None

    return {
        "outcome": outcome_name,
        "subgroup": subgroup_name,
        "n_subjects": n_subjects,
        "n_events": n_events,
        "n_evaluable_5yr": n_evaluable_5yr,
        "n_5yr_events": n_5yr_events,
        "c_index": pt_c,
        "c_index_95ci": ci_c,
        "auroc_5yr": pt_auc,
        "auroc_5yr_95ci": ci_auc,
        "bootstrap_seed": seed,
        "n_boot": n_boot,
        "resampling_unit": RESAMPLING_UNIT,
        "null_reason": auc_reason,
    }


def main() -> None:
    logger.info("Loading NHANES test cohort...")
    df = pd.read_parquet(DATA_PATH)
    test_df = df[(df["eligstat"] == 1) & (df["cycle"].isin(["2011-2012", "2013-2014"]))].copy()
    logger.info("Test cohort size: %d", len(test_df))

    bundle = joblib.load(MODEL_PATH)
    mort_feats = bundle["mortality_features"]
    train_meds = bundle["train_medians"]
    X_test = test_df[mort_feats].fillna(train_meds)

    scores_all = bundle["cox_full_all"].predict(X_test)
    scores_cvd = bundle["cph_cvd"].predict_partial_hazard(X_test).values.ravel()

    subgroups_spec = [
        ("non_hispanic_asian", (test_df["race_ethnicity"] == 6.0).values, "Non-Hispanic Asian"),
        ("non_hispanic_white", (test_df["race_ethnicity"] == 3.0).values, "Non-Hispanic White"),
        ("non_hispanic_black", (test_df["race_ethnicity"] == 4.0).values, "Non-Hispanic Black"),
        ("hispanic", (test_df["race_ethnicity"].isin([1.0, 2.0])).values, "Hispanic / Mexican American"),
        ("males", (test_df["sex_male"] == 1).values, "Males"),
        ("females", (test_df["sex_male"] == 0).values, "Females"),
        ("synthetic_zero_event_control", np.zeros(len(test_df), dtype=bool), "Synthetic Zero-Event Control Cohort"),
    ]

    out_data: dict[str, Any] = {
        "metadata": {
            "title": "NHANES 2011-2014 Subgroup Mortality Risk Evaluation",
            "date": "2026-10-17",
            "cohort": "NHANES 2011-2014 Linked Mortality Follow-up",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_boot": N_BOOTSTRAP,
            "resampling_unit": RESAMPLING_UNIT,
            "zero_event_policy": "metrics set to null with reason string; never asserted as 1.0 or trivial maximum",
        },
        "subgroups": {},
    }

    for key, mask, label in subgroups_spec:
        logger.info("Evaluating subgroup: %s (%s, n=%d)...", key, label, int(mask.sum()))
        if mask.sum() == 0:
            out_data["subgroups"][key] = {
                "subgroup_label": label,
                "all_cause_mortality": {
                    "outcome": "all_cause_mortality",
                    "subgroup": label,
                    "n_subjects": 0,
                    "n_events": 0,
                    "c_index": None,
                    "c_index_95ci": None,
                    "auroc_5yr": None,
                    "auroc_5yr_95ci": None,
                    "null_reason": "zero_subjects_in_subgroup",
                },
                "cardiovascular_mortality": {
                    "outcome": "cardiovascular_mortality",
                    "subgroup": label,
                    "n_subjects": 0,
                    "n_events": 0,
                    "c_index": None,
                    "c_index_95ci": None,
                    "auroc_5yr": None,
                    "auroc_5yr_95ci": None,
                    "null_reason": "zero_subjects_in_subgroup",
                },
            }
            continue

        dur = test_df.loc[mask, "followup_years"].values
        ev_all = test_df.loc[mask, "event_all_cause"].values
        st5_all = test_df.loc[mask, "status_5yr_all_cause"].values
        sc_all = scores_all[mask]

        ev_cvd = test_df.loc[mask, "event_cvd"].values
        st5_cvd = test_df.loc[mask, "status_5yr_cvd"].values
        sc_cvd = scores_cvd[mask]

        all_res = compute_subgroup_outcome(ev_all, dur, st5_all, sc_all, "all_cause_mortality", label)
        cvd_res = compute_subgroup_outcome(ev_cvd, dur, st5_cvd, sc_cvd, "cardiovascular_mortality", label)

        out_data["subgroups"][key] = {
            "subgroup_label": label,
            "all_cause_mortality": all_res,
            "cardiovascular_mortality": cvd_res,
        }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    logger.info("Wrote audit subgroup results to %s", OUTPUT_JSON)

    # Print summary for non_hispanic_asian
    asian = out_data["subgroups"]["non_hispanic_asian"]
    print("\n=== NON-HISPANIC ASIAN SUBGROUP RESULTS ===")
    print(f"All-Cause Mortality (n={asian['all_cause_mortality']['n_subjects']}, events={asian['all_cause_mortality']['n_events']}, 5yr_events={asian['all_cause_mortality']['n_5yr_events']}):")
    print(f"  C-index:   {asian['all_cause_mortality']['c_index']}  95% CI: {asian['all_cause_mortality']['c_index_95ci']}")
    print(f"  5yr AUROC: {asian['all_cause_mortality']['auroc_5yr']}  95% CI: {asian['all_cause_mortality']['auroc_5yr_95ci']}")
    print(f"Cardiovascular Mortality (n={asian['cardiovascular_mortality']['n_subjects']}, events={asian['cardiovascular_mortality']['n_events']}, 5yr_events={asian['cardiovascular_mortality']['n_5yr_events']}):")
    print(f"  C-index:   {asian['cardiovascular_mortality']['c_index']}  95% CI: {asian['cardiovascular_mortality']['c_index_95ci']}")
    print(f"  5yr AUROC: {asian['cardiovascular_mortality']['auroc_5yr']}  95% CI: {asian['cardiovascular_mortality']['auroc_5yr_95ci']}")


if __name__ == "__main__":
    main()
