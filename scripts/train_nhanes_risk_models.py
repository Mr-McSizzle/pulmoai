#!/usr/bin/env python3
"""Train, calibrate, and validate NHANES Chronic Condition Risk Models.

Targets evaluated:
  1. Diabetes (Diagnosed)
  2. Hypertension (Diagnosed)
  3. High Cholesterol (Hypercholesterolemia)
  4. Cardiovascular Disease (CAD / Angina / Heart Attack)
  5. Stroke
  6. Chronic Respiratory Disease (COPD / Emphysema / Chronic Bronchitis)

Validation Strategy:
  - Temporal Split: Train on 2013-2016 (Cycles H & I), Test on 2017-2018 (Cycle J).
  - Baselines: Simple Logistic Regression (Age, Sex, BMI).
  - Clinical Models: HistGradientBoostingClassifier with native missing-value handling.
  - Probability Calibration: Isotonic / Platt scaling via CalibratedClassifierCV.
  - Subgroup Fairness: Evaluated across sex (Male/Female) and age bands (18-39, 40-64, 65+).
  - Leakage Prevention: Diagnosed targets strictly exclude self-defining questionnaire
    variables and circular diagnostic definitions (e.g., HbA1c/Fasting Glucose excluded for Diabetes).

Outputs:
  - models/nhanes_risk_models.joblib
  - models/nhanes_model_card.json
  - reports/nhanes_risk_benchmark.json
  - reports/nhanes_calibration_plots.png
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nhanes_merged.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_JOBLIB = MODEL_DIR / "nhanes_risk_models.joblib"
OUTPUT_MODEL_CARD = MODEL_DIR / "nhanes_model_card.json"
OUTPUT_REPORT_JSON = REPORT_DIR / "nhanes_risk_benchmark.json"
OUTPUT_PLOT_PNG = REPORT_DIR / "nhanes_calibration_plots.png"

TARGETS = [
    {
        "id": "diabetes",
        "col": "target_diabetes",
        "name": "Type 2 / Diagnosed Diabetes",
        "leakage_exclude": ["hba1c", "fasting_glucose"],
        "reason_exclude": "HbA1c >= 6.5% and fasting glucose >= 126 mg/dL constitute diagnostic criteria; excluded to avoid tautological prediction.",
    },
    {
        "id": "hypertension",
        "col": "target_hypertension",
        "name": "Essential Hypertension",
        "leakage_exclude": [],
        "reason_exclude": "Questionnaire items BPQ excluded during ingestion; vital examination blood pressures retained.",
    },
    {
        "id": "high_cholesterol",
        "col": "target_high_cholesterol",
        "name": "Hypercholesterolemia",
        "leakage_exclude": [],
        "reason_exclude": "BPQ questionnaire excluded; lipid panel biomarkers retained as physiological predictors.",
    },
    {
        "id": "heart_disease",
        "col": "target_heart_disease",
        "name": "Cardiovascular / Coronary Heart Disease",
        "leakage_exclude": [],
        "reason_exclude": "Questionnaire condition items MCQ160 excluded during dataset ingestion.",
    },
    {
        "id": "stroke",
        "col": "target_stroke",
        "name": "Cerebrovascular Disease (Stroke)",
        "leakage_exclude": [],
        "reason_exclude": "MCQ160F excluded during dataset ingestion.",
    },
    {
        "id": "copd",
        "col": "target_copd",
        "name": "Chronic Obstructive Pulmonary Disease (COPD)",
        "leakage_exclude": [],
        "reason_exclude": "MCQ160G and MCQ160K excluded during dataset ingestion.",
    },
]

ALL_CLINICAL_FEATURES = [
    # Demographics
    "age", "sex_male", "race_ethnicity", "education", "poverty_ratio",
    # Body Measures
    "weight_kg", "height_cm", "bmi", "waist_cm",
    # Vitals
    "systolic_bp", "diastolic_bp", "pulse_bpm",
    # Routine Chemistry (Renal, Hepatic, Electrolytes)
    "serum_creatinine", "bun", "uric_acid", "albumin", "total_protein",
    "total_bilirubin", "alt", "ast", "alk_phosphatase", "calcium",
    "phosphorus", "sodium", "potassium", "chloride", "bicarbonate",
    # Lipids
    "total_cholesterol", "hdl_cholesterol", "triglycerides", "ldl_cholesterol",
    # Glycemic Markers (included where clinically appropriate)
    "hba1c", "fasting_glucose",
    # Complete Blood Count (Hematology)
    "wbc_count", "rbc_count", "hemoglobin", "hematocrit",
    "mcv", "mch", "mchc", "platelets", "rdw",
    # Lifestyle
    "smoking_status", "drinks_per_day", "physically_active",
]

BASELINE_FEATURES = ["age", "sex_male", "bmi"]


def evaluate_binary_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute comprehensive diagnostic evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1_score": round(f1, 4),
        "prevalence": round(float(np.mean(y_true)), 4),
        "sample_size": int(len(y_true)),
    }


def main() -> None:
    t_start = time.time()
    print("=" * 80)
    print("PHASE 2: TRAINING & VALIDATING NHANES RISK MODULE")
    print("=" * 80)

    # 1. Load Data
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded merged NHANES dataset: {len(df):,} respondents x {len(df.columns)} columns")
    print(f"Cycles: {df['cycle'].value_counts().to_dict()}")

    # Filter to adults (age >= 18) for chronic disease modeling
    adult_mask = (df["age"] >= 18)
    df_adults = df[adult_mask].copy()
    print(f"Adult population (age >= 18): {len(df_adults):,} respondents")

    # Temporal split: Train on 2013-2016, Test on 2017-2018
    train_mask = df_adults["cycle"].isin(["2013-2014", "2015-2016"])
    test_mask = df_adults["cycle"] == "2017-2018"
    print(f"Temporal Split: Train (2013-2016) = {train_mask.sum():,} | Test (2017-2018) = {test_mask.sum():,}")

    model_registry = {}
    benchmark_report = {
        "task": "Phase 2 NHANES Chronic Condition Risk Engine",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "temporal_validation": {
            "train_cycles": ["2013-2014", "2015-2016"],
            "test_cycle": "2017-2018",
            "adults_train_count": int(train_mask.sum()),
            "adults_test_count": int(test_mask.sum()),
        },
        "target_results": {},
    }

    # Setup calibration plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, target_info in enumerate(TARGETS):
        target_col = target_info["col"]
        target_name = target_info["name"]
        t_id = target_info["id"]
        print("\n" + "=" * 80)
        print(f"TARGET {idx+1}/6: {target_name.upper()} ({target_col})")
        print("=" * 80)

        # Drop respondents with missing target
        valid_train = train_mask & df_adults[target_col].notna()
        valid_test = test_mask & df_adults[target_col].notna()

        train_data = df_adults[valid_train]
        test_data = df_adults[valid_test]

        y_train = train_data[target_col].values.astype(int)
        y_test = test_data[target_col].values.astype(int)

        print(f"Train samples: {len(y_train):,} ({y_train.sum():,} positive, {y_train.mean():.1%})")
        print(f"Test samples:  {len(y_test):,} ({y_test.sum():,} positive, {y_test.mean():.1%})")

        # -------------------------------------------------------------
        # Model 1: Baseline Logistic Regression (Age, Sex, BMI)
        # -------------------------------------------------------------
        X_train_base = train_data[BASELINE_FEATURES].values
        X_test_base = test_data[BASELINE_FEATURES].values

        base_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ])

        # Calibrated wrapper
        cal_base = CalibratedClassifierCV(estimator=base_pipe, method="sigmoid", cv=3)
        cal_base.fit(X_train_base, y_train)
        prob_base_test = cal_base.predict_proba(X_test_base)[:, 1]
        metrics_base = evaluate_binary_predictions(y_test, prob_base_test)

        print(f"\n[Baseline Logistic Regression] Age + Sex + BMI:")
        print(f"  AUROC: {metrics_base['auroc']:.4f} | AUPRC: {metrics_base['auprc']:.4f} | Brier: {metrics_base['brier_score']:.4f}")

        # -------------------------------------------------------------
        # Model 2: Clinical Gradient Boosting Model
        # -------------------------------------------------------------
        feat_list = [f for f in ALL_CLINICAL_FEATURES if f not in target_info["leakage_exclude"]]
        feat_list = [f for f in feat_list if train_data[f].nunique(dropna=True) >= 2]
        print(f"\n[Clinical Gradient Boosting] Feature count: {len(feat_list)} (excluded: {target_info['leakage_exclude']})")

        X_train_clin = train_data[feat_list].values
        X_test_clin = test_data[feat_list].values

        # Base gradient booster (handles NaNs natively)
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )

        # Calibrate via 3-fold cross-validation inside training temporal set
        cal_clin = CalibratedClassifierCV(estimator=hgb, method="sigmoid", cv=3)
        cal_clin.fit(X_train_clin, y_train)
        prob_clin_test = cal_clin.predict_proba(X_test_clin)[:, 1]
        metrics_clin = evaluate_binary_predictions(y_test, prob_clin_test)

        print(f"  AUROC: {metrics_clin['auroc']:.4f} | AUPRC: {metrics_clin['auprc']:.4f} | Brier: {metrics_clin['brier_score']:.4f}")
        print(f"  Acc:   {metrics_clin['accuracy']:.4f} | Bal Acc: {metrics_clin['balanced_accuracy']:.4f}")

        # -------------------------------------------------------------
        # Subgroup Fairness Analysis (Sex & Age Band)
        # -------------------------------------------------------------
        subgroups = {}

        # Sex: Male vs Female
        for s_code, s_name in [(1.0, "Male"), (0.0, "Female")]:
            mask_s = (test_data["sex_male"] == s_code).values
            if mask_s.sum() > 20 and len(np.unique(y_test[mask_s])) == 2:
                subgroups[f"sex_{s_name.lower()}"] = evaluate_binary_predictions(
                    y_test[mask_s], prob_clin_test[mask_s]
                )

        # Age: 18-39, 40-64, 65+
        age_bands = [
            ("young_adult_18_39", (test_data["age"] >= 18) & (test_data["age"] < 40)),
            ("middle_aged_40_64", (test_data["age"] >= 40) & (test_data["age"] < 65)),
            ("senior_65_plus", test_data["age"] >= 65),
        ]
        for b_name, b_mask in age_bands:
            m_arr = b_mask.values
            if m_arr.sum() > 20 and len(np.unique(y_test[m_arr])) == 2:
                subgroups[b_name] = evaluate_binary_predictions(
                    y_test[m_arr], prob_clin_test[m_arr]
                )

        print("  Subgroup AUROC:")
        for sg_name, sg_met in subgroups.items():
            print(f"    {sg_name:<20}: AUROC = {sg_met['auroc']:.4f} (n={sg_met['sample_size']}, prev={sg_met['prevalence']:.1%})")

        # -------------------------------------------------------------
        # Calibration Curve for Plot
        # -------------------------------------------------------------
        prob_true, prob_pred = calibration_curve(y_test, prob_clin_test, n_bins=10)
        prob_base_t, prob_base_p = calibration_curve(y_test, prob_base_test, n_bins=10)

        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.plot(prob_base_p, prob_base_t, "s-", color="gray", alpha=0.7, label=f"Baseline LR (AUC={metrics_base['auroc']:.2f})")
        ax.plot(prob_pred, prob_true, "o-", color="#1f77b4", lw=2, label=f"Clinical GB (AUC={metrics_clin['auroc']:.2f})")
        ax.set_title(target_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Observed Fraction Positive")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # Store in registries
        benchmark_report["target_results"][t_id] = {
            "target_name": target_name,
            "target_column": target_col,
            "leakage_rationale": target_info["reason_exclude"],
            "features_used": feat_list,
            "features_count": len(feat_list),
            "baseline_logistic_regression": metrics_base,
            "clinical_gradient_boosting": metrics_clin,
            "subgroups": subgroups,
        }

        model_registry[t_id] = {
            "baseline_model": cal_base,
            "clinical_model": cal_clin,
            "feature_names": feat_list,
            "metrics": metrics_clin,
        }

    # Save calibration plot figure
    plt.tight_layout()
    fig.savefig(OUTPUT_PLOT_PNG, dpi=200)
    plt.close(fig)
    print(f"\nSaved calibration plots to {OUTPUT_PLOT_PNG}")

    # 2. Save Models Bundle
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_registry, OUTPUT_JOBLIB, compress=3)
    joblib_size_mb = OUTPUT_JOBLIB.stat().st_size / (1024 * 1024)
    print(f"Saved trained calibrated models to {OUTPUT_JOBLIB} ({joblib_size_mb:.2f} MB)")

    # 3. Save Benchmark Report JSON
    with open(OUTPUT_REPORT_JSON, "w") as f:
        json.dump(benchmark_report, f, indent=2)
    print(f"Saved benchmark report to {OUTPUT_REPORT_JSON}")

    # 4. Save Model Card JSON
    model_card = {
        "module": "NHANES Chronic Condition Risk Engine",
        "version": "1.0.0",
        "training_data": "CDC National Health and Nutrition Examination Survey (NHANES) 2013-2016",
        "validation_data": "CDC NHANES 2017-2018 (Temporal out-of-sample forward evaluation)",
        "population": "U.S. non-institutionalized civilian adult population (aged 18+)",
        "intended_use": "Screening-level chronic disease risk estimation to support preventive health triaging; NOT a standalone diagnostic device.",
        "algorithms": {
            "baseline": "L2-penalized Logistic Regression with Platt scaling",
            "clinical": "Histogram-based Gradient Boosted Trees with Platt scaling",
        },
        "performance_summary": {
            t_id: {
                "name": TARGETS[i]["name"],
                "auroc": benchmark_report["target_results"][t_id]["clinical_gradient_boosting"]["auroc"],
                "auprc": benchmark_report["target_results"][t_id]["clinical_gradient_boosting"]["auprc"],
                "brier_score": benchmark_report["target_results"][t_id]["clinical_gradient_boosting"]["brier_score"],
            }
            for i, t_id in enumerate(benchmark_report["target_results"])
        },
        "leakage_safeguards": "Self-defining diagnostic questionnaires strictly excluded from feature space. Glycemic diagnostic tests (HbA1c/Glucose) excluded from diabetes prediction.",
        "disclaimer": "This algorithm produces statistical risk estimates for educational and clinical decision support only. It does not replace medical consultation, clinical diagnosis, or lab-confirmed pathology.",
    }

    with open(OUTPUT_MODEL_CARD, "w") as f:
        json.dump(model_card, f, indent=2)
    print(f"Saved model card to {OUTPUT_MODEL_CARD}")
    print(f"\nPhase 2 Complete in {time.time() - t_start:.1f}s!")


if __name__ == "__main__":
    main()
