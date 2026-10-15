# Model Card: Future Mortality Risk Engine (US Survey Data)

## Model Details
- **Architecture:** Regularized Cox Proportional Hazards (`sksurv.linear_model.CoxPHSurvivalAnalysis` with L2 penalty $\alpha = 0.01$) and Breslow survival baseline estimation (`lifelines.CoxPHFitter`).
- **Artifact Names & Paths:**
  - Deployed Checkpoint: `models/future_risk_models_v3.joblib`
  - Previous Version: `models/future_risk_models_v2.joblib`
- **SHA-256 Checkpoint Hashes:**
  - `future_risk_models_v3.joblib`: `76bc925d2fd9f955547528be5fda0940a755b647005e5bd36ca181c9f89adcc6`
  - `future_risk_models_v2.joblib`: `1e753750d167ad1ef91bbc951d6489020e32d9706d2b2086581e61317d44d2b0`
  - Linked Data (`data/processed/nhanes_mortality_linked.parquet`): `95f87b6495b2cf1c1fcf549b0eb6a7c36b803f29053896dfa209ecbb73ad6ba2`
- **Developers:** PulmoAI Machine Learning Research Group.

## Intended Use & Estimand Specification
- **Primary Estimand:** Future mortality risk over 5-year and 10-year horizons based on continuous follow-up of CDC NHANES participants linked to the National Death Index (NDI) through December 31, 2019.
- **Endpoints Modeled:**
  1. All-Cause Mortality (death from any cause)
  2. Cardiovascular Disease (CVD) Mortality (ICD-10 codes I00-I09, I11, I13, I20-I51, I60-I69)
- **User-Facing Feature Inputs (23 features):** Age, sex, BMI, waist circumference, systolic BP, diastolic BP, resting pulse, BP medication status, total cholesterol, HDL cholesterol, fasting glucose, HbA1c, serum creatinine, BUN, albumin, uric acid, smoking status, alcohol consumption, physical activity, and diagnosed comorbidities (diabetes, hypertension, heart disease, stroke).
- **Fairness & Privacy Policy:** Race and ethnicity variables are strictly excluded from user-facing mortality feature inputs.

## Training and Temporal Holdout Evaluation Data
- **Training Cohort:** 6 NHANES survey cycles (1999-2000 through 2009-2010; 12 survey years, $n = 35,332$ adult participants, 7,733 all-cause deaths, 2,478 CVD deaths).
- **Temporal Holdout Cohort:** 2 later NHANES survey cycles (2011-2012 and 2013-2014; 4 survey years, $n = 11,949$ adult participants, 1,095 all-cause deaths, 347 CVD deaths).

## Quantitative Metrics (2011-2014 Temporal Holdout)
- **All-Cause Mortality (Full Model):**
  - Harrell's C-index: 0.8621 (95% CI: [0.8509, 0.8738])
  - 5-Year AUROC: 0.8713 (95% CI: [0.8599, 0.8854])
  - Survey-Weighted 5-Year AUROC: 0.8559
  - 5-Year Brier Score: 0.0468 (Survey-weighted: 0.0380)
  - Decile Calibration O/E Ratio: 1.007 (95% CI: [0.937, 1.082])
  - Calibration-in-the-Large Intercept: 0.0096
- **Cardiovascular Mortality (Full Model):**
  - Harrell's C-index: 0.8948 (95% CI: [0.8768, 0.9115])
  - 5-Year AUROC: 0.9074 (95% CI: [0.8907, 0.9261])
  - 5-Year Brier Score: 0.0184
  - Decile Calibration O/E Ratio: 1.032 (95% CI: [0.910, 1.170])
  - Calibration-in-the-Large Intercept: 0.0356

## Subgroup Discrimination (Bootstrap 95% CIs)
- **Non-Hispanic Asian Subgroup ($n=1,529$, 53 deaths):**
  - C-index: 0.8618 (95% CI: [0.8032, 0.9140])
  - 5-Year AUROC: 0.8834 (95% CI: [0.8146, 0.9465])
- **Males ($n=5,802$, 596 deaths):** C-index: 0.8527 [0.8347, 0.8683], 5-Yr AUROC: 0.8651 [0.8461, 0.8850]
- **Females ($n=6,147$, 499 deaths):** C-index: 0.8670 [0.8507, 0.8809], 5-Yr AUROC: 0.8775 [0.8571, 0.8949]

## Primary Prevention Comparison (Age 40-79 without Prior CVD, $n=5,899$)
- **Published 2013 ACC/AHA PCE:** C-index 0.7391 [0.6910, 0.7860], 5-Yr AUROC 0.7178 [0.6554, 0.7776]
- **Cox Model with ONLY PCE Inputs (9 features):** C-index 0.7864 [0.7431, 0.8256], 5-Yr AUROC 0.7771 [0.7296, 0.8339]
- **Full Multivariable Cox Model:** C-index 0.8297 [0.7885, 0.8663], 5-Yr AUROC 0.8234 [0.7738, 0.8732]
- **Incremental Value:** Full model adds $\Delta C = +0.0432$ (95% CI: [0.0244, 0.0658]) and $\Delta \text{AUROC} = +0.0463$ (95% CI: [0.0231, 0.0686]) over PCE inputs alone.

## Limitations and Privacy Warnings
1. **NCHS Data Perturbation Notice:** Public-use Linked Mortality Files apply synthetic data substitution techniques to protect participant identity. Population-level survival estimates remain statistically valid, but individual record follow-up times carry synthetic noise.
2. **Clinical Barrier:** Full model gains require specialized venipuncture laboratory assays (`fasting_glucose`, `hba1c`, `serum_creatinine`, `bun`, `albumin`, `uric_acid`) that patients often do not have in routine settings.
3. **Primary Prevention Limitation:** PCE was derived for 10-year fatal + non-fatal hard ASCVD; in NHANES-LMF, only fatal CVD events are observable. PCE calibration is therefore unverified and only discrimination is claimed.
