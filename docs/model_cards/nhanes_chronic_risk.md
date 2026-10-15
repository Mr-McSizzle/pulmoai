# Model Card: NHANES Cross-Sectional Chronic Disease Risk Engine

## Model Details
- **Architecture:** Multi-target penalized Logistic Regression and Gradient Boosted Trees with Platt scaling / isotonic calibration.
- **Deployed Checkpoint:** `models/nhanes_risk_models.joblib`
- **SHA-256 Checkpoint Hash:** `ac86d702fd069919354ca8c20d26168c5a08ce91589ede2eb53f74c8c9a71b7e`
- **Training Data:** `data/processed/nhanes_merged.parquet` (`7023347b0a331165cf45a16df599c235cb58d5cf05bdf9b64ea7d8efd2aa49c6`)
- **Developers:** PulmoAI Machine Learning Research Group.

## Intended Use & Estimand Specification
- **Primary Estimand:** Cross-sectional probability of a CURRENT doctor-diagnosed chronic condition in civilian US adults based on standard biometric and laboratory features.
- **Explicit Estimand Clarification:** This model does NOT predict future prospective disease incidence or lifetime risk. It estimates the statistical likelihood that an adult in a national examination survey with matching biometrics currently reports a diagnosis.
- **Conditions Modeled:**
  1. Diabetes (diagnosed, with HbA1c/glucose withheld from input)
  2. Essential Hypertension (doctor-diagnosed)
  3. Hypercholesterolemia (doctor-diagnosed)
  4. Cardiovascular Disease (prior coronary disease / myocardial infarction)
  5. Cerebrovascular Disease (prior stroke)
  6. Chronic Obstructive Pulmonary Disease (COPD / chronic bronchitis / emphysema)

## Training and Temporal Holdout Evaluation Data
- **Training Cycles:** NHANES Cycles G, H, I (2011-2016; 6 survey years, $n = 12,105$ adult participants).
- **Later-Cycle Temporal Holdout:** NHANES Cycle J (2017-2018; 2 survey years, $n = 5,856$ adult participants).

## Quantitative Metrics on Temporal Holdout (Cycle J 2017-2018)
| Target Condition | AUROC (95% CI) | Survey-Weighted AUROC | AUPRC (Baseline Prev) | Brier Score | Calib. Slope |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Diabetes** | 0.8525 [0.838, 0.867] | 0.8410 | 0.6120 (0.1500) | 0.0968 | 1.042 |
| **Hypertension** | 0.8405 [0.828, 0.853] | 0.8290 | 0.7710 (0.3644) | 0.1574 | 0.985 |
| **High Cholesterol** | 0.7887 [0.774, 0.803] | 0.7750 | 0.6540 (0.3368) | 0.1734 | 1.012 |
| **Heart Disease** | 0.8357 [0.816, 0.855] | 0.8210 | 0.3620 (0.0826) | 0.0652 | 1.080 |
| **Stroke** | 0.7862 [0.758, 0.814] | 0.7680 | 0.2180 (0.0491) | 0.0447 | 1.120 |
| **COPD** | 0.7491 [0.724, 0.774] | 0.7350 | 0.2840 (0.0810) | 0.0690 | 0.965 |

## Known Failure Modes and Subgroup Biases
1. **Low Positive Predictive Value on Rare Conditions:** On low-prevalence outcomes like stroke (4.91% prevalence) and heart disease (8.26% prevalence), precision/PPV remains below 25% despite high AUROC ($>0.78$), leading to false positives if used as a screening cutoff.
2. **Subgroup Heterogeneity:** Older adults ($\ge 80$ years) exhibit compressed discrimination (AUROC drops by 0.10-0.18 across conditions compared to adults aged 20-39).
3. **Self-Reported Label Noise:** Survey labels reflect doctor diagnoses reported by participants; undiagnosed individuals who have never visited a physician are labeled as negatives.

## Out-of-Scope Uses
- Do not use as a standalone diagnostic tool or for clinical treatment decisions.
- Do not use on pediatric populations ($< 18$ years).
- Do not interpret outputs as future lifetime risk or prospective disease incidence.
