# Indian Diabetes Risk Score (IDRS) Source Specification

## Source Citation
- **Authors:** V. Mohan, R. Deepa, M. Deepa, S. Somannavar, M. Datta
- **Title:** A simplified Indian Diabetes Risk Score for screening for undiagnosed diabetic subjects
- **Journal:** *Journal of the Association of Physicians of India (JAPI)*, September 2005, Vol. 53, pp. 759–763
- **PMID:** 16334618
- **URL:** https://pubmed.ncbi.nlm.nih.gov/16334618/
- **CURES Cohort:** Developed in the Chennai Urban Rural Epidemiology Study (CURES) on 26,001 individuals (CURES-Phase 1) and validated on 1,402 subjects (CURES-Phase 3) using oral glucose tolerance tests.

## Table 1: Indian Diabetes Risk Score (Numbers Only)

| Parameter | Category / Cutoff | Score |
| :--- | :--- | :--- |
| **Age (years)** | < 35 | 0 |
| | 35 – 49 | 20 |
| | ≥ 50 | 30 |
| **Abdominal Obesity (Waist Circumference)** | Female < 80 cm, Male < 90 cm | 0 |
| | Female 80 – 89 cm, Male 90 – 99 cm | 10 |
| | Female ≥ 90 cm, Male ≥ 100 cm | 20 |
| **Physical Activity** | Regular exercise AND strenuous manual work | 0 |
| | Regular exercise OR strenuous manual work | 20 |
| | No exercise AND sedentary activities | 30 |
| **Family History of Diabetes** | Neither parent diabetic | 0 |
| | One parent diabetic | 10 |
| | Both parents diabetic | 20 |

**Total Score Range:** 0 to 100
- **Low Risk:** < 30
- **Moderate Risk:** 30 – 50 (or < 60)
- **High Risk:** ≥ 60

## Physical Activity Categories and NHANES Proxy Mapping

The original IDRS defines **3 physical activity categories**:
1. **0 points:** Regular vigorous exercise and strenuous manual work.
2. **20 points:** Regular moderate exercise or strenuous manual work (either exercise or strenuous occupational work).
3. **30 points:** No exercise and sedentary activities (neither leisure-time exercise nor manual occupational work).

### NHANES Proxy Mapping
In US NHANES (PAQ / PAD modules):
- **Vigorous activity (0 points):** Individuals reporting vigorous leisure-time physical activity or heavy occupational work (`PAQ605 = 1` or `PAQ650 = 1` or `PAD200 = 1`).
- **Moderate activity (20 points):** Individuals reporting moderate leisure-time physical activity (`PAQ620 = 1` or `PAD320 = 1`) without vigorous work.
- **Sedentary (30 points):** Individuals reporting neither vigorous nor moderate physical activity (`activity_level = 2.0`).

### Family History Mapping
NHANES questionnaire item `MCQ300C` records self-reported diabetes in any close biological relative as binary (`1 = Yes`, `2 = No`), without distinguishing one parent versus both parents. In PulmoAI's NHANES evaluation:
- No family history (`MCQ300C = 2`) maps to **0 points** (neither parent).
- Positive family history (`MCQ300C = 1`) is mapped conservatively to **10 points** (one parent diabetic).
