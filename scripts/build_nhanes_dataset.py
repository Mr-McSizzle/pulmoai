#!/usr/bin/env python3
"""Merge raw NHANES XPT files (2013-2018) into a clean, leakage-free dataset.

Merges by SEQN across cycles 2013-2014 (H), 2015-2016 (I), and 2017-2018 (J).
Derives standardized clinical, demographic, body measure, lab, and lifestyle features.
Defines 6 chronic disease targets with strict handling of refused/don't know codes.
Saves to data/processed/nhanes_merged.parquet and writes reports/nhanes_data_dictionary.csv.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nhanes"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_PARQUET = PROCESSED_DIR / "nhanes_merged.parquet"
OUTPUT_DICT_CSV = REPORT_DIR / "nhanes_data_dictionary.csv"

CYCLES = {
    "2013-2014": "H",
    "2015-2016": "I",
    "2017-2018": "J",
}


def clean_refused_dontknow(series: pd.Series | object, invalid_values: tuple[int, ...] = (7, 9, 77, 99, 777, 999)) -> pd.Series:
    """Map CDC Refused and Don't Know codes to NaN."""
    s = pd.Series(series) if not isinstance(series, pd.Series) else series.copy()
    s = s.replace(list(invalid_values), np.nan)
    return s


def load_cycle_data(cycle: str, suffix: str) -> pd.DataFrame:
    """Load and merge all 16 XPT files for a single cycle by SEQN."""
    cycle_dir = RAW_DIR / cycle
    print(f"\nProcessing Cycle {cycle} (suffix _{suffix})...")

    # Start with Demographics (all participants)
    demo_path = cycle_dir / f"DEMO_{suffix}.XPT"
    df_cycle = pd.read_sas(demo_path, format="xport")
    df_cycle["SEQN"] = df_cycle["SEQN"].astype(int)
    df_cycle["cycle"] = cycle
    print(f"  Loaded DEMO: {len(df_cycle):,} respondents.")

    other_modules = [
        "BMX", "BPX", "GHB", "GLU", "TCHOL", "HDL", "TRIGLY",
        "BIOPRO", "CBC", "SMQ", "ALQ", "PAQ", "DIQ", "BPQ", "MCQ"
    ]

    for mod in other_modules:
        path = cycle_dir / f"{mod}_{suffix}.XPT"
        if not path.exists():
            print(f"  Warning: {path} not found!")
            continue
        mod_df = pd.read_sas(path, format="xport")
        mod_df["SEQN"] = mod_df["SEQN"].astype(int)

        # Drop duplicate columns other than SEQN
        overlap_cols = [c for c in mod_df.columns if c in df_cycle.columns and c != "SEQN"]
        if overlap_cols:
            mod_df = mod_df.drop(columns=overlap_cols)

        df_cycle = df_cycle.merge(mod_df, on="SEQN", how="left")
        print(f"  Merged {mod:<8}: {len(mod_df):,} records.")

    return df_cycle


def derive_features_and_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Derive clean, standardized features and target variables with data dictionary."""
    data_dict = []
    out = pd.DataFrame(index=df.index)

    # Key / Metadata
    out["SEQN"] = df["SEQN"].astype(int)
    out["cycle"] = df["cycle"].astype(str)
    data_dict.append({"column": "SEQN", "category": "Metadata", "description": "Respondent Sequence Number", "type": "int"})
    data_dict.append({"column": "cycle", "category": "Metadata", "description": "NHANES Survey Cycle", "type": "string"})

    # 1. Demographics
    out["age"] = pd.to_numeric(df.get("RIDAGEYR"), errors="coerce")
    data_dict.append({"column": "age", "category": "Demographics", "description": "Age in years at screening (0-80+)", "type": "float"})

    # Gender: 1=Male, 2=Female
    gender = pd.to_numeric(df.get("RIAGENDR"), errors="coerce")
    out["sex_male"] = np.where(gender == 1, 1.0, np.where(gender == 2, 0.0, np.nan))
    data_dict.append({"column": "sex_male", "category": "Demographics", "description": "Biological sex (1=Male, 0=Female)", "type": "binary"})

    out["race_ethnicity"] = pd.to_numeric(df.get("RIDRETH3"), errors="coerce")
    data_dict.append({"column": "race_ethnicity", "category": "Demographics", "description": "Race/Hispanic origin code (1-7)", "type": "categorical"})

    educ = clean_refused_dontknow(pd.to_numeric(df.get("DMDEDUC2"), errors="coerce"))
    out["education"] = educ
    data_dict.append({"column": "education", "category": "Demographics", "description": "Education level (adults 20+, 1-5)", "type": "categorical"})

    out["poverty_ratio"] = pd.to_numeric(df.get("INDFMPIR"), errors="coerce")
    data_dict.append({"column": "poverty_ratio", "category": "Demographics", "description": "Family income to poverty threshold ratio (0-5)", "type": "float"})

    # 2. Body Measures (BMX)
    out["weight_kg"] = pd.to_numeric(df.get("BMXWT"), errors="coerce")
    data_dict.append({"column": "weight_kg", "category": "Body Measures", "description": "Weight in kg", "type": "float"})

    out["height_cm"] = pd.to_numeric(df.get("BMXHT"), errors="coerce")
    data_dict.append({"column": "height_cm", "category": "Body Measures", "description": "Standing height in cm", "type": "float"})

    out["bmi"] = pd.to_numeric(df.get("BMXBMI"), errors="coerce")
    data_dict.append({"column": "bmi", "category": "Body Measures", "description": "Body Mass Index (kg/m^2)", "type": "float"})

    out["waist_cm"] = pd.to_numeric(df.get("BMXWAIST"), errors="coerce")
    data_dict.append({"column": "waist_cm", "category": "Body Measures", "description": "Waist circumference in cm", "type": "float"})

    # 3. Blood Pressure Exam (BPX)
    # Average of valid readings across readings 1, 2, 3
    sys_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3"] if c in df.columns]
    dia_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3"] if c in df.columns]

    sys_vals = df[sys_cols].apply(pd.to_numeric, errors="coerce")
    dia_vals = df[dia_cols].apply(pd.to_numeric, errors="coerce")

    # Exclude diastolic 0 (often recorded when reading failed)
    dia_vals = dia_vals.replace(0, np.nan)

    out["systolic_bp"] = sys_vals.mean(axis=1)
    data_dict.append({"column": "systolic_bp", "category": "Vitals", "description": "Mean systolic blood pressure (mmHg)", "type": "float"})

    out["diastolic_bp"] = dia_vals.mean(axis=1)
    data_dict.append({"column": "diastolic_bp", "category": "Vitals", "description": "Mean diastolic blood pressure (mmHg)", "type": "float"})

    out["pulse_bpm"] = pd.to_numeric(df.get("BPXPLS"), errors="coerce")
    data_dict.append({"column": "pulse_bpm", "category": "Vitals", "description": "60-second resting pulse (bpm)", "type": "float"})

    # 4. Standard Biochemistry & Labs (BIOPRO)
    biopro_map = {
        "LBXSCR": ("serum_creatinine", "Serum creatinine (mg/dL)"),
        "LBXSBU": ("bun", "Blood urea nitrogen (mg/dL)"),
        "LBXSUA": ("uric_acid", "Serum uric acid (mg/dL)"),
        "LBXSAL": ("albumin", "Serum albumin (g/dL)"),
        "LBXSTP": ("total_protein", "Serum total protein (g/dL)"),
        "LBXSTB": ("total_bilirubin", "Total bilirubin (mg/dL)"),
        "LBXSATSI": ("alt", "Alanine aminotransferase / ALT (U/L)"),
        "LBXSASSI": ("ast", "Aspartate aminotransferase / AST (U/L)"),
        "LBXSAPSI": ("alk_phosphatase", "Alkaline phosphatase (U/L)"),
        "LBXSCA": ("calcium", "Total calcium (mg/dL)"),
        "LBXSPH": ("phosphorus", "Serum phosphorus (mg/dL)"),
        "LBXSNASI": ("sodium", "Serum sodium (mmol/L)"),
        "LBXSKSI": ("potassium", "Serum potassium (mmol/L)"),
        "LBXSCLSI": ("chloride", "Serum chloride (mmol/L)"),
        "LBXSC3SI": ("bicarbonate", "Serum bicarbonate / CO2 (mmol/L)"),
    }
    for raw_c, (new_c, desc) in biopro_map.items():
        out[new_c] = pd.to_numeric(df.get(raw_c), errors="coerce")
        data_dict.append({"column": new_c, "category": "Routine Chemistry", "description": desc, "type": "float"})

    # 5. Lipid Panel (TCHOL, HDL, TRIGLY)
    out["total_cholesterol"] = pd.to_numeric(df.get("LBXTC"), errors="coerce")
    data_dict.append({"column": "total_cholesterol", "category": "Lipids", "description": "Total cholesterol (mg/dL)", "type": "float"})

    out["hdl_cholesterol"] = pd.to_numeric(df.get("LBDHDD"), errors="coerce")
    data_dict.append({"column": "hdl_cholesterol", "category": "Lipids", "description": "HDL-cholesterol (mg/dL)", "type": "float"})

    out["triglycerides"] = pd.to_numeric(df.get("LBXTR"), errors="coerce")
    data_dict.append({"column": "triglycerides", "category": "Lipids", "description": "Fasting triglycerides (mg/dL)", "type": "float"})

    out["ldl_cholesterol"] = pd.to_numeric(df.get("LBDLDL"), errors="coerce")
    data_dict.append({"column": "ldl_cholesterol", "category": "Lipids", "description": "LDL-cholesterol (mg/dL)", "type": "float"})

    # 6. Glycemic Biomarkers (GHB, GLU)
    out["hba1c"] = pd.to_numeric(df.get("LBXGH"), errors="coerce")
    data_dict.append({"column": "hba1c", "category": "Glycemic Labs", "description": "Glycohemoglobin / HbA1c (%)", "type": "float"})

    out["fasting_glucose"] = pd.to_numeric(df.get("LBXGLU"), errors="coerce")
    data_dict.append({"column": "fasting_glucose", "category": "Glycemic Labs", "description": "Fasting plasma glucose (mg/dL)", "type": "float"})

    # 7. Complete Blood Count (CBC)
    cbc_map = {
        "LBXWBCSI": ("wbc_count", "White blood cell count (10^3/uL)"),
        "LBXRBCSI": ("rbc_count", "Red blood cell count (10^6/uL)"),
        "LBXHGB": ("hemoglobin", "Hemoglobin (g/dL)"),
        "LBXHCT": ("hematocrit", "Hematocrit (%)"),
        "LBXMCVSI": ("mcv", "Mean cell volume (fL)"),
        "LBXMCHSI": ("mch", "Mean cell hemoglobin (pg)"),
        "LBXMC": ("mchc", "Mean cell hemoglobin conc (g/dL)"),
        "LBXPLTSI": ("platelets", "Platelet count (10^3/uL)"),
        "LBXRDW": ("rdw", "Red cell distribution width (%)"),
    }
    for raw_c, (new_c, desc) in cbc_map.items():
        out[new_c] = pd.to_numeric(df.get(raw_c), errors="coerce")
        data_dict.append({"column": new_c, "category": "Hematology (CBC)", "description": desc, "type": "float"})

    # 8. Lifestyle (SMQ, ALQ, PAQ)
    # Smoking: SMQ020 (1=Yes, 2=No), SMQ040 (1=Every day, 2=Some days, 3=Not at all)
    smq020 = clean_refused_dontknow(pd.to_numeric(df.get("SMQ020"), errors="coerce"))
    smq040 = clean_refused_dontknow(pd.to_numeric(df.get("SMQ040"), errors="coerce"))
    # Derived smoking status: 0=Never (smq020==2), 1=Former (smq020==1 and smq040==3), 2=Current (smq020==1 and smq040 in [1, 2])
    smoking = np.full(len(df), np.nan)
    smoking[smq020 == 2] = 0.0
    smoking[(smq020 == 1) & (smq040 == 3)] = 1.0
    smoking[(smq020 == 1) & (smq040.isin([1, 2]))] = 2.0
    out["smoking_status"] = smoking
    data_dict.append({"column": "smoking_status", "category": "Lifestyle", "description": "Smoking status (0=Never, 1=Former, 2=Current)", "type": "categorical"})

    # Alcohol drinks per day (ALQ130)
    alq130 = clean_refused_dontknow(pd.to_numeric(df.get("ALQ130"), errors="coerce"))
    # For non-drinkers (ALQ101==2 or ALQ110==2), set drinks to 0
    alq101 = clean_refused_dontknow(pd.to_numeric(df.get("ALQ101"), errors="coerce"))
    alq110 = clean_refused_dontknow(pd.to_numeric(df.get("ALQ110"), errors="coerce"))
    drinks = alq130.copy()
    drinks[(alq101 == 2) | (alq110 == 2)] = 0.0
    out["drinks_per_day"] = drinks
    data_dict.append({"column": "drinks_per_day", "category": "Lifestyle", "description": "Average alcoholic drinks per day (0 for non-drinkers)", "type": "float"})

    # Physical activity: vigorous or moderate activity
    paq605 = clean_refused_dontknow(pd.to_numeric(df.get("PAQ605"), errors="coerce"))
    paq620 = clean_refused_dontknow(pd.to_numeric(df.get("PAQ620"), errors="coerce"))
    paq650 = clean_refused_dontknow(pd.to_numeric(df.get("PAQ650"), errors="coerce"))
    active = np.where((paq605 == 1) | (paq620 == 1) | (paq650 == 1), 1.0, 0.0)
    # If all responses are NaN, set to NaN
    all_paq_nan = paq605.isna() & paq620.isna() & paq650.isna()
    active = np.where(all_paq_nan, np.nan, active)
    out["physically_active"] = active
    data_dict.append({"column": "physically_active", "category": "Lifestyle", "description": "Reports vigorous or moderate physical activity (1=Yes, 0=No)", "type": "binary"})

    # 9. Targets (from Questionnaires)
    # A. Diabetes: DIQ010 (1=Yes, 2=No, 3=Borderline)
    diq010 = clean_refused_dontknow(pd.to_numeric(df.get("DIQ010"), errors="coerce"))
    target_diab = np.where(diq010 == 1, 1.0, np.where(diq010.isin([2, 3]), 0.0, np.nan))
    out["target_diabetes"] = target_diab
    data_dict.append({"column": "target_diabetes", "category": "Target", "description": "Diagnosed Diabetes (DIQ010 == 1)", "type": "binary"})

    # B. Hypertension: BPQ020 (1=Yes, 2=No)
    bpq020 = clean_refused_dontknow(pd.to_numeric(df.get("BPQ020"), errors="coerce"))
    target_htn = np.where(bpq020 == 1, 1.0, np.where(bpq020 == 2, 0.0, np.nan))
    out["target_hypertension"] = target_htn
    data_dict.append({"column": "target_hypertension", "category": "Target", "description": "Diagnosed Hypertension (BPQ020 == 1)", "type": "binary"})

    # C. High Cholesterol: BPQ080 (1=Yes, 2=No)
    bpq080 = clean_refused_dontknow(pd.to_numeric(df.get("BPQ080"), errors="coerce"))
    target_chol = np.where(bpq080 == 1, 1.0, np.where(bpq080 == 2, 0.0, np.nan))
    out["target_high_cholesterol"] = target_chol
    data_dict.append({"column": "target_high_cholesterol", "category": "Target", "description": "Diagnosed High Cholesterol (BPQ080 == 1)", "type": "binary"})

    # D. Heart Disease: MCQ160C (coronary), MCQ160D (angina), MCQ160E (heart attack)
    mcq160c = clean_refused_dontknow(pd.to_numeric(df.get("MCQ160C"), errors="coerce"))
    mcq160d = clean_refused_dontknow(pd.to_numeric(df.get("MCQ160D"), errors="coerce"))
    mcq160e = clean_refused_dontknow(pd.to_numeric(df.get("MCQ160E"), errors="coerce"))
    any_heart_yes = (mcq160c == 1) | (mcq160d == 1) | (mcq160e == 1)
    all_heart_no = (mcq160c == 2) & (mcq160d == 2) & (mcq160e == 2)
    target_heart = np.where(any_heart_yes, 1.0, np.where(all_heart_no, 0.0, np.nan))
    out["target_heart_disease"] = target_heart
    data_dict.append({"column": "target_heart_disease", "category": "Target", "description": "Coronary heart disease, angina, or heart attack (MCQ160C/D/E)", "type": "binary"})

    # E. Stroke: MCQ160F (1=Yes, 2=No)
    mcq160f = clean_refused_dontknow(pd.to_numeric(df.get("MCQ160F"), errors="coerce"))
    target_stroke = np.where(mcq160f == 1, 1.0, np.where(mcq160f == 2, 0.0, np.nan))
    out["target_stroke"] = target_stroke
    data_dict.append({"column": "target_stroke", "category": "Target", "description": "Diagnosed Stroke (MCQ160F == 1)", "type": "binary"})

    # F. COPD: MCQ160G (emphysema == 1) OR MCQ160K (chronic bronchitis == 1)
    mcq160g = clean_refused_dontknow(pd.to_numeric(df["MCQ160G"] if "MCQ160G" in df else np.nan, errors="coerce"))
    mcq160k = clean_refused_dontknow(pd.to_numeric(df["MCQ160K"] if "MCQ160K" in df else np.nan, errors="coerce"))
    any_copd_yes = (mcq160g == 1) | (mcq160k == 1)
    all_copd_no = (mcq160g == 2) & (mcq160k == 2)
    target_copd = np.where(any_copd_yes, 1.0, np.where(all_copd_no, 0.0, np.nan))
    out["target_copd"] = target_copd
    data_dict.append({"column": "target_copd", "category": "Target", "description": "Diagnosed COPD, emphysema, or chronic bronchitis (MCQ160G/K)", "type": "binary"})

    return out, data_dict


def main() -> None:
    t_start = time.time()
    print("=" * 80)
    print("BUILDING MERGED NHANES DATASET (2013-2018)")
    print("=" * 80)

    # 1. Load each cycle
    cycle_dfs = []
    for cycle, suffix in CYCLES.items():
        df_c = load_cycle_data(cycle, suffix)
        cycle_dfs.append(df_c)

    # 2. Concatenate all cycles
    print("\nConcatenating all survey cycles...")
    merged_raw = pd.concat(cycle_dfs, ignore_index=True)
    print(f"Total merged respondents across 3 cycles: {len(merged_raw):,}")

    # 3. Clean and derive features
    print("\nDeriving standardized clinical features and target variables...")
    clean_df, data_dict = derive_features_and_targets(merged_raw)

    # 4. Save Parquet
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(OUTPUT_PARQUET, index=False)
    parquet_size_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
    print(f"\nSaved clean merged dataset to {OUTPUT_PARQUET} ({parquet_size_mb:.2f} MB)")

    # 5. Save Data Dictionary
    dict_df = pd.DataFrame(data_dict)
    # Add non-null counts and percentage
    non_null_counts = clean_df.notna().sum()
    dict_df["non_null_count"] = dict_df["column"].map(non_null_counts)
    dict_df["completeness_pct"] = (dict_df["non_null_count"] / len(clean_df) * 100).round(1)

    dict_df.to_csv(OUTPUT_DICT_CSV, index=False)
    print(f"Saved data dictionary to {OUTPUT_DICT_CSV}")

    # Print Target Distribution Summary
    print("\n" + "=" * 80)
    print("CHRONIC DISEASE TARGET SUMMARY (Adults with recorded response)")
    print("=" * 80)
    target_cols = [c for c in clean_df.columns if c.startswith("target_")]
    for t in target_cols:
        s = clean_df[t].dropna()
        pos = int(s.sum())
        total = len(s)
        prev = (pos / total) * 100
        print(f"  {t:<25} : {pos:>5,} positive / {total:>6,} respondents ({prev:5.1f}% prevalence)")

    print(f"\nElapsed time: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
