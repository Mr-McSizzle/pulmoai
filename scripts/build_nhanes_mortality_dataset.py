#!/usr/bin/env python3
"""Build linked NHANES-LMF prospective cohort dataset (1999-2014).

Links 8 NHANES cycles (1999-2000 through 2013-2014) to CDC Public-Use Linked Mortality Files.
Harmonizes demographics, vitals, body measures, labs, lifestyle, self-reported chronic diagnoses,
and prospective mortality endpoints (all-cause and cardiovascular death).

Saves to:
- data/processed/nhanes_mortality_linked.parquet
- reports/nhanes_mortality_dictionary.csv
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "nhanes"
LMF_DIR = RAW_DIR / "mortality"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_PARQUET = PROCESSED_DIR / "nhanes_mortality_linked.parquet"
OUTPUT_DICT_CSV = REPORT_DIR / "nhanes_mortality_dictionary.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_mortality_dataset")

CYCLES = [
    ("1999-2000", "1999_2000", ""),
    ("2001-2002", "2001_2002", "_B"),
    ("2003-2004", "2003_2004", "_C"),
    ("2005-2006", "2005_2006", "_D"),
    ("2007-2008", "2007_2008", "_E"),
    ("2009-2010", "2009_2010", "_F"),
    ("2011-2012", "2011_2012", "_G"),
    ("2013-2014", "2013_2014", "_H"),
]


def get_clean(df: pd.DataFrame, *col_names: str) -> pd.Series:
    """Return cleaned numeric series for first matching column, replacing Refused/Don't Know with NaN."""
    for col in col_names:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.replace([7, 9, 77, 99, 777, 999], np.nan)
            return s
    return pd.Series(np.nan, index=df.index, dtype=float)


def get_num(df: pd.DataFrame, *col_names: str) -> pd.Series:
    """Return numeric series for first matching column, or NaN series matching index."""
    for col in col_names:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def parse_lmf(lmf_path: Path) -> pd.DataFrame:
    """Parse CDC NCHS fixed-width Public-Use Linked Mortality File."""
    records = []
    with open(lmf_path, "r", encoding="ascii") as f:
        for line in f:
            seqn = int(line[0:6].strip())
            elig = int(line[14:15].strip()) if line[14:15].strip() else None
            mort = int(line[15:16].strip()) if line[15:16].strip() not in ("", ".") else None
            ucod = int(line[16:19].strip()) if line[16:19].strip() not in ("", ".") else None
            diab = int(line[19:20].strip()) if line[19:20].strip() not in ("", ".") else None
            hyp = int(line[20:21].strip()) if line[20:21].strip() not in ("", ".") else None
            p_int = float(line[42:45].strip()) if line[42:45].strip() not in ("", ".") else None
            p_exm = float(line[45:48].strip()) if line[45:48].strip() not in ("", ".") else None
            records.append({
                "SEQN": seqn,
                "eligstat": elig,
                "mortstat": mort,
                "ucod_leading": ucod,
                "diabetes_mcod": diab,
                "hyperten_mcod": hyp,
                "permth_int": p_int,
                "permth_exm": p_exm,
            })
    return pd.DataFrame(records)


def load_cycle_raw(cycle: str, sfx: str) -> pd.DataFrame:
    """Load and merge XPT files for a single cycle by SEQN."""
    cycle_dir = RAW_DIR / cycle

    files_to_load = []
    if cycle == "1999-2000":
        files_to_load = [
            "DEMO.XPT", "BMX.XPT", "BPX.XPT", "BPQ.XPT", "DIQ.XPT", "MCQ.XPT",
            "SMQ.XPT", "ALQ.XPT", "PAQ.XPT", "LAB13.XPT", "LAB10.XPT", "LAB18.XPT",
            "LAB25.XPT", "LAB10AM.XPT",
        ]
    elif cycle in ("2001-2002", "2003-2004"):
        files_to_load = [
            f"DEMO{sfx}.XPT", f"BMX{sfx}.XPT", f"BPX{sfx}.XPT", f"BPQ{sfx}.XPT",
            f"DIQ{sfx}.XPT", f"MCQ{sfx}.XPT", f"SMQ{sfx}.XPT", f"ALQ{sfx}.XPT",
            f"PAQ{sfx}.XPT", f"L13{sfx}.XPT", f"L10{sfx}.XPT", f"L40{sfx}.XPT",
            f"L25{sfx}.XPT", f"L10AM{sfx}.XPT",
        ]
    else:  # 2005-2014
        files_to_load = [
            f"DEMO{sfx}.XPT", f"BMX{sfx}.XPT", f"BPX{sfx}.XPT", f"BPQ{sfx}.XPT",
            f"DIQ{sfx}.XPT", f"MCQ{sfx}.XPT", f"SMQ{sfx}.XPT", f"ALQ{sfx}.XPT",
            f"PAQ{sfx}.XPT", f"TCHOL{sfx}.XPT", f"HDL{sfx}.XPT", f"GHB{sfx}.XPT",
            f"BIOPRO{sfx}.XPT", f"CBC{sfx}.XPT", f"GLU{sfx}.XPT",
        ]

    demo_fn = files_to_load[0]
    df_cycle = pd.read_sas(cycle_dir / demo_fn, format="xport")
    df_cycle["SEQN"] = df_cycle["SEQN"].astype(int)
    df_cycle["cycle"] = cycle

    for fn in files_to_load[1:]:
        p = cycle_dir / fn
        if not p.exists():
            continue
        try:
            mod_df = pd.read_sas(p, format="xport")
            mod_df["SEQN"] = mod_df["SEQN"].astype(int)
            overlap = [c for c in mod_df.columns if c in df_cycle.columns and c != "SEQN"]
            if overlap:
                mod_df = mod_df.drop(columns=overlap)
            df_cycle = df_cycle.merge(mod_df, on="SEQN", how="left")
        except Exception as e:
            logger.warning(f"Error reading {p}: {e}")

    return df_cycle


def harmonize_cycle(df: pd.DataFrame, lmf_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and standardize variables across cycles."""
    out = pd.DataFrame(index=df.index)
    out["SEQN"] = df["SEQN"].astype(int)
    out["cycle"] = df["cycle"].astype(str)

    # 1. Demographics
    out["age"] = get_num(df, "RIDAGEYR")
    gender = get_num(df, "RIAGENDR")
    out["sex_male"] = np.where(gender == 1, 1.0, np.where(gender == 2, 0.0, np.nan))

    out["race_ethnicity"] = get_num(df, "RIDRETH3", "RIDRETH1")
    out["race_black"] = np.where(out["race_ethnicity"] == 4, 1.0, 0.0)
    out["race_white"] = np.where(out["race_ethnicity"] == 3, 1.0, 0.0)

    out["education"] = get_clean(df, "DMDEDUC2")
    out["poverty_ratio"] = get_num(df, "INDFMPIR")

    # Survey weights
    out["survey_weight_mec"] = get_num(df, "WTMEC2YR", "WTMEC4YR")
    out["survey_weight_int"] = get_num(df, "WTINT2YR", "WTINT4YR")
    out["psu"] = get_num(df, "SDMVPSU")
    out["strata"] = get_num(df, "SDMVSTRA")

    # 2. Body Measures
    out["weight_kg"] = get_num(df, "BMXWT")
    out["height_cm"] = get_num(df, "BMXHT")
    out["bmi"] = get_num(df, "BMXBMI")
    out["waist_cm"] = get_num(df, "BMXWAIST")

    # 3. Blood Pressure
    sys_cols = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3"] if c in df.columns]
    dia_cols = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3"] if c in df.columns]
    sys_vals = df[sys_cols].apply(pd.to_numeric, errors="coerce") if sys_cols else pd.DataFrame(index=df.index)
    dia_vals = df[dia_cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan) if dia_cols else pd.DataFrame(index=df.index)
    out["systolic_bp"] = sys_vals.mean(axis=1) if not sys_vals.empty else np.nan
    out["diastolic_bp"] = dia_vals.mean(axis=1) if not dia_vals.empty else np.nan
    out["pulse_bpm"] = get_num(df, "BPXPLS")

    # Blood pressure medication
    bpq050a = get_clean(df, "BPQ050A")
    bpq040a = get_clean(df, "BPQ040A")
    bp_med = np.where((bpq050a == 1) | (bpq040a == 1), 1.0, 0.0)
    all_bp_med_nan = bpq050a.isna() & bpq040a.isna()
    bp_med = np.where(all_bp_med_nan, np.nan, bp_med)
    out["bp_medication"] = bp_med

    # 4. Lipids
    out["total_cholesterol"] = get_num(df, "LBXTC")
    out["hdl_cholesterol"] = get_num(df, "LBDHDD", "LBDHDL")

    # 5. Glycemic Labs
    out["fasting_glucose"] = get_num(df, "LBXGLU")
    out["hba1c"] = get_num(df, "LBXGH")

    # 6. Biochemistry
    out["serum_creatinine"] = get_num(df, "LBXSCR")
    out["bun"] = get_num(df, "LBXSBU")
    out["albumin"] = get_num(df, "LBXSAL")
    out["uric_acid"] = get_num(df, "LBXSUA")
    out["alt"] = get_num(df, "LBXSATSI")
    out["ast"] = get_num(df, "LBXSASSI")
    out["alk_phosphatase"] = get_num(df, "LBXSAPSI")
    out["calcium"] = get_num(df, "LBXSCA")
    out["potassium"] = get_num(df, "LBXSKSI")
    out["sodium"] = get_num(df, "LBXSNASI")

    # 7. Hematology
    out["wbc_count"] = get_num(df, "LBXWBCSI")
    out["rbc_count"] = get_num(df, "LBXRBCSI")
    out["hemoglobin"] = get_num(df, "LBXHGB")
    out["hematocrit"] = get_num(df, "LBXHCT")
    out["platelets"] = get_num(df, "LBXPLTSI")

    # 8. Lifestyle
    smq020 = get_clean(df, "SMQ020")
    smq040 = get_clean(df, "SMQ040")
    smoking = np.full(len(df), np.nan)
    smoking[smq020 == 2] = 0.0
    smoking[(smq020 == 1) & (smq040 == 3)] = 1.0
    smoking[(smq020 == 1) & (smq040.isin([1, 2]))] = 2.0
    out["smoking_status"] = smoking
    out["current_smoker"] = np.where(smoking == 2.0, 1.0, np.where(np.isin(smoking, [0.0, 1.0]), 0.0, np.nan))

    # Alcohol
    alq130 = get_clean(df, "ALQ130", "ALD130")
    alq101 = get_clean(df, "ALQ101")
    alq110 = get_clean(df, "ALQ110")
    drinks = alq130.copy()
    drinks[(alq101 == 2) | (alq110 == 2)] = 0.0
    out["drinks_per_day"] = drinks

    # Physical activity harmonization
    if "PAQ605" in df.columns:
        p_vig = (get_clean(df, "PAQ605") == 1) | (get_clean(df, "PAQ650") == 1)
        p_mod = (get_clean(df, "PAQ620") == 1) | (get_clean(df, "PAQ665") == 1)
        all_paq_nan = df[["PAQ605", "PAQ620", "PAQ650"]].isna().all(axis=1)
    else:
        p_vig = get_clean(df, "PAD200") == 1
        p_mod = get_clean(df, "PAD320") == 1
        all_paq_nan = df[["PAD200", "PAD320"]].isna().all(axis=1) if "PAD200" in df.columns else pd.Series(True, index=df.index)

    act_level = np.full(len(df), np.nan)
    act_level[p_vig] = 0.0  # Vigorous
    act_level[(~p_vig) & p_mod] = 1.0  # Moderate
    act_level[(~p_vig) & (~p_mod) & (~all_paq_nan)] = 2.0  # Sedentary
    out["activity_level"] = act_level
    out["physically_active"] = np.where(np.isin(act_level, [0.0, 1.0]), 1.0, np.where(act_level == 2.0, 0.0, np.nan))

    # Family history of diabetes
    fam_diab = get_clean(df, "MCQ300C", "MCQ250A")
    out["family_history_diabetes"] = np.where(fam_diab == 1, 1.0, np.where(fam_diab == 2, 0.0, np.nan))

    # 9. Self-Reported Chronic Disease Targets
    diq010 = get_clean(df, "DIQ010")
    out["target_diabetes"] = np.where(diq010 == 1, 1.0, np.where(diq010.isin([2, 3]), 0.0, np.nan))

    bpq020 = get_clean(df, "BPQ020")
    out["target_hypertension"] = np.where(bpq020 == 1, 1.0, np.where(bpq020 == 2, 0.0, np.nan))

    bpq080 = get_clean(df, "BPQ080")
    out["target_high_cholesterol"] = np.where(bpq080 == 1, 1.0, np.where(bpq080 == 2, 0.0, np.nan))

    mcq160c = get_clean(df, "MCQ160C")
    mcq160d = get_clean(df, "MCQ160D")
    mcq160e = get_clean(df, "MCQ160E")
    any_heart_yes = (mcq160c == 1) | (mcq160d == 1) | (mcq160e == 1)
    all_heart_no = (mcq160c == 2) & (mcq160d == 2) & (mcq160e == 2)
    out["target_heart_disease"] = np.where(any_heart_yes, 1.0, np.where(all_heart_no, 0.0, np.nan))

    mcq160f = get_clean(df, "MCQ160F")
    out["target_stroke"] = np.where(mcq160f == 1, 1.0, np.where(mcq160f == 2, 0.0, np.nan))

    mcq160g = get_clean(df, "MCQ160G")
    mcq160k = get_clean(df, "MCQ160K")
    any_copd_yes = (mcq160g == 1) | (mcq160k == 1)
    all_copd_no = (mcq160g == 2) & (mcq160k == 2)
    out["target_copd"] = np.where(any_copd_yes, 1.0, np.where(all_copd_no, 0.0, np.nan))

    # 10. Merge with Linked Mortality File (LMF)
    merged = out.merge(lmf_df, on="SEQN", how="left")

    followup_m = merged["permth_exm"].fillna(merged["permth_int"])
    merged["followup_months"] = followup_m
    merged["followup_years"] = followup_m / 12.0

    is_eligible = merged["eligstat"] == 1
    merged["event_all_cause"] = np.where(is_eligible & (merged["mortstat"] == 1), 1.0, np.where(is_eligible & (merged["mortstat"] == 0), 0.0, np.nan))
    is_cvd_death = is_eligible & (merged["mortstat"] == 1) & (merged["ucod_leading"].isin([1, 5]))
    merged["event_cvd"] = np.where(is_cvd_death, 1.0, np.where(is_eligible, 0.0, np.nan))

    merged["status_5yr_all_cause"] = np.where(
        is_eligible & (merged["event_all_cause"] == 1) & (merged["followup_months"] <= 60), 1.0,
        np.where(is_eligible & (merged["followup_months"] >= 60), 0.0, np.nan)
    )
    merged["status_5yr_cvd"] = np.where(
        is_eligible & (merged["event_cvd"] == 1) & (merged["followup_months"] <= 60), 1.0,
        np.where(is_eligible & (merged["followup_months"] >= 60), 0.0, np.nan)
    )
    merged["status_10yr_all_cause"] = np.where(
        is_eligible & (merged["event_all_cause"] == 1) & (merged["followup_months"] <= 120), 1.0,
        np.where(is_eligible & (merged["followup_months"] >= 120), 0.0, np.nan)
    )
    merged["status_10yr_cvd"] = np.where(
        is_eligible & (merged["event_cvd"] == 1) & (merged["followup_months"] <= 120), 1.0,
        np.where(is_eligible & (merged["followup_months"] >= 120), 0.0, np.nan)
    )

    return merged


def main() -> None:
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("BUILDING HARMONIZED NHANES-LMF DATASET (1999-2014)")
    logger.info("=" * 80)

    cycle_dfs = []
    for cycle, lmf_tag, sfx in CYCLES:
        lmf_path = LMF_DIR / f"NHANES_{lmf_tag}_MORT_2019_PUBLIC.dat"
        if not lmf_path.exists():
            raise FileNotFoundError(f"Missing LMF file: {lmf_path}")
        logger.info(f"Loading Cycle {cycle} (LMF: {lmf_path.name})...")
        lmf_df = parse_lmf(lmf_path)
        raw_df = load_cycle_raw(cycle, sfx)
        clean_cycle = harmonize_cycle(raw_df, lmf_df)
        logger.info(f"  Cycle {cycle}: {len(clean_cycle):,} respondents ({int((clean_cycle['eligstat'] == 1).sum()):,} eligible adults, {int(clean_cycle['event_all_cause'].sum()):,} deaths).")
        cycle_dfs.append(clean_cycle)

    full_df = pd.concat(cycle_dfs, ignore_index=True)
    logger.info("=" * 80)
    logger.info(f"TOTAL MERGED COHORT: {len(full_df):,} participants across 8 cycles (1999-2014)")
    eligible_df = full_df[full_df["eligstat"] == 1]
    logger.info(f"TOTAL ELIGIBLE ADULT COHORT: {len(eligible_df):,} participants")
    logger.info(f"TOTAL ALL-CAUSE DEATHS: {int(eligible_df['event_all_cause'].sum()):,} ({eligible_df['event_all_cause'].mean()*100:.2f}%)")
    logger.info(f"TOTAL CVD DEATHS: {int(eligible_df['event_cvd'].sum()):,} ({eligible_df['event_cvd'].mean()*100:.2f}%)")

    # Partition summary
    train_mask = full_df["cycle"].isin(["1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008", "2009-2010"]) & (full_df["eligstat"] == 1)
    test_mask = full_df["cycle"].isin(["2011-2012", "2013-2014"]) & (full_df["eligstat"] == 1)
    logger.info(f"Training cohort (1999-2010): {train_mask.sum():,} eligible adults, {int(full_df.loc[train_mask, 'event_all_cause'].sum()):,} deaths.")
    logger.info(f"Temporal test holdout (2011-2014): {test_mask.sum():,} eligible adults, {int(full_df.loc[test_mask, 'event_all_cause'].sum()):,} deaths.")

    # Save to parquet
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(OUTPUT_PARQUET, index=False)
    parquet_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
    logger.info(f"Saved parquet to {OUTPUT_PARQUET} ({parquet_mb:.2f} MB)")

    # Data dictionary
    dict_rows = []
    for col in full_df.columns:
        s = full_df[col]
        non_null = int(s.notna().sum())
        pct = round(non_null / len(full_df) * 100, 1)
        dtype_str = str(s.dtype)
        dict_rows.append({
            "column": col,
            "dtype": dtype_str,
            "non_null_count": non_null,
            "completeness_pct": pct,
        })
    dict_df = pd.DataFrame(dict_rows)
    dict_df.to_csv(OUTPUT_DICT_CSV, index=False)
    logger.info(f"Saved data dictionary to {OUTPUT_DICT_CSV}")
    logger.info(f"Total processing time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
