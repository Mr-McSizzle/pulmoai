#!/usr/bin/env python3
"""Unified Re-Benchmark Harness (Phase 1 / Step 3).

Evaluates:
  A. Raw genes, atlas only, common gene set (new reference)
  B. Raw genes, atlas + Melms (17 studies)
  C. Hallmark pathway scores, atlas only
  D. Hallmark pathway scores, atlas + Melms (17 studies)

Uses ONE consistent evaluation harness across all configs:
  - Leave-One-Study-Out Cross-Validation (LOGO-CV)
  - Fold-local SelectKBest(500, f_classif) inside training folds only
  - Fold-local StandardScaler inside training folds only
  - LinearSVC(C=0.1, class_weight='balanced')
  - Predictions via direct decision boundary (no post-hoc calibration distortion)

Computes:
  - Accuracy, balanced accuracy, macro-F1, per-class precision/recall, confusion matrices
  - Held-out Melms fold metrics (trained on 16 atlas studies, tested on 26 Melms donors)
  - COVID-19 recall split across 3 sub-cohorts: GSE158127, GSE171668, GSE171524
  - Study-clustered bootstrap 95% CIs (1,000 cluster resamples with replacement)
  - Paired differences: A vs C, B vs D, A vs B, C vs D
Saves output to reports/lung_v3_benchmark.json.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_JSON = REPORT_DIR / "lung_v3_benchmark.json"

LABELS = ("Donor", "COPD", "PF", "COVID19")
LABEL_TO_CODE = {l: i for i, l in enumerate(LABELS)}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABELS))), zero_division=0
    )
    per_class = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "precision": round(float(p[i]), 4),
            "recall": round(float(r[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(s[i]),
        }

    conf = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "confusion_matrix": conf.tolist(),
    }


def run_losog_cv(
    X: np.ndarray,
    y: np.ndarray,
    studies: np.ndarray,
    config_name: str,
    top_features: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Execute Leave-One-Study-Out CV with fold-local scaling and feature selection."""
    logo = LeaveOneGroupOut()
    oof_preds = np.full(len(y), -1, dtype=np.int8)

    t0 = time.time()
    for fold_num, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=studies), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Fold-local feature selection
        if top_features is not None and top_features < X_train.shape[1]:
            selector = SelectKBest(score_func=f_classif, k=top_features)
            X_train_sel = selector.fit_transform(X_train, y_train)
            X_test_sel = selector.transform(X_test)
        else:
            X_train_sel = X_train
            X_test_sel = X_test

        # Fold-local scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        # Classifier
        clf = LinearSVC(
            C=0.1,
            class_weight="balanced",
            dual="auto",
            max_iter=10000,
            random_state=seed + fold_num,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train_scaled, y_train)

        oof_preds[test_idx] = clf.predict(X_test_scaled)

    elapsed = time.time() - t0
    metrics = compute_metrics(y, oof_preds)
    metrics["runtime_seconds"] = round(elapsed, 2)
    metrics["n_samples"] = len(y)
    metrics["n_features"] = int(X.shape[1])
    metrics["n_studies"] = int(len(np.unique(studies)))

    return oof_preds, metrics


def compute_study_clustered_bootstrap(
    study_list: list[str],
    y_true: np.ndarray,
    preds_dict: dict[str, np.ndarray],
    studies: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    """Perform study-level cluster bootstrap resampling with replacement."""
    rng = np.random.RandomState(seed)
    unique_studies = np.array(study_list)
    n_studies = len(unique_studies)

    # Pre-compute sample indices per study
    study_indices = {s: np.where(studies == s)[0] for s in unique_studies}

    boot_results = {k: {"macro_f1": [], "balanced_accuracy": []} for k in preds_dict}

    for _ in range(n_bootstraps):
        sampled_studies = rng.choice(unique_studies, size=n_studies, replace=True)
        sampled_sample_indices = np.concatenate([study_indices[s] for s in sampled_studies])

        y_boot = y_true[sampled_sample_indices]

        # Only evaluate if all classes present in bootstrap sample
        if len(np.unique(y_boot)) < len(LABELS):
            continue

        for k, p in preds_dict.items():
            p_boot = p[sampled_sample_indices]
            mf1 = f1_score(y_boot, p_boot, average="macro", zero_division=0)
            bacc = balanced_accuracy_score(y_boot, p_boot)
            boot_results[k]["macro_f1"].append(mf1)
            boot_results[k]["balanced_accuracy"].append(bacc)

    ci_out = {}
    for k in preds_dict:
        mf1_arr = np.array(boot_results[k]["macro_f1"])
        bacc_arr = np.array(boot_results[k]["balanced_accuracy"])
        ci_out[k] = {
            "macro_f1_95ci": [round(float(np.percentile(mf1_arr, 2.5)), 4), round(float(np.percentile(mf1_arr, 97.5)), 4)],
            "balanced_accuracy_95ci": [round(float(np.percentile(bacc_arr, 2.5)), 4), round(float(np.percentile(bacc_arr, 97.5)), 4)],
        }

    # Paired differences
    paired_comparisons = [
        ("A", "C", "Pathways vs Genes (Atlas Only)"),
        ("B", "D", "Pathways vs Genes (Atlas + Melms)"),
        ("A", "B", "Adding Melms to Raw Genes"),
        ("C", "D", "Adding Melms to Pathways"),
    ]
    paired_out = {}
    for k1, k2, desc in paired_comparisons:
        if k1 in preds_dict and k2 in preds_dict:
            diff_mf1 = np.array(boot_results[k2]["macro_f1"]) - np.array(boot_results[k1]["macro_f1"])
            diff_bacc = np.array(boot_results[k2]["balanced_accuracy"]) - np.array(boot_results[k1]["balanced_accuracy"])
            paired_out[f"{k2}_minus_{k1}"] = {
                "description": desc,
                "diff_macro_f1_mean": round(float(np.mean(diff_mf1)), 4),
                "diff_macro_f1_95ci": [round(float(np.percentile(diff_mf1, 2.5)), 4), round(float(np.percentile(diff_mf1, 97.5)), 4)],
                "diff_balanced_accuracy_mean": round(float(np.mean(diff_bacc)), 4),
                "diff_balanced_accuracy_95ci": [round(float(np.percentile(diff_bacc, 2.5)), 4), round(float(np.percentile(diff_bacc, 97.5)), 4)],
            }

    return {"individual_cis": ci_out, "paired_differences": paired_out}


def main() -> None:
    print("=" * 80)
    print("PHASE 1 / STEP 3: UNIFIED RE-BENCHMARK (ONE HARNESS FOR EVERYTHING)")
    print("=" * 80)

    # 1. Load Atlas Data
    with np.load(DATA_DIR / "allcells_pseudobulk_counts.npz", allow_pickle=True) as a:
        atlas_counts = a["counts"]
        atlas_studies = a["studies"].astype(str)
        atlas_labels = a["labels"].astype(str)
        atlas_features = a["feature_names"].astype(str)
        atlas_sample_ids = a["sample_ids"].astype(str)

    y_atlas = np.array([LABEL_TO_CODE[l] for l in atlas_labels], dtype=np.int8)

    # Normalize Atlas Raw CPM -> log1p
    atlas_lib = atlas_counts.sum(axis=1, keepdims=True)
    atlas_cpm = (atlas_counts / np.maximum(atlas_lib, 1)) * 1e6
    atlas_logcpm_full = np.log1p(atlas_cpm).astype(np.float32)

    # Load Atlas Pathways
    X_pathways_atlas = np.load(DATA_DIR / "atlas_pathway_hallmark.npy").astype(np.float32)

    # 2. Load Melms Data
    melms_npz_path = DATA_DIR / "melms_profiles.npz"
    if not melms_npz_path.exists():
        raise FileNotFoundError(f"Melms profiles not found at {melms_npz_path}. Run build_melms_pseudobulk.py first.")

    with np.load(melms_npz_path, allow_pickle=True) as m:
        X_genes_melms = m["expression"].astype(np.float32)
        melms_studies = m["studies"].astype(str)
        melms_labels = m["labels"].astype(str)
        common_genes = m["feature_names"].astype(str)
        melms_donors = m["donors"].astype(str)

    y_melms = np.array([LABEL_TO_CODE[l] for l in melms_labels], dtype=np.int8)

    # Load Melms Pathways
    melms_pw_path = DATA_DIR / "melms_pathway_hallmark.npy"
    if not melms_pw_path.exists():
        raise FileNotFoundError(f"Melms pathways not found at {melms_pw_path}.")
    X_pathways_melms = np.load(melms_pw_path).astype(np.float32)

    # Restrict Atlas Raw Genes to COMMON GENE SET
    atlas_feat_map = {f: i for i, f in enumerate(atlas_features)}
    atlas_common_indices = [atlas_feat_map[g] for g in common_genes]
    X_genes_atlas = atlas_logcpm_full[:, atlas_common_indices]

    print(f"Dataset summary:")
    print(f"  Atlas: {len(y_atlas)} profiles across {len(np.unique(atlas_studies))} studies | {len(common_genes)} common genes")
    print(f"  Melms: {len(y_melms)} profiles (GSE171524) | {len(common_genes)} common genes")
    print(f"  Pathways: {X_pathways_atlas.shape[1]} Hallmark pathways")

    # Combine Atlas + Melms
    X_genes_all = np.vstack([X_genes_atlas, X_genes_melms])
    X_pathways_all = np.vstack([X_pathways_atlas, X_pathways_melms])
    y_all = np.concatenate([y_atlas, y_melms])
    studies_all = np.concatenate([atlas_studies, melms_studies])
    sample_ids_all = np.concatenate([atlas_sample_ids, melms_donors])

    print(f"Combined 17-study cohort: {len(y_all)} profiles across {len(np.unique(studies_all))} studies.")
    print(f"Combined disease counts: {pd.Series([LABELS[i] for i in y_all]).value_counts().to_dict()}")

    # 3. Run Benchmark Configurations
    configs = {
        "A": {
            "name": "Raw genes, atlas only (Common Gene Set)",
            "X": X_genes_atlas,
            "y": y_atlas,
            "studies": atlas_studies,
            "top_features": 500,
        },
        "B": {
            "name": "Raw genes, atlas + Melms (17 studies)",
            "X": X_genes_all,
            "y": y_all,
            "studies": studies_all,
            "top_features": 500,
        },
        "C": {
            "name": "Hallmark pathway scores, atlas only",
            "X": X_pathways_atlas,
            "y": y_atlas,
            "studies": atlas_studies,
            "top_features": None,
        },
        "D": {
            "name": "Hallmark pathway scores, atlas + Melms (17 studies)",
            "X": X_pathways_all,
            "y": y_all,
            "studies": studies_all,
            "top_features": None,
        },
    }

    benchmark_results = {}
    preds_dict = {}

    for key, cfg in configs.items():
        print(f"\n--- Running Configuration {key}: {cfg['name']} ---")
        oof_preds, m = run_losog_cv(
            cfg["X"], cfg["y"], cfg["studies"], cfg["name"], top_features=cfg["top_features"]
        )
        benchmark_results[key] = {
            "config_key": key,
            "description": cfg["name"],
            "metrics": m,
        }
        preds_dict[key] = oof_preds
        print(f"  Accuracy: {m['accuracy']:.4f} | Balanced Accuracy: {m['balanced_accuracy']:.4f} | Macro-F1: {m['macro_f1']:.4f}")
        for l in LABELS:
            pcm = m["per_class"][l]
            print(f"    {l:<7}: Prec={pcm['precision']:.3f}, Rec={pcm['recall']:.3f}, F1={pcm['f1']:.3f} (n={pcm['support']})")

    # 4. Held-out Melms Fold Metrics (Configurations B and D)
    print("\n" + "=" * 80)
    print("HELD-OUT MELMS FOLD ANALYSIS (Trained on 16 Atlas Studies, Tested on Melms)")
    print("=" * 80)
    melms_mask = (studies_all == "GSE171524")
    y_melms_true = y_all[melms_mask]

    held_out_melms_analysis = {}
    for key in ["B", "D"]:
        p_melms = preds_dict[key][melms_mask]
        m_melms = compute_metrics(y_melms_true, p_melms)
        held_out_melms_analysis[key] = {
            "description": configs[key]["name"],
            "metrics": m_melms,
        }
        print(f"\nConfiguration {key} on Melms (n=26):")
        print(f"  Accuracy: {m_melms['accuracy']:.4f} | Balanced Accuracy: {m_melms['balanced_accuracy']:.4f} | Macro-F1: {m_melms['macro_f1']:.4f}")
        for l in ["Donor", "COVID19"]:
            pcm = m_melms["per_class"][l]
            print(f"    {l:<7}: Recall = {pcm['recall']:.1%} ({int(pcm['recall']*pcm['support'])}/{pcm['support']})")

    # 5. COVID-19 Recall Split by Sub-Cohort
    print("\n" + "=" * 80)
    print("COVID-19 RECALL SPLIT BY SUB-COHORT")
    print("=" * 80)
    # Target cohorts: GSE158127, GSE171668, GSE171524
    covid_cohorts = ["GSE158127", "GSE171668", "GSE171524"]
    covid_subcohort_results = {}

    for c in covid_cohorts:
        c_mask = (studies_all == c) & (y_all == LABEL_TO_CODE["COVID19"])
        n_covid = int(np.sum(c_mask))
        covid_subcohort_results[c] = {"total_covid": n_covid, "models": {}}
        print(f"\nCohort {c} (Total COVID-19 samples = {n_covid}):")
        for key in ["A", "B", "C", "D"]:
            if key in ["A", "C"] and c == "GSE171524":
                continue
            if key in ["A", "C"]:
                c_mask_atlas = (atlas_studies == c) & (y_atlas == LABEL_TO_CODE["COVID19"])
                corr = int(np.sum(preds_dict[key][c_mask_atlas] == LABEL_TO_CODE["COVID19"]))
                rec = corr / max(n_covid, 1)
            else:
                corr = int(np.sum(preds_dict[key][c_mask] == LABEL_TO_CODE["COVID19"]))
                rec = corr / max(n_covid, 1)
            covid_subcohort_results[c]["models"][key] = {
                "correct": corr,
                "recall": round(float(rec), 4),
            }
            print(f"  Config {key} ({configs[key]['name'][:30]}...): Recall = {corr}/{n_covid} ({rec:.1%})")

    # 6. Study-Clustered Bootstrap 95% CIs
    print("\n" + "=" * 80)
    print("COMPUTING STUDY-CLUSTERED BOOTSTRAP 95% CIs (1,000 resamples of studies)...")
    print("=" * 80)

    # For Atlas-only (A, C)
    atlas_study_list = sorted(np.unique(atlas_studies).tolist())
    boot_atlas = compute_study_clustered_bootstrap(
        atlas_study_list, y_atlas, {"A": preds_dict["A"], "C": preds_dict["C"]}, atlas_studies, n_bootstraps=1000
    )

    # For Atlas + Melms (B, D)
    all_study_list = sorted(np.unique(studies_all).tolist())
    boot_all = compute_study_clustered_bootstrap(
        all_study_list, y_all, {"B": preds_dict["B"], "D": preds_dict["D"]}, studies_all, n_bootstraps=1000
    )

    # Combined paired difference between A and B, C and D:
    # Need bootstrap across common resamples
    # In common resamples across 16 atlas studies + Melms:
    rng = np.random.RandomState(42)
    boot_diffs = {
        "B_minus_A": {"macro_f1": [], "balanced_accuracy": []},
        "D_minus_C": {"macro_f1": [], "balanced_accuracy": []},
    }
    # For A and C on 17-study bootstrap, map atlas predictions (Melms test samples are ignored for A/C)
    for _ in range(1000):
        sampled_studies = rng.choice(all_study_list, size=len(all_study_list), replace=True)
        # 17-study sample
        idx_17 = np.concatenate([np.where(studies_all == s)[0] for s in sampled_studies])
        # 16-study subset for A & C
        sampled_atlas = [s for s in sampled_studies if s != "GSE171524"]
        if not sampled_atlas:
            continue
        idx_atlas = np.concatenate([np.where(atlas_studies == s)[0] for s in sampled_atlas])

        y_17 = y_all[idx_17]
        y_at = y_atlas[idx_atlas]

        if len(np.unique(y_17)) < len(LABELS) or len(np.unique(y_at)) < len(LABELS):
            continue

        mf1_B = f1_score(y_17, preds_dict["B"][idx_17], average="macro", zero_division=0)
        mf1_A = f1_score(y_at, preds_dict["A"][idx_atlas], average="macro", zero_division=0)
        bacc_B = balanced_accuracy_score(y_17, preds_dict["B"][idx_17])
        bacc_A = balanced_accuracy_score(y_at, preds_dict["A"][idx_atlas])

        mf1_D = f1_score(y_17, preds_dict["D"][idx_17], average="macro", zero_division=0)
        mf1_C = f1_score(y_at, preds_dict["C"][idx_atlas], average="macro", zero_division=0)
        bacc_D = balanced_accuracy_score(y_17, preds_dict["D"][idx_17])
        bacc_C = balanced_accuracy_score(y_at, preds_dict["C"][idx_atlas])

        boot_diffs["B_minus_A"]["macro_f1"].append(mf1_B - mf1_A)
        boot_diffs["B_minus_A"]["balanced_accuracy"].append(bacc_B - bacc_A)
        boot_diffs["D_minus_C"]["macro_f1"].append(mf1_D - mf1_C)
        boot_diffs["D_minus_C"]["balanced_accuracy"].append(bacc_D - bacc_C)

    combined_paired = {
        "A_vs_C (Pathways vs Genes, Atlas Only)": boot_atlas["paired_differences"]["C_minus_A"],
        "B_vs_D (Pathways vs Genes, Atlas + Melms)": boot_all["paired_differences"]["D_minus_B"],
        "A_vs_B (Adding Melms to Raw Genes)": {
            "description": "B (Atlas+Melms) minus A (Atlas Only)",
            "diff_macro_f1_mean": round(float(np.mean(boot_diffs["B_minus_A"]["macro_f1"])), 4),
            "diff_macro_f1_95ci": [round(float(np.percentile(boot_diffs["B_minus_A"]["macro_f1"], 2.5)), 4), round(float(np.percentile(boot_diffs["B_minus_A"]["macro_f1"], 97.5)), 4)],
            "diff_balanced_accuracy_mean": round(float(np.mean(boot_diffs["B_minus_A"]["balanced_accuracy"])), 4),
            "diff_balanced_accuracy_95ci": [round(float(np.percentile(boot_diffs["B_minus_A"]["balanced_accuracy"], 2.5)), 4), round(float(np.percentile(boot_diffs["B_minus_A"]["balanced_accuracy"], 97.5)), 4)],
        },
        "C_vs_D (Adding Melms to Pathways)": {
            "description": "D (Atlas+Melms) minus C (Atlas Only)",
            "diff_macro_f1_mean": round(float(np.mean(boot_diffs["D_minus_C"]["macro_f1"])), 4),
            "diff_macro_f1_95ci": [round(float(np.percentile(boot_diffs["D_minus_C"]["macro_f1"], 2.5)), 4), round(float(np.percentile(boot_diffs["D_minus_C"]["macro_f1"], 97.5)), 4)],
            "diff_balanced_accuracy_mean": round(float(np.mean(boot_diffs["D_minus_C"]["balanced_accuracy"])), 4),
            "diff_balanced_accuracy_95ci": [round(float(np.percentile(boot_diffs["D_minus_C"]["balanced_accuracy"], 2.5)), 4), round(float(np.percentile(boot_diffs["D_minus_C"]["balanced_accuracy"], 97.5)), 4)],
        },
    }

    bootstrap_summary = {
        "atlas_configs": boot_atlas["individual_cis"],
        "all_configs": boot_all["individual_cis"],
        "paired_differences": combined_paired,
    }

    print("\nBootstrap 95% Confidence Intervals:")
    for k in ["A", "C"]:
        print(f"  Config {k}: Macro-F1 95% CI: {boot_atlas['individual_cis'][k]['macro_f1_95ci']} | Bal Acc 95% CI: {boot_atlas['individual_cis'][k]['balanced_accuracy_95ci']}")
    for k in ["B", "D"]:
        print(f"  Config {k}: Macro-F1 95% CI: {boot_all['individual_cis'][k]['macro_f1_95ci']} | Bal Acc 95% CI: {boot_all['individual_cis'][k]['balanced_accuracy_95ci']}")

    print("\nPaired Differences (95% CI):")
    for comp_name, data in combined_paired.items():
        print(f"  {comp_name}:")
        print(f"    Macro-F1 Diff: {data['diff_macro_f1_mean']:+.4f} (95% CI: {data['diff_macro_f1_95ci']})")
        print(f"    Bal Acc Diff:  {data['diff_balanced_accuracy_mean']:+.4f} (95% CI: {data['diff_balanced_accuracy_95ci']})")

    # 7. Assemble Final JSON Report
    final_report = {
        "task": "Phase 1 / Step 3 Unified Lung Disease Cross-Study Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "methodology": {
            "harness": "Leave-One-Study-Out Cross-Validation (LOGO-CV)",
            "pipeline": "SelectKBest(500, f_classif) -> StandardScaler -> LinearSVC(C=0.1, class_weight='balanced')",
            "no_leakage_guarantee": "Feature selection and scaling fit inside training folds only; test studies strictly unseen",
            "common_gene_set_size": len(common_genes),
            "calibration_note": "Uncalibrated decision boundary used for evaluation to prevent sigmoid probability distortion on severely imbalanced rare classes (COVID19)",
        },
        "configurations": benchmark_results,
        "held_out_melms_analysis": held_out_melms_analysis,
        "covid_subcohort_analysis": covid_subcohort_results,
        "bootstrap_results": bootstrap_summary,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSaved complete benchmark report to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
