#!/usr/bin/env python3
"""Run sanity check on Melms et al. (GSE171524) nuclear pseudobulk profiles.

Within Melms only (26 donors: 19 COVID-19, 7 normal controls), performs
Leave-One-Donor-Out Cross-Validation using the project pipeline
(StandardScaler -> feature selection -> linear model with balanced weights).
Reports ROC-AUC, accuracy, balanced accuracy, COVID sensitivity, and control specificity.
Confirms disease signal exists in nuclear single-nucleus RNA-seq data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_NPZ = PROJECT_ROOT / "data" / "processed" / "melms_profiles.npz"


def main() -> None:
    print("=" * 80)
    print("MELMS ET AL. (GSE171524) NUCLEAR PSEUDOBULK SANITY CHECK")
    print("=" * 80)

    if not DATA_NPZ.exists():
        raise FileNotFoundError(f"Melms profiles not found at {DATA_NPZ}")

    with np.load(DATA_NPZ, allow_pickle=True) as data:
        X = data["expression"].astype(np.float32)
        donors = data["donors"].astype(str)
        labels = data["labels"].astype(str)
        features = data["feature_names"].astype(str)

    n_samples, n_features = X.shape
    print(f"Loaded Melms profiles: {n_samples} donors x {n_features} genes")
    print(f"Disease distribution:\n{pd.Series(labels).value_counts().to_string()}")

    # Binary labels: COVID19 = 1, Donor = 0
    y = np.array([1 if l == "COVID19" else 0 for l in labels], dtype=np.int8)

    loo = LeaveOneOut()

    # We evaluate two linear models:
    # 1. LinearSVC with decision_function
    # 2. Logistic Regression (L2, balanced) with predict_proba
    models = {
        "Linear SVM (C=0.1, SelectKBest 500)": {
            "pipe": Pipeline([
                ("select", SelectKBest(score_func=f_classif, k=min(500, n_features))),
                ("scale", StandardScaler()),
                ("clf", LinearSVC(C=0.1, class_weight="balanced", dual="auto", random_state=42)),
            ]),
            "score_type": "decision",
        },
        "Logistic Regression (C=0.1, SelectKBest 500)": {
            "pipe": Pipeline([
                ("select", SelectKBest(score_func=f_classif, k=min(500, n_features))),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=42)),
            ]),
            "score_type": "proba",
        },
        "Linear SVM (C=0.1, All Common Genes)": {
            "pipe": Pipeline([
                ("scale", StandardScaler()),
                ("clf", LinearSVC(C=0.1, class_weight="balanced", dual="auto", random_state=42)),
            ]),
            "score_type": "decision",
        },
    }

    results = {}
    for name, config in models.items():
        pipe = config["pipe"]
        score_type = config["score_type"]

        oof_scores = np.zeros(n_samples, dtype=np.float64)
        oof_preds = np.zeros(n_samples, dtype=np.int8)

        for train_idx, test_idx in loo.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            pipe.fit(X_train, y_train)

            if score_type == "decision":
                score = pipe.decision_function(X_test)[0]
                pred = 1 if score > 0 else 0
            else:
                score = pipe.predict_proba(X_test)[0, 1]
                pred = int(pipe.predict(X_test)[0])

            oof_scores[test_idx[0]] = score
            oof_preds[test_idx[0]] = pred

        auc = roc_auc_score(y, oof_scores)
        acc = accuracy_score(y, oof_preds)
        bal_acc = balanced_accuracy_score(y, oof_preds)
        f1 = f1_score(y, oof_preds, average="binary")
        cm = confusion_matrix(y, oof_preds)
        # cm: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        covid_recall = tp / max(tp + fn, 1)
        donor_recall = tn / max(tn + fp, 1)

        results[name] = {
            "roc_auc": round(float(auc), 4),
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "f1_score": round(float(f1), 4),
            "covid_recall": round(float(covid_recall), 4),
            "donor_recall": round(float(donor_recall), 4),
            "confusion_matrix": cm.tolist(),
        }

        print(f"\nModel: {name}")
        print(f"  ROC-AUC: {auc:.4f} ({auc*100:.1f}%)")
        print(f"  Accuracy: {acc:.4f} | Balanced Accuracy: {bal_acc:.4f}")
        print(f"  COVID-19 Recall (Sensitivity): {tp}/{tp+fn} ({covid_recall:.1%})")
        print(f"  Normal Control Recall (Specificity): {tn}/{tn+fp} ({donor_recall:.1%})")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("\n" + "=" * 80)
    print("SANITY CHECK SUMMARY")
    print("=" * 80)
    for name, r in results.items():
        print(f"{name:<45} : AUC = {r['roc_auc']:.4f} | Bal Acc = {r['balanced_accuracy']:.4f} | COVID Recall = {r['covid_recall']:.1%}")


if __name__ == "__main__":
    main()
