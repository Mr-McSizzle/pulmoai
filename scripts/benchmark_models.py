#!/usr/bin/env python3
"""Compare lung-disease classifiers on cached pseudobulk profiles."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ("Donor", "COPD", "PF", "COVID19")
TOP_FEATURES = 500
BASE_MODEL_NAMES = (
    "Logistic Regression",
    "Linear SVM",
    "RBF SVM",
    "Random Forest",
    "Extra Trees",
    "AdaBoost",
    "Histogram Gradient Boosting",
    "Neural Network (MLP)",
)
ENSEMBLE_NAME = "Hard Voting Ensemble"
MODEL_NAMES = (*BASE_MODEL_NAMES, ENSEMBLE_NAME)

try:
    from xgboost import XGBClassifier
except Exception as error:  # Optional dependency; do not install implicitly.
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = f"{type(error).__name__}: {error}"
else:
    XGBOOST_IMPORT_ERROR = None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
        warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'", category=UserWarning)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(len(LABELS)), average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=np.arange(len(LABELS)), average="weighted", zero_division=0)),
        }


def base_estimator(name: str, feature_count: int, seed: int):
    if name == "Logistic Regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=3000, solver="lbfgs", random_state=seed
            )),
        ])
    if name == "Linear SVM":
        return Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LinearSVC(
                C=0.1, class_weight="balanced", dual="auto", max_iter=10000, random_state=seed
            )),
        ])
    if name == "RBF SVM":
        return Pipeline([
            ("scale", StandardScaler()),
            ("classifier", SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced")),
        ])
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=120, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=2, random_state=seed,
        )
    if name == "Extra Trees":
        return ExtraTreesClassifier(
            n_estimators=120, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=2, random_state=seed,
        )
    if name == "AdaBoost":
        stump = DecisionTreeClassifier(max_depth=2, min_samples_leaf=2, class_weight="balanced", random_state=seed)
        return AdaBoostClassifier(
            estimator=stump, n_estimators=100, learning_rate=0.3, random_state=seed
        )
    if name == "Histogram Gradient Boosting":
        return HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=15, learning_rate=0.08,
            l2_regularization=1.0, class_weight="balanced", random_state=seed,
        )
    if name == "Neural Network (MLP)":
        return Pipeline([
            ("scale", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(64,), alpha=0.01, learning_rate_init=0.001,
                max_iter=250, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=15, random_state=seed, tol=1e-3,
            )),
        ])
    if name == "XGBoost":
        if XGBClassifier is None:
            raise RuntimeError(f"XGBoost unavailable: {XGBOOST_IMPORT_ERROR}")
        return XGBClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
            objective="multi:softmax", num_class=len(LABELS),
            eval_metric="mlogloss", tree_method="hist", n_jobs=2,
            random_state=seed,
        )
    raise ValueError(f"unknown model: {name}")


def make_pipeline(name: str, feature_count: int, seed: int) -> Pipeline:
    if name == ENSEMBLE_NAME:
        members = [
            ("logistic", base_estimator("Logistic Regression", feature_count, seed)),
            ("linear_svm", base_estimator("Linear SVM", feature_count, seed)),
            ("extra_trees", base_estimator("Extra Trees", feature_count, seed)),
        ]
        classifier = VotingClassifier(estimators=members, voting="hard", n_jobs=1)
    else:
        classifier = base_estimator(name, feature_count, seed)
    return Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=min(TOP_FEATURES, feature_count))),
        ("classifier", classifier),
    ])


def load_profiles(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {"counts", "sample_ids", "studies", "orig_ids", "labels", "cell_counts", "feature_names"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"pseudobulk archive is missing arrays: {sorted(missing)}")
        data = {name: archive[name] for name in required}
    labels = data["labels"].astype(str)
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError(f"unexpected disease labels: {unknown}")
    label_to_code = {label: index for index, label in enumerate(LABELS)}
    data["y"] = np.asarray([label_to_code[label] for label in labels], dtype=np.int8)
    data["labels"] = labels
    data["studies"] = data["studies"].astype(str)
    data["sample_ids"] = data["sample_ids"].astype(str)
    data["orig_ids"] = data["orig_ids"].astype(str)
    data["feature_names"] = data["feature_names"].astype(str)
    counts = np.asarray(data["counts"])
    if counts.ndim != 2 or counts.shape[0] != len(labels) or counts.shape[1] != len(data["feature_names"]):
        raise ValueError("counts, labels, and feature names have incompatible dimensions")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("pseudobulk counts must be finite and nonnegative")
    totals = counts.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("every pseudobulk profile must have positive total counts")
    data["expression"] = np.log1p(counts / totals * 1_000_000).astype(np.float32)
    return data


def run_benchmark(data: dict, report_dir: Path, model_dir: Path, seed: int) -> dict:
    expression = data["expression"]
    y = data["y"]
    studies = data["studies"]
    unique_studies = sorted(np.unique(studies).tolist())
    if len(unique_studies) < 3:
        raise ValueError("at least three source studies are required for study-held-out evaluation")
    if set(np.unique(y).tolist()) != set(range(len(LABELS))):
        raise ValueError("all four disease classes must be present")

    available_models = list(MODEL_NAMES)
    unavailable_models = []
    if XGBClassifier is not None:
        available_models.append("XGBoost")
    else:
        unavailable_models.append({
            "model": "XGBoost",
            "status": "not run; package is not installed",
            "details": XGBOOST_IMPORT_ERROR,
        })

    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    outer_splitter = LeaveOneGroupOut()
    candidate_predictions = {name: np.full(len(y), -1, dtype=np.int8) for name in available_models}
    selected_predictions = np.full(len(y), -1, dtype=np.int8)
    fold_rows = []
    inner_rows = []
    fit_seconds = {name: 0.0 for name in available_models}
    started = time.perf_counter()

    for fold_number, (train_idx, test_idx) in enumerate(
        outer_splitter.split(expression, y, groups=studies), start=1
    ):
        heldout_study = str(studies[test_idx[0]])
        train_classes = set(y[train_idx].tolist())
        if train_classes != set(range(len(LABELS))):
            raise ValueError(f"holding out {heldout_study} removes a disease class from training")
        print(f"\nStudy holdout {fold_number}/{len(unique_studies)}: {heldout_study}", flush=True)
        smallest_class = min(int(np.sum(y[train_idx] == label)) for label in range(len(LABELS)))
        n_splits = min(3, smallest_class)
        inner_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + fold_number)
        inner_scores = {}

        for model_name in available_models:
            inner_oof = np.full(len(train_idx), -1, dtype=np.int8)
            for inner_train_local, inner_valid_local in inner_splitter.split(expression[train_idx], y[train_idx]):
                inner_train = train_idx[inner_train_local]
                inner_valid = train_idx[inner_valid_local]
                if set(y[inner_train].tolist()) != set(range(len(LABELS))):
                    raise ValueError(f"an inner fold lacks a class while holding out {heldout_study}")
                estimator = make_pipeline(model_name, expression.shape[1], seed + fold_number)
                fit_start = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(expression[inner_train], y[inner_train])
                fit_seconds[model_name] += time.perf_counter() - fit_start
                inner_oof[inner_valid_local] = estimator.predict(expression[inner_valid]).astype(np.int8)
            scores = metric_dict(y[train_idx], inner_oof)
            inner_scores[model_name] = scores
            inner_rows.append({
                "heldout_study": heldout_study,
                "candidate_model": model_name,
                **scores,
            })

        selected_model = max(
            available_models,
            key=lambda name: (
                inner_scores[name]["macro_f1"],
                inner_scores[name]["balanced_accuracy"],
                -available_models.index(name),
            ),
        )

        fold_candidate_scores = {}
        for model_name in available_models:
            estimator = make_pipeline(model_name, expression.shape[1], seed + fold_number)
            fit_start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(expression[train_idx], y[train_idx])
            fit_seconds[model_name] += time.perf_counter() - fit_start
            candidate_predictions[model_name][test_idx] = estimator.predict(expression[test_idx]).astype(np.int8)
            fold_candidate_scores[model_name] = metric_dict(y[test_idx], candidate_predictions[model_name][test_idx])

        selected_predictions[test_idx] = candidate_predictions[selected_model][test_idx]
        fold_rows.append({
            "heldout_study": heldout_study,
            "test_sample_groups": int(len(test_idx)),
            "selected_model_from_inner_cv": selected_model,
            **metric_dict(y[test_idx], selected_predictions[test_idx]),
            "inner_selected_macro_f1": inner_scores[selected_model]["macro_f1"],
        })
        print(
            f"Selected {selected_model}; unseen-study macro-F1="
            f"{fold_rows[-1]['macro_f1']:.3f}", flush=True,
        )

    if (selected_predictions < 0).any() or any((values < 0).any() for values in candidate_predictions.values()):
        raise RuntimeError("not every sample group received an out-of-fold prediction")

    final_mean_inner = {
        name: {
            metric: float(np.mean([row[metric] for row in inner_rows if row["candidate_model"] == name]))
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        }
        for name in available_models
    }
    final_model_name = max(
        available_models,
        key=lambda name: (
            final_mean_inner[name]["macro_f1"],
            final_mean_inner[name]["balanced_accuracy"],
            -available_models.index(name),
        ),
    )
    comparison_rows = []
    for name in available_models:
        scores = metric_dict(y, candidate_predictions[name])
        comparison_rows.append({
            "model": name,
            "status": "evaluated",
            **scores,
            "mean_inner_macro_f1": final_mean_inner[name]["macro_f1"],
            "mean_inner_balanced_accuracy": final_mean_inner[name]["balanced_accuracy"],
            "fit_seconds_in_nested_cv": round(fit_seconds[name], 2),
            "selected_for_final_fit": name == final_model_name,
        })
    for item in unavailable_models:
        comparison_rows.append({
            "model": item["model"],
            "status": item["status"],
            "accuracy": "",
            "balanced_accuracy": "",
            "macro_f1": "",
            "weighted_f1": "",
            "mean_inner_macro_f1": "",
            "mean_inner_balanced_accuracy": "",
            "fit_seconds_in_nested_cv": "",
            "selected_for_final_fit": False,
        })

    final_estimator = make_pipeline(final_model_name, expression.shape[1], seed)
    fit_start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_estimator.fit(expression, y)
    calibrated = CalibratedClassifierCV(
        final_estimator, method="sigmoid", cv=5,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated.fit(expression, y)
    model_path = model_dir / "ensemble_benchmark_best.joblib"
    joblib.dump({
        "pipeline": calibrated,
        "model_name": final_model_name,
        "feature_names": data["feature_names"].tolist(),
        "label_order": list(LABELS),
        "normalization": "sample-level summed counts -> counts per million -> log1p",
        "sample_grouping": "gse_alias + orig.ident; patient identity is unverified",
        "validation": "nested leave-one-study-out outer evaluation; final model refit on all sample groups",
        "model_input_contract": "one pseudobulk count row in feature_names order; normalize as documented before pipeline.predict",
        "calibration": "Platt scaling (sigmoid) via CalibratedClassifierCV with 5-fold CV on all training profiles",
    }, model_path, compress=3)
    final_fit_seconds = time.perf_counter() - fit_start
    total_seconds = time.perf_counter() - started

    y_names = np.asarray(LABELS, dtype=object)[y]
    selected_names = np.asarray(LABELS, dtype=object)[selected_predictions]
    selected_metrics = metric_dict(y, selected_predictions)
    confusion = confusion_matrix(y, selected_predictions, labels=np.arange(len(LABELS)))
    class_report = classification_report(
        y, selected_predictions, labels=np.arange(len(LABELS)), target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    write_csv(report_dir / "algorithm_comparison.csv", comparison_rows)
    write_csv(report_dir / "inner_model_selection.csv", inner_rows)
    write_csv(report_dir / "outer_study_folds.csv", fold_rows)
    prediction_rows = []
    for index in range(len(y)):
        row = {
            "sample_group_id": str(data["sample_ids"][index]),
            "study": str(studies[index]),
            "orig_ident": str(data["orig_ids"][index]),
            "true_disease": str(y_names[index]),
            "selected_procedure_prediction": str(selected_names[index]),
        }
        row.update({
            f"{name}_prediction": str(np.asarray(LABELS, dtype=object)[candidate_predictions[name][index]])
            for name in available_models
        })
        prediction_rows.append(row)
    write_csv(report_dir / "out_of_fold_predictions.csv", prediction_rows)
    write_csv(report_dir / "unavailable_models.csv", unavailable_models)
    (report_dir / "classification_report.txt").write_text(
        classification_report(y, selected_predictions, labels=np.arange(len(LABELS)), target_names=LABELS, zero_division=0),
        encoding="utf-8",
    )
    np.savetxt(report_dir / "selected_procedure_confusion_matrix.csv", confusion, delimiter=",", fmt="%d")

    comparison_frame = pd.DataFrame([row for row in comparison_rows if row["status"] == "evaluated"])
    order = comparison_frame.sort_values("macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    colors = ["#187c70" if name == final_model_name else "#8295a8" for name in order["model"]]
    ax.barh(order["model"], order["macro_f1"], color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 on held-out studies")
    ax.set_title("Classifier comparison — each entire source study held out")
    for position, value in enumerate(order["macro_f1"]):
        ax.text(float(value) + 0.01, position, f"{value:.3f}", va="center", fontsize=9)
    fig.savefig(report_dir / "algorithm_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    image = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=35, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("True disease")
    ax.set_title("Nested model-selection procedure — unseen-study predictions")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            ax.text(column, row, str(confusion[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(report_dir / "selected_procedure_confusion_matrix.png", dpi=180)
    plt.close(fig)

    summary = {
        "status": "completed; exploratory research benchmark",
        "source_profiles": str(PROJECT_ROOT / "data" / "processed" / "allcells_pseudobulk_counts.npz"),
        "sample_group_count": int(len(y)),
        "source_study_count": int(len(unique_studies)),
        "cells_aggregated_into_profiles": int(np.sum(data["cell_counts"])),
        "feature_count": int(expression.shape[1]),
        "label_order": list(LABELS),
        "sample_group_counts": {label: int(np.sum(data["labels"] == label)) for label in LABELS},
        "validation": {
            "outer": "LeaveOneGroupOut with gse_alias as group; every source study is held out once",
            "inner": "stratified 3-fold sample-group CV within outer training data; used only to select a model per outer fold",
            "final_selection_rule": "highest mean inner-CV macro-F1 across outer training folds; balanced accuracy breaks ties",
            "reported_selected_procedure_metrics": selected_metrics,
            "confusion_matrix_labels_rows_true_columns_predicted": list(LABELS),
            "confusion_matrix": confusion.tolist(),
            "classification_report": class_report,
            "selected_model_by_heldout_study": [
                {"study": row["heldout_study"], "model": row["selected_model_from_inner_cv"]}
                for row in fold_rows
            ],
            "fixed_candidate_heldout_study_metrics": [
                {key: row[key] for key in ("model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}
                for row in comparison_rows if row["status"] == "evaluated"
            ],
        },
        "models": {
            "evaluated": available_models,
            "unavailable": unavailable_models,
            "inner_cv_mean_scores": final_mean_inner,
            "selected_final_algorithm": final_model_name,
            "final_model_path": str(model_path),
        },
        "runtime_seconds": {
            "nested_model_fits_by_candidate": {name: round(seconds, 2) for name, seconds in fit_seconds.items()},
            "final_fit_and_save_seconds": round(final_fit_seconds, 2),
            "total_benchmark_seconds": round(total_seconds, 2),
        },
        "limitations": [
            "The 1,470,730 cells are aggregated into 372 study-scoped source sample/library profiles; they are not 1,470,730 independent labeled examples.",
            "The H5AD has no verified biological patient identifier; sample groups must not be called patients.",
            "Study-held-out evaluation is intentionally difficult; labels are confounded with source study and class coverage is uneven.",
            "Model comparisons are exploratory and the selected model's outer-fold score is for the selection procedure, not a clinical performance claim.",
            "Linear regression and reinforcement learning are excluded because the target is supervised multiclass disease classification.",
            "The hard-voting ensemble uses Logistic Regression, Linear SVM, and Extra Trees; it uses hard voting.",
            "XGBoost was not run because the package is not installed; no package was downloaded.",
        ],
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "xgboost_available": XGBClassifier is not None,
        },
    }
    with (report_dir / "analysis_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    write_results_markdown(report_dir / "RESULTS.md", summary, comparison_rows)
    return summary


def write_results_markdown(path: Path, summary: dict, comparison_rows: list[dict]) -> None:
    metrics = summary["validation"]["reported_selected_procedure_metrics"]
    lines = [
        "# Multi-Algorithm Benchmark",
        "",
        "This benchmark reuses the already-created pseudobulk profiles; it does not reread or copy the 5.9 GB H5AD.",
        "",
        f"- Data: {summary['cells_aggregated_into_profiles']:,} cells represented as {summary['sample_group_count']} sample/library groups, across {summary['source_study_count']} studies.",
        f"- Final development algorithm selected from inner-CV results: **{summary['models']['selected_final_algorithm']}**.",
        f"- Nested selection-procedure unseen-study scores: accuracy **{metrics['accuracy']:.1%}**, balanced accuracy **{metrics['balanced_accuracy']:.1%}**, macro-F1 **{metrics['macro_f1']:.1%}**, weighted-F1 **{metrics['weighted_f1']:.1%}**.",
        "- The saved final model is refit on all profiles after evaluation; report the nested held-out-study scores, not its training fit.",
        "",
        "## Candidate comparison",
        "",
        "| Model | Held-out-study accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | Mean inner macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        (item for item in comparison_rows if item["status"] == "evaluated"),
        key=lambda item: item["macro_f1"], reverse=True,
    ):
        mark = " **(selected for final fit)**" if row["selected_for_final_fit"] else ""
        lines.append(
            f"| {row['model']}{mark} | {row['accuracy']:.1%} | {row['balanced_accuracy']:.1%} | "
            f"{row['macro_f1']:.1%} | {row['weighted_f1']:.1%} | {row['mean_inner_macro_f1']:.1%} |"
        )
    for item in summary["models"]["unavailable"]:
        lines.append(f"| {item['model']} | Not run | Not run | Not run | Not run | Not run |")
    lines.extend([
        "",
        "## How to read this",
        "",
        "Every candidate uses the same 16 leave-one-study-out test folds. The selected final algorithm is determined only from inner-CV scores inside the outer training data; the headline score evaluates that model-selection procedure on unseen studies. Fixed-candidate scores are useful for comparison but should not be treated as an independent external test after choosing a winner.",
        "",
        "The ensemble is hard voting among Logistic Regression, Linear SVM, and Extra Trees. It does not average raw SVM decision scores as if they were calibrated probabilities.",
        "",
        "XGBoost is reported as unavailable because it is not installed in this environment. No dependency was downloaded. Linear regression is for continuous outcomes, and reinforcement learning does not match this labeled classification task.",
        "",
        "This is exploratory research software, not a clinical diagnostic model. See `analysis_summary.json` for per-class metrics, runtime, fold selections, and limitations.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "cd ~/Desktop/\"ml project\"",
        "MPLCONFIGDIR=\"$PWD/.mplconfig\" .venv/bin/python scripts/benchmark_models.py",
        "```",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "allcells_pseudobulk_counts.npz",
        help="Cached pseudobulk profiles from the completed atlas aggregation.",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "reports" / "ensemble_benchmark",
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=PROJECT_ROOT / "models",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.profiles.is_file():
        parser.error(f"profile archive does not exist: {args.profiles}")
    data = load_profiles(args.profiles)
    summary = run_benchmark(data, args.report_dir, args.model_dir, args.seed)
    print(json.dumps({
        "report": str(args.report_dir / "RESULTS.md"),
        "summary": str(args.report_dir / "analysis_summary.json"),
        "selected_model": summary["models"]["selected_final_algorithm"],
        "heldout_study_metrics": summary["validation"]["reported_selected_procedure_metrics"],
        "runtime_seconds": summary["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
