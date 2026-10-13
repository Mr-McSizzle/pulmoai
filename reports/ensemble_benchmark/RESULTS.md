# Multi-Algorithm Benchmark

This benchmark reuses the already-created pseudobulk profiles; it does not reread or copy the 5.9 GB H5AD.

- Data: 1,470,730 cells represented as 372 sample/library groups, across 16 studies.
- Final development algorithm selected from inner-CV results: **Linear SVM**.
- Nested selection-procedure unseen-study scores: accuracy **63.4%**, balanced accuracy **56.1%**, macro-F1 **52.2%**, weighted-F1 **66.7%**.
- The saved final model is refit on all profiles after evaluation; report the nested held-out-study scores, not its training fit.

## Candidate comparison

| Model | Held-out-study accuracy | Balanced accuracy | Macro-F1 | Weighted-F1 | Mean inner macro-F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 64.5% | 56.5% | 54.1% | 67.6% | 90.0% |
| Linear SVM **(selected for final fit)** | 64.8% | 57.9% | 53.5% | 67.9% | 90.3% |
| Hard Voting Ensemble | 63.7% | 52.5% | 49.8% | 66.5% | 89.9% |
| Neural Network (MLP) | 59.1% | 48.3% | 46.1% | 62.4% | 79.2% |
| RBF SVM | 59.9% | 46.1% | 43.5% | 62.0% | 82.7% |
| Histogram Gradient Boosting | 55.4% | 44.3% | 42.5% | 59.1% | 85.6% |
| AdaBoost | 57.0% | 45.7% | 42.2% | 59.1% | 81.1% |
| Random Forest | 56.7% | 44.2% | 42.0% | 59.9% | 82.6% |
| Extra Trees | 57.3% | 42.4% | 40.9% | 59.5% | 82.9% |
| XGBoost | Not run | Not run | Not run | Not run | Not run |

## How to read this

Every candidate uses the same 16 leave-one-study-out test folds. The selected final algorithm is determined only from inner-CV scores inside the outer training data; the headline score evaluates that model-selection procedure on unseen studies. Fixed-candidate scores are useful for comparison but should not be treated as an independent external test after choosing a winner.

The ensemble is hard voting among Logistic Regression, Linear SVM, and Extra Trees. It does not average raw SVM decision scores as if they were calibrated probabilities.

XGBoost is reported as unavailable because it is not installed in this environment. No dependency was downloaded. Linear regression is for continuous outcomes, and reinforcement learning does not match this labeled classification task.

This is exploratory research software, not a clinical diagnostic model. See `analysis_summary.json` for per-class metrics, runtime, fold selections, and limitations.

## Reproduce

```bash
cd ~/Desktop/"ml project"
MPLCONFIGDIR="$PWD/.mplconfig" .venv/bin/python scripts/benchmark_models.py
```
