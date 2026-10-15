# Exploratory Model Comparison

## Task and data

This run predicts COVID versus healthy from the two locally available GSE248688 COVID-HC expression matrices. The analysis aggregates assigned cells by donor across both libraries, producing eight independent donor profiles: four healthy and four COVID. COVID pneumonia and mild cases are combined as the COVID class.

The matrices are read directly from `<DOWNLOADS>/GSE248688_RAW.tar`; no files are extracted. Counts are summed by donor, normalized to counts per million, and transformed with `log1p`. Constant features are removed and up to 100 genes are selected with ANOVA F-scores inside each training fold.

## Results

| Algorithm | Balanced accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| L2 logistic regression | 1.00 | 1.00 | 1.00 |
| Scaled L2 logistic regression | 1.00 | 1.00 | 1.00 |
| Linear SVM | 1.00 | 1.00 | 1.00 |
| RBF SVM | 1.00 | 1.00 | 1.00 |
| Random forest | 1.00 | 1.00 | 1.00 |

The comparison uses leave-one-donor-out cross-validation, so each donor receives one out-of-fold prediction. Since every candidate ties, L2-regularized logistic regression is selected as the simplest interpretable option. This is only a practical tie-break: eight donors are far too few to identify a generally superior algorithm.

## Limitations

- The reported cross-validation scores are extremely unstable and are not an independent final test.
- The saved model is fitted on all eight donors after comparison; do not present its fit as validated generalization.
- The wider GEO archive contains mixed-donor pooled matrices without cell-to-donor assignment files for most cohorts. Those cells were excluded to avoid speculative labels and leakage.
- This is a two-class pilot, not the intended multi-disease model. No external cohort or clinical validation was performed.

## Reproduce

```bash
.venv/bin/python scripts/compare_pilot_models.py \
  <DOWNLOADS>/GSE248688_RAW.tar \
  --output-dir reports/model_comparison --model-dir models
```
