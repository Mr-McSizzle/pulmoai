# Model card: pooled lung transcriptomic classifier

## Artifact and input contract

The default service loads `models/lung_uncalibrated_dual_rule.joblib` (SHA-256 `f56ef0883e3d1e00f7c496e4b9f8edd9492b3c71210afcf08b7ba860d0fca043`). It uses a balanced LinearSVC decision rule, with separately calibrated probability estimates. Probabilities do not necessarily select the same class as the balanced decision rule.

The deployed input contract is one complete raw-count pseudobulk profile over 30,566 model genes, normalized with CPM and log1p. The application validates the saved artifact's exact feature names rather than filling missing genes. The architecture uses feature selection inside fitted pipelines; model weights are not validated causal biomarkers.

## Data and identity limits

The repository summary reports 16 source studies, 372 sample/library profiles and 1,470,730 aggregated cells. Profile counts are 128 Donor, 46 COPD, 179 PF and 19 COVID-19. These are not verified independent patient counts. Study/disease confounding and batch effects limit interpretation.

The released atlas has no verified biological-sex annotations or independent patient IDs. The default model is pooled. Optional sex-specific research candidates require additional annotated data, evaluation and deliberate local deployment; no such candidates are supplied by this release.

## Historical evaluation of the dual-rule procedure

Source: `reports/audit_lung_part_a.json`, `configuration_comparison.model_c_dual_rule`.

| Metric | Reported value | Study-bootstrap 95% interval |
| --- | --- | --- |
| Overall accuracy | 64.78% | 42.86%–81.78% |
| Balanced accuracy | 57.92% | 41.29%–75.93% |
| Macro-F1 | 0.5349 | 0.3766–0.7184 |
| COVID-19 recall | 21.05% (4/19) | Not supplied in this report entry |

These are historical leave-one-study-out procedure results, not a new evaluation of this software release or clinical validation. The audit's deployment flag describes the state when the audit was written; the current application selects the dual-rule artifact by its configured path. Matching per-class metrics and a confusion matrix are not substituted from an older model-selection procedure.

## Interpretation and scope

Results are exploratory expression-class assignments, not diagnoses. Poor COVID-19 recall prevents reliable acute COVID screening. Post-mortem single-nucleus tissue differs materially from the training input contract. Do not infer disease severity, treatment response, sex, or individual diagnostic certainty from these outputs.

The application does not claim regulatory clearance or clinical suitability. See `docs/UPGRADE_REVIEW.md` for operational limits, `docs/PRESENTATION_UPDATE.md` for evidence handling, and `LICENSE`/`docs/WEIGHTS.md` for distribution information.
