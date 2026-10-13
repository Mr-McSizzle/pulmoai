# Data inventory

## Source dataset

- Full source file: `/Users/nisarg/Downloads/atlas-001.h5ad` (6,310,632,229 bytes).
- The source is kept in Downloads and is not copied or modified by the project.
- `data/atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad` is a 15,908,864-byte partial/incomplete file with a similar name. It is not the source used for the current model and must not be substituted for the full atlas.

## Derived data

- `data/processed/allcells_pseudobulk_counts.npz` is generated from the full H5AD. It contains the 372 × 30,566 aggregated count profiles and profile/study/feature metadata. It is ignored by Git and can be regenerated from the source.
- Cells are grouped by `gse_alias + orig.ident`. These are sample/library groups, not verified unique patients; the source does not provide a verified patient ID in the metadata currently used.
- Labels are Donor, COPD, PF, and COVID19. Sample-group class counts are documented in `reports/ensemble_benchmark/analysis_summary.json`.
- CP1 validation found 1,470,730 nonblank, unique cell IDs within studies; no cell identifier was reused across studies. All four required observation fields had zero missing codes, all 372 study-scoped groups had a single consistent disease label, and all 30,566 feature identifiers were unique, nonblank, and ordered exactly as in the model bundle. Detailed results are in `reports/atlas_metadata_validation.json`.
- The source `.obs` fields are `Disease`, `gse_alias`, `orig.ident`, and `celltypeuse`; there is no explicit verified patient/subject ID field. Therefore the profile count is not a patient count.
- The saved classifier expects one profile of raw pseudobulk counts with the exact 30,566 feature identifiers. `src/pulmoai/preprocessing.py` accepts either a one-row wide CSV (genes as columns, optional `sample_id`) or a long CSV with exactly `gene_id,count` (or `gene,count`) columns. Genes may be shuffled; the parser aligns them to model order and rejects any missing/extra genes, duplicate identifiers, multiple profiles, negative/non-finite/fractional values, and zero-total profiles.
- Preprocessing matches training: raw counts are normalized by the profile total to CPM, then transformed with `log1p`. The fitted model handles top-500 feature selection, scaling, and Linear SVM classification. No missing gene is silently filled with zero. Normalized/log expression, images, routine blood reports, and multi-profile uploads are not supported.

Do not extract or copy another multi-gigabyte atlas file unless the workflow explicitly requires it. Inspect new input metadata and feature identifiers before using it for inference.
