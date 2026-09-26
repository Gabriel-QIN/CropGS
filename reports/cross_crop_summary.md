# Cross-crop Data Audit Summary

The audit is intentionally conservative: CV design is based on observed joins and coverage, not the README alone.

| Crop | Genotype samples | Markers | Phenotype rows | Environments | Traits | Genotype missing | Direct env join |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Maize | 1879 | 50000 | 12854 | 14 | 1 | 0.636% | 100.0% |
| Rice | 1000 | 60000 | 5881 | 7 | 1 | 0.000% | 100.0% |
| Wheat | 1953 | 10297 | 5108 | 12 | 1 | 20.229% | 0.0% |
| Soybean | 11604 | 41756 | 12007 | 31 | 2 | 0.738% | 100.0% |

## Findings

- **Maize:** 4080 monomorphic markers detected.
- **Rice:** has single-environment materials; report coverage-stratified CV-G performance; 12187 monomorphic markers detected.
- **Wheat:** phenotype and weather environment IDs use different year-location order; mapping is required; has single-environment materials; report coverage-stratified CV-G performance; 5 monomorphic markers detected.
- **Soybean:** largest genotype matrix and two traits; environment weather missingness must be imputed within a declared weather preprocessing step; has single-environment materials; report coverage-stratified CV-G performance; 429 monomorphic markers detected.

## Recommended audit order

1. Resolve and version any environment ID mapping (especially Wheat).
2. Freeze genotype/phenotype linkage and replicate policy.
3. Build CV-G and coverage-stratified diagnostics before CV-GCluster/CV-E/CV-GE.
4. Keep the generated PCA/clustering output as an audit diagnostic; do not use cluster labels as a model feature without fold-safe construction.

Cross-crop figure: `reports/figures/cross_crop_dataset_overview.svg`
