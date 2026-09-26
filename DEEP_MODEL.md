# Deep model: inputs, outputs, and leakage controls

## What is the current model?

The first reproducible baseline is `global_mean`, `environment_mean`,
`genotype_mean`, `additive_mean`, and `genomic_ridge` (a small marker-ridge /
GBLUP-like proxy). It is useful for checking the scoring pipeline, but it is
not a deep model.

`scripts/run_deep_baseline.py` adds `deep_gxe_net`, a regularised
DeepGxE-Net-L. It uses a marker-block convolution followed by a small
Transformer encoder, a weather MLP, and a gated genotype-by-environment
fusion head. Because the released files do not include chromosome and
position metadata, marker blocks follow the stable marker-column order and are
not described as chromosome-aware.

## Model input

One training example is one phenotype record: a `genotype_id × environment_id`
pair. The model receives three tensors:

| Tensor | Shape | Construction |
|---|---|---|
| genotype dosage | `[batch, P]` | Fold-selected dosage values, imputed and standardised using outer-training genotypes only. `P` defaults to at most 4,096 markers. |
| genotype missing mask | `[batch, P]` | `1` where the original dosage was missing, `0` otherwise; this prevents imputation from hiding missingness. |
| environment features | `[batch, F]` | `n_days` plus mean, SD, min, max, first, last, and missing fraction for each weather variable. Imputation/scaling uses outer-training environments only. |

The target is one `Trait_*` value. Traits are trained independently so a
missing `Trait_1` row cannot silently change the `Trait_2` target table.

## Model output

The head emits one scalar prediction for the requested trait for each input
record. OOF files retain the original row/genotype/environment IDs and include
`true`, `prediction`, fold, selected-marker count, parameter count, and seed.
The competition score is `(Pearson + Spearman) / 2`; MSE/RMSE/MAE and
environment-macro scores are also written.

## Why the larger model is not allowed to overfit silently

- Outer folds are the existing leakage checks: CV-G, CV-GCluster, CV-E, and
  strict CV-GE. CV0 is debug-only.
- Marker selection, missing-value imputation, dosage scaling, weather
  imputation, and weather scaling are fitted separately inside each outer
  training fold.
- Early stopping uses a deterministic inner training-only split. The outer
  validation rows are never used to choose an epoch, feature, or weight.
- The final fold model is retrained on all outer-training rows for the epoch
  selected by the inner split.
- AdamW weight decay, dropout, Huber loss, gradient clipping, fixed seeds, and
  a finite marker/hidden-size budget are enabled by default.
- `results/metrics/deep_distribution.csv` reports genotype missingness,
  environment feature shift, group overlap, selected markers, runtime, and
  parameter count for every fold. Non-zero genotype or environment overlap is
  a hard failure for the relevant protocol, not a score to average away.

## Run

The base Python 3.13 environment has a CUDA 13 wheel that is incompatible
with this host's driver. Use the existing CUDA 12.8 environment:

```bash
python scripts/run_deep_baseline.py \
  --protocol cv_g --protocol cv_gcluster --protocol cv_ge \
  --device cuda:0
```

For the expensive leave-one-environment-out test, add `--protocol cv_e` as a
separate run. Smoke tests may use `--fold-limit 1`, but those metrics must not
be used for model selection or competition reporting.
