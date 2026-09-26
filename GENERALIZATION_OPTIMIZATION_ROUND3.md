# Generalization Optimization — Round 3

## Decision

Promote the Maize phenotype-blind weather OOD gate at strength 4. Retain the
original mean-imputed Wheat RKHS; reject all tested low-rank imputation ranks.
The locked Diagnostic Holdout was not read or used. Validation sets with fewer
than 100 observations remain excluded.

## Maize: environment OOD gate

The in-support predictor remains Reaction-Norm GBLUP. For an unseen environment,
weather features are transformed by fold-local PCA and compared with the closest
training environment. The gate activates only beyond the q90 training-support
radius and blends toward a training-only calibrated additive GBLUP. It uses no
validation phenotype or environment label semantics.

| Variant | CV-G | CV-GCluster | CV-E | CV-GE | Robust | OOD-E | OOD-GE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ungated Reaction-Norm | 0.643 | 0.614 | 0.372 | 0.329 | 0.490 | 0.160 | 0.215 |
| Gate strength 1 | 0.643 | 0.614 | 0.372 | 0.329 | 0.490 | 0.198 | 0.240 |
| Gate strength 2 | 0.643 | 0.614 | 0.372 | 0.329 | 0.490 | 0.248 | 0.268 |
| **Gate strength 4** | **0.643** | **0.614** | **0.372** | **0.329** | **0.490** | **0.266** | **0.278** |
| Gate strength 8 | 0.643 | 0.614 | 0.372 | 0.329 | 0.490 | 0.266 | 0.278 |

Strengths 4 and 8 are identical because active OOD environments already reach
the clipping boundary. Strength 4 is selected as the smallest saturated setting.
The gate leaves CV-G, CV-GCluster, CV-E, CV-GE, and OOD-G predictions unchanged.

## Wheat: low-rank missing-genotype imputation

Truncated SVD is fitted on outer-training genotypes only. It reconstructs only
missing entries; observed dosages are never overwritten.

| Imputation | CV-G | CV-GCluster | CV-E | CV-GE | Robust | Worst | OOD-E | OOD-GE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Allele mean** | 0.802 | 0.779 | 0.382 | 0.207 | 0.5425 | **0.0119** | **0.379** | **0.211** |
| Low rank 8 | 0.803 | 0.779 | 0.383 | 0.204 | 0.5424 | -0.0008 | 0.346 | 0.155 |
| Low rank 16 | 0.804 | 0.779 | 0.384 | 0.197 | 0.5411 | 0.0033 | 0.344 | 0.151 |
| Low rank 32 | 0.805 | 0.780 | 0.388 | 0.197 | 0.5426 | 0.0092 | 0.349 | 0.144 |
| Low rank 64 | 0.805 | 0.780 | 0.386 | 0.202 | 0.5436 | -0.0087 | 0.346 | 0.151 |

Rank 64's Robust gain is only 0.0011 and is accompanied by a negative worst
fold plus material OOD-E/OOD-GE losses. No low-rank variant is promoted. This
protects the blind-test objective from selecting a small in-distribution gain
that does not survive environmental shift.

## Final configuration change

- Maize Trait_1: weather-OOD-gated Reaction-Norm GBLUP, gate strength 4.
- Wheat Trait_1: unchanged RKHS with fold-local allele-mean imputation.

Full fold-level outputs and predictions are under
`results/optimization/maize_weather_ood_gate_s4/` and
`results/optimization/wheat_lowrank{8,16,32,64}/`.
