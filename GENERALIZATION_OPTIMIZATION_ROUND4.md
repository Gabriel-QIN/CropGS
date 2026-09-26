# Generalization Optimization — Round 4

## Outcome

No configuration is promoted in this round. The selected Maize weather-OOD
gate is stable to reasonable weather-geometry changes, and Wheat's existing
40% marker-missingness ceiling remains the best blind-test compromise.
Diagnostic Holdout remained locked; validation subsets below 100 observations
were excluded.

## Maize gate robustness

The gate strength was fixed at 4. Candidate fallback predictors and gate
geometries were varied without using validation phenotypes to construct the
gate.

| Variant | Robust | OOD-E | OOD-GE |
|---|---:|---:|---:|
| **Selected: PCA-5/q90, calibrated additive GBLUP** | **0.4897** | **0.2661** | **0.2784** |
| PCA-3/q90 | 0.4901 | 0.2661 | 0.2784 |
| PCA-8/q90 | 0.4897 | 0.2661 | 0.2784 |
| PCA-5/q75 | 0.4897 | 0.2661 | 0.2784 |
| PCA-5/q95 | 0.4897 | 0.2661 | 0.2784 |
| Distributional RKHS fallback | 0.4897 | 0.2492 | 0.2677 |
| GBLUP+environment fallback | 0.4897 | 0.2486 | 0.2719 |
| Distributional Reaction-Norm fallback | 0.4897 | 0.2047 | 0.2312 |

PCA-3 changes RobustScore by only +0.0004 and leaves OOD scores identical.
This is below the promotion threshold and introduces an extra active core-CV
gate, so PCA-5/q90 remains selected. The OOD gain is therefore not dependent
on a narrow PCA dimension or support quantile.

## Wheat missing-genotype representation

Two additional families were evaluated: shrinkage of rank-64 reconstruction
toward allele-mean imputation, and fold-local marker missingness ceilings.

### Low-rank shrinkage

| Rank-64 reconstruction weight | Robust | Worst | OOD-E | OOD-GE |
|---:|---:|---:|---:|---:|
| **0.00 / allele mean** | **0.5425** | **0.0119** | **0.3792** | **0.2110** |
| 0.25 | 0.5428 | 0.0048 | 0.3699 | 0.1915 |
| 0.50 | 0.5431 | -0.0013 | 0.3612 | 0.1731 |
| 0.75 | 0.5433 | -0.0054 | 0.3532 | 0.1610 |
| 1.00 | 0.5436 | -0.0087 | 0.3461 | 0.1506 |

The monotonic tradeoff shows that low-rank reconstruction buys a tiny average
CV gain by steadily sacrificing tail and environment-OOD performance. It is
rejected for blind testing.

### Marker missingness ceiling

| Maximum marker missingness | Robust | Worst | OOD-E | OOD-GE |
|---:|---:|---:|---:|---:|
| 0.10 | 0.5366 | 0.0225 | 0.3405 | 0.1558 |
| 0.20 | 0.5440 | -0.0550 | 0.3590 | 0.1917 |
| 0.30 | 0.5445 | 0.0043 | 0.3686 | 0.2041 |
| 0.35 | 0.5432 | -0.0118 | 0.3738 | 0.2060 |
| **0.40 selected** | **0.5425** | **0.0119** | **0.3792** | **0.2110** |
| 0.50 | 0.5418 | 0.0080 | 0.3792 | 0.2150 |

The 0.30 ceiling has the highest RobustScore, but its +0.0020 average gain is
paired with lower WorstScore and lower OOD-E/OOD-GE. The 0.50 ceiling improves
only OOD-GE by 0.0041 while reducing core robustness. The existing 0.40 setting
remains the safest balanced choice.

## Verification

- Fixed-weight low-rank shrinkage is outer-fold-local and never overwrites an
  observed genotype dosage.
- Maize gate construction remains phenotype-blind.
- The unified leaderboard includes all Round-4 candidates for auditability.
