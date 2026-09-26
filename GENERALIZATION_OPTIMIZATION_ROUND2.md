# Generalization Optimization — Round 2

## Outcome

Rice receives one validated upgrade: an equal-weight, two-scale environment
RKHS ensemble using RBF bandwidths `0.5 × median distance` and
`1.0 × median distance`. No target-dependent ensemble weights are fitted.

| Rice Trait_1 | CV-G | CV-GCluster | CV-E | CV-GE | Robust | OOD-E | OOD-GE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Previous RKHS | 0.727 | 0.617 | 0.333 | 0.276 | 0.488 | 0.299 | 0.082 |
| Equal multi-scale RKHS | 0.729 | 0.612 | 0.371 | 0.360 | **0.518** | 0.321 | 0.114 |

The gain is concentrated where Rice was weakest: unseen environments and
simultaneous genotype/environment OOD. CV-GCluster changes by -0.005, while
RobustScore improves by +0.030.

## Experiments not promoted

- Environment location/scale calibration improved some adversarial Maize
  folds but reduced ordinary CV-E/CV-GE RobustScore.
- Cross-fitted non-negative stacking did not beat the strongest single model.
- Selecting the best RBF bandwidth/residual ratio directly on pooled outer OOF
  looked strong, but outer-fold-excluded selection was unstable. Those direct
  grid maxima are not treated as validated gains.
- Three-scale Rice RKHS (`0.5/1/2`) underperformed the two-scale ensemble.
- Wheat and Maize multi-scale averages did not improve their existing robust
  champions, so their selected models remain unchanged.

## Selected development models

- Maize Trait_1: Reaction-Norm GBLUP, RobustScore 0.490.
- Rice Trait_1: equal two-scale RKHS, RobustScore 0.518.
- Wheat Trait_1: default RKHS, RobustScore 0.543.
- Soybean Trait_1: default RKHS, RobustScore 0.780.
- Soybean Trait_2: default RKHS, RobustScore 0.860.

Diagnostic Holdout remained locked throughout this round. All decisions above
use development folds only; validation subsets below 100 observations remain
excluded.
