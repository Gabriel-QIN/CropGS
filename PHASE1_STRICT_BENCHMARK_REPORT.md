# Phase-1 Strict Genomic Benchmark

## Status

The Phase-1 development benchmark is complete for all four crops and all five
crop-trait targets. The locked Diagnostic Holdout was excluded before any split or model
fit.  Model selection uses only development CV.

Implemented and executed:

- frozen per-observation registries in `splits/`;
- CV-G, phenotype-blind CV-GCluster, leave-one-environment-out CV-E, and
  crossover-purged CV-GE;
- phenotype-blind adversarial OOD-G/OOD-E/OOD-GE at 10%, 15%, and 20%;
- automatic skip of every validation set with fewer than 100 observations;
- fold-local marker QC, allele-frequency imputation, VanRaden-I scaling,
  genotype PCA implementation, weather standardization/PCA, and kernel fit;
- exact linear full-marker GBLUP without a dense observation GRM;
- GBLUP + environment, Reaction-Norm GBLUP, and multi-kernel RKHS;
- pooled Pearson, Spearman, CompetitionScore, RMSE, MAE, environment-macro,
  median/worst environment, fold SD, runtime, and variance/kernel components.

`RR-BLUP` and linear `GBLUP` are the same ridge solution under the usual
primal/dual reparameterization.  The implementation uses the dual,
matrix-free form so every QC-passed marker is retained; it is not the old
512-marker approximation.

## Strict-CV leaderboard

Competition scores below are pooled OOF scores. `Robust` is the unweighted
mean of CV-G, CV-GCluster, CV-E, and CV-GE.

| Crop / trait | Best model | CV-G | CV-GCluster | CV-E | CV-GE | Robust |
|---|---|---:|---:|---:|---:|---:|
| Maize Trait_1 | Weather-OOD-gated Reaction-Norm | 0.643 | 0.614 | 0.372 | 0.329 | **0.490** |
| Rice Trait_1 | Equal multi-scale RKHS | 0.729 | 0.612 | 0.371 | 0.360 | **0.518** |
| Wheat Trait_1 | RKHS | 0.802 | 0.779 | 0.382 | 0.207 | **0.543** |
| Soybean Trait_1 | RKHS | 0.887 | 0.827 | 0.729 | 0.679 | **0.780** |
| Soybean Trait_2 | RKHS | 0.926 | 0.842 | 0.869 | 0.804 | **0.860** |

The complete 20-row model table is `results/model_leaderboard.csv`.

## Adversarial stress test

Scores are the mean across eligible 10%/15%/20% distance thresholds. Threshold
sets below 100 observations are absent rather than reported as results.

| Crop / trait | Selected model | OOD-G | OOD-E | OOD-GE |
|---|---|---:|---:|---:|
| Maize Trait_1 | Weather-OOD gate (strength 4) | 0.561 | 0.266 | 0.278 |
| Rice Trait_1 | Equal multi-scale RKHS | 0.741 | 0.321 | 0.114 |
| Wheat Trait_1 | RKHS | 0.793 | 0.379 | 0.211 |
| Soybean Trait_1 | RKHS | 0.918 | 0.730 | 0.837 |
| Soybean Trait_2 | RKHS | 0.935 | 0.861 | 0.869 |

## Interpretation

1. Environment extrapolation, not marker capacity, is the main bottleneck for
   Maize, Rice, and Wheat. Genotype OOD remains much stronger than E/GE OOD.
2. Linear Reaction-Norm remains the Maize in-support base model. A phenotype-blind
   weather-distance gate now switches extreme out-of-support environments toward
   calibrated additive GBLUP, improving OOD-E and OOD-GE without changing any
   core-CV prediction.
3. Rice has only six development environments after locking the Diagnostic
   Holdout. Linear environment effects become unstable; RBF-E RKHS is the only
   strong all-regime Phase-1 candidate, but OOD-GE remains weak.
4. Wheat's linear Reaction-Norm catastrophically fails adversarial environment
   extrapolation (OOD-E -0.354, OOD-GE -0.642). It must not be selected from
   CV-G alone. RKHS is materially safer.
5. Soybean is currently robust, especially Trait_2. Its mostly one-environment
   genotype structure still argues against a high-rank free interaction model.

## Next modeling gate

The next authorized search should remain development-only and focus on:

- environment mean/scale calibration for Maize, Rice, and Wheat;
- fold-local genomic PCA/RBF and environment bandwidth ablations;
- additive fallback plus distance-gated RKHS/Reaction-Norm ensemble;
- Wheat mean-imputation versus low-rank imputation;
- only after those, compact low-rank neural residual models.

The Diagnostic Holdout remains locked and must not be used for any of these
choices.

## Verification

- 911 strict fold-model fits in the final converged run;
- zero validation sets below 100 observations reported;
- zero non-finite scores/predictions;
- zero conjugate-gradient solves reaching the 400-iteration cap;
- maximum final CG residual: 0.00543;
- nine automated component/leakage tests passing;
- all registry content hashes verified after construction.
