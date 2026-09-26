# AI Breeding Challenge — Final Models and Inference Guide

## 1. Delivery status

Five final target artifacts have been trained on the complete **development**
partition certified by the frozen registries. The Diagnostic Holdout was not
read, scored, or added to training. These artifacts are the models to use when
the official blind prediction files are released.

Deliverables:

- final artifacts: `artifacts/final/*.pkl`;
- integrity and training metadata: `artifacts/final/manifest.json`;
- batch CLI: `scripts/predict_final.py`;
- Python API: `src/inference/final_predictor.py`;
- HTTP JSON API: `scripts/serve_inference_api.py`;
- reproducible final training: `scripts/train_final_models.py`;
- example request: `examples/final_inference_requests.csv`;
- strict benchmark table: `results/model_leaderboard.csv`.

## 2. Final model portfolio

| Crop / target | Final model | Development observations | Genotypes | Environments | Selected SNPs | Artifact |
|---|---|---:|---:|---:|---:|---:|
| Maize Trait_1 | Weather-OOD-gated Reaction-Norm GBLUP | 7,576 | 1,503 | 11 | 39,221 | 1.7 MB |
| Rice Trait_1 | Equal two-scale environment RKHS | 4,444 | 800 | 6 | 29,706 | 2.1 MB |
| Wheat Trait_1 | Full-marker RKHS | 3,534 | 1,563 | 10 | 9,517 | 0.6 MB |
| Soybean Trait_1 | Full-marker RKHS | 8,147 | 7,868 | 25 | 37,979 | 3.8 MB |
| Soybean Trait_2 | Full-marker RKHS | 8,166 | 7,872 | 25 | 37,981 | 3.8 MB |

All genomic transforms use fold-local/full-development MAF filtering,
missingness filtering, allele-frequency mean imputation, VanRaden-I centering,
and every SNP passing QC. Wheat retains the validated 40% marker-missingness
ceiling; low-rank imputation was rejected because its small average-CV gain did
not survive environment OOD.

### Model details

**Maize.** The in-support model is linear Reaction-Norm GBLUP with additive G,
weather E, and G×E components. Weather is summarized into dynamic statistics,
standardized, and projected to five training-fitted PCs. For an unseen
environment, its nearest training-weather distance is compared with the q90
training support radius. Beyond that radius, a phenotype-blind gate of strength
4 switches toward a training-only location/scale-calibrated additive GBLUP.

**Rice.** Two RKHS members share full-marker G and use weather-RBF environment
kernels at bandwidth multipliers 0.5 and 1.0. Predictions are averaged 50/50;
no target-fitted ensemble weight is used.

**Wheat and Soybean.** A full-marker RKHS combines G, weather-RBF E, and G×E.
The RBF bandwidth multiplier and residual ratio are 1.0 and 0.2.

The serialized predictor stores marker effects, environment effects, and G×E
marker effects—not the training genotype matrix. Inference is CPU-only and does
not require PyTorch or a GPU.

## 3. Strict development performance

CompetitionScore is `(Pearson + Spearman) / 2`. RobustScore is the unweighted
mean of CV-G, phenotype-blind CV-GCluster, CV-E, and crossover-purged CV-GE.

| Crop / target | CV-G | CV-GCluster | CV-E | CV-GE | Robust |
|---|---:|---:|---:|---:|---:|
| Maize Trait_1 | 0.643 | 0.614 | 0.372 | 0.329 | **0.490** |
| Rice Trait_1 | 0.729 | 0.612 | 0.371 | 0.360 | **0.518** |
| Wheat Trait_1 | 0.802 | 0.779 | 0.382 | 0.207 | **0.543** |
| Soybean Trait_1 | 0.887 | 0.827 | 0.729 | 0.679 | **0.780** |
| Soybean Trait_2 | 0.926 | 0.842 | 0.869 | 0.804 | **0.860** |

| Crop / target | OOD-G | OOD-E | OOD-GE |
|---|---:|---:|---:|
| Maize Trait_1 | 0.561 | 0.266 | 0.278 |
| Rice Trait_1 | 0.741 | 0.321 | 0.114 |
| Wheat Trait_1 | 0.793 | 0.379 | 0.211 |
| Soybean Trait_1 | 0.918 | 0.730 | 0.837 |
| Soybean Trait_2 | 0.935 | 0.861 | 0.869 |

These are out-of-fold development estimates, not scores computed on the final
full-development artifacts. No claim is made about the still-hidden official
blind labels. Validation subsets with fewer than 100 observations are absent,
not reported as scores.

## 4. Inference input contract

The inference dataset root must follow the released layout:

```text
NEW_DATASET/
  Maize/Maize/Genotypes.csv
  Maize/Maize/Environment.csv
  Rice/Rice/Genotypes.csv
  Rice/Rice/Environment.csv
  Wheat/Wheat/Genotypes.csv
  Wheat/Wheat/Environment.csv
  Soybean/Soybean/Genotypes.csv
  Soybean/Soybean/Environment.csv
```

The request CSV has at least:

```csv
genotype_id,environment_id
Sample1953,Loc3_2015
Sample1953,Loc2_2016
```

Extra columns such as `request_id` are preserved in the output.

Requirements and safeguards:

1. Marker IDs are aligned by name; column order may differ.
2. Missing selected markers and missing calls are allele-mean imputed.
3. Dosage must be 0/1/2 with the same allele orientation as training, and -1
   or the released missing tokens for missing calls. For nucleotide-coded data,
   use the official combined release whenever possible so allele coding stays
   consistent.
4. Every requested genotype must exist in that crop's `Genotypes.csv`.
5. Every requested environment must have weather rows in `Environment.csv`.
6. The environment variables must match the training schema. New environment
   IDs are supported; unseen weather is exactly what CV-E/OOD-E tested.

## 5. Batch CLI

Artifacts are already trained. Wheat example:

```bash
python scripts/predict_final.py \
  --artifact artifacts/final/wheat__trait_1.pkl \
  --data-root NEW_DATASET \
  --input prediction_requests.csv \
  --output wheat_predictions.csv
```

Artifact names:

```text
artifacts/final/maize__trait_1.pkl
artifacts/final/rice__trait_1.pkl
artifacts/final/wheat__trait_1.pkl
artifacts/final/soybean__trait_1.pkl
artifacts/final/soybean__trait_2.pkl
```

Output columns include `prediction`, `gate_weight`,
`nearest_environment_distance`, and `selected_marker_missing_fraction`.
The gate fields are zero for crops without the Maize OOD gate. A high selected
marker missing fraction is a warning to inspect genotype panel compatibility;
it is not silently discarded.

The organizer's final submission column names are not yet available. When the
official template arrives, retain row order and copy `prediction` into the
required target column without retraining the model.

## 6. Python API

```python
from pathlib import Path
from src.inference.final_predictor import load_bundle

model = load_bundle(Path("artifacts/final/maize__trait_1.pkl"))
records = [
    {"request_id": "M1", "genotype_id": "Sample_1/Sample_9", "environment_id": "WIH2_2020"}
]
predictions = model.predict_dataset(
    data_root=Path("NEW_DATASET"),
    records=records,
    mapping_path=Path("data/interim/environment_id_map.csv"),
    cache_dir=Path("data/interim/inference_genotype_cache"),
)
print(predictions[0]["prediction"])
```

For an upstream pipeline that already has dosage arrays and weather feature
vectors, call `model.predict_arrays(...)` directly. See the type signature in
`src/inference/final_predictor.py`.

## 7. HTTP JSON API

Start the service:

```bash
python scripts/serve_inference_api.py \
  --artifact-dir artifacts/final \
  --data-root NEW_DATASET \
  --host 127.0.0.1 \
  --port 8080
```

Health check:

```bash
curl --noproxy '*' http://127.0.0.1:8080/health
```

Prediction request:

```bash
curl --noproxy '*' -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "crop": "Wheat",
    "trait": "Trait_1",
    "records": [
      {"request_id": "W1", "genotype_id": "Sample1953", "environment_id": "Loc3_2015"}
    ]
  }'
```

The service uses only Python's standard HTTP server plus the numerical model
dependencies. It is intended for trusted internal/batch deployment. Put an
authenticated production gateway in front of it before exposing it to a
network.

## 8. Rebuilding final artifacts

Normally, use the supplied artifacts. Rebuild only after the frozen registry,
training release, or selected configuration intentionally changes:

```bash
python scripts/train_final_models.py \
  --data-root Dataset \
  --splits splits \
  --output artifacts/final
```

The trainer refuses a registry that does not certify Diagnostic Holdout
exclusion. `manifest.json` records registry hashes, marker hashes, artifact
SHA-256 checksums, sample counts, and artifact sizes.

The artifacts are Python pickle files and must be treated as trusted build
outputs. Never load a pickle received from an untrusted party.

## 9. Verification performed

- 14 automated preprocessing, leakage, kernel, calibration, and portable-export
  tests pass.
- Portable exported kernel predictions match the original fitted matrix-free
  model within `2e-5` in the equivalence test.
- All five artifacts load and produce finite predictions on real released
  genotype × environment requests.
- Batch CLI produced `results/predictions/final_inference_smoke.csv`.
- HTTP `/health` and `/predict` returned HTTP 200 with all five models loaded.
- Full artifact checksums are in `artifacts/final/manifest.json`.

## 10. Blind-test operating procedure

1. Keep the final artifacts frozen.
2. Verify new marker IDs, dosage orientation, genotype IDs, environment weather,
   and missing fractions before scoring.
3. Run the batch CLI separately for each crop/trait.
4. Confirm output row count/order against the official template.
5. Inspect Maize `gate_weight` and all crops' marker missing fraction; do not
   tune against any hidden or post-submission feedback labels.
6. Archive the input hashes, artifact manifest, predictions, and exact command
   line with the competition submission.

The principal unresolved risk remains environment extrapolation for Maize,
Rice, and Wheat—especially Rice OOD-GE. The final portfolio deliberately favors
strict-CV/OOD stability over larger neural models or small in-distribution
gains that failed stress testing.
