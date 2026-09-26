# CropGS

Robust genomic selection and trait prediction for major crops using genomic,
environmental, and weather data.

CropGS provides leakage-safe genomic prediction for maize, rice, wheat, and
soybean. It combines full-marker genomic kernels, weather-aware genotype ×
environment modeling, adversarial OOD validation, compact trained artifacts,
and batch/Python/HTTP inference interfaces.

## Final models

| Crop / target | Selected model | CV-G | CV-E | CV-GE | Robust |
|---|---|---:|---:|---:|---:|
| Maize Trait_1 | Weather-OOD-gated Reaction-Norm GBLUP | 0.643 | 0.372 | 0.329 | **0.490** |
| Rice Trait_1 | Equal two-scale RKHS | 0.729 | 0.371 | 0.360 | **0.518** |
| Wheat Trait_1 | Full-marker RKHS | 0.802 | 0.382 | 0.207 | **0.543** |
| Soybean Trait_1 | Full-marker RKHS | 0.887 | 0.729 | 0.679 | **0.780** |
| Soybean Trait_2 | Full-marker RKHS | 0.926 | 0.869 | 0.804 | **0.860** |

Scores use `(Pearson + Spearman) / 2`. `Robust` is the mean of CV-G,
phenotype-blind CV-GCluster, CV-E, and crossover-purged CV-GE. Validation sets
with fewer than 100 observations are excluded. The Diagnostic Holdout was not
used for model selection or final training.

## Installation

```bash
git clone https://github.com/Gabriel-QIN/CropGS.git
cd CropGS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The released crop data are not redistributed. Place them under a dataset root:

```text
Dataset/
  Maize/Maize/{Genotypes,Environment,Phenotypes}.csv
  Rice/Rice/{Genotypes,Environment,Phenotypes}.csv
  Wheat/Wheat/{Genotypes,Environment,Phenotypes}.csv
  Soybean/Soybean/{Genotypes,Environment,Phenotypes}.csv
```

## Batch inference

The five compact final artifacts are included in `artifacts/final/`.

Create a request file:

```csv
request_id,genotype_id,environment_id
example_1,Sample1953,Loc3_2015
```

Run prediction:

```bash
python scripts/predict_final.py \
  --artifact artifacts/final/wheat__trait_1.pkl \
  --data-root Dataset \
  --input examples/final_inference_requests.csv \
  --output wheat_predictions.csv
```

Available artifacts:

```text
maize__trait_1.pkl
rice__trait_1.pkl
wheat__trait_1.pkl
soybean__trait_1.pkl
soybean__trait_2.pkl
```

## Python API

```python
from pathlib import Path
from src.inference.final_predictor import load_bundle

model = load_bundle(Path("artifacts/final/maize__trait_1.pkl"))
predictions = model.predict_dataset(
    data_root=Path("Dataset"),
    records=[{
        "request_id": "M1",
        "genotype_id": "Sample_1/Sample_9",
        "environment_id": "WIH2_2020",
    }],
    mapping_path=Path("data/interim/environment_id_map.csv"),
)
```

## HTTP API

```bash
python scripts/serve_inference_api.py \
  --artifact-dir artifacts/final \
  --data-root Dataset \
  --port 8080
```

```bash
curl --noproxy '*' -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "crop": "Wheat",
    "trait": "Trait_1",
    "records": [{
      "request_id": "W1",
      "genotype_id": "Sample1953",
      "environment_id": "Loc3_2015"
    }]
  }'
```

## Rebuild final models

```bash
python scripts/train_final_models.py \
  --data-root Dataset \
  --splits splits \
  --output artifacts/final
```

The trainer accepts only frozen registries that certify Diagnostic Holdout
exclusion. Artifact and marker hashes are recorded in
`artifacts/final/manifest.json`.

## Repository layout

```text
artifacts/final/       compact trained final models
configs/               selected model configurations
data/interim/          environment-ID mapping (generated matrices ignored)
examples/              inference request examples
reports/               data-audit summaries and figures
scripts/               audit, benchmark, training, and inference CLIs
splits/                frozen leakage-safe split registries
src/                    data, preprocessing, model, validation, and API code
tests/                  component, leakage, and portable-export tests
```

See [FINAL_MODEL_INFERENCE_REPORT.md](FINAL_MODEL_INFERENCE_REPORT.md) for the
complete technical description, OOD results, input contract, API reference,
and blind-test operating procedure.

## Verification

```bash
pip install -r requirements-dev.txt
pytest -q
```

The final delivery passes 14 automated tests. Portable model predictions match
the original matrix-free model within the tested numerical tolerance.

## License

CropGS is released under the GNU General Public License v3.0.
