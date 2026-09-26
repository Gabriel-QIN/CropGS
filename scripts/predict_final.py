#!/usr/bin/env python3
"""Batch inference with one exported final genomic-selection artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.final_predictor import load_bundle, read_request_csv, write_prediction_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root containing Crop/Crop/{Genotypes,Environment}.csv")
    parser.add_argument("--input", type=Path, required=True, help="CSV with genotype_id,environment_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=Path("data/interim/environment_id_map.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim/inference_genotype_cache"))
    args = parser.parse_args()
    bundle = load_bundle(args.artifact)
    records = read_request_csv(args.input)
    predictions = bundle.predict_dataset(args.data_root, records, args.mapping, args.cache_dir)
    write_prediction_csv(args.output, predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
