#!/usr/bin/env python3
"""Phenotype-blind weather-distance gate for conservative Maize prediction."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.data.io import read_phenotypes
from src.evaluation.metrics import metric_dict
from src.preprocessing.deep_features import read_environment_features
from src.preprocessing.genomic import FoldEnvironmentTransformer
from src.validation.registry import read_registry, registry_splits


def read_model(path: Path, model: str) -> dict[tuple[str, str, str], dict]:
    result = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["crop"] == "Maize" and row["model"] == model:
                result[(row["protocol"], row["fold"], row["row_id"])] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("results/phase1_strict/maize/oof_predictions.csv"))
    parser.add_argument("--calibrated", type=Path, default=Path("results/optimization/distributional_maize/oof_predictions.csv"))
    parser.add_argument("--fallback-model", default="distributional_gblup")
    parser.add_argument("--output", type=Path, default=Path("results/optimization/maize_weather_ood_gate"))
    parser.add_argument("--gate-strength", type=float, default=1.0, help="Multiplier on the phenotype-blind distance exceedance weight")
    parser.add_argument("--environment-components", type=int, default=5)
    parser.add_argument("--support-quantile", type=float, default=0.90)
    parser.add_argument("--run-label", default="maize_weather_ood_gate")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    base = read_model(args.base, "reaction_norm_gblup")
    calibrated = read_model(args.calibrated, args.fallback_model)
    registry = read_registry(Path("splits/maize.json"))
    all_rows, _, _ = read_phenotypes(Path("Dataset"), "Maize", Path("data/interim/environment_id_map.csv"))
    allowed = {int(item["row_id"]) for item in registry["observations"]}
    rows = [row for row in all_rows if row["row_id"] in allowed]
    weather, _, _ = read_environment_features(Path("Dataset"), "Maize", Path("data/interim/environment_id_map.csv"), "dynamic")
    output, diagnostics, fold_metrics = [], [], []
    protocols = ["cv_g", "cv_gcluster", "cv_e", "cv_ge"] + [f"ood_{kind}_{pct}" for kind in ("g", "e", "ge") for pct in (10, 15, 20)]
    for protocol in protocols:
        for split in registry_splits(rows, registry, protocol, 100):
            train_env = sorted({rows[i]["environment_id"] for i in split.train_idx})
            valid_env = sorted({rows[i]["environment_id"] for i in split.validation_idx})
            transformer = FoldEnvironmentTransformer(args.environment_components).fit(weather, train_env)
            train_pc = transformer.transform(weather, train_env)
            valid_pc = transformer.transform(weather, valid_env)
            if len(train_pc) > 1:
                pair = np.sqrt(np.sum((train_pc[:, None, :] - train_pc[None, :, :]) ** 2, axis=2))
                pair[pair <= 1e-12] = np.inf
                reference = np.min(pair, axis=1)
                threshold = max(float(np.quantile(reference, args.support_quantile)), 1e-6)
            else:
                threshold = 1.0
            env_weight = {}
            for environment, pc in zip(valid_env, valid_pc):
                if environment in train_env:
                    distance, weight = 0.0, 0.0
                else:
                    distance = float(np.min(np.linalg.norm(train_pc - pc, axis=1)))
                    weight = float(np.clip(args.gate_strength * (distance - threshold) / threshold, 0.0, 1.0))
                env_weight[environment] = weight
                diagnostics.append({"protocol": protocol, "fold": split.fold, "environment_id": environment, "nearest_distance": distance, "training_q90_nearest_distance": threshold, "calibrated_weight": weight})
            records = []
            for index in split.validation_idx:
                row = rows[index]; key = (protocol, str(split.fold), str(row["row_id"]))
                if key not in base or key not in calibrated: continue
                weight = env_weight[row["environment_id"]]
                prediction = (1.0 - weight) * float(base[key]["prediction"]) + weight * float(calibrated[key]["prediction"])
                record = {**base[key], "model": args.run_label, "prediction": prediction}; output.append(record); records.append(record)
            if records:
                metrics = metric_dict(np.asarray([float(r["true"]) for r in records]), np.asarray([float(r["prediction"]) for r in records]))
                metrics.update({"crop": "Maize", "trait": "Trait_1", "protocol": protocol, "fold": split.fold, "model": args.run_label, "runtime_seconds": 0.0}); fold_metrics.append(metrics)
    fields = ["crop", "trait", "protocol", "fold", "model", "row_id", "genotype_id", "environment_id", "true", "prediction"]
    with (args.output / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{key: row[key] for key in fields} for row in output])
    with (args.output / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields_m = sorted({key for row in fold_metrics for key in row}); writer = csv.DictWriter(handle, fieldnames=fields_m); writer.writeheader(); writer.writerows(fold_metrics)
    with (args.output / "gate_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0])); writer.writeheader(); writer.writerows(diagnostics)
    print(f"Wrote {len(output)} gated predictions")
    return 0


if __name__ == "__main__": raise SystemExit(main())
