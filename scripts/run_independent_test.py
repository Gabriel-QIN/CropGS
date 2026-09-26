#!/usr/bin/env python3
"""Train on frozen development rows and evaluate eligible easy/hard tests once."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_deep_baseline import (
    TrainConfig,
    build_row_features,
    inner_split_positions,
    make_model,
    predict,
    set_seed,
    target_scale,
    train_fixed_epochs,
    train_with_inner_early_stopping,
    write_csv,
)
from src.data.io import load_genotype_dosage, read_phenotypes
from src.evaluation.metrics import metric_dict, summarize_by_environment
from src.preprocessing.deep_features import FoldFeatureTransformer, read_environment_features


CAPACITY = {
    "small": dict(d_model=96, n_heads=4, n_layers=2, head_hidden=128, max_markers=2048),
    "base": dict(d_model=192, n_heads=6, n_layers=3, head_hidden=256, max_markers=4096),
    "large": dict(d_model=384, n_heads=8, n_layers=6, head_hidden=512, max_markers=8192),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", required=True, choices=["Maize", "Rice", "Wheat", "Soybean"])
    parser.add_argument("--architecture", choices=["gated", "reaction_norm"], required=True)
    parser.add_argument("--environment-feature-set", choices=["basic", "dynamic"], required=True)
    parser.add_argument("--capacity", choices=list(CAPACITY), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim/genotype_cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--minimum-test-rows", type=int, default=100)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()

    import torch

    device = torch.device(args.device)
    config = TrainConfig(**CAPACITY[args.capacity])
    mapping = args.data_root.parent / "data" / "interim" / "environment_id_map.csv"
    rows, traits, _ = read_phenotypes(args.data_root, args.crop, mapping)
    manifest_path = args.results_root / "folds" / args.crop.lower() / "independent_test.csv"
    role_by_row: dict[int, str] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            role_by_row[int(record["row_id"])] = record["role"]
    environment_features, environment_names, _ = read_environment_features(
        args.data_root, args.crop, mapping, feature_set=args.environment_feature_set
    )
    genotype_ids, dosage, _ = load_genotype_dosage(args.data_root, args.crop, cache_dir=args.cache_dir, use_cache=True)
    genotype_index = {value: index for index, value in enumerate(genotype_ids)}
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for trait in traits:
        finite = [index for index, row in enumerate(rows) if np.isfinite(row[trait])]
        development_idx = np.asarray([index for index in finite if role_by_row[int(rows[index]["row_id"])] == "development"], dtype=int)
        evaluation_idx = np.asarray([index for index in finite if role_by_row[int(rows[index]["row_id"])].startswith("test_")], dtype=int)
        train_genotypes = np.asarray(
            sorted({genotype_index[rows[index]["genotype_id"]] for index in development_idx if rows[index]["genotype_id"] in genotype_index}), dtype=int
        )
        transformer = FoldFeatureTransformer(max_markers=config.max_markers, environment_components=config.environment_components).fit(
            dosage,
            train_genotypes,
            environment_features,
            [rows[index]["environment_id"] for index in development_idx],
            environment_names,
        )
        all_idx = np.concatenate((development_idx, evaluation_idx))
        x, missing, environment, _, _ = build_row_features(rows, all_idx, genotype_index, dosage, transformer, environment_features)
        y = np.asarray([float(rows[index][trait]) for index in all_idx], dtype=np.float32)
        n_development = len(development_idx)
        inner_train, inner_stop = inner_split_positions(rows, development_idx, "cv_ge", args.seed)
        position = {int(index): pos for pos, index in enumerate(development_idx)}
        inner_train_pos = np.asarray([position[int(index)] for index in inner_train], dtype=int)
        inner_stop_pos = np.asarray([position[int(index)] for index in inner_stop], dtype=int)
        set_seed(args.seed)
        selector = make_model(transformer.n_markers, transformer.n_environment_features, config, args.architecture)
        best_epoch, _, _ = train_with_inner_early_stopping(
            selector, x[:n_development], missing[:n_development], environment[:n_development], y[:n_development],
            inner_train_pos, inner_stop_pos, config, device, args.seed,
        )
        set_seed(args.seed + 1)
        model = make_model(transformer.n_markers, transformer.n_environment_features, config, args.architecture)
        train_fixed_epochs(
            model, x[:n_development], missing[:n_development], environment[:n_development], y[:n_development],
            np.arange(n_development), best_epoch, config, device, args.seed,
        )
        center, scale = target_scale(y[:n_development])
        test_prediction = predict(
            model, x[n_development:], missing[n_development:], environment[n_development:],
            np.arange(len(evaluation_idx)), center, scale, config, device,
        )
        for index, value in zip(evaluation_idx, test_prediction):
            prediction_rows.append(
                {
                    "crop": args.crop,
                    "trait": trait,
                    "role": role_by_row[int(rows[index]["row_id"])],
                    "row_id": rows[index]["row_id"],
                    "genotype_id": rows[index]["genotype_id"],
                    "environment_id": rows[index]["environment_id"],
                    "true": float(rows[index][trait]),
                    "prediction": float(value),
                    "architecture": args.architecture,
                    "feature_set": args.environment_feature_set,
                    "capacity": args.capacity,
                    "parameter_count": model.parameter_count(),
                    "best_epoch": best_epoch,
                }
            )
        for role in ["test_easy_g", "test_hard_g", "test_hard_e", "test_hard_ge"]:
            selected = [record for record in prediction_rows if record["trait"] == trait and record["role"] == role]
            if len(selected) < args.minimum_test_rows:
                metric_rows.append(
                    {
                        "crop": args.crop, "trait": trait, "role": role, "status": "skipped_insufficient_n", "n": len(selected),
                        "architecture": args.architecture, "feature_set": args.environment_feature_set, "capacity": args.capacity,
                    }
                )
                continue
            y_true = np.asarray([record["true"] for record in selected], dtype=float)
            y_pred = np.asarray([record["prediction"] for record in selected], dtype=float)
            environments = np.asarray([record["environment_id"] for record in selected], dtype=object)
            metrics = metric_dict(y_true, y_pred)
            metrics.update(summarize_by_environment(environments, y_true, y_pred))
            metrics.update(
                {
                    "crop": args.crop, "trait": trait, "role": role, "status": "evaluated",
                    "architecture": args.architecture, "feature_set": args.environment_feature_set, "capacity": args.capacity,
                    "parameter_count": model.parameter_count(), "best_epoch": best_epoch,
                }
            )
            metric_rows.append(metrics)

    metric_fields = [
        "crop", "trait", "role", "status", "architecture", "feature_set", "capacity", "parameter_count", "best_epoch",
        "n", "pearson", "spearman", "competition_score", "mse", "rmse", "mae", "n_environments", "environment_macro_score", "environment_weighted_score",
    ]
    prediction_fields = [
        "crop", "trait", "role", "row_id", "genotype_id", "environment_id", "true", "prediction", "architecture", "feature_set", "capacity", "parameter_count", "best_epoch",
    ]
    write_csv(args.results_root / "metrics" / f"{args.run_tag}_metrics.csv", metric_rows, metric_fields)
    write_csv(args.results_root / "predictions" / f"{args.run_tag}_predictions.csv", prediction_rows, prediction_fields)
    print(json.dumps(metric_rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
