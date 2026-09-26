#!/usr/bin/env python3
"""Run the mandatory strict full-marker Phase-1 genomic benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.io import load_genotype_dosage, read_phenotypes
from src.evaluation.metrics import metric_dict, summarize_by_environment
from src.models.kernel_gs import MatrixFreeReactionNorm, environment_kernel_factors
from src.models.environment_calibration import EnvironmentLocationScale
from src.preprocessing.deep_features import read_environment_features
from src.preprocessing.genomic import FoldEnvironmentTransformer, FullMarkerTransformer
from src.validation.leakage import assert_protocol_isolation, assert_registry_locked
from src.validation.registry import read_registry, registry_splits


def unique_mapping(values: list[str]) -> tuple[list[str], np.ndarray]:
    unique = sorted(set(values))
    lookup = {value: i for i, value in enumerate(unique)}
    return unique, np.asarray([lookup[value] for value in values], dtype=np.int64)


def checkpoint(results: Path, fold_rows: list[dict], prediction_rows: list[dict]) -> None:
    if fold_rows:
        fields = sorted({key for row in fold_rows for key in row})
        with (results / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(fold_rows)
    if prediction_rows:
        with (results / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
            writer.writeheader(); writer.writerows(prediction_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--results", type=Path, default=Path("results/phase1"))
    parser.add_argument("--crop", action="append", choices=["Maize", "Rice", "Wheat", "Soybean"])
    all_protocols = ["cv_g", "cv_gcluster", "cv_e", "cv_ge"] + [f"ood_{regime}_{pct}" for regime in ("g", "e", "ge") for pct in (10, 15, 20)]
    parser.add_argument("--protocol", action="append", choices=all_protocols)
    base_models = ["gblup", "gblup_environment", "reaction_norm_gblup", "rkhs"]
    distributional_models = ["distributional_gblup", "distributional_reaction_norm_gblup", "distributional_rkhs"]
    parser.add_argument("--model", action="append", choices=base_models + distributional_models)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-validation-rows", type=int, default=100)
    parser.add_argument("--max-cg-iterations", type=int, default=60)
    parser.add_argument("--residual-ratio", type=float, default=0.2)
    parser.add_argument("--rbf-bandwidth", type=float, default=1.0)
    parser.add_argument("--run-label", help="Override recorded model name for a single-model configuration run")
    parser.add_argument("--genotype-imputation", choices=["mean", "low_rank"], default="mean")
    parser.add_argument("--imputation-rank", type=int, default=32)
    parser.add_argument("--imputation-weight", type=float, default=1.0)
    parser.add_argument("--max-marker-missing", type=float, help="Override crop-default fold-local marker missingness ceiling")
    parser.add_argument("--limit-folds", type=int)
    args = parser.parse_args()
    crops = args.crop or ["Maize", "Rice", "Wheat", "Soybean"]
    protocols = args.protocol or ["cv_g", "cv_gcluster", "cv_e", "cv_ge"]
    models = args.model or ["gblup", "reaction_norm_gblup", "rkhs"]
    if args.run_label and len(models) != 1:
        parser.error("--run-label requires exactly one --model")
    mapping = args.data_root.parent / "data" / "interim" / "environment_id_map.csv"
    args.results.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for crop in crops:
        registry_path = args.splits / f"{crop.lower()}.json"
        assert_registry_locked(registry_path)
        registry = read_registry(registry_path)
        all_rows, traits, _ = read_phenotypes(args.data_root, crop, mapping)
        allowed = {int(item["row_id"]) for item in registry["observations"]}
        rows = [row for row in all_rows if int(row["row_id"]) in allowed]
        sample_ids, dosage, marker_ids = load_genotype_dosage(args.data_root, crop, Path("data/interim/genotype_cache"))
        genotype_index = {value: i for i, value in enumerate(sample_ids)}
        weather, weather_names, _ = read_environment_features(args.data_root, crop, mapping, feature_set="dynamic")
        for protocol in protocols:
            splits = registry_splits(rows, registry, protocol, args.minimum_validation_rows)
            if args.limit_folds is not None:
                splits = splits[: args.limit_folds]
            for split in splits:
                assert_protocol_isolation(rows, split.train_idx, split.validation_idx, protocol)
                for trait in traits:
                    train_idx = split.train_idx[np.asarray([np.isfinite(rows[i][trait]) for i in split.train_idx])]
                    valid_idx = split.validation_idx[np.asarray([np.isfinite(rows[i][trait]) for i in split.validation_idx])]
                    if len(valid_idx) < args.minimum_validation_rows:
                        continue
                    train_gids, train_obs_g = unique_mapping([rows[i]["genotype_id"] for i in train_idx])
                    test_gids, test_obs_g = unique_mapping([rows[i]["genotype_id"] for i in valid_idx])
                    train_gidx = np.asarray([genotype_index[value] for value in train_gids])
                    test_gidx = np.asarray([genotype_index[value] for value in test_gids])
                    max_marker_missing = args.max_marker_missing if args.max_marker_missing is not None else (0.40 if crop == "Wheat" else 0.25)
                    genomic = FullMarkerTransformer(min_maf=0.01, max_missing=max_marker_missing, low_rank_imputation_rank=args.imputation_rank if args.genotype_imputation == "low_rank" else None, low_rank_imputation_weight=args.imputation_weight).fit(dosage, train_gidx, sample_ids)
                    x_train = genomic.transform(dosage, train_gidx)
                    x_test = genomic.transform(dosage, test_gidx)
                    train_env_ids = [rows[i]["environment_id"] for i in train_idx]
                    test_env_ids = [rows[i]["environment_id"] for i in valid_idx]
                    env_transform = FoldEnvironmentTransformer(n_components={"Rice": 3, "Maize": 5, "Wheat": 5, "Soybean": 6}[crop]).fit(weather, train_env_ids)
                    unique_train_e, train_obs_e = unique_mapping(train_env_ids)
                    unique_test_e, test_obs_e = unique_mapping(test_env_ids)
                    train_e_pc = env_transform.transform(weather, unique_train_e)
                    test_e_pc = env_transform.transform(weather, unique_test_e)
                    artifact_dir = args.results / "artifacts" / crop.lower() / trait / protocol / str(split.fold)
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    np.save(artifact_dir / "marker_indices.npy", genomic.marker_indices)
                    (artifact_dir / "preprocessing.json").write_text(json.dumps({**genomic.metadata(), "marker_ids_sha256": __import__("hashlib").sha256("\n".join(marker_ids[i] for i in genomic.marker_indices).encode()).hexdigest(), "environment_features": weather_names, "environment_components": int(train_e_pc.shape[1])}, indent=2), encoding="utf-8")
                    y_train = np.asarray([rows[i][trait] for i in train_idx], dtype=np.float32)
                    y_valid = np.asarray([rows[i][trait] for i in valid_idx], dtype=np.float32)
                    calibrator = None
                    if any(name.startswith("distributional_") for name in models):
                        calibrator = EnvironmentLocationScale().fit(train_env_ids, y_train, unique_train_e, train_e_pc)
                        standardized_y = calibrator.normalize(train_env_ids, y_train)
                    for model_name in models:
                        started = time.monotonic()
                        distributional = model_name.startswith("distributional_")
                        underlying_name = model_name.removeprefix("distributional_")
                        kernel_kind = "rbf" if underlying_name == "rkhs" else "linear"
                        train_factor_unique, test_factor_unique = environment_kernel_factors(train_e_pc, test_e_pc, kernel_kind, args.rbf_bandwidth)
                        train_factor, test_factor = train_factor_unique[train_obs_e], test_factor_unique[test_obs_e]
                        model = MatrixFreeReactionNorm(underlying_name, residual_ratio=args.residual_ratio, max_iter=args.max_cg_iterations, device=args.device).fit(x_train, train_obs_g, train_factor, standardized_y if distributional else y_train, genomic.denominator)
                        prediction = model.predict(x_test, test_obs_g, test_factor)
                        if distributional:
                            assert calibrator is not None
                            prediction = calibrator.restore(prediction, test_env_ids, test_e_pc[test_obs_e])
                        elapsed = time.monotonic() - started
                        metrics = metric_dict(y_valid, prediction)
                        metrics.update(summarize_by_environment(np.asarray(test_env_ids, dtype=object), y_valid, prediction))
                        recorded_name = args.run_label or model_name
                        metrics.update({"crop": crop, "trait": trait, "protocol": protocol, "fold": split.fold, "model": recorded_name, "runtime_seconds": elapsed, "n_train": len(train_idx), "n_validation": len(valid_idx), "n_markers": len(genomic.marker_indices), "cg_iterations": model.fit_.cg_iterations, "cg_residual": model.fit_.cg_residual, "residual_ratio": args.residual_ratio, "rbf_bandwidth": args.rbf_bandwidth, "calibration_mean_alpha": calibrator.mean_alpha_ if distributional else "", "calibration_scale_alpha": calibrator.scale_alpha_ if distributional else "", **{f"variance_{key}": value for key, value in model.fit_.variance_components.items()}})
                        fold_rows.append(metrics)
                        for idx, pred in zip(valid_idx, prediction):
                            prediction_rows.append({"crop": crop, "trait": trait, "protocol": protocol, "fold": split.fold, "model": recorded_name, "row_id": rows[idx]["row_id"], "genotype_id": rows[idx]["genotype_id"], "environment_id": rows[idx]["environment_id"], "true": rows[idx][trait], "prediction": float(pred)})
                        checkpoint(args.results, fold_rows, prediction_rows)
                        print(f"[{crop} {trait} {protocol} {split.fold} {model_name}] n={len(valid_idx)} score={metrics['competition_score']:.4f} time={elapsed:.1f}s", flush=True)
                    del x_train, x_test
    checkpoint(args.results, fold_rows, prediction_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
