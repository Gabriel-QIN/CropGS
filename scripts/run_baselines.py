#!/usr/bin/env python3
"""Run leakage-aware baseline models across the declared split protocols."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.io import load_genotype_dosage, read_phenotypes, select_genomic_markers
from src.evaluation.metrics import metric_dict, summarize_by_environment
from src.validation.splits import FoldSplit, assert_no_leakage, build_splits, write_manifest


ALL_PROTOCOLS = ["cv0", "cv_g", "cv_gcluster", "cv_e", "cv_ge"]
DEFAULT_MODELS = ["global_mean", "environment_mean", "genotype_mean", "additive_mean", "genomic_ridge"]


def fit_mean_predictions(rows: list[dict], train_idx: np.ndarray, validation_idx: np.ndarray, trait: str, model: str) -> np.ndarray:
    y_train = np.asarray([rows[i][trait] for i in train_idx], dtype=float)
    valid_train = np.isfinite(y_train)
    y_train = y_train[valid_train]
    train_idx = train_idx[valid_train]
    if len(y_train) == 0:
        return np.zeros(len(validation_idx), dtype=float)
    global_mean = float(np.mean(y_train))
    if model == "global_mean":
        return np.full(len(validation_idx), global_mean, dtype=float)
    group_key = "environment_id" if model == "environment_mean" else "genotype_id"
    if model not in {"environment_mean", "genotype_mean", "additive_mean"}:
        raise ValueError(model)
    group_values: dict[str, list[float]] = defaultdict(list)
    for index, value in zip(train_idx, y_train):
        group_values[rows[index][group_key]].append(float(value))
    group_means = {key: float(np.mean(values)) for key, values in group_values.items()}
    if model in {"environment_mean", "genotype_mean"}:
        return np.asarray([group_means.get(rows[i][group_key], global_mean) for i in validation_idx], dtype=float)
    environment_values: dict[str, list[float]] = defaultdict(list)
    genotype_values: dict[str, list[float]] = defaultdict(list)
    for index, value in zip(train_idx, y_train):
        environment_values[rows[index]["environment_id"]].append(float(value))
        genotype_values[rows[index]["genotype_id"]].append(float(value))
    environment_means = {key: float(np.mean(values)) for key, values in environment_values.items()}
    genotype_means = {key: float(np.mean(values)) for key, values in genotype_values.items()}
    return np.asarray(
        [
            global_mean
            + (genotype_means.get(rows[i]["genotype_id"], global_mean) - global_mean if rows[i]["genotype_id"] in genotype_means else 0.0)
            + (environment_means.get(rows[i]["environment_id"], global_mean) - global_mean if rows[i]["environment_id"] in environment_means else 0.0)
            for i in validation_idx
        ],
        dtype=float,
    )


def fit_genomic_ridge(
    rows: list[dict],
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    trait: str,
    genotype_index: dict[str, int],
    dosage: np.ndarray,
    alpha: float = 10.0,
    max_markers: int = 512,
) -> np.ndarray:
    """Primal marker-ridge baseline, equivalent to a low-rank GBLUP approximation."""
    y_by_genotype: dict[str, list[float]] = defaultdict(list)
    for index in train_idx:
        value = float(rows[index][trait])
        if np.isfinite(value):
            y_by_genotype[rows[index]["genotype_id"]].append(value)
    training_genotypes = [gid for gid in sorted(y_by_genotype) if gid in genotype_index]
    if len(training_genotypes) < 3:
        return fit_mean_predictions(rows, train_idx, validation_idx, trait, "global_mean")
    training_indices = np.asarray([genotype_index[gid] for gid in training_genotypes], dtype=int)
    y = np.asarray([np.mean(y_by_genotype[gid]) for gid in training_genotypes], dtype=np.float64)
    marker_idx = select_genomic_markers(dosage[training_indices], max_markers=max_markers, min_maf=0.01, max_missing=0.25)
    if len(marker_idx) == 0:
        return fit_mean_predictions(rows, train_idx, validation_idx, trait, "global_mean")
    x_train = dosage[training_indices][:, marker_idx].astype(np.float32)
    observed = x_train >= 0
    counts = observed.sum(axis=0)
    means = np.divide(np.where(observed, x_train, 0).sum(axis=0), counts, out=np.zeros(len(marker_idx), dtype=np.float32), where=counts > 0)
    x_train[~observed] = np.take(means, np.where(~observed)[1])
    centers = x_train.mean(axis=0)
    scales = x_train.std(axis=0)
    scales[scales < 1e-6] = 1.0
    x_train = (x_train - centers) / scales
    y_center = y - float(np.mean(y))
    gram = x_train.T @ x_train
    gram.flat[:: gram.shape[0] + 1] += alpha
    try:
        beta = np.linalg.solve(gram.astype(np.float64), (x_train.T @ y_center).astype(np.float64))
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(gram.astype(np.float64), (x_train.T @ y_center).astype(np.float64), rcond=None)[0]

    validation_genotypes = [rows[index]["genotype_id"] for index in validation_idx]
    known = np.asarray([genotype_index.get(gid, -1) for gid in validation_genotypes], dtype=int)
    pred = np.full(len(validation_idx), float(np.mean(y)), dtype=float)
    known_mask = known >= 0
    if np.any(known_mask):
        x_val = dosage[known[known_mask]][:, marker_idx].astype(np.float32)
        observed_val = x_val >= 0
        x_val[~observed_val] = np.take(means, np.where(~observed_val)[1])
        x_val = (x_val - centers) / scales
        pred[known_mask] += x_val @ beta
    return pred


def cluster_map(audit_crop: dict) -> dict[str, int]:
    structure = audit_crop["genotypes"].get("structure", {})
    return {sample_id: int(label) for sample_id, label in zip(audit_crop["genotypes"]["sample_ids"], structure.get("cluster_labels", []))}


def write_oof(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["crop", "protocol", "model", "trait", "row_id", "genotype_id", "environment_id", "fold", "true", "prediction"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run_crop(
    crop: str,
    data_root: Path,
    audit_crop: dict,
    results_root: Path,
    cache_dir: Path,
    protocols: list[str],
    models: list[str],
    genomic_protocols: set[str],
    seed: int,
    use_cache: bool,
) -> tuple[list[dict], list[dict]]:
    rows, traits, _ = read_phenotypes(data_root, crop, data_root.parent / "data" / "interim" / "environment_id_map.csv")
    cluster_labels = cluster_map(audit_crop)
    genotype_ids: list[str] = []
    dosage = None
    genotype_index: dict[str, int] = {}
    if "genomic_ridge" in models and any(protocol in genomic_protocols for protocol in protocols):
        print(f"[{crop}] loading genotype dosage/cache", flush=True)
        genotype_ids, dosage, _ = load_genotype_dosage(data_root, crop, cache_dir=cache_dir, use_cache=use_cache)
        genotype_index = {gid: index for index, gid in enumerate(genotype_ids)}
    all_metric_rows: list[dict] = []
    all_fold_metric_rows: list[dict] = []
    for protocol in protocols:
        print(f"[{crop}] building {protocol}", flush=True)
        splits = build_splits(rows, protocol, seed=seed, cluster_labels=cluster_labels if protocol == "cv_gcluster" else None)
        for split in splits:
            assert_no_leakage(rows, split, protocol)
        manifest_path = results_root / "folds" / crop.lower() / f"{protocol}.csv"
        write_manifest(manifest_path, rows, protocol, splits)
        model_records: dict[tuple[str, str], list[dict]] = defaultdict(list)
        protocol_models = [model for model in models if model != "genomic_ridge" or protocol in genomic_protocols]
        for trait in traits:
            for split in splits:
                train_idx = split.train_idx[np.asarray([np.isfinite(rows[i][trait]) for i in split.train_idx])]
                validation_idx = split.validation_idx[np.asarray([np.isfinite(rows[i][trait]) for i in split.validation_idx])]
                if not len(validation_idx):
                    continue
                predictions: dict[str, np.ndarray] = {}
                for model in protocol_models:
                    if model == "genomic_ridge":
                        assert dosage is not None
                        predictions[model] = fit_genomic_ridge(rows, train_idx, validation_idx, trait, genotype_index, dosage)
                    else:
                        predictions[model] = fit_mean_predictions(rows, train_idx, validation_idx, trait, model)
                for model, values in predictions.items():
                    for index, prediction in zip(validation_idx, values):
                        model_records[(model, trait)].append(
                            {
                                "crop": crop,
                                "protocol": protocol,
                                "model": model,
                                "trait": trait,
                                "row_id": rows[index]["row_id"],
                                "genotype_id": rows[index]["genotype_id"],
                                "environment_id": rows[index]["environment_id"],
                                "fold": split.fold,
                                "true": float(rows[index][trait]),
                                "prediction": float(prediction),
                            }
                        )
        oof_records = [record for records in model_records.values() for record in records]
        write_oof(results_root / "oof" / crop.lower() / protocol / "baseline_oof.csv", oof_records)
        for (model, trait), records in sorted(model_records.items()):
            y = np.asarray([record["true"] for record in records], dtype=float)
            prediction = np.asarray([record["prediction"] for record in records], dtype=float)
            environments = np.asarray([record["environment_id"] for record in records], dtype=object)
            metrics = metric_dict(y, prediction)
            metrics.update(summarize_by_environment(environments, y, prediction))
            fold_scores = []
            for fold in sorted(set(str(record["fold"]) for record in records)):
                fold_records = [record for record in records if str(record["fold"]) == fold]
                fold_y = np.asarray([record["true"] for record in fold_records], dtype=float)
                fold_prediction = np.asarray([record["prediction"] for record in fold_records], dtype=float)
                fold_metrics = metric_dict(fold_y, fold_prediction)
                fold_metrics.update({"crop": crop, "protocol": protocol, "model": model, "trait": trait, "fold": fold})
                all_fold_metric_rows.append(fold_metrics)
                fold_scores.append(float(fold_metrics["competition_score"]))
            metrics.update({"fold_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0, "worst_fold_score": float(np.min(fold_scores)), "best_fold_score": float(np.max(fold_scores))})
            metrics.update({"crop": crop, "protocol": protocol, "model": model, "trait": trait})
            all_metric_rows.append(metrics)
        print(f"[{crop}] {protocol}: {sum(len(v) for v in model_records.values())} OOF predictions", flush=True)
    return all_metric_rows, all_fold_metric_rows


def write_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "crop", "protocol", "model", "trait", "n", "pearson", "spearman", "competition_score", "mse", "rmse", "mae",
        "pearson_defined", "spearman_defined", "n_environments", "environment_macro_score", "environment_weighted_score",
        "environment_macro_pearson", "environment_macro_spearman", "fold_std", "worst_fold_score", "best_fold_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_fold_metrics(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["crop", "protocol", "model", "trait", "fold", "n", "pearson", "spearman", "competition_score", "mse", "rmse", "mae", "pearson_defined", "spearman_defined"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_summary(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Baseline performance summary",
        "",
        "Competition score is `(Pearson + Spearman) / 2`; constant predictions are assigned correlation 0 and marked undefined in the CSV. `genomic_ridge` uses up to 512 genotype-only markers and is a low-rank GBLUP-like baseline, not full-marker GBLUP.",
        "",
        "| Crop | Protocol | Model | Trait | N | Score | Pearson | Spearman | Env macro | Fold SD | Worst fold | MSE |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["crop"], item["trait"], item["protocol"], -float(item["competition_score"]))):
        lines.append(
            f"| {row['crop']} | {row['protocol']} | {row['model']} | {row['trait']} | {row['n']} | {float(row['competition_score']):.4f} | {float(row['pearson']):.4f} | {float(row['spearman']):.4f} | {float(row['environment_macro_score']):.4f} | {float(row['fold_std']):.4f} | {float(row['worst_fold_score']):.4f} | {float(row['mse']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--audit-json", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim/genotype_cache"))
    parser.add_argument("--crop", action="append", choices=["Maize", "Rice", "Wheat", "Soybean"])
    parser.add_argument("--protocol", action="append", choices=ALL_PROTOCOLS)
    parser.add_argument("--model", action="append", choices=DEFAULT_MODELS)
    parser.add_argument("--genomic-protocol", action="append", choices=ALL_PROTOCOLS, help="Protocols that run genomic_ridge; default excludes expensive CV-E.")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    crops = args.crop or ["Maize", "Rice", "Wheat", "Soybean"]
    protocols = args.protocol or ALL_PROTOCOLS
    models = args.model or DEFAULT_MODELS
    genomic_protocols = set(args.genomic_protocol or ["cv_g", "cv_gcluster", "cv_ge"])
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    metrics: list[dict] = []
    fold_metrics: list[dict] = []
    for crop in crops:
        crop_metrics, crop_fold_metrics = run_crop(crop, args.data_root, audit[crop], args.results_root, args.cache_dir, protocols, models, genomic_protocols, args.seed, not args.no_cache)
        metrics.extend(crop_metrics)
        fold_metrics.extend(crop_fold_metrics)
    write_metrics(args.results_root / "metrics" / "baseline_metrics.csv", metrics)
    write_fold_metrics(args.results_root / "metrics" / "baseline_fold_metrics.csv", fold_metrics)
    write_summary(args.results_root / "metrics" / "baseline_summary.md", metrics)
    print(f"Wrote {len(metrics)} metric rows to {args.results_root / 'metrics' / 'baseline_metrics.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
