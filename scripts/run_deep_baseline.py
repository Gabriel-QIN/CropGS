#!/usr/bin/env python3
"""Train a leakage-controlled deep genotype x environment baseline.

The command deliberately reuses the validation protocols from
``src.validation.splits``.  It is a development benchmark, not a claim that a
larger neural network automatically generalises better than GBLUP.  The
default run is CV-G; add CV-GCluster/CV-E/CV-GE explicitly for the stress
tests described in MODEL_TRAINING_VALIDATION_PLAN.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - gives a useful CLI error
    raise SystemExit("PyTorch is required for run_deep_baseline.py; install torch>=2.0 first") from exc

from src.data.io import load_genotype_dosage, read_phenotypes
from src.evaluation.metrics import metric_dict, summarize_by_environment
from src.models.deep.genotype_environment_net import GenotypeEnvironmentNet, ReactionNormNet
from src.preprocessing.deep_features import FoldFeatureTransformer, read_environment_features
from src.validation.splits import assert_no_leakage, build_splits, stable_bucket, write_manifest


ALL_PROTOCOLS = ["cv0", "cv_g", "cv_gcluster", "cv_e", "cv_ge"]
DEFAULT_CROPS = ["Maize", "Rice", "Wheat", "Soybean"]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 45
    patience: int = 8
    batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 2e-4
    dropout: float = 0.20
    max_grad_norm: float = 1.0
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 3
    block_size: int = 256
    head_hidden: int = 256
    max_markers: int = 4096
    min_maf: float = 0.01
    max_missing: float = 0.25
    environment_components: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN operations are available for this model and make
    # fold comparisons reproducible.  We do not force deterministic algorithms
    # globally because some CUDA kernels are not implemented in that mode.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but torch.cuda.is_available() is false")
    return device


def finite_indices(rows: list[dict], indices: Iterable[int], trait: str) -> np.ndarray:
    indices = np.asarray(list(indices), dtype=int)
    return indices[np.asarray([np.isfinite(rows[index][trait]) for index in indices], dtype=bool)]


def inner_split_positions(
    rows: list[dict],
    outer_train_idx: np.ndarray,
    protocol: str,
    seed: int,
    min_validation: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Make an early-stopping split using only outer-training rows.

    The split is group-aware: genotype groups are held out for the usual
    protocol, environments for LOEO, and both groups for CV-GE when enough
    rows remain.  It is never used to select the outer validation result.
    """

    outer_train_idx = np.asarray(outer_train_idx, dtype=int)
    if len(outer_train_idx) < 4:
        return outer_train_idx, outer_train_idx[:0]
    if protocol == "cv_e":
        keys = [f"environment|{rows[index]['environment_id']}" for index in outer_train_idx]
        assignments = np.asarray([stable_bucket(key, seed + 17, 5) for key in keys])
        validation = assignments == 0
        training = ~validation
    elif protocol == "cv_ge":
        genotype_assignments = np.asarray(
            [stable_bucket(f"inner-genotype|{rows[index]['genotype_id']}", seed + 17, 5) for index in outer_train_idx]
        )
        environment_assignments = np.asarray(
            [stable_bucket(f"inner-environment|{rows[index]['environment_id']}", seed + 19, 5) for index in outer_train_idx]
        )
        validation = (genotype_assignments == 0) & (environment_assignments == 0)
        training = (genotype_assignments != 0) & (environment_assignments != 0)
        if int(validation.sum()) < min_validation or int(training.sum()) < min_validation:
            validation = genotype_assignments == 0
            training = ~validation
    else:
        keys = [f"genotype|{rows[index]['genotype_id']}" for index in outer_train_idx]
        assignments = np.asarray([stable_bucket(key, seed + 17, 5) for key in keys])
        validation = assignments == 0
        training = ~validation
    if int(validation.sum()) < min_validation or int(training.sum()) < min_validation:
        # The fallback is still deterministic and uses no target values.  It
        # is only reachable for very small folds or a highly unbalanced group.
        row_assignments = np.asarray([stable_bucket(f"row|{rows[index]['row_id']}", seed + 23, 5) for index in outer_train_idx])
        validation = row_assignments == 0
        training = ~validation
    if not validation.any() or not training.any():
        cut = max(1, int(round(len(outer_train_idx) * 0.85)))
        return outer_train_idx[:cut], outer_train_idx[cut:]
    return outer_train_idx[training], outer_train_idx[validation]


def build_row_features(
    rows: list[dict],
    row_indices: np.ndarray,
    genotype_index: dict[str, int],
    dosage: np.ndarray,
    transformer: FoldFeatureTransformer,
    environment_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Materialise one fold's observation-level tensors from genotype IDs."""

    unique_ids = sorted({rows[index]["genotype_id"] for index in row_indices})
    known = np.asarray([genotype_index.get(genotype_id, -1) for genotype_id in unique_ids], dtype=int)
    genotype_dosage = np.zeros((len(unique_ids), transformer.n_markers), dtype=np.float32)
    genotype_missing = np.ones((len(unique_ids), transformer.n_markers), dtype=np.float32)
    known_mask = known >= 0
    if known_mask.any():
        transformed, missing = transformer.transform_genotypes(np.asarray(dosage[known[known_mask]]))
        genotype_dosage[known_mask] = transformed
        genotype_missing[known_mask] = missing
    genotype_lookup = {genotype_id: position for position, genotype_id in enumerate(unique_ids)}
    genotype_positions = np.asarray([genotype_lookup[rows[index]["genotype_id"]] for index in row_indices], dtype=int)
    environment_ids = [rows[index]["environment_id"] for index in row_indices]
    environment = transformer.transform_environments(environment_ids, environment_features)
    return genotype_dosage[genotype_positions], genotype_missing[genotype_positions], environment, known, genotype_lookup


def target_scale(y: np.ndarray) -> tuple[float, float]:
    center = float(np.mean(y))
    scale = float(np.std(y))
    return center, scale if np.isfinite(scale) and scale >= 1e-6 else 1.0


def make_model(n_markers: int, n_environment_features: int, config: TrainConfig, architecture: str = "gated") -> GenotypeEnvironmentNet:
    model_class = {"gated": GenotypeEnvironmentNet, "reaction_norm": ReactionNormNet}[architecture]
    return model_class(
        n_markers=n_markers,
        n_environment_features=n_environment_features,
        block_size=config.block_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        head_hidden=config.head_hidden,
    )


def loader_for(
    dosage: np.ndarray,
    missing: np.ndarray,
    environment: np.ndarray,
    y: np.ndarray,
    positions: np.ndarray,
    center: float,
    scale: float,
    config: TrainConfig,
    seed: int,
) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(np.asarray(dosage[positions], dtype=np.float32)),
        torch.from_numpy(np.asarray(missing[positions], dtype=np.float32)),
        torch.from_numpy(np.asarray(environment[positions], dtype=np.float32)),
        torch.from_numpy(((np.asarray(y[positions], dtype=np.float32) - center) / scale).astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(tensors, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=torch.cuda.is_available())


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for dosage, missing, environment, target in loader:
            prediction = model(dosage.to(device, non_blocking=True), missing.to(device, non_blocking=True), environment.to(device, non_blocking=True))
            loss = criterion(prediction, target.to(device, non_blocking=True))
            total += float(loss.item()) * len(target)
            count += len(target)
    return total / max(1, count)


def train_with_inner_early_stopping(
    model: GenotypeEnvironmentNet,
    dosage: np.ndarray,
    missing: np.ndarray,
    environment: np.ndarray,
    y: np.ndarray,
    training_positions: np.ndarray,
    stopping_positions: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    seed: int,
) -> tuple[int, dict[str, torch.Tensor], list[dict[str, float]]]:
    inner_center, inner_scale = target_scale(y[training_positions])
    train_loader = loader_for(dosage, missing, environment, y, training_positions, inner_center, inner_scale, config, seed)
    stopping_loader = loader_for(dosage, missing, environment, y, stopping_positions, inner_center, inner_scale, config, seed + 1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.SmoothL1Loss(beta=1.0)
    best_loss = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] = {}
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for batch_dosage, batch_missing, batch_environment, batch_target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch_dosage.to(device, non_blocking=True),
                batch_missing.to(device, non_blocking=True),
                batch_environment.to(device, non_blocking=True),
            )
            loss = criterion(prediction, batch_target.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            train_total += float(loss.item()) * len(batch_target)
            train_count += len(batch_target)
        train_loss = train_total / max(1, train_count)
        validation_loss = evaluate_loss(model, stopping_loader, device, criterion) if len(stopping_positions) else train_loss
        history.append({"epoch": float(epoch), "train_loss": train_loss, "stopping_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    if not best_state:
        best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    return best_epoch, best_state, history


def train_fixed_epochs(
    model: GenotypeEnvironmentNet,
    dosage: np.ndarray,
    missing: np.ndarray,
    environment: np.ndarray,
    y: np.ndarray,
    positions: np.ndarray,
    epochs: int,
    config: TrainConfig,
    device: torch.device,
    seed: int,
) -> None:
    center, scale = target_scale(y[positions])
    loader = loader_for(dosage, missing, environment, y, positions, center, scale, config, seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.SmoothL1Loss(beta=1.0)
    for _ in range(max(1, epochs)):
        model.train()
        for batch_dosage, batch_missing, batch_environment, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch_dosage.to(device, non_blocking=True),
                batch_missing.to(device, non_blocking=True),
                batch_environment.to(device, non_blocking=True),
            )
            loss = criterion(prediction, batch_target.to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()


def predict(
    model: GenotypeEnvironmentNet,
    dosage: np.ndarray,
    missing: np.ndarray,
    environment: np.ndarray,
    positions: np.ndarray,
    y_center: float,
    y_scale: float,
    config: TrainConfig,
    device: torch.device,
) -> np.ndarray:
    if not len(positions):
        return np.empty(0, dtype=np.float64)
    dummy_y = np.zeros(len(positions), dtype=np.float32)
    tensors = TensorDataset(
        torch.from_numpy(np.asarray(dosage[positions], dtype=np.float32)),
        torch.from_numpy(np.asarray(missing[positions], dtype=np.float32)),
        torch.from_numpy(np.asarray(environment[positions], dtype=np.float32)),
        torch.from_numpy(dummy_y),
    )
    loader = DataLoader(tensors, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_dosage, batch_missing, batch_environment, _ in loader:
            value = model(
                batch_dosage.to(device, non_blocking=True),
                batch_missing.to(device, non_blocking=True),
                batch_environment.to(device, non_blocking=True),
            )
            output.append(value.detach().cpu().numpy())
    return np.concatenate(output).astype(np.float64) * y_scale + y_center


def cluster_map(audit_crop: dict) -> dict[str, int]:
    structure = audit_crop["genotypes"].get("structure", {})
    return {sample_id: int(label) for sample_id, label in zip(audit_crop["genotypes"]["sample_ids"], structure.get("cluster_labels", []))}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def run_crop(
    crop: str,
    data_root: Path,
    audit_crop: dict,
    results_root: Path,
    cache_dir: Path,
    protocols: list[str],
    config: TrainConfig,
    seed: int,
    device: torch.device,
    save_checkpoints: bool,
    fold_limit: int | None,
    run_tag: str,
    architecture: str,
    environment_feature_set: str,
    development_only: bool,
    minimum_validation_rows: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    mapping_path = data_root.parent / "data" / "interim" / "environment_id_map.csv"
    rows, traits, _ = read_phenotypes(data_root, crop, mapping_path)
    if development_only:
        manifest_path = results_root / "folds" / crop.lower() / "independent_test.csv"
        with manifest_path.open("r", newline="", encoding="utf-8") as fh:
            development_row_ids = {int(row["row_id"]) for row in csv.DictReader(fh) if row["role"] == "development"}
        rows = [row for row in rows if int(row["row_id"]) in development_row_ids]
        if not rows:
            raise ValueError(f"No development rows found in {manifest_path}")
    environment_features, environment_feature_names, _ = read_environment_features(data_root, crop, mapping_path, feature_set=environment_feature_set)
    print(f"[{crop}] loading genotype dosage/cache", flush=True)
    genotype_ids, dosage, _ = load_genotype_dosage(data_root, crop, cache_dir=cache_dir, use_cache=True)
    genotype_index = {genotype_id: index for index, genotype_id in enumerate(genotype_ids)}
    cluster_labels = cluster_map(audit_crop)
    metric_rows: list[dict] = []
    fold_metric_rows: list[dict] = []
    oof_by_protocol: dict[str, list[dict]] = defaultdict(list)
    distribution_rows: list[dict] = []
    history_rows: list[dict] = []

    for protocol in protocols:
        print(f"[{crop}] building {protocol}", flush=True)
        all_splits = build_splits(rows, protocol, seed=seed, cluster_labels=cluster_labels if protocol == "cv_gcluster" else None)
        for split in all_splits:
            assert_no_leakage(rows, split, protocol)
        # Keep the complete deterministic outer split next to the OOF files.
        # A debug fold limit changes compute only; it never changes the
        # canonical manifest used for comparison.
        manifest_name = f"development_{protocol}.csv" if development_only else f"{protocol}.csv"
        write_manifest(results_root / "folds" / crop.lower() / manifest_name, rows, protocol, all_splits)
        splits = all_splits[:fold_limit] if fold_limit is not None else all_splits
        for trait in traits:
            model_records: list[dict] = []
            for split in splits:
                train_idx = finite_indices(rows, split.train_idx, trait)
                validation_idx = finite_indices(rows, split.validation_idx, trait)
                if len(validation_idx) < minimum_validation_rows:
                    print(f"[{crop}] {protocol} trait={trait} fold={split.fold}: SKIPPED_INSUFFICIENT_N n={len(validation_idx)}", flush=True)
                    continue
                if len(train_idx) < 4:
                    continue
                train_genotype_indices = np.asarray(
                    sorted({genotype_index[rows[index]["genotype_id"]] for index in train_idx if rows[index]["genotype_id"] in genotype_index}),
                    dtype=int,
                )
                transformer = FoldFeatureTransformer(
                    max_markers=config.max_markers,
                    min_maf=config.min_maf,
                    max_missing=config.max_missing,
                    environment_components=config.environment_components,
                ).fit(
                    dosage,
                    train_genotype_indices,
                    environment_features,
                    [rows[index]["environment_id"] for index in train_idx],
                    environment_feature_names,
                )
                all_idx = np.concatenate((train_idx, validation_idx))
                genotype_dosage, genotype_missing, environment, _, _ = build_row_features(
                    rows,
                    all_idx,
                    genotype_index,
                    dosage,
                    transformer,
                    environment_features,
                )
                n_train = len(train_idx)
                train_x = genotype_dosage[:n_train]
                train_m = genotype_missing[:n_train]
                train_e = environment[:n_train]
                validation_x = genotype_dosage[n_train:]
                validation_m = genotype_missing[n_train:]
                validation_e = environment[n_train:]
                feature_dosage = np.concatenate((train_x, validation_x), axis=0)
                feature_missing = np.concatenate((train_m, validation_m), axis=0)
                feature_environment = np.concatenate((train_e, validation_e), axis=0)
                y = np.asarray([float(rows[index][trait]) for index in all_idx], dtype=np.float32)
                fold_token = stable_bucket(f"fold|{split.fold}", seed, 1000003)
                inner_seed = seed + fold_token
                inner_train_global, inner_validation_global = inner_split_positions(rows, train_idx, protocol, inner_seed)
                global_to_local = {int(index): position for position, index in enumerate(train_idx)}
                inner_train_positions = np.asarray([global_to_local[int(index)] for index in inner_train_global], dtype=int)
                inner_validation_positions = np.asarray([global_to_local[int(index)] for index in inner_validation_global], dtype=int)
                set_seed(seed + 1009 * fold_token + len(metric_rows))
                model = make_model(transformer.n_markers, transformer.n_environment_features, config, architecture)
                parameter_count = model.parameter_count()
                fold_start = time.time()
                best_epoch, _, history = train_with_inner_early_stopping(
                    model,
                    feature_dosage[:n_train],
                    feature_missing[:n_train],
                    feature_environment[:n_train],
                    y[:n_train],
                    inner_train_positions,
                    inner_validation_positions,
                    config,
                    device,
                    inner_seed,
                )
                for item in history:
                    history_rows.append(
                        {
                            "crop": crop,
                            "protocol": protocol,
                            "trait": trait,
                            "fold": split.fold,
                            "phase": "inner_early_stopping",
                            "epoch": int(item["epoch"]),
                            "train_loss": item["train_loss"],
                            "stopping_loss": item["stopping_loss"],
                            "best_epoch": best_epoch,
                        }
                    )
                # Refit on every outer-training observation for the epoch count
                # selected inside the fold.  The outer validation labels are
                # never used in this phase.
                set_seed(seed + 2009 * fold_token + len(metric_rows))
                final_model = make_model(transformer.n_markers, transformer.n_environment_features, config, architecture)
                train_fixed_epochs(final_model, feature_dosage[:n_train], feature_missing[:n_train], feature_environment[:n_train], y[:n_train], np.arange(n_train), best_epoch, config, device, seed)
                y_center, y_scale = target_scale(y[:n_train])
                predictions = predict(
                    final_model,
                    feature_dosage[n_train:],
                    feature_missing[n_train:],
                    feature_environment[n_train:],
                    np.arange(len(validation_idx)),
                    y_center,
                    y_scale,
                    config,
                    device,
                )
                for index, prediction in zip(validation_idx, predictions):
                    record = {
                        "crop": crop,
                        "protocol": protocol,
                        "model": architecture,
                        "trait": trait,
                        "row_id": rows[index]["row_id"],
                        "genotype_id": rows[index]["genotype_id"],
                        "environment_id": rows[index]["environment_id"],
                        "fold": split.fold,
                        "true": float(rows[index][trait]),
                        "prediction": float(prediction),
                        "seed": seed,
                        "n_selected_markers": transformer.n_markers,
                        "parameter_count": parameter_count,
                        "best_epoch": best_epoch,
                    }
                    model_records.append(record)
                    oof_by_protocol[protocol].append(record)
                train_genotype_ids = [rows[index]["genotype_id"] for index in train_idx]
                validation_genotype_ids = [rows[index]["genotype_id"] for index in validation_idx]
                train_genotype_unique = np.asarray(sorted({genotype_index[g] for g in train_genotype_ids if g in genotype_index}), dtype=int)
                validation_genotype_unique = np.asarray(sorted({genotype_index[g] for g in validation_genotype_ids if g in genotype_index}), dtype=int)
                distribution = transformer.distribution_summary(
                    dosage,
                    train_genotype_unique,
                    validation_genotype_unique,
                    [rows[index]["environment_id"] for index in train_idx],
                    [rows[index]["environment_id"] for index in validation_idx],
                    environment_features,
                )
                distribution.update(
                    {
                        "crop": crop,
                        "protocol": protocol,
                        "trait": trait,
                        "fold": split.fold,
                        "n_train_rows": len(train_idx),
                        "n_validation_rows": len(validation_idx),
                        "n_train_genotypes": len(set(train_genotype_ids)),
                        "n_validation_genotypes": len(set(validation_genotype_ids)),
                        "genotype_overlap": len(set(train_genotype_ids) & set(validation_genotype_ids)),
                        "environment_overlap": len(
                            set(rows[index]["environment_id"] for index in train_idx)
                            & set(rows[index]["environment_id"] for index in validation_idx)
                        ),
                        "runtime_seconds": time.time() - fold_start,
                        "parameter_count": parameter_count,
                    }
                )
                distribution_rows.append(distribution)
                if save_checkpoints:
                    checkpoint_path = results_root / "checkpoints" / "deep" / crop.lower() / protocol / f"{trait}_fold_{split.fold}.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state": {name: tensor.detach().cpu() for name, tensor in final_model.state_dict().items()},
                            "model_config": config.__dict__,
                            "architecture": architecture,
                            "environment_feature_set": environment_feature_set,
                            "n_markers": transformer.n_markers,
                            "marker_indices": transformer.marker_indices,
                            "transformer": transformer,
                            "crop": crop,
                            "protocol": protocol,
                            "trait": trait,
                            "fold": split.fold,
                            "seed": seed,
                        },
                        checkpoint_path,
                    )
                print(f"[{crop}] {protocol} trait={trait} fold={split.fold}: n={len(validation_idx)} best_epoch={best_epoch} markers={transformer.n_markers} score_pending", flush=True)
            if not model_records:
                continue
            y_true = np.asarray([record["true"] for record in model_records], dtype=float)
            y_pred = np.asarray([record["prediction"] for record in model_records], dtype=float)
            environments = np.asarray([record["environment_id"] for record in model_records], dtype=object)
            metrics = metric_dict(y_true, y_pred)
            metrics.update(summarize_by_environment(environments, y_true, y_pred))
            fold_scores: list[float] = []
            for fold in sorted({str(record["fold"]) for record in model_records}):
                fold_records = [record for record in model_records if str(record["fold"]) == fold]
                fold_y = np.asarray([record["true"] for record in fold_records], dtype=float)
                fold_pred = np.asarray([record["prediction"] for record in fold_records], dtype=float)
                fold_metrics = metric_dict(fold_y, fold_pred)
                fold_metrics.update({"crop": crop, "protocol": protocol, "model": architecture, "trait": trait, "fold": fold})
                fold_metric_rows.append(fold_metrics)
                fold_scores.append(float(fold_metrics["competition_score"]))
            metrics.update(
                {
                    "crop": crop,
                    "protocol": protocol,
                    "model": architecture,
                    "trait": trait,
                    "fold_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
                    "worst_fold_score": float(np.min(fold_scores)),
                    "best_fold_score": float(np.max(fold_scores)),
                }
            )
            metric_rows.append(metrics)
        write_csv(
            results_root / "oof" / crop.lower() / protocol / f"{run_tag}_oof.csv",
            oof_by_protocol[protocol],
            ["crop", "protocol", "model", "trait", "row_id", "genotype_id", "environment_id", "fold", "true", "prediction", "seed", "n_selected_markers", "parameter_count", "best_epoch"],
        )
    return metric_rows, fold_metric_rows, distribution_rows, history_rows


def write_summary(path: Path, rows: list[dict], architecture: str, environment_feature_set: str) -> None:
    lines = [
        "# Deep baseline performance summary",
        "",
        f"`{architecture}` uses fold-fitted dosage/missing-mask features and `{environment_feature_set}` weather features. Early stopping uses an inner training-only split; the outer validation fold is used once for OOF scoring.",
        "",
        "| Crop | Protocol | Trait | N | Score | Pearson | Spearman | Env macro | Fold SD | Worst fold | MSE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["crop"], item["trait"], item["protocol"])):
        lines.append(
            f"| {row['crop']} | {row['protocol']} | {row['trait']} | {row['n']} | {float(row['competition_score']):.4f} | {float(row['pearson']):.4f} | {float(row['spearman']):.4f} | {float(row['environment_macro_score']):.4f} | {float(row['fold_std']):.4f} | {float(row['worst_fold_score']):.4f} | {float(row['mse']):.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--audit-json", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim/genotype_cache"))
    parser.add_argument("--crop", action="append", choices=DEFAULT_CROPS)
    parser.add_argument("--protocol", action="append", choices=ALL_PROTOCOLS)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--max-markers", type=int, default=4096)
    parser.add_argument("--min-maf", type=float, default=0.01)
    parser.add_argument("--max-missing", type=float, default=0.25)
    parser.add_argument("--environment-components", type=int, default=None, help="fold-fitted low-rank weather representation; e.g. 6")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--fold-limit", type=int, default=None, help="debug limit per protocol; never use for final comparison")
    parser.add_argument("--run-tag", default="deep", help="output prefix, e.g. deep_strict; keep alphanumeric/underscore")
    parser.add_argument("--architecture", choices=["gated", "reaction_norm"], default="gated")
    parser.add_argument("--environment-feature-set", choices=["basic", "dynamic"], default="basic")
    parser.add_argument("--capacity", choices=["custom", "small", "base", "large"], default="custom", help="preset model/marker capacity; custom uses explicit flags")
    parser.add_argument("--development-only", action="store_true", help="exclude the frozen independent test manifest from all CV/model fitting")
    parser.add_argument("--minimum-validation-rows", type=int, default=100, help="skip folds below this size; no correlation is reported")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    crops = args.crop or DEFAULT_CROPS
    protocols = args.protocol or ["cv_g"]
    if not args.run_tag.replace("_", "").isalnum():
        raise ValueError("--run-tag must contain only letters, numbers, and underscores")
    device = device_from_arg(args.device)
    capacity = {
        "small": dict(d_model=96, n_heads=4, n_layers=2, head_hidden=128, max_markers=2048),
        "base": dict(d_model=192, n_heads=6, n_layers=3, head_hidden=256, max_markers=4096),
        "large": dict(d_model=384, n_heads=8, n_layers=6, head_hidden=512, max_markers=8192),
    }.get(args.capacity, {})
    config = TrainConfig(
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        d_model=capacity.get("d_model", args.d_model),
        n_heads=capacity.get("n_heads", args.n_heads),
        n_layers=capacity.get("n_layers", args.n_layers),
        head_hidden=capacity.get("head_hidden", args.head_hidden),
        max_markers=capacity.get("max_markers", args.max_markers),
        min_maf=args.min_maf,
        max_missing=args.max_missing,
        environment_components=args.environment_components,
    )
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    metrics: list[dict] = []
    fold_metrics: list[dict] = []
    distributions: list[dict] = []
    histories: list[dict] = []
    print(f"Using device={device}; architecture={args.architecture}; env_features={args.environment_feature_set}; capacity={args.capacity}; config={config}", flush=True)
    for crop in crops:
        crop_metrics, crop_fold_metrics, crop_distributions, crop_histories = run_crop(
            crop,
            args.data_root,
            audit[crop],
            args.results_root,
            args.cache_dir,
            protocols,
            config,
            args.seed,
            device,
            args.save_checkpoints,
            args.fold_limit,
            args.run_tag,
            args.architecture,
            args.environment_feature_set,
            args.development_only,
            args.minimum_validation_rows,
        )
        metrics.extend(crop_metrics)
        fold_metrics.extend(crop_fold_metrics)
        distributions.extend(crop_distributions)
        histories.extend(crop_histories)
    metric_fields = [
        "crop", "protocol", "model", "trait", "n", "pearson", "spearman", "competition_score", "mse", "rmse", "mae",
        "pearson_defined", "spearman_defined", "n_environments", "environment_macro_score", "environment_weighted_score",
        "environment_macro_pearson", "environment_macro_spearman", "fold_std", "worst_fold_score", "best_fold_score",
    ]
    fold_fields = ["crop", "protocol", "model", "trait", "fold", "n", "pearson", "spearman", "competition_score", "mse", "rmse", "mae", "pearson_defined", "spearman_defined"]
    distribution_fields = [
        "crop", "protocol", "trait", "fold", "n_train_rows", "n_validation_rows", "n_train_genotypes", "n_validation_genotypes",
        "genotype_overlap", "environment_overlap", "n_selected_markers", "train_genotype_missing_fraction", "validation_genotype_missing_fraction",
        "environment_standardized_mean_shift", "n_train_environments", "n_validation_environments", "runtime_seconds", "parameter_count",
    ]
    history_fields = ["crop", "protocol", "trait", "fold", "phase", "epoch", "train_loss", "stopping_loss", "best_epoch"]
    config_path = args.results_root / "metrics" / f"{args.run_tag}_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "run_tag": args.run_tag,
                "crops": crops,
                "protocols": protocols,
                "seed": args.seed,
                "device": str(device),
                "architecture": args.architecture,
                "environment_feature_set": args.environment_feature_set,
                "capacity": args.capacity,
                "development_only": args.development_only,
                "minimum_validation_rows": args.minimum_validation_rows,
                "train_config": config.__dict__,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(args.results_root / "metrics" / f"{args.run_tag}_metrics.csv", metrics, metric_fields)
    write_csv(args.results_root / "metrics" / f"{args.run_tag}_fold_metrics.csv", fold_metrics, fold_fields)
    write_csv(args.results_root / "metrics" / f"{args.run_tag}_distribution.csv", distributions, distribution_fields)
    write_csv(args.results_root / "metrics" / f"{args.run_tag}_training_history.csv", histories, history_fields)
    write_summary(args.results_root / "metrics" / f"{args.run_tag}_summary.md", metrics, args.architecture, args.environment_feature_set)
    print(f"Wrote {len(metrics)} metric rows and {len(distributions)} distribution rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
