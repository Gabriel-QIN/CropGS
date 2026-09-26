"""Deterministic validation protocols for genotype-by-environment data."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train_idx: np.ndarray
    validation_idx: np.ndarray
    purge_idx: np.ndarray


def stable_bucket(value: str, seed: int, n_buckets: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_buckets


def _rows_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([row["genotype_id"] for row in rows], dtype=object), np.asarray([row["environment_id"] for row in rows], dtype=object)


def _group_assignment(values: Iterable[str], seed: int, n_folds: int, prefix: str) -> dict[str, int]:
    unique = sorted(set(values))
    return {value: stable_bucket(f"{prefix}|{value}", seed, n_folds) for value in unique}


def _environment_assignment(environments: dict[str, int], n_folds: int) -> dict[str, int]:
    """Greedy balance by observed row count, independent of phenotype values."""
    loads = [0] * n_folds
    assignment: dict[str, int] = {}
    for environment, count in sorted(environments.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(n_folds), key=lambda i: (loads[i], i))
        assignment[environment] = fold
        loads[fold] += count
    return assignment


def build_splits(rows: list[dict], protocol: str, seed: int = 20260922, cluster_labels: dict[str, int] | None = None) -> list[FoldSplit]:
    genotype_ids, environment_ids = _rows_arrays(rows)
    n_rows = len(rows)
    all_indices = np.arange(n_rows, dtype=int)
    if protocol == "cv0":
        n_folds = 5
        row_folds = np.asarray([stable_bucket(f"row|{int(row['row_id'])}", seed, n_folds) for row in rows])
        return [FoldSplit(f, all_indices[row_folds != f], all_indices[row_folds == f], np.empty(0, dtype=int)) for f in range(n_folds)]
    if protocol == "cv_g":
        assignment = _group_assignment(genotype_ids.tolist(), seed, 5, "genotype")
        row_folds = np.asarray([assignment[value] for value in genotype_ids])
        return [FoldSplit(f, all_indices[row_folds != f], all_indices[row_folds == f], np.empty(0, dtype=int)) for f in range(5)]
    if protocol == "cv_gcluster":
        if not cluster_labels:
            raise ValueError("cv_gcluster requires genotype-only cluster labels")
        row_folds = np.asarray([int(cluster_labels[value]) for value in genotype_ids])
        fold_values = sorted(set(row_folds.tolist()))
        return [FoldSplit(fold, all_indices[row_folds != fold], all_indices[row_folds == fold], np.empty(0, dtype=int)) for fold in fold_values]
    if protocol == "cv_e":
        fold_values = sorted(set(environment_ids.tolist()))
        return [FoldSplit(f, all_indices[environment_ids != f], all_indices[environment_ids == f], np.empty(0, dtype=int)) for f in fold_values]
    if protocol == "cv_ge":
        genotype_assignment = _group_assignment(genotype_ids.tolist(), seed, 5, "genotype")
        environment_counts: dict[str, int] = {}
        for environment in environment_ids.tolist():
            environment_counts[environment] = environment_counts.get(environment, 0) + 1
        environment_assignment = _environment_assignment(environment_counts, 5)
        genotype_folds = np.asarray([genotype_assignment[value] for value in genotype_ids])
        environment_folds = np.asarray([environment_assignment[value] for value in environment_ids])
        splits = []
        for fold in range(5):
            validation = (genotype_folds == fold) & (environment_folds == fold)
            train = (genotype_folds != fold) & (environment_folds != fold)
            purge = ~(validation | train)
            splits.append(FoldSplit(fold, all_indices[train], all_indices[validation], all_indices[purge]))
        return splits
    raise ValueError(f"Unknown validation protocol: {protocol}")


def assert_no_leakage(rows: list[dict], split: FoldSplit, protocol: str) -> None:
    train = set(split.train_idx.tolist())
    validation = set(split.validation_idx.tolist())
    if not train.isdisjoint(validation):
        raise RuntimeError(f"row leakage in {protocol} fold {split.fold}")
    if protocol in {"cv_g", "cv_gcluster", "cv_ge"}:
        train_groups = {rows[i]["genotype_id"] for i in split.train_idx}
        validation_groups = {rows[i]["genotype_id"] for i in split.validation_idx}
        if not train_groups.isdisjoint(validation_groups):
            raise RuntimeError(f"genotype leakage in {protocol} fold {split.fold}")
    if protocol in {"cv_e", "cv_ge"}:
        train_environments = {rows[i]["environment_id"] for i in split.train_idx}
        validation_environments = {rows[i]["environment_id"] for i in split.validation_idx}
        if not train_environments.isdisjoint(validation_environments):
            raise RuntimeError(f"environment leakage in {protocol} fold {split.fold}")


def write_manifest(path: Path, rows: list[dict], protocol: str, splits: list[FoldSplit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row_id", "genotype_id", "environment_id", "protocol", "fold", "role", "purge_reason"])
        for split in splits:
            train = set(split.train_idx.tolist())
            validation = set(split.validation_idx.tolist())
            purge = set(split.purge_idx.tolist())
            for index, row in enumerate(rows):
                if index in train:
                    role, reason = "train", ""
                elif index in validation:
                    role, reason = "validation", ""
                elif index in purge:
                    role, reason = "purge", "genotype_or_environment_holdout"
                else:
                    raise AssertionError((protocol, split.fold, index))
                writer.writerow([row["row_id"], row["genotype_id"], row["environment_id"], protocol, split.fold, role, reason])
