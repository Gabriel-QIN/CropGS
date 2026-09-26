"""Read frozen strict split registries and materialize outer folds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.validation.splits import FoldSplit


def read_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def registry_splits(rows: list[dict], registry: dict, protocol: str, minimum_rows: int = 100) -> list[FoldSplit]:
    by_row = {int(item["row_id"]): item for item in registry["observations"]}
    assignments = np.asarray([by_row[int(row["row_id"])]["outer_fold"].get(protocol) for row in rows], dtype=object)
    folds = registry["protocols"][protocol]["folds"]
    result: list[FoldSplit] = []
    all_idx = np.arange(len(rows), dtype=int)
    for fold in folds:
        if protocol.startswith("ood_"):
            validation_mask = assignments == "validation"
            train_mask = assignments == "train"
            purge = assignments == "purge"
        elif protocol == "cv_ge":
            g_fold, e_fold = str(fold).split("|")
            g = np.asarray([str(value).split("|")[0] for value in assignments])
            e = np.asarray([str(value).split("|")[1] for value in assignments])
            validation_mask = (g == g_fold) & (e == e_fold)
            train_mask = (g != g_fold) & (e != e_fold)
            purge = ~(validation_mask | train_mask)
        else:
            validation_mask = np.asarray([str(value) == str(fold) for value in assignments])
            train_mask = ~validation_mask
            purge = np.zeros(len(rows), dtype=bool)
        validation = all_idx[validation_mask]
        if len(validation) < minimum_rows:
            continue
        result.append(FoldSplit(fold, all_idx[train_mask], validation, all_idx[purge]))
    return result
