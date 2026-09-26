#!/usr/bin/env python3
"""Leakage-safe non-negative stacking from strict OOF predictions."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import metric_dict, summarize_by_environment


BASE_MODELS = ["gblup", "gblup_environment", "reaction_norm_gblup", "rkhs"]
CORE_PROTOCOLS = ["cv_g", "cv_gcluster", "cv_e", "cv_ge"]


def read_predictions(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        target = path / "oof_predictions.csv"
        if target.exists():
            with target.open("r", newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
    return [row for row in rows if row["model"] in BASE_MODELS]


def matrix(records: list[dict], models: list[str]):
    grouped = defaultdict(dict)
    for row in records:
        grouped[(row["row_id"], row["fold"])][row["model"]] = row
    complete = [value for value in grouped.values() if all(model in value for model in models)]
    complete.sort(key=lambda value: (value[models[0]]["fold"], int(value[models[0]]["row_id"])))
    x = np.asarray([[float(value[model]["prediction"]) for model in models] for value in complete])
    y = np.asarray([float(value[models[0]]["true"]) for value in complete])
    folds = np.asarray([value[models[0]]["fold"] for value in complete], dtype=object)
    return complete, x, y, folds


def fit_weights(x: np.ndarray, y: np.ndarray, folds: np.ndarray, penalty: float = 0.05) -> np.ndarray:
    n_models = x.shape[1]
    equal = np.full(n_models, 1.0 / n_models)
    counts = {fold: int(np.sum(folds == fold)) for fold in set(folds.tolist())}
    sample_weight = np.asarray([1.0 / counts[fold] for fold in folds])
    sample_weight /= np.mean(sample_weight)
    scale = max(float(np.var(y)), 1e-8)
    def objective(w):
        return float(np.mean(sample_weight * (y - x @ w) ** 2) / scale + penalty * np.sum((w - equal) ** 2))
    result = minimize(objective, equal, method="SLSQP", bounds=[(0.0, 1.0)] * n_models, constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    return result.x if result.success else equal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/optimization/crossfit_stacking"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = read_predictions(args.inputs)
    keys = sorted({(r["crop"], r["trait"], r["protocol"]) for r in records})
    output, weight_rows, fold_metrics = [], [], []
    full_weights = {}
    # Core protocols: outer-fold cross-fitting of the meta learner.
    for crop, trait, protocol in [key for key in keys if key[2] in CORE_PROTOCOLS]:
        subset = [r for r in records if (r["crop"], r["trait"], r["protocol"]) == (crop, trait, protocol)]
        complete, x, y, folds = matrix(subset, BASE_MODELS)
        if not len(complete) or len(set(folds.tolist())) < 2:
            continue
        full_weights[(crop, trait, protocol)] = fit_weights(x, y, folds)
        prediction = np.empty(len(y))
        for held in sorted(set(folds.tolist())):
            train_mask, test_mask = folds != held, folds == held
            weights = fit_weights(x[train_mask], y[train_mask], folds[train_mask])
            prediction[test_mask] = x[test_mask] @ weights
            weight_rows.append({"crop": crop, "trait": trait, "protocol": protocol, "held_fold": held, **{f"weight_{m}": w for m, w in zip(BASE_MODELS, weights)}})
            metrics = metric_dict(y[test_mask], prediction[test_mask]); metrics.update({"crop": crop, "trait": trait, "protocol": protocol, "fold": held, "model": "stacking_raw", "runtime_seconds": 0.0}); fold_metrics.append(metrics)
        for value, pred in zip(complete, prediction):
            source = value[BASE_MODELS[0]]
            output.append({**{k: source[k] for k in ["crop", "trait", "protocol", "fold", "row_id", "genotype_id", "environment_id", "true"]}, "model": "stacking_raw", "prediction": float(pred)})
    # Stress protocols: apply weights learned only from the corresponding core
    # regime.  OOD target values never fit ensemble weights.
    source_protocol = {"ood_g_": "cv_gcluster", "ood_e_": "cv_e", "ood_ge_": "cv_ge"}
    for crop, trait, protocol in [key for key in keys if key[2].startswith("ood_")]:
        core = next((value for prefix, value in source_protocol.items() if protocol.startswith(prefix)), None)
        weights = full_weights.get((crop, trait, core))
        if weights is None:
            continue
        subset = [r for r in records if (r["crop"], r["trait"], r["protocol"]) == (crop, trait, protocol)]
        complete, x, y, folds = matrix(subset, BASE_MODELS)
        prediction = x @ weights
        for value, pred in zip(complete, prediction):
            source = value[BASE_MODELS[0]]
            output.append({**{k: source[k] for k in ["crop", "trait", "protocol", "fold", "row_id", "genotype_id", "environment_id", "true"]}, "model": "stacking_raw", "prediction": float(pred)})
        if len(y):
            metrics = metric_dict(y, prediction); metrics.update({"crop": crop, "trait": trait, "protocol": protocol, "fold": "stress", "model": "stacking_raw", "runtime_seconds": 0.0}); fold_metrics.append(metrics)
    with (args.output / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["crop", "trait", "protocol", "fold", "model", "row_id", "genotype_id", "environment_id", "true", "prediction"]); writer.writeheader(); writer.writerows(output)
    with (args.output / "stacking_weights.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["crop", "trait", "protocol", "held_fold"] + [f"weight_{m}" for m in BASE_MODELS]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(weight_rows)
    with (args.output / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in fold_metrics for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(fold_metrics)
    print(f"Wrote {len(output)} predictions and {len(weight_rows)} cross-fitted weight rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
