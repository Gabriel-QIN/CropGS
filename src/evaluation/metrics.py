"""Pearson/Spearman-first metrics used by the competition baseline."""

from __future__ import annotations

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def safe_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, bool]:
    if len(y_true) < 2:
        return 0.0, False
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0, False
    value = float(np.corrcoef(a, b)[0, 1])
    return (value if np.isfinite(value) else 0.0), True


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | bool]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pearson, pearson_defined = safe_correlation(y_true, y_pred)
    spearman, spearman_defined = safe_correlation(rankdata(y_true), rankdata(y_pred))
    error = y_true - y_pred
    mse = float(np.mean(error ** 2)) if len(error) else float("nan")
    return {
        "n": int(len(y_true)),
        "pearson": pearson,
        "spearman": spearman,
        "competition_score": (pearson + spearman) / 2.0,
        "mse": mse,
        "rmse": float(np.sqrt(mse)) if np.isfinite(mse) else float("nan"),
        "mae": float(np.mean(np.abs(error))) if len(error) else float("nan"),
        "pearson_defined": pearson_defined,
        "spearman_defined": spearman_defined,
    }


def summarize_by_environment(environment_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    per_environment = []
    for environment in sorted(set(environment_ids.tolist())):
        mask = environment_ids == environment
        per_environment.append((int(np.sum(mask)), metric_dict(y_true[mask], y_pred[mask])))
    if not per_environment:
        return {"n_environments": 0, "environment_macro_score": float("nan"), "environment_weighted_score": float("nan"), "environment_macro_pearson": float("nan"), "environment_macro_spearman": float("nan")}
    weights = np.asarray([item[0] for item in per_environment], dtype=float)
    scores = np.asarray([item[1]["competition_score"] for item in per_environment], dtype=float)
    pearsons = np.asarray([item[1]["pearson"] for item in per_environment], dtype=float)
    spearmans = np.asarray([item[1]["spearman"] for item in per_environment], dtype=float)
    return {
        "n_environments": int(len(per_environment)),
        "environment_macro_score": float(np.mean(scores)),
        "environment_weighted_score": float(np.average(scores, weights=weights)),
        "environment_macro_pearson": float(np.mean(pearsons)),
        "environment_macro_spearman": float(np.mean(spearmans)),
        "environment_median_score": float(np.median(scores)),
        "environment_worst_score": float(np.min(scores)),
    }
