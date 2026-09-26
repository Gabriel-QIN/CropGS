#!/usr/bin/env python3
"""Cross-fit convex ensemble weights across already out-of-fold predictions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import metric_dict


def read_oof(path: Path) -> dict[int, dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return {int(row["row_id"]): row for row in csv.DictReader(fh)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-a", type=Path, required=True)
    parser.add_argument("--candidate-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=21)
    args = parser.parse_args()
    a, b = read_oof(args.candidate_a), read_oof(args.candidate_b)
    common = sorted(set(a) & set(b))
    if not common or set(a) != set(b):
        raise ValueError("Candidates must contain the same non-empty OOF row IDs")
    folds = sorted({a[index]["fold"] for index in common})
    output: list[dict] = []
    for held_fold in folds:
        meta = [index for index in common if a[index]["fold"] != held_fold]
        held = [index for index in common if a[index]["fold"] == held_fold]
        y_meta = np.asarray([float(a[index]["true"]) for index in meta])
        pred_a_meta = np.asarray([float(a[index]["prediction"]) for index in meta])
        pred_b_meta = np.asarray([float(b[index]["prediction"]) for index in meta])
        candidates = np.linspace(0.0, 1.0, args.grid_size)
        scores = [metric_dict(y_meta, weight * pred_a_meta + (1.0 - weight) * pred_b_meta)["competition_score"] for weight in candidates]
        best_weight = float(candidates[int(np.argmax(scores))])
        for index in held:
            prediction = best_weight * float(a[index]["prediction"]) + (1.0 - best_weight) * float(b[index]["prediction"])
            output.append(
                {
                    "row_id": index,
                    "crop": a[index]["crop"],
                    "trait": a[index]["trait"],
                    "fold": held_fold,
                    "true": float(a[index]["true"]),
                    "prediction": prediction,
                    "weight_a": best_weight,
                    "weight_b": 1.0 - best_weight,
                }
            )
    y = np.asarray([row["true"] for row in output])
    prediction = np.asarray([row["prediction"] for row in output])
    overall = metric_dict(y, prediction)
    fold_scores = []
    for fold in folds:
        selected = [row for row in output if row["fold"] == fold]
        fold_scores.append(metric_dict(np.asarray([row["true"] for row in selected]), np.asarray([row["prediction"] for row in selected]))["competition_score"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(
        {
            **overall,
            "fold_std": float(np.std(fold_scores, ddof=1)),
            "worst_fold": float(np.min(fold_scores)),
            "best_fold": float(np.max(fold_scores)),
            "fold_weights_a": {fold: next(row["weight_a"] for row in output if row["fold"] == fold) for fold in folds},
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
