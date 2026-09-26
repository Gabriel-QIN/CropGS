#!/usr/bin/env python3
"""Combine crop-specific strict benchmark outputs into the unified leaderboard."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import metric_dict, summarize_by_environment


def read_many(paths: list[Path], filename: str) -> list[dict]:
    result = []
    for path in paths:
        target = path / filename
        if target.exists():
            with target.open("r", newline="", encoding="utf-8") as handle:
                result.extend(csv.DictReader(handle))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="Phase-1 result directories")
    parser.add_argument("--output", type=Path, default=Path("results/model_leaderboard.csv"))
    args = parser.parse_args()
    oof = read_many(args.inputs, "oof_predictions.csv")
    fold = read_many(args.inputs, "fold_metrics.csv")
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in oof:
        grouped[(row["crop"], row["trait"], row["model"], row["protocol"])].append(row)
    regime = {}
    for key, records in grouped.items():
        y = np.asarray([float(r["true"]) for r in records])
        p = np.asarray([float(r["prediction"]) for r in records])
        e = np.asarray([r["environment_id"] for r in records], dtype=object)
        value = metric_dict(y, p); value.update(summarize_by_environment(e, y, p)); regime[key] = value
    keys = sorted({key[:3] for key in grouped})
    rows = []
    label = {"cv_g": "CV-G", "cv_gcluster": "CV-GCluster", "cv_e": "CV-E", "cv_ge": "CV-GE"}
    for crop, trait, model in keys:
        scores = {label[p]: regime[(crop, trait, model, p)]["competition_score"] for p in label if (crop, trait, model, p) in regime}
        ood_scores = {}
        for family, output_name in (("ood_g_", "OOD-G"), ("ood_e_", "OOD-E"), ("ood_ge_", "OOD-GE")):
            values = [value["competition_score"] for (c, t, m, p), value in regime.items() if (c, t, m) == (crop, trait, model) and p.startswith(family)]
            ood_scores[output_name] = float(np.mean(values)) if values else "not_run"
        related = [r for r in fold if (r["crop"], r["trait"], r["model"]) == (crop, trait, model)]
        fold_scores = np.asarray([float(r["competition_score"]) for r in related])
        rmses = np.asarray([float(r["rmse"]) for r in related])
        runtimes = np.asarray([float(r["runtime_seconds"]) for r in related])
        distributional = model.startswith("distributional_")
        base_model = model.removeprefix("distributional_")
        params = "full-marker VanRaden-I; fold-local MAF/missing QC; " + ("weather-PCA RBF E kernel" if base_model == "rkhs" else "linear compact E kernel" if base_model != "gblup" else "G kernel only")
        if distributional:
            params += "; training-only weather ridge location/scale calibration"
        if "multiscale_equal" in model:
            params = "equal-weight RKHS ensemble; environment RBF bandwidths 0.5x and 1.0x median; residual ratio 0.2"
        elif model.startswith("maize_weather_ood_gate"):
            strength = model.removeprefix("maize_weather_ood_gate_s") if "_s" in model else "1"
            params = f"Reaction-Norm GBLUP with phenotype-blind fold-local weather-distance OOD gate (strength {strength}) to calibrated additive GBLUP"
        elif model.startswith("maize_gate_fb_"):
            fallback = model.removeprefix("maize_gate_fb_").rstrip("_")
            params = f"Reaction-Norm GBLUP with phenotype-blind weather OOD gate (strength 4); fallback {fallback}"
        elif model.startswith("maize_gate_pc"):
            params = f"Reaction-Norm GBLUP with phenotype-blind weather OOD gate geometry {model.removeprefix('maize_gate_')}"
        elif model.startswith("wheat_rkhs_lowrank"):
            rank = model.removeprefix("wheat_rkhs_lowrank")
            params = f"RKHS; fold-local truncated-SVD missing-genotype imputation rank {rank}; observed dosages unchanged"
        elif model.startswith("wheat_rkhs_lr64w"):
            weight = int(model.removeprefix("wheat_rkhs_lr64w")) / 100.0
            params = f"RKHS; fold-local rank-64 SVD missing-genotype residual with weight {weight:.2f}; observed dosages unchanged"
        elif model.startswith("wheat_rkhs_maxmiss"):
            ceiling = int(model.removeprefix("wheat_rkhs_maxmiss")) / 100.0
            params = f"RKHS; fold-local allele-mean imputation; marker missingness ceiling {ceiling:.2f}"
        elif model == "rkhs_nested_tuned":
            params = "outer-fold-excluded RKHS bandwidth/residual-ratio selection"
        elif model == "stacking_raw":
            params = "cross-fitted non-negative stacking of four full-marker kernel models"
        rows.append({"crop": crop, "trait": trait, "model": model, "params": params, **{name: scores.get(name, "") for name in label.values()}, **ood_scores, "RobustScore": float(np.mean(list(scores.values()))) if scores else "", "WorstScore": float(np.min(fold_scores)) if len(fold_scores) else "", "SD": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0, "RMSE": float(np.mean(rmses)) if len(rmses) else "", "runtime": float(np.sum(runtimes)) if len(runtimes) else "", "n_folds": len(related), "status": "complete" if len(scores) == 4 else "partial"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["crop", "trait", "model", "params", "CV-G", "CV-GCluster", "CV-E", "CV-GE", "OOD-G", "OOD-E", "OOD-GE", "RobustScore", "WorstScore", "SD", "RMSE", "runtime", "n_folds", "status"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
