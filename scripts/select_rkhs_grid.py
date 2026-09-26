#!/usr/bin/env python3
"""Cross-select RKHS hyperparameters without scoring a fold on itself."""

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
from src.evaluation.metrics import metric_dict

CORE = ["cv_g", "cv_gcluster", "cv_e", "cv_ge"]
OOD_SOURCE = {"ood_g_": "cv_gcluster", "ood_e_": "cv_e", "ood_ge_": "cv_ge"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/optimization/rkhs_nested_tuned"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for directory in args.inputs:
        path = directory / "oof_predictions.csv"
        if not path.exists(): continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["model"] == "rkhs": row["model"] = "rkhs_rr02_bw1"
                if row["model"].startswith("rkhs_rr"): rows.append(row)
    grouped = defaultdict(list)
    for row in rows: grouped[(row["crop"], row["trait"], row["protocol"], row["model"], row["fold"])].append(row)
    output, selections, fold_metrics = [], [], []
    full_best = {}
    crops = sorted({(r["crop"], r["trait"]) for r in rows})
    for crop, trait in crops:
        configs = sorted({r["model"] for r in rows if r["crop"] == crop and r["trait"] == trait})
        for protocol in CORE:
            eligible = [config for config in configs if any(key[:4] == (crop, trait, protocol, config) for key in grouped)]
            if not eligible: continue
            folds = sorted({key[4] for key in grouped if key[:3] == (crop, trait, protocol)})
            fold_score = {}
            for config in eligible:
                for fold in folds:
                    records = grouped.get((crop, trait, protocol, config, fold), [])
                    if records:
                        fold_score[config, fold] = metric_dict(np.asarray([float(r["true"]) for r in records]), np.asarray([float(r["prediction"]) for r in records]))["competition_score"]
            complete = [config for config in eligible if all((config, fold) in fold_score for fold in folds)]
            if not complete: continue
            full_best[crop, trait, protocol] = max(complete, key=lambda config: np.mean([fold_score[config, fold] for fold in folds]))
            for held in folds:
                training_folds = [fold for fold in folds if fold != held]
                chosen = max(complete, key=lambda config: np.mean([fold_score[config, fold] for fold in training_folds]))
                records = grouped[crop, trait, protocol, chosen, held]
                selections.append({"crop": crop, "trait": trait, "protocol": protocol, "held_fold": held, "chosen_config": chosen})
                y = np.asarray([float(r["true"]) for r in records]); pred = np.asarray([float(r["prediction"]) for r in records])
                metrics = metric_dict(y, pred); metrics.update({"crop": crop, "trait": trait, "protocol": protocol, "fold": held, "model": "rkhs_nested_tuned", "runtime_seconds": 0.0}); fold_metrics.append(metrics)
                for row in records: output.append({**row, "model": "rkhs_nested_tuned"})
    # OOD predictions use a configuration selected on the corresponding core
    # protocol only; no OOD phenotype participates in selection.
    for crop, trait in crops:
        protocols = sorted({r["protocol"] for r in rows if r["crop"] == crop and r["trait"] == trait and r["protocol"].startswith("ood_")})
        for protocol in protocols:
            source = next((value for prefix, value in OOD_SOURCE.items() if protocol.startswith(prefix)), None)
            chosen = full_best.get((crop, trait, source))
            if chosen is None: continue
            records = [r for r in rows if (r["crop"], r["trait"], r["protocol"], r["model"]) == (crop, trait, protocol, chosen)]
            if not records: continue
            y = np.asarray([float(r["true"]) for r in records]); pred = np.asarray([float(r["prediction"]) for r in records])
            metrics = metric_dict(y, pred); metrics.update({"crop": crop, "trait": trait, "protocol": protocol, "fold": "stress", "model": "rkhs_nested_tuned", "runtime_seconds": 0.0}); fold_metrics.append(metrics)
            for row in records: output.append({**row, "model": "rkhs_nested_tuned"})
    fields = ["crop", "trait", "protocol", "fold", "model", "row_id", "genotype_id", "environment_id", "true", "prediction"]
    with (args.output / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{key: row[key] for key in fields} for row in output])
    with (args.output / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields_m = sorted({key for row in fold_metrics for key in row}); writer = csv.DictWriter(handle, fieldnames=fields_m); writer.writeheader(); writer.writerows(fold_metrics)
    with (args.output / "selected_configs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["crop", "trait", "protocol", "held_fold", "chosen_config"]); writer.writeheader(); writer.writerows(selections)
    with (args.output / "full_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["crop", "trait", "protocol", "chosen_config"])
        for key, value in sorted(full_best.items()): writer.writerow([*key, value])
    print(f"Wrote {len(output)} nested-selected predictions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
