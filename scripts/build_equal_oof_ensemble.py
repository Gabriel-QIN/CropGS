#!/usr/bin/env python3
"""Build a label-free equal-weight ensemble from aligned OOF files."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import metric_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--member", action="append", required=True, help="DIRECTORY:MODEL")
    parser.add_argument("--label", required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(dict)
    member_names = []
    for index, spec in enumerate(args.member):
        directory, model = spec.rsplit(":", 1); member_name = f"m{index}"; member_names.append(member_name)
        with (Path(directory) / "oof_predictions.csv").open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["crop"] == args.crop and row["model"] == model:
                    grouped[(row["crop"], row["trait"], row["protocol"], row["fold"], row["row_id"])][member_name] = row
    output = []
    for values in grouped.values():
        if not all(name in values for name in member_names): continue
        source = values[member_names[0]]
        prediction = float(np.mean([float(values[name]["prediction"]) for name in member_names]))
        output.append({**source, "model": args.label, "prediction": prediction})
    fold_metrics = []
    fold_groups = defaultdict(list)
    for row in output: fold_groups[(row["crop"], row["trait"], row["protocol"], row["fold"])].append(row)
    for (crop, trait, protocol, fold), records in fold_groups.items():
        metrics = metric_dict(np.asarray([float(r["true"]) for r in records]), np.asarray([float(r["prediction"]) for r in records]))
        metrics.update({"crop": crop, "trait": trait, "protocol": protocol, "fold": fold, "model": args.label, "runtime_seconds": 0.0}); fold_metrics.append(metrics)
    fields = ["crop", "trait", "protocol", "fold", "model", "row_id", "genotype_id", "environment_id", "true", "prediction"]
    with (args.output / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows([{key: row[key] for key in fields} for row in output])
    with (args.output / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields_m = sorted({key for row in fold_metrics for key in row}); writer = csv.DictWriter(handle, fieldnames=fields_m); writer.writeheader(); writer.writerows(fold_metrics)
    print(f"Wrote {len(output)} equal-ensemble predictions")
    return 0


if __name__ == "__main__": raise SystemExit(main())
