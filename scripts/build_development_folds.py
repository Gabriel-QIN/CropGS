#!/usr/bin/env python3
"""Freeze all development-only CV manifests after independent-test removal."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.io import read_phenotypes
from src.validation.splits import assert_no_leakage, build_splits, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--audit-json", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--minimum-validation-rows", type=int, default=100)
    args = parser.parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    mapping = args.data_root.parent / "data" / "interim" / "environment_id_map.csv"
    summary: list[dict] = []
    for crop in ["Maize", "Rice", "Wheat", "Soybean"]:
        rows, _, _ = read_phenotypes(args.data_root, crop, mapping)
        with (args.results_root / "folds" / crop.lower() / "independent_test.csv").open("r", newline="", encoding="utf-8") as fh:
            development_ids = {int(row["row_id"]) for row in csv.DictReader(fh) if row["role"] == "development"}
        development = [row for row in rows if int(row["row_id"]) in development_ids]
        genotype = audit[crop]["genotypes"]
        clusters = {sample: int(label) for sample, label in zip(genotype["sample_ids"], genotype["structure"]["cluster_labels"])}
        for protocol in ["cv_g", "cv_gcluster", "cv_e", "cv_ge"]:
            splits = build_splits(development, protocol, seed=args.seed, cluster_labels=clusters if protocol == "cv_gcluster" else None)
            for split in splits:
                assert_no_leakage(development, split, protocol)
                summary.append(
                    {
                        "crop": crop,
                        "protocol": protocol,
                        "fold": split.fold,
                        "n_train": len(split.train_idx),
                        "n_validation": len(split.validation_idx),
                        "n_purge": len(split.purge_idx),
                        "status": "eligible" if len(split.validation_idx) >= args.minimum_validation_rows else "skipped_insufficient_n",
                    }
                )
            write_manifest(args.results_root / "folds" / crop.lower() / f"development_{protocol}.csv", development, protocol, splits)
    path = args.results_root / "folds" / "development_fold_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {len(summary)} fold diagnostics to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
