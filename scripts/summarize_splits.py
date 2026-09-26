#!/usr/bin/env python3
"""Write a compact summary of the generated validation manifests."""

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
from src.validation.splits import assert_no_leakage, build_splits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--audit-json", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--output", type=Path, default=Path("results/folds/split_summary.csv"))
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    output_rows = []
    mapping = args.data_root.parent / "data" / "interim" / "environment_id_map.csv"
    for crop in ["Maize", "Rice", "Wheat", "Soybean"]:
        rows, _, _ = read_phenotypes(args.data_root, crop, mapping)
        cluster_labels = {
            sample_id: int(label)
            for sample_id, label in zip(audit[crop]["genotypes"]["sample_ids"], audit[crop]["genotypes"]["structure"].get("cluster_labels", []))
        }
        for protocol in ["cv0", "cv_g", "cv_gcluster", "cv_e", "cv_ge"]:
            splits = build_splits(rows, protocol, seed=args.seed, cluster_labels=cluster_labels if protocol == "cv_gcluster" else None)
            for split in splits:
                assert_no_leakage(rows, split, protocol)
                train_rows = [rows[i] for i in split.train_idx]
                validation_rows = [rows[i] for i in split.validation_idx]
                output_rows.append(
                    {
                        "crop": crop,
                        "protocol": protocol,
                        "fold": split.fold,
                        "n_total": len(rows),
                        "n_train": len(split.train_idx),
                        "n_validation": len(split.validation_idx),
                        "n_purge": len(split.purge_idx),
                        "n_train_genotypes": len({row["genotype_id"] for row in train_rows}),
                        "n_validation_genotypes": len({row["genotype_id"] for row in validation_rows}),
                        "n_train_environments": len({row["environment_id"] for row in train_rows}),
                        "n_validation_environments": len({row["environment_id"] for row in validation_rows}),
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} split rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
