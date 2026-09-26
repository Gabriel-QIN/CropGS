#!/usr/bin/env python3
"""Freeze phenotype-blind development/easy/hard independent test manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.io import read_phenotypes
from src.preprocessing.deep_features import read_environment_features
from src.validation.independent_test import assert_independent_design, build_independent_test_design, independent_role


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--audit-json", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--output-root", type=Path, default=Path("results/folds"))
    parser.add_argument("--genotype-tail-fraction", type=float, default=0.10)
    parser.add_argument("--hard-environment-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-test-rows", type=int, default=100)
    args = parser.parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    mapping = args.data_root.parent / "data" / "interim" / "environment_id_map.csv"
    summary: list[dict] = []
    design_metadata: dict[str, object] = {
        "selection_uses_phenotype_values": False,
        "genotype_tail_fraction": args.genotype_tail_fraction,
        "hard_environment_fraction": args.hard_environment_fraction,
        "minimum_test_rows": args.minimum_test_rows,
        "difficulty_definition": {
            "easy_g": "central genotype PCA distance, non-outlier environment",
            "hard_g": "distant genotype PCA distance, non-outlier environment",
            "hard_e": "development genotype, weather-outlier environment",
            "hard_ge": "distant genotype PCA distance, weather-outlier environment",
        },
        "crops": {},
    }
    for crop in ["Maize", "Rice", "Wheat", "Soybean"]:
        rows, _, _ = read_phenotypes(args.data_root, crop, mapping)
        environment_features, _, _ = read_environment_features(args.data_root, crop, mapping, feature_set="dynamic")
        genotype = audit[crop]["genotypes"]
        design = build_independent_test_design(
            genotype["sample_ids"],
            np.asarray(genotype["structure"]["pca_scores"], dtype=float),
            environment_features,
            genotype_fraction_per_tail=args.genotype_tail_fraction,
            hard_environment_fraction=args.hard_environment_fraction,
            observed_pairs=[(row["genotype_id"], row["environment_id"]) for row in rows],
        )
        roles = [independent_role(row["genotype_id"], row["environment_id"], design) for row in rows]
        assert_independent_design(rows, roles)
        path = args.output_root / crop.lower() / "independent_test.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            fields = ["row_id", "genotype_id", "environment_id", "role", "genotype_novelty", "environment_novelty"]
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row, role in zip(rows, roles):
                writer.writerow(
                    {
                        "row_id": row["row_id"],
                        "genotype_id": row["genotype_id"],
                        "environment_id": row["environment_id"],
                        "role": role,
                        "genotype_novelty": design.genotype_novelty.get(row["genotype_id"], ""),
                        "environment_novelty": design.environment_novelty.get(row["environment_id"], ""),
                    }
                )
        manifest_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        design_metadata["crops"][crop] = {
            "hard_environments": sorted(design.hard_environments),
            "n_easy_genotypes": len(design.easy_genotypes),
            "n_hard_genotypes": len(design.hard_genotypes),
            "manifest": str(path),
            "sha256": manifest_hash,
        }
        counts = Counter(roles)
        for role in ["development", "test_easy_g", "test_hard_g", "test_hard_e", "test_hard_ge", "purge_mixed"]:
            selected = [rows[index] for index, value in enumerate(roles) if value == role]
            summary.append(
                {
                    "crop": crop,
                    "role": role,
                    "n_rows": counts.get(role, 0),
                    "n_genotypes": len({row["genotype_id"] for row in selected}),
                    "n_environments": len({row["environment_id"] for row in selected}),
                    "status": "eligible" if role in {"development", "purge_mixed"} or counts.get(role, 0) >= args.minimum_test_rows else "skipped_insufficient_n",
                }
            )
        print(f"{crop}: {dict(counts)}; hard environments={sorted(design.hard_environments)}")
    summary_path = args.output_root / "independent_test_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    metadata_path = args.output_root / "independent_test_design.json"
    metadata_path.write_text(json.dumps(design_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
