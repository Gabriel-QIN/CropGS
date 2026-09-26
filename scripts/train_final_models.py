#!/usr/bin/env python3
"""Train and export the selected development-only final model artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.final_predictor import save_bundle, train_final_bundle


TARGETS = {
    "Maize": ["Trait_1"],
    "Rice": ["Trait_1"],
    "Wheat": ["Trait_1"],
    "Soybean": ["Trait_1", "Trait_2"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--splits", type=Path, default=Path("splits"))
    parser.add_argument("--mapping", type=Path, default=Path("data/interim/environment_id_map.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim/genotype_cache"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/final"))
    parser.add_argument("--crop", action="append", choices=sorted(TARGETS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    crops = args.crop or list(TARGETS)
    manifest = {"schema_version": 1, "diagnostic_holdout_used": False, "artifacts": []}
    for crop in crops:
        for trait in TARGETS[crop]:
            print(f"Training {crop}/{trait} ...", flush=True)
            bundle = train_final_bundle(args.data_root, args.splits, crop, trait, args.mapping, args.cache_dir)
            path = args.output / f"{crop.lower()}__{trait.lower()}.pkl"
            entry = save_bundle(bundle, path)
            manifest["artifacts"].append(entry)
            print(f"Saved {path} ({entry['size_bytes'] / 1e6:.1f} MB)", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
