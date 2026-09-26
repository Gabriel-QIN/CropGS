"""Hard leakage guards.  Violations are benchmark failures, not warnings."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable


def require_disjoint(train: Iterable[str], validation: Iterable[str], label: str) -> None:
    overlap = set(train) & set(validation)
    if overlap:
        raise RuntimeError(f"{label} leakage: {len(overlap)} overlapping IDs; example={next(iter(overlap))}")


def assert_protocol_isolation(rows: list[dict], train_idx, validation_idx, protocol: str) -> None:
    train = [rows[int(i)] for i in train_idx]
    valid = [rows[int(i)] for i in validation_idx]
    if protocol in {"cv_g", "cv_gcluster", "cv_ge"} or protocol.startswith("ood_g") or protocol.startswith("ood_ge"):
        require_disjoint((r["genotype_id"] for r in train), (r["genotype_id"] for r in valid), "genotype")
    if protocol in {"cv_e", "cv_ge"} or protocol.startswith("ood_e") or protocol.startswith("ood_ge"):
        require_disjoint((r["environment_id"] for r in train), (r["environment_id"] for r in valid), "environment")


def assert_diagnostic_holdout_excluded(row_ids: Iterable[int], lock_manifest: Path) -> None:
    import csv
    with lock_manifest.open("r", newline="", encoding="utf-8") as handle:
        forbidden = {int(row["row_id"]) for row in csv.DictReader(handle) if row["role"] != "development"}
    overlap = set(int(value) for value in row_ids) & forbidden
    if overlap:
        raise RuntimeError(f"Diagnostic Holdout leakage: {len(overlap)} rows entered development")


def assert_wheat_mapping(mapping_path: Path) -> None:
    import csv
    phenotype: dict[str, str] = {}
    weather: dict[str, str] = {}
    with mapping_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["crop"] != "Wheat":
                continue
            p, w, c = row["phenotype_environment_id"], row["weather_environment_id"], row["canonical_environment_id"]
            if p in phenotype or w in weather:
                raise RuntimeError("Wheat environment mapping is not one-to-one")
            phenotype[p], weather[w] = c, c
    if not phenotype or set(phenotype.values()) != set(weather.values()):
        raise RuntimeError("Wheat phenotype/weather canonical mapping is incomplete")


def assert_registry_locked(registry_path: Path) -> None:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not payload.get("diagnostic_holdout_excluded") or not payload.get("registry_hash"):
        raise RuntimeError(f"Unfrozen strict registry: {registry_path}")
    claimed = payload.pop("registry_hash")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if claimed != actual:
        raise RuntimeError(f"Strict registry hash mismatch: {registry_path}")
