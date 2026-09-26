"""Small, dependency-light readers for the released crop CSV files."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MISSING_TOKENS = {"", "NA", "N/A", "N", "NN", "-9", ".", "?", "nan", "NaN"}
CROP_ID_COLUMNS = {"Maize": "Hybrid", "Rice": "Sample", "Wheat": "Sample", "Soybean": "Sample"}


def canonical_environment_map(mapping_path: Path | None, crop: str) -> dict[str, str]:
    if mapping_path is None or not mapping_path.exists():
        return {}
    mapping: dict[str, str] = {}
    with mapping_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("crop") == crop:
                canonical = row["canonical_environment_id"]
                # The phenotype and weather releases may use opposite
                # location/year orders (Wheat is the current example).  Both
                # source IDs resolve to the same frozen canonical key so a
                # weather feature lookup cannot silently return all-missing.
                mapping[row["phenotype_environment_id"]] = canonical
                if row.get("weather_environment_id"):
                    mapping[row["weather_environment_id"]] = canonical
    return mapping


def read_phenotypes(data_root: Path, crop: str, mapping_path: Path | None = None) -> tuple[list[dict[str, Any]], list[str], str]:
    path = data_root / crop / crop / "Phenotypes.csv"
    env_map = canonical_environment_map(mapping_path, crop)
    id_column = CROP_ID_COLUMNS[crop]
    rows: list[dict[str, Any]] = []
    traits: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        traits = [name for name in (reader.fieldnames or []) if name.startswith("Trait_")]
        for row_id, row in enumerate(reader):
            genotype_id = (row.get(id_column) or "").strip()
            original_environment = (row.get("Environment") or "").strip()
            environment_id = env_map.get(original_environment, original_environment)
            parsed: dict[str, Any] = {
                "row_id": row_id,
                "genotype_id": genotype_id,
                "environment_id": environment_id,
                "original_environment_id": original_environment,
            }
            for trait in traits:
                value = row.get(trait, "")
                try:
                    number = float(value)
                    parsed[trait] = number if math.isfinite(number) else float("nan")
                except (TypeError, ValueError):
                    parsed[trait] = float("nan")
            rows.append(parsed)
    return rows, traits, id_column


def _parse_token(token: str) -> tuple[str, object]:
    token = token.strip()
    if token in MISSING_TOKENS:
        return "missing", None
    if token in {"0", "1", "2"}:
        return "numeric", int(token)
    try:
        value = float(token)
        if math.isfinite(value) and value in {0.0, 1.0, 2.0}:
            return "numeric", int(value)
    except ValueError:
        pass
    return "categorical", token


def _encode_categorical(token: str, minor_allele: str | None) -> int:
    if minor_allele is None:
        return 0
    if len(token) == 1:
        return int(token == minor_allele)
    return int(token[0] == minor_allele) + int(token[1] == minor_allele)


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_genotype_dosage(
    data_root: Path,
    crop: str,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Load a sample-by-marker int8 dosage matrix.

    Nucleotide rows are converted to minor-allele dosage per marker.  The
    missing code is -1.  The cache is deliberately genotype-only and records
    the input signature, so it is safe to reuse across validation protocols.
    """
    path = data_root / crop / crop / "Genotypes.csv"
    cache_dir = cache_dir or data_root.parent / "data" / "interim" / "genotype_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = crop.lower()
    matrix_path = cache_dir / f"{stem}_dosage.npy"
    meta_path = cache_dir / f"{stem}_dosage.json"
    header: list[str]
    with path.open("r", newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    sample_ids = header[1:]
    n_markers = max(0, sum(1 for _ in path.open("rb")) - 1)
    signature = _file_signature(path)
    if use_cache and matrix_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("signature") == signature and meta.get("sample_ids") == sample_ids and meta.get("n_markers") == n_markers:
                matrix = np.load(matrix_path, mmap_mode="r")
                return sample_ids, matrix, list(meta.get("marker_ids", []))
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    matrix = np.empty((len(sample_ids), n_markers), dtype=np.int8)
    marker_ids: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        row_index = 0
        for row in reader:
            if row_index >= n_markers:
                break
            marker_ids.append(row[0] if row else "")
            values = row[1:]
            if len(values) < len(sample_ids):
                values += [""] * (len(sample_ids) - len(values))
            elif len(values) > len(sample_ids):
                values = values[: len(sample_ids)]
            parsed = [_parse_token(value) for value in values]
            kinds = {kind for kind, _ in parsed if kind != "missing"}
            codes = np.full(len(sample_ids), -1, dtype=np.int8)
            if kinds and kinds <= {"numeric"}:
                for index, (kind, value) in enumerate(parsed):
                    if kind == "numeric":
                        codes[index] = int(value)
            else:
                allele_counts: dict[str, int] = {}
                for kind, value in parsed:
                    if kind != "categorical":
                        continue
                    token = str(value)
                    for allele in token:
                        allele_counts[allele] = allele_counts.get(allele, 0) + 1
                ordered = sorted(allele_counts.items(), key=lambda item: (item[1], item[0]))
                minor = ordered[0][0] if ordered else None
                for index, (kind, value) in enumerate(parsed):
                    if kind == "categorical":
                        codes[index] = _encode_categorical(str(value), minor)
            matrix[:, row_index] = codes
            row_index += 1
    if row_index < n_markers:
        matrix = matrix[:, :row_index]
        marker_ids = marker_ids[:row_index]
    if use_cache:
        np.save(matrix_path, matrix)
        meta_path.write_text(json.dumps({"signature": signature, "sample_ids": sample_ids, "marker_ids": marker_ids, "n_markers": len(marker_ids)}, ensure_ascii=False), encoding="utf-8")
        matrix = np.load(matrix_path, mmap_mode="r")
    return sample_ids, matrix, marker_ids


def select_genomic_markers(matrix: np.ndarray, max_markers: int = 512, min_maf: float = 0.01, max_missing: float = 0.2) -> np.ndarray:
    """Select markers using genotype-only QC, deterministically."""
    observed = matrix >= 0
    count = observed.sum(axis=0)
    dosage_sum = np.where(observed, matrix, 0).sum(axis=0)
    allele_frequency = np.divide(dosage_sum, 2.0 * count, out=np.zeros(matrix.shape[1], dtype=float), where=count > 0)
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    missing = 1.0 - count / max(1, matrix.shape[0])
    polymorphic = np.asarray([len(np.unique(matrix[:, j][matrix[:, j] >= 0])) > 1 for j in range(matrix.shape[1])])
    valid = np.where(polymorphic & (maf >= min_maf) & (missing <= max_missing))[0]
    if len(valid) > max_markers:
        valid = valid[np.linspace(0, len(valid) - 1, max_markers).astype(int)]
    return valid.astype(int)
