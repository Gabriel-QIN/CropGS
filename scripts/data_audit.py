#!/usr/bin/env python3
"""Audit the released crop datasets without requiring pandas.

The genotype files are SNP-by-sample matrices and can be large.  This script
reads them one marker at a time, keeps a compact int8 dosage matrix in memory
for reproducible structure/QC calculations, and writes Markdown/JSON/SVG
artifacts under the requested output directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


MISSING_TOKENS = {"", "NA", "N/A", "N", "NN", "-9", ".", "?", "nan", "NaN"}
COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if a.size == 0:
        return {"min": None, "p05": None, "median": None, "p95": None, "max": None, "mean": None, "sd": None}
    q = np.quantile(a, [0.05, 0.5, 0.95])
    return {
        "min": float(np.min(a)),
        "p05": float(q[0]),
        "median": float(q[1]),
        "p95": float(q[2]),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
    }


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def count_data_rows(path: Path) -> int:
    with path.open("rb") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def normalized_env_id(value: str) -> str:
    """Normalize common year/location naming permutations for diagnostics only."""
    value = value.strip()
    m = re.match(r"^(\d{4})[-_](.+)$", value)
    if m:
        return f"{m.group(2)}_{m.group(1)}"
    m = re.match(r"^(.+?)[-_](\d{4})$", value)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return value.replace("-", "_")


def audit_environment(path: Path) -> dict:
    header = read_header(path)
    rows = 0
    env_counts: Counter[str] = Counter()
    env_missing: Counter[str] = Counter()
    env_dates: defaultdict[str, list[str]] = defaultdict(list)
    column_missing: Counter[str] = Counter()
    numeric: defaultdict[str, list[float]] = defaultdict(list)
    duplicate_rows = 0
    seen_rows: set[tuple[str, ...]] = set()
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            rows += 1
            t = tuple(row)
            if t in seen_rows:
                duplicate_rows += 1
            seen_rows.add(t)
            if len(row) != len(header):
                continue
            env = row[0]
            env_counts[env] += 1
            if len(row) > 1:
                env_dates[env].append(row[1])
            for col, value in zip(header[2:], row[2:]):
                if value in MISSING_TOKENS:
                    column_missing[col] += 1
                    env_missing[env] += 1
                else:
                    number = safe_float(value)
                    if math.isfinite(number):
                        numeric[col].append(number)
    env_summaries = []
    for env in sorted(env_counts):
        dates = env_dates[env]
        env_summaries.append(
            {
                "environment": env,
                "records": env_counts[env],
                "missing_values": env_missing[env],
                "date_first": dates[0] if dates else None,
                "date_last": dates[-1] if dates else None,
                "unique_dates": len(set(dates)),
            }
        )
    return {
        "file": str(path),
        "columns": header,
        "rows": rows,
        "n_environments": len(env_counts),
        "environment_summaries": env_summaries,
        "column_missing": dict(column_missing),
        "numeric_summary": {k: quantiles(v) for k, v in numeric.items()},
        "duplicate_rows": duplicate_rows,
    }


def audit_phenotypes(path: Path) -> dict:
    header = read_header(path)
    id_col = next((c for c in header if c not in {"Environment"} and not c.startswith("Trait_")), None)
    trait_cols = [c for c in header if c.startswith("Trait_")]
    rows = 0
    env_counts: Counter[str] = Counter()
    sample_counts: Counter[str] = Counter()
    sample_envs: defaultdict[str, set[str]] = defaultdict(set)
    env_samples: defaultdict[str, set[str]] = defaultdict(set)
    key_counts: Counter[tuple[str, str]] = Counter()
    trait_values: defaultdict[str, list[float]] = defaultdict(list)
    trait_env_values: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    trait_missing: Counter[str] = Counter()
    malformed = 0
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            env = (row.get("Environment") or "").strip()
            sample = (row.get(id_col or "") or "").strip()
            if not env or not sample:
                malformed += 1
                continue
            env_counts[env] += 1
            sample_counts[sample] += 1
            sample_envs[sample].add(env)
            env_samples[env].add(sample)
            key_counts[(env, sample)] += 1
            for trait in trait_cols:
                value = row.get(trait, "")
                if value in MISSING_TOKENS or value is None:
                    trait_missing[trait] += 1
                    continue
                number = safe_float(value)
                if not math.isfinite(number):
                    trait_missing[trait] += 1
                    continue
                trait_values[trait].append(number)
                trait_env_values[trait][env].append(number)
    coverage = Counter(len(envs) for envs in sample_envs.values())
    env_table = []
    for env in sorted(env_counts):
        row = {"environment": env, "records": env_counts[env], "unique_samples": len(env_samples[env])}
        for trait in trait_cols:
            row[f"{trait}_summary"] = quantiles(trait_env_values[trait].get(env, []))
        env_table.append(row)
    dup_keys = {f"{e}::{s}": n for (e, s), n in key_counts.items() if n > 1}
    return {
        "file": str(path),
        "columns": header,
        "id_column": id_col,
        "trait_columns": trait_cols,
        "rows": rows,
        "n_environments": len(env_counts),
        "n_samples": len(sample_counts),
        "environment_summaries": env_table,
        "sample_coverage": {str(k): v for k, v in sorted(coverage.items())},
        "coverage_min": min(coverage) if coverage else None,
        "coverage_max": max(coverage) if coverage else None,
        "trait_missing": dict(trait_missing),
        "trait_summary": {k: quantiles(v) for k, v in trait_values.items()},
        "duplicate_environment_sample_keys": dup_keys,
        "n_duplicate_keys": len(dup_keys),
        "malformed_rows": malformed,
        "environment_ids": sorted(env_counts),
        "sample_ids": sorted(sample_counts),
        "sample_env_map": {k: sorted(v) for k, v in sample_envs.items()},
        "trait_environment_summary": {
            trait: {env: quantiles(vals) for env, vals in by_env.items()}
            for trait, by_env in trait_env_values.items()
        },
    }


def parse_dosage_token(token: str) -> tuple[str, object]:
    token = token.strip()
    if token in MISSING_TOKENS:
        return "missing", None
    if token in {"0", "1", "2"}:
        return "numeric", int(token)
    try:
        number = float(token)
        if math.isfinite(number) and number in {0.0, 1.0, 2.0}:
            return "numeric", int(number)
    except ValueError:
        pass
    return "categorical", token


def encode_categorical(token: str, minor_allele: str | None) -> int:
    if minor_allele is None:
        return 0
    chars = list(token)
    if len(chars) == 1:
        return int(chars[0] == minor_allele)
    return int(chars[0] == minor_allele) + int(chars[1] == minor_allele)


def hash_groups(digests: Iterable[str]) -> list[list[int]]:
    groups: defaultdict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(digests):
        groups[digest].append(index)
    return [members for members in groups.values() if len(members) > 1]


def randomized_pca(matrix: np.ndarray, n_components: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    n, p = matrix.shape
    if n == 0 or p == 0:
        return np.empty((n, 0)), np.empty(0)
    k = min(n_components, n - 1, p)
    if k <= 0:
        return np.zeros((n, 0)), np.empty(0)
    l = min(k + 5, n, p)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((p, l), dtype=np.float32)
    y = matrix @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    b = q.T @ matrix
    ub, singular, _ = np.linalg.svd(b, full_matrices=False)
    scores = (q @ ub[:, :k]) * singular[:k]
    variance = (singular[:k] ** 2) / max(1, n - 1)
    total = float(np.sum(matrix.astype(np.float64) ** 2) / max(1, n - 1))
    ratios = variance / total if total > 0 else np.zeros_like(variance)
    return scores.astype(np.float32), ratios.astype(float)


def kmeans(points: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=int)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = np.empty((k, points.shape[1]), dtype=np.float32)
    first = int(rng.integers(n))
    centers[0] = points[first]
    distances = np.full(n, np.inf, dtype=np.float32)
    for i in range(1, k):
        distances = np.minimum(distances, np.sum((points - centers[i - 1]) ** 2, axis=1))
        total = float(distances.sum())
        if total <= 0:
            centers[i] = points[int(rng.integers(n))]
        else:
            centers[i] = points[int(rng.choice(n, p=distances / total))]
    labels = np.zeros(n, dtype=int)
    for _ in range(50):
        new_labels = np.argmin(((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            members = points[labels == i]
            if len(members):
                centers[i] = members.mean(axis=0)
    return labels


def structure_audit(matrix: np.ndarray, marker_maf: np.ndarray, marker_missing: np.ndarray, marker_polymorphic: np.ndarray) -> dict:
    valid = np.where(marker_polymorphic & (marker_maf > 0) & (marker_missing < 1))[0]
    if len(valid) == 0:
        return {"n_markers_used": 0, "pca_scores": [], "explained_variance_ratio": [], "cluster_labels": [], "cluster_counts": {}}
    max_markers = min(3000, len(valid))
    if len(valid) > max_markers:
        indices = np.linspace(0, len(valid) - 1, max_markers).astype(int)
        valid = valid[indices]
    x = matrix[:, valid].astype(np.float32)
    observed = x >= 0
    counts = observed.sum(axis=0)
    sums = np.where(observed, x, 0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float32), where=counts > 0)
    x[~observed] = np.take(means, np.where(~observed)[1])
    x -= means
    std = x.std(axis=0)
    keep = std > 1e-6
    x = x[:, keep]
    if x.shape[1]:
        x /= std[keep]
    scores, ratios = randomized_pca(x, n_components=5)
    k = min(5, max(2, len(matrix) // 300)) if len(matrix) >= 300 else 2
    labels = kmeans(scores[:, : min(5, scores.shape[1])], k) if scores.shape[1] else np.zeros(len(matrix), dtype=int)
    return {
        "n_markers_used": int(x.shape[1]),
        "pca_scores": scores[:, :2].round(6).tolist(),
        "explained_variance_ratio": [float(v) for v in ratios],
        "cluster_labels": labels.tolist(),
        "cluster_counts": {str(int(i)): int(np.sum(labels == i)) for i in sorted(set(labels.tolist()))},
    }


def pairwise_similarity(matrix: np.ndarray, seed: int = 42) -> dict:
    n, p = matrix.shape
    if n < 2 or p == 0:
        return {"n_pairs": 0}
    rng = np.random.default_rng(seed)
    sample_n = min(n, 400)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    marker_idx = np.linspace(0, p - 1, min(1500, p)).astype(int)
    a = matrix[sample_idx[:, None], marker_idx]
    pairs = min(30000, sample_n * (sample_n - 1) // 2)
    i = rng.integers(0, sample_n, size=pairs)
    j = rng.integers(0, sample_n, size=pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    sims = []
    for left, right in zip(i, j):
        x, y = a[left], a[right]
        observed = (x >= 0) & (y >= 0)
        if np.any(observed):
            sims.append(float(1.0 - np.mean(np.abs(x[observed].astype(float) - y[observed].astype(float)) / 2.0)))
    return {"n_pairs": len(sims), "similarity_summary": quantiles(sims)}


def audit_genotypes(path: Path, structure: bool = True) -> dict:
    header = read_header(path)
    marker_col = header[0] if header else None
    sample_ids = header[1:]
    n_samples = len(sample_ids)
    n_markers = count_data_rows(path)
    matrix = np.empty((n_samples, n_markers), dtype=np.int8)
    marker_missing = np.zeros(n_markers, dtype=np.float32)
    marker_maf = np.full(n_markers, np.nan, dtype=np.float32)
    marker_het = np.full(n_markers, np.nan, dtype=np.float32)
    marker_alleles = np.zeros(n_markers, dtype=np.int16)
    marker_polymorphic = np.zeros(n_markers, dtype=bool)
    marker_hashes: list[str] = []
    token_counts: Counter[str] = Counter()
    encoding_counts: Counter[str] = Counter()
    malformed_rows = 0
    bad_lengths: Counter[int] = Counter()
    marker_ids: list[str] = []
    categorical_rows = 0
    numeric_rows = 0

    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        row_index = 0
        for row in reader:
            if row_index >= n_markers:
                break
            marker_ids.append(row[0] if row else "")
            values = row[1:]
            if len(values) != n_samples:
                malformed_rows += 1
                bad_lengths[len(values)] += 1
                if len(values) < n_samples:
                    values = values + [""] * (n_samples - len(values))
                else:
                    values = values[:n_samples]
            parsed: list[tuple[str, object]] = []
            for token in values:
                kind, value = parse_dosage_token(token)
                parsed.append((kind, value))
                if token not in MISSING_TOKENS:
                    token_counts[token] += 1
                encoding_counts[kind] += 1
            kinds = {kind for kind, _ in parsed if kind != "missing"}
            observed_values = {value for kind, value in parsed if kind != "missing"}
            marker_polymorphic[row_index] = len(observed_values) > 1
            numeric = kinds and kinds <= {"numeric"}
            codes = np.full(n_samples, -1, dtype=np.int8)
            observed = np.zeros(n_samples, dtype=bool)
            if numeric:
                numeric_rows += 1
                for j, (kind, value) in enumerate(parsed):
                    if kind == "numeric":
                        codes[j] = int(value)
                        observed[j] = True
                dosage = codes[observed].astype(float)
                marker_het[row_index] = float(np.mean(dosage == 1)) if dosage.size else np.nan
                allele_freq = float(np.mean(dosage) / 2.0) if dosage.size else np.nan
                marker_maf[row_index] = min(allele_freq, 1.0 - allele_freq) if math.isfinite(allele_freq) else np.nan
                marker_alleles[row_index] = 2
            else:
                categorical_rows += 1
                alleles: Counter[str] = Counter()
                for kind, value in parsed:
                    if kind != "categorical":
                        continue
                    token = str(value)
                    for allele in token:
                        alleles[allele] += 1
                marker_alleles[row_index] = len(alleles)
                ordered = sorted(alleles.items(), key=lambda item: (item[1], item[0]))
                minor = ordered[0][0] if ordered else None
                total_alleles = sum(alleles.values())
                minor_count = alleles[minor] if minor is not None else 0
                if total_alleles:
                    marker_maf[row_index] = float(min(minor_count, total_alleles - minor_count) / total_alleles)
                for j, (kind, value) in enumerate(parsed):
                    if kind == "categorical":
                        token = str(value)
                        codes[j] = encode_categorical(token, minor)
                        observed[j] = True
                marker_het[row_index] = float(
                    np.mean([len(str(value)) >= 2 and str(value)[0] != str(value)[1] for kind, value in parsed if kind == "categorical"])
                ) if any(kind == "categorical" for kind, _ in parsed) else np.nan
            matrix[:, row_index] = codes
            marker_missing[row_index] = float(np.mean(~observed))
            marker_hashes.append(hashlib.blake2b(codes.tobytes(), digest_size=16).hexdigest())
            row_index += 1

    if row_index < n_markers:
        matrix = matrix[:, :row_index]
        marker_missing = marker_missing[:row_index]
        marker_maf = marker_maf[:row_index]
        marker_het = marker_het[:row_index]
        marker_alleles = marker_alleles[:row_index]
    sample_missing = np.mean(matrix < 0, axis=1)
    sample_hashes = [hashlib.blake2b(matrix[i].tobytes(), digest_size=16).hexdigest() for i in range(n_samples)]
    duplicate_marker_groups = hash_groups(marker_hashes)
    duplicate_sample_groups = hash_groups(sample_hashes)
    valid_maf = marker_maf[np.isfinite(marker_maf)]
    all_missing_markers = int(np.sum(marker_missing >= 1))
    monomorphic = int(np.sum(~marker_polymorphic & (marker_missing < 1)))
    missing_marker_count = int(np.sum(marker_missing > 0))
    pca_info = structure_audit(matrix, marker_maf, marker_missing, marker_polymorphic) if structure else {}
    similarity = pairwise_similarity(matrix)
    return {
        "file": str(path),
        "marker_column": marker_col,
        "sample_ids": sample_ids,
        "n_samples": n_samples,
        "n_markers": int(matrix.shape[1]),
        "encoding": "numeric dosage" if numeric_rows >= categorical_rows else "nucleotide genotype",
        "encoding_counts": dict(encoding_counts),
        "token_counts": dict(token_counts),
        "marker_id_first": marker_ids[:5],
        "marker_id_last": marker_ids[-5:],
        "marker_id_unique": len(set(marker_ids)),
        "sample_id_unique": len(set(sample_ids)),
        "malformed_rows": malformed_rows,
        "bad_row_lengths": {str(k): v for k, v in bad_lengths.items()},
        "missing_genotype_rate": float(np.mean(matrix < 0)),
        "sample_missing_summary": quantiles(sample_missing.tolist()),
        "marker_missing_summary": quantiles(marker_missing.tolist()),
        "maf_summary": quantiles(valid_maf.tolist()),
        "heterozygosity_summary": quantiles(marker_het[np.isfinite(marker_het)].tolist()),
        "n_monomorphic_markers": monomorphic,
        "n_all_missing_markers": all_missing_markers,
        "n_polymorphic_markers": int(np.sum(marker_polymorphic)),
        "n_markers_with_missing": missing_marker_count,
        "n_allele_summary": quantiles(marker_alleles.tolist()),
        "n_multiallelic_markers": int(np.sum(marker_alleles > 2)),
        "duplicate_marker_groups": duplicate_marker_groups,
        "n_duplicate_marker_groups": len(duplicate_marker_groups),
        "duplicate_sample_groups": duplicate_sample_groups,
        "n_duplicate_sample_groups": len(duplicate_sample_groups),
        "duplicate_marker_groups_ids": [[marker_ids[i] for i in group] for group in duplicate_marker_groups[:20]],
        "duplicate_sample_groups_ids": [[sample_ids[i] for i in group] for group in duplicate_sample_groups[:20]],
        "worst_missing_samples": [
            {"sample_id": sample_ids[int(i)], "missing_rate": float(sample_missing[int(i)])}
            for i in np.argsort(-sample_missing)[:20]
        ],
        "worst_missing_markers": [
            {"marker_id": marker_ids[int(i)], "missing_rate": float(marker_missing[int(i)])}
            for i in np.argsort(-marker_missing)[:20]
        ],
        "structure": pca_info,
        "pairwise_similarity": similarity,
        "_matrix": matrix,
        "_marker_missing": marker_missing,
        "_marker_maf": marker_maf,
        "_marker_het": marker_het,
        "_marker_polymorphic": marker_polymorphic,
        "_sample_missing": sample_missing,
    }


def linkage_audit(env: dict, pheno: dict, geno: dict) -> dict:
    env_ids = set(env["environment_summaries"][i]["environment"] for i in range(len(env["environment_summaries"])))
    pheno_envs = set(pheno["environment_ids"])
    genotype_ids = set(geno["sample_ids"])
    pheno_samples = set(pheno["sample_ids"])
    direct_env = sorted(pheno_envs & env_ids)
    normalized_env_matches = []
    by_normalized: defaultdict[str, list[str]] = defaultdict(list)
    for value in env_ids:
        by_normalized[normalized_env_id(value)].append(value)
    for value in sorted(pheno_envs):
        matches = sorted(by_normalized.get(normalized_env_id(value), []))
        normalized_env_matches.append({"phenotype_environment": value, "environment_file_matches": matches})
    return {
        "environment_direct_match_count": len(direct_env),
        "environment_direct_match_fraction": len(direct_env) / len(pheno_envs) if pheno_envs else None,
        "environment_ids_only_in_phenotypes": sorted(pheno_envs - env_ids),
        "environment_ids_only_in_environment_file": sorted(env_ids - pheno_envs),
        "normalized_environment_matches": normalized_env_matches,
        "sample_direct_match_count": len(pheno_samples & genotype_ids),
        "sample_pheno_not_in_genotype": sorted(pheno_samples - genotype_ids)[:100],
        "sample_genotype_not_in_pheno": sorted(genotype_ids - pheno_samples)[:100],
        "sample_pheno_not_in_genotype_count": len(pheno_samples - genotype_ids),
        "sample_genotype_not_in_pheno_count": len(genotype_ids - pheno_samples),
    }


def svg_escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def svg_text(x: float, y: float, text: object, size: int = 12, anchor: str = "start", fill: str = "#172033", weight: str = "400", rotate: float | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial,sans-serif" font-size="{size}px" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{transform}>{svg_escape(text)}</text>'


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{svg_escape(title)}</title>',
        f'<desc id="desc">{svg_escape(title)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(28, 30, title, 18, weight="600"),
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def chart_axes(lines: list[str], x: float, y: float, w: float, h: float, x_label: str, y_label: str) -> None:
    lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#f8fafc" stroke="#cbd5e1"/>')
    lines.append(svg_text(x + w / 2, y + h + 34, x_label, 11, anchor="middle", fill="#475569"))
    lines.append(svg_text(x - 36, y + h / 2, y_label, 11, anchor="middle", fill="#475569", rotate=-90))


def histogram(lines: list[str], values: Sequence[float], x: float, y: float, w: float, h: float, title: str, x_label: str, bins: int = 20, color: str = COLORS[0]) -> None:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    chart_axes(lines, x, y, w, h, x_label, "count")
    lines.append(svg_text(x, y - 10, title, 13, weight="600"))
    if arr.size == 0:
        lines.append(svg_text(x + w / 2, y + h / 2, "no finite values", 11, anchor="middle"))
        return
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    max_count = max(1, int(counts.max()))
    for i, count in enumerate(counts):
        bx = x + i * w / bins + 1
        bh = (count / max_count) * (h - 2)
        lines.append(f'<rect x="{bx:.1f}" y="{y + h - bh:.1f}" width="{max(1, w / bins - 2):.1f}" height="{bh:.1f}" fill="{color}" opacity="0.82"/>')
    lines.append(svg_text(x, y + h + 16, f"{lo:.3g}", 10, fill="#64748b"))
    lines.append(svg_text(x + w, y + h + 16, f"{hi:.3g}", 10, anchor="end", fill="#64748b"))


def bar_chart(lines: list[str], labels: Sequence[str], values: Sequence[float], x: float, y: float, w: float, h: float, title: str, y_label: str, color: str = COLORS[0]) -> None:
    chart_axes(lines, x, y, w, h, "environment", y_label)
    lines.append(svg_text(x, y - 10, title, 13, weight="600"))
    n = max(1, len(values))
    vmax = max(1.0, max(values) if values else 1.0)
    gap = min(5, w / n * 0.2)
    bw = max(2, w / n - gap)
    for i, (label, value) in enumerate(zip(labels, values)):
        bx = x + i * w / n + gap / 2
        bh = float(value) / vmax * (h - 2)
        lines.append(f'<rect x="{bx:.1f}" y="{y + h - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}" opacity="0.82"/>')
        if n <= 18:
            lines.append(svg_text(bx + bw / 2, y + h + 12, label, 9, anchor="end", fill="#475569", rotate=-45))
    lines.append(svg_text(x, y - 2, f"0", 10, fill="#64748b"))
    lines.append(svg_text(x, y + 10, f"max {vmax:.3g}", 10, fill="#64748b"))


def scatter(lines: list[str], xvals: Sequence[float], yvals: Sequence[float], labels: Sequence[int], x: float, y: float, w: float, h: float, title: str) -> None:
    chart_axes(lines, x, y, w, h, "PC1", "PC2")
    lines.append(svg_text(x, y - 10, title, 13, weight="600"))
    a = np.asarray(xvals, dtype=float)
    b = np.asarray(yvals, dtype=float)
    if not len(a):
        return
    xmin, xmax = float(np.nanmin(a)), float(np.nanmax(a))
    ymin, ymax = float(np.nanmin(b)), float(np.nanmax(b))
    if xmin == xmax:
        xmax = xmin + 1
    if ymin == ymax:
        ymax = ymin + 1
    for xv, yv, label in zip(a, b, labels):
        px = x + (xv - xmin) / (xmax - xmin) * w
        py = y + h - (yv - ymin) / (ymax - ymin) * h
        color = COLORS[int(label) % len(COLORS)]
        lines.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.4" fill="{color}" opacity="0.55"/>')
    lines.append(svg_text(x, y + h + 16, f"{xmin:.3g}", 10, fill="#64748b"))
    lines.append(svg_text(x + w, y + h + 16, f"{xmax:.3g}", 10, anchor="end", fill="#64748b"))


def presence_matrix(lines: list[str], sample_env_map: dict[str, list[str]], environments: Sequence[str], x: float, y: float, w: float, h: float, title: str) -> None:
    """Draw a compact genotype/sample by environment presence matrix.

    Rows are sorted by number of observed environments so sparse and widely
    replicated materials are visible without rendering thousands of labels.
    """
    lines.append(svg_text(x, y - 10, title, 13, weight="600"))
    sample_ids = sorted(sample_env_map, key=lambda s: (len(sample_env_map[s]), s))
    if not sample_ids or not environments:
        lines.append(svg_text(x + w / 2, y + h / 2, "no observations", 11, anchor="middle"))
        return
    # Keep the complete matrix for small datasets and a deterministic sample for
    # very large soybean matrices; coverage counts remain in the Markdown/JSON.
    max_rows = 650
    if len(sample_ids) > max_rows:
        picks = np.linspace(0, len(sample_ids) - 1, max_rows).astype(int)
        sample_ids = [sample_ids[i] for i in picks]
    env_index = {env: i for i, env in enumerate(environments)}
    cell_w = w / len(environments)
    cell_h = h / max(1, len(sample_ids))
    for row_i, sample_id in enumerate(sample_ids):
        observed = set(sample_env_map[sample_id])
        for env, col_i in env_index.items():
            if env not in observed:
                continue
            lines.append(f'<rect x="{x + col_i * cell_w:.2f}" y="{y + row_i * cell_h:.2f}" width="{max(0.5, cell_w - 0.25):.2f}" height="{max(0.5, cell_h - 0.25):.2f}" fill="{COLORS[2]}" opacity="0.78"/>')
    for i, env in enumerate(environments):
        if len(environments) <= 20:
            lines.append(svg_text(x + (i + 0.5) * cell_w, y + h + 13, env, 8, anchor="end", fill="#475569", rotate=-45))
    lines.append(svg_text(x - 6, y + 8, "sparse", 9, anchor="end", fill="#64748b"))
    lines.append(svg_text(x - 6, y + h, "dense", 9, anchor="end", fill="#64748b"))
    lines.append(svg_text(x + w, y + h + 28, f"rows shown: {len(sample_ids)}/{len(sample_env_map)}", 10, anchor="end", fill="#64748b"))


def write_crop_figures(crop: str, env: dict, pheno: dict, geno: dict, out_dir: Path) -> list[str]:
    crop_dir = out_dir / crop.lower()
    crop_dir.mkdir(parents=True, exist_ok=True)
    envs = [row["environment"] for row in pheno["environment_summaries"]]
    counts = [row["records"] for row in pheno["environment_summaries"]]
    coverage_labels = list(pheno["sample_coverage"].keys())
    coverage_values = [pheno["sample_coverage"][k] for k in coverage_labels]
    trait = next(iter(pheno["trait_summary"]), None)
    trait_values = []
    for row in pheno["trait_environment_summary"].get(trait, {}).values() if trait else []:
        if row.get("median") is not None:
            trait_values.append(row["median"])
    outputs = []

    lines = svg_header(1100, 720, f"{crop}: phenotype and environment distribution")
    bar_chart(lines, envs, counts, 70, 80, 980, 220, "Phenotype records by environment", "records", COLORS[0])
    bar_chart(lines, coverage_labels, coverage_values, 70, 390, 450, 220, "Genotype/sample environment coverage", "samples", COLORS[2])
    histogram(lines, trait_values, 600, 390, 450, 220, f"Per-environment {trait or 'trait'} medians", "trait median", color=COLORS[1])
    path = crop_dir / "phenotype_environment_distribution.svg"
    write_svg(path, lines)
    outputs.append(str(path))

    lines = svg_header(1100, 720, f"{crop}: genotype by environment presence")
    environments = [row["environment"] for row in pheno["environment_summaries"]]
    presence_matrix(lines, pheno["sample_env_map"], environments, 90, 70, 920, 560, "Material × environment observation matrix")
    path = crop_dir / "genotype_environment_presence.svg"
    write_svg(path, lines)
    outputs.append(str(path))

    lines = svg_header(1100, 720, f"{crop}: genotype quality distributions")
    histogram(lines, geno["_marker_maf"], 70, 80, 450, 240, "Marker MAF", "MAF", color=COLORS[0])
    histogram(lines, geno["_marker_missing"], 600, 80, 450, 240, "Marker missingness", "missing rate", color=COLORS[1])
    histogram(lines, geno["_sample_missing"], 70, 410, 450, 240, "Sample missingness", "missing rate", color=COLORS[2])
    histogram(lines, geno["_marker_het"], 600, 410, 450, 240, "Marker heterozygosity", "heterozygosity", color=COLORS[3])
    path = crop_dir / "genotype_qc_distribution.svg"
    write_svg(path, lines)
    outputs.append(str(path))

    structure = geno.get("structure", {})
    scores = structure.get("pca_scores", [])
    if scores:
        labels = structure.get("cluster_labels", [0] * len(scores))
        lines = svg_header(760, 620, f"{crop}: genotype PCA and genetic clusters")
        scatter(lines, [s[0] for s in scores], [s[1] for s in scores], labels, 80, 80, 580, 420, "PCA on polymorphic markers (audit subset)")
        lines.append(svg_text(80, 555, f"markers used: {structure.get('n_markers_used', 'NA')}; clusters: {len(structure.get('cluster_counts', {}))}", 11, fill="#475569"))
        ratios = structure.get("explained_variance_ratio", [])
        lines.append(svg_text(80, 575, f"PC1 explained: {fmt(ratios[0] if ratios else None, 3)}; PC2 explained: {fmt(ratios[1] if len(ratios) > 1 else None, 3)}", 11, fill="#475569"))
        path = crop_dir / "genotype_pca.svg"
        write_svg(path, lines)
        outputs.append(str(path))
    return outputs


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(out)


def crop_report(crop: str, env: dict, pheno: dict, geno: dict, linkage: dict, figure_paths: list[str]) -> str:
    env_summary = env["environment_summaries"]
    pheno_summary = pheno["environment_summaries"]
    structure = geno.get("structure", {})
    env_by_normalized: defaultdict[str, list[dict]] = defaultdict(list)
    for item in env_summary:
        env_by_normalized[normalized_env_id(item["environment"])].append(item)
    unknowns = []
    if linkage["environment_direct_match_fraction"] != 1:
        unknowns.append("Environment ID is not a complete direct match between Environment.csv and Phenotypes.csv; resolve mapping before CV-E/CV-GE.")
    if pheno["n_duplicate_keys"]:
        unknowns.append("Duplicate environment × sample keys require replicate handling before model fitting.")
    if geno["n_monomorphic_markers"]:
        unknowns.append("Monomorphic markers should be removed inside each training fold or by a pre-declared genotype-only rule.")
    if geno["n_multiallelic_markers"]:
        unknowns.append("Some markers have more than two observed alleles; verify whether they are coding artifacts or true multiallelic loci.")
    if pheno["coverage_min"] is not None and pheno["coverage_min"] < 2:
        unknowns.append("Some materials occur in only one environment, so genotype holdout and G×E estimates are not equally supported for all materials.")
    lines = [f"# {crop} Data Audit", "", "Generated by `scripts/data_audit.py` from the raw release files.", "", "## Inventory", ""]
    lines.append(markdown_table(
        ["Asset", "Rows", "Columns / entities", "Missing / duplicates"],
        [
            ["Environment", env["rows"], f"{len(env['columns'])} columns; {env['n_environments']} environments", f"{sum(env['column_missing'].values())} missing cells; {env['duplicate_rows']} duplicate rows"],
            ["Phenotypes", pheno["rows"], f"{pheno['n_samples']} samples; {pheno['n_environments']} environments; {len(pheno['trait_columns'])} traits", f"{sum(pheno['trait_missing'].values())} missing trait cells; {pheno['n_duplicate_keys']} duplicate keys"],
            ["Genotypes", geno["n_markers"], f"{geno['n_samples']} samples; {geno['encoding']}", f"{geno['missing_genotype_rate']:.4%} missing; {geno['n_duplicate_marker_groups']} duplicate marker groups; {geno['n_duplicate_sample_groups']} duplicate sample groups"],
        ],
    ))
    lines += ["", "## Environment coverage", "", markdown_table(["Environment", "Environment records", "Phenotype records", "Unique samples", "Direct / normalized ID"], [
        [row["environment"], next((e["records"] for e in env_summary if e["environment"] == row["environment"]), next((e["records"] for e in env_by_normalized.get(normalized_env_id(row["environment"]), [])), "NA")), row["records"], row["unique_samples"], "direct" if row["environment"] in {e["environment"] for e in env_summary} else ("normalized" if env_by_normalized.get(normalized_env_id(row["environment"])) else "no")] for row in pheno_summary
    ]), ""]
    lines += ["## Genotype QC", "", markdown_table(["Metric", "Value"], [
        ["Marker count", geno["n_markers"]], ["Sample count", geno["n_samples"]], ["Encoding", geno["encoding"]], ["Overall missing genotype rate", f"{geno['missing_genotype_rate']:.4%}"], ["Marker MAF summary", fmt(geno["maf_summary"])], ["Marker heterozygosity summary", fmt(geno["heterozygosity_summary"])], ["Polymorphic markers", geno["n_polymorphic_markers"]], ["Monomorphic markers", geno["n_monomorphic_markers"]], ["All-missing markers", geno["n_all_missing_markers"]], ["Multiallelic markers", geno["n_multiallelic_markers"]], ["Marker ID unique", geno["marker_id_unique"]], ["Sample ID unique", geno["sample_id_unique"]], ["Sample missingness summary", fmt(geno["sample_missing_summary"])], ["Marker missingness summary", fmt(geno["marker_missing_summary"])], ["Worst missing samples", "; ".join(f"{x['sample_id']}={x['missing_rate']:.1%}" for x in geno["worst_missing_samples"][:5])], ["Worst missing markers", "; ".join(f"{x['marker_id']}={x['missing_rate']:.1%}" for x in geno["worst_missing_markers"][:5])],
    ]), ""]
    lines += ["## Missingness details", "", markdown_table(["Asset", "Column-wise missing cells"], [["Environment", "; ".join(f"{k}={v}" for k, v in env["column_missing"].items()) or "none"], ["Phenotypes", "; ".join(f"{k}={v}" for k, v in pheno["trait_missing"].items()) or "none"]]), ""]
    lines += ["## Linkage audit", "", markdown_table(["Check", "Result"], [
        ["Phenotype environments with exact Environment.csv match", f"{linkage['environment_direct_match_count']}/{pheno['n_environments']} ({fmt(linkage['environment_direct_match_fraction'], 3)})"], ["Phenotype samples found in genotype header", linkage["sample_direct_match_count"]], ["Phenotype samples absent from genotype", linkage["sample_pheno_not_in_genotype_count"]], ["Genotype samples absent from phenotype", linkage["sample_genotype_not_in_pheno_count"]], ["Duplicate phenotype keys", pheno["n_duplicate_keys"]],
    ]), ""]
    if linkage["environment_ids_only_in_phenotypes"] or linkage["environment_ids_only_in_environment_file"]:
        lines += ["### Environment ID discrepancies", "", f"Phenotype-only IDs: `{', '.join(linkage['environment_ids_only_in_phenotypes']) or 'none'}`", "", f"Environment-file-only IDs: `{', '.join(linkage['environment_ids_only_in_environment_file']) or 'none'}`", ""]
        lines.append("Normalized diagnostic matches:")
        lines.append("")
        lines.append(markdown_table(["Phenotype ID", "Environment file candidates"], [[m["phenotype_environment"], ", ".join(m["environment_file_matches"]) or "none"] for m in linkage["normalized_environment_matches"]]))
        lines.append("")
    lines += ["## Population structure audit", "", markdown_table(["Metric", "Value"], [["Markers used for audit PCA", structure.get("n_markers_used", "NA")], ["Explained variance PC1", fmt((structure.get("explained_variance_ratio") or [None])[0], 4)], ["Explained variance PC2", fmt((structure.get("explained_variance_ratio") or [None, None])[1], 4)], ["Cluster counts", structure.get("cluster_counts", "NA")], ["Sampled pairwise similarity", geno["pairwise_similarity"].get("similarity_summary", "NA")]]), ""]
    lines += ["## Current CV implications", ""]
    if linkage["environment_direct_match_fraction"] != 1:
        lines.append("- **Block CV-E/CV-GE implementation until environment IDs are mapped.** For this crop, direct joins are incomplete; a mapping artifact must be versioned and validated against date/location metadata.")
    else:
        lines.append("- Environment IDs join directly, so CV-E and CV-GE can be generated after checking environment sample sizes.")
    if pheno["coverage_min"] is not None and pheno["coverage_min"] <= 1:
        lines.append("- Use CV-G as the primary protocol, but report coverage-stratified performance; single-environment materials cannot support a fully crossed G×E estimate.")
    if len(pheno_summary) < 10:
        lines.append("- CV-E has few environment groups; use leave-one-environment-out or repeated fixed folds, not a high-k random split.")
    else:
        lines.append("- CV-E can use grouped environment folds, stratified by environment sample count and time/location where metadata permit.")
    lines.append("- Genotype preprocessing and any phenotype-informed marker selection must be fitted inside each training fold; genotype-only ID/QC checks are reported separately.")
    lines += ["", "## Open issues", ""]
    if unknowns:
        lines.extend([f"- {issue}" for issue in unknowns])
    else:
        lines.append("- No blocking issue detected by the automated checks; inspect plots and retain all audit artifacts with the experiment configuration.")
    lines += ["", "## Figures", ""]
    for figure in figure_paths:
        lines.append(f"- `{figure}`")
    return "\n".join(lines) + "\n"


def cross_crop_report(results: dict, figure_path: str) -> str:
    rows = []
    for crop, result in results.items():
        env = result["environment"]
        pheno = result["phenotypes"]
        geno = result["genotypes"]
        link = result["linkage"]
        rows.append([crop, geno["n_samples"], geno["n_markers"], pheno["rows"], pheno["n_environments"], len(pheno["trait_columns"]), f"{geno['missing_genotype_rate']:.3%}", f"{link['environment_direct_match_fraction']:.1%}"])
    lines = ["# Cross-crop Data Audit Summary", "", "The audit is intentionally conservative: CV design is based on observed joins and coverage, not the README alone.", "", markdown_table(["Crop", "Genotype samples", "Markers", "Phenotype rows", "Environments", "Traits", "Genotype missing", "Direct env join"], rows), "", "## Findings", ""]
    for crop, result in results.items():
        pheno = result["phenotypes"]
        geno = result["genotypes"]
        link = result["linkage"]
        notes = []
        if crop == "Soybean":
            notes.append("largest genotype matrix and two traits; environment weather missingness must be imputed within a declared weather preprocessing step")
        if crop == "Wheat" and link["environment_direct_match_fraction"] != 1:
            notes.append("phenotype and weather environment IDs use different year-location order; mapping is required")
        if pheno["coverage_min"] is not None and pheno["coverage_min"] <= 1:
            notes.append("has single-environment materials; report coverage-stratified CV-G performance")
        if geno["n_monomorphic_markers"]:
            notes.append(f"{geno['n_monomorphic_markers']} monomorphic markers detected")
        lines.append(f"- **{crop}:** " + ("; ".join(notes) if notes else "no blocking issue found by automated checks") + ".")
    lines += ["", "## Recommended audit order", "", "1. Resolve and version any environment ID mapping (especially Wheat).", "2. Freeze genotype/phenotype linkage and replicate policy.", "3. Build CV-G and coverage-stratified diagnostics before CV-GCluster/CV-E/CV-GE.", "4. Keep the generated PCA/clustering output as an audit diagnostic; do not use cluster labels as a model feature without fold-safe construction.", "", f"Cross-crop figure: `{figure_path}`", ""]
    return "\n".join(lines)


def write_cross_crop_figure(results: dict, path: Path) -> None:
    crops = list(results)
    metrics = [
        ("Genotype samples", [results[c]["genotypes"]["n_samples"] for c in crops], COLORS[0]),
        ("Markers", [results[c]["genotypes"]["n_markers"] for c in crops], COLORS[1]),
        ("Phenotype rows", [results[c]["phenotypes"]["rows"] for c in crops], COLORS[2]),
        ("Environments", [results[c]["phenotypes"]["n_environments"] for c in crops], COLORS[3]),
    ]
    lines = svg_header(1100, 800, "Cross-crop dataset size and coverage")
    for i, (label, values, color) in enumerate(metrics):
        y = 80 + i * 170
        bar_chart(lines, crops, values, 100, y, 880, 105, label, label, color)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_svg(path, lines)


def strip_private(value: object) -> object:
    if isinstance(value, dict):
        return {k: strip_private(v) for k, v in value.items() if not k.startswith("_")}
    if isinstance(value, list):
        return [strip_private(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--skip-structure", action="store_true", help="skip PCA, clusters, and pairwise similarity")
    parser.add_argument("--crop", action="append", choices=["Maize", "Rice", "Wheat", "Soybean"], help="audit only selected crop(s)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    crops = args.crop or ["Maize", "Rice", "Wheat", "Soybean"]
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for crop in crops:
        crop_root = args.data_root / crop / crop
        env_path = crop_root / "Environment.csv"
        pheno_path = crop_root / "Phenotypes.csv"
        geno_path = crop_root / "Genotypes.csv"
        for required in (env_path, pheno_path, geno_path):
            if not required.exists():
                raise FileNotFoundError(required)
        print(f"[{crop}] auditing environment", flush=True)
        environment = audit_environment(env_path)
        print(f"[{crop}] auditing phenotypes", flush=True)
        phenotypes = audit_phenotypes(pheno_path)
        print(f"[{crop}] auditing genotypes ({geno_path.stat().st_size / 1e6:.1f} MB)", flush=True)
        genotypes = audit_genotypes(geno_path, structure=not args.skip_structure)
        linkage = linkage_audit(environment, phenotypes, genotypes)
        print(f"[{crop}] writing figures", flush=True)
        figures = write_crop_figures(crop, environment, phenotypes, genotypes, args.figures_dir)
        report = crop_report(crop, environment, phenotypes, genotypes, linkage, figures)
        (args.reports_dir / f"{crop.lower()}_data_audit.md").write_text(report, encoding="utf-8")
        results[crop] = {"environment": environment, "phenotypes": phenotypes, "genotypes": genotypes, "linkage": linkage, "figures": figures}
        # Remove the private matrix before the next crop is audited.
        del genotypes["_matrix"]
        del genotypes["_marker_missing"]
        del genotypes["_marker_maf"]
        del genotypes["_marker_het"]
        del genotypes["_marker_polymorphic"]
        del genotypes["_sample_missing"]
    cross_figure = args.figures_dir / "cross_crop_dataset_overview.svg"
    write_cross_crop_figure(results, cross_figure)
    (args.reports_dir / "cross_crop_summary.md").write_text(cross_crop_report(results, str(cross_figure)), encoding="utf-8")
    (args.reports_dir / "data_audit.json").write_text(json.dumps(strip_private(results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed audit for {', '.join(crops)}. Reports: {args.reports_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
