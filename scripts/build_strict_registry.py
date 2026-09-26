#!/usr/bin/env python3
"""Build phenotype-blind, immutable strict split registries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.io import load_genotype_dosage, read_phenotypes, select_genomic_markers
from src.preprocessing.deep_features import read_environment_features
from src.validation.leakage import assert_diagnostic_holdout_excluded, assert_wheat_mapping
from src.validation.splits import stable_bucket


def balanced_groups(weights: dict[str, int], n_folds: int) -> dict[str, int]:
    loads = [0] * n_folds
    result: dict[str, int] = {}
    for key, weight in sorted(weights.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(n_folds), key=lambda value: (loads[value], value))
        result[key] = fold
        loads[fold] += weight
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--output", type=Path, default=Path("splits"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim/genotype_cache"))
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--cluster-markers", type=int, default=8192)
    args = parser.parse_args()
    mapping = args.data_root.parent / "data" / "interim" / "environment_id_map.csv"
    assert_wheat_mapping(mapping)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for crop in ["Maize", "Rice", "Wheat", "Soybean"]:
        all_rows, _, _ = read_phenotypes(args.data_root, crop, mapping)
        lock = Path("results/folds") / crop.lower() / "independent_test.csv"
        with lock.open("r", newline="", encoding="utf-8") as handle:
            development_ids = {int(row["row_id"]) for row in csv.DictReader(handle) if row["role"] == "development"}
        rows = [row for row in all_rows if int(row["row_id"]) in development_ids]
        assert_diagnostic_holdout_excluded((row["row_id"] for row in rows), lock)
        sample_ids, dosage, _ = load_genotype_dosage(args.data_root, crop, args.cache_dir)
        index = {value: i for i, value in enumerate(sample_ids)}
        genotype_ids = sorted({row["genotype_id"] for row in rows})
        g_indices = np.asarray([index[value] for value in genotype_ids], dtype=int)
        marker_idx = select_genomic_markers(dosage[g_indices], max_markers=args.cluster_markers, min_maf=0.01, max_missing=0.40 if crop == "Wheat" else 0.25)
        raw = np.asarray(dosage[g_indices][:, marker_idx], dtype=np.float32)
        observed = raw >= 0
        means = np.divide(np.where(observed, raw, 0).sum(0), observed.sum(0), out=np.zeros(raw.shape[1]), where=observed.sum(0) > 0)
        raw[~observed] = np.take(means, np.where(~observed)[1])
        raw -= raw.mean(0)
        raw /= np.where(raw.std(0) < 1e-6, 1.0, raw.std(0))
        pcs = PCA(n_components=min(32, len(genotype_ids) - 1), svd_solver="randomized", random_state=args.seed).fit_transform(raw)
        row_counts = Counter(row["genotype_id"] for row in rows)
        candidates = []
        labels_by_k = {}
        for k in (8, 12, 16):
            labels = KMeans(k, random_state=args.seed, n_init=20).fit_predict(pcs)
            labels_by_k[k] = labels
            cluster_rows = np.bincount(labels, weights=[row_counts[g] for g in genotype_ids], minlength=k)
            feasible = bool(np.min(cluster_rows) >= 100)
            imbalance = float(np.std(cluster_rows) / np.mean(cluster_rows))
            candidates.append({"k": k, "min_rows": int(np.min(cluster_rows)), "imbalance_cv": imbalance, "feasible": feasible})
        feasible = [item for item in candidates if item["feasible"]]
        chosen = min(feasible or candidates, key=lambda item: (item["imbalance_cv"], -item["min_rows"]))["k"]
        cluster_map = {g: int(label) for g, label in zip(genotype_ids, labels_by_k[chosen])}
        genotype_weights = {g: row_counts[g] for g in genotype_ids}
        cv_g = balanced_groups(genotype_weights, 5)
        cluster_weights = {str(c): sum(row_counts[g] for g in genotype_ids if cluster_map[g] == c) for c in range(chosen)}
        ge_g = balanced_groups(cluster_weights, 5)
        environment_weights = Counter(row["environment_id"] for row in rows)
        ge_e = balanced_groups(dict(environment_weights), 5)
        # Phenotype-blind adversarial distances.  Genotype PCs and standardized
        # weather summaries use no target values and only determine stress-test
        # membership; all model preprocessing is still refitted per fold.
        pc_scale = np.where(pcs.std(0) < 1e-6, 1.0, pcs.std(0))
        genotype_distance = {g: float(np.linalg.norm((value - pcs.mean(0)) / pc_scale)) for g, value in zip(genotype_ids, pcs)}
        env_features, _, _ = read_environment_features(args.data_root, crop, mapping, feature_set="dynamic")
        env_ids = sorted(environment_weights)
        env_matrix = np.asarray([env_features[e] for e in env_ids], dtype=np.float32)
        for j in range(env_matrix.shape[1]):
            observed_column = env_matrix[:, j][np.isfinite(env_matrix[:, j])]
            fill = float(np.median(observed_column)) if len(observed_column) else 0.0
            env_matrix[:, j] = np.where(np.isfinite(env_matrix[:, j]), env_matrix[:, j], fill)
        env_scale = np.where(env_matrix.std(0) < 1e-6, 1.0, env_matrix.std(0))
        env_distance = {e: float(np.linalg.norm(value)) for e, value in zip(env_ids, (env_matrix - env_matrix.mean(0)) / env_scale)}
        held_g, held_e = {}, {}
        for percentage in (10, 15, 20):
            ng = max(1, int(np.ceil(len(genotype_ids) * percentage / 100)))
            ne = max(1, int(np.ceil(len(env_ids) * percentage / 100)))
            held_g[percentage] = set(sorted(genotype_ids, key=lambda g: (-genotype_distance[g], g))[:ng])
            held_e[percentage] = set(sorted(env_ids, key=lambda e: (-env_distance[e], e))[:ne])
        observations = []
        for row in rows:
            gid, eid = row["genotype_id"], row["environment_id"]
            cluster = cluster_map[gid]
            outer = {"cv_g": str(cv_g[gid]), "cv_gcluster": str(cluster), "cv_e": eid, "cv_ge": f"{ge_g[str(cluster)]}|{ge_e[eid]}"}
            inner = {"cv_g": stable_bucket(gid, args.seed + 1, 4), "cv_gcluster": stable_bucket(str(cluster), args.seed + 1, 4), "cv_e": stable_bucket(eid, args.seed + 1, 4), "cv_ge": stable_bucket(f"{gid}|{eid}", args.seed + 1, 4)}
            for percentage in (10, 15, 20):
                g_held, e_held = gid in held_g[percentage], eid in held_e[percentage]
                outer[f"ood_g_{percentage}"] = "validation" if g_held else "train"
                outer[f"ood_e_{percentage}"] = "validation" if e_held else "train"
                outer[f"ood_ge_{percentage}"] = "validation" if g_held and e_held else "purge" if g_held or e_held else "train"
                inner[f"ood_g_{percentage}"] = stable_bucket(gid, args.seed + percentage, 4)
                inner[f"ood_e_{percentage}"] = stable_bucket(eid, args.seed + percentage, 4)
                inner[f"ood_ge_{percentage}"] = stable_bucket(f"{gid}|{eid}", args.seed + percentage, 4)
            observations.append({
                "row_id": int(row["row_id"]), "genotype_id": gid, "environment_id": eid,
                "genetic_cluster": cluster,
                "outer_fold": outer, "inner_fold": inner,
                "split_type": "development",
            })
        protocols = {
            "cv_g": {"folds": [str(i) for i in range(5)]},
            "cv_gcluster": {"folds": [str(i) for i in range(chosen)]},
            "cv_e": {"folds": sorted(environment_weights)},
            "cv_ge": {"folds": [f"{i}|{i}" for i in range(5)], "crossover_purge": True},
        }
        for percentage in (10, 15, 20):
            for regime in ("g", "e", "ge"):
                protocols[f"ood_{regime}_{percentage}"] = {"folds": ["stress"], "adversarial_fraction": percentage / 100}
        payload = {"schema_version": 1, "crop": crop, "seed": args.seed, "diagnostic_holdout_excluded": True, "cluster_selection": {"candidates": candidates, "chosen_k": chosen, "construction": "genotype-only structural split; model preprocessing remains outer-fold-local"}, "protocols": protocols, "observations": observations}
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload["registry_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        path = args.output / f"{crop.lower()}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append({"crop": crop, "n": len(rows), "chosen_k": chosen, "registry_hash": payload["registry_hash"]})
        print(f"[{crop}] wrote {len(rows)} development rows, K={chosen}", flush=True)
    (args.output / "registry_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
