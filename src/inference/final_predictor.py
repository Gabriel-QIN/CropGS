"""Portable, development-only final genomic-selection models.

Artifacts contain marker effects and environment-kernel projections rather
than the training genotype matrix.  They are therefore CPU-portable and small
enough for batch or service inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.data.io import canonical_environment_map, load_genotype_dosage, read_phenotypes
from src.models.environment_calibration import EnvironmentLocationScale
from src.models.kernel_gs import MatrixFreeReactionNorm
from src.preprocessing.deep_features import read_environment_features
from src.preprocessing.genomic import FoldEnvironmentTransformer, FullMarkerTransformer
from src.validation.leakage import assert_registry_locked
from src.validation.registry import read_registry


ARTIFACT_SCHEMA_VERSION = 1


def _unique_mapping(values: Sequence[str]) -> tuple[list[str], np.ndarray]:
    unique = sorted(set(values))
    lookup = {value: index for index, value in enumerate(unique)}
    return unique, np.asarray([lookup[value] for value in values], dtype=np.int64)


@dataclass
class PortableEnvironmentKernel:
    kind: str
    bandwidth_multiplier: float = 1.0
    linear_scale: float = 1.0
    train_pc: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None
    bandwidth2: float = 1.0

    @classmethod
    def fit(cls, train_pc: np.ndarray, kind: str, bandwidth_multiplier: float = 1.0) -> tuple["PortableEnvironmentKernel", np.ndarray]:
        train = np.asarray(train_pc, dtype=np.float64)
        if kind == "linear":
            scale = float(np.sqrt(max(1, train.shape[1])))
            obj = cls(kind=kind, bandwidth_multiplier=bandwidth_multiplier, linear_scale=scale)
            return obj, (train / scale).astype(np.float32)
        distances = np.sum((train[:, None, :] - train[None, :, :]) ** 2, axis=2)
        positive = distances[distances > 1e-12]
        bandwidth2 = (float(np.median(positive)) if len(positive) else 1.0) * bandwidth_multiplier**2
        kernel = np.exp(-distances / max(2.0 * bandwidth2, 1e-12))
        values, vectors = np.linalg.eigh((kernel + kernel.T) / 2.0)
        keep = values > 1e-8
        obj = cls(
            kind=kind,
            bandwidth_multiplier=bandwidth_multiplier,
            train_pc=train.astype(np.float32),
            eigenvectors=vectors[:, keep].astype(np.float32),
            eigenvalues=values[keep].astype(np.float32),
            bandwidth2=bandwidth2,
        )
        return obj, (vectors[:, keep] * np.sqrt(values[keep])).astype(np.float32)

    def transform(self, test_pc: np.ndarray) -> np.ndarray:
        test = np.asarray(test_pc, dtype=np.float64)
        if self.kind == "linear":
            return (test / self.linear_scale).astype(np.float32)
        if self.train_pc is None or self.eigenvectors is None or self.eigenvalues is None:
            raise RuntimeError("RBF environment kernel is incomplete")
        distance = np.sum((test[:, None, :] - self.train_pc[None, :, :]) ** 2, axis=2)
        cross = np.exp(-distance / max(2.0 * self.bandwidth2, 1e-12))
        return (cross @ (self.eigenvectors / np.sqrt(self.eigenvalues))).astype(np.float32)


@dataclass
class PortableKernelPredictor:
    mean: float
    marker_effect: np.ndarray
    environment_effect: np.ndarray | None = None
    interaction_effect: np.ndarray | None = None
    variance_components: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_fitted(cls, model: MatrixFreeReactionNorm) -> "PortableKernelPredictor":
        if model.fit_ is None or model.device != "cpu":
            raise ValueError("Portable export requires a fitted CPU model")
        x = np.asarray(model._state["x"], dtype=np.float32)
        obs = np.asarray(model._state["obs_g"], dtype=np.int64)
        env = np.asarray(model._state["e"], dtype=np.float32)
        alpha = np.asarray(model.fit_.alpha, dtype=np.float32)
        denominator = float(model._state["denominator"])
        aggregate = np.bincount(obs, weights=alpha, minlength=x.shape[0]).astype(np.float32)
        marker = model.fit_.variance_components["G"] * (x.T @ aggregate) / denominator / float(model._state["ng"])
        environment = None
        if model.model != "gblup":
            environment = model.fit_.variance_components["E"] * (env.T @ alpha) / float(model._state["ne"])
        interaction = None
        if model.model in {"reaction_norm_gblup", "rkhs"}:
            interaction = np.empty((env.shape[1], x.shape[1]), dtype=np.float32)
            for column in range(env.shape[1]):
                weighted = env[:, column] * alpha
                aggregate = np.bincount(obs, weights=weighted, minlength=x.shape[0]).astype(np.float32)
                interaction[column] = model.fit_.variance_components["GxE"] * (x.T @ aggregate) / denominator / float(model._state["nge"])
        return cls(
            mean=float(model.fit_.mean),
            marker_effect=np.asarray(marker, dtype=np.float32),
            environment_effect=None if environment is None else np.asarray(environment, dtype=np.float32),
            interaction_effect=interaction,
            variance_components=dict(model.fit_.variance_components),
        )

    def predict(self, genotype: np.ndarray, observation_genotype: np.ndarray, environment_factor: np.ndarray) -> np.ndarray:
        x = np.asarray(genotype, dtype=np.float32)
        obs = np.asarray(observation_genotype, dtype=np.int64)
        env = np.asarray(environment_factor, dtype=np.float32)
        result = self.mean + (x @ self.marker_effect)[obs]
        if self.environment_effect is not None:
            result = result + env @ self.environment_effect
        if self.interaction_effect is not None:
            per_genotype = x @ self.interaction_effect.T
            result = result + np.sum(env * per_genotype[obs], axis=1)
        return np.asarray(result, dtype=np.float32)


@dataclass
class KernelMember:
    predictor: PortableKernelPredictor
    environment_kernel: PortableEnvironmentKernel
    weight: float = 1.0


@dataclass
class FinalModelBundle:
    crop: str
    trait: str
    model_name: str
    selected_marker_ids: list[str]
    allele_frequency: np.ndarray
    environment_transformer: FoldEnvironmentTransformer
    environment_feature_names: list[str]
    training_environment_ids: list[str]
    training_environment_pc: np.ndarray
    members: list[KernelMember]
    metadata: dict[str, Any]
    fallback: KernelMember | None = None
    calibrator: EnvironmentLocationScale | None = None
    gate_threshold: float | None = None
    gate_strength: float = 0.0

    def _transform_aligned_genotypes(self, dosage: np.ndarray, marker_ids: Sequence[str]) -> tuple[np.ndarray, float]:
        marker_lookup = {str(marker): index for index, marker in enumerate(marker_ids)}
        n_samples = int(dosage.shape[0])
        transformed = np.zeros((n_samples, len(self.selected_marker_ids)), dtype=np.float32)
        missing = 0
        for output_index, marker in enumerate(self.selected_marker_ids):
            input_index = marker_lookup.get(marker)
            if input_index is None:
                missing += n_samples
                continue
            values = np.asarray(dosage[:, input_index])
            observed = values >= 0
            missing += int(np.sum(~observed))
            transformed[observed, output_index] = values[observed].astype(np.float32) - 2.0 * self.allele_frequency[output_index]
        fraction = missing / max(1, n_samples * len(self.selected_marker_ids))
        return transformed, float(fraction)

    def predict_arrays(
        self,
        records: Sequence[dict[str, Any]],
        sample_ids: Sequence[str],
        dosage: np.ndarray,
        marker_ids: Sequence[str],
        environment_features: dict[str, np.ndarray],
    ) -> list[dict[str, Any]]:
        if not records:
            return []
        sample_lookup = {str(value): index for index, value in enumerate(sample_ids)}
        genotype_ids = [str(record["genotype_id"]) for record in records]
        environment_ids = [str(record["environment_id"]) for record in records]
        absent = sorted({value for value in genotype_ids if value not in sample_lookup})
        if absent:
            raise KeyError(f"Genotypes absent from inference matrix: {absent[:10]}")
        absent_env = sorted({value for value in environment_ids if value not in environment_features})
        if absent_env:
            raise KeyError(f"Environments absent from Environment.csv: {absent_env[:10]}")
        unique_g, obs_g = _unique_mapping(genotype_ids)
        incoming_indices = np.asarray([sample_lookup[value] for value in unique_g], dtype=np.int64)
        x, missing_fraction = self._transform_aligned_genotypes(np.asarray(dosage[incoming_indices]), marker_ids)
        unique_e, obs_e = _unique_mapping(environment_ids)
        test_pc = self.environment_transformer.transform(environment_features, unique_e)
        prediction = np.zeros(len(records), dtype=np.float32)
        for member in self.members:
            factor = member.environment_kernel.transform(test_pc)[obs_e]
            prediction += member.weight * member.predictor.predict(x, obs_g, factor)
        gate_weight = np.zeros(len(records), dtype=np.float32)
        nearest_distance = np.zeros(len(records), dtype=np.float32)
        if self.fallback is not None:
            if self.calibrator is None or self.gate_threshold is None:
                raise RuntimeError("Maize OOD fallback is incomplete")
            fallback_factor = self.fallback.environment_kernel.transform(test_pc)[obs_e]
            standardized = self.fallback.predictor.predict(x, obs_g, fallback_factor)
            fallback_prediction = self.calibrator.restore(standardized, environment_ids, test_pc[obs_e])
            training_set = set(self.training_environment_ids)
            for index, environment in enumerate(unique_e):
                if environment in training_set:
                    distance = weight = 0.0
                else:
                    distance = float(np.min(np.linalg.norm(self.training_environment_pc - test_pc[index], axis=1)))
                    weight = float(np.clip(self.gate_strength * (distance - self.gate_threshold) / self.gate_threshold, 0.0, 1.0))
                nearest_distance[obs_e == index] = distance
                gate_weight[obs_e == index] = weight
            prediction = (1.0 - gate_weight) * prediction + gate_weight * fallback_prediction
        output = []
        for index, (record, value) in enumerate(zip(records, prediction)):
            output.append(
                {
                    **record,
                    "crop": self.crop,
                    "trait": self.trait,
                    "prediction": float(value),
                    "gate_weight": float(gate_weight[index]),
                    "nearest_environment_distance": float(nearest_distance[index]),
                    "selected_marker_missing_fraction": missing_fraction,
                }
            )
        return output

    def predict_dataset(
        self,
        data_root: Path,
        records: Sequence[dict[str, Any]],
        mapping_path: Path | None = None,
        cache_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        mapping = canonical_environment_map(mapping_path, self.crop)
        normalized = [{**record, "environment_id": mapping.get(str(record["environment_id"]), str(record["environment_id"]))} for record in records]
        sample_ids, dosage, marker_ids = load_genotype_dosage(data_root, self.crop, cache_dir)
        environment, names, _ = read_environment_features(data_root, self.crop, mapping_path, feature_set="dynamic")
        if names != self.environment_feature_names:
            raise ValueError("Environment feature schema differs from the training artifact")
        return self.predict_arrays(normalized, sample_ids, dosage, marker_ids, environment)


def _fit_kernel(
    model_name: str,
    x: np.ndarray,
    obs_g: np.ndarray,
    factor: np.ndarray,
    y: np.ndarray,
    denominator: float,
    residual_ratio: float = 0.2,
) -> PortableKernelPredictor:
    model = MatrixFreeReactionNorm(model_name, residual_ratio=residual_ratio, max_iter=400, tolerance=1e-5, device="cpu")
    model.fit(x, obs_g, factor, y, denominator)
    return PortableKernelPredictor.from_fitted(model)


def train_final_bundle(
    data_root: Path,
    splits_dir: Path,
    crop: str,
    trait: str,
    mapping_path: Path | None = None,
    cache_dir: Path | None = None,
) -> FinalModelBundle:
    """Fit one selected final target on development observations only."""
    registry_path = splits_dir / f"{crop.lower()}.json"
    assert_registry_locked(registry_path)
    registry = read_registry(registry_path)
    if not registry.get("diagnostic_holdout_excluded", False):
        raise RuntimeError("Registry does not certify Diagnostic Holdout exclusion")
    all_rows, traits, _ = read_phenotypes(data_root, crop, mapping_path)
    if trait not in traits:
        raise KeyError(f"Unknown trait {trait}; available: {traits}")
    allowed = {int(item["row_id"]) for item in registry["observations"] if item.get("split_type") == "development"}
    rows = [row for row in all_rows if int(row["row_id"]) in allowed and np.isfinite(row[trait])]
    sample_ids, dosage, marker_ids = load_genotype_dosage(data_root, crop, cache_dir)
    genotype_lookup = {value: index for index, value in enumerate(sample_ids)}
    unique_g, obs_g = _unique_mapping([row["genotype_id"] for row in rows])
    if any(value not in genotype_lookup for value in unique_g):
        raise KeyError("Training phenotype contains genotype absent from Genotypes.csv")
    train_gidx = np.asarray([genotype_lookup[value] for value in unique_g])
    max_missing = 0.40 if crop == "Wheat" else 0.25
    genomic = FullMarkerTransformer(min_maf=0.01, max_missing=max_missing).fit(dosage, train_gidx, sample_ids)
    x = genomic.transform(dosage, train_gidx)
    selected_marker_ids = [marker_ids[index] for index in genomic.marker_indices]
    environment, environment_names, _ = read_environment_features(data_root, crop, mapping_path, feature_set="dynamic")
    components = {"Rice": 3, "Maize": 5, "Wheat": 5, "Soybean": 6}[crop]
    environment_ids = [row["environment_id"] for row in rows]
    unique_e, obs_e = _unique_mapping(environment_ids)
    env_transform = FoldEnvironmentTransformer(components).fit(environment, unique_e)
    train_pc = env_transform.transform(environment, unique_e)
    y = np.asarray([row[trait] for row in rows], dtype=np.float32)
    members: list[KernelMember] = []
    fallback = None
    calibrator = None
    gate_threshold = None
    gate_strength = 0.0
    if crop == "Maize":
        factorizer, unique_factor = PortableEnvironmentKernel.fit(train_pc, "linear")
        factor = unique_factor[obs_e]
        members.append(KernelMember(_fit_kernel("reaction_norm_gblup", x, obs_g, factor, y, genomic.denominator), factorizer, 1.0))
        calibrator = EnvironmentLocationScale().fit(environment_ids, y, unique_e, train_pc)
        normalized = calibrator.normalize(environment_ids, y)
        fallback_predictor = _fit_kernel("gblup", x, obs_g, factor, normalized, genomic.denominator)
        fallback = KernelMember(fallback_predictor, factorizer, 1.0)
        pair = np.sqrt(np.sum((train_pc[:, None, :] - train_pc[None, :, :]) ** 2, axis=2))
        pair[pair <= 1e-12] = np.inf
        gate_threshold = max(float(np.quantile(np.min(pair, axis=1), 0.90)), 1e-6)
        gate_strength = 4.0
        model_name = "maize_weather_ood_gate"
    elif crop == "Rice":
        for bandwidth, weight in ((0.5, 0.5), (1.0, 0.5)):
            factorizer, unique_factor = PortableEnvironmentKernel.fit(train_pc, "rbf", bandwidth)
            predictor = _fit_kernel("rkhs", x, obs_g, unique_factor[obs_e], y, genomic.denominator)
            members.append(KernelMember(predictor, factorizer, weight))
        model_name = "equal_multiscale_rkhs"
    else:
        factorizer, unique_factor = PortableEnvironmentKernel.fit(train_pc, "rbf", 1.0)
        predictor = _fit_kernel("rkhs", x, obs_g, unique_factor[obs_e], y, genomic.denominator)
        members.append(KernelMember(predictor, factorizer, 1.0))
        model_name = "rkhs"
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "selection_basis": "development-only strict CV; Diagnostic Holdout locked",
        "registry_hash": registry["registry_hash"],
        "n_training_observations": len(rows),
        "n_training_genotypes": len(unique_g),
        "n_training_environments": len(unique_e),
        "n_selected_markers": len(selected_marker_ids),
        "selected_marker_sha256": hashlib.sha256("\n".join(selected_marker_ids).encode()).hexdigest(),
        "genotype_coding_contract": "0/1/2 dosage must use the same allele orientation as training; missing=-1",
    }
    return FinalModelBundle(
        crop=crop,
        trait=trait,
        model_name=model_name,
        selected_marker_ids=selected_marker_ids,
        allele_frequency=np.asarray(genomic.allele_frequency, dtype=np.float32),
        environment_transformer=env_transform,
        environment_feature_names=environment_names,
        training_environment_ids=unique_e,
        training_environment_pc=train_pc,
        members=members,
        fallback=fallback,
        calibrator=calibrator,
        gate_threshold=gate_threshold,
        gate_strength=gate_strength,
        metadata=metadata,
    )


def save_bundle(bundle: FinalModelBundle, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    sha256 = hashlib.sha256(payload).hexdigest()
    return {"crop": bundle.crop, "trait": bundle.trait, "model": bundle.model_name, "path": path.name, "sha256": sha256, "size_bytes": len(payload), **bundle.metadata}


def load_bundle(path: Path) -> FinalModelBundle:
    # Pickle artifacts are trusted local build outputs; never load untrusted files.
    bundle = pickle.loads(path.read_bytes())
    if not isinstance(bundle, FinalModelBundle):
        raise TypeError(f"Not a FinalModelBundle: {path}")
    if bundle.metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported artifact schema: {bundle.metadata.get('schema_version')}")
    return bundle


def read_request_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"genotype_id", "environment_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Request CSV must contain genotype_id and environment_id")
    return rows


def write_prediction_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["genotype_id", "environment_id", "crop", "trait", "prediction"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
