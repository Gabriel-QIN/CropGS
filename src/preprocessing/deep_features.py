"""Fold-fitted features for the deep genotype x environment model.

The important contract in this module is that marker selection, dosage
imputation, standardisation, and weather standardisation are all fitted from
the outer training fold only.  Validation environments can therefore be
represented by weather that is available at prediction time without exposing
their phenotype labels or environment IDs as categorical features.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.data.io import MISSING_TOKENS, canonical_environment_map, select_genomic_markers


def _float_or_nan(value: str | None) -> float:
    if value is None or value.strip() in MISSING_TOKENS:
        return float("nan")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def read_environment_features(
    data_root: Path,
    crop: str,
    mapping_path: Path | None = None,
    feature_set: str = "basic",
) -> tuple[dict[str, np.ndarray], list[str], dict[str, int]]:
    """Aggregate each environment's weather time series into fixed features.

    For every weather variable we retain mean, standard deviation, minimum,
    maximum, first/last observed value, and missing fraction.  The last two
    values preserve a small amount of temporal shape without pretending that
    rows from different environments have the same calendar.  ``n_days`` is a
    separate feature.  Missing all-day variables remain NaN and are imputed by
    :class:`FoldFeatureTransformer` using training environments only.
    """

    path = data_root / crop / crop / "Environment.csv"
    environment_map = canonical_environment_map(mapping_path, crop)
    by_environment: dict[str, list[list[float]]] = {}
    variable_names: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        variable_names = [name for name in fieldnames if name.startswith("Variable_")]
        for row in reader:
            raw_environment = (row.get("Environment") or "").strip()
            environment = environment_map.get(raw_environment, raw_environment)
            by_environment.setdefault(environment, []).append([_float_or_nan(row.get(name)) for name in variable_names])

    if feature_set not in {"basic", "dynamic"}:
        raise ValueError(f"Unknown environment feature set: {feature_set}")
    feature_names = ["n_days"]
    for name in variable_names:
        names = [
            f"{name}__mean", f"{name}__std", f"{name}__min", f"{name}__max",
            f"{name}__first", f"{name}__last", f"{name}__missing_fraction",
        ]
        if feature_set == "dynamic":
            names.extend(
                [
                    f"{name}__q10", f"{name}__q25", f"{name}__median", f"{name}__q75", f"{name}__q90",
                    f"{name}__trend", f"{name}__diff_mean", f"{name}__diff_std",
                    f"{name}__early_mean", f"{name}__middle_mean", f"{name}__late_mean",
                ]
            )
        feature_names.extend(names)

    features: dict[str, np.ndarray] = {}
    n_days: dict[str, int] = {}
    for environment, records in by_environment.items():
        values = np.asarray(records, dtype=np.float32)
        output: list[float] = [float(len(values))]
        for column in range(values.shape[1]):
            series = values[:, column]
            observed = series[np.isfinite(series)]
            if len(observed):
                output.extend([float(np.mean(observed)), float(np.std(observed)), float(np.min(observed)), float(np.max(observed)), float(observed[0]), float(observed[-1]), float(1.0 - len(observed) / len(series))])
                if feature_set == "dynamic":
                    quantiles = np.quantile(observed, [0.10, 0.25, 0.50, 0.75, 0.90])
                    observed_positions = np.flatnonzero(np.isfinite(series)).astype(np.float64)
                    if len(observed) > 1 and np.std(observed_positions) > 0:
                        trend = float(np.cov(observed_positions, observed, ddof=0)[0, 1] / np.var(observed_positions))
                        differences = np.diff(observed)
                        diff_mean, diff_std = float(np.mean(differences)), float(np.std(differences))
                    else:
                        trend = diff_mean = diff_std = 0.0
                    thirds = np.array_split(observed, 3)
                    phase_means = [float(np.mean(part)) if len(part) else float(np.mean(observed)) for part in thirds]
                    output.extend([*(float(value) for value in quantiles), trend, diff_mean, diff_std, *phase_means])
            else:
                output.extend([float("nan")] * 6 + [1.0])
                if feature_set == "dynamic":
                    output.extend([float("nan")] * 11)
        features[environment] = np.asarray(output, dtype=np.float32)
        n_days[environment] = int(len(values))
    return features, feature_names, n_days


@dataclass
class FoldFeatureTransformer:
    """Fit genotype and environment transformations on one training fold."""

    max_markers: int = 4096
    min_maf: float = 0.01
    max_missing: float = 0.25
    environment_components: int | None = None
    marker_indices: np.ndarray | None = field(default=None, init=False)
    marker_impute: np.ndarray | None = field(default=None, init=False)
    marker_center: np.ndarray | None = field(default=None, init=False)
    marker_scale: np.ndarray | None = field(default=None, init=False)
    environment_impute: np.ndarray | None = field(default=None, init=False)
    environment_center: np.ndarray | None = field(default=None, init=False)
    environment_scale: np.ndarray | None = field(default=None, init=False)
    environment_feature_names: list[str] = field(default_factory=list, init=False)
    environment_projection: np.ndarray | None = field(default=None, init=False)

    def fit(
        self,
        dosage: np.ndarray,
        training_genotype_indices: np.ndarray,
        environment_features: Mapping[str, np.ndarray],
        training_environment_ids: Sequence[str],
        environment_feature_names: Sequence[str],
    ) -> "FoldFeatureTransformer":
        training_genotype_indices = np.asarray(training_genotype_indices, dtype=int)
        if len(training_genotype_indices) == 0:
            raise ValueError("Cannot fit fold features without training genotypes")
        training_matrix = np.asarray(dosage[training_genotype_indices])
        self.marker_indices = select_genomic_markers(
            training_matrix,
            max_markers=self.max_markers,
            min_maf=self.min_maf,
            max_missing=self.max_missing,
        )
        if len(self.marker_indices) == 0:
            raise ValueError("No polymorphic markers passed fold-level QC")
        selected = training_matrix[:, self.marker_indices].astype(np.float32, copy=True)
        observed = selected >= 0
        counts = observed.sum(axis=0)
        self.marker_impute = np.divide(
            np.where(observed, selected, 0.0).sum(axis=0),
            counts,
            out=np.zeros(len(self.marker_indices), dtype=np.float32),
            where=counts > 0,
        )
        selected[~observed] = np.take(self.marker_impute, np.where(~observed)[1])
        self.marker_center = selected.mean(axis=0).astype(np.float32)
        self.marker_scale = selected.std(axis=0).astype(np.float32)
        self.marker_scale[self.marker_scale < 1e-6] = 1.0

        self.environment_feature_names = list(environment_feature_names)
        train_environment_ids = list(dict.fromkeys(str(value) for value in training_environment_ids))
        raw_environment = np.asarray(
            [environment_features[environment] for environment in train_environment_ids if environment in environment_features],
            dtype=np.float32,
        )
        if raw_environment.ndim != 2 or raw_environment.shape[0] == 0:
            raise ValueError("No environment features available for the training fold")
        # Compute medians column by column so an all-missing weather variable
        # does not emit a warning.  Such a column gets a neutral zero fallback
        # and its missing-fraction feature remains available to the network.
        self.environment_impute = np.zeros(raw_environment.shape[1], dtype=np.float32)
        for column in range(raw_environment.shape[1]):
            observed = raw_environment[:, column][np.isfinite(raw_environment[:, column])]
            if len(observed):
                self.environment_impute[column] = float(np.median(observed))
        filled = np.where(np.isfinite(raw_environment), raw_environment, self.environment_impute)
        self.environment_center = filled.mean(axis=0).astype(np.float32)
        self.environment_scale = filled.std(axis=0).astype(np.float32)
        self.environment_scale[self.environment_scale < 1e-6] = 1.0
        if self.environment_components is not None:
            standardized = (filled - self.environment_center) / self.environment_scale
            max_rank = min(standardized.shape[0] - 1, standardized.shape[1])
            n_components = min(int(self.environment_components), max_rank)
            if n_components > 0:
                _, _, vt = np.linalg.svd(standardized, full_matrices=False)
                self.environment_projection = vt[:n_components].T.astype(np.float32)
        return self

    @property
    def n_markers(self) -> int:
        return int(len(self.marker_indices)) if self.marker_indices is not None else 0

    @property
    def n_environment_features(self) -> int:
        if self.environment_projection is not None:
            return int(self.environment_projection.shape[1])
        return int(len(self.environment_feature_names))

    def transform_genotypes(self, dosage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.marker_indices is None or self.marker_impute is None or self.marker_center is None or self.marker_scale is None:
            raise RuntimeError("FoldFeatureTransformer has not been fitted")
        selected = np.asarray(dosage[:, self.marker_indices]).astype(np.float32, copy=True)
        missing = (selected < 0).astype(np.float32)
        selected[selected < 0] = np.take(self.marker_impute, np.where(selected < 0)[1])
        selected = (selected - self.marker_center) / self.marker_scale
        return selected.astype(np.float32), missing.astype(np.float32)

    def transform_environments(
        self,
        environment_ids: Sequence[str],
        environment_features: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        if self.environment_impute is None or self.environment_center is None or self.environment_scale is None:
            raise RuntimeError("FoldFeatureTransformer has not been fitted")
        raw = np.asarray(
            [environment_features.get(str(environment), np.full(len(self.environment_feature_names), np.nan, dtype=np.float32)) for environment in environment_ids],
            dtype=np.float32,
        )
        raw = np.where(np.isfinite(raw), raw, self.environment_impute)
        standardized = ((raw - self.environment_center) / self.environment_scale).astype(np.float32)
        if self.environment_projection is not None:
            standardized = standardized @ self.environment_projection
        return standardized.astype(np.float32)

    def distribution_summary(
        self,
        dosage: np.ndarray,
        train_genotype_indices: np.ndarray,
        validation_genotype_indices: np.ndarray,
        training_environment_ids: Sequence[str],
        validation_environment_ids: Sequence[str],
        environment_features: Mapping[str, np.ndarray],
    ) -> dict[str, float | int]:
        """Return fold-level shift diagnostics without using phenotype values."""

        marker_indices = self.marker_indices
        assert marker_indices is not None
        train_raw = np.asarray(dosage[np.asarray(train_genotype_indices, dtype=int)][:, marker_indices])
        valid_raw = np.asarray(dosage[np.asarray(validation_genotype_indices, dtype=int)][:, marker_indices])
        train_missing = float(np.mean(train_raw < 0)) if train_raw.size else float("nan")
        valid_missing = float(np.mean(valid_raw < 0)) if valid_raw.size else float("nan")
        train_env = self.transform_environments(training_environment_ids, environment_features)
        valid_env = self.transform_environments(validation_environment_ids, environment_features)
        env_shift = float(np.mean(np.abs(np.mean(valid_env, axis=0) - np.mean(train_env, axis=0)))) if len(valid_env) and len(train_env) else float("nan")
        return {
            "n_selected_markers": int(len(marker_indices)),
            "train_genotype_missing_fraction": train_missing,
            "validation_genotype_missing_fraction": valid_missing,
            "environment_standardized_mean_shift": env_shift,
            "n_train_environments": int(len(set(training_environment_ids))),
            "n_validation_environments": int(len(set(validation_environment_ids))),
        }
