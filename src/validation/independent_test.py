"""Phenotype-blind independent test construction and difficulty labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class IndependentTestDesign:
    easy_genotypes: frozenset[str]
    hard_genotypes: frozenset[str]
    hard_environments: frozenset[str]
    genotype_novelty: dict[str, float]
    environment_novelty: dict[str, float]


def _standardized_radius(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    center = np.nanmedian(matrix, axis=0)
    scale = np.nanstd(matrix, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    standardized = (np.where(np.isfinite(matrix), matrix, center) - center) / scale
    return np.sqrt(np.mean(standardized**2, axis=1))


def build_independent_test_design(
    genotype_ids: Sequence[str],
    genotype_pca_scores: np.ndarray,
    environment_features: Mapping[str, np.ndarray],
    genotype_fraction_per_tail: float = 0.10,
    hard_environment_fraction: float = 0.20,
    observed_pairs: Sequence[tuple[str, str]] | None = None,
) -> IndependentTestDesign:
    """Select central/distant genotypes and distant environments without y.

    Central held-out genotypes create the easy test: they are new IDs but lie
    close to the population centre.  Distant held-out genotypes and weather
    outlier environments create progressively harder extrapolation tests.
    """

    genotype_ids = [str(value) for value in genotype_ids]
    pca = np.asarray(genotype_pca_scores, dtype=np.float64)
    if pca.shape[0] != len(genotype_ids):
        raise ValueError("PCA rows and genotype IDs must align")
    genotype_distance = _standardized_radius(pca)
    order = np.argsort(genotype_distance, kind="mergesort")
    n_tail = max(1, int(round(len(genotype_ids) * genotype_fraction_per_tail)))
    if 2 * n_tail >= len(genotype_ids):
        raise ValueError("genotype holdout tails leave no development genotypes")
    environment_ids = sorted(environment_features)
    environment_matrix = np.asarray([environment_features[value] for value in environment_ids], dtype=np.float64)
    environment_distance = _standardized_radius(environment_matrix)
    n_hard_environment = max(1, int(round(len(environment_ids) * hard_environment_fraction)))
    n_hard_environment = min(n_hard_environment, max(1, len(environment_ids) - 1))
    environment_order = np.argsort(environment_distance, kind="mergesort")
    hard_environments = frozenset(environment_ids[index] for index in environment_order[-n_hard_environment:])
    hard_candidate_ids: set[str] | None = None
    if observed_pairs is not None:
        hard_candidate_ids = {str(genotype) for genotype, environment in observed_pairs if str(environment) in hard_environments}
    hard_candidate_indices = [index for index in order if hard_candidate_ids is None or genotype_ids[index] in hard_candidate_ids]
    if len(hard_candidate_indices) < n_tail:
        hard_candidate_indices = order.tolist()
    hard_indices = hard_candidate_indices[-n_tail:]
    hard = frozenset(genotype_ids[index] for index in hard_indices)
    # Central easy genotypes are selected after hard candidates so the sets
    # remain disjoint even in a very small crop release.
    easy_indices = [index for index in order if genotype_ids[index] not in hard][:n_tail]
    easy = frozenset(genotype_ids[index] for index in easy_indices)
    return IndependentTestDesign(
        easy_genotypes=easy,
        hard_genotypes=hard,
        hard_environments=hard_environments,
        genotype_novelty={value: float(genotype_distance[index]) for index, value in enumerate(genotype_ids)},
        environment_novelty={value: float(environment_distance[index]) for index, value in enumerate(environment_ids)},
    )


def independent_role(genotype_id: str, environment_id: str, design: IndependentTestDesign) -> str:
    """Assign one row to development or a frozen test difficulty."""

    genotype_id = str(genotype_id)
    environment_id = str(environment_id)
    is_easy = genotype_id in design.easy_genotypes
    is_hard = genotype_id in design.hard_genotypes
    is_hard_environment = environment_id in design.hard_environments
    if not is_easy and not is_hard and not is_hard_environment:
        return "development"
    if is_easy and not is_hard_environment:
        return "test_easy_g"
    if is_hard and not is_hard_environment:
        return "test_hard_g"
    if not is_easy and not is_hard and is_hard_environment:
        return "test_hard_e"
    if is_hard and is_hard_environment:
        return "test_hard_ge"
    # Central/easy genotypes in a weather-outlier environment mix opposing
    # difficulty definitions.  Purging them keeps easy and hard interpretable.
    return "purge_mixed"


def assert_independent_design(rows: list[dict], roles: Sequence[str]) -> None:
    if len(rows) != len(roles):
        raise AssertionError("role count does not match row count")
    development = [rows[index] for index, role in enumerate(roles) if role == "development"]
    easy = [rows[index] for index, role in enumerate(roles) if role == "test_easy_g"]
    hard_g = [rows[index] for index, role in enumerate(roles) if role == "test_hard_g"]
    hard_e = [rows[index] for index, role in enumerate(roles) if role == "test_hard_e"]
    hard_ge = [rows[index] for index, role in enumerate(roles) if role == "test_hard_ge"]
    development_g = {row["genotype_id"] for row in development}
    development_e = {row["environment_id"] for row in development}
    assert development_g.isdisjoint({row["genotype_id"] for row in easy})
    assert development_g.isdisjoint({row["genotype_id"] for row in hard_g})
    assert development_g.isdisjoint({row["genotype_id"] for row in hard_ge})
    assert development_e.isdisjoint({row["environment_id"] for row in hard_e})
    assert development_e.isdisjoint({row["environment_id"] for row in hard_ge})
