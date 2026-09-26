"""Strict fold-local full-marker genomic preprocessing.

The fitted object is intentionally stateful: every learned statistic carries a
fingerprint of the training genotypes.  Benchmark code can therefore reject an
artifact fitted on a different outer fold instead of silently leaking it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


def fingerprint(values: list[str] | np.ndarray) -> str:
    joined = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass
class FullMarkerTransformer:
    min_maf: float = 0.01
    max_missing: float = 0.25
    low_rank_imputation_rank: int | None = None
    low_rank_imputation_weight: float = 1.0
    marker_indices: np.ndarray | None = field(default=None, init=False)
    allele_frequency: np.ndarray | None = field(default=None, init=False)
    scale: np.ndarray | None = field(default=None, init=False)
    denominator: float = field(default=0.0, init=False)
    training_fingerprint: str = field(default="", init=False)
    imputation_components: np.ndarray | None = field(default=None, init=False)

    def fit(self, dosage: np.ndarray, genotype_indices: np.ndarray, genotype_ids: list[str]) -> "FullMarkerTransformer":
        indices = np.unique(np.asarray(genotype_indices, dtype=int))
        if not len(indices):
            raise ValueError("No training genotypes")
        x = np.asarray(dosage[indices])
        observed = x >= 0
        count = observed.sum(axis=0)
        allele_sum = np.where(observed, x, 0).sum(axis=0, dtype=np.float64)
        p = np.divide(allele_sum, 2.0 * count, out=np.zeros(x.shape[1]), where=count > 0)
        maf = np.minimum(p, 1.0 - p)
        missing = 1.0 - count / len(indices)
        valid = (count > 0) & (maf >= self.min_maf) & (missing <= self.max_missing) & (p > 0) & (p < 1)
        self.marker_indices = np.flatnonzero(valid).astype(np.int32)
        if not len(self.marker_indices):
            raise ValueError("No markers passed fold-local QC")
        self.allele_frequency = p[self.marker_indices].astype(np.float32)
        # VanRaden I: centre by 2p and divide the relationship matrix by
        # sum(2p(1-p)).  scale is retained for optional correlation kernels.
        self.scale = np.sqrt(2.0 * self.allele_frequency * (1.0 - self.allele_frequency)).astype(np.float32)
        self.scale[self.scale < 1e-6] = 1.0
        self.denominator = float(np.sum(2.0 * self.allele_frequency * (1.0 - self.allele_frequency)))
        if self.low_rank_imputation_rank is not None:
            if not 0.0 <= self.low_rank_imputation_weight <= 1.0:
                raise ValueError("low_rank_imputation_weight must be in [0, 1]")
            from sklearn.utils.extmath import randomized_svd
            selected = x[:, self.marker_indices].astype(np.float32, copy=True)
            selected_observed = selected >= 0
            impute = 2.0 * self.allele_frequency
            selected[~selected_observed] = np.take(impute, np.where(~selected_observed)[1])
            selected -= impute
            rank = min(int(self.low_rank_imputation_rank), selected.shape[0] - 1, selected.shape[1])
            if rank > 0:
                _, _, vt = randomized_svd(selected, n_components=rank, n_iter=4, random_state=20260924)
                self.imputation_components = vt.astype(np.float32)
        self.training_fingerprint = fingerprint([genotype_ids[i] for i in indices])
        return self

    def assert_fitted_on(self, genotype_ids: list[str]) -> None:
        if not self.training_fingerprint:
            raise RuntimeError("FullMarkerTransformer is not fitted")
        if self.training_fingerprint != fingerprint(genotype_ids):
            raise RuntimeError("Fold-local genomic artifact was fitted on different genotypes")

    def transform(self, dosage: np.ndarray, genotype_indices: np.ndarray, standardized: bool = False) -> np.ndarray:
        if self.marker_indices is None or self.allele_frequency is None:
            raise RuntimeError("FullMarkerTransformer is not fitted")
        x = np.asarray(dosage[np.asarray(genotype_indices, dtype=int)][:, self.marker_indices]).astype(np.float32, copy=True)
        missing = x < 0
        impute = 2.0 * self.allele_frequency
        x[missing] = np.take(impute, np.where(missing)[1])
        x -= impute
        if self.imputation_components is not None and np.any(missing):
            reconstructed = (x @ self.imputation_components.T) @ self.imputation_components
            x[missing] = self.low_rank_imputation_weight * reconstructed[missing]
        if standardized:
            assert self.scale is not None
            x /= self.scale
        return x

    def metadata(self) -> dict[str, object]:
        return {
            "min_maf": self.min_maf,
            "max_missing": self.max_missing,
            "n_markers": 0 if self.marker_indices is None else int(len(self.marker_indices)),
            "denominator": self.denominator,
            "training_fingerprint": self.training_fingerprint,
            "low_rank_imputation_rank": self.low_rank_imputation_rank,
            "low_rank_imputation_weight": self.low_rank_imputation_weight,
        }


@dataclass
class FoldEnvironmentTransformer:
    n_components: int = 4
    impute_: np.ndarray | None = field(default=None, init=False)
    center_: np.ndarray | None = field(default=None, init=False)
    scale_: np.ndarray | None = field(default=None, init=False)
    projection_: np.ndarray | None = field(default=None, init=False)
    training_fingerprint: str = field(default="", init=False)

    def fit(self, features: dict[str, np.ndarray], training_environments: list[str]) -> "FoldEnvironmentTransformer":
        ids = sorted(set(training_environments))
        if not ids:
            raise ValueError("No training environments")
        matrix = np.asarray([features[value] for value in ids], dtype=np.float32)
        self.impute_ = np.zeros(matrix.shape[1], dtype=np.float32)
        for j in range(matrix.shape[1]):
            observed = matrix[:, j][np.isfinite(matrix[:, j])]
            self.impute_[j] = float(np.median(observed)) if len(observed) else 0.0
        filled = np.where(np.isfinite(matrix), matrix, self.impute_)
        self.center_ = filled.mean(0)
        self.scale_ = filled.std(0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        standardized = (filled - self.center_) / self.scale_
        rank = min(self.n_components, max(1, len(ids) - 1), standardized.shape[1])
        _, _, vt = np.linalg.svd(standardized, full_matrices=False)
        self.projection_ = vt[:rank].T.astype(np.float32)
        self.training_fingerprint = fingerprint(ids)
        return self

    def transform(self, features: dict[str, np.ndarray], environments: list[str]) -> np.ndarray:
        if self.impute_ is None or self.center_ is None or self.scale_ is None or self.projection_ is None:
            raise RuntimeError("FoldEnvironmentTransformer is not fitted")
        raw = np.asarray([features.get(value, np.full(len(self.impute_), np.nan)) for value in environments], dtype=np.float32)
        filled = np.where(np.isfinite(raw), raw, self.impute_)
        return (((filled - self.center_) / self.scale_) @ self.projection_).astype(np.float32)


@dataclass
class FoldGenotypePCA:
    """PCA fitted only to outer-training genotypes after fold-local QC."""
    n_components: int = 128
    mean_: np.ndarray | None = field(default=None, init=False)
    components_: np.ndarray | None = field(default=None, init=False)
    training_fingerprint: str = field(default="", init=False)

    def fit(self, transformed_training_dosage: np.ndarray, training_ids: list[str]) -> "FoldGenotypePCA":
        from sklearn.decomposition import PCA
        x = np.asarray(transformed_training_dosage, dtype=np.float32)
        rank = min(self.n_components, x.shape[0] - 1, x.shape[1])
        if rank < 1:
            raise ValueError("Insufficient training genotypes for PCA")
        estimator = PCA(n_components=rank, svd_solver="randomized", random_state=20260922).fit(x)
        self.mean_ = estimator.mean_.astype(np.float32)
        self.components_ = estimator.components_.astype(np.float32)
        self.training_fingerprint = fingerprint(training_ids)
        return self

    def transform(self, transformed_dosage: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("FoldGenotypePCA is not fitted")
        return ((np.asarray(transformed_dosage, dtype=np.float32) - self.mean_) @ self.components_.T).astype(np.float32)
