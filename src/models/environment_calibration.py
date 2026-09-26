"""Training-only environment location/scale calibration.

Phenotypes from a held-out environment never enter this object.  Seen
environments use their training statistics; unseen environments are inferred
from fold-local weather PCs with ridge penalties selected by environment-level
leave-one-out CV.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class EnvironmentLocationScale:
    alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    means_: dict[str, float] = field(default_factory=dict, init=False)
    scales_: dict[str, float] = field(default_factory=dict, init=False)
    mean_model_: Ridge | None = field(default=None, init=False)
    scale_model_: Ridge | None = field(default=None, init=False)
    mean_alpha_: float = field(default=float("nan"), init=False)
    scale_alpha_: float = field(default=float("nan"), init=False)
    mean_bounds_: tuple[float, float] = field(default=(0.0, 0.0), init=False)
    scale_bounds_: tuple[float, float] = field(default=(1.0, 1.0), init=False)

    @staticmethod
    def _select_alpha(x: np.ndarray, target: np.ndarray, alphas: tuple[float, ...]) -> float:
        if len(target) < 3:
            return 100.0
        losses = []
        for alpha in alphas:
            prediction = np.empty(len(target), dtype=float)
            for held in range(len(target)):
                mask = np.arange(len(target)) != held
                prediction[held] = Ridge(alpha=alpha).fit(x[mask], target[mask]).predict(x[held : held + 1])[0]
            losses.append(float(np.mean((target - prediction) ** 2)))
        return float(alphas[int(np.argmin(losses))])

    def fit(self, environment_ids: list[str], y: np.ndarray, unique_environment_ids: list[str], unique_environment_pc: np.ndarray) -> "EnvironmentLocationScale":
        y = np.asarray(y, dtype=float)
        global_scale = max(float(np.std(y)), 1e-6)
        for environment in unique_environment_ids:
            values = y[np.asarray(environment_ids, dtype=object) == environment]
            self.means_[environment] = float(np.mean(values))
            scale = float(np.std(values, ddof=1)) if len(values) > 1 else global_scale
            self.scales_[environment] = max(scale, global_scale * 0.05, 1e-6)
        mean_target = np.asarray([self.means_[e] for e in unique_environment_ids])
        log_scale_target = np.log(np.asarray([self.scales_[e] for e in unique_environment_ids]))
        x = np.asarray(unique_environment_pc, dtype=float)
        self.mean_alpha_ = self._select_alpha(x, mean_target, self.alphas)
        self.scale_alpha_ = self._select_alpha(x, log_scale_target, self.alphas)
        self.mean_model_ = Ridge(alpha=self.mean_alpha_).fit(x, mean_target)
        self.scale_model_ = Ridge(alpha=self.scale_alpha_).fit(x, log_scale_target)
        mean_sd = max(float(np.std(mean_target)), 1e-6)
        self.mean_bounds_ = (float(np.min(mean_target) - mean_sd), float(np.max(mean_target) + mean_sd))
        scale_values = np.exp(log_scale_target)
        self.scale_bounds_ = (max(float(np.quantile(scale_values, 0.05)) * 0.5, 1e-6), float(np.quantile(scale_values, 0.95)) * 2.0)
        return self

    def normalize(self, environment_ids: list[str], y: np.ndarray) -> np.ndarray:
        return np.asarray([(float(value) - self.means_[environment]) / self.scales_[environment] for environment, value in zip(environment_ids, y)], dtype=np.float32)

    def parameters(self, environment_ids: list[str], environment_pc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.mean_model_ is None or self.scale_model_ is None:
            raise RuntimeError("EnvironmentLocationScale is not fitted")
        inferred_mean = np.clip(self.mean_model_.predict(environment_pc), *self.mean_bounds_)
        inferred_scale = np.clip(np.exp(self.scale_model_.predict(environment_pc)), *self.scale_bounds_)
        means = np.asarray([self.means_.get(e, inferred_mean[i]) for i, e in enumerate(environment_ids)], dtype=np.float32)
        scales = np.asarray([self.scales_.get(e, inferred_scale[i]) for i, e in enumerate(environment_ids)], dtype=np.float32)
        return means, scales

    def restore(self, standardized_prediction: np.ndarray, environment_ids: list[str], environment_pc: np.ndarray) -> np.ndarray:
        means, scales = self.parameters(environment_ids, environment_pc)
        return means + scales * np.asarray(standardized_prediction, dtype=np.float32)
