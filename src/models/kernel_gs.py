"""Matrix-free full-marker GBLUP and reaction-norm kernel ridge models.

No dense observation GRM is formed.  The identity
``Z X X' Z' v = Z X (X' (Z' v))`` keeps the exact VanRaden linear kernel
scalable and the analogous low-rank identity is used for GxE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


def _cg_numpy(matvec, b: np.ndarray, tolerance: float, max_iter: int, diagonal: np.ndarray | None = None) -> tuple[np.ndarray, int, float]:
    x = np.zeros_like(b)
    r = b.copy()
    z = r / diagonal if diagonal is not None else r.copy()
    p = z.copy()
    rr = float(r @ z)
    target = tolerance * max(1.0, float(np.linalg.norm(b)))
    for iteration in range(1, max_iter + 1):
        ap = matvec(p)
        alpha = rr / max(float(p @ ap), 1e-20)
        x += alpha * p
        r -= alpha * ap
        residual = float(np.linalg.norm(r))
        if residual <= target:
            return x, iteration, residual
        z = r / diagonal if diagonal is not None else r.copy()
        rr_new = float(r @ z)
        p = z + (rr_new / max(rr, 1e-20)) * p
        rr = rr_new
    return x, max_iter, float(np.linalg.norm(r))


def environment_kernel_factors(
    train_environment: np.ndarray,
    test_environment: np.ndarray,
    kind: Literal["linear", "rbf"] = "linear",
    bandwidth_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test factors whose product gives the environment kernel."""
    train = np.asarray(train_environment, dtype=np.float64)
    test = np.asarray(test_environment, dtype=np.float64)
    if kind == "linear":
        scale = np.sqrt(max(1, train.shape[1]))
        return (train / scale).astype(np.float32), (test / scale).astype(np.float32)
    distances = np.sum((train[:, None, :] - train[None, :, :]) ** 2, axis=2)
    positive = distances[distances > 1e-12]
    bandwidth2 = (float(np.median(positive)) if len(positive) else 1.0) * bandwidth_multiplier**2
    k_train = np.exp(-distances / max(2.0 * bandwidth2, 1e-12))
    values, vectors = np.linalg.eigh((k_train + k_train.T) / 2.0)
    keep = values > 1e-8
    train_factor = vectors[:, keep] * np.sqrt(values[keep])
    cross_dist = np.sum((test[:, None, :] - train[None, :, :]) ** 2, axis=2)
    k_cross = np.exp(-cross_dist / max(2.0 * bandwidth2, 1e-12))
    test_factor = k_cross @ (vectors[:, keep] / np.sqrt(values[keep]))
    return train_factor.astype(np.float32), test_factor.astype(np.float32)


@dataclass
class KernelFit:
    alpha: np.ndarray
    mean: float
    variance_components: dict[str, float]
    cg_iterations: int
    cg_residual: float


class MatrixFreeReactionNorm:
    """Exact linear full-marker kernel with optional E and GxE kernels."""

    def __init__(self, model: str = "reaction_norm_gblup", residual_ratio: float = 0.2, max_iter: int = 80, tolerance: float = 1e-5, device: str = "cpu"):
        if model not in {"gblup", "gblup_environment", "reaction_norm_gblup", "rkhs"}:
            raise ValueError(model)
        self.model = model
        self.residual_ratio = float(residual_ratio)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.device = device
        self.fit_: KernelFit | None = None
        self._state: dict[str, np.ndarray | float] = {}

    @staticmethod
    def _genomic_op(x: np.ndarray, obs_g: np.ndarray, v: np.ndarray, denominator: float) -> np.ndarray:
        aggregate = np.bincount(obs_g, weights=v, minlength=x.shape[0]).astype(np.float32)
        genetic = x @ (x.T @ aggregate) / denominator
        return genetic[obs_g]

    def _component_ops(self, x: np.ndarray, obs_g: np.ndarray, e_factor: np.ndarray, denominator: float):
        def g(v):
            return self._genomic_op(x, obs_g, v, denominator)
        def e(v):
            return e_factor @ (e_factor.T @ v)
        def ge(v):
            result = np.zeros_like(v)
            for column in range(e_factor.shape[1]):
                weighted = e_factor[:, column] * v
                result += e_factor[:, column] * g(weighted)
            return result
        return g, e, ge

    @staticmethod
    def _normalizers(x: np.ndarray, obs_g: np.ndarray, e: np.ndarray, denominator: float) -> tuple[float, float, float]:
        gdiag_unique = np.sum(x * x, axis=1) / denominator
        gdiag = gdiag_unique[obs_g]
        ediag = np.sum(e * e, axis=1)
        return max(float(np.mean(gdiag)), 1e-8), max(float(np.mean(ediag)), 1e-8), max(float(np.mean(gdiag * ediag)), 1e-8)

    def fit(self, x_genotype: np.ndarray, observation_genotype: np.ndarray, environment_factor: np.ndarray, y: np.ndarray, denominator: float) -> "MatrixFreeReactionNorm":
        if self.device != "cpu":
            return self._fit_torch(x_genotype, observation_genotype, environment_factor, y, denominator)
        x = np.asarray(x_genotype, dtype=np.float32)
        obs_g = np.asarray(observation_genotype, dtype=np.int64)
        e = np.asarray(environment_factor, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        mean = float(np.mean(y))
        centered = y - mean
        g, env, ge = self._component_ops(x, obs_g, e, denominator)
        ng, ne, nge = self._normalizers(x, obs_g, e, denominator)
        ops = {"G": lambda v: g(v) / ng}
        if self.model != "gblup":
            ops["E"] = lambda v: env(v) / ne
        if self.model in {"reaction_norm_gblup", "rkhs"}:
            ops["GxE"] = lambda v: ge(v) / nge
        # Kernel-target alignment provides deterministic, training-only
        # non-negative component estimates.  A residual floor stabilizes OOD.
        energy = max(float(centered @ centered), 1e-12)
        raw = {name: max(float(centered @ op(centered)) / energy, 1e-6) for name, op in ops.items()}
        total = sum(raw.values())
        signal = 1.0 - min(max(self.residual_ratio, 0.05), 0.8)
        weights = {name: signal * value / total for name, value in raw.items()}
        weights["Residual"] = 1.0 - signal

        def matvec(v):
            result = weights["Residual"] * v
            for name, op in ops.items():
                result = result + weights[name] * op(v)
            return result

        gdiag = np.sum(x * x, axis=1)[obs_g] / denominator / ng
        diagonal = weights["Residual"] + weights["G"] * gdiag
        if self.model != "gblup":
            ediag = np.sum(e * e, axis=1) / ne
            diagonal += weights["E"] * ediag
        if self.model in {"reaction_norm_gblup", "rkhs"}:
            diagonal += weights["GxE"] * (gdiag * (np.sum(e * e, axis=1) / ne)) / max(nge / (ng * ne), 1e-8)
        alpha, iterations, residual = _cg_numpy(matvec, centered.copy(), self.tolerance, self.max_iter, np.maximum(diagonal, 1e-6))
        self.fit_ = KernelFit(alpha, mean, weights, iterations, residual)
        self._state = {"x": x, "obs_g": obs_g, "e": e, "denominator": float(denominator), "ng": ng, "ne": ne, "nge": nge}
        return self

    def _fit_torch(self, x_genotype, observation_genotype, environment_factor, y, denominator):
        import torch
        device = torch.device(self.device)
        x = torch.as_tensor(np.asarray(x_genotype), dtype=torch.float32, device=device)
        obs = torch.as_tensor(np.asarray(observation_genotype), dtype=torch.long, device=device)
        e = torch.as_tensor(np.asarray(environment_factor), dtype=torch.float32, device=device)
        target = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=device)
        mean = target.mean()
        yc = target - mean

        def genomic(v):
            aggregate = torch.zeros(x.shape[0], dtype=torch.float32, device=device).scatter_add_(0, obs, v)
            return (x @ (x.T @ aggregate) / denominator)[obs]
        def env(v):
            return e @ (e.T @ v)
        def interaction(v):
            result = torch.zeros_like(v)
            for column in range(e.shape[1]):
                result.add_(e[:, column] * genomic(e[:, column] * v))
            return result
        gdiag = (x.square().sum(1) / denominator)[obs]
        ediag = e.square().sum(1)
        ng = torch.clamp(gdiag.mean(), min=1e-8)
        ne = torch.clamp(ediag.mean(), min=1e-8)
        nge = torch.clamp((gdiag * ediag).mean(), min=1e-8)
        ops = {"G": lambda v: genomic(v) / ng}
        if self.model != "gblup":
            ops["E"] = lambda v: env(v) / ne
        if self.model in {"reaction_norm_gblup", "rkhs"}:
            ops["GxE"] = lambda v: interaction(v) / nge
        energy = torch.clamp(yc @ yc, min=1e-12)
        raw = {name: torch.clamp((yc @ op(yc)) / energy, min=1e-6) for name, op in ops.items()}
        total = sum(raw.values())
        signal = 1.0 - min(max(self.residual_ratio, 0.05), 0.8)
        weights = {name: float((signal * value / total).item()) for name, value in raw.items()}
        weights["Residual"] = 1.0 - signal
        def matvec(v):
            result = weights["Residual"] * v
            for name, op in ops.items():
                result = result + weights[name] * op(v)
            return result
        diagonal = weights["Residual"] + weights["G"] * gdiag / ng
        if self.model != "gblup":
            diagonal = diagonal + weights["E"] * ediag / ne
        if self.model in {"reaction_norm_gblup", "rkhs"}:
            diagonal = diagonal + weights["GxE"] * (gdiag * ediag) / nge
        diagonal = torch.clamp(diagonal, min=1e-6)
        solution = torch.zeros_like(yc)
        residual = yc.clone()
        preconditioned = residual / diagonal
        direction = preconditioned.clone()
        rr = residual @ preconditioned
        threshold = self.tolerance * max(1.0, float(torch.linalg.norm(yc).item()))
        final_residual = float(torch.linalg.norm(residual).item())
        iteration = 0
        for iteration in range(1, self.max_iter + 1):
            product = matvec(direction)
            step = rr / torch.clamp(direction @ product, min=1e-20)
            solution.add_(step * direction)
            residual.sub_(step * product)
            final_residual = float(torch.linalg.norm(residual).item())
            if final_residual <= threshold:
                break
            preconditioned = residual / diagonal
            rr_new = residual @ preconditioned
            direction = preconditioned + (rr_new / torch.clamp(rr, min=1e-20)) * direction
            rr = rr_new
        self.fit_ = KernelFit(solution.detach().cpu().numpy(), float(mean.item()), weights, iteration, final_residual)
        self._state = {"x": x, "obs_g": obs, "e": e, "denominator": float(denominator), "ng": float(ng.item()), "ne": float(ne.item()), "nge": float(nge.item()), "alpha_torch": solution}
        return self

    def predict(self, x_test_genotype: np.ndarray, test_observation_genotype: np.ndarray, test_environment_factor: np.ndarray) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("Model is not fitted")
        if self.device != "cpu":
            return self._predict_torch(x_test_genotype, test_observation_genotype, test_environment_factor)
        x_train = self._state["x"]
        obs_train = self._state["obs_g"]
        e_train = self._state["e"]
        assert isinstance(x_train, np.ndarray) and isinstance(obs_train, np.ndarray) and isinstance(e_train, np.ndarray)
        x_test = np.asarray(x_test_genotype, dtype=np.float32)
        obs_test = np.asarray(test_observation_genotype, dtype=np.int64)
        e_test = np.asarray(test_environment_factor, dtype=np.float32)
        alpha = self.fit_.alpha
        aggregate = np.bincount(obs_train, weights=alpha, minlength=x_train.shape[0]).astype(np.float32)
        base_unique = x_test @ (x_train.T @ aggregate) / float(self._state["denominator"])
        g_cross = base_unique[obs_test] / float(self._state["ng"])
        result = self.fit_.variance_components["G"] * g_cross
        if self.model != "gblup":
            e_cross = e_test @ (e_train.T @ alpha) / float(self._state["ne"])
            result += self.fit_.variance_components["E"] * e_cross
        if self.model in {"reaction_norm_gblup", "rkhs"}:
            ge_cross = np.zeros(len(obs_test), dtype=np.float32)
            for column in range(e_train.shape[1]):
                weighted = e_train[:, column] * alpha
                agg = np.bincount(obs_train, weights=weighted, minlength=x_train.shape[0]).astype(np.float32)
                cross = x_test @ (x_train.T @ agg) / float(self._state["denominator"])
                ge_cross += e_test[:, column] * cross[obs_test]
            ge_cross /= float(self._state["nge"])
            result += self.fit_.variance_components["GxE"] * ge_cross
        return self.fit_.mean + result

    def _predict_torch(self, x_test_genotype, test_observation_genotype, test_environment_factor):
        import torch
        assert self.fit_ is not None
        x_train = self._state["x"]
        obs_train = self._state["obs_g"]
        e_train = self._state["e"]
        alpha = self._state["alpha_torch"]
        assert isinstance(x_train, torch.Tensor)
        device = x_train.device
        x_test = torch.as_tensor(np.asarray(x_test_genotype), dtype=torch.float32, device=device)
        obs_test = torch.as_tensor(np.asarray(test_observation_genotype), dtype=torch.long, device=device)
        e_test = torch.as_tensor(np.asarray(test_environment_factor), dtype=torch.float32, device=device)
        aggregate = torch.zeros(x_train.shape[0], device=device).scatter_add_(0, obs_train, alpha)
        base = (x_test @ (x_train.T @ aggregate) / float(self._state["denominator"]))[obs_test] / float(self._state["ng"])
        result = self.fit_.variance_components["G"] * base
        if self.model != "gblup":
            result = result + self.fit_.variance_components["E"] * (e_test @ (e_train.T @ alpha)) / float(self._state["ne"])
        if self.model in {"reaction_norm_gblup", "rkhs"}:
            interaction = torch.zeros(len(obs_test), device=device)
            for column in range(e_train.shape[1]):
                weighted = e_train[:, column] * alpha
                agg = torch.zeros(x_train.shape[0], device=device).scatter_add_(0, obs_train, weighted)
                cross = (x_test @ (x_train.T @ agg) / float(self._state["denominator"]))[obs_test]
                interaction.add_(e_test[:, column] * cross)
            result = result + self.fit_.variance_components["GxE"] * interaction / float(self._state["nge"])
        return (result + self.fit_.mean).detach().cpu().numpy()
