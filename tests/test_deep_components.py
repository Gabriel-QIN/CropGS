"""Small dependency-light tests for fold features and model tensor shapes."""

from __future__ import annotations

import unittest

import numpy as np

from src.preprocessing.deep_features import FoldFeatureTransformer
from src.validation.independent_test import assert_independent_design, build_independent_test_design, independent_role


class FoldFeatureTransformerTest(unittest.TestCase):
    def test_fit_transform_is_finite_and_preserves_missing_mask(self) -> None:
        dosage = np.asarray(
            [
                [0, 1, 2, -1, 0, 1],
                [0, 2, 1, 0, 1, 2],
                [2, 1, 0, 1, 2, 0],
                [1, -1, 2, 2, 0, 1],
                [2, 2, 1, 0, -1, 2],
            ],
            dtype=np.int8,
        )
        environments = {
            "E1": np.asarray([1.0, 10.0, np.nan], dtype=np.float32),
            "E2": np.asarray([2.0, 11.0, 5.0], dtype=np.float32),
        }
        transformer = FoldFeatureTransformer(max_markers=4, max_missing=0.5).fit(
            dosage,
            np.asarray([0, 1, 2], dtype=int),
            environments,
            ["E1"],
            ["days", "weather", "missing"],
        )
        transformed, missing = transformer.transform_genotypes(dosage)
        weather = transformer.transform_environments(["E1", "E2"], environments)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.isfinite(weather).all())
        self.assertEqual(transformed.shape, missing.shape)
        self.assertGreaterEqual(float(missing.sum()), 1.0)
        self.assertEqual(transformer.n_markers, len(transformer.marker_indices))


class DeepModelShapeTest(unittest.TestCase):
    def test_forward_shape(self) -> None:
        try:
            import torch
            from src.models.deep.genotype_environment_net import GenotypeEnvironmentNet
        except ImportError as exc:  # pragma: no cover - environment-specific
            self.skipTest(str(exc))
        model = GenotypeEnvironmentNet(n_markers=17, n_environment_features=5, block_size=8, d_model=24, n_heads=4, n_layers=1, head_hidden=32)
        prediction = model(torch.randn(3, 17), torch.zeros(3, 17), torch.randn(3, 5))
        self.assertEqual(tuple(prediction.shape), (3,))

    def test_reaction_norm_forward_shape(self) -> None:
        import torch
        from src.models.deep.genotype_environment_net import ReactionNormNet

        model = ReactionNormNet(n_markers=17, n_environment_features=5, block_size=8, d_model=24, n_heads=4, n_layers=1, head_hidden=32)
        prediction = model(torch.randn(3, 17), torch.zeros(3, 17), torch.randn(3, 5))
        self.assertEqual(tuple(prediction.shape), (3,))


class IndependentTestDesignTest(unittest.TestCase):
    def test_roles_are_leakage_free(self) -> None:
        genotype_ids = [f"G{i}" for i in range(20)]
        pca = np.column_stack((np.arange(20), np.arange(20) % 3)).astype(float)
        environments = {f"E{i}": np.asarray([float(i), float(i % 2)]) for i in range(5)}
        pairs = [(genotype, environment) for genotype in genotype_ids for environment in environments]
        design = build_independent_test_design(genotype_ids, pca, environments, 0.1, 0.2, pairs)
        rows = [{"genotype_id": genotype, "environment_id": environment} for genotype, environment in pairs]
        roles = [independent_role(row["genotype_id"], row["environment_id"], design) for row in rows]
        assert_independent_design(rows, roles)
        self.assertIn("test_easy_g", roles)
        self.assertIn("test_hard_ge", roles)


if __name__ == "__main__":
    unittest.main()
