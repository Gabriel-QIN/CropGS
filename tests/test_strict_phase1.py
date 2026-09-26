from __future__ import annotations

import unittest

import numpy as np

from src.models.kernel_gs import MatrixFreeReactionNorm, environment_kernel_factors
from src.models.environment_calibration import EnvironmentLocationScale
from src.inference.final_predictor import PortableEnvironmentKernel, PortableKernelPredictor
from src.preprocessing.genomic import FoldGenotypePCA, FullMarkerTransformer
from src.validation.leakage import assert_protocol_isolation


class StrictPreprocessingTest(unittest.TestCase):
    def test_full_marker_statistics_are_train_only(self) -> None:
        dosage = np.asarray([[0, 0, 2], [2, 1, 0], [2, 2, 0]], dtype=np.int8)
        transformer = FullMarkerTransformer(min_maf=0, max_missing=1).fit(dosage, np.asarray([0, 1]), ["A", "B", "C"])
        np.testing.assert_allclose(transformer.allele_frequency, [0.5, 0.25, 0.5])
        transformer.assert_fitted_on(["A", "B"])
        with self.assertRaises(RuntimeError):
            transformer.assert_fitted_on(["A", "B", "C"])

    def test_low_rank_imputation_is_finite(self) -> None:
        dosage = np.asarray([[0, -1, 2, 0], [1, 1, -1, 1], [2, 2, 0, -1], [1, 0, 1, 2]], dtype=np.int8)
        transformer = FullMarkerTransformer(min_maf=0, max_missing=1, low_rank_imputation_rank=2).fit(dosage, np.arange(3), ["A", "B", "C", "D"])
        transformed = transformer.transform(dosage, np.arange(4))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertIsNotNone(transformer.imputation_components)

    def test_low_rank_imputation_weight_zero_matches_mean_imputation(self) -> None:
        dosage = np.asarray([[0, -1, 2, 0], [1, 1, -1, 1], [2, 2, 0, -1], [1, 0, 1, 2]], dtype=np.int8)
        train = np.arange(3)
        mean = FullMarkerTransformer(min_maf=0, max_missing=1).fit(dosage, train, ["A", "B", "C", "D"])
        shrunk = FullMarkerTransformer(min_maf=0, max_missing=1, low_rank_imputation_rank=2, low_rank_imputation_weight=0.0).fit(dosage, train, ["A", "B", "C", "D"])
        np.testing.assert_allclose(shrunk.transform(dosage, np.arange(4)), mean.transform(dosage, np.arange(4)), atol=1e-6)

    def test_low_rank_imputation_weight_is_bounded(self) -> None:
        dosage = np.asarray([[0, -1], [1, 1], [2, 2]], dtype=np.int8)
        with self.assertRaises(ValueError):
            FullMarkerTransformer(min_maf=0, max_missing=1, low_rank_imputation_rank=1, low_rank_imputation_weight=1.1).fit(dosage, np.arange(3), ["A", "B", "C"])

    def test_leakage_is_runtime_error(self) -> None:
        rows = [{"genotype_id": "G1", "environment_id": "E1"}, {"genotype_id": "G1", "environment_id": "E2"}]
        with self.assertRaises(RuntimeError):
            assert_protocol_isolation(rows, np.asarray([0]), np.asarray([1]), "cv_g")

    def test_fold_pca_fits_and_transforms(self) -> None:
        x = np.arange(30, dtype=np.float32).reshape(6, 5)
        pca = FoldGenotypePCA(3).fit(x[:5], [f"G{i}" for i in range(5)])
        self.assertEqual(pca.transform(x).shape, (6, 3))


class KernelModelTest(unittest.TestCase):
    def test_reaction_norm_fits_small_signal(self) -> None:
        x = np.asarray([[-1, -1], [0, 0], [1, 1]], dtype=np.float32)
        obs_g = np.asarray([0, 1, 2, 0, 1, 2])
        env = np.asarray([[-1], [-1], [-1], [1], [1], [1]], dtype=np.float32)
        y = x[obs_g, 0] + 0.5 * env[:, 0] + x[obs_g, 0] * env[:, 0]
        model = MatrixFreeReactionNorm("reaction_norm_gblup", residual_ratio=0.05, max_iter=100).fit(x, obs_g, env, y, denominator=2.0)
        pred = model.predict(x, obs_g, env)
        self.assertGreater(np.corrcoef(y, pred)[0, 1], 0.95)
        self.assertIn("GxE", model.fit_.variance_components)

    def test_rbf_environment_cross_factor(self) -> None:
        train = np.asarray([[0.0], [1.0], [2.0]])
        test = np.asarray([[0.5]])
        a, b = environment_kernel_factors(train, test, "rbf")
        self.assertEqual(a.shape[0], 3)
        self.assertEqual(b.shape[0], 1)
        self.assertTrue(np.isfinite(b).all())

    def test_portable_kernel_export_matches_fitted_model(self) -> None:
        x = np.asarray([[-1, 0], [0, 1], [1, -1]], dtype=np.float32)
        obs = np.asarray([0, 1, 2, 0, 1, 2])
        pc = np.asarray([[-1.0], [1.0]], dtype=np.float32)
        factorizer, unique_factor = PortableEnvironmentKernel.fit(pc, "rbf", 1.0)
        factor = unique_factor[np.asarray([0, 0, 0, 1, 1, 1])]
        y = np.asarray([0.0, 1.0, 2.0, 1.0, 2.5, 1.5], dtype=np.float32)
        model = MatrixFreeReactionNorm("rkhs", residual_ratio=0.2, max_iter=100).fit(x, obs, factor, y, denominator=2.0)
        portable = PortableKernelPredictor.from_fitted(model)
        expected = model.predict(x, obs, factor)
        actual = portable.predict(x, obs, factorizer.transform(pc)[np.asarray([0, 0, 0, 1, 1, 1])])
        np.testing.assert_allclose(actual, expected, atol=2e-5)

    def test_environment_calibration_never_uses_unseen_targets(self) -> None:
        ids = ["E1"] * 3 + ["E2"] * 3 + ["E3"] * 3
        y = np.asarray([0, 1, 2, 10, 12, 14, 20, 23, 26], dtype=float)
        pc = np.asarray([[-1.0], [0.0], [1.0]])
        calibration = EnvironmentLocationScale().fit(ids, y, ["E1", "E2", "E3"], pc)
        normalized = calibration.normalize(ids, y)
        self.assertTrue(np.isfinite(normalized).all())
        restored = calibration.restore(np.zeros(2), ["E2", "UNSEEN"], np.asarray([[0.0], [2.0]]))
        self.assertAlmostEqual(restored[0], 12.0, places=5)
        self.assertTrue(np.isfinite(restored[1]))


if __name__ == "__main__":
    unittest.main()
