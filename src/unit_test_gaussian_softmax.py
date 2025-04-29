import unittest
import numpy as np
from gaussian_softmax_rbm import GaussianSoftmaxRBM


class TestGaussianSoftmaxRBM(unittest.TestCase):
    def setUp(self):
        self.rbm = GaussianSoftmaxRBM(n_components=5, random_state=42, sigma=1.0)
        self.rbm.components_ = np.random.randn(5, 4)
        self.rbm.intercept_hidden_ = np.random.randn(5)
        self.rbm.intercept_visible_ = np.random.randn(4)
        self.rng = np.random.default_rng(seed=42)

        # Simulate small input data
        self.v = np.random.randn(3, 4)  # 3 samples, 4 visible units

    def test_mean_hiddens_shape_and_sum(self):
        probs = self.rbm._mean_hiddens(self.v)
        self.assertEqual(probs.shape, (3, 5))
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(3), atol=1e-6)

    def test_sample_hiddens_shape_and_one_hot(self):
        samples = self.rbm._sample_hiddens(self.v, self.rng)
        self.assertEqual(samples.shape, (3, 5))
        row_sums = np.sum(samples, axis=1)
        np.testing.assert_array_equal(row_sums, np.ones(3))  # one-hot vectors

    def test_free_energy_output_shape(self):
        energy = self.rbm._free_energy(self.v)
        self.assertEqual(energy.shape, (3,))
        self.assertTrue(np.all(np.isfinite(energy)))

    def test_determinism_with_seed(self):
        self.rbm.random_state = 123
        rng1 = np.random.default_rng(seed=123)
        rng2 = np.random.default_rng(seed=123)

        samples1 = self.rbm._sample_hiddens(self.v, rng1)
        samples2 = self.rbm._sample_hiddens(self.v, rng2)
        np.testing.assert_array_equal(samples1, samples2)


if __name__ == '__main__':
    unittest.main()
