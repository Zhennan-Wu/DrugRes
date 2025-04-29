import unittest
import numpy as np
from sklearn.utils import check_random_state
from bernoulli_softmax_rbm import BernoulliSoftmaxRBM


class TestBernoulliSoftmaxRBM(unittest.TestCase):
    def setUp(self):
        self.n_samples = 5
        self.n_features = 6
        self.hidden_group_sizes = [3, 2]  # total of 5 hidden units
        self.n_components = sum(self.hidden_group_sizes)
        self.rbm = BernoulliSoftmaxRBM(n_components=self.n_components)
        self.rbm.hidden_group_sizes = self.hidden_group_sizes

        rng = check_random_state(0)
        self.X = rng.rand(self.n_samples, self.n_features)
        
        # Initialize model parameters manually
        self.rbm.components_ = rng.normal(0, 0.1, (self.n_components, self.n_features))
        self.rbm.intercept_hidden_ = rng.normal(0, 0.1, self.n_components)
        self.rbm.intercept_visible_ = rng.normal(0, 0.1, self.n_features)
        self.rng = rng

    def test_mean_hiddens_shape(self):
        h_probs = self.rbm._mean_hiddens(self.X)
        self.assertEqual(h_probs.shape, (self.n_samples, self.n_components))

    def test_mean_hiddens_softmax_groupwise(self):
        h_probs = self.rbm._mean_hiddens(self.X)
        start = 0
        for size in self.hidden_group_sizes:
            end = start + size
            group_sums = h_probs[:, start:end].sum(axis=1)
            np.testing.assert_allclose(group_sums, 1.0, rtol=1e-5)
            start = end

    def test_sample_hiddens_one_hot(self):
        h_sample = self.rbm._sample_hiddens(self.X, self.rng)
        self.assertEqual(h_sample.shape, (self.n_samples, self.n_components))
        
        # Verify one-hot encoding per group
        start = 0
        for size in self.hidden_group_sizes:
            end = start + size
            group = h_sample[:, start:end]
            group_sums = group.sum(axis=1)
            np.testing.assert_array_equal(group_sums, np.ones(self.n_samples))
            start = end

    def test_free_energy_shape(self):
        energy = self.rbm._free_energy(self.X)
        self.assertEqual(energy.shape, (self.n_samples,))
        self.assertTrue(np.all(np.isfinite(energy)))


if __name__ == "__main__":
    unittest.main()
