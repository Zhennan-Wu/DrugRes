import unittest
import numpy as np
from sklearn.utils import check_random_state
from gaussian_rbm import GaussianRBM


class TestGaussianRBM(unittest.TestCase):
    def setUp(self):
        self.n_samples = 10
        self.n_features = 6
        self.n_components = 3
        self.sigma = 0.5
        self.rng = check_random_state(42)

        self.model = GaussianRBM(n_components=self.n_components, sigma=self.sigma, random_state=42)
        self.model.components_ = self.rng.normal(size=(self.n_components, self.n_features))
        self.model.intercept_hidden_ = self.rng.normal(size=self.n_components)
        self.model.intercept_visible_ = self.rng.normal(size=self.n_features)

        self.X = self.rng.normal(size=(self.n_samples, self.n_features))
        self.H = self.rng.binomial(1, 0.5, size=(self.n_samples, self.n_components))

    def test_hidden_probabilities_shape_and_range(self):
        probs = self.model._mean_hiddens(self.X)
        self.assertEqual(probs.shape, (self.n_samples, self.n_components))
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))

    def test_sample_visibles_shape_and_distribution(self):
        visibles = self.model._sample_visibles(self.H, self.rng)
        self.assertEqual(visibles.shape, (self.n_samples, self.n_features))
        self.assertTrue(np.isfinite(visibles).all())

    def test_free_energy_output(self):
        free_energy = self.model._free_energy(self.X)
        self.assertEqual(free_energy.shape, (self.n_samples,))
        self.assertTrue(np.isfinite(free_energy).all())

    def test_sigma_scalar_vs_vector(self):
        # Use a vector sigma for each feature
        self.model.sigma = np.full(self.n_features, self.sigma)
        energy_vector = self.model._free_energy(self.X)

        # Use scalar sigma
        self.model.sigma = self.sigma
        energy_scalar = self.model._free_energy(self.X)

        self.assertEqual(energy_vector.shape, energy_scalar.shape)
        self.assertTrue(np.allclose(energy_vector, energy_scalar, atol=1e-6))

if __name__ == '__main__':
    unittest.main()
