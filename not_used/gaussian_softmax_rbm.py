from gaussian_rbm import GaussianRBM
from sklearn.utils.extmath import safe_sparse_dot
from scipy.special import softmax
import numpy as np


class GaussianSoftmaxRBM(GaussianRBM):
    def __init__(
        self,
        n_components=256,
        *,
        learning_rate=0.1,
        batch_size=10,
        n_iter=10,
        verbose=0,
        random_state=None,
        grad_max=1.0,
        sigma=0.3,
    ):
        super().__init__(
            n_components=n_components,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_iter=n_iter,
            verbose=verbose,
            random_state=random_state,
            grad_max=grad_max,
            sigma=sigma
        )

    def _mean_hiddens(self, v):
        """Computes the universal softmax probabilities P(h|v)."""
        v_normalized = v / self.sigma
        activations = safe_sparse_dot(v_normalized, self.components_.T) + self.intercept_hidden_
        return softmax(activations, axis=1)

    def _sample_hiddens(self, v, rng):
        """Sample one-hot vectors from the universal softmax distribution."""
        probs = self._mean_hiddens(v)
        h = np.zeros_like(probs)

        # Sample one index per row using softmax probs
        indices = [rng.choice(probs.shape[1], p=row) for row in probs]
        h[np.arange(probs.shape[0]), indices] = 1
        return h

    def _free_energy(self, v):
        """Computes the free energy for universal softmax."""
        diff = v - self.intercept_visible_
        quad_term = 0.5 * np.sum((diff / self.sigma) ** 2, axis=1)

        v_normalized = v / self.sigma
        hidden_act = safe_sparse_dot(v_normalized, self.components_.T) + self.intercept_hidden_

        # Log-sum-exp over all hidden units
        hidden_term = np.log(np.sum(np.exp(hidden_act), axis=1))
        return quad_term - hidden_term
