from bernoulli_rbm import BernoulliRBM
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.utils import check_random_state
from scipy.special import expit
from utils import clip_gradients
import numpy as np


class GaussianRBM(BernoulliRBM):
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
            grad_max=grad_max
        )
        self.sigma = sigma

    def _mean_hiddens(self, v):
        """Computes the probabilities P(h=1|v).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        # Normalize visible units by sigma
        v_normalized = v / self.sigma  # assumes self.sigma is either a vector or scalar
        p = safe_sparse_dot(v_normalized, self.components_.T)
        p += self.intercept_hidden_
        return expit(p, out=p)
    
    def _mean_visibles(self, h):
        """Compute mean of Gaussian visible units given hidden units."""
        return np.dot(h, self.components_) + self.intercept_visible_
    
    def _sample_visibles(self, h, rng):
        mean = self._mean_visibles(h)
        return rng.normal(loc=mean, scale=self.sigma)

    def _free_energy(self, v):
        """Computes the free energy for Gaussian RBM:
        F(v) = 0.5 * ||v - b||^2 - sum_j log(1 + exp(v @ W_j + c_j))

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : ndarray of shape (n_samples,)
            The value of the free energy.
        """
        # print("intercept_visible_ stats:")
        # print("intercept_visible_ stats: Min:", self.intercept_visible_.min())
        # print("intercept_visible_ stats: Max:", self.intercept_visible_.max())
        # print("intercept_visible_ stats: Mean:", self.intercept_visible_.mean())
        # print("intercept_visible_ stats: Any NaNs:", np.isnan(self.intercept_visible_).any())
        # print("intercept_visible_ stats: Any Infs:", np.isinf(self.intercept_visible_).any())

        # Quadratic term for Gaussian visible units
        quadratic_term = 0.5 * np.square(v - self.intercept_visible_).sum(axis=1)
        
        # Hidden unit activation input
        hidden_input = safe_sparse_dot(v, self.components_.T) + self.intercept_hidden_

        # Explicit debug inspection
        # print("hidden Any NaNs?", np.isnan(hidden_input).any())
        # print("hidden Any Infs?", np.isinf(hidden_input).any())
        # print("hidden Max value:", np.nanmax(hidden_input))
        # print("hidden Min value:", np.nanmin(hidden_input))

        # Log-sum-exp over hidden units
        hidden_term = np.logaddexp(0, hidden_input).sum(axis=1)

        return quadratic_term - hidden_term

    def score_samples(self, X):
        """Compute the pseudo-likelihood of X for continuous-valued input.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Continuous-valued visible units.

        Returns
        -------
        pseudo_likelihood : ndarray of shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).

        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a version of X with Gaussian noise added
        to a randomly chosen dimension per sample, and returns the log
        of the logistic function of the difference.
        """
        check_is_fitted(self)
        v = check_array(X, accept_sparse=False)
        # v = self._validate_data(X, accept_sparse=False, reset=False)
        rng = check_random_state(self.random_state)

        # Randomly pick one dimension per sample to corrupt
        batch_size, n_features = v.shape
        ind = (np.arange(batch_size), rng.randint(0, n_features, size=batch_size))

        # Copy and corrupt one element per row with Gaussian noise
        v_ = v.copy()
        noise_std = getattr(self, 'corruption_std', 0.1)  # default noise std
        v_[ind] += rng.normal(loc=0.0, scale=noise_std, size=batch_size)

        fe = self._free_energy(v)
        fe_ = self._free_energy(v_)

        return -n_features * np.logaddexp(0, -(fe_ - fe))
    
    def _fit(self, v_pos, rng):
        """Inner fit for one mini-batch.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        Parameters
        ----------
        v_pos : ndarray of shape (n_samples, n_features)
            The data to use for training.

        rng : RandomState instance
            Random number generator to use for sampling.
        """
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_, rng)
        h_neg = self._mean_hiddens(v_neg)
        v_pos_normalized = v_pos/self.sigma
        v_neg_normalized = v_neg/self.sigma

        lr = float(self.learning_rate) / v_pos.shape[0]
        update = safe_sparse_dot(v_pos_normalized.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg_normalized)
        update = clip_gradients(update, max_norm=self.grad_max)
        # print(f"Gradient max: {update.max()}")
        # print(f"Gradient min {update.min()}")
        # print(f"Gradient mean, {update.mean()}")
        self.components_ += lr * update
        # self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        # self.intercept_visible_ += lr * (
        #     np.asarray(v_pos.sum(axis=0)).squeeze() - v_neg.sum(axis=0)
        # )
        self.intercept_hidden_[:] = 0.
        self.intercept_visible_[:] = 0.

        h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
        self.h_samples_ = np.floor(h_neg, h_neg)


