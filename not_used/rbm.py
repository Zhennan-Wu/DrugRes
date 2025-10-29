import numpy as np
import time
from numbers import Integral, Real
import scipy.sparse as sp
from scipy.special import expit, softmax 
import torch
from torch.utils.data import DataLoader

from sklearn.neural_network import BernoulliRBM
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils import check_random_state, gen_even_slices
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
import matplotlib.pyplot as plt
import os


class RBM(BernoulliRBM):
    def __init__(self, n_components=2, learning_rate=0.1, lr_decay_factor = 0.5, lr_no_decay_length = 100, lr_decay = False, batch_size=10, n_iter=10, verbose=0, savefile="./", random_state=None, add_bias=False, target_in_model=False, hybrid=False, input_dist='bernoulli', latent_dist='bernoulli',target_dist='gaussian'):
        super().__init__(n_components=n_components, learning_rate=learning_rate,
                         batch_size=batch_size, n_iter=n_iter, verbose=verbose, random_state=random_state)
        self.add_bias = add_bias
        self.input_dist = input_dist # 'bernoulli' or 'gaussian'
        self.latent_dist = latent_dist # 'bernoulli' or 'multinomial'
        self.target_dist = target_dist # 'bernoulli' or 'gaussian'
        self.target_in_model = target_in_model
        self.hybrid = hybrid
        self.lr_decay_factor = lr_decay_factor
        self.lr_no_decay_length = lr_no_decay_length
        self.lr_decay = lr_decay
        
        if (not os.path.exists(savefile)):
            os.makedirs(savefile)
        self.savefile = savefile

    def transform(self, X, y):
        """Compute the hidden layer activation probabilities, P(h=1|v=X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Latent representations of the data.
        """
        check_is_fitted(self)
        return self._mean_hiddens(X, y)

    def _mean_hiddens(self, v, t):
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
        if (self.input_dist == 'bernoulli'):
            p = safe_sparse_dot(v, self.components_.T)
        elif (self.input_dist == 'gaussian'):
            p = safe_sparse_dot(v/self.sigma, self.components_.T)
        else:
            raise ValueError("Invalid input distribution: {}".format(self.input_dist))
        
        if (self.add_bias):
            p += self.intercept_hidden_

        if (self.target_in_model):
            if (self.target_dist == 'bernoulli'):
                p += safe_sparse_dot(t, self.target_components_)
            elif (self.target_dist == 'gaussian'):
                p += safe_sparse_dot(t/self.target_sigma, self.target_components_)
            else:
                raise ValueError("Invalid target distribution: {}".format(self.target_dist))
        
        if (self.latent_dist == 'bernoulli'):
            p = expit(p, out=p)
        elif (self.latent_dist == 'multinomial'):
            p = softmax(p, axis=1)
        else:
            raise ValueError("Invalid latent distribution: {}".format(self.latent_dist))
        return p

    def _sample_hiddens(self, v, t, rng):
        """Sample from the distribution P(h|v).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer to sample from.

        rng : RandomState instance
            Random number generator to use.

        Returns
        -------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer.
        """
        p = self._mean_hiddens(v, t)
        if (self.latent_dist == 'bernoulli'):
            samples = rng.uniform(size=p.shape) < p
        elif (self.latent_dist == 'multinomial'):
            # Verify that rows are normalized
            assert np.allclose(p.sum(axis=1), 1), "Rows of p must be normalized, instead get {}".format(p.sum(axis=1))
            # samples = [rng.multinomial(self.sample_size, pval) for pval in p]
            samples = [np.bincount(np.random.choice(len(pval), size=self.sample_size, p=pval), minlength=len(pval)) for pval in p]

            samples = np.array(samples)
        else:
            raise ValueError("Invalid latent distribution: {}".format(self.latent_dist))
        return samples

    def _mean_visibles(self, h):
        """Computes the probabilities P(v=1|h).

        Parameters
        ----------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer.

        Returns
        -------
        v : ndarray of shape (n_samples, n_features)
            Corresponding mean field values for the visible layer.
        """
        p = np.dot(h, self.components_)
        if (self.input_dist == 'gaussian'):
            p *= self.sigma
            if (self.add_bias):
                p += self.intercept_visible_
        elif (self.input_dist == 'bernoulli'):
            if (self.add_bias):
                p += self.intercept_visible_
            p = expit(p, out=p)
        else:
            raise ValueError("Invalid input distribution: {}".format(self.input_dist))

        return p
    
    def _sample_visibles(self, h, rng, lower_bound=0, upper_bound=1):
        """Sample from the distribution P(v|h).

        Parameters
        ----------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        rng : RandomState instance
            Random number generator to use.

        Returns
        -------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = self._mean_visibles(h)
        if (self.input_dist == 'gaussian'):
            samples = rng.normal(p, self.sigma, size=p.shape)
        elif (self.input_dist == 'bernoulli'):
            samples = rng.uniform(size=p.shape) < p
        else:
            raise ValueError("Invalid input distribution: {}".format(self.input_dist))
        samples = np.clip(samples, lower_bound, upper_bound)
        return samples

    def _mean_targets(self, h):
        """Computes the probabilities P(t|h).

        Parameters
        ----------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer.

        Returns
        -------
        t : ndarray of shape (n_samples, n_targets)
            Corresponding mean field values for the target layer.
        """
        p = np.dot(h, self.target_components_.T)
        if (self.target_dist == 'gaussian'):
            p *= self.target_sigma
            if (self.add_bias):
                p += self.intercept_target_
        else:
            raise ValueError("Invalid target distribution: {}".format(self.target_dist))
        return p
    
    def _sample_targets(self, h, rng, lower_bound=0, upper_bound=1):
        """Sample from the distribution P(t|h).

        Parameters
        ----------
        h : ndarray of shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        rng : RandomState instance
            Random number generator to use.

        Returns
        -------
        t : ndarray of shape (n_samples, n_targets)
            Values of the target layer.
        """
        p = self._mean_targets(h)
        if (self.target_dist == 'gaussian'):
            samples = rng.normal(p, self.target_sigma, size=p.shape)
        else:
            raise ValueError("Invalid target distribution: {}".format(self.target_dist))
        samples = np.clip(samples, lower_bound, upper_bound)
        return samples
    
    def _free_energy(self, v, t, h):
        """Computes the free energy F(v) = - log sum_h exp(-E(v,h)).

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        free_energy : ndarray of shape (n_samples,)
            The value of the free energy.
        """
        if (self.add_bias and self.input_dist == 'gaussian'):
            input_energy = np.sum(((v - self.intercept_visible_) / self.sigma) ** 2, axis=1)/2 -np.sum((safe_sparse_dot(v/self.sigma, self.components_.T) + self.intercept_hidden_)*h, axis=1)
        elif (self.add_bias and self.input_dist == 'bernoulli'):
            input_energy = -safe_sparse_dot(v, self.intercept_visible_) - np.sum((safe_sparse_dot(v, self.components_.T) + self.intercept_hidden_)*h, axis=1)
        elif (self.input_dist == 'gaussian'):
            input_energy = np.sum((v / self.sigma) ** 2, axis=1)/2 - np.sum((safe_sparse_dot(v/self.sigma, self.components_.T))*h, axis=1)
        elif (self.input_dist == 'bernoulli'):
            input_energy = - np.sum(safe_sparse_dot(v, self.components_.T)*h, axis=1)
        else:
            raise ValueError("Invalid input distribution: {}".format(self.input_dist))
        
        if (self.target_in_model):
            if (self.add_bias and self.target_dist == 'gaussian'):
                target_energy = np.sum(((t - self.intercept_target_) / self.target_sigma) ** 2, axis=1)/2 - np.sum((safe_sparse_dot(t/self.target_sigma, self.target_components_) + self.intercept_hidden_)*h, axis=1)
            elif (self.target_dist == 'gaussian'):
                target_energy = np.sum((t / self.target_sigma) ** 2, axis=1)/2 - np.sum(safe_sparse_dot(t/self.target_sigma, self.target_components_)*h, axis=1)
            else:
                raise ValueError("Invalid target distribution: {}".format(self.target_dist))
        else:
            target_energy = np.zeros(v.shape[0], dtype=np.float32)
        return np.mean(input_energy + target_energy)

    def gibbs(self, v, t):
        """Perform one Gibbs sampling step.

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer to start from.

        Returns
        -------
        v_new : ndarray of shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step.
        """
        check_is_fitted(self)
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        h_ = self._sample_hiddens(v, t, self.random_state_)
        v_ = self._sample_visibles(h_, self.random_state_)
        t_ = self._sample_targets(h_, self.random_state_)

        return v_, t_

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        """Fit the model to the partial segment of the data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        first_pass = not hasattr(self, "components_")
        X = self._validate_data(
            X, accept_sparse="csr", dtype=np.float32, reset=first_pass
        )
        if not hasattr(self, "random_state_"):
            self.random_state_ = check_random_state(self.random_state)
        if not hasattr(self, "components_"):
            self.components_ = np.asarray(
                self.random_state_.normal(0, 0.01, (self.n_components, X.shape[1])),
                order="F",
            )
            self._n_features_out = self.components_.shape[0]
        if not hasattr(self, "intercept_hidden_"):
            self.intercept_hidden_ = np.zeros(
                self.n_components,
            )
        if not hasattr(self, "intercept_visible_"):
            self.intercept_visible_ = np.zeros(
                X.shape[1],
            )
        if not hasattr(self, "h_samples_"):
            self.h_samples_ = np.zeros((self.batch_size, self.n_components))

        self._fit(X, y, self.random_state_)

    def _fit(self, v_pos, t_pos, lr, rng):
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
        h_pos = self._mean_hiddens(v_pos, t_pos)
        v_neg = self._sample_visibles(self.h_samples_, rng)
        t_neg = self._sample_targets(self.h_samples_, rng)
        h_neg = self._mean_hiddens(v_neg, t_neg)
        

        update = safe_sparse_dot(v_pos.T, h_pos, dense_output=True).T
        update -= np.dot(h_neg.T, v_neg)
        self.components_ += lr * update * self.hybrid_alpha
        self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0)) * self.hybrid_alpha
        self.intercept_visible_ += lr * (
            np.asarray(v_pos.sum(axis=0)).squeeze() - v_neg.sum(axis=0)
        ) * self.hybrid_alpha

        if (self.target_in_model):
            target_lr = lr
            update_target = safe_sparse_dot(h_pos.T, t_pos/self.target_sigma, dense_output=True).T
            update_target -= np.dot(h_neg.T, t_neg/self.target_sigma).T
            self.target_components_ += target_lr * update_target * self.hybrid_alpha
            self.intercept_target_ += target_lr * (np.sum(t_pos/(self.target_sigma**2), axis=0) - np.sum(t_neg/(self.target_sigma**2), axis=0)) * self.hybrid_alpha
        
        if (self.hybrid):
            raise NotImplementedError("Hybrid training not implemented yet")
        
        if (self.latent_dist == 'multinomial'):
            assert np.allclose(h_neg.sum(axis=1), 1), "Rows of p must be normalized, instead get {}".format(h_neg.sum(axis=1))
            check_sum = np.sum(h_neg, axis=1)
            non_ones = check_sum[~np.isclose(check_sum, 1.0, atol=1e-6)]
            if non_ones.size > 0:
                print("Matrix contains non-1 values:", non_ones)
                raise ValueError("Matrix contains values other than 1!")
            self.h_samples_ = [rng.multinomial(self.sample_size, pval) for pval in h_neg]
            self.h_samples_ = np.array(self.h_samples_)
        elif (self.latent_dist == 'bernoulli'):
            h_neg[rng.uniform(size=h_neg.shape) < h_neg] = 1.0  # sample binomial
            self.h_samples_ = np.floor(h_neg, h_neg)
        else:
            raise ValueError("Invalid latent distribution: {}".format(self.latent_dist))

    def estimate_energy(self, X, y, rng, repeat=10):
        """Compute the pseudo-likelihood of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Values of the visible layer. Must be all-boolean (not checked).

        Returns
        -------
        pseudo_likelihood : ndarray of shape (n_samples,)
            Value of the pseudo-likelihood (proxy for likelihood).

        Notes
        -----
        This method is not deterministic: it computes a quantity called the
        free energy on X, then on a randomly corrupted version of X, and
        returns the log of the logistic function of the difference.
        """
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        y = self._validate_data(y, accept_sparse="csr", reset=False)
        energy = 0
        for _ in range(repeat):
            h = self._sample_hiddens(X, y, rng)
            energy += self._free_energy(X, y, h)
        return energy / repeat


    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_size=100, sigma = 0.1, target_sigma = 0.1, hybrid_alpha=1., showplot=False):
        """Fit the model to the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs).

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        X = self._validate_data(X, accept_sparse="csr", dtype=np.float32)
        y = self._validate_data(y, accept_sparse="csr", dtype=np.float32)
        n_samples = X.shape[0]
        
        rng = check_random_state(self.random_state)

        self.sample_size = sample_size
        self.components_ = np.asarray(
            rng.normal(0, 0.01, (self.n_components, X.shape[1])),
            order="F",
            dtype=X.dtype,
        )
        self.sigma = sigma
        self.target_sigma = target_sigma
        self.hybrid_alpha = hybrid_alpha
        self.target_components_ = np.asarray(
            rng.normal(0, 0.01, (y.shape[1], self.n_components)),
            order="F",
            dtype=y.dtype,
        )
        self.intercept_target_ = np.zeros(y.shape[1], dtype=y.dtype)
        self._n_features_out = self.components_.shape[0]
        self.intercept_hidden_ = np.zeros(self.n_components, dtype=X.dtype)
        self.intercept_visible_ = np.zeros(X.shape[1], dtype=X.dtype)
        self.h_samples_ = np.zeros((self.batch_size, self.n_components), dtype=X.dtype)

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(
            gen_even_slices(n_batches * self.batch_size, n_batches, n_samples=n_samples)
        )
        energy_mean_tracking = []
        energy_var_tracking = []
        for epoch in range(1, self.n_iter + 1):
            energy = []
            if (self.lr_decay):
                lr = float(self.learning_rate) * self.lr_decay_factor ** (epoch // self.lr_no_decay_length)
            else:
                lr = float(self.learning_rate)
            for batch_slice in batch_slices:
                self._fit(X[batch_slice], y[batch_slice], lr, rng)
                energy.append(self.estimate_energy(X, y, rng))
            energy_mean_tracking.append(np.mean(np.array(energy)))
            energy_var_tracking.append(np.var(np.array(energy)))
        
        x = np.array(list(range(1, len(energy_mean_tracking) + 1)))
        y = np.array(energy_mean_tracking)
        y_upper = y + np.array(energy_var_tracking)
        y_lower =y - np.array(energy_var_tracking)
        print(y)
        print(energy_var_tracking)
        plt.plot(x, y, label="Mean Line")
        plt.fill_between(x, y_lower, y_upper, alpha=0.2, label="Variance Range")
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("Energy vs Iteration")
        plt.legend()
        if (showplot):
            # Add a caption
            plt.figtext(0.5, 0.02, self.savefile + "energy.png", ha='center', fontsize=10, color='gray')

            # Show the plot
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(self.savefile + "energy.png")
        plt.close()

        return self

    def fit_dataloader(self, dataloader, v_dim, t_dim, components=None, target_components=None, visible_bias=None, hidden_bias=None, target_bias=None, sample_size=100, sigma=0.1, target_sigma=0.1, hybrid_alpha=1., showplot=False):
        """Fit the model to the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs).

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        rng = check_random_state(self.random_state)

        self.sample_size = sample_size
        self.hybrid_alpha = hybrid_alpha
        if (components is not None):
            self.components_ = components.cpu().numpy()
        else:
            self.components_ = np.asarray(
                rng.normal(0, 0.01, (self.n_components, v_dim)),
                order="F",
                dtype=np.float32,
            )
        self.sigma = sigma
        self.target_sigma = target_sigma
        if (target_components is not None):
            self.target_components_ = target_components.cpu().numpy()
        else:
            self.target_components_ = np.asarray(
                rng.normal(0, 0.01, (t_dim, self.n_components)),
                order="F",
                dtype=np.float32,
            )
        if (target_bias is not None):
            self.intercept_target_ = target_bias.cpu().numpy()
        else:
            self.intercept_target_ = np.zeros(t_dim, dtype=np.float32)
        self._n_features_out = self.components_.shape[0]
        if (hidden_bias is not None):
            self.intercept_hidden_ = hidden_bias.cpu().numpy()
        else:
            self.intercept_hidden_ = np.zeros(self.n_components, dtype=np.float32)
        if (visible_bias is not None):
            self.intercept_visible_ = visible_bias.cpu().numpy()
        else:
            self.intercept_visible_ = np.zeros(v_dim, dtype=np.float32)
        self.h_samples_ = np.zeros((dataloader.batch_size, self.n_components), dtype=np.float32)

        energy_mean_tracking = []
        energy_var_tracking = []
        for epoch in range(1, self.n_iter + 1):
            energy = []
            if (self.lr_decay):
                lr = float(self.learning_rate) * self.lr_decay_factor ** (epoch // self.lr_no_decay_length)
            else:
                lr = float(self.learning_rate)
            for X, y in dataloader:
                X = X.detach().cpu().numpy()
                y = y.detach().cpu().numpy().reshape(-1, t_dim)
                self._fit(X, y, lr, rng)
                energy.append(self.estimate_energy(X, y, rng))
            energy_mean_tracking.append(np.mean(np.array(energy)))
            energy_var_tracking.append(np.var(np.array(energy)))
        
        x = np.array(list(range(1, len(energy_mean_tracking) + 1)))
        y = np.array(energy_mean_tracking)
        y_upper = y + np.array(energy_var_tracking)
        y_lower =y - np.array(energy_var_tracking)
        # print(y)
        # print(energy_var_tracking)
        plt.plot(x, y, label="Mean Line")
        plt.fill_between(x, y_lower, y_upper, alpha=0.2, label="Variance Range")
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("Energy vs Iteration")
        plt.legend()
        if (showplot):
            # Add a caption
            plt.figtext(0.5, 0.02, self.savefile + "energy.png", ha='center', fontsize=10, color='gray')

            # Show the plot
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(self.savefile + "energy.png")
        plt.close()

        return self
