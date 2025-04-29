from pyro_bernoulli_rbm import BernoulliRBM
import torch
import pyro
import time
import pyro.distributions as dist


class MixedRBM(BernoulliRBM):
    def __init__(
        self,
        hidden_dim, 
        learning_rate, 
        sigma=0.3, 
        grad_max=1e6,
        device=None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            grad_max=grad_max,
            device=device
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
        sigma = torch.ones_like(v)
        sigma[:, self.gaussian_indices] = self.sigma
        v_normalized = v / sigma  # assumes self.sigma is either a vector or scalar
        p = torch.sigmoid(torch.matmul(v_normalized, self.W) +self.b_h)
        return p

    def _mean_visibles(self, h):
        """Compute mean of Gaussian visible units given hidden units."""
        return torch.matmul(h, self.W.T) + self.b_v

    def _sample_visibles(self, h):
        mean = self._mean_visibles(h)
        g_part = pyro.sample("v", dist.Normal(loc=mean, scale=self.sigma))
        b_part = pyro.sample("v", dist.Bernoulli(torch.clamp(mean, 0, 1)))
        samples = b_part.clone()
        samples[:, self.gaussian_indices] = g_part[:, self.gaussian_indices]
        return samples
    
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
        quadratic_term = 0.5 * torch.sum(((v[:, self.gaussian_indices] - self.b_v[self.gaussian_indices])/self.sigma) ** 2, dim=1)

        
        # Hidden unit activation input
        g_hidden_input = torch.matmul(v[:, self.gaussian_indices]/self.sigma, self.W[self.gaussian_indices, :]) + self.b_h

        # Explicit debug inspection
        # print("hidden Any NaNs?", np.isnan(hidden_input).any())
        # print("hidden Any Infs?", np.isinf(hidden_input).any())
        # print("hidden Max value:", np.nanmax(hidden_input))
        # print("hidden Min value:", np.nanmin(hidden_input))

        # Log-sum-exp over hidden units
        g_hidden_term = torch.logaddexp(torch.zeros_like(g_hidden_input), g_hidden_input).sum(dim=1)
        
        b_free_energy = -torch.sum(torch.log(1 + torch.exp(torch.matmul(v[:, self.bernoulli_indices], self.W[self.bernoulli_indices, :])))) - torch.sum(v[:, self.bernoulli_indices] * self.b_v[self.bernoulli_indices])
        return quadratic_term - g_hidden_term + b_free_energy
    
    def _fit(self, v_pos):
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
        v_neg = self._sample_visibles(self.h_samples_)
        h_neg = self._mean_hiddens(v_neg)

        sigma = torch.ones_like(v_pos)
        sigma[:, self.gaussian_indices] = self.sigma
        v_pos_normalized = v_pos/sigma
        v_neg_normalized = v_neg/sigma

        update = (torch.matmul(v_pos_normalized.T, h_pos).T - torch.matmul(h_neg.T, v_neg_normalized))/v_pos.shape[0]
        update = torch.clamp(update, -self.grad_max, self.grad_max)
        # print(f"Gradient max: {update.max()}")
        # print(f"Gradient min {update.min()}")
        # print(f"Gradient mean, {update.mean()}")
        self.W += self.learning_rate * update
        # self.intercept_hidden_ += lr * (h_pos.sum(axis=0) - h_neg.sum(axis=0))
        # self.intercept_visible_ += lr * (
        #     np.asarray(v_pos.sum(axis=0)).squeeze() - v_neg.sum(axis=0)
        # )
        self.b_v += self.learning_rate * (v_pos_normalized.mean(0) - v_neg_normalized.mean(0))
        self.b_h += self.learning_rate * (h_pos.mean(0) - h_neg.mean(0))
        self.h_samples_ = pyro.sample("h_neg", dist.Bernoulli(probs=h_neg))

    def score_samples(self, v):
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

        # Randomly pick one dimension per sample to corrupt
        batch_size, data_dim = v.shape
        # Random indices for corruption: shape = (batch_size,)
        random_row_indices = torch.arange(batch_size, device=self.device)
        random_col_indices = torch.randint(0, len(self.gaussian_indices), (batch_size,), device=self.device)
        gaussian_cols = torch.tensor(self.gaussian_indices)[random_col_indices]

        # Create a corrupted version v_
        v_corrupted = v.clone()

        # Generate Gaussian noise
        noise_std = getattr(self, 'corruption_std', 0.1)
        noise = torch.randn(batch_size, device=self.device) * noise_std

        # Apply noise to two features per sample
        v_corrupted[random_row_indices, gaussian_cols] += noise

        random_col_indices = torch.randint(0, len(self.bernoulli_indices), (batch_size,), device=self.device)
        bernoulli_cols = torch.tensor(self.bernoulli_indices)[random_col_indices]

        # Flip the selected bits
        v_corrupted[torch.arange(batch_size), bernoulli_cols] = 1 - v_corrupted[torch.arange(batch_size), bernoulli_cols]

        fe = self._free_energy(v)
        fe_corrupted = self._free_energy(v_corrupted)

        # Compute pseudo-likelihood
        logits = fe_corrupted - fe
        pseudo_likelihood = -data_dim * torch.nn.functional.softplus(-logits)

        return pseudo_likelihood

    def fit(self, X, epochs, gaussian_indices, verbose=0, learning_rate=None):
        """
        Fit the RBM model to the data.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, visible_dim).
        y : None
            Not used.
        
        Returns
        -------
        self : BernoulliRBM
            Fitted RBM model.
        """
        X = X.to(self.device)
        self.batch_size = X.shape[0]
        self.visible_dim = X.shape[1]
        self.n_iter = epochs

        self.W = torch.randn(self.visible_dim, self.hidden_dim, dtype=X.dtype, device=self.device)
        self.b_v = torch.zeros(self.visible_dim, dtype=X.dtype, device=self.device)
        self.b_h = torch.zeros(self.hidden_dim, dtype=X.dtype, device=self.device)
        self.h_samples_ = torch.zeros((self.batch_size, self.hidden_dim), dtype=X.dtype).to(self.device)
        if learning_rate is not None:
            self.learning_rate = learning_rate
                # Convert to tensor if needed
        
        self.gaussian_indices = gaussian_indices
        indices_tensor = torch.tensor(self.gaussian_indices, device=X.device)

        # Get the complementary indices
        all_indices = torch.arange(X.size(1), device=X.device)
        self.bernoulli_indices = all_indices[~torch.isin(all_indices, indices_tensor)]

        begin = time.time()
        for epoch in range(1, self.n_iter+1):
            self._fit(X)
            if verbose:
                end = time.time()
                print(
                    "[%s] Iteration %d, pseudo-likelihood = %.2f, time = %.2fs"
                    % (
                        type(self).__name__,
                        epoch,
                        self.score_samples(X).mean(),
                        end - begin,
                    )
                )
                begin = end

        return self

    def batch_fit(self, data_loader, epochs, gaussian_indices, verbose=0, learning_rate=None):
        """
        Fit the RBM model to the data.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, visible_dim).
        y : None
            Not used.
        
        Returns
        -------
        self : BernoulliRBM
            Fitted RBM model.
        """
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                X = batch[0]  # if dataset returns (data, label)
            else:
                X = batch      # if dataset returns only data
            break  # only need the first batch
        self.batch_size = X.shape[0]
        self.visible_dim = X.shape[1]
        self.n_iter = epochs

        self.W = torch.randn(self.visible_dim, self.hidden_dim, dtype=X.dtype, device=self.device)
        self.b_v = torch.zeros(self.visible_dim, dtype=X.dtype, device=self.device)
        self.b_h = torch.zeros(self.hidden_dim, dtype=X.dtype, device=self.device)
        self.h_samples_ = torch.zeros((self.batch_size, self.hidden_dim), dtype=X.dtype).to(self.device)
        if learning_rate is not None:
            self.learning_rate = learning_rate

        self.gaussian_indices = gaussian_indices
        indices_tensor = torch.tensor(self.gaussian_indices, device=X.device)

        # Get the complementary indices
        all_indices = torch.arange(X.size(1), device=X.device)
        self.bernoulli_indices = all_indices[~torch.isin(all_indices, indices_tensor)]

        begin = time.time()
        for epoch in range(1, self.n_iter+1):
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    X = batch[0]  # if dataset returns (data, label)
                else:
                    X = batch      # if dataset returns only data
                X = X.to(self.device)
                self._fit(X)
                
            if verbose:
                end = time.time()
                print(
                    "[%s] Iteration %d, pseudo-likelihood = %.2f, time = %.2fs"
                    % (
                        type(self).__name__,
                        epoch,
                        self.score_samples(X).mean(),
                        end - begin,
                    )
                )
                begin = end
                
        return self