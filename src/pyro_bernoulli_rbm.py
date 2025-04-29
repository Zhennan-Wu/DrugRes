import time
import torch
import pyro
import pyro.distributions as dist


class BernoulliRBM:
    def __init__(self, hidden_dim, learning_rate, grad_max = 1e6, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.grad_max = grad_max

    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_dim={self.hidden_dim}, learning_rate={self.learning_rate})"

    def transform(self, X):
        """
        Transform the input data X into hidden representations.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_samples, visible_dim).
        
        Returns
        -------
        h : torch.Tensor
            Transformed hidden representations of shape (n_samples, hidden_dim).
        """
        X = X.to(self.device)
        h = self._mean_hiddens(X)
        return h

    def _mean_hiddens(self, v):
        """Computes the probabilities P(h=1|v)."""
        # Compute the activation of hidden units given visible units
        p = torch.sigmoid(torch.matmul(v, self.W) + self.b_h)
        return p
        
    def _sample_hiddens(self, v):
        """Sample hidden units from the Bernoulli distribution."""
        h = self._mean_hiddens(v)
        return pyro.sample("h", dist.Bernoulli(h))
    
    def _sample_visibles(self, h):
        """Sample visible units from the Bernoulli distribution."""
        v = torch.sigmoid(torch.matmul(h, self.W.T) + self.b_v)
        return pyro.sample("v", dist.Bernoulli(v))
    
    def _free_energy(self, v):
        """Computes the free energy for Bernoulli RBM."""
        free_energy = -torch.sum(torch.log(1 + torch.exp(torch.matmul(v, self.W) + self.b_h))) - torch.sum(v * self.b_v)
        return free_energy
    
    def gibbs(self, v):
        """Perform one step of Gibbs sampling."""
        h = self._sample_hiddens(v)
        v_new = self._sample_visibles(h)
        return v_new
    
    def partial_fit(self, v, learning_rate=None):
        """
        Perform one step of contrastive divergence.
        
        Parameters
        ----------
        v : torch.Tensor
            Input data of shape (n_samples, visible_dim).
        y : None
            Not used.
        
        Returns
        -------
        self : BernoulliRBM
            Fitted RBM model.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        v = v.to(self.device)

        self._fit(v)
        return self

    def _fit(self, v_pos):
        """Fit the RBM model to the data."""
        h_pos = self._mean_hiddens(v_pos)
        v_neg = self._sample_visibles(self.h_samples_)
        h_neg = self._mean_hiddens(v_neg)

        update = (torch.matmul(v_pos.T, h_pos) - torch.matmul(v_neg.T, h_neg))/v_pos.shape[0]
        update = torch.clamp(update, -self.grad_max, self.grad_max)
        self.W += self.learning_rate * update
        self.b_v += self.learning_rate * (v_pos.mean(0) - v_neg.mean(0))
        self.b_h += self.learning_rate * (h_pos.mean(0) - h_neg.mean(0))
        self.h_samples_ = pyro.sample("h_neg", dist.Bernoulli(probs=h_neg))

    def score_samples(self, v):
        """
        Compute pseudo-likelihood for batch of visible samples v.

        Args:
            v: torch.Tensor, shape (batch_size, data_dim)
            Visible data (assumed binary 0/1).
        Returns:
            torch.Tensor of shape (batch_size,)
        """

        batch_size, data_dim = v.shape

        # Randomly select one feature per sample to flip
        random_indices = torch.randint(0, data_dim, (batch_size,), device=self.device)

        # Create a corrupted version v_
        v_corrupted = v.clone()

        # Flip the selected bits
        v_corrupted[torch.arange(batch_size), random_indices] = 1 - v_corrupted[torch.arange(batch_size), random_indices]

        # Compute free energies
        fe = self._free_energy(v)           # (batch_size,)
        fe_corrupted = self._free_energy(v_corrupted)  # (batch_size,)

        # Compute pseudo-likelihood
        logits = fe_corrupted - fe
        pseudo_likelihood = -data_dim * torch.nn.functional.softplus(-logits)

        return pseudo_likelihood

    def fit(self, X, epochs, verbose=0, learning_rate=None):
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

    def batch_fit(self, data_loader, epochs, verbose=0, learning_rate=None):
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

