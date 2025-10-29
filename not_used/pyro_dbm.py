import torch
import pyro
import pyro.distributions as dist


class DBM:

    def __init__(self, dbn_pipeline, learning_rate=0.06, grad_max=1e6, sigma = 0.3, device=None):
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.W = [step[1].W.to(self.device) for step in dbn_pipeline]
        self.n_layers = len(self.W)

        self.b_h = []
        layer_hidden_biases = [step[1].b_h for step in dbn_pipeline]
        layer_visible_biases = [step[1].b_v for step in dbn_pipeline]
        layer_visible_biases.append(layer_hidden_biases[-1])
        for layer in range(self.n_layers):
            self.b_h.append((layer_hidden_biases[layer] + layer_visible_biases[layer+1])/2.)
        self.b_v = layer_visible_biases[0]
        
        self.layer_types = [step[0] for step in dbn_pipeline]
        self.grad_max = grad_max
        self.learning_rate = learning_rate
        self.sigma = sigma

    def __repr__(self):
        return f"{self.__class__.__name__}(depth={self.n_layers}, learning_rate={self.learning_rate})"
    
    def _mean_intermeidate(self, layer):
        if (layer == 0):
            raise ValueError("To sample observation, please use the sample_observation method.")
        elif (layer == self.n_layers):
            raise ValueError("To sample observation, please use the sample_latent method.")
        
        W_lower = self.W[layer-1]
        state_lower = self.markov_samples[layer-1]
        layer_lower = self.layer_types[layer-1]

        W_upper = self.W[layer]
        state_upper = self.markov_samples[layer+1]
        layer_upper = self.layer_types[layer]

        bias = self.b_h[layer]

        if "gaussian" in layer_lower:
            sigma = torch.ones_like(state_lower)*self.sigma
            p = torch.matmul(state_lower/sigma, W_lower) + bias
        elif "mixed" in layer_lower:
            sigma = torch.ones_like(state_lower)
            sigma[:, self.gaussian_indices] = self.sigma
            p = torch.matmul(state_lower/sigma, W_lower) + bias        
        else:
            p = torch.matmul(state_lower, W_lower) + bias

        if "softmax" in layer_upper:
            # binarized mean or simple sum, this is a question
            p += torch.matmul(torch.sum(state_upper, dim=1), W_upper.T)
        else:
            p += torch.matmul(state_upper, W_upper.T)
        
        p = torch.sigmoid(p)
        return p
    
    def _sample_intermeidate(self, layer):
        p = self._mean_intermeidate(layer)
        self.markov_samples[layer] = pyro.sample(f"h_{layer}", dist.Bernoulli(p))
    
    def _sample_latent(self):
        W_lower = self.W[-1]
        state_lower = self.markov_samples[-2]
        layer_lower = self.layer_types[-1]
        bias = self.b_h[-1]

        if "gaussian" in layer_lower:
            sigma = torch.ones_like(state_lower)*self.sigma
            p = torch.matmul(state_lower/sigma, W_lower) + bias
        elif "mixed" in layer_lower:
            sigma = torch.ones_like(state_lower)
            sigma[:, self.gaussian_indices] = self.sigma
            p = torch.matmul(state_lower/sigma, W_lower) + bias           
        else:
            p = torch.matmul(state_lower, W_lower) + bias
        
        if "softmax" in layer_lower:
            p = torch.softmax(p, dim=1)
            batch_size, num_classes = p.shape
            M = 10  # Number of samples per row

            # Expand p to shape [batch_size, M, num_classes]
            p_expanded = p.unsqueeze(1).expand(batch_size, M, num_classes)

            # Reshape to [batch_size * M, num_classes] for sampling
            p_reshaped = p_expanded.reshape(-1, num_classes)

            # Sample from categorical
            cat_dist = dist.Categorical(probs=p_reshaped)
            sampled_indices = pyro.sample("sampled_indices", cat_dist)

            # Convert to one-hot: shape [batch_size * M, num_classes]
            one_hot_flat = torch.nn.functional.one_hot(sampled_indices, num_classes=num_classes).float()

            # Reshape back to [batch_size, M, num_classes]
            one_hot = one_hot_flat.view(batch_size, M, num_classes)
            self.markov_samples[-1] = one_hot
        else:
            p = torch.sigmoid(p)
            self.markov_samples[-1] = pyro.sample(f"h_{self.n_layers}", dist.Bernoulli(p))
    
    def _sample_observation(self):
        W_upper = self.W[0]
        state_upper = self.markov_samples[1]
        layer_upper = self.layer_types[0]
        bias = self.b_v

        if "gaussian" in layer_upper:
            p = (torch.matmul(state_upper, W_upper.T) + bias)*self.sigma
            self.markov_samples[0] = pyro.sample(f"v", dist.Normal(p, self.sigma))
        elif "mixed" in layer_upper:
            p = torch.matmul(state_upper, W_upper.T) + bias
            g_part = pyro.sample("v", dist.Normal(loc=p*self.sigma, scale=self.sigma))
            b_part = pyro.sample("v", dist.Bernoulli(torch.clamp(p, 0, 1)))
            b_part[:, self.gaussian_indices] = g_part[:, self.gaussian_indices]
            self.markov_samples[0] = b_part
        else:
            p = torch.sigmoid(torch.matmul(state_upper, W_upper.T) + bias)
            self.markov_samples[0] = pyro.sample(f"v", dist.Bernoulli(p))
    
    def _mean_field_hidden(self, layer):
        if (layer == 0):
            raise ValueError("No need to approximate observation distribution.")          
        elif (layer == self.n_layers):
            W_lower = self.W[layer-1]
            state_lower = self.mu[layer-1]
            layer_lower = self.layer_types[layer-1]

            W_upper = torch.zeros_like(W_lower.T)
            state_upper = torch.zeros_like(state_lower)

            bias = self.b_h[layer]
        else:
            W_lower = self.W[layer-1]
            state_lower = self.mu[layer-1]
            layer_lower = self.layer_types[layer-1]

            W_upper = self.W[layer]
            state_upper = self.mu[layer+1]

            bias = self.b_h[layer]

        if "gaussian" in layer_lower:
            sigma = torch.ones_like(state_lower)*self.sigma
            p = torch.matmul(state_lower/sigma, W_lower) + bias
        elif "mixed" in layer_lower:
            sigma = torch.ones_like(state_lower)
            sigma[:, self.gaussian_indices] = self.sigma
            p = torch.matmul(state_lower/sigma, W_lower) + bias        
        else:
            p = torch.matmul(state_lower, W_lower) + bias

        p += torch.matmul(state_upper, W_upper.T)
        p = torch.sigmoid(p)
        return p
    
    def mean_field_update(self, data, max_iter=30, threshold=0.1):
        self.mu = [data]
        for layer in range(self.n_layers):
            self.mu.append(torch.rand((data.shape[0], self.W[layer].shape[1])).to(self.device))

        consequtive_convergence = 0
        for _ in range(max_iter):
            diffs = []
            for layer in range(1, self.n_layers+1):
                p = self._mean_field_hidden(layer)
                diffs.append(torch.mean(torch.abs(p - self.mu[layer])))
            if (max(diffs) < threshold):
                consequtive_convergence += 1
            else:
                consequtive_convergence = 0
            
            if (consequtive_convergence > 5):
                break
        if (consequtive_convergence):
            print(f"Converged after {consequtive_convergence} iterations.")
        else:
            print(f"Did not converge after {max_iter} iterations.")
    
    def gibbs_step(self):
        for layer in range(self.n_layers+1):
            if (layer == 0):
                self._sample_observation()
            elif (layer == self.n_layers):
                self._sample_latent()
            else:
                self._sample_intermeidate(layer)
    
    def gibbs_update(self, data, max_iter):
        self.markov_samples = [data]
        for layer in range(self.n_layers):
            self.markov_samples.append(torch.sigmoid(torch.matmul(self.markov_samples[-1], self.W[layer]) + self.b_h[layer]))

        for _ in range(max_iter):
            self.gibbs_step()

    def _fit(self, data, lr, gibbs_iter, mf_iter, mf_thres):
        self.mean_field_update(data, mf_iter, mf_thres)
        self.gibbs_update(data, gibbs_iter)

        for layer in range(self.n_layers):
            update = (torch.matmul(self.mu[layer].T, self.mu[layer+1]) - torch.matmul(self.markov_samples[layer].T, self.markov_samples[layer+1]))/self.batch_size
            update = torch.clamp(update, -self.grad_max, self.grad_max)
            self.W[layer] += lr * update
    
            self.b_h[layer] += lr * (self.mu[layer+1].mean(0) - self.markov_samples[layer+1].mean(0))
        self.b_v += lr * (self.mu[0].mean(0) - self.markov_samples[0].mean(0))
    
    def partial_fit(self, data, lr, gibbs_iter, mf_iter, mf_thres):
        """
        Perform one step of contrastive divergence.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data of shape (n_samples, visible_dim).
        y : None
            Not used.
        
        Returns
        -------
        self : DBM
            Fitted DBM model.
        """
        data = data.to(self.device)
        self._fit(data, lr, gibbs_iter, mf_iter, mf_thres)

    def fit(self, data, epochs, gibbs_iter=10, mf_iter=30, mf_thres=0.1, learning_rate=None, gaussian_indice=None):
        """
        Fit the DBM model to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data of shape (n_samples, visible_dim).
        epochs : int
            Number of training epochs.
        gibbs_iter : int
            Number of Gibbs sampling iterations.
        mf_iter : int
            Number of mean field iterations.
        mf_thres : float
            Convergence threshold for mean field iterations.
        
        Returns
        -------
        self : DBM
            Fitted DBM model.
        """
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gaussian_indice is not None:
            self.gaussian_indices = gaussian_indice
        data = data.to(self.device)
        self.batch_size = data.shape[0]
        self.n_iter = epochs

        for epoch in range(1, self.n_iter+1):
            lr = self.learning_rate * (1 - (epoch-1)/self.n_iter)
            self._fit(data, lr, gibbs_iter, mf_iter, mf_thres)
        
        return self
    
    def batch_fit(self, data_loader, epochs, gibbs_iter=10, mf_iter=30, mf_thres=0.1, learning_rate=None, gaussian_indice=None):
        """
        Fit the DBM model to the data.
        
        Parameters
        ----------
        data : torch.Tensor
            Input data of shape (n_samples, visible_dim).
        epochs : int
            Number of training epochs.
        gibbs_iter : int
            Number of Gibbs sampling iterations.
        mf_iter : int
            Number of mean field iterations.
        mf_thres : float
            Convergence threshold for mean field iterations.
        
        Returns
        -------
        self : DBM
            Fitted DBM model.
        """
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                X = batch[0]  # if dataset returns (data, label)
            else:
                X = batch      # if dataset returns only data
            break  # only need the first batch
        self.batch_size = X.shape[0]

        if learning_rate is not None:
            self.learning_rate = learning_rate
        if gaussian_indice is not None:
            self.gaussian_indices = gaussian_indice
        self.n_iter = epochs

        for epoch in range(1, self.n_iter+1):
            lr = self.learning_rate * (1 - (epoch-1)/self.n_iter)
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    X = batch[0]  # if dataset returns (data, label)
                else:
                    X = batch      # if dataset returns only data
                X = X.to(self.device) 
                self._fit(X, lr, gibbs_iter, mf_iter, mf_thres)
        return self       

    def transform(self, data):
        data = data.to(self.device)
        self.mean_field_update(data)
        return self.mu[-1]