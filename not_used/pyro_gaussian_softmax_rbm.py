from pyro_gaussian_rbm import GaussianRBM
import torch
import pyro
import pyro.distributions as dist


class GaussianSoftmaxRBM(GaussianRBM):
    def __init__(
        self,
        hidden_dim,
        learning_rate=0.1,
        sigma=0.3,
        grad_max=1e6,
        M=10,
        device=None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            sigma=sigma,
            grad_max=grad_max,
            device=device
        )
        self.M = M

    def _mean_hiddens(self, v):
        """Computes the universal softmax probabilities P(h|v)."""
        v_normalized = v / self.sigma
        p = torch.softmax(torch.matmul(v_normalized, self.W) +self.b_h)
        return p

    def _sample_hiddens(self, v):
        """Sample one-hot vectors from the universal softmax distribution."""
        p = self._mean_hiddens(v)
        batch_size, num_classes = p.shape
        
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
        one_hot = one_hot_flat.view(batch_size, self.M, num_classes)

        one_hot = torch.sum(one_hot, dim=1)
        return one_hot