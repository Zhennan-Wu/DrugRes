from pyro_bernoulli_rbm import BernoulliRBM
import torch
import pyro
import pyro.distributions as dist


class BernoulliSoftmaxRBM(BernoulliRBM):
    def __init__(
        self,
        hidden_dim,
        learning_rate,
        grad_max=1e6,
        device=None
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            grad_max=grad_max,
            device=device
        )

    def _mean_hiddens(self, v):
        """Computes the universal softmax probabilities P(h|v)."""
        p = torch.softmax(torch.matmul(v, self.W) + self.b_h)
        return p

    def _sample_hiddens(self, v):
        """Sample one-hot vectors from the universal softmax distribution."""
        probs = self._mean_hiddens(v)
        _, num_classes = probs.shape

        # For each row, sample a single index according to probs
        cat_dist = dist.Categorical(probs=probs)
        sampled_indices = pyro.sample("sampled_indices", cat_dist)

        # Turn sampled indices into one-hot encoding
        one_hot = torch.nn.functional.one_hot(sampled_indices, num_classes=num_classes).float()

        return one_hot
