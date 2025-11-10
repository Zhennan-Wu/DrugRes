import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal, InverseGamma, Multinomial, Categorical, Beta

from tqdm import trange
import copy
import math

from hdmm_utils_torch import mix_weights, suffix_sum, get_unique_rows_and_positions, advanced_multi_index_select, safe_update_scatter, stats_by_label, safe_positive, rand_uniform
from vis import likelihood_visualization


def truncated_stick_breaking(param_alpha: torch.Tensor, param_beta: torch.Tensor, sample_shape: tuple, truncate_dim: int = -1) -> torch.Tensor:
    """
    Truncated stick-breaking process to generate mixture weights.

    Args:
        param_alpha: Tensor of alpha parameters for Beta distributions.
        param_beta: Tensor of beta parameters for Beta distributions.
        sample_shape: Shape of the samples to draw.
        truncate_dim: Dimension along which to truncate the stick-breaking.
    Returns:
        Tensor of mixture weights with last weight set to 1.
    """
    beta_samples = Beta(safe_positive(param_alpha), safe_positive(param_beta)).sample(sample_shape)
    assert beta_samples.shape == sample_shape + param_alpha.shape
    if truncate_dim == -1:
        beta_samples = torch.cat((beta_samples[..., :-1], torch.ones_like(beta_samples[..., -1:])), dim=-1)  # last stick = 1
        weight = mix_weights(beta_samples, axis=-1)
        assert weight.shape == beta_samples.shape
        assert torch.allclose(weight.sum(dim=-1), torch.ones(weight.shape[:-1], device=weight.device))
    elif truncate_dim == 0:
        beta_samples = torch.cat((beta_samples[:-1], torch.ones_like(beta_samples[-1:])), dim=0)  # last stick = 1
        weight = mix_weights(beta_samples, axis=0)
        assert weight.shape == beta_samples.shape
        assert torch.allclose(weight.sum(dim=0), torch.ones(weight.shape[1:], device=weight.device))

    return weight


def gen_next_level_prior(G_parent, alpha_param):
    param_alpha = alpha_param * G_parent
    param_beta = suffix_sum(param_alpha)

    return [param_alpha, param_beta]


def dirichlet_posterior(
    obs: torch.Tensor, z_gen: torch.Tensor,
    params: torch.Tensor,
    num_components: int,
    scaling_constant: float = 1.0,
) -> torch.Tensor:
    """
    Batched Dirichlet posterior sampling.

    Args:
        obs: Tensor of shape (B, N_obs, V) or (N_obs, V)
              Assigned one-hot or count observations.
        params: Tensor of shape (V,) or (B, V)
              Dirichlet prior concentration parameters.
        scaling_constant: Float to scale counts before adding to params.

    Returns:
        sample: Tensor of shape (B, V)
              Samples from the Dirichlet posterior per batch.
    """
    data_stats = stats_by_label(obs.reshape(-1, obs.shape[-1]), z_gen.flatten(), num_components) 
    value = data_stats[-1]  # (K, V) or (V,)

    # Broadcast params if needed
    if params.dim() == 1:
        params = params.unsqueeze(0).expand(num_components, -1)  # (K, V)
    if num_components == 1:
        assert value.dim() == 1
        value = value.unsqueeze(0)  # (1, V)

    assert value.shape == params.shape
    # Posterior concentration parameters
    new_params = params + value * scaling_constant

    # Sample from Dirichlet posterior for each batch
    dist = Dirichlet(new_params)
    sample = dist.sample()  # (K, V)

    return sample, new_params


def nig_posterior(reg: torch.Tensor, 
                  z_reg: torch.Tensor,
                  num_components: int,
                  params: list,
                  scale_constant: float = 1.0):
    """
    Sample regression component parameters given assigned observations and Normal-Inverse-Gamma prior.

    Args:
        obs: (N_obs,) tensor of assigned regression observations
        params: list of four scalars [mu, kappa, alpha, beta] for the NIG prior
        scale_constant: float to scale the counts before updating parameters
        generator: optional torch.Generator for reproducible sampling

    Returns:
        new_mu: sampled mean parameter (scalar tensor)
        new_sigma: sampled variance parameter (scalar tensor)
    """
    # Ensure reg is a 1D float tensor
    means, _, sum_vars, counts, _ = stats_by_label(reg.flatten(), z_reg.flatten(), num_components)
    means = means.squeeze()

    assert means.shape == (num_components,)
    assert sum_vars.shape == (num_components,)
    assert counts.shape == (num_components,)

    mu0, kappa0, alpha0, beta0 = [torch.as_tensor(p, dtype=torch.float32) for p in params]
    assert mu0.shape == (num_components,)
    assert kappa0.shape == (num_components,)
    assert alpha0.shape == (num_components,)
    assert beta0.shape == (num_components,)

    # Posterior updates
    kappa_n = kappa0 + counts * scale_constant
    mu_n = (kappa0 * mu0 + counts * scale_constant * means) / kappa_n
    alpha_n = alpha0 + counts * scale_constant / 2.0
    beta_n = beta0 + 0.5 * scale_constant * sum_vars + \
             (kappa0 * counts * scale_constant * (means - mu0) ** 2) / (2.0 * kappa_n)

    # Sample from the posterior
    assert kappa_n.shape == (num_components,)
    assert alpha_n.shape == (num_components,)
    assert beta_n.shape == (num_components,)
    assert mu_n.shape == (num_components,)

    sigma_sample = InverseGamma(alpha_n, beta_n).sample()
    mu_sample = Normal(mu_n, torch.sqrt(sigma_sample / kappa_n)).sample()

    assert mu_sample.shape == (num_components,)
    assert sigma_sample.shape == (num_components,)

    return mu_sample, sigma_sample, mu_n, kappa_n, alpha_n, beta_n


class HDMM(nn.Module):
    def __init__(self, struct_upbd, *args, **kwargs):
        super().__init__()
        torch.set_grad_enabled(False)
        torch.set_default_dtype(torch.float32)

        self.struct_upbd = struct_upbd
        self.K = int(struct_upbd["G0"])
        self.param_dims = list(struct_upbd.values())[::-1]
        self.cluster_dims = self.param_dims[:-1][::-1]

        self.vocab_size = kwargs.get("vocab_size", 10000)
        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = kwargs.get("seed", 0)
        torch.manual_seed(self.seed)

        self.best_log_prob = -torch.inf
        
        self.struct_params = nn.ParameterDict()
        self.init_tunable_hyperparameters()
        self.init_structure()
        self.init_mixture_components()

    @torch.no_grad()
    def init_tunable_hyperparameters(self):
        """
        Initialize tunable hyperparameters in PyTorch version.
        Each parameter is registered as nn.Parameter so they become trainable.
        """

        # Core scalar hyperparameters
        self.struct_params["gamma"] = nn.Parameter(rand_uniform((), 0.1, 100.0)).to(self.device)

        # Hierarchical alpha/eta initialization
        for depth in range(len(self.param_dims)):
            child_level = depth + 1

            # α parameter
            self.struct_params[f"alpha{depth}"] = nn.Parameter(rand_uniform(tuple(self.param_dims[-child_level:]), 10.0, 100.0)).to(self.device)

            # η parameter
            if depth < len(self.param_dims) - 1:
                self.struct_params[f"eta{depth}"] = nn.Parameter(rand_uniform(tuple(self.param_dims[-child_level:-1]), 0.1, 1.0)).to(self.device)

    @torch.no_grad()
    def init_structure(self):
        """
        Initialize hierarchical structure variables in PyTorch version.
        Sets up global sticks (G) and local sticks (LG) along with their Beta parameters (P, LP).
        """
        self.SV = {}
        self.best_SV = {}

        self._setup_mixture_params()

        for depth in range(len(self.param_dims)):
            # ----------------------------------------------
            # Hierarchical structure levels
            # ----------------------------------------------
            if (depth == 0):
                param_alpha = torch.tensor(1.0).to(self.device)
                param_beta = self.struct_params["gamma"] 
            else:               
                param_alpha = self.struct_params[f"alpha{depth-1}"] * self.SV[f"G{depth-1}"]
                param_beta = suffix_sum(param_alpha)
                assert param_alpha.shape == tuple(self.param_dims[-depth:])
            self._setup_struct_values(depth, param_alpha, param_beta)

            # ----------------------------------------------
            # Cluster-specific local weights (η)
            # ----------------------------------------------
            if depth < len(self.param_dims) - 1:
                eta = self.struct_params[f"eta{depth}"]
                a = torch.ones_like(eta)
                self._setup_cluster_values(depth, a, eta)

    @torch.no_grad()
    def _setup_mixture_params(self):
        """
        Initialize mixture model parameters in PyTorch version.
        """
        self.SV["dir_alpha"] = rand_uniform((self.vocab_size,), 0.1, 1.0).unsqueeze(0).expand(self.K, -1).to(self.device)
        self.SV["nig_mu"] = rand_uniform((), 0.1, 100.0).expand(self.K).to(self.device)
        self.SV["nig_kappa"] = rand_uniform((), 0.1, 100.0).expand(self.K).to(self.device)
        self.SV["nig_alpha"] = rand_uniform((), 0.1, 100.0).expand(self.K).to(self.device)
        self.SV["nig_beta"] = rand_uniform((), 0.1, 100.0).expand(self.K).to(self.device)

        self.SV["dir_alpha_Pos"] = self.SV["dir_alpha"].detach().clone()
        self.SV["nig_mu_Pos"] = self.SV["nig_mu"].detach().clone()
        self.SV["nig_kappa_Pos"] = self.SV["nig_kappa"].detach().clone()
        self.SV["nig_alpha_Pos"] = self.SV["nig_alpha"].detach().clone()
        self.SV["nig_beta_Pos"] = self.SV["nig_beta"].detach().clone()

        self.best_SV["dir_alpha"] = self.SV["dir_alpha"].detach().clone()
        self.best_SV["nig_mu"] = self.SV["nig_mu"].detach().clone()
        self.best_SV["nig_kappa"] = self.SV["nig_kappa"].detach().clone()
        self.best_SV["nig_alpha"] = self.SV["nig_alpha"].detach().clone()
        self.best_SV["nig_beta"] = self.SV["nig_beta"].detach().clone()

        self.best_SV["dir_alpha_Pos"] = self.SV["dir_alpha"].detach().clone()
        self.best_SV["nig_mu_Pos"] = self.SV["nig_mu"].detach().clone()
        self.best_SV["nig_kappa_Pos"] = self.SV["nig_kappa"].detach().clone()
        self.best_SV["nig_alpha_Pos"] = self.SV["nig_alpha"].detach().clone()
        self.best_SV["nig_beta_Pos"] = self.SV["nig_beta"].detach().clone()

    @torch.no_grad()
    def _setup_struct_values(self, depth, param_alpha, param_beta):
        """
        Setup global structure variables at a given depth.
        """
        self.SV[f"G{depth}"] = truncated_stick_breaking(param_alpha, param_beta, sample_shape=(self.param_dims[-(depth+1)],), truncate_dim=-1)
        assert self.SV[f"G{depth}"].shape == tuple(self.param_dims[-(depth+1):])
        assert self.SV[f"G{depth}"].sum(dim=-1).allclose(torch.ones(self.SV[f"G{depth}"].shape[:-1], device=self.device))
        self.SV[f"P{depth}"] = [param_alpha.unsqueeze(0).expand(tuple(self.param_dims[-(depth+1):])), param_beta.unsqueeze(0).expand(tuple(self.param_dims[-(depth+1):]))]  
        assert self.SV[f"P{depth}"][0].shape == tuple(self.param_dims[-(depth+1):])
        assert self.SV[f"P{depth}"][1].shape == tuple(self.param_dims[-(depth+1):])
        # save posterior structure variables of a iteration for potential best structure
        self.SV[f"Posterior{depth}"] = [param.detach().clone() for param in self.SV[f"P{depth}"]]   

        # save best structure variables from this batch
        self.best_SV[f"P{depth}"] = [param.detach().clone() for param in self.SV[f"P{depth}"]]
        self.best_SV[f"G{depth}"] = self.SV[f"G{depth}"]
        # save posterior structure variables across batches
        self.best_SV[f"Posterior{depth}"] = [param.detach().clone() for param in self.SV[f"P{depth}"]]

    @torch.no_grad()
    def _setup_cluster_values(self, depth, param_alpha, param_beta):
        """
        Setup local structure variables at a given depth.
        """
        self.SV[f"LG{depth}"] = truncated_stick_breaking(param_alpha, param_beta, sample_shape=(self.param_dims[-(depth+2)],), truncate_dim=0)
        assert self.SV[f"LG{depth}"].shape == tuple(self.param_dims[-(depth+2):-1])
        assert self.SV[f"LG{depth}"].sum(dim=0).allclose(torch.ones(self.SV[f"LG{depth}"].shape[1:], device=self.device))
        self.SV[f"LP{depth}"] = [param_alpha.unsqueeze(0).expand(tuple(self.param_dims[-(depth+2):-1])), param_beta.unsqueeze(0).expand(tuple(self.param_dims[-(depth+2):-1]))]
        assert self.SV[f"LP{depth}"][0].shape == tuple(self.param_dims[-(depth+2):-1])
        assert self.SV[f"LP{depth}"][1].shape == tuple(self.param_dims[-(depth+2):-1])
        # save posterior structure variables of a iteration for potential best structure
        self.SV[f"LPosterior{depth}"] = [param.detach().clone() for param in self.SV[f"LP{depth}"]]

        # save best structure variables from this batch
        self.best_SV[f"LP{depth}"] = [param.detach().clone() for param in self.SV[f"LP{depth}"]]
        self.best_SV[f"LG{depth}"] = self.SV[f"LG{depth}"]
        # save posterior structure variables across batches
        self.best_SV[f"LPosterior{depth}"] = [param.detach().clone() for param in self.SV[f"LP{depth}"]]

    @torch.no_grad()
    def init_mixture_components(self):
        """
        Initialize mixture components in PyTorch version:
        - Dirichlet topics over vocab
        - Normal–InverseGamma regression parameters
        """
        # -----------------------
        # Mixture components
        # -----------------------

        alpha = self.SV["nig_alpha"]
        beta = self.SV["nig_beta"]
        mu = self.SV["nig_mu"]
        kappa = self.SV["nig_kappa"]
        dir_alpha = self.SV["dir_alpha"]

        self.mixture_components = {}
        self.best_mixture_components = {}
        self.mixture_components_posterior = {}

        # --- Generation (Dirichlet over vocabulary) ---
        self.mixture_components["generation"] = Dirichlet(dir_alpha).sample()  # (K, vocab_size)
        assert self.mixture_components["generation"].shape == (self.K, self.vocab_size)

        # --- Regression components via NIG prior ---
        # InverseGamma(alpha, beta)
        sigma = InverseGamma(
            alpha,
            beta,
        ).sample()  # (K,)
        assert sigma.shape == (self.K,)

        # Normal(mu, sqrt(sigma/kappa))
        mu = Normal(
            mu,
            torch.sqrt(sigma / kappa)
        ).sample()
        assert mu.shape == (self.K,)

        self.mixture_components["regression_sigma"] = sigma
        self.mixture_components["regression_mu"] = mu

        # Deep copies for posterior/best tracking
        for k in ["generation", "regression_mu", "regression_sigma"]:
            self.best_mixture_components[k] = self.mixture_components[k]
            self.mixture_components_posterior[k] = self.mixture_components[k].clone()

    @torch.no_grad()
    def init_latent_variables(self, obs: torch.Tensor):
        """
        Initialize latent variables (PyTorch version of JAX code).

        Args:
            obs: (N, M, D) observed data tensor
        Returns:
            z_gen: (N, M) generator category assignments
            z_reg: (N,) regression category assignments
            local_category_assignments: (N, num_levels) local hierarchical assignments
            doc_values: dict of per-document Beta params and mixture weights
        """
        N, M, _ = obs.shape

        # ------------------------------
        # Sample global latent assignments
        # ------------------------------
        z_gen = torch.randint(low=0, high=self.K, size=(N, M),  device=self.device)
        z_reg = torch.randint(low=0, high=self.K, size=(N,), device=self.device)

        # ------------------------------
        # Sample local hierarchical assignments
        # ------------------------------
        local_category_assignments = []
        for max_cat in self.cluster_dims:
            cats = torch.randint(low=0, high=max_cat, size=(N,), device=self.device)
            local_category_assignments.append(cats)
        local_category_assignments = torch.stack(local_category_assignments, dim=1)  # (N, num_levels)
        assert local_category_assignments.shape == (N, len(self.cluster_dims))

        # ------------------------------
        # Build per-document Beta/G mixture parameters
        # ------------------------------
        doc_values = {}
        rev_idx = torch.flip(local_category_assignments, dims=[1])

        # Extract level index (deepest hierarchy)
        num_levels = len(self.cluster_dims)
        index_dims = torch.arange(num_levels, device=self.device)

        params_0, params_1 = gen_next_level_prior(self.struct_params[f"alpha{num_levels}"], self.SV[f"G{num_levels}"])

        doc_params_0 = advanced_multi_index_select(params_0, rev_idx, dims=index_dims).to(self.device)
        assert torch.all(params_0[tuple(rev_idx[:, i] for i in range(rev_idx.shape[1]))] == doc_params_0)
        doc_params_1 = advanced_multi_index_select(params_1, rev_idx, dims=index_dims).to(self.device)
        assert torch.all(params_1[tuple(rev_idx[:, i] for i in range(rev_idx.shape[1]))] == doc_params_1)

         # Compute G mixture weights
        doc_values["P"] = [doc_params_0, doc_params_1]
        doc_values["G"] = truncated_stick_breaking(doc_params_0, doc_params_1, sample_shape=(), truncate_dim=-1)
        assert doc_values["G"].shape == (N, self.K)
        assert torch.allclose(doc_values["G"].sum(dim=-1), torch.ones(N, device=self.device))

        return z_gen, z_reg, local_category_assignments, doc_values
    
    @torch.no_grad()
    def _update_struct_posterior(self, lr):
        for parent_level in range(len(self.param_dims)):
            self.best_SV[f"Posterior{parent_level}"][0] = (1-lr)*self.best_SV[f"Posterior{parent_level}"][0] + lr*self.best_SV[f"P{parent_level}"][0]
            self.best_SV[f"Posterior{parent_level}"][1] = (1-lr)*self.best_SV[f"Posterior{parent_level}"][1] + lr*self.best_SV[f"P{parent_level}"][1]

            if (parent_level < len(self.param_dims) - 1):
                self.best_SV[f"LPosterior{parent_level}"][0] = (1-lr)*self.best_SV[f"LPosterior{parent_level}"][0] + lr*self.best_SV[f"LP{parent_level}"][0]
                self.best_SV[f"LPosterior{parent_level}"][1] = (1-lr)*self.best_SV[f"LPosterior{parent_level}"][1] + lr*self.best_SV[f"LP{parent_level}"][1]

        self.best_SV["dir_alpha_Pos"] = (1-lr)*self.best_SV["dir_alpha_Pos"] + lr*self.best_SV["dir_alpha"]
        self.best_SV["nig_mu_Pos"] = (1-lr)*self.best_SV["nig_mu_Pos"] + lr*self.best_SV["nig_mu"]
        self.best_SV["nig_kappa_Pos"] = (1-lr)*self.best_SV["nig_kappa_Pos"] + lr*self.best_SV["nig_kappa"]
        self.best_SV["nig_alpha_Pos"] = (1-lr)*self.best_SV["nig_alpha_Pos"] + lr*self.best_SV["nig_alpha"]
        self.best_SV["nig_beta_Pos"] = (1-lr)*self.best_SV["nig_beta_Pos"] + lr*self.best_SV["nig_beta"]

        self.mixture_components_posterior["generation"] = (1-lr)*self.mixture_components_posterior["generation"] + lr*self.best_mixture_components["generation"]
        self.mixture_components_posterior["regression_mu"] = (1-lr)*self.mixture_components_posterior["regression_mu"] + lr*self.best_mixture_components["regression_mu"]
        self.mixture_components_posterior["regression_sigma"] = (1-lr)*self.mixture_components_posterior["regression_sigma"] + lr*self.best_mixture_components["regression_sigma"]

    @torch.no_grad()
    def _set_struct_to_best(self):
        for k in ["generation", "regression_mu", "regression_sigma"]:
            self.mixture_components[k] = self.best_mixture_components[k].detach().clone()

        self.SV["dir_alpha"] = self.best_SV["dir_alpha"].detach().clone()
        self.SV["nig_mu"] = self.best_SV["nig_mu"].detach().clone()
        self.SV["nig_kappa"] = self.best_SV["nig_kappa"].detach().clone()
        self.SV["nig_alpha"] = self.best_SV["nig_alpha"].detach().clone()
        self.SV["nig_beta"] = self.best_SV["nig_beta"].detach().clone()

        for depth in range(len(self.param_dims)):
            self.SV[f"P{depth}"] = [self.best_SV[f"P{depth}"][0].detach().clone(), self.best_SV[f"P{depth}"][1].detach().clone()]

            self.SV[f"G{depth}"] = truncated_stick_breaking(self.SV[f"P{depth}"][0], self.SV[f"P{depth}"][1], sample_shape=(), truncate_dim=-1)
            assert self.SV[f"G{depth}"].shape == tuple(self.param_dims[-(depth+1):])

            if depth < len(self.param_dims) - 1:
                self.SV[f"LP{depth}"] = [self.best_SV[f"LP{depth}"][0].detach().clone(), self.best_SV[f"LP{depth}"][1].detach().clone()]

                self.SV[f"LG{depth}"] = truncated_stick_breaking(self.SV[f"LP{depth}"][0], self.SV[f"LP{depth}"][1], sample_shape=(), truncate_dim=0)
                assert self.SV[f"LG{depth}"].shape == tuple(self.param_dims[-(depth+2):-1])

    @torch.no_grad()
    def _update_best_struct(self, log_prob):
        if log_prob > self.best_log_prob:
            self.best_log_prob = log_prob

            for k in ["generation", "regression_mu", "regression_sigma"]:
                self.best_mixture_components[k] = self.mixture_components[k]

            self.best_SV["dir_alpha"] = self.SV["dir_alpha"]
            self.best_SV["nig_mu"] = self.SV["nig_mu"]
            self.best_SV["nig_kappa"] = self.SV["nig_kappa"]
            self.best_SV["nig_alpha"] = self.SV["nig_alpha"]
            self.best_SV["nig_beta"] = self.SV["nig_beta"]

            for parent_level in range(len(self.param_dims)):
                self.best_SV[f"P{parent_level}"] = [self.SV[f"Posterior{parent_level}"][0], self.SV[f"Posterior{parent_level}"][1]]

                if parent_level < len(self.param_dims) - 1:
                    self.best_SV[f"LP{parent_level}"] = [self.SV[f"LPosterior{parent_level}"][0], self.SV[f"LPosterior{parent_level}"][1]]

    @torch.no_grad()
    def _update_struct_prior(self):
        """
        PyTorch equivalent of JAX update_struct_prior().
        Refreshes Prior and G variables by sampling new Beta sticks.
        """

        self.SV["dir_alpha"] = self.best_SV["dir_alpha_Pos"].detach().clone()
        self.SV["nig_mu"] = self.best_SV["nig_mu_Pos"].detach().clone()
        self.SV["nig_kappa"] = self.best_SV["nig_kappa_Pos"].detach().clone()
        self.SV["nig_alpha"] = self.best_SV["nig_alpha_Pos"].detach().clone()
        self.SV["nig_beta"] = self.best_SV["nig_beta_Pos"].detach().clone()

        self.SV["dir_alpha_Pos"] = self.SV["dir_alpha"].detach().clone()
        self.SV["nig_mu_Pos"] = self.SV["nig_mu"].detach().clone()
        self.SV["nig_kappa_Pos"] = self.SV["nig_kappa"].detach().clone()
        self.SV["nig_alpha_Pos"] = self.SV["nig_alpha"].detach().clone()
        self.SV["nig_beta_Pos"] = self.SV["nig_beta"].detach().clone()

        # Loop over each hierarchy level
        for parent_level in range(len(self.param_dims)):
            # -----------------------------
            # Copy Posterior -> Prior
            # -----------------------------
            self.SV[f"P{parent_level}"] = [self.best_SV[f"Posterior{parent_level}"][0].detach().clone(), self.best_SV[f"Posterior{parent_level}"][1].detach().clone()]

            # -----------------------------
            # Resample Gₗ stick-breaking weights
            # -----------------------------
            self.SV[f"G{parent_level}"] = truncated_stick_breaking(self.SV[f"P{parent_level}"][0], self.SV[f"P{parent_level}"][1], sample_shape=(), truncate_dim=-1)
            assert self.SV[f"G{parent_level}"].shape == tuple(self.param_dims[-(parent_level+1):])

            self.SV[f"Posterior{parent_level}"] = [self.SV[f"P{parent_level}"][0].detach().clone(), self.SV[f"P{parent_level}"][1].detach().clone()]

            # -----------------------------
            # If not last level: update local sticks (L)
            # -----------------------------
            if parent_level < len(self.param_dims) - 1:
                self.SV[f"LP{parent_level}"] = [self.best_SV[f"LPosterior{parent_level}"][0].detach().clone(), self.best_SV[f"LPosterior{parent_level}"][1].detach().clone()]

                self.SV[f"LG{parent_level}"] = truncated_stick_breaking(self.SV[f"LP{parent_level}"][0], self.SV[f"LP{parent_level}"][1], sample_shape=(), truncate_dim=0)
                assert self.SV[f"LG{parent_level}"].shape == tuple(self.param_dims[-(parent_level+2):-1])

                self.SV[f"LPosterior{parent_level}"] = [self.SV[f"LP{parent_level}"][0].detach().clone(), self.SV[f"LP{parent_level}"][1].detach().clone()]

        # -----------------------------
        # Copy posterior mixture components
        # -----------------------------
        for k in ["generation", "regression_mu", "regression_sigma"]:
            self.mixture_components[k] = self.mixture_components_posterior[k].detach().clone()

    @torch.no_grad()
    def forward(self, obs, *args, **kwargs):
        pass

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, *args, **kwargs):
        """
        Perform Gibbs inference for new documents (PyTorch version).

        Args:
            obs: (N, M, V) tensor of word observations
            reg: (N,) optional tensor of regression targets
            num_iters: int, number of Gibbs iterations
            generator: torch.Generator for reproducible RNG

        Returns:
            z_gen: (N, M) tensor — word category assignments
            z_reg: (N,) tensor — regression category assignments
            local_category_assignments: (N, num_levels)
            doc_values: dict of document-specific parameters
            log_probs: np.ndarray of log-likelihood trace
        """
        num_iters = kwargs.get("num_iters", 100)
        best_z_gen = None
        best_z_reg = None
        best_local_category_assignments = None
        best_doc_values = None

        N, M, _ = obs.shape
        reg = kwargs.get("reg", None)
        obs = obs.to(self.device)
        log_probs = []

        # Initialize latent variables
        z_gen, z_reg, local_category_assignments, doc_values = self.init_latent_variables(obs)

        # --- Gibbs sampling loop ---
        pbar = trange(num_iters, desc="Inference Gibbs Sampling")
        for it in pbar:
            # ------------------------
            # 1. Sample word-level categories
            # ------------------------
            z_gen = self.words_cat_gibbs(obs, doc_values["G"])

            # ------------------------
            # 2. Sample document weights
            # ------------------------
            doc_values = self.docs_weight_gibbs(
                doc_values,
                z_gen,
                z_reg,
                scale_constant=1.0,
                predict=True
            )

            # ------------------------
            # 3. Sample hierarchical categories
            # ------------------------
            with torch.no_grad():
                for depth in range(len(self.cluster_dims)):
                    cats = self.collapsed_docs_cat_gibbs(
                        depth=depth,
                        z_gen=z_gen,
                        z_reg=z_reg,
                        parent_cats=local_category_assignments[:, :depth],
                        predict=True
                    )
                    local_category_assignments[:, depth] = cats

            # ------------------------
            # 4. Update document priors
            # ------------------------
            doc_values = self.update_docs_prior(doc_values, torch.flip(local_category_assignments, dims=[1]))

            # ------------------------
            # 5. Compute log-likelihood
            # ------------------------
            log_prob = self.compute_log_likelihood(obs, z_gen, z_reg, reg, predict=True)
            log_prob_val = log_prob.item() if torch.is_tensor(log_prob) else float(log_prob)

            if log_prob_val > max(log_probs, default=-float("inf")):
                best_z_gen = z_gen.clone()
                best_z_reg = z_reg.clone()
                best_local_category_assignments = local_category_assignments.clone()
                best_doc_values = {"G": doc_values["G"].clone(), "P": [doc_values["P"][0].clone(), doc_values["P"][1].clone()]}

            log_probs.append(log_prob_val)
            pbar.set_description(f"Inference Gibbs Sampling (Iter {it+1}) LogProb {log_prob_val:.2f}")

        # Convert to numpy for visualization compatibility
        return (
            best_z_gen,
            best_z_reg,
            best_local_category_assignments,
            best_doc_values,
            torch.tensor(log_probs)
        )
        

    @torch.no_grad()
    def infer(self, obs: torch.Tensor, **kwargs):
        """
        Full Gibbs inference for the HDMM model (PyTorch version).
        """
        lr = kwargs.get("lr", 0.1)
        num_iters = kwargs.get("num_iters", 100)
        known_cats = kwargs.get("known_cats", None)
        known_mixtures = kwargs.get("known_mixtures", None)
        known_struct = kwargs.get("known_struct", None)
        known_words = kwargs.get("known_words", None)
        datasize = kwargs.get("datasize", obs.shape[0])
        epoch = kwargs.get("epoch", 0)
        sanity_check = kwargs.get("sanity_check", True)
        plot_gap = kwargs.get("plot_gap", 50)
        log_dir = kwargs.get("log_dir", None)

        best_z_gen = None
        best_z_reg = None
        best_local_category_assignments = None
        best_doc_values = None
        obs = obs.to(self.device)

        if epoch > 0:
            self._update_struct_prior()

        skip_depth = []
        skip_struct = []

        N, M, _ = obs.shape
        scale_constant = datasize / N
        reg = kwargs.get("reg", None)
        reg = reg.to(self.device) if reg is not None else None
        log_probs = []
        if sanity_check:
            assert (obs.sum(dim=-1) == 1).all(), "Observations must be one-hot encoded."
            assert ((obs == 0) | (obs == 1)).all(), "Observations must be binary one-hot vectors."
            # assert torch.all((reg >= 0) & (reg <= 1)) if reg is not None else True, f"Regression targets must be non-negative, instead got min {reg.min().item()} and max {reg.max().item()}."

        # --- Initialize latent variables ---
        z_gen, z_reg, local_category_assignments, doc_values = self.init_latent_variables(obs)
        if sanity_check:
            assert z_gen.shape == (N, M)
            assert (z_gen >= 0).all() and (z_gen < self.K).all()
            assert z_reg.shape == (N,)
            assert (z_reg >= 0).all() and (z_reg < self.K).all()
            assert local_category_assignments.shape == (N, len(self.cluster_dims))
            assert all(torch.all(local_category_assignments[:, i] < self.cluster_dims[i]) for i in range(len(self.cluster_dims)))
            assert doc_values["G"].shape == (N, self.K)
            assert torch.allclose(doc_values["G"].sum(dim=-1), torch.ones(N, device=self.device))
            assert doc_values["P"][0].shape == (N, self.K)
            assert doc_values["P"][1].shape == (N, self.K)

        if known_words is not None:
            z_gen = known_words.to(self.device).to(torch.int64)
            if sanity_check:
                assert z_gen.shape == (N, M)
                assert (z_gen >= 0).all() and (z_gen < self.K).all()

        # --- Freeze known category depths ---
        if known_cats is not None:
            for depth, cats in known_cats.items():
                local_category_assignments[:, depth] = cats
                if sanity_check:
                    assert cats.shape == (N,)
                    assert (cats >= 0).all() and (cats < self.cluster_dims[depth]).all()
                skip_depth.append(depth)

        if known_mixtures is not None:
            self.mixture_components["generation"] = known_mixtures["generation"].to(self.device)
            if sanity_check:
                assert self.mixture_components["generation"].shape == (self.K, self.vocab_size)
                assert torch.allclose(self.mixture_components["generation"].sum(dim=-1), torch.ones(self.K, device=self.device))
                assert self.mixture_components["generation"].min() >= 0

        if known_struct is not None:
            skip_struct = list(known_struct.keys())
            for depth in range(len(self.param_dims)):
                if depth in skip_struct:
                    self.SV[f"G{depth+1}"] = known_struct[depth].to(self.device)
                    if sanity_check:
                        assert self.SV[f"G{depth+1}"].shape == tuple(self.param_dims[-(depth+2):]), f"for depth {depth+1} expected shape {tuple(self.param_dims[-(depth+1):])}, got {self.SV[f'G{depth+1}'].shape}"
                        assert torch.allclose(self.SV[f"G{depth+1}"].sum(dim=-1), torch.ones(self.SV[f"G{depth+1}"].shape[:-1], device=self.device))
        # --- Gibbs Sampling Loop ---
        pbar = trange(num_iters, desc="Gibbs Sampling")

        for it in pbar:
            # ------------------------
            # 1. Sample document-level word categories
            # ------------------------
            if known_words is None:
                z_gen = self.words_cat_gibbs(obs, doc_values["G"], sanity_check=sanity_check)
                if sanity_check:
                    assert z_gen.shape == (N, M)
                    assert (z_gen >= 0).all() and (z_gen < self.K).all()

            # ------------------------
            # 2. Sample regression categories
            # ------------------------
            z_reg = self.regs_cat_gibbs(reg, doc_values["G"], sanity_check=sanity_check)
            if sanity_check:
                assert z_reg.shape == (N,)
                assert (z_reg >= 0).all() and (z_reg < self.K).all()

            # ------------------------
            # 3. Update document-level stick-breaking weights
            # ------------------------
            doc_values = self.docs_weight_gibbs(
                doc_values,
                z_gen,
                z_reg,
                scale_constant,
                predict=False,
                sanity_check=sanity_check
            )
            if sanity_check:
                assert doc_values["G"].shape == (N, self.K)
                assert torch.allclose(doc_values["G"].sum(dim=-1), torch.ones(N, device=self.device))
                assert doc_values["P"][0].shape == (N, self.K)
                assert doc_values["P"][1].shape == (N, self.K)

            # ------------------------
            # 4. Update hierarchical document category assignments
            # ------------------------
            for depth in range(len(self.cluster_dims)):
                if depth in skip_depth:
                    continue
                cats = self.collapsed_docs_cat_gibbs(
                    depth,
                    z_gen,
                    z_reg,
                    parent_cats=local_category_assignments[:, :depth],
                    predict=False,
                    sanity_check=sanity_check
                )
                if sanity_check:
                    assert cats.shape == (N,)
                    assert (cats >= 0).all() and (cats < self.cluster_dims[depth]).all()

                with torch.no_grad():
                    local_category_assignments[:, depth] = cats

            # ------------------------
            # 5. Update document priors
            # ------------------------
            doc_values = self.update_docs_prior(doc_values, torch.flip(local_category_assignments, dims=[1]), sanity_check=sanity_check)
            if sanity_check:
                assert doc_values["G"].shape == (N, self.K)
                assert torch.allclose(doc_values["G"].sum(dim=-1), torch.ones(N, device=self.device))
                assert doc_values["P"][0].shape == (N, self.K)
                assert doc_values["P"][1].shape == (N, self.K)
            # ------------------------
            # 6. Sample generation components
            # ------------------------
            if known_mixtures is None:
                self.gen_mix_gibbs(obs, z_gen, scale_constant, sanity_check=sanity_check)

            # ------------------------
            # 7. Sample regression components
            # ------------------------
            self.reg_mix_gibbs(reg, z_reg, scale_constant, sanity_check=sanity_check)

            # ------------------------
            # 8. Update structural weights
            # ------------------------

            for depth in range(len(self.param_dims)):
                if depth == 0:
                    unique_rows = None
                    positions = None
                    rev_cats = None
                else:
                    unique_rows, positions = get_unique_rows_and_positions(local_category_assignments[:, :depth])
                    rev_cats = torch.flip(unique_rows, dims=[1]).to(self.device)
                if depth not in skip_struct:
                    self.struct_weights_gibbs(depth, rev_cats, positions, z_gen, z_reg, scale_constant, sanity_check=sanity_check)
                if depth < len(self.param_dims) - 1:
                    self.struct_cluster_gibbs(depth, rev_cats, positions, local_category_assignments, scale_constant, sanity_check=sanity_check)

            # ------------------------
            # 9. Compute log-likelihood and update best state
            # ------------------------
            log_prob = self.compute_log_likelihood(obs, z_gen, z_reg, reg)
            log_probs.append(log_prob.item() if torch.is_tensor(log_prob) else float(log_prob))
            
            if (log_prob > self.best_log_prob):
                best_z_gen = z_gen.clone()
                best_z_reg = z_reg.clone()
                best_local_category_assignments = local_category_assignments.clone()
                best_doc_values = {"P": [doc_values["P"][0].clone(), doc_values["P"][1].clone()], "G": doc_values["G"].clone()}

            self._update_best_struct(
                log_prob
            )
            # ------------------------
            # 10. Optional visualization
            # ------------------------
            if it > 0 and (it+1) % plot_gap == 0:
                likelihood_visualization(torch.tensor(log_probs), torch.zeros_like(torch.tensor(log_probs)), epoch=it, log_dir=log_dir)

            pbar.set_description(f"Gibbs Sampling (Iter {it+1}) LogProb {log_probs[-1]:.2f}")

        # ------------------------
        # Return results
        # ------------------------
        # return (
        #     best_z_gen,
        #     best_z_reg,
        #     best_local_category_assignments,
        #     best_doc_values,
        #     torch.tensor(log_probs)
        # )
        
        return (
            z_gen,
            z_reg,
            local_category_assignments,
            doc_values,
            torch.tensor(log_probs)
        )
    
    @torch.no_grad()
    def compute_log_likelihood(self, obs: torch.Tensor, z_gen: torch.Tensor,
                               z_reg: torch.Tensor, reg: torch.Tensor,
                               predict: bool = False, sanity_check: bool = True) -> torch.Tensor:
        """
        Compute total log-likelihood for HDMM (PyTorch version of JAX code).

        Args:
            obs: (N, M, V) tensor of one-hot word observations.
            z_gen: (N, M) tensor of generator component assignments.
            z_reg: (N,) tensor of regression component assignments.
            reg: (N,) tensor of regression targets.
            predict: if True, skip regression likelihood.

        Returns:
            log_prob: scalar tensor (sum of all log-probs).
        """
        log_prob = torch.tensor(0.0, device=self.device)

        # -------------------------------
        # Multinomial likelihood (word generation)
        # -------------------------------
        gen_param = self.mixture_components["generation"][z_gen]  # (N, M, V)
        if sanity_check:
            assert gen_param.shape == (obs.shape[0], obs.shape[1], self.vocab_size)
            assert (gen_param >= 0).all()
            assert torch.allclose(gen_param.sum(dim=-1), torch.ones(gen_param.shape[:-1], device=self.device))
        gen_param = torch.clamp(gen_param, min=1e-12, max=1.0)
        gen_param = gen_param / gen_param.sum(dim=-1, keepdim=True)

        word_dist = Multinomial(total_count=1, probs=gen_param)
        word_prob = word_dist.log_prob(obs)  # (N, M)
        log_prob = log_prob + word_prob.sum()

        # -------------------------------
        # Normal likelihood (regression)
        # -------------------------------
        if not predict:
            mu = self.mixture_components["regression_mu"][z_reg]      # (N,)
            sigma = self.mixture_components["regression_sigma"][z_reg]  # (N,)
            if sanity_check:
                assert mu.shape == (obs.shape[0],)
                assert sigma.shape == (obs.shape[0],)
                assert (sigma > 0).all()
            sigma = torch.clamp(sigma, min=1e-8)  # ensure positive
            reg_dist = Normal(loc=mu, scale=sigma)
            reg_prob = reg_dist.log_prob(reg)  # (N,)
            log_prob = log_prob + reg_prob.sum()

        return log_prob

    @torch.no_grad()
    def gen_mix_gibbs(self, obs: torch.Tensor, z_gen: torch.Tensor, scale_constant: float, sanity_check: bool = True):
        """
        Gibbs sampling step for the generation component of mixture k (PyTorch version).

        Args:
            obs_k: (N_obs, V) tensor of one-hot or count word observations assigned to component k.
            k: integer index of the component to update.
            scale_constant: scaling constant for posterior update.
            sanity_check: whether to perform sanity checks on the output.
        """

        # Compute prior Dirichlet parameters
        dir_alpha = self.SV["dir_alpha"]

        # Sample new generation parameters from the posterior
        generation_components, new_dir_alpha = dirichlet_posterior(obs, z_gen, dir_alpha, self.K, scale_constant)
        if sanity_check:
            assert generation_components.shape == (self.K, self.vocab_size)
            assert (generation_components >= 0).all()
            assert torch.allclose(generation_components.sum(dim=-1), torch.ones(self.K, device=self.device))
            assert new_dir_alpha.shape == dir_alpha.shape

        self.mixture_components["generation"] = generation_components.to(self.device)
        self.SV["dir_alpha_Pos"] = new_dir_alpha.to(self.device)

    @torch.no_grad()
    def reg_mix_gibbs(self, reg: torch.Tensor, z_reg: torch.Tensor, scale_constant: float, sanity_check: bool = True):
        """
        Gibbs sampling step for the regression component of mixture k (PyTorch version).

        Args:
            reg_k: (N_obs,) tensor of regression observations assigned to component k.
            k: integer index of the mixture component to update.
            scale_constant: scaling constant for posterior update.
        """

        # Extract prior NIG parameters
        mu0 = self.SV["nig_mu"]
        kappa0 = self.SV["nig_kappa"]
        alpha0 = self.SV["nig_alpha"]
        beta0 = self.SV["nig_beta"]

        # Call the PyTorch version of NIG posterior
        new_mu, new_sigma, new_nig_mu, new_nig_kappa, new_nig_alpha, new_nig_beta = nig_posterior(
            reg,
            z_reg,
            self.K,
            (mu0, kappa0, alpha0, beta0),
            scale_constant
        )
        if sanity_check:
            assert new_mu.shape == (self.K,)
            assert new_sigma.shape == (self.K,)
            assert (new_sigma > 0).all()
            assert new_nig_mu.shape == (self.K,)
            assert new_nig_kappa.shape == (self.K,)
            assert new_nig_alpha.shape == (self.K,)
            assert new_nig_beta.shape == (self.K,)

        self.mixture_components["regression_mu"] = new_mu.to(self.device)
        self.mixture_components["regression_sigma"] = new_sigma.to(self.device)

        self.SV["nig_mu_Pos"] = new_nig_mu.to(self.device)
        self.SV["nig_kappa_Pos"] = new_nig_kappa.to(self.device)
        self.SV["nig_alpha_Pos"] = new_nig_alpha.to(self.device)
        self.SV["nig_beta_Pos"] = new_nig_beta.to(self.device)

    @torch.no_grad()
    def struct_weights_gibbs(self,
                             depth: int,
                             rev_cat: torch.Tensor,
                             row_idx: torch.Tensor,
                             z_gen: torch.Tensor,
                             z_reg: torch.Tensor,
                             scale_constant: float,
                             sanity_check: bool = True):
        """
        Gibbs sampling step for updating hierarchical structural weights (PyTorch version).

        Args:
            depth: int, current hierarchical depth.
            rev_cat: tensor of reversed category indices selecting location in structure.
            matching_z_gen: tensor of generation indices matching current path.
            matching_z_reg: tensor of regression indices matching current path.
            scale_constant: scaling factor for posterior updates.
        """

        # 1️⃣ Compute conditional Beta parameters for this node
        new_params = self._cat_weight_conditional(
            depth, rev_cat, row_idx, z_gen, z_reg, scale_constant, sanity_check
        )
        new_params = [param.to(self.device) for param in new_params]

        # 2️⃣ Sample new Beta sticks
        new_weights = truncated_stick_breaking(new_params[0], new_params[1], sample_shape=(), truncate_dim=-1)
        new_weights = new_weights.to(self.device)
        assert torch.allclose(new_weights.sum(dim=-1), torch.ones(new_weights.shape[0], device=self.device))

        # Update G and next level P
        self._update_struct_slice(depth, rev_cat, new_weights, new_params, sanity_check=sanity_check)

    @torch.no_grad()
    def _cat_weight_conditional(self,
                                depth: int,
                                rev_cat: torch.Tensor,
                                row_idx: torch.Tensor,
                                z_gen: torch.Tensor,
                                z_reg: torch.Tensor,
                                scale_constant: float,
                                sanity_check: bool = True):
        """
        Compute posterior Beta parameters for category-level stick-breaking weights (PyTorch version).

        Args:
            depth: int — hierarchy level (0 = top level)
            rev_cat: tensor of reversed category indices for current path
            word_cats: (N_word,) tensor of word-level category assignments
            reg_cats: (N_reg,) tensor of regression category assignments
            scale_constant: float — scaling factor for posterior update

        Returns:
            new_params: [alpha_new, beta_new] — updated Beta parameters (each (K,))
        """
        # ----------------------------
        # Get prior parameters
        # ----------------------------
        if depth == 0:
            params = [
                self.SV["P0"][0],
                self.SV["P0"][1]
            ]
            word_cats_group = [z_gen]
            reg_cats_group = [z_reg]
        else:
            idx_dims = torch.arange(0, rev_cat.shape[1], device=self.device)
            params = [
                advanced_multi_index_select(self.SV[f"P{depth}"][0], rev_cat, dims=idx_dims),
                advanced_multi_index_select(self.SV[f"P{depth}"][1], rev_cat, dims=idx_dims)
            ]
            if sanity_check:
                assert torch.allclose(params[0], self.SV[f"P{depth}"][0][tuple(rev_cat[:, i] for i in range(rev_cat.shape[1]))])
                assert torch.allclose(params[1], self.SV[f"P{depth}"][1][tuple(rev_cat[:, i] for i in range(rev_cat.shape[1]))])

            word_cats_group = [z_gen[i] for i in row_idx]
            reg_cats_group = [z_reg[i] for i in row_idx]
            assert params[0].shape == (rev_cat.shape[0], self.K), f"depth {depth} Shape mismatch: params {params[0].shape} vs rev_cat {rev_cat.shape}"
        alpha_bias, beta_bias = self._beta_group_bias(word_cats_group, reg_cats_group)
        if (depth == 0):
            alpha_bias = alpha_bias.to(self.device).squeeze()
            beta_bias = beta_bias.to(self.device).squeeze()
        # ----------------------------
        # Compute category counts
        # ----------------------------
        assert alpha_bias.shape == params[0].shape, f"depth {depth} Shape mismatch: alpha {alpha_bias.shape} vs params {params[0].shape}"

        # ----------------------------
        # Update Beta parameters
        # ----------------------------
        new_params = [
            params[0] + alpha_bias * scale_constant,
            params[1] + beta_bias * scale_constant
        ]

        return new_params

    @torch.no_grad()
    def _update_struct_slice(self, depth: int, rev_cats_slice: torch.Tensor, new_weights: torch.Tensor, new_params: list, sanity_check: bool = True):
        """
        Update structural weights at a specific hierarchy level and category path (PyTorch version).

        Args:
            depth: int — hierarchy level to update.
            cats: tensor — category indices path within hierarchy.
            new_weights: tensor — new mixture weights to set.
        """
        if depth == 0:
            self.SV[f"G{depth}"] = new_weights
            self.SV[f"Posterior{depth}"] = new_params
        else:
            self.SV[f"G{depth}"] = safe_update_scatter(
                self.SV[f"G{depth}"],
                rev_cats_slice,
                new_weights,
                dim=-1
            )
            if sanity_check:
                assert torch.allclose(self.SV[f"G{depth}"][tuple(rev_cats_slice[:, i] for i in range(rev_cats_slice.shape[1]))], new_weights)
            self.SV[f"Posterior{depth}"] = [
                safe_update_scatter(
                    self.SV[f"Posterior{depth}"][0],
                    rev_cats_slice,
                    new_params[0],
                    dim=-1
                ),
                safe_update_scatter(
                    self.SV[f"Posterior{depth}"][1],
                    rev_cats_slice,
                    new_params[1],
                    dim=-1
                )
            ]   
            if sanity_check:
                assert torch.allclose(self.SV[f"Posterior{depth}"][0][tuple(rev_cats_slice[:, i] for i in range(rev_cats_slice.shape[1]))], new_params[0])
                assert torch.allclose(self.SV[f"Posterior{depth}"][1][tuple(rev_cats_slice[:, i] for i in range(rev_cats_slice.shape[1]))], new_params[1])
            assert self.SV[f"G{depth}"].shape == tuple(self.param_dims[-(depth+1):])
            assert self.SV[f"Posterior{depth}"][0].shape == tuple(self.param_dims[-(depth+1):])
            assert self.SV[f"Posterior{depth}"][1].shape == tuple(self.param_dims[-(depth+1):])

        if depth + 1 < len(self.param_dims):
            param_alpha, param_beta = gen_next_level_prior(self.struct_params[f"alpha{depth}"], self.SV[f"G{depth}"])         
            assert param_alpha.shape == tuple(self.param_dims[-(depth+1):])
            assert param_beta.shape == tuple(self.param_dims[-(depth+1):])

            self.SV[f"P{depth+1}"] = [param_alpha.unsqueeze(0).expand(tuple(self.param_dims[-(depth+2):])).to(self.device), param_beta.unsqueeze(0).expand(tuple(self.param_dims[-(depth+2):])).to(self.device)]

    @torch.no_grad()
    def _beta_group_bias(self, word_cats_group, reg_cats_group, predict: bool = False):

        alpha_bias = torch.stack([torch.sum(self._docs_cat_count(word_cats, reg_cats, predict), dim=0) for word_cats, reg_cats in zip(word_cats_group, reg_cats_group)])

        # ----------------------------
        # Compute suffix sum of counts
        # ----------------------------
        beta_bias = suffix_sum(alpha_bias)
        return alpha_bias, beta_bias
                
    @torch.no_grad()
    def struct_cluster_gibbs(self,
                             depth: int,
                             rev_cats: torch.Tensor,
                             row_idx: torch.Tensor,
                             cats: torch.Tensor,
                             scale_constant: float,
                             sanity_check: bool = True):
        """
        Gibbs sampling step for local (intra-level) cluster weights in HDMM hierarchy (PyTorch version).

        Args:
            depth: int — hierarchy level to update.
            row_idx: tensor or int — index/indices of current data point(s) to consider.
            cats: tensor — current category indices path within hierarchy.
            local_category_assignments: (N, L) tensor — full table of local category assignments.
            scale_constant: float — scaling constant for posterior updates.
        """

        # 1️⃣ Compute conditional Beta parameters for local cluster node
        new_params = self._cluster_weight_conditional(
            depth,
            rev_cats,
            row_idx,
            cats,
            scale_constant,
            sanity_check
        )
        new_params = [param.to(self.device) for param in new_params]

        # 2️⃣ Sample new Beta sticks
        new_weights = truncated_stick_breaking(new_params[0], new_params[1], sample_shape=(), truncate_dim=-1)
        new_weights = new_weights.to(self.device)
        assert torch.allclose(new_weights.sum(dim=-1), torch.ones_like(new_weights.sum(dim=-1)))

        if depth == 0:
            self.SV[f"LG{depth}"] = new_weights
        else:
            self.SV[f"LG{depth}"] = safe_update_scatter(
                self.SV[f"LG{depth}"],
                rev_cats,
                new_weights,
                dim=0
            )
            assert self.SV[f"LG{depth}"].shape == tuple(self.param_dims[-(depth+2):-1])
            if sanity_check:
                assert all(torch.allclose(self.SV[f"LG{depth}"][(torch.arange(self.cluster_dims[depth]),) + tuple(rev_cats[j, i] for i in range(rev_cats.shape[1]))], new_weights[j]) for j in range(new_weights.shape[0]))

    @torch.no_grad()
    def _cluster_weight_conditional(self,
                                    depth: int,
                                    rev_cats: torch.Tensor,
                                    row_idx: torch.Tensor,
                                    local_cluster_cats: torch.Tensor,
                                    scale_constant: float,
                                    sanity_check: bool = True):
        """
        Compute posterior Beta parameters for super-cluster-level stick-breaking weights (PyTorch version).

        Args:
            depth: int — hierarchy level
            cats: tensor — current category index path selecting location in LPrior tensors
            local_cluster_cats: (N_cluster,) tensor — cluster-level category assignments
            scale_constant: float — scaling factor for posterior update

        Returns:
            new_params: [alpha_new, beta_new] — updated Beta parameters (each shape (S,))
        """
        S = self.cluster_dims[depth]  # number of clusters at this depth

        # ----------------------------
        # Retrieve prior parameters from LPrior
        # ----------------------------
        if depth == 0:
            params = [
                self.SV["LP0"][0],
                self.SV["LP0"][1]
            ]
            cats_group = [local_cluster_cats[:, depth]]
        else:
            idx_dims = torch.arange(1, rev_cats.shape[1]+1, device=self.device)
            # print("rev_cats:", rev_cats.shape)
            # print("LP shape:", self.SV[f"LP{depth}"][0].shape)
            # print("rev_cats:", rev_cats)
            params = [
                advanced_multi_index_select(self.SV[f"LP{depth}"][0], rev_cats, dims=idx_dims).to(self.device),
                advanced_multi_index_select(self.SV[f"LP{depth}"][1], rev_cats, dims=idx_dims).to(self.device)
            ]

            cats_group = [local_cluster_cats[i][:, depth] for i in row_idx]
            assert params[0].shape[0] == rev_cats.shape[0], f"depth {depth} cluster Shape mismatch: params {params[0].shape} vs rev_cats {rev_cats.shape}"
            assert params[0].shape[1] == S, f"depth {depth} cluster Shape mismatch: params {params[0].shape} vs S {S}"

            if sanity_check:
                assert torch.allclose(params[0], self.SV[f"LP{depth}"][0][tuple(torch.arange(S, device=self.device)), rev_cats])
                assert torch.allclose(params[1], self.SV[f"LP{depth}"][1][tuple(torch.arange(S, device=self.device)), rev_cats])
        # ----------------------------
        # Compute category counts
        # ----------------------------
        cat_count = torch.stack([torch.bincount(cats_group[i], minlength=S).to(torch.float32) for i in range(len(cats_group))]).to(self.device)
        if (depth == 0):
            cat_count = cat_count.squeeze()
        alpha_bias = cat_count
        beta_bias = suffix_sum(alpha_bias)

        assert alpha_bias.shape == params[0].shape, f"depth {depth} cluster Shape mismatch: alpha {alpha_bias.shape} vs params {params[0].shape}"

        # ----------------------------
        # Update Beta parameters
        # ----------------------------
        alpha_new = params[0] + alpha_bias * scale_constant
        beta_new = params[1] + beta_bias * scale_constant

        return [alpha_new, beta_new]

    @torch.no_grad()
    def words_cat_gibbs(self, obs: torch.Tensor, doc_weights: torch.Tensor, sanity_check: bool = True):
        """
        Vectorized Gibbs sampling for all documents and words (PyTorch version).

        Args:
            obs: (N, M, V) tensor — one-hot word vectors per document.
            doc_weights: (N, K) tensor — mixture weights for each document.
        Returns:
            z_gen: (N, M) tensor — sampled topic indices for each word.
        """
        N, M, V = obs.shape
        K = doc_weights.shape[-1]

        # Compute log probability per topic
        gen_param = self.mixture_components["generation"]  # (K, V)
        gen_param = torch.clamp(gen_param, min=1e-12, max=1.0)
        gen_param = gen_param / gen_param.sum(-1, keepdim=True)

        log_probs = obs @ torch.log(gen_param.T)  # (N, M, K)
        unnormalized = log_probs + torch.log(doc_weights.unsqueeze(1) + 1e-12)
        probs = torch.softmax(unnormalized, dim=-1)    
        assert probs.shape == (N, M, K)
        z_gen = torch.multinomial(probs.view(-1, K), 1).squeeze(-1).view(N, M)

        return z_gen

    @torch.no_grad()
    def regs_cat_gibbs(self, reg: torch.Tensor, doc_weights: torch.Tensor, sanity_check: bool = True):
        """
        Vectorized Gibbs sampling for regression categories across all documents (PyTorch version).

        Args:
            reg: (N,) tensor — regression scores per document
            doc_weights: (N, K) tensor — mixture weights per document
        Returns:
            z_reg: (N,) tensor — sampled regression category indices
        """
        N = reg.shape[0]
        K = doc_weights.shape[-1]

        # Extract regression mixture components
        mu = self.mixture_components["regression_mu"]      # (K,)
        sigma = self.mixture_components["regression_sigma"]  # (K,)

        mu = mu.to(self.device)
        sigma = sigma.to(self.device)
        # Normal log probability under each component
        log_probs = Normal(mu, sigma).log_prob(reg.unsqueeze(1))  # (N, K)
        unnormalized = log_probs + torch.log(doc_weights + 1e-12)
        probs = torch.softmax(unnormalized, dim=-1)
        assert probs.shape == (N, K)

        # Sample from categorical distribution
        z_reg = torch.multinomial(probs, 1).squeeze(-1)

        return z_reg

    @torch.no_grad()
    def collapsed_docs_cat_gibbs(self,
                                 depth: int,
                                 z_gen: torch.Tensor,
                                 z_reg: torch.Tensor,
                                 parent_cats: torch.Tensor,
                                 predict: bool = False,
                                 sanity_check: bool = True):
        """
        Gibbs sampling step for collapsed document category assignment (PyTorch version).

        Args:
            depth: int, hierarchical depth (0 = top level).
            obs: (M, V) tensor of one-hot or count word vectors.
            reg: scalar or tensor, regression target for the document.
            z_gen: (M,) tensor of word-level topic assignments.
            z_reg: scalar, regression component assignment.
            parent_cats: (depth,) tensor of ancestor categories.
            predict: bool, if True skip regression likelihood.

        Returns:
            cat: scalar tensor, sampled category index.
            prob: (num_cats,) tensor, categorical probabilities.
        """
        # ----------------------------
        # Retrieve relevant weights
        # ----------------------------
        if depth == 0:
            weight = self.SV[f"G{depth + 1}"].unsqueeze(0).expand(z_gen.shape[0], -1, -1)  # (N, C, K)
            cluster_weight = self.SV[f"LG{depth}"].unsqueeze(0).expand(z_gen.shape[0], -1)  # (N, C)
        else:
            rev_idx = torch.flip(parent_cats, dims=[1])
            struct_dims = torch.arange(1, rev_idx.shape[1]+1, device=self.device)
            weight = advanced_multi_index_select(self.SV[f"G{depth + 1}"], rev_idx, dims=struct_dims).to(self.device) # (C, K)
            if sanity_check:
                assert all(torch.allclose(weight[n], self.SV[f"G{depth + 1}"][torch.arange(self.cluster_dims[depth]), rev_idx[n]]) for n in range(rev_idx.shape[0]))

            cluster_weight = advanced_multi_index_select(self.SV[f"LG{depth}"], rev_idx, dims=struct_dims).to(self.device)
            if sanity_check:
                assert all(torch.allclose(cluster_weight[n], self.SV[f"LG{depth}"][torch.arange(self.cluster_dims[depth]), rev_idx[n]]) for n in range(rev_idx.shape[0]))
        assert weight.shape[1] == self.cluster_dims[depth], \
            f"weight.shape[1]={weight.shape[1]}, expected {self.cluster_dims[depth]}"
        assert cluster_weight.shape[1] == self.cluster_dims[depth], \
            f"cluster_weight.shape[1]={cluster_weight.shape[1]}, expected {self.cluster_dims[depth]}"
        assert weight.shape[-1] == self.K, \
            f"weight.shape[-1]={weight.shape[-1]}, expected {self.K}"

        cats_counts = self._docs_cat_count(z_gen, z_reg, predict).to(self.device) # (N, K)
        # print("depth:", depth)
        # print("cats_counts:", cats_counts.shape)
        # print("cats_counts sample", cats_counts[0])
        # print("rev_idx:", rev_idx.shape if depth > 0 else "N/A")
        # print("rev_idx sample", rev_idx[0] if depth > 0 else "N/A")
        # print("weight:", weight.shape)
        # print("weight sample", weight[0])
        # print("ref", self.SV[f"G{depth + 1}"])
        # print("cluster_weight:", cluster_weight.shape)
        # print("cluster_weight sample", cluster_weight[0])
        # print("ref", self.SV[f"LG{depth}"])

        # Compute log probabilities under each cluster
        log_prob = cats_counts.unsqueeze(1).expand(-1, weight.shape[1], -1).to(dtype=weight.dtype) * torch.log(weight + 1e-12)                     # (N, C, K)
        log_prob = log_prob.sum(dim=-1)  # (N, C)
        # print("log_prob:", log_prob.shape)
        # print("log_prob sample", log_prob[0])

        # --- Add cluster weights ---
        unnorm = log_prob + torch.log(cluster_weight + 1e-12)  # (N, C)
        prob = torch.softmax(unnorm, dim=-1)
        # print("prob:", prob.shape)
        # print("prob sample", prob[0])
        # print("depth", depth)
        # print("prob", prob)
        # --- Sample category ---
        level_cat = torch.multinomial(prob, 1).squeeze(-1)
        # print("level_cat", level_cat)

        return level_cat
    
    @torch.no_grad()
    def docs_weight_gibbs(self, doc_values: dict,
                                    z_gen: torch.Tensor,
                                    z_reg: torch.Tensor,
                                    scale_constant: float,
                                    predict: bool = False,
                                    sanity_check: bool = True):
        """
        Vectorized Gibbs update of document-level stick-breaking weights (PyTorch version).

        Args:
            doc_values: dict with fields ["B"], ["Prior"], ["P"], ["G"]
            z_gen: (N, M) tensor — word category assignments
            z_reg: (N,) tensor — regression category assignments
            scale_constant: float — scaling factor
            predict: bool — skip regression updates if True
        Returns:
            Updated doc_values dict with fields ["P"], ["B"], ["G"]
        """
        N = z_gen.shape[0]
        K = self.K

        Prior0 = doc_values["P"][0].to(self.device)   # (N, K)
        Prior1 = doc_values["P"][1].to(self.device)   # (N, K)
        params = [Prior0, Prior1]

        new_params = self._docs_weight_conditional(params, z_gen, z_reg, scale_constant, predict, sanity_check)
        new_params = [param.to(self.device) for param in new_params]

        doc_values["G"] = truncated_stick_breaking(new_params[0], new_params[1], sample_shape=(), truncate_dim=-1)
        return doc_values

    @torch.no_grad()
    def _docs_weight_conditional(self,
                                params: list,
                                gen_cats: torch.Tensor,
                                reg_cats: torch.Tensor,
                                scale_constant: float,
                                predict: bool = False,
                                sanity_check: bool = True):
        """
        Compute posterior Beta parameters for document-level stick-breaking weights (PyTorch version).

        Args:
            params: list [alpha, beta] each (K,) tensor — prior Beta parameters.
            word_cats: (N_word,) tensor — word category assignments.
            reg_cats: (N_reg,) tensor — regression category assignments.
            scale_constant: float — scaling factor for posterior update.
            predict: bool — if True, skip regression category contributions.

        Returns:
            new_params: [alpha_new, beta_new] — updated Beta parameters (each (K,))
        """

        # ----------------------------
        # Construct α_bias and β_bias
        # ----------------------------
        alpha_bias, beta_bias = self._beta_data_bias(gen_cats, reg_cats, predict)

        # ----------------------------
        # Update Beta parameters
        # ----------------------------
        assert alpha_bias.shape == params[0].shape, f"Shape mismatch: alpha {alpha_bias.shape} vs params {params[0].shape}"
        alpha_new = params[0] + alpha_bias * scale_constant
        beta_new = params[1] + beta_bias * scale_constant

        return [alpha_new, beta_new]
    
    @torch.no_grad()
    def update_docs_prior(self, doc_values, rev_cat: torch.Tensor, sanity_check: bool = True):
        """
        Compute document-specific prior parameters (PyTorch version).

        Args:
            rev_cat: tensor of reversed category indices for the document
                     (e.g., shape (num_levels,)).

        Returns:
            a, b: flattened Beta prior parameters for this document
        """
        depth = len(self.cluster_dims)

        # Gather relevant G and alpha tensors at the deepest level
        G_depth = advanced_multi_index_select(self.SV[f"G{depth}"], rev_cat, dims=torch.arange(depth)).to(self.device)
        alpha_depth = advanced_multi_index_select(self.struct_params[f"alpha{depth}"], rev_cat, dims=torch.arange(depth)).to(self.device)
        if sanity_check:
            assert torch.allclose(G_depth, self.SV[f"G{depth}"][tuple(rev_cat[:, i] for i in range(rev_cat.shape[1]))])
        # Use the PyTorch version of gen_next_level_prior
        # The function should accept tensors of the same shape as in JAX version
        a, b = gen_next_level_prior(
            G_depth,
            alpha_depth
        )
        doc_values["P"] = [a, b]

        return doc_values

    @torch.no_grad()
    def _docs_cat_count(self, gen_cats: torch.Tensor,
                         reg_cats: torch.Tensor, predict: bool = False):
        """
        Compute document-level category counts (PyTorch version).

        Args:
            gen_cats: (N_word,) tensor — word category assignments.
            reg_cats: (N_reg,) tensor — regression category assignments.
        Returns:
            cat_count: (K,) tensor — counts per category.
        """
        K = self.K
        # ----------------------------
        # Compute category counts
        # ----------------------------
        cat_count = torch.zeros(gen_cats.shape[0], K, dtype=torch.int64, device=self.device)
        cat_count.scatter_add_(1, gen_cats, torch.ones_like(gen_cats))

        if not predict:
            if reg_cats.dim() == 1:
                reg_cats = reg_cats.unsqueeze(1)
            cat_count.scatter_add_(1, reg_cats, torch.ones_like(reg_cats))
        assert cat_count.shape == (gen_cats.shape[0], K), f"Shape mismatch: cat_count {cat_count.shape} vs expected {(gen_cats.shape[0], K)}"

        return cat_count

    @torch.no_grad()
    def _beta_data_bias(self, word_cats, reg_cats, predict: bool = False):
        
        alpha_bias = self._docs_cat_count(word_cats, reg_cats, predict)

        # ----------------------------
        # Compute suffix sum of counts
        # ----------------------------
        beta_bias = suffix_sum(alpha_bias)
        return alpha_bias, beta_bias


if __name__ == "__main__":
    def random_one_hot(N, M, V, generator=None, device=None, dtype=torch.float32):
        """
        Generate a (N, M, V) tensor where each [n, m, :] is a one-hot vector.

        Args:
            N, M, V: dimensions
            generator: optional torch.Generator for reproducibility
            device: optional torch.device
            dtype: torch.dtype (float32 default)

        Returns:
            (N, M, V) tensor of floats in {0, 1}, one-hot along the last dim.
        """
        # choose a random category index for each (N, M)
        idx = torch.randint(0, V, (N, M), generator=generator, device=device)
        # convert to one-hot along the last dimension
        return torch.nn.functional.one_hot(idx, num_classes=V).to(dtype)
    # --- define a toy hierarchical structure ---
    toy_struct = {"G0": 5, "G1": 3, "G2": 2}

    # --- initialize model ---
    model = HDMM(toy_struct, vocab_size=11, device=torch.device("cpu"))
    print("Model initialized.")

    # --- synthetic data ---
    N, M, V = 7, 17, 11
    generator = torch.Generator().manual_seed(0)

    # binary word presence matrix (N, M, V)
    obs = random_one_hot(N, M, V, generator=generator)

    # regression targets (N,)
    reg = torch.randn(N, generator=generator)
    # normalize to within [0, 1]
    reg = (reg - reg.min()) / (reg.max() - reg.min())

    # --- inference ---
    z_gen, z_reg, local_category_assignments, mc, log_prob = model.infer(
        obs=obs,
        reg=reg,
        num_iters=200,
        generator=generator,
    )
    print("Inference completed.")

    # --- visualize log likelihood evolution ---
    likelihood_visualization(
        log_prob.detach().cpu(),
        torch.zeros_like(log_prob.detach().cpu()),
        epoch=0
    )