import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import lgamma
from torch.distributions import Dirichlet, Normal, InverseGamma, Multinomial, Categorical, Beta, Gamma

from tqdm import trange
import copy
import math

from hdmm_utils_torch import mix_weights, suffix_sum, get_unique_rows_and_positions, advanced_multi_index_select, safe_update_scatter, stats_by_label, safe_positive, rand_uniform, broadcast_to_largest, random_row_mix, mix_update
from vis import likelihood_visualization


def assert_valid_dirichlet_param(t: torch.Tensor):
    assert torch.isfinite(t).all(), "Found +inf/-inf in Dirichlet parameter"
    assert not torch.isnan(t).any(), "Found NaNs in Dirichlet parameter"
    assert (t > 0).all(), "Dirichlet parameters must be strictly positive"


def truncated_stick_breaking(param_alpha: torch.Tensor, param_beta: torch.Tensor, sample_shape: tuple, truncate_dim: int = -1, arg_max=False) -> torch.Tensor:
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
    if arg_max:
        beta_samples = (safe_positive(param_alpha) - 1) / (safe_positive(param_alpha) + safe_positive(param_beta) - 2)
    else:
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


def dirichlet_multinomial_logpmf(counts: torch.Tensor,
                                 alpha: torch.Tensor) -> torch.Tensor:
    """
    Vectorized log Dirichlet–Multinomial for all (n, m) pairs.

    Args
    ----
    counts : (N, K)
        Integer counts per sample.
    alpha  : (M, K)
        Dirichlet concentration parameters.

    Returns
    -------
    log_p : (N, M)
        log p(counts[n] | alpha[m]) for all pairs.
    """
    counts = counts.to(torch.float64)
    alpha  = alpha.to(torch.float64)
    assert torch.all(counts >= 0), "Counts must be non-negative."
    assert torch.all(alpha > 0), "Alpha parameters must be positive."


    # Expand to broadcast shapes (N, 1, K) and (1, M, K)
    counts_ = counts[:, None, :]      # (N, 1, K)
    alpha_  = alpha[None, :, :]       # (1, M, K)

    # Compute common terms
    alpha0 = alpha_.sum(dim=-1)       # (1, M)
    N = counts_.sum(dim=-1)           # (N, 1)

    logp = torch.lgamma(alpha0) - torch.lgamma(N + alpha0)
    logp = logp + (torch.lgamma(counts_ + alpha_) - torch.lgamma(alpha_)).sum(dim=-1)

    return logp   # shape (N, M)


def estimate_kappa_batched(
    X: torch.Tensor,          # (n0, n1, ..., K)
    theta: torch.Tensor,      # (n1, ..., K)
    kappa_init: torch.Tensor, # (n1, ..., K)
    gamma_shape=1.0,          # a0 (can be float or tensor broadcastable to kappa)
    gamma_rate=0.0,           # b0 (can be float or tensor broadcastable to kappa)
    max_iters: int = 1,
    tol: float = 1e-6,
    eps: float = 1e-8,
    max_kappa: float = 1e2,   # hard ceiling to avoid +inf
):
    """
    Fully robust Newton solver for Dirichlet concentration κ with Gamma(a0, b0) prior.

    Guarantees:
        - κ > 0
        - κ finite (no inf, no nan)
        - κ has same shape as kappa_init

    Prior: κ ~ Gamma(gamma_shape, gamma_rate)  (shape–rate parameterization)
    """

    device = X.device
    out_dtype = kappa_init.dtype

    # Work in float64 for numerical stability
    X = X.to(device=device, dtype=torch.float64)
    theta = theta.to(device=device, dtype=torch.float64)

    # scalar κ per location (shape: n1 x n2 x ... x nm)
    kappa = kappa_init[..., 0].to(device=device, dtype=torch.float64)

    # Broadcast Gamma prior params to shape of kappa
    a0 = torch.as_tensor(gamma_shape, dtype=torch.float64, device=device)
    b0 = torch.as_tensor(gamma_rate, dtype=torch.float64, device=device)
    a0 = a0.expand_as(kappa)
    b0 = b0.expand_as(kappa)

    # average log X over samples
    s = X.log().mean(dim=0)   # (..., K)

    for _ in range(max_iters):
        # 1) Force κ to be strictly positive & within finite range
        kappa = torch.clamp(kappa, min=eps, max=max_kappa)

        # 2) Compute κ·θ
        kappa_theta = kappa.unsqueeze(-1) * theta  # (..., K)

        # 3) Log-likelihood gradient g_ll and Hessian gp_ll
        term1 = torch.digamma(kappa)                              # (...)
        term2 = (theta * torch.digamma(kappa_theta)).sum(dim=-1)
        term3 = (theta * s).sum(dim=-1)
        g_ll = term1 - term2 + term3

        term1p = torch.polygamma(1, kappa)
        term2p = (theta**2 * torch.polygamma(1, kappa_theta)).sum(dim=-1)
        gp_ll = term1p - term2p

        # 4) Add Gamma(a0, b0) prior: log p(κ) = (a0-1) log κ - b0 κ + const
        #    ∂/∂κ log p(κ)      = (a0-1)/κ - b0
        #    ∂²/∂κ² log p(κ)    = -(a0-1)/κ²
        grad_prior = (a0 - 1.0) / kappa - b0
        hess_prior = -(a0 - 1.0) / (kappa ** 2 + eps)   # add eps to avoid /0

        # Posterior gradient and Hessian
        g_post = g_ll + grad_prior
        gp_post = gp_ll + hess_prior

        # Replace tiny or zero Hessian with safe small number
        gp_post = torch.where(gp_post.abs() < eps,
                              torch.full_like(gp_post, eps),
                              gp_post)

        delta = g_post / gp_post

        # 5) Newton update
        kappa_new = kappa - delta

        # ---- SAFETY LAYERS ----

        # A) positivity: if κ_new <= eps, fallback to conservative half-step
        kappa_new = torch.where(kappa_new <= eps,
                                0.5 * kappa,
                                kappa_new)

        # B) inf or nan fallback → also conservative half-step
        bad = torch.isnan(kappa_new) | torch.isinf(kappa_new)
        if bad.any():
            kappa_new = torch.where(bad, 0.5 * kappa, kappa_new)

        # C) clamp final κ to finite positive interval
        kappa_new = torch.clamp(kappa_new, min=eps, max=max_kappa)

        # 6) Convergence
        if torch.max(torch.abs(kappa_new - kappa)) < tol:
            kappa = kappa_new
            break

        kappa = kappa_new

    # Expand κ back to shape (..., K)
    K = kappa_init.shape[-1]
    kappa_final = kappa.unsqueeze(-1).expand(*kappa_init.shape)

    # Convert back to original dtype
    kappa_final = kappa_final.to(dtype=out_dtype)

    # 7) FINAL CHECK — ensure no NaN or INF
    bad = torch.isnan(kappa_final) | torch.isinf(kappa_final)
    if bad.any():
        kappa_final = torch.where(
            bad, torch.full_like(kappa_final, eps), kappa_final
        )

    # Final clamp ensures everything is finite & positive
    kappa_final = torch.clamp(kappa_final, min=eps, max=max_kappa)

    return kappa_final


class HDMM(nn.Module):
    def __init__(self, struct_upbd, gamma_alpha, gamma_rate, *args, **kwargs):
        super().__init__()
        torch.set_grad_enabled(False)
        torch.set_default_dtype(torch.float32)

        self.struct_upbd = struct_upbd
        self.K = int(struct_upbd["G0"])
        self.gamma_alpha = gamma_alpha
        self.gamma_rate = gamma_rate
        self.param_dims = list(struct_upbd.values())[::-1]
        self.cluster_dims = self.param_dims[:-1][::-1]

        self.vocab_size = kwargs.get("vocab_size", 10000)
        self.reg_weight = kwargs.get("reg_weight", 1)
        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = kwargs.get("seed", 0)
        torch.manual_seed(self.seed)

        self.best_log_prob = -torch.inf
        
        self.struct_params = {}
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
        self.struct_params["gamma"] = Gamma(5, 5).sample().to(self.device)
        print(f"Initialized gamma: {self.struct_params['gamma'].item():.4f}")
        # Hierarchical alpha/eta initialization
        for depth in range(len(self.param_dims)):
            child_level = depth + 1

            if depth < len(self.param_dims) - 1:
            # α parameter
                self.struct_params[f"gamma_prior{depth}"] = (self.gamma_alpha[depth], self.gamma_rate[depth])  # shape parameters for Gamma prior on alpha
                self.struct_params[f"alpha{depth}"] = Gamma(self.struct_params[f"gamma_prior{depth}"][0], self.struct_params[f"gamma_prior{depth}"][1]).sample(tuple(self.param_dims[-child_level:-1])).unsqueeze(-1).expand(*self.param_dims[-child_level:]).to(self.device)
                print(f"Initialized alpha{depth}: {self.struct_params[f'alpha{depth}']}")
                # η parameter
                self.struct_params[f"eta{depth}"] = Gamma(5,5).sample(tuple(self.param_dims[-child_level:-1])).to(self.device)
        self.struct_params[f"gamma_prior{len(self.param_dims) - 1}"] = (self.gamma_alpha[len(self.param_dims) - 1], self.gamma_rate[len(self.param_dims) - 1])  # shape parameters for Gamma prior on alpha
        self.struct_params[f"alpha{len(self.param_dims) - 1}"] = Gamma(self.struct_params[f"gamma_prior{len(self.param_dims) - 1}"][0], self.struct_params[f"gamma_prior{len(self.param_dims) - 1}"][1]).sample(tuple(self.param_dims)).to(self.device)
        # print(f"Initialized alpha{len(self.param_dims) - 1}: {self.struct_params[f'alpha{len(self.param_dims) - 1}']}")

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
        self.SV["dir_alpha"] = rand_uniform((self.K, self.vocab_size), 0.1/self.vocab_size, 2./self.vocab_size).to(self.device)
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
    def update_doc_cats(self, z_gen, z_reg, predict=False):
        counts = self._docs_cat_count(z_gen, z_reg, predict=predict)
        clusters, shapes = broadcast_to_largest([self.SV[f"LG{depth}"] for depth in range(len(self.cluster_dims))])
        clusters_log = sum([torch.log(clusters[depth]) for depth in range(len(self.cluster_dims))])
        struct_weight = self.struct_params[f"alpha{len(self.cluster_dims)}"] * self.SV[f"G{len(self.cluster_dims)}"]
        struct_weight = struct_weight.clamp(min=1e-6)
        dirmul = dirichlet_multinomial_logpmf(counts, struct_weight.reshape(-1, self.K))
        flat_cluster_log = clusters_log.flatten().unsqueeze(0).expand(counts.shape[0], -1)  # (N, num_categories)
        assert flat_cluster_log.shape == dirmul.shape
        cat_log = dirmul + flat_cluster_log  # (N, num_categories)
        docs_abs_cat = Categorical(logits=cat_log).sample()
        multi_idx = torch.unravel_index(docs_abs_cat, shapes)
        rev_idx = torch.stack(multi_idx, dim=1)  # (N, num_levels)
        assert rev_idx.shape == (z_gen.shape[0], len(self.cluster_dims))
        local_category_assignments = torch.flip(rev_idx, dims=[1])
        return local_category_assignments, cat_log

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
            local_category_assignments, log_cat = self.update_doc_cats(z_gen, z_reg, predict=True)
            # with torch.no_grad():
            #     for depth in range(len(self.cluster_dims)):
            #         cats = self.collapsed_docs_cat_gibbs(
            #             depth=depth,
            #             z_gen=z_gen,
            #             z_reg=z_reg,
            #             parent_cats=local_category_assignments[:, :depth],
            #             predict=True
            #         )
            #         local_category_assignments[:, depth] = cats

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
        # return (
        #     best_z_gen,
        #     best_z_reg,
        #     best_local_category_assignments,
        #     best_doc_values,
        #     torch.tensor(log_probs)
        # )
        return (
            z_gen,
            local_category_assignments,
            log_cat,
            doc_values,
            torch.tensor(log_probs)
        )
    
    @torch.no_grad()
    def pre_learn(self, obs: torch.Tensor, iter: int = 100, **kwargs):
        """
        Pre-learn the global structure of the HDMM model.
        This function can be expanded to include pre-training steps if necessary.
        Currently, it serves as a placeholder for potential future functionality.
        """
        uniform_weight = torch.ones(obs.shape[0], self.K, device=self.device) / self.K
        for _ in range(iter):
            z_gen = self.words_cat_gibbs(obs, uniform_weight)
            self.mixture_components["generation"] = self.gen_mix_gibbs(obs, z_gen, scale_constant=1.0, sanity_check=True)
        return z_gen

        
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
        known_regs = kwargs.get("known_regs", None)
        known_clusters = kwargs.get("known_clusters", None)
        datasize = kwargs.get("datasize", obs.shape[0])
        epoch = kwargs.get("epoch", 0)
        sanity_check = kwargs.get("sanity_check", True)
        plot_gap = kwargs.get("plot_gap", 50)
        log_dir = kwargs.get("log_dir", None)
        # reserve_rate = kwargs.get("reserve_rate", [1. - 1e-3, 0.99, 0.9, 0.])
        reserve_rate = kwargs.get("reserve_rate", [0., 0., 0., 0.])
        heuristic_prelearn = kwargs.get("heuristic_prelearn", True)
        word_por = reserve_rate[0]
        old_por = reserve_rate[1]
        struct_old_por = reserve_rate[2]
        param_por = reserve_rate[3]
        gamma_reg = kwargs.get("gamma_reg", None)
        if gamma_reg is None:
            gamma_reg = [False]*len(self.param_dims)
        max_kappa = kwargs.get("max_kappa", None)
        kappa_iter = kwargs.get("kappa_iter", 1)

        if max_kappa is None:
            max_kappa = [10*(i+1) for i in range(len(self.param_dims))]

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
        
        if known_regs is not None:
            z_reg = known_regs.to(self.device).to(torch.int64)
            if sanity_check:
                assert z_reg.shape == (N,)
                assert (z_reg >= 0).all() and (z_reg < self.K).all()


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
            skip_struct = [k+1 for k in skip_struct]  # shift by 1 due to 0-indexing
            for depth in range(len(self.param_dims)):
                if depth in skip_struct:
                    self.SV[f"G{depth}"] = known_struct[depth-1].to(self.device)
                    if sanity_check:
                        assert self.SV[f"G{depth}"].shape == tuple(self.param_dims[-(depth+1):]), f"for depth {depth} expected shape {tuple(self.param_dims[-(depth+1):])}, got {self.SV[f'G{depth}'].shape}"
                        assert torch.allclose(self.SV[f"G{depth}"].sum(dim=-1), torch.ones(self.SV[f"G{depth}"].shape[:-1], device=self.device))

        if known_clusters is not None:
            for depth in range(len(self.param_dims) - 1):
                if depth in known_clusters.keys():
                    self.SV[f"LG{depth}"] = known_clusters[depth].to(self.device)
                    if sanity_check:
                        assert self.SV[f"LG{depth}"].shape == tuple(self.param_dims[-(depth+2):-1]), f"for depth {depth} expected shape {tuple(self.param_dims[-(depth+2):-1])}, got {self.SV[f'LG{depth}'].shape}"
                        assert torch.allclose(self.SV[f"LG{depth}"].sum(dim=0), torch.ones(self.SV[f"LG{depth}"].shape[1:], device=self.device))

        # --- Gibbs Sampling Loop ---
        pbar = trange(num_iters, desc="Gibbs Sampling")
        if known_mixtures is None and heuristic_prelearn:
            print("initializing mixture components with pre-learning...")
            z_gen = self.pre_learn(obs)

        for it in pbar:
            data_mask = (torch.rand((N,), device=self.device) < old_por).float()

            # ------------------------
            # 1. Sample document-level word categories
            # ------------------------

            if known_words is None:
                z_gen_new = self.words_cat_gibbs(obs, doc_values["G"], sanity_check=sanity_check)
                if sanity_check:
                    assert z_gen_new.shape == (N, M)
                    assert (z_gen_new >= 0).all() and (z_gen_new < self.K).all()
                # z_gen = mix_update(z_gen, z_gen_new, mask=data_mask)
                z_gen_flatten = random_row_mix(z_gen.flatten(), z_gen_new.flatten(), p=word_por)
                z_gen = z_gen_flatten.view(N, M)
            # ------------------------
            # 2. Sample regression categories
            # ------------------------
            if known_regs is None:
                z_reg_new = self.regs_cat_gibbs(reg, doc_values["G"], sanity_check=sanity_check)
                if sanity_check:
                    assert z_reg_new.shape == (N,)
                    assert (z_reg_new >= 0).all() and (z_reg_new < self.K).all()
                # z_reg = mix_update(z_reg, z_reg_new, mask=data_mask)
                z_reg = random_row_mix(z_reg, z_reg_new, p=old_por)

            # ------------------------
            # 3. Update document-level stick-breaking weights
            # ------------------------
            doc_values_new = self.docs_weight_gibbs(
                doc_values,
                z_gen,
                z_reg,
                scale_constant,
                predict=False,
                sanity_check=sanity_check
            )
            if sanity_check:
                assert doc_values_new["G"].shape == (N, self.K)
                assert torch.allclose(doc_values_new["G"].sum(dim=-1), torch.ones(N, device=self.device))
                assert doc_values_new["P"][0].shape == (N, self.K)
                assert doc_values_new["P"][1].shape == (N, self.K)
            doc_values = {
                "G": mix_update(doc_values["G"], doc_values_new["G"], mask=data_mask),
                "P": [
                    mix_update(doc_values["P"][0], doc_values_new["P"][0], mask=data_mask),
                    mix_update(doc_values["P"][1], doc_values_new["P"][1], mask=data_mask)
                ]
            }

            # ------------------------
            # 4. Update hierarchical document category assignments
            # ------------------------
            # for depth in range(len(self.cluster_dims)):
            #     if depth in skip_depth:
            #         continue
            #     cats = self.collapsed_docs_cat_gibbs(
            #         depth,
            #         z_gen,
            #         z_reg,
            #         parent_cats=local_category_assignments[:, :depth],
            #         predict=False,
            #         sanity_check=sanity_check
            #     )
            #     if sanity_check:
            #         assert cats.shape == (N,)
            #         assert (cats >= 0).all() and (cats < self.cluster_dims[depth]).all()

            #     with torch.no_grad():
            #         local_category_assignments[:, depth] = cats
            if known_cats is None:
                local_category_assignments_new, _ = self.update_doc_cats(z_gen, z_reg)
                if sanity_check:
                    assert local_category_assignments_new.shape == (N, len(self.cluster_dims))
                    assert all(torch.all(local_category_assignments_new[:, i] < self.cluster_dims[i]) for i in range(len(self.cluster_dims)))
                # local_category_assignments = mix_update(local_category_assignments, local_category_assignments_new, mask=data_mask)
                local_category_assignments = random_row_mix(local_category_assignments, local_category_assignments_new, p=old_por)

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
                generation_components = self.gen_mix_gibbs(obs, z_gen, scale_constant, sanity_check=sanity_check)
                self.mixture_components["generation"] = random_row_mix(self.mixture_components["generation"], generation_components, p=struct_old_por)

            # ------------------------
            # 7. Sample regression components
            # ------------------------
            reg_mu_new, reg_sigma_new = self.reg_mix_gibbs(reg, z_reg, scale_constant, sanity_check=sanity_check)
            self.mixture_components["regression_mu"] = random_row_mix(self.mixture_components["regression_mu"], reg_mu_new, p=struct_old_por)
            self.mixture_components["regression_sigma"] = random_row_mix(self.mixture_components["regression_sigma"], reg_sigma_new, p=struct_old_por)

            # ------------------------
            # 8. Update structural weights
            # ------------------------

            for depth in range(len(self.param_dims)):
                struct_mask = (torch.rand((self.param_dims[-(depth+1)],), device=self.device) < struct_old_por).float()

                if depth == 0:
                    unique_rows = None
                    positions = None
                    rev_cats = None
                else:
                    unique_rows, positions = get_unique_rows_and_positions(local_category_assignments[:, :depth])
                    rev_cats = torch.flip(unique_rows, dims=[1]).to(self.device)
                if depth not in skip_struct:
                    new_G, new_Posterior = self.struct_weights_gibbs(depth, rev_cats, positions, z_gen, z_reg, scale_constant, sanity_check=sanity_check)
                    self.SV[f"G{depth}"] = mix_update(self.SV[f"G{depth}"], new_G, mask=struct_mask)
                    self.SV[f"Posterior{depth}"] = [
                        mix_update(self.SV[f"Posterior{depth}"][0], new_Posterior[0], mask=struct_mask),
                        mix_update(self.SV[f"Posterior{depth}"][1], new_Posterior[1], mask=struct_mask)
                    ]

                    if depth + 1 < len(self.param_dims):
                        param_alpha, param_beta = gen_next_level_prior(self.struct_params[f"alpha{depth}"], self.SV[f"G{depth}"])         
                        assert param_alpha.shape == tuple(self.param_dims[-(depth+1):])
                        assert param_beta.shape == tuple(self.param_dims[-(depth+1):])

                        self.SV[f"P{depth+1}"] = [param_alpha.unsqueeze(0).expand(tuple(self.param_dims[-(depth+2):])).to(self.device), param_beta.unsqueeze(0).expand(tuple(self.param_dims[-(depth+2):])).to(self.device)]
                        # self.SV[f"G{depth+1}"] = truncated_stick_breaking(self.SV[f"P{depth+1}"][0], self.SV[f"P{depth+1}"][1], sample_shape=(), truncate_dim=-1)
                    
                if known_clusters is None:
                    if depth < len(self.param_dims) - 1:
                        new_LG = self.struct_cluster_gibbs(depth, rev_cats, positions, local_category_assignments, scale_constant, sanity_check=sanity_check)
                        self.SV[f"LG{depth}"] = random_row_mix(self.SV[f"LG{depth}"], new_LG, p=struct_old_por)


            # ------------------------
            # 9. Update structural params
            # ------------------------

            # print("initial alpha0:", self.struct_params[f"alpha0"])
            prior = self.SV[f"G0"]
            param = self.struct_params[f"alpha0"]
            unique_childs, child_poses = get_unique_rows_and_positions(local_category_assignments[:, :1])
            weights = []
            for child_row, child_pos in zip(unique_childs, child_poses):
                rev_child_idx = torch.flip(child_row, dims=[0]).to(self.device)
                weight = self.SV[f"G1"][tuple(rev_child_idx)]
                weights.append(weight)
            weights = torch.stack(weights, dim=0)
            if gamma_reg[0]:
                new_param = (1 - param_por) * estimate_kappa_batched(weights, prior, param, gamma_shape=self.struct_params[f"gamma_prior0"][0], gamma_rate=self.struct_params[f"gamma_prior0"][1], max_iters=kappa_iter) + param_por * param
            else:
                new_param = (1 - param_por) * estimate_kappa_batched(weights, prior, param, max_kappa=max_kappa[0], max_iters=kappa_iter) + param_por * param
            assert_valid_dirichlet_param(new_param)
            self.struct_params[f"alpha0"] = new_param
            
            if len(self.cluster_dims) > 1:
                for depth in range(1, len(self.cluster_dims)):
                    unique_rows, positions = get_unique_rows_and_positions(local_category_assignments[:, :depth])
                    for row, pos in zip(unique_rows, positions):
                        rev_idx  = torch.flip(row, dims=[0]).to(self.device)
                        prior = self.SV[f"G{depth}"][tuple(rev_idx)]
                        param = self.struct_params[f"alpha{depth}"][tuple(rev_idx)]
                        same_parent_childs = local_category_assignments[pos]
                        unique_childs, child_poses = get_unique_rows_and_positions(same_parent_childs[:, :depth+1])
                        weights = []
                        for child_row, child_pos in zip(unique_childs, child_poses):
                            rev_child_idx = torch.flip(child_row, dims=[0]).to(self.device)
                            weight = self.SV[f"G{depth+1}"][tuple(rev_child_idx)]
                            weights.append(weight)
                        weights = torch.stack(weights, dim=0)
                        if gamma_reg[depth]:
                            new_param = (1 - param_por) * estimate_kappa_batched(weights, prior, param, gamma_shape=self.struct_params[f"gamma_prior{depth}"][0], gamma_rate=self.struct_params[f"gamma_prior{depth}"][1], max_iters=kappa_iter) + param_por * param
                        else:
                            new_param = (1 - param_por) * estimate_kappa_batched(weights, prior, param, max_kappa=max_kappa[depth], max_iters=kappa_iter) + param_por * param
                        assert_valid_dirichlet_param(new_param)
                        self.struct_params[f"alpha{depth}"] = safe_update_scatter(
                            self.struct_params[f"alpha{depth}"],
                            rev_idx,
                            new_param,
                            dim=-1
                        )
            
            unique_rows, positions = get_unique_rows_and_positions(local_category_assignments)
            for row, pos in zip(unique_rows, positions):
                rev_idx = torch.flip(row, dims=[0]).to(self.device)
                doc_weights = doc_values["G"][pos]
                prior = self.SV[f"G{len(self.param_dims)-1}"][tuple(rev_idx)]
                param = self.struct_params[f"alpha{len(self.param_dims)-1}"][tuple(rev_idx)]
                if gamma_reg[len(self.param_dims)-1]:
                    new_alpha = (1 - param_por) * estimate_kappa_batched(doc_weights, prior, param, gamma_shape=self.struct_params[f"gamma_prior{len(self.param_dims)-1}"][0], gamma_rate=self.struct_params[f"gamma_prior{len(self.param_dims)-1}"][1], max_iters=kappa_iter) + param_por * param
                else:
                    new_alpha = (1 - param_por) * estimate_kappa_batched(doc_weights, prior, param, max_kappa=max_kappa[len(self.param_dims)-1], max_iters=kappa_iter) + param_por * param
                assert_valid_dirichlet_param(new_alpha)

                self.struct_params[f"alpha{len(self.param_dims)-1}"] = safe_update_scatter(
                    self.struct_params[f"alpha{len(self.param_dims)-1}"],
                    rev_idx,
                    new_alpha,
                    dim=-1
                )

            # ------------------------
            # 9. Compute log-likelihood and update best state
            # ------------------------
            log_prob = self.compute_log_likelihood(obs, z_gen, z_reg, reg)
            log_probs.append(log_prob.item() if torch.is_tensor(log_prob) else float(log_prob))
            
            # if (log_prob > self.best_log_prob):
            #     best_z_gen = z_gen.clone()
            #     best_z_reg = z_reg.clone()
            #     best_local_category_assignments = local_category_assignments.clone()
            #     best_doc_values = {"P": [doc_values["P"][0].clone(), doc_values["P"][1].clone()], "G": doc_values["G"].clone()}

            # self._update_best_struct(
            #     log_prob
            # )
            # ------------------------
            # 10. Optional visualization
            # ------------------------
            if it > 0 and (it+1) % plot_gap == 0:
                likelihood_visualization(torch.tensor(log_probs), torch.zeros_like(torch.tensor(log_probs)), epoch=it, log_dir=log_dir)

            pbar.set_description(f"Gibbs Sampling (Iter {it}) LogProb {log_prob:.2f}")

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
        print("Final learned alpha parameters:")
        for depth in range(len(self.cluster_dims)):
            print(f"alpha {depth}", self.struct_params[f"alpha{depth}"])
            
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

        # self.mixture_components["generation"] = generation_components.to(self.device)
        self.SV["dir_alpha_Pos"] = new_dir_alpha.to(self.device)

        return generation_components.to(self.device)

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

        # self.mixture_components["regression_mu"] = new_mu.to(self.device)
        # self.mixture_components["regression_sigma"] = new_sigma.to(self.device)

        self.SV["nig_mu_Pos"] = new_nig_mu.to(self.device)
        self.SV["nig_kappa_Pos"] = new_nig_kappa.to(self.device)
        self.SV["nig_alpha_Pos"] = new_nig_alpha.to(self.device)
        self.SV["nig_beta_Pos"] = new_nig_beta.to(self.device)
        return new_mu.to(self.device), new_sigma.to(self.device)

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
        new_G, new_Posterior = self._update_struct_slice(depth, rev_cat, new_weights, new_params, sanity_check=sanity_check)
        return new_G, new_Posterior

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
                # print("depth:", depth)
                # print("params alpha:", params[0])
                # print("ref", rev_cat)
                # print("SV", self.SV[f"P{depth}"][0])
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
        # print("depth:", depth)
        # print("old params alpha", params[0])
        # print("old params beta", params[1])
        # print("alpha bias", alpha_bias)
        # print("beta bias", beta_bias)
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
            # self.SV[f"G{depth}"] = new_weights
            # self.SV[f"Posterior{depth}"] = new_params
            new_G = new_weights.to(self.device)
            new_Posterior = [new_params[0].to(self.device), new_params[1].to(self.device)]
        else:
            # self.SV[f"G{depth}"] = safe_update_scatter(
            #     self.SV[f"G{depth}"],
            #     rev_cats_slice,
            #     new_weights,
            #     dim=-1
            # )
            # if sanity_check:
            #     assert torch.allclose(self.SV[f"G{depth}"][tuple(rev_cats_slice[:, i] for i in range(rev_cats_slice.shape[1]))], new_weights)
            new_G = safe_update_scatter(
                self.SV[f"G{depth}"],
                rev_cats_slice,
                new_weights,
                dim=-1
            )
            if sanity_check:
                assert torch.allclose(new_G[tuple(rev_cats_slice[:, i] for i in range(rev_cats_slice.shape[1]))], new_weights)   
            new_Posterior = [
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
                assert torch.allclose(new_Posterior[0][tuple(rev_cats_slice[:, i] for i in range(rev_cats_slice.shape[1]))], new_params[0])
                assert torch.allclose(new_Posterior[1][tuple(rev_cats_slice[:, i] for i in range(rev_cats_slice.shape[1]))], new_params[1])
            assert new_G.shape == tuple(self.param_dims[-(depth+1):])
            assert new_Posterior[0].shape == tuple(self.param_dims[-(depth+1):])
            assert new_Posterior[1].shape == tuple(self.param_dims[-(depth+1):])      
            # self.SV[f"Posterior{depth}"] = [
            #     safe_update_scatter(
            #         self.SV[f"Posterior{depth}"][0],
            #         rev_cats_slice,
            #         new_params[0],
            #         dim=-1
            #     ),
            #     safe_update_scatter(
            #         self.SV[f"Posterior{depth}"][1],
            #         rev_cats_slice,
            #         new_params[1],
            #         dim=-1
            #     )
            # ]   
            # if sanity_check:
            #     assert torch.allclose(self.SV[f"Posterior{depth}"][0][tuple(rev_cats_slice[:, i] for i in range(rev_cats_slice.shape[1]))], new_params[0])
            #     assert torch.allclose(self.SV[f"Posterior{depth}"][1][tuple(rev_cats_slice[:, i] for i in range(rev_cats_slice.shape[1]))], new_params[1])
            # assert self.SV[f"G{depth}"].shape == tuple(self.param_dims[-(depth+1):])
            # assert self.SV[f"Posterior{depth}"][0].shape == tuple(self.param_dims[-(depth+1):])
            # assert self.SV[f"Posterior{depth}"][1].shape == tuple(self.param_dims[-(depth+1):])

        # if depth + 1 < len(self.param_dims):
        #     param_alpha, param_beta = gen_next_level_prior(self.struct_params[f"alpha{depth}"], self.SV[f"G{depth}"])         
        #     assert param_alpha.shape == tuple(self.param_dims[-(depth+1):])
        #     assert param_beta.shape == tuple(self.param_dims[-(depth+1):])

        #     self.SV[f"P{depth+1}"] = [param_alpha.unsqueeze(0).expand(tuple(self.param_dims[-(depth+2):])).to(self.device), param_beta.unsqueeze(0).expand(tuple(self.param_dims[-(depth+2):])).to(self.device)]
        return new_G, new_Posterior

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
            # self.SV[f"LG{depth}"] = new_weights
            new_LG = new_weights.to(self.device)
        else:
            new_LG = safe_update_scatter(
                self.SV[f"LG{depth}"],
                rev_cats,
                new_weights,
                dim=0
            )
            assert new_LG.shape == tuple(self.param_dims[-(depth+2):-1])
            if sanity_check:
                assert all(torch.allclose(new_LG[(torch.arange(self.cluster_dims[depth]),) + tuple(rev_cats[j, i] for i in range(rev_cats.shape[1]))], new_weights[j]) for j in range(new_weights.shape[0]))

            # self.SV[f"LG{depth}"] = safe_update_scatter(
            #     self.SV[f"LG{depth}"],
            #     rev_cats,
            #     new_weights,
            #     dim=0
            # )
            # assert self.SV[f"LG{depth}"].shape == tuple(self.param_dims[-(depth+2):-1])
            # if sanity_check:
            #     assert all(torch.allclose(self.SV[f"LG{depth}"][(torch.arange(self.cluster_dims[depth]),) + tuple(rev_cats[j, i] for i in range(rev_cats.shape[1]))], new_weights[j]) for j in range(new_weights.shape[0]))
        return new_LG

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
                expected = torch.stack([self.SV[f"LP{depth}"][0][(slice(None),) + tuple(rev_cats[j])] for j in range(rev_cats.shape[0])])
                assert torch.allclose(params[0], expected)
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
        weight = doc_weights.unsqueeze(1).expand(N, M, K)  # (N, M, K)
        weight = torch.clamp(weight, min=1e-12, max=1.0)
        weight = weight / weight.sum(-1, keepdim=True)
        unnormalized = log_probs + torch.log(weight)
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
            cat_count.scatter_add_(1, reg_cats, int(self.reg_weight)*torch.ones_like(reg_cats))
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