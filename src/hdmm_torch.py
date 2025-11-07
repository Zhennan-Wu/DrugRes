import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal, InverseGamma, Multinomial, Categorical, Beta

from tqdm import trange
import copy

from hdmm_utils_torch import mix_weights, suffix_sum, get_unique_rows_and_positions, advanced_multi_index_select, safe_update_scatter, stats_by_label

from vis import likelihood_visualization


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
    _, _, _, value = stats_by_label(obs.reshape(-1, obs.shape[-1]), z_gen.flatten(), num_components)  # (K, V) or (V,)

    # Broadcast params if needed
    if params.dim() == 1:
        params = params.unsqueeze(0).expand(num_components, -1)  # (K, V)

    # Posterior concentration parameters
    new_params = params + value * scaling_constant

    # Sample from Dirichlet posterior for each batch
    dist = Dirichlet(new_params)
    sample = dist.sample()  # (K, V)

    return sample


def nig_posterior(obs: torch.Tensor, labels: torch.Tensor,
                  num_components: int,
                  params: list,
                  scale_constant: float = 1.0,
                  generator: torch.Generator = None):
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
    obs = torch.atleast_2d(obs).float()
    # Ensure obs is a 1D float tensor
    means, _, sum_vars, counts = stats_by_label(obs.reshape(-1, obs.shape[-1]), labels.flatten(), num_components)

    mu0, kappa0, alpha0, beta0 = [torch.as_tensor(p, dtype=torch.float32) for p in params]

    # Posterior updates
    kappa_n = kappa0 + counts * scale_constant
    mu_n = (kappa0 * mu0 + counts * scale_constant * means) / kappa_n
    alpha_n = alpha0 + counts * scale_constant / 2.0
    beta_n = beta0 + 0.5 * scale_constant * sum_vars + \
             (kappa0 * counts * scale_constant * (means - mu0) ** 2) / (2.0 * kappa_n)

    # Sample from the posterior
    sigma_sample = InverseGamma(alpha_n, beta_n).sample()
    mu_sample = Normal(mu_n, torch.sqrt(sigma_sample / kappa_n)).sample()

    return mu_sample, sigma_sample


class HDMM(nn.Module):
    def __init__(self, struct_upbd, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.struct_upbd = struct_upbd
        self.K = int(struct_upbd["G0"])
        self.param_dims = list(struct_upbd.values())[::-1]
        self.cluster_dims = self.param_dims[:-1][::-1]
        self.struct_params = nn.ParameterDict()
        self.init_tunable_hyperparameters()
        self.init_mixture_components()
        self.init_structure()
        self.best_log_prob = -torch.inf

    def init_tunable_hyperparameters(self):
        """
        Initialize tunable hyperparameters in PyTorch version.
        Each parameter is registered as nn.Parameter so they become trainable.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Random initialization helper
        def rand_uniform(shape=(), minval=0.0, maxval=1.0):
            return (minval + (maxval - minval) * torch.rand(shape, device=device))

        # Core scalar hyperparameters
        self.struct_params["gamma"] = nn.Parameter(rand_uniform((), 0.0, 100.0))
        self.struct_params["dir_alpha"] = nn.Parameter(rand_uniform((), 0.0, 1.0))
        self.struct_params["nig_mu"] = nn.Parameter(rand_uniform((), 0.0, 100.0))
        self.struct_params["nig_kappa"] = nn.Parameter(rand_uniform((), 0.0, 100.0))
        self.struct_params["nig_alpha"] = nn.Parameter(rand_uniform((), 0.0, 100.0))
        self.struct_params["nig_beta"] = nn.Parameter(rand_uniform((), 0.0, 100.0))

        # Convert struct_upbd to dimensional lists
        self.param_dims = list(self.struct_upbd.values())
        self.param_dims.reverse()
        self.cluster_dims = self.param_dims[:-1]
        self.cluster_dims.reverse()

        # Hierarchical alpha/eta initialization
        for depth in range(len(self.param_dims)):
            child_level = depth + 1

            # α parameter
            self.struct_params[f"alpha{depth}"] = nn.Parameter(rand_uniform(tuple(self.param_dims[-child_level:]), 0.0, 100.0))

            # η parameter
            if depth < len(self.struct_upbd) - 1:
                self.struct_params[f"eta{depth}"] = nn.Parameter(rand_uniform(tuple(self.param_dims[-(child_level+1):-1]), 0.0, 100.0))

    def _setup_struct_values(self, depth, param_alpha, param_beta):

        beta = Beta(param_alpha, param_beta).sample(tuple(self.param_dims[-(depth+1)],))
        assert beta.shape == tuple(self.param_dims[-(depth+1):])
        beta = torch.cat((beta[:-1], torch.ones_like(beta[-1:]))) # last stick = 1

        self.SV[f"P{depth}"] = [param_alpha, param_beta]
        # self.SV[f"Prior{depth}"] = [param_alpha.clone(), param_beta.clone()]
        # self.SV[f"Posterior{depth}"] = [param_alpha.clone(), param_beta.clone()]
        self.SV[f"G{depth}"] = mix_weights(beta)
        assert self.SV[f"G{depth}"].shape == tuple(self.param_dims[-(depth+1):])

        self.best_SV[f"P{depth}"] = [param_alpha.clone(), param_beta.clone()]
        # self.best_SV[f"Prior{depth}"] = [param_alpha.clone(), param_beta.clone()]
        self.best_SV[f"Posterior{depth}"] = [param_alpha.clone(), param_beta.clone()]

    def _setup_cluster_values(self, depth, param_alpha, param_beta):
        beta = Beta(param_alpha, param_beta).sample(tuple(self.param_dims[-(depth+3):-1],))
        beta = torch.cat((beta[:-1], torch.ones_like(beta[-1:]))) # last stick = 1
        assert beta.shape == tuple(self.param_dims[-(depth+3):-1])

        self.SV[f"LP{depth}"] = [param_alpha, param_beta]
        # self.SV[f"LPrior{depth}"] = [param_alpha.clone(), param_beta.clone()]
        # self.SV[f"LPosterior{depth}"] = [param_alpha.clone(), param_beta.clone()]
        self.SV[f"LG{depth}"] = mix_weights(beta, axis=0)
        assert self.SV[f"LG{depth}"].shape == tuple(self.param_dims[-(depth+3):-1])

        self.best_SV[f"LP{depth}"] = [param_alpha.clone(), param_beta.clone()]
        # self.best_SV[f"LPrior{depth}"] = [param_alpha.clone(), param_beta.clone()]
        self.best_SV[f"LPosterior{depth}"] = [param_alpha.clone(), param_beta.clone()]
        # self.best_SV[f"LG{depth}"] = self.SV[f"LG{depth}"].clone()

    def init_structure(self):
        device = next(iter(self.struct_params.values())).device

        self.SV = {}
        self.best_SV = {}

        for depth in range(len(self.param_dims)):
            # ----------------------------------------------
            # Hierarchical structure levels
            # ----------------------------------------------
            if (depth == 0):
                param_alpha = torch.ones(device=device)
                param_beta = self.struct_params["gamma"] 
            else:               
                param_alpha = self.struct_params[f"alpha{depth-1}"] * self.SV[f"G{depth-1}"]
                # weight = mix_weights(Beta(self.SV[f"P{depth-1}"][0], self.SV[f"P{depth-1}"][1]).sample(tuple(self.param_dims[-depth],)))
                # param_alpha = self.struct_params[f"alpha{depth-1}"] * weight
                param_beta = suffix_sum(param_alpha)

            self._setup_struct_values(depth, param_alpha, param_beta)

            # ----------------------------------------------
            # Cluster-specific local weights (η)
            # ----------------------------------------------
            if depth < len(self.param_dims) - 1:
                eta = self.struct_params[f"eta{depth}"]
                a = torch.ones_like(eta)
                self._setup_cluster_values(depth, a, eta)

    def init_latent_variables(self, obs: torch.Tensor, *args, **kwargs):
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
        device = obs.device
        N, M, _ = obs.shape

        # ------------------------------
        # Sample global latent assignments
        # ------------------------------
        z_gen = torch.randint(low=0, high=self.K, size=(N, M),  device=device)
        z_reg = torch.randint(low=0, high=self.K, size=(N,), device=device)

        # ------------------------------
        # Sample local hierarchical assignments
        # ------------------------------
        local_category_assignments = []
        for max_cat in self.cluster_dims:
            cats = torch.randint(low=0, high=max_cat, size=(N,), device=device)
            local_category_assignments.append(cats)
        local_category_assignments = torch.stack(local_category_assignments, dim=1)  # (N, num_levels)

        # ------------------------------
        # Build per-document Beta/G mixture parameters
        # ------------------------------
        doc_values = {}
        rev_idx = torch.flip(local_category_assignments, dims=[1])

        # Extract level index (deepest hierarchy)
        num_levels = len(self.cluster_dims)
        index_dims = torch.arange(num_levels, device=device)
        param_0 = advanced_multi_index_select(self.SV[f"P{num_levels}"][0], rev_idx, dims=index_dims)
        param_1 = advanced_multi_index_select(self.SV[f"P{num_levels}"][1], rev_idx, dims=index_dims)
        # weight = advanced_multi_index_select(self.SV[f"G{num_levels}"], rev_idx, dims=index_dims)
        betas = Beta(param_0, param_1).sample()
        betas = torch.cat((betas[..., :-1], torch.ones_like(betas[..., -1:])), dim=-1)  # last stick = 1
        weight = mix_weights(betas)

        doc_values["P"] = [param_0, param_1]
        # doc_values["Prior"] = [param_0.clone(), param_1.clone()]
        doc_values["G"] = weight

        return z_gen, z_reg, local_category_assignments, doc_values
    
    def _update_struct_posterior(self, lr):
        for parent_level in range(len(self.struct_upbd)):
            self.best_SV[f"Posterior{parent_level}"][0] = (1-lr)*self.best_SV[f"Posterior{parent_level}"][0] + lr*self.best_SV[f"P{parent_level}"][0]
            self.best_SV[f"Posterior{parent_level}"][1] = (1-lr)*self.best_SV[f"Posterior{parent_level}"][1] + lr*self.best_SV[f"P{parent_level}"][1]

            if (parent_level < len(self.struct_upbd) - 1):
                self.best_SV[f"LPosterior{parent_level}"][0] = (1-lr)*self.best_SV[f"LPosterior{parent_level}"][0] + lr*self.best_SV[f"LP{parent_level}"][0]
                self.best_SV[f"LPosterior{parent_level}"][1] = (1-lr)*self.best_SV[f"LPosterior{parent_level}"][1] + lr*self.best_SV[f"LP{parent_level}"][1]

        self.mixture_components_posterior["generation"] = (1-lr)*self.mixture_components_posterior["generation"] + lr*self.best_mixture_components["generation"]
        self.mixture_components_posterior["regression_mu"] = (1-lr)*self.mixture_components_posterior["regression_mu"] + lr*self.best_mixture_components["regression_mu"]
        self.mixture_components_posterior["regression_sigma"] = (1-lr)*self.mixture_components_posterior["regression_sigma"] + lr*self.best_mixture_components["regression_sigma"]

    def _set_struct_to_best(self):
        for k in ["generation", "regression_mu", "regression_sigma"]:
            self.mixture_components_posterior[k] = self.best_mixture_components[k].clone()
        for parent_level in range(len(self.param_dims)):
            self.SV[f"P{parent_level}"] = [self.best_SV[f"P{parent_level}"][0].clone(), self.best_SV[f"P{parent_level}"][1].clone()]
            # self.SV[f"Prior{parent_level}"] = [self.best_SV[f"Prior{parent_level}"][0].clone(), self.best_SV[f"Prior{parent_level}"][1].clone()]
            # self.SV[f"Posterior{parent_level}"] = [self.best_SV[f"Posterior{parent_level}"][0].clone(), self.best_SV[f"Posterior{parent_level}"][1].clone()]
            beta =  Beta(self.SV[f"P{parent_level}"][0], self.SV[f"P{parent_level}"][1]).sample()
            beta = torch.cat((beta[:-1], torch.ones_like(beta[-1:])))  # last stick = 1
            self.SV[f"G{parent_level}"] = mix_weights(beta)
            if parent_level < len(self.param_dims) - 1:
                self.SV[f"LP{parent_level}"] = [self.best_SV[f"LP{parent_level}"][0].clone(), self.best_SV[f"LP{parent_level}"][1].clone()]
                # self.SV[f"LPrior{parent_level}"] = [self.best_SV[f"LPrior{parent_level}"][0].clone(), self.best_SV[f"LPrior{parent_level}"][1].clone()]
                # self.SV[f"LPosterior{parent_level}"] = [self.best_SV[f"LPosterior{parent_level}"][0].clone(), self.best_SV[f"LPosterior{parent_level}"][1].clone()]
                cluster_beta = Beta(self.SV[f"LP{parent_level}"][0], self.SV[f"LP{parent_level}"][1]).sample()
                cluster_beta = torch.cat((cluster_beta[:-1], torch.ones_like(cluster_beta[-1:])))  # last stick = 1
                self.SV[f"LG{parent_level}"] = mix_weights(cluster_beta, axis=0)

    def _update_best_struct(self, log_prob, predict=False, **kwargs):
        if log_prob > self.best_log_prob:
            self.best_log_prob = log_prob
            for k in ["generation", "regression_mu", "regression_sigma"]:
                self.best_mixture_components_posterior[k] = self.mixture_components[k].clone()
            for parent_level in range(len(self.param_dims)):
                self.best_SV[f"P{parent_level}"] = [self.SV[f"P{parent_level}"][0].clone(), self.SV[f"P{parent_level}"][1].clone()]
                # self.best_SV[f"Prior{parent_level}"] = [self.SV[f"Prior{parent_level}"][0].clone(), self.SV[f"Prior{parent_level}"][1].clone()]
                # self.best_SV[f"Posterior{parent_level}"] = [self.SV[f"Posterior{parent_level}"][0].clone(), self.SV[f"Posterior{parent_level}"][1].clone()]
                # self.best_SV[f"G{parent_level}"] = self.SV[f"G{parent_level}"].clone()
                if parent_level < len(self.param_dims) - 1:
                    self.best_SV[f"LP{parent_level}"] = [self.SV[f"LP{parent_level}"][0].clone(), self.SV[f"LP{parent_level}"][1].clone()]
                    # self.best_SV[f"LPrior{parent_level}"] = [self.SV[f"LPrior{parent_level}"][0].clone(), self.SV[f"LPrior{parent_level}"][1].clone()]
                    # self.best_SV[f"LPosterior{parent_level}"] = [self.SV[f"LPosterior{parent_level}"][0].clone(), self.SV[f"LPosterior{parent_level}"][1].clone()]
                    # self.best_SV[f"LG{parent_level}"] = self.SV[f"LG{parent_level}"].clone()

    def _update_struct_prior(self):
        """
        PyTorch equivalent of JAX update_struct_prior().
        Refreshes Prior and G variables by sampling new Beta sticks.
        """
        device = next(iter(self.SV.values()))[0].device
        # Loop over each hierarchy level
        for parent_level in range(len(self.param_dims)):
            # -----------------------------
            # Copy Posterior -> Prior
            # -----------------------------
            # self.SV[f"Prior{parent_level}"]  = [self.best_SV[f"Posterior{parent_level}"][0].clone(), self.best_SV[f"Posterior{parent_level}"][1].clone()]
            self.SV[f"P{parent_level}"] = [self.best_SV[f"Posterior{parent_level}"][0].clone(), self.best_SV[f"Posterior{parent_level}"][1].clone()]

            # -----------------------------
            # Resample Gₗ stick-breaking weights
            # -----------------------------
            a = self.SV[f"P{parent_level}"][0]
            b = self.SV[f"P{parent_level}"][1]
            beta = Beta(a, b).sample()
            beta = torch.cat((beta[:-1], torch.ones_like(beta[-1:])))  # last stick = 1
            self.SV[f"G{parent_level}"] = mix_weights(beta)

            # -----------------------------
            # If not last level: update local sticks (L)
            # -----------------------------
            if parent_level < len(self.struct_upbd) - 1:
                # self.SV[f"LPrior{parent_level}"] = [self.best_SV[f"LPosterior{parent_level}"][0].clone(), self.best_SV[f"LPosterior{parent_level}"][1].clone()]
                self.SV[f"LP{parent_level}"] = [self.best_SV[f"LPosterior{parent_level}"][0].clone(), self.best_SV[f"LPosterior{parent_level}"][1].clone()]

                a_local = self.SV[f"LP{parent_level}"][0]
                b_local = self.SV[f"LP{parent_level}"][1]
                beta_local = Beta(a_local, b_local).sample()
                beta_local = torch.cat((beta_local[:-1], torch.ones_like(beta_local[-1:])))  # last stick = 1
                self.SV[f"LG{parent_level}"] = mix_weights(beta_local, axis=0)

        # -----------------------------
        # Copy posterior mixture components
        # -----------------------------
        for k in ["generation", "regression_mu", "regression_sigma"]:
            self.mixture_components[k] = self.mixture_components_posterior[k].clone()

    def forward(self, obs, *args, **kwargs):
        pass

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
        generator = kwargs.get("generator", torch.Generator().manual_seed(3))
        device = obs.device

        self.set_struct_to_best()

        N, M, _ = obs.shape
        reg = args[0] if len(args) > 0 else None

        log_probs = []

        # Initialize latent variables
        z_gen, z_reg, local_category_assignments, doc_values = self.init_latent_variables(obs, *args, **kwargs)
        self.update_best_latent(
            z_gen=z_gen,
            z_reg=z_reg,
            local_category_assignments=local_category_assignments,
            doc_values=doc_values,
        )

        # --- Gibbs sampling loop ---
        pbar = trange(num_iters, desc="Inference Gibbs Sampling")
        for it in pbar:
            # ------------------------
            # 1. Sample word-level categories
            # ------------------------
            z_gen = self.vectorized_word_cat_gibbs(obs, doc_values["G"])

            # ------------------------
            # 2. Sample document weights
            # ------------------------
            doc_values = self.vectorized_doc_weight_gibbs(
                doc_values,
                z_gen,
                z_reg,
                scale_constant=1.0,
                predict=True
            )

            # ------------------------
            # 3. Sample hierarchical categories
            # ------------------------
            for depth in range(len(self.cluster_dims)):
                cats, probs = self.collapsed_doc_cats_gibbs_batch(
                    depth=depth,
                    obs=obs,
                    reg=reg,
                    z_gen=z_gen,
                    z_reg=z_reg,
                    local_category_assignments=local_category_assignments,
                    predict=True
                )
                local_category_assignments[:, depth] = cats

            # ------------------------
            # 4. Update document priors
            # ------------------------
            doc_values = self.vectorized_update_docs_prior(doc_values, torch.flip(local_category_assignments, dims=[1]))

            # ------------------------
            # 5. Compute log-likelihood
            # ------------------------
            log_prob = self.compute_log_likelihood(obs, z_gen, z_reg, reg, predict=True)
            log_prob_val = log_prob.item() if torch.is_tensor(log_prob) else float(log_prob)

            if log_prob_val > max(log_probs, default=-float("inf")):
                self.update_best_latent(
                    z_gen=z_gen,
                    z_reg=z_reg,
                    local_category_assignments=local_category_assignments,
                    doc_values=doc_values,
                )

            log_probs.append(log_prob_val)
            pbar.set_description(f"Inference Gibbs Sampling (Iter {it+1}) LogProb {log_prob_val:.2f}")

        # Convert to numpy for visualization compatibility
        return (
            z_gen,
            z_reg,
            local_category_assignments,
            doc_values,
            torch.tensor(log_probs),
        )

    def infer(self, obs: torch.Tensor, *args, **kwargs):
        """
        Full Gibbs inference for the HDMM model (PyTorch version).
        """
        lr = kwargs.get("lr", 0.1)
        self._update_struct_posterior(lr)
        self._set_struct_to_best()
        num_iters = kwargs.get("num_iters", 100)
        known_cats = kwargs.get("known_cats", None)
        known_mixtures = kwargs.get("known_mixtures", None)
        known_struct = kwargs.get("known_struct", None)
        known_words = kwargs.get("known_words", None)
        datasize = kwargs.get("datasize", obs.shape[0])
        epoch = kwargs.get("epoch", 0)
        best_z_gen = None
        best_z_reg = None
        best_local_category_assignments = None
        best_doc_values = None

        if epoch > 0:
            self._update_struct_prior()

        skip_depth = []

        N, M, _ = obs.shape
        scale_constant = datasize / N
        reg = args[0] if len(args) > 0 else None
        log_probs = []

        # --- Initialize latent variables ---
        z_gen, z_reg, local_category_assignments, doc_values = self.init_latent_variables(obs, *args, **kwargs)

        if known_words is not None:
            z_gen = known_words

        # --- Freeze known category depths ---
        if known_cats is not None:
            for depth, cats in known_cats.items():
                local_category_assignments[:, depth] = cats
                skip_depth.append(depth)

        if known_mixtures is not None:
            self.mixture_components["generation"] = known_mixtures["generation"]

        # --- Gibbs Sampling Loop ---
        pbar = trange(num_iters, desc="Gibbs Sampling")

        for it in pbar:
            # ------------------------
            # 1. Sample document-level word categories
            # ------------------------
            if known_words is None:
                z_gen = self.words_cat_gibbs(obs, doc_values["G"])

            # ------------------------
            # 2. Sample regression categories
            # ------------------------
            z_reg = self.regs_cat_gibbs(reg, doc_values["G"])

            # ------------------------
            # 3. Update document-level stick-breaking weights
            # ------------------------
            doc_values = self.docs_weight_gibbs(
                doc_values,
                z_gen,
                z_reg,
                scale_constant,
                predict=False
            )

            # ------------------------
            # 4. Update hierarchical document category assignments
            # ------------------------
            for depth in range(len(self.cluster_dims)):
                if depth in skip_depth:
                    continue
                cats = self.collapsed_docs_cat_gibbs(
                    depth,
                    obs,
                    reg,
                    z_gen,
                    z_reg,
                    local_category_assignments,
                    predict=False
                )

            with torch.no_grad():
                local_category_assignments[:, depth] = cats

            # ------------------------
            # 5. Update document priors
            # ------------------------
            doc_values = self.update_docs_prior(torch.flip(local_category_assignments, dims=[1]))

            # ------------------------
            # 6. Sample generation components
            # ------------------------
            if known_mixtures is None:
                for k in range(self.K):
                    word_idx = (z_gen == k).nonzero(as_tuple=True)
                    if word_idx[0].numel() > 0:
                        obs_k = obs[word_idx]
                        self.gen_mix_gibbs(obs_k, k, scale_constant)

            # ------------------------
            # 7. Sample regression components
            # ------------------------
            for k in range(self.K):
                reg_idx = (z_reg == k).nonzero(as_tuple=True)
                if reg_idx[0].numel() > 0:
                    reg_k = reg[reg_idx]
                    self.reg_mix_gibbs(reg_k, k, scale_constant)

            # ------------------------
            # 8. Update structural weights
            # ------------------------
            if known_struct is not None:
                for depth, struct_val in known_struct.items():
                    self.SV[f"G{depth+1}"] = struct_val

                    # Collect unique category paths
                    if depth > 0:
                        unique_rows, positions = get_unique_rows_and_positions(local_category_assignments[:, :depth])
                    else:
                        unique_rows, positions = [(slice(None),)], [(slice(None),)]

                    for row, row_idx in zip(unique_rows, positions):
                        rev_cat = torch.flip(row, dims=[0]) if depth > 0 else row
                        if depth < len(self.cluster_dims):
                            self.struct_cluster_gibbs(depth, row_idx, row, rev_cat, scale_constant)
            else:
                for depth in range(len(self.param_dims)):
                    # Collect unique prefix category assignments
                    if depth > 0:
                        unique_rows, positions = get_unique_rows_and_positions(local_category_assignments[:, :depth])
                    else:
                        unique_rows, positions = [(slice(None),)], [(slice(None),)]

                    for row, row_idx in zip(unique_rows, positions):
                        rev_cat = torch.flip(row, dims=[0]) if depth > 0 else row
                        self.struct_weights_gibbs(depth, rev_cat, z_gen[row_idx], z_reg[row_idx], scale_constant)
                        if depth < len(self.cluster_dims):
                            self.struct_cluster_gibbs(depth, row_idx, row, rev_cat, scale_constant)

            # ------------------------
            # 9. Compute log-likelihood and update best state
            # ------------------------
            log_prob = self.compute_log_likelihood(obs, z_gen, z_reg, reg)
            log_probs.append(log_prob.item() if torch.is_tensor(log_prob) else float(log_prob))

            self._update_best_struct(
                log_prob
            )
            
            if (log_prob > self.best_log_prob):
                best_z_gen = z_gen.clone()
                best_z_reg = z_reg.clone()
                best_local_category_assignments = local_category_assignments.clone()
                best_doc_values = copy.deepcopy(doc_values)

            # ------------------------
            # 10. Optional visualization
            # ------------------------
            if it > 0 and it % 50 == 0:
                likelihood_visualization(torch.tensor(log_probs), torch.zeros_like(torch.tensor(log_probs)), epoch=it, log_dir=None)

            pbar.set_description(f"Gibbs Sampling (Iter {it+1}) LogProb {log_probs[-1]:.2f}")

        # ------------------------
        # Return results
        # ------------------------
        return (
            best_z_gen,
            best_z_reg,
            best_local_category_assignments,
            best_doc_values,
            torch.tensor(log_probs)
        )
        
    def compute_log_likelihood(self, obs: torch.Tensor, z_gen: torch.Tensor,
                               z_reg: torch.Tensor, reg: torch.Tensor,
                               predict: bool = False) -> torch.Tensor:
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
        device = obs.device
        log_prob = torch.tensor(0.0, device=device)

        # -------------------------------
        # Multinomial likelihood (word generation)
        # -------------------------------
        gen_param = self.mixture_components["generation"][z_gen]  # (N, M, V)
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
            sigma = torch.clamp(sigma, min=1e-8)  # ensure positive
            reg_dist = Normal(loc=mu, scale=sigma)
            reg_prob = reg_dist.log_prob(reg)  # (N,)
            log_prob = log_prob + reg_prob.sum()

        return log_prob
    
    def gen_mix_gibbs(self, obs: torch.Tensor, scale_constant: float):
        """
        Gibbs sampling step for the generation component of mixture k (PyTorch version).

        Args:
            obs_k: (N_obs, V) tensor of one-hot or count word observations assigned to component k.
            k: integer index of the component to update.
            scale_constant: scaling constant for posterior update.
        """
        device = obs.device
        dtype = obs.dtype

        # Compute prior Dirichlet parameters
        dir_alpha = self.struct_params["dir_alpha"] * torch.ones(
            (self.vocab_size,), dtype=dtype, device=device
        )

        # Sample new generation parameters from the posterior
        generation_components = dirichlet_posterior(obs, dir_alpha, scale_constant)

        self.mixture_components["generation"] = generation_components

    def reg_mix_gibbs(self, reg: torch.Tensor, scale_constant: float):
        """
        Gibbs sampling step for the regression component of mixture k (PyTorch version).

        Args:
            reg_k: (N_obs,) tensor of regression observations assigned to component k.
            k: integer index of the mixture component to update.
            scale_constant: scaling constant for posterior update.
        """
        device = reg.device

        # Extract prior NIG parameters
        mu0 = self.struct_params["nig_mu"]
        kappa0 = self.struct_params["nig_kappa"]
        alpha0 = self.struct_params["nig_alpha"]
        beta0 = self.struct_params["nig_beta"]

        # Call the PyTorch version of NIG posterior
        new_mu, new_sigma = nig_posterior(
            reg,
            (mu0, kappa0, alpha0, beta0),
            scale_constant
        )

        self.mixture_components["regression_mu"] = new_mu
        self.mixture_components["regression_sigma"] = new_sigma

    def struct_weights_gibbs(self,
                             depth: int,
                             rev_cat: torch.Tensor,
                             matching_z_gen: torch.Tensor,
                             matching_z_reg: torch.Tensor,
                             scale_constant: float):
        """
        Gibbs sampling step for updating hierarchical structural weights (PyTorch version).

        Args:
            depth: int, current hierarchical depth.
            rev_cat: tensor of reversed category indices selecting location in structure.
            matching_z_gen: tensor of generation indices matching current path.
            matching_z_reg: tensor of regression indices matching current path.
            scale_constant: scaling factor for posterior updates.
        """
        device = rev_cat.device

        # 1️⃣ Compute conditional Beta parameters for this node
        new_params = self._cat_weight_conditional(
            depth, rev_cat, matching_z_gen, matching_z_reg, scale_constant
        )

        # 2️⃣ Sample new Beta sticks
        beta = Beta(new_params[0], new_params[1]).sample()
        beta = torch.cat((beta[:-1], torch.ones_like(beta[-1:])))  # enforce final stick = 1
        new_weights = mix_weights(beta)

        # Update G and next level P
        self._update_struct_slice(depth, rev_cat, new_weights)

    def _cat_weight_conditional(self,
                                depth: int,
                                rev_cat: torch.Tensor,
                                word_cats: torch.Tensor,
                                reg_cats: torch.Tensor,
                                scale_constant: float):
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
        device = rev_cat.device
        K = self.K  # number of components

        # ----------------------------
        # Get prior parameters
        # ----------------------------
        if depth == 0:
            params = [
                self.SV["P0"][0],
                self.SV["P0"][1]
            ]
        else:
            idx_dims = torch.arange(len(rev_cat), device=device)
            params = [
                advanced_multi_index_select(self.SV[f"P{depth}"][0], rev_cat, dims=idx_dims),
                advanced_multi_index_select(self.SV[f"P{depth}"][1], rev_cat, dims=idx_dims)
            ]

        # ----------------------------
        # Compute category counts
        # ----------------------------
        alpha_bias, beta_bias = self._beta_data_bias(word_cats, reg_cats)

        # ----------------------------
        # Update Beta parameters
        # ----------------------------
        new_params = [
            params[0] + alpha_bias * scale_constant,
            params[1] + beta_bias * scale_constant
        ]

        return new_params

    def _update_struct_slice(self, depth: int, rev_cats_slice: torch.Tensor, new_weights: torch.Tensor):
        """
        Update structural weights at a specific hierarchy level and category path (PyTorch version).

        Args:
            depth: int — hierarchy level to update.
            cats: tensor — category indices path within hierarchy.
            new_weights: tensor — new mixture weights to set.
        """
        self.SV[f"G{depth}"] = safe_update_scatter(
            self.SV[f"G{depth}"],
            rev_cats_slice,
            new_weights
        )
        if depth + 1 < len(self.param_dims):
            param_alpha, param_beta = gen_next_level_prior(self.struct_params[f"alpha{depth}"], self.SV[f"G{depth}"])         
            beta = Beta(param_alpha, param_beta).sample(tuple(self.param_dims[-(depth+2)],))
            assert beta.shape == tuple(self.param_dims[-(depth+2):])
            beta = torch.cat((beta[:-1], torch.ones_like(beta[-1:]))) # last stick = 1

            self.SV[f"P{depth+1}"] = [param_alpha, param_beta]
            
    def struct_cluster_gibbs(self,
                             depth: int,
                             row_idx: torch.Tensor,
                             rev_cats: torch.Tensor,
                             local_category_assignments: torch.Tensor,
                             scale_constant: float):
        """
        Gibbs sampling step for local (intra-level) cluster weights in HDMM hierarchy (PyTorch version).

        Args:
            depth: int — hierarchy level to update.
            row_idx: tensor or int — index/indices of current data point(s) to consider.
            cats: tensor — current category indices path within hierarchy.
            local_category_assignments: (N, L) tensor — full table of local category assignments.
            scale_constant: float — scaling constant for posterior updates.
        """
        device = rev_cats.device

        # 1️⃣ Compute conditional Beta parameters for local cluster node
        new_params = self._cluster_weight_conditional(
            depth,
            rev_cats,
            local_category_assignments[:, depth][row_idx],
            scale_constant
        )

        # 2️⃣ Sample new Beta sticks
        beta_dist = Beta(*new_params).sample()
        beta = torch.cat((beta_dist[:-1], torch.ones_like(beta_dist[-1:])))  # enforce final stick = 1

        new_weights = mix_weights(beta, axis=0)

        self.SV[f"LG{depth}"] = safe_update_scatter(
            self.SV[f"LG{depth}"],
            rev_cats,
            new_weights
        )

    def _cluster_weight_conditional(self,
                                    depth: int,
                                    rev_cats: torch.Tensor,
                                    local_cluster_cats: torch.Tensor,
                                    scale_constant: float):
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
        device = rev_cats.device
        S = self.cluster_dims[depth]  # number of clusters at this depth

        # ----------------------------
        # Retrieve prior parameters from LPrior
        # ----------------------------
        params = [
            advanced_multi_index_select(self.SV[f"LP{depth}"][0], rev_cats, dims=torch.arange(len(rev_cats), device=device)),
            advanced_multi_index_select(self.SV[f"LP{depth}"][1], rev_cats, dims=torch.arange(len(rev_cats), device=device))
        ]

        # ----------------------------
        # Compute category counts
        # ----------------------------
        cat_count = torch.bincount(local_cluster_cats.flatten(), minlength=S).to(torch.float32)

        alpha_bias = cat_count
        beta_bias = suffix_sum(alpha_bias)

        # ----------------------------
        # Update Beta parameters
        # ----------------------------
        alpha_new = params[0] + alpha_bias * scale_constant
        beta_new = params[1] + beta_bias * scale_constant

        return [alpha_new, beta_new]

    def _docs_weight_conditional(self,
                                params: list,
                                gen_cats: torch.Tensor,
                                reg_cats: torch.Tensor,
                                scale_constant: float,
                                predict: bool = False):
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
        alpha_new = params[0] + alpha_bias * scale_constant
        beta_new = params[1] + beta_bias * scale_constant

        return [alpha_new, beta_new]

    
    def words_cat_gibbs(self, obs: torch.Tensor, doc_weights: torch.Tensor):
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
        device = obs.device

        # Compute log probability per topic
        gen_param = self.mixture_components["generation"]  # (K, V)
        gen_param = torch.clamp(gen_param, min=1e-12, max=1.0)
        gen_param = gen_param / gen_param.sum(-1, keepdim=True)

        log_probs = obs @ torch.log(gen_param)  # (N, M, K)
        unnormalized = log_probs + torch.log(doc_weights.unsqueeze(1) + 1e-12)
        probs = torch.softmax(unnormalized, dim=-1)    
        z_gen = torch.multinomial(probs.view(-1, K), 1).squeeze(-1).view(N, M)

        return z_gen

    def regs_cat_gibbs(self, reg: torch.Tensor, doc_weights: torch.Tensor):
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
        device = reg.device

        # Extract regression mixture components
        mu = self.mixture_components["regression_mu"]      # (K,)
        sigma = self.mixture_components["regression_sigma"]  # (K,)

        mu = mu.to(device)
        sigma = sigma.to(device)
        # Normal log probability under each component
        log_probs = Normal(mu, sigma).log_prob(reg.unsqueeze(1))  # (N, K)
        unnormalized = log_probs + torch.log(doc_weights + 1e-12)
        probs = torch.softmax(unnormalized, dim=-1)

        # Sample from categorical distribution
        z_reg = torch.multinomial(probs, 1).squeeze(-1)

        return z_reg

    def collapsed_docs_cat_gibbs(self,
                                 depth: int,
                                 obs: torch.Tensor,
                                 reg: torch.Tensor,
                                 z_gen: torch.Tensor,
                                 z_reg: torch.Tensor,
                                 parent_cats: torch.Tensor,
                                 predict: bool = False):
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
        eps = 1e-12
        device = obs.device

        # ----------------------------
        # Retrieve relevant weights
        # ----------------------------
        if depth == 0:
            weight = self.SV[f"G{depth + 1}"]
            cluster_weight = self.SV[f"LG{depth}"].flatten()
        else:
            rev_idx = torch.flip(parent_cats, dims=[0])
            struct_dims = torch.arange(1, len(rev_idx)+1, device=device)
            weight = advanced_multi_index_select(self.SV[f"G{depth + 1}"], rev_idx, dims=struct_dims) # (C, K)
            cluster_dims = torch.arange(1, len(rev_idx), device=device) 
            cluster_weight = advanced_multi_index_select(self.SV[f"LG{depth}"], rev_idx, dims=cluster_dims)
            assert cluster_weight.dim() == 1, f"cluster_weight.dim()={cluster_weight.dim()}, expected 1"

        assert weight.shape[0] == self.cluster_dims[depth], \
            f"weight.shape[0]={weight.shape[0]}, expected {self.cluster_dims[depth]}"
        assert cluster_weight.shape[0] == self.cluster_dims[depth], \
            f"cluster_weight.shape[0]={cluster_weight.shape[0]}, expected {self.cluster_dims[depth]}"

        cats_counts = self._docs_cat_count(z_gen, z_reg, predict) # (N, K)

        # Compute log probabilities under each cluster
        log_prob = torch.log(weight + 1e-12) * cats_counts.unsqueeze(0)  # (C, K)
        log_prob = torch.sum(log_prob, dim=1)                           # (C,)

        # --- Add cluster weights ---
        unnorm = log_prob + torch.log(cluster_weight + 1e-12)
        prob = torch.softmax(unnorm, dim=-1)

        # --- Sample category ---
        level_cat = torch.multinomial(prob, 1).squeeze(-1)

        return level_cat
    
    def docs_weight_gibbs(self, doc_values: dict,
                                    z_gen: torch.Tensor,
                                    z_reg: torch.Tensor,
                                    scale_constant: float,
                                    predict: bool = False):
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
        device = z_gen.device

        Prior0 = doc_values["P"][0].to(device)   # (N, K)
        Prior1 = doc_values["P"][1].to(device)   # (N, K)
        params = [Prior0, Prior1]

        new_params = self._docs_weight_conditional(params, z_gen, z_reg, scale_constant, predict)
        new_beta = Beta(new_params[0], new_params[1]).sample()
        new_beta = torch.cat([new_beta[:, :-1], torch.ones((N, 1), device=device)], dim=-1)  # last stick = 1
        new_G = mix_weights(new_beta)

        doc_values["G"] = new_G
        return doc_values

    def update_docs_prior(self, rev_cat: torch.Tensor):
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
        G_depth = advanced_multi_index_select(self.SV[f"G{depth}"], rev_cat, dims=torch.arange(depth))
        alpha_depth = advanced_multi_index_select(self.struct_params[f"alpha{depth}"], rev_cat, dims=torch.arange(depth))

        # Use the PyTorch version of gen_next_level_prior
        # The function should accept tensors of the same shape as in JAX version
        a, b = gen_next_level_prior(
            torch.atleast_2d(G_depth),
            torch.atleast_2d(alpha_depth)
        )

        return a.flatten(), b.flatten()


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
        cat_count = torch.zeros(gen_cats.shape[0], K, dtype=torch.int64)
        cat_count.scatter_add_(1, gen_cats, torch.ones_like(gen_cats))

        if not predict:
            reg_count = torch.bincount(reg_cats, minlength=K)
            cat_count = cat_count + reg_count
        
        return cat_count

    def _beta_data_bias(self, word_cats, reg_cats, predict: bool = False):
        
        alpha_bias = self._docs_cat_count(word_cats, reg_cats, predict)

        # ----------------------------
        # Compute suffix sum of counts
        # ----------------------------
        beta_bias = suffix_sum(alpha_bias)
        return alpha_bias, beta_bias
    

if __name__ == "__main__":
    # --- define a toy hierarchical structure ---
    toy_struct = {"G0": 5, "G1": 3, "G2": 2}

    # --- initialize model ---
    model = HDMM(toy_struct, vocab_size=11)
    print("Model initialized.")

    # --- synthetic data ---
    N, M, V = 7, 17, 11
    generator = torch.Generator().manual_seed(0)

    # binary word presence matrix (N, M, V)
    obs = torch.randint(0, 2, (N, M, V), generator=generator).float()

    # regression targets (N,)
    reg = torch.randn(N, generator=generator)

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
        log_prob.detach().cpu().numpy(),
        torch.zeros_like(log_prob.detach().cpu().numpy()),
        epoch=0
    )

    # --- optional: prediction test ---
    # test_data = torch.randint(0, 2, (5, M, V), generator=generator).float()
    # local_cats, doc_weights, log_prob = model.predict(
    #     obs=test_data,
    #     num_iters=50,
    #     generator=generator,
    # )
    # likelihood_visualization(
    #     log_prob.detach().cpu().numpy(),
    #     np.zeros_like(log_prob.detach().cpu().numpy()),
    #     epoch=1
    # )
    # print("Prediction completed.")