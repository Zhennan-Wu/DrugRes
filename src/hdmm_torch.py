import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Normal, InverseGamma, Multinomial, Categorical, Beta

from tqdm import trange
import copy

from hdmm_utils_torch import mix_weights, suffix_sum, get_unique_rows_and_positions, advanced_multi_index_select, safe_update_scatter, stats_by_label, safe_positive
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
    if truncate_dim == -1:
        beta_samples = torch.cat((beta_samples[..., :-1], torch.ones_like(beta_samples[..., -1:])), dim=-1)  # last stick = 1
        weight = mix_weights(beta_samples, axis=-1)
    elif truncate_dim == 0:
        beta_samples = torch.cat((beta_samples[:-1], torch.ones_like(beta_samples[-1:])), dim=0)  # last stick = 1
        weight = mix_weights(beta_samples, axis=0)

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
    _, _, _, value = stats_by_label(obs.reshape(-1, obs.shape[-1]), z_gen.flatten(), num_components)  # (K, V) or (V,)

    # Broadcast params if needed
    if params.dim() == 1:
        params = params.unsqueeze(0).expand(num_components, -1)  # (K, V)

    # Posterior concentration parameters
    new_params = params + value.unsqueeze(1) * scaling_constant

    # Sample from Dirichlet posterior for each batch
    dist = Dirichlet(new_params)
    sample = dist.sample()  # (K, V)

    return sample


def nig_posterior(reg: torch.Tensor, z_reg: torch.Tensor,
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
    # Ensure reg is a 1D float tensor
    means, _, sum_vars, counts = stats_by_label(reg, z_reg.flatten(), num_components)
    means = means.squeeze()
    # counts = counts.squeeze()
    # sum_vars = sum_vars.squeeze()

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
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.struct_upbd = struct_upbd
        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.vocab_size = self.kwargs.get("vocab_size", 10000)
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

        # Random initialization helper
        def rand_uniform(shape=(), minval=0.0, maxval=1.0):
            return (minval + (maxval - minval) * torch.rand(shape, device=self.device))

        # Core scalar hyperparameters
        self.struct_params["gamma"] = nn.Parameter(rand_uniform((), 0.1, 100.0)).to(self.device)
        self.struct_params["dir_alpha"] = nn.Parameter(rand_uniform((self.vocab_size,), 0.1, 1.0)).to(self.device)
        self.struct_params["nig_mu"] = nn.Parameter(rand_uniform((), 0.1, 100.0)).to(self.device)
        self.struct_params["nig_kappa"] = nn.Parameter(rand_uniform((), 0.1, 100.0)).to(self.device)
        self.struct_params["nig_alpha"] = nn.Parameter(rand_uniform((), 0.1, 100.0)).to(self.device)
        self.struct_params["nig_beta"] = nn.Parameter(rand_uniform((), 0.1, 100.0)).to(self.device)

        # Convert struct_upbd to dimensional lists
        self.param_dims = list(self.struct_upbd.values())
        self.param_dims.reverse()
        self.cluster_dims = self.param_dims[:-1]
        self.cluster_dims.reverse()

        # Hierarchical alpha/eta initialization
        for depth in range(len(self.param_dims)):
            child_level = depth + 1

            # α parameter
            self.struct_params[f"alpha{depth}"] = nn.Parameter(rand_uniform(tuple(self.param_dims[-child_level:]), 10.0, 100.0)).to(self.device)

            # η parameter
            if depth < len(self.param_dims) - 1:
                self.struct_params[f"eta{depth}"] = nn.Parameter(rand_uniform(tuple(self.param_dims[-child_level:-1]), 0.1, 1.0)).to(self.device)

    def init_structure(self):
        self.SV = {}
        self.best_SV = {}

        for depth in range(len(self.param_dims)):
            # ----------------------------------------------
            # Hierarchical structure levels
            # ----------------------------------------------
            if (depth == 0):
                param_alpha = torch.tensor(1.0).to(self.device)
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

    def _setup_struct_values(self, depth, param_alpha, param_beta):
        self.SV[f"G{depth}"] = truncated_stick_breaking(param_alpha, param_beta, sample_shape=(self.param_dims[-(depth+1)],), truncate_dim=-1)
        assert self.SV[f"G{depth}"].shape == tuple(self.param_dims[-(depth+1):])
        self.SV[f"P{depth}"] = [param_alpha.unsqueeze(0).expand(tuple(self.param_dims[-(depth+1):])), param_beta.unsqueeze(0).expand(tuple(self.param_dims[-(depth+1):]))]  
        # save posterior structure variables of a iteration for potential best structure
        self.SV[f"Posterior{depth}"] = [param for param in self.SV[f"P{depth}"]]   

        # save best structure variables from this batch
        self.best_SV[f"P{depth}"] = [param for param in self.SV[f"P{depth}"]]
        self.best_SV[f"G{depth}"] = self.SV[f"G{depth}"]
        # save posterior structure variables across batches
        self.best_SV[f"Posterior{depth}"] = [param.detach().clone() for param in self.SV[f"P{depth}"]]

    def _setup_cluster_values(self, depth, param_alpha, param_beta):

        self.SV[f"LG{depth}"] = truncated_stick_breaking(param_alpha, param_beta, sample_shape=(self.param_dims[-(depth+2)],), truncate_dim=0)
        assert self.SV[f"LG{depth}"].shape == tuple(self.param_dims[-(depth+2):-1])
        self.SV[f"LP{depth}"] = [param_alpha.unsqueeze(0).expand(tuple(self.param_dims[-(depth+2):-1])), param_beta.unsqueeze(0).expand(tuple(self.param_dims[-(depth+2):-1]))]
        # save posterior structure variables of a iteration for potential best structure    
        self.SV[f"LPosterior{depth}"] = [param for param in self.SV[f"LP{depth}"]]

        # save best structure variables from this batch
        self.best_SV[f"LP{depth}"] = [param for param in self.SV[f"LP{depth}"]]
        self.best_SV[f"LG{depth}"] = self.SV[f"LG{depth}"]
        # save posterior structure variables across batches
        self.best_SV[f"LPosterior{depth}"] = [param.detach().clone() for param in self.SV[f"LP{depth}"]]

    def init_mixture_components(self):
        """
        Initialize mixture components in PyTorch version:
        - Dirichlet topics over vocab
        - Normal–InverseGamma regression parameters
        """
        # -----------------------
        # Mixture components
        # -----------------------

        alpha = self.struct_params["nig_alpha"]
        beta = self.struct_params["nig_beta"]
        mu = self.struct_params["nig_mu"]
        kappa = self.struct_params["nig_kappa"]
        dir_alpha = self.struct_params["dir_alpha"]

        # Safe broadcasting for autograd
        alpha_vec = alpha.expand(self.K) if alpha.numel() == 1 else torch.ones(self.K, device=self.device) * alpha
        beta_vec  = beta.expand(self.K) if beta.numel() == 1 else torch.ones(self.K, device=self.device) * beta
        mu_vec    = mu.expand(self.K) if mu.numel() == 1 else torch.ones(self.K, device=self.device) * mu
        kappa_vec = kappa.expand(self.K) if kappa.numel() == 1 else torch.ones(self.K, device=self.device) * kappa

        self.mixture_components = {}
        self.best_mixture_components = {}
        self.mixture_components_posterior = {}

        # --- Generation (Dirichlet over vocabulary) ---
        dir_alpha = self.struct_params["dir_alpha"]
        self.mixture_components["generation"] = Dirichlet(dir_alpha).sample((self.K,))  # (K, vocab_size)
        assert self.mixture_components["generation"].shape == (self.K, self.vocab_size)

        # --- Regression components via NIG prior ---
        # InverseGamma(alpha, beta)
        sigma = InverseGamma(
            alpha_vec,
            beta_vec,
        ).sample()  # (K,)
        assert sigma.shape == (self.K,)

        # Normal(mu, sqrt(sigma/kappa))
        mu = Normal(
            mu_vec,
            torch.sqrt(sigma / kappa_vec)
        ).sample()
        assert mu.shape == (self.K,)

        self.mixture_components["regression_sigma"] = sigma
        self.mixture_components["regression_mu"] = mu

        # Deep copies for posterior/best tracking
        for k in ["generation", "regression_mu", "regression_sigma"]:
            self.best_mixture_components[k] = self.mixture_components[k]
            self.mixture_components_posterior[k] = self.mixture_components[k].clone()

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
        doc_params_1 = advanced_multi_index_select(params_1, rev_idx, dims=index_dims).to(self.device)
        doc_values["P"] = [doc_params_0, doc_params_1]
        doc_values["G"] = truncated_stick_breaking(doc_params_0, doc_params_1, sample_shape=(), truncate_dim=-1)

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
            self.mixture_components_posterior[k] = self.best_mixture_components[k]
        for depth in range(len(self.param_dims)):
            self.SV[f"P{depth}"] = [self.best_SV[f"P{depth}"][0], self.best_SV[f"P{depth}"][1]]

            self.SV[f"G{depth}"] = truncated_stick_breaking(self.SV[f"P{depth}"][0], self.SV[f"P{depth}"][1], sample_shape=(), truncate_dim=-1)

            if depth < len(self.param_dims) - 1:
                self.SV[f"LP{depth}"] = [self.best_SV[f"LP{depth}"][0], self.best_SV[f"LP{depth}"][1]]

                self.SV[f"LG{depth}"] = truncated_stick_breaking(self.SV[f"LP{depth}"][0], self.SV[f"LP{depth}"][1], sample_shape=(), truncate_dim=0)

    def _update_best_struct(self, log_prob):
        if log_prob > self.best_log_prob:
            self.best_log_prob = log_prob

            for k in ["generation", "regression_mu", "regression_sigma"]:
                self.best_mixture_components[k] = self.mixture_components[k]

            for parent_level in range(len(self.param_dims)):
                self.best_SV[f"P{parent_level}"] = [self.SV[f"Posterior{parent_level}"][0], self.SV[f"Posterior{parent_level}"][1]]

                if parent_level < len(self.param_dims) - 1:
                    self.best_SV[f"LP{parent_level}"] = [self.SV[f"LPosterior{parent_level}"][0], self.SV[f"LPosterior{parent_level}"][1]]

    def _update_struct_prior(self):
        """
        PyTorch equivalent of JAX update_struct_prior().
        Refreshes Prior and G variables by sampling new Beta sticks.
        """
        # Loop over each hierarchy level
        for parent_level in range(len(self.param_dims)):
            # -----------------------------
            # Copy Posterior -> Prior
            # -----------------------------
            self.SV[f"P{parent_level}"] = [self.best_SV[f"Posterior{parent_level}"][0].clone(), self.best_SV[f"Posterior{parent_level}"][1].clone()]

            # -----------------------------
            # Resample Gₗ stick-breaking weights
            # -----------------------------
            self.SV[f"G{parent_level}"] = truncated_stick_breaking(self.SV[f"P{parent_level}"][0], self.SV[f"P{parent_level}"][1], sample_shape=(), truncate_dim=-1)

            self.SV[f"Posterior{parent_level}"] = [self.SV[f"P{parent_level}"][0], self.SV[f"P{parent_level}"][1]]

            # -----------------------------
            # If not last level: update local sticks (L)
            # -----------------------------
            if parent_level < len(self.struct_upbd) - 1:
                self.SV[f"LP{parent_level}"] = [self.best_SV[f"LPosterior{parent_level}"][0].clone(), self.best_SV[f"LPosterior{parent_level}"][1].clone()]

                self.SV[f"LG{parent_level}"] = truncated_stick_breaking(self.SV[f"LP{parent_level}"][0], self.SV[f"LP{parent_level}"][1], sample_shape=(), truncate_dim=0)

                self.SV[f"LPosterior{parent_level}"] = [self.SV[f"LP{parent_level}"][0], self.SV[f"LP{parent_level}"][1]]

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
        best_z_gen = None
        best_z_reg = None
        best_local_category_assignments = None
        best_doc_values = None

        self._set_struct_to_best()

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
            for depth in range(len(self.cluster_dims)):
                cats = self.collapsed_docs_cat_gibbs(
                    depth=depth,
                    z_gen=z_gen,
                    z_reg=z_reg,
                    parent_cats=local_category_assignments[:, :depth+1],
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
                best_doc_values = copy.deepcopy(doc_values)

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
        obs = obs.to(self.device)

        if epoch > 0:
            self._update_struct_prior()

        skip_depth = []

        N, M, _ = obs.shape
        scale_constant = datasize / N
        reg = kwargs.get("reg", None)
        reg = reg.to(self.device) if reg is not None else None
        log_probs = []

        # --- Initialize latent variables ---
        z_gen, z_reg, local_category_assignments, doc_values = self.init_latent_variables(obs)

        if known_words is not None:
            z_gen = known_words

        # --- Freeze known category depths ---
        if known_cats is not None:
            for depth, cats in known_cats.items():
                local_category_assignments[:, depth] = cats
                skip_depth.append(depth)

        if known_mixtures is not None:
            self.mixture_components["generation"] = known_mixtures["generation"].to(self.device)

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
                    z_gen,
                    z_reg,
                    parent_cats=local_category_assignments[:, :depth+1],
                    predict=False
                )

            with torch.no_grad():
                local_category_assignments[:, depth] = cats

            # ------------------------
            # 5. Update document priors
            # ------------------------
            doc_values = self.update_docs_prior(doc_values, torch.flip(local_category_assignments, dims=[1]))
            # ------------------------
            # 6. Sample generation components
            # ------------------------
            if known_mixtures is None:
                self.gen_mix_gibbs(obs, z_gen, scale_constant)

            # ------------------------
            # 7. Sample regression components
            # ------------------------
            self.reg_mix_gibbs(reg, z_reg, scale_constant)

            # ------------------------
            # 8. Update structural weights
            # ------------------------
            if known_struct is not None:
                for depth, struct_val in known_struct.items():
                    self.SV[f"G{depth+1}"] = struct_val.to(self.device)

            else:
                for depth in range(len(self.param_dims)):
                    if depth in skip_depth:
                        continue
                    if depth == 0:
                        unique_rows = None
                        positions = None
                        rev_cats = None
                    else:
                        unique_rows, positions = get_unique_rows_and_positions(local_category_assignments[:, :depth])
                        rev_cats = torch.flip(unique_rows, dims=[1]).to(self.device)
                    self.struct_weights_gibbs(depth, rev_cats, positions, z_gen, z_reg, scale_constant)
                    if depth < len(self.param_dims) - 1:
                        self.struct_cluster_gibbs(depth, rev_cats, positions, local_category_assignments, scale_constant)

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
        log_prob = torch.tensor(0.0, device=self.device)

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

    def gen_mix_gibbs(self, obs: torch.Tensor, z_gen: torch.Tensor, scale_constant: float):
        """
        Gibbs sampling step for the generation component of mixture k (PyTorch version).

        Args:
            obs_k: (N_obs, V) tensor of one-hot or count word observations assigned to component k.
            k: integer index of the component to update.
            scale_constant: scaling constant for posterior update.
        """
        dtype = obs.dtype

        # Compute prior Dirichlet parameters
        dir_alpha = self.struct_params["dir_alpha"] * torch.ones(
            (self.vocab_size,), dtype=dtype, device=self.device
        )

        # Sample new generation parameters from the posterior
        generation_components = dirichlet_posterior(obs, z_gen, dir_alpha, self.K, scale_constant)

        self.mixture_components["generation"] = generation_components.to(self.device)

    def reg_mix_gibbs(self, reg: torch.Tensor, z_reg: torch.Tensor, scale_constant: float):
        """
        Gibbs sampling step for the regression component of mixture k (PyTorch version).

        Args:
            reg_k: (N_obs,) tensor of regression observations assigned to component k.
            k: integer index of the mixture component to update.
            scale_constant: scaling constant for posterior update.
        """

        # Extract prior NIG parameters
        mu0 = self.struct_params["nig_mu"]
        kappa0 = self.struct_params["nig_kappa"]
        alpha0 = self.struct_params["nig_alpha"]
        beta0 = self.struct_params["nig_beta"]

        # Call the PyTorch version of NIG posterior
        new_mu, new_sigma = nig_posterior(
            reg,
            z_reg,
            self.K,
            (mu0, kappa0, alpha0, beta0),
            scale_constant
        )
        self.mixture_components["regression_mu"] = new_mu.to(self.device)
        self.mixture_components["regression_sigma"] = new_sigma.to(self.device)

    def struct_weights_gibbs(self,
                             depth: int,
                             rev_cat: torch.Tensor,
                             row_idx: torch.Tensor,
                             z_gen: torch.Tensor,
                             z_reg: torch.Tensor,
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

        # 1️⃣ Compute conditional Beta parameters for this node
        new_params = self._cat_weight_conditional(
            depth, rev_cat, row_idx, z_gen, z_reg, scale_constant
        )
        new_params = [param.to(self.device) for param in new_params]

        # 2️⃣ Sample new Beta sticks
        new_weights = truncated_stick_breaking(new_params[0], new_params[1], sample_shape=(), truncate_dim=-1)
        new_weights = new_weights.to(self.device)

        # Update G and next level P
        self._update_struct_slice(depth, rev_cat, new_weights, new_params)

    def _cat_weight_conditional(self,
                                depth: int,
                                rev_cat: torch.Tensor,
                                row_idx: torch.Tensor,
                                z_gen: torch.Tensor,
                                z_reg: torch.Tensor,
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
        K = self.K  # number of components

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
            # print("struct ")
            # print("rev_cat:", rev_cat)
            # print("idx_dims:", idx_dims)
            # print("params[0] shape:", params[0].shape)
            # print("sv[p{depth}][0] shape:", self.SV[f"P{depth}"][0].shape)
            word_cats_group = [z_gen[i] for i in row_idx]
            reg_cats_group = [z_reg[i] for i in row_idx]
        alpha_bias, beta_bias = self._beta_group_bias(word_cats_group, reg_cats_group)
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

    def _update_struct_slice(self, depth: int, rev_cats_slice: torch.Tensor, new_weights: torch.Tensor, new_params: list):
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

        if depth + 1 < len(self.param_dims):
            param_alpha, param_beta = gen_next_level_prior(self.struct_params[f"alpha{depth}"], self.SV[f"G{depth}"])         
            self.SV[f"P{depth+1}"] = [param_alpha.unsqueeze(0).expand(tuple(self.param_dims[-(depth+2):])).to(self.device), param_beta.unsqueeze(0).expand(tuple(self.param_dims[-(depth+2):])).to(self.device)]

    def _beta_group_bias(self, word_cats_group, reg_cats_group, predict: bool = False):

        alpha_bias = torch.stack([torch.sum(self._docs_cat_count(word_cats, reg_cats, predict), dim=0) for word_cats, reg_cats in zip(word_cats_group, reg_cats_group)])

        # ----------------------------
        # Compute suffix sum of counts
        # ----------------------------
        beta_bias = suffix_sum(alpha_bias)
        return alpha_bias, beta_bias
                
    def struct_cluster_gibbs(self,
                             depth: int,
                             rev_cats: torch.Tensor,
                             row_idx: torch.Tensor,
                             cats: torch.Tensor,
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

        # 1️⃣ Compute conditional Beta parameters for local cluster node
        new_params = self._cluster_weight_conditional(
            depth,
            rev_cats,
            row_idx,
            cats,
            scale_constant
        )
        new_params = [param.to(self.device) for param in new_params]

        # 2️⃣ Sample new Beta sticks
        new_weights = truncated_stick_breaking(new_params[0], new_params[1], sample_shape=(), truncate_dim=0)
        new_weights = new_weights.to(self.device)
        if depth == 0:
            self.SV[f"LG{depth}"] = new_weights
        else:
            self.SV[f"LG{depth}"] = safe_update_scatter(
                self.SV[f"LG{depth}"],
                rev_cats,
                new_weights,
                dim=0
            )

    def _cluster_weight_conditional(self,
                                    depth: int,
                                    rev_cats: torch.Tensor,
                                    row_idx: torch.Tensor,
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
            # print("depth", depth)
            # print("rev_cats", rev_cats)
            # print("idx_dims", idx_dims)
            # print("LP shape", self.SV[f"LP{depth}"][0].shape)
            params = [
                advanced_multi_index_select(self.SV[f"LP{depth}"][0], rev_cats, dims=idx_dims).to(self.device),
                advanced_multi_index_select(self.SV[f"LP{depth}"][1], rev_cats, dims=idx_dims).to(self.device)
            ]
            cats_group = [local_cluster_cats[i][:, depth] for i in row_idx]
        # ----------------------------
        # Compute category counts
        # ----------------------------
        cat_count = torch.stack([torch.bincount(cats_group[i], minlength=S).to(torch.float32) for i in range(len(cats_group))])

        alpha_bias = cat_count
        beta_bias = suffix_sum(alpha_bias)
        alpha_bias = alpha_bias.squeeze()
        beta_bias = beta_bias.squeeze()

        assert alpha_bias.shape == params[0].shape, f"depth {depth} cluster Shape mismatch: alpha {alpha_bias.shape} vs params {params[0].shape}"

        # ----------------------------
        # Update Beta parameters
        # ----------------------------
        alpha_new = params[0] + alpha_bias * scale_constant
        beta_new = params[1] + beta_bias * scale_constant

        return [alpha_new, beta_new]

    @torch.no_grad()
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

        # Compute log probability per topic
        gen_param = self.mixture_components["generation"]  # (K, V)
        gen_param = torch.clamp(gen_param, min=1e-12, max=1.0)
        gen_param = gen_param / gen_param.sum(-1, keepdim=True)

        log_probs = obs @ torch.log(gen_param.T)  # (N, M, K)
        unnormalized = log_probs + torch.log(doc_weights.unsqueeze(1) + 1e-12)
        probs = torch.softmax(unnormalized, dim=-1)    
        z_gen = torch.multinomial(probs.view(-1, K), 1).squeeze(-1).view(N, M)

        return z_gen

    @torch.no_grad()
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

        # Extract regression mixture components
        mu = self.mixture_components["regression_mu"]      # (K,)
        sigma = self.mixture_components["regression_sigma"]  # (K,)

        mu = mu.to(self.device)
        sigma = sigma.to(self.device)
        # Normal log probability under each component
        log_probs = Normal(mu, sigma).log_prob(reg.unsqueeze(1))  # (N, K)
        unnormalized = log_probs + torch.log(doc_weights + 1e-12)
        probs = torch.softmax(unnormalized, dim=-1)

        # Sample from categorical distribution
        z_reg = torch.multinomial(probs, 1).squeeze(-1)

        return z_reg

    @torch.no_grad()
    def collapsed_docs_cat_gibbs(self,
                                 depth: int,
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

        # ----------------------------
        # Retrieve relevant weights
        # ----------------------------
        if depth == 0:
            weight = self.SV[f"G{depth + 1}"].unsqueeze(0).expand(z_gen.shape[0], -1, -1)  # (N, C, K)
            cluster_weight = self.SV[f"LG{depth}"].unsqueeze(0).expand(z_gen.shape[0], -1)  # (N, C)
        else:
            rev_idx = torch.flip(parent_cats, dims=[1])[:, :-1]
            struct_dims = torch.arange(1, rev_idx.shape[1]+1, device=self.device)
            weight = advanced_multi_index_select(self.SV[f"G{depth + 1}"], rev_idx, dims=struct_dims).to(self.device) # (C, K)
            cluster_dims = torch.arange(1, rev_idx.shape[1]+1, device=self.device) 
            cluster_weight = advanced_multi_index_select(self.SV[f"LG{depth}"], rev_idx, dims=cluster_dims).to(self.device)

        assert weight.shape[1] == self.cluster_dims[depth], \
            f"weight.shape[1]={weight.shape[1]}, expected {self.cluster_dims[depth]}"
        assert cluster_weight.shape[1] == self.cluster_dims[depth], \
            f"cluster_weight.shape[1]={cluster_weight.shape[1]}, expected {self.cluster_dims[depth]}"

        cats_counts = self._docs_cat_count(z_gen, z_reg, predict).to(self.device) # (N, K)

        # Compute log probabilities under each cluster
        log_prob = cats_counts.unsqueeze(1).to(dtype=weight.dtype) * torch.log(weight + 1e-12)                     # (N, C)
        log_prob = log_prob.sum(dim=-1)  # (N, C)

        # --- Add cluster weights ---
        unnorm = log_prob + torch.log(cluster_weight + 1e-12)  # (N, C)
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

        Prior0 = doc_values["P"][0].to(self.device)   # (N, K)
        Prior1 = doc_values["P"][1].to(self.device)   # (N, K)
        params = [Prior0, Prior1]

        new_params = self._docs_weight_conditional(params, z_gen, z_reg, scale_constant, predict)
        new_params = [param.to(self.device) for param in new_params]

        doc_values["G"] = truncated_stick_breaking(new_params[0], new_params[1], sample_shape=(), truncate_dim=-1)
        return doc_values

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
    
    def update_docs_prior(self, doc_values, rev_cat: torch.Tensor):
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

        # Use the PyTorch version of gen_next_level_prior
        # The function should accept tensors of the same shape as in JAX version
        a, b = gen_next_level_prior(
            G_depth,
            alpha_depth
        )
        doc_values["P"] = [a, b]

        return doc_values

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

        return cat_count

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
    model = HDMM(toy_struct, vocab_size=11)
    print("Model initialized.")

    # --- synthetic data ---
    N, M, V = 7, 17, 11
    generator = torch.Generator().manual_seed(0)

    # binary word presence matrix (N, M, V)
    obs = random_one_hot(N, M, V, generator=generator)

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
        log_prob.detach().cpu(),
        torch.zeros_like(log_prob.detach().cpu()),
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