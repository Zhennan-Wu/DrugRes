# hdmm_torch.py
# PyTorch rewrite of your JAX/NumPyro HDMM

import math
import copy
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Dirichlet, Beta, Normal, Multinomial, InverseGamma, Categorical
)

Tensor = torch.Tensor


# -------------------------
# Utilities (PyTorch)
# -------------------------

def tdevice(x: Tensor) -> torch.device:
    return x.device if isinstance(x, torch.Tensor) else torch.device("cpu")


def mix_weights(beta: Tensor, dim: int = -1) -> Tensor:
    """
    Compute mixture weights from stick-breaking proportions beta.
    beta: (..., K)
    Returns: weights of same shape as beta
    """
    eps = 1e-10
    one = torch.ones(1, device=beta.device, dtype=beta.dtype)

    # remaining = cumprod(1 - beta) with a leading 1 and drop last
    remaining_all = torch.cumprod(1 - beta + eps, dim=dim)
    # prepend ones along dim
    pad_shape = list(beta.shape)
    pad_shape[dim] = 1
    ones = torch.ones(pad_shape, device=beta.device, dtype=beta.dtype)
    # shift: [1, (1-v1), (1-v1)(1-v2), ..., drop last]
    remaining = torch.cat([ones, remaining_all.narrow(dim, 0, beta.size(dim)-1)], dim=dim)
    return beta * remaining


def suffix_sum(x: Tensor) -> Tensor:
    """
    Suffix sums along last dim: each entry is sum of elements to its right.
    Last element along that dim is 0.
    """
    rev = torch.flip(x, dims=[-1])
    rev_cumsum = torch.cumsum(rev, dim=-1)
    suffix = torch.flip(rev_cumsum, dims=[-1]) - x
    return torch.clamp(suffix, min=1e-10)


def gen_next_level_prior(G_parent: Tensor, alpha_param: Tensor) -> List[Tensor]:
    """
    Given parent G and alpha params, produce Beta parameters (alpha, beta)
    for stick-breaking at next level.
    Shapes broadcast as in your JAX version.
    """
    param_alpha = alpha_param * G_parent
    param_beta = suffix_sum(param_alpha)
    return [param_alpha, param_beta]


def gather_middle_slice(x: Tensor, idx: Tensor) -> Tensor:
    """
    x: (D0, D1, ..., D{k-1})
    idx: (k-2,) selecting D1..D{k-2}
    Returns: (D0, D{k-1})
    """
    assert x.ndim >= 3, "x needs at least 3 dims"
    # flatten dims 1..k-2 into one, pick row by flat index, keep D0 and last dim
    middle_shape = torch.tensor(x.shape[1:-1], device=x.device)
    if middle_shape.numel() == 0:
        # degenerate: there is no middle to index; return x[:, 0, :]
        flat_x = x.reshape(x.shape[0], -1, x.shape[-1])
        return flat_x[:, 0, :]

    strides = torch.cumprod(torch.cat([middle_shape.new_ones(1), middle_shape[:-1]]), dim=0)
    flat_index = int(torch.sum(idx.to(middle_shape.device) * strides).item())
    flat_x = x.reshape(x.shape[0], -1, x.shape[-1])
    return flat_x[:, flat_index, :]


def partial_index(a: Tensor, idx, mode: str = "clip") -> Tensor:
    """
    Return a[idx[0], idx[1], ..., idx[k-1], ...] where k < a.ndim.
    idx can be Tensor shape (k,) or a tuple of ints.
    """
    if isinstance(idx, tuple) and any(isinstance(i, slice) for i in idx):
        return a

    if isinstance(idx, tuple):
        idx = torch.tensor(idx, device=a.device, dtype=torch.long)
    else:
        idx = torch.atleast_1d(torch.as_tensor(idx, device=a.device, dtype=torch.long))

    k = idx.numel()
    prefix_shape = a.shape[:k]
    sizes = torch.tensor(prefix_shape, device=a.device)
    if mode == "clip":
        idx = torch.clamp(idx, 0, sizes - 1)
    elif mode == "wrap":
        idx = torch.remainder(idx, sizes)

    # flat index for first k dims
    strides = torch.tensor(
        list(reversed(torch.cumprod(torch.tensor(prefix_shape[::-1], device=a.device), dim=0).tolist()))[1:] + [1],
        device=a.device,
        dtype=torch.long
    )
    flat_idx = int(torch.sum(idx * strides).item())

    sub = a.reshape(-1, *a.shape[k:])
    return sub[flat_idx]


def set_by_multi_index(a: Tensor, idx, value: Tensor) -> Tensor:
    """
    Set a[idx[0], idx[1], ..., idx[k-1], ...] = value
    Works by flattening first k dims; returns a new tensor.
    """
    if isinstance(idx, tuple) and any(isinstance(i, slice) for i in idx):
        return a.clone().index_put_((idx,), value)

    if isinstance(idx, tuple):
        idx_t = torch.tensor(idx, device=a.device, dtype=torch.long)
    else:
        idx_t = torch.atleast_1d(torch.as_tensor(idx, device=a.device, dtype=torch.long))
    k = idx_t.numel()
    prefix_shape = a.shape[:k]
    tail_shape = a.shape[k:]

    flat_idx = torch.ravel_multi_index(idx_t, torch.Size(prefix_shape)).item()
    sub = a.reshape(-1, *tail_shape).clone()
    sub[flat_idx] = value
    return sub.reshape(a.shape)


def get_unique_rows_and_positions(x: Tensor):
    """
    x: (N, D) int tensor
    Returns (unique_rows: (U, D), positions: list[LongTensor indices per unique row])
    """
    # Convert to a 1D key per row by hashing (robust for moderate D)
    if x.ndim != 2:
        raise ValueError("x must be (N, D)")
    N, D = x.shape
    # simple mixed-radix hash
    base = int(x.max().item() + 1) if N > 0 else 1
    multipliers = (base ** torch.arange(D, device=x.device)).long()
    keys = (x.long() * multipliers).sum(dim=1)

    unique_keys, inv = torch.unique(keys, return_inverse=True)
    unique_rows_idx = []
    positions = []
    for i in range(unique_keys.numel()):
        pos = (inv == i).nonzero(as_tuple=False).flatten()
        unique_rows_idx.append(pos[0].item())
        positions.append(pos)
    unique_rows = x[torch.tensor(unique_rows_idx, device=x.device)]
    return unique_rows, positions


# -------------------------
# Posterior updaters
# -------------------------

def dirichlet_posterior(rng: torch.Generator, obs: Tensor, prior: Tensor, scale: float) -> Tensor:
    """
    obs: (N_obs, V) one-hot bag
    prior: (V,)
    """
    value = obs.sum(dim=0)
    new_param = prior + value * scale
    return Dirichlet(new_param).sample(rng=rng)


def nig_posterior(rng: torch.Generator, obs: Tensor, params: Tuple[Tensor, Tensor, Tensor, Tensor], scale: float):
    """
    Normal-Inverse-Gamma posterior for scalar Normal likelihood.
    params: (mu0, kappa0, alpha0, beta0) scalars (as 0-d or 1-d tensors)
    Returns (new_mu, new_sigma)
    """
    mu0, kappa0, alpha0, beta0 = params
    count = float(obs.numel())
    mean = obs.mean()
    sum_var = torch.sum((obs - mean) ** 2)

    kappa = kappa0 + count * scale
    mu = (kappa0 * mu0 + count * mean * scale) / kappa
    alpha = alpha0 + 0.5 * count * scale
    beta = beta0 + 0.5 * sum_var * scale + (kappa0 * count * scale * (mean - mu0) ** 2) / (2.0 * kappa)

    sigma = InverseGamma(alpha, beta).sample(rng=rng)
    mu_samp = Normal(mu, torch.sqrt(sigma / kappa)).sample(rng=rng)
    return mu_samp.squeeze(), sigma.squeeze()


def gaussian_mixture_posterior(
    rng: torch.Generator, score: Tensor, weight: Tensor, components: Tuple[Tensor, Tensor], unknown_latent: bool = False
) -> Tensor:
    """
    score: scalar tensor
    weight: (K,)
    components: (mu[K], sigma[K])
    """
    mu, sigma = components
    reg_dist = Normal(loc=mu, scale=torch.sqrt(sigma))
    log_probs = reg_dist.log_prob(score)  # (K,)
    if unknown_latent:
        unnorm = log_probs
    else:
        unnorm = log_probs + torch.log(weight + 1e-12)
    prob = F.softmax(unnorm, dim=-1)
    return Categorical(probs=prob).sample(rng=rng)


def topic_mixture_posterior(
    rng: torch.Generator, word: Tensor, weight: Tensor, components: Tensor
) -> Tensor:
    """
    word: (V,) one-hot
    weight: (K,)
    components: (K, V)
    """
    gen_dist = Multinomial(total_count=1.0, probs=components)  # broadcasts over K in log_prob via index
    # log P(word | k) for each k
    log_probs = gen_dist.log_prob(word.expand_as(components))  # (K,)
    unnorm = log_probs + torch.log(weight + 1e-12)
    prob = F.softmax(unnorm, dim=-1)
    return Categorical(probs=prob).sample(rng=rng)


def beta_mixture_posterior(
    rng: torch.Generator, doc_nu: Tensor, cat_param: List[Tensor], cluster_prob: Tensor
) -> Tuple[Tensor, Tensor]:
    """
    Sample a cluster index from Beta stick values and cluster priors.
    doc_nu: (C,)
    cat_param: [alpha(C,), beta(C,)]
    cluster_prob: (C,)
    """
    non_trivial_thres = 1e-2
    doc_alpha, doc_beta = cat_param
    eps = 1e-8
    doc_alpha = torch.clamp(doc_alpha, min=eps)
    doc_beta = torch.clamp(doc_beta, min=eps)
    doc_nu = torch.clamp(doc_nu, min=eps, max=1 - eps)

    # Beta log-prob for the first C-1 sticks
    log_prob_all = Beta(doc_alpha[..., :-1], doc_beta[..., :-1]).log_prob(doc_nu[..., :-1])
    lbd_mask = (doc_nu[..., :-1] > non_trivial_thres)
    upbd_mask = (doc_nu[..., :-1] < 1 - non_trivial_thres)
    masked = torch.where(lbd_mask & upbd_mask, log_prob_all, torch.zeros_like(log_prob_all))
    doc_nu_cat_log_prob = masked.sum()

    unnorm = doc_nu_cat_log_prob + torch.log(cluster_prob + 1e-12)
    prob = F.softmax(unnorm, dim=-1)
    cat = Categorical(probs=prob).sample(rng=rng)
    return cat, prob.unsqueeze(0)  # keep prob as 1D (match JAX code returning at least 1d)


# -------------------------
# HDMM (PyTorch)
# -------------------------

class HDMM(nn.Module):
    def __init__(self, struct_upbd: Dict[str, int], *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.struct_upbd = struct_upbd
        self.K = int(struct_upbd["G0"])
        self.best_log_prob = float("-inf")

        # Tunable hyperparameters (PyTorch Parameters)
        self._init_tunable_hyperparameters()

        # Derived dims
        self.param_dims = list(self.struct_upbd.values())[::-1]  # reverse
        self.cluster_dims = self.param_dims[:-1][::-1]           # no G0, then reverse back

        # Initialize structure and mixture components
        self._init_structure()
        self._init_mixture_components()

        # Best snapshots
        self.best_struct_values = copy.deepcopy(self.struct_values)
        self.best_mixture_components = copy.deepcopy(self.mixture_components)
        self.best_z_gen = None
        self.best_z_reg = None
        self.best_local_category_assignments = None
        self.best_doc_values = None

    # ---------- Hyperparameters ----------
    def _pos(self, x: Tensor) -> Tensor:
        # Enforce positivity (softplus with small offset)
        return F.softplus(x) + 1e-8

    def _init_tunable_hyperparameters(self):
        # Scalars
        self.model_gamma     = nn.Parameter(torch.rand(1))           # >0
        self.model_dir_alpha = nn.Parameter(torch.rand(1))           # >0
        self.model_nig_mu    = nn.Parameter(torch.rand(1))
        self.model_nig_kappa = nn.Parameter(torch.rand(1))           # >0
        self.model_nig_alpha = nn.Parameter(torch.rand(1))           # >0
        self.model_nig_beta  = nn.Parameter(torch.rand(1))           # >0

        # Tensors alpha/eta across hierarchy
        # Build dimension helpers (we need them here)
        dims = list(self.struct_upbd.values())[::-1]
        cluster_dims = dims[:-1][::-1]

        self.alpha_params = nn.ParameterList()
        self.eta_params   = nn.ParameterList()

        # alpha/eta for all but last level
        for parent_level in range(len(self.struct_upbd) - 1):
            child_level = parent_level + 1
            shape_base = tuple(dims[-child_level:-1])  # base (without the last K for expand)
            base_alpha = nn.Parameter(torch.rand(*shape_base))
            self.alpha_params.append(base_alpha)  # store base; expand later when used

            eta_shape = tuple(cluster_dims[:child_level]) if child_level > 0 else (1,)
            eta = nn.Parameter(torch.rand(*eta_shape))
            self.eta_params.append(eta)

        # last level alpha (no eta)
        last_idx = len(self.struct_upbd) - 1
        base_last = nn.Parameter(torch.rand(*tuple(dims[:-1])))
        self.alpha_last = base_last

        # store vocab size
        self.vocab_size = self.kwargs.get("vocab_size", 10000)

    # ---------- Structure ----------
    def _init_structure(self):
        """
        Sample initial stick-breaking weights across hierarchy
        and cluster routing weights (LG).
        """
        device = next(self.parameters()).device
        gen = torch.Generator(device=device)
        gen.manual_seed(1)

        self.struct_values = {}

        # Top level P0
        B0_a = torch.ones(self.K, device=device)
        B0_b = torch.full((self.K,), self._pos(self.model_gamma).item(), device=device)
        beta0 = Beta(B0_a, B0_b).sample(rng=gen)
        beta0[-1] = 1.0
        G0 = mix_weights(beta0)

        self.struct_values["P0"] = [B0_a, B0_b]
        self.struct_values["Prior0"] = [B0_a.clone(), B0_b.clone()]
        self.struct_values["Posterior0"] = [B0_a.clone(), B0_b.clone()]
        self.struct_values["G0"] = G0

        # Lower levels G1..G{L}
        dims = self.param_dims
        cluster_dims = self.cluster_dims

        for parent_level in range(len(self.struct_upbd) - 1):
            child_level = parent_level + 1
            full_dim = child_level + 1

            # Build alpha tensor by expanding base across last dim
            base_alpha = self.alpha_params[parent_level]
            # create expanded alpha to (..., K_last) with ones
            expanded = base_alpha.unsqueeze(-1) * torch.ones(*dims[-child_level:], device=device)
            alpha_param = expanded

            G_parent = self.struct_values[f"G{parent_level}"]

            # param_alpha shape should match dims[-full_dim:]
            # param_alpha = alpha_param * G_parent (broadcast)
            param_alpha = alpha_param * G_parent
            param_beta = suffix_sum(param_alpha)

            a = param_alpha  # already correct shape
            b = param_beta

            beta = Beta(a, b).sample(rng=gen)
            beta[..., -1] = 1.0
            G_child = mix_weights(beta)

            self.struct_values[f"P{child_level}"] = [a.clone(), b.clone()]
            self.struct_values[f"Prior{child_level}"] = [a.clone(), b.clone()]
            self.struct_values[f"Posterior{child_level}"] = [a.clone(), b.clone()]
            self.struct_values[f"G{child_level}"] = G_child.clone()

        # Cluster routing weights LG at each parent level
        for parent_level in range(len(self.struct_upbd) - 1):
            eta = self.eta_params[parent_level]
            a = torch.ones_like(eta)
            b = self._pos(eta)
            beta = Beta(a, b).sample(rng=gen)
            beta[..., -1] = 1.0
            LG = mix_weights(beta)
            self.struct_values[f"LP{parent_level}"] = [a.clone(), b.clone()]
            self.struct_values[f"LPrior{parent_level}"] = [a.clone(), b.clone()]
            self.struct_values[f"LPosterior{parent_level}"] = [a.clone(), b.clone()]
            self.struct_values[f"LG{parent_level}"] = LG.clone()

    # ---------- Mixture components ----------
    def _init_mixture_components(self):
        device = next(self.parameters()).device
        gen = torch.Generator(device=device)
        gen.manual_seed(2)

        self.mixture_components = {}

        dir_alpha = self._pos(self.model_dir_alpha)
        generation = Dirichlet(dir_alpha.expand(self.vocab_size)).sample((self.K,), rng=gen)
        self.mixture_components["generation"] = generation  # (K, V)

        # NIG regression
        alpha = self._pos(self.model_nig_alpha).expand(self.K)
        beta = self._pos(self.model_nig_beta).expand(self.K)
        sigma = InverseGamma(alpha, beta).sample(rng=gen)
        mu0 = self.model_nig_mu.expand(self.K)
        kappa = self._pos(self.model_nig_kappa).expand(self.K)
        mu = Normal(mu0, torch.sqrt(sigma / kappa)).sample(rng=gen)

        self.mixture_components["regression_sigma"] = sigma
        self.mixture_components["regression_mu"] = mu

        self.mixture_components_posterior = copy.deepcopy(self.mixture_components)

    # ---------- Latents / chains ----------
    def init_latent_variables(self, obs: Tensor, *args, **kwargs):
        device = obs.device
        N, M, _ = obs.shape
        gen = torch.Generator(device=device)
        gen.manual_seed(5)

        z_gen = torch.randint(low=0, high=self.K, size=(N, M), device=device, generator=gen)
        z_reg = torch.randint(low=0, high=self.K, size=(N,), device=device, generator=gen)

        local_category_assignments = []
        for max_cat in self.cluster_dims:
            cats = torch.randint(low=0, high=max_cat, size=(N,), device=device, generator=gen)
            local_category_assignments.append(cats)
        local_category_assignments = torch.stack(local_category_assignments, dim=1)  # (N, num_levels)

        # Build doc_values P and G by gathering according to reversed cats
        doc_values = {}
        rev_idx = torch.flip(local_category_assignments, dims=[1])

        depth = len(self.cluster_dims)
        # P_depth shapes match param_dims
        P0 = self.struct_values[f"P{depth}"][0]
        P1 = self.struct_values[f"P{depth}"][1]
        Gd = self.struct_values[f"G{depth}"]

        def gather_doc_params(P):
            # broadcast P to (N, ...) pick along middle dims
            P_exp = P.unsqueeze(0).expand(N, *P.shape)
            return gather_middle_slice(P_exp, rev_idx[0]) if rev_idx.shape[1] > 0 else P_exp[:, 0, :]

        # To mirror JAX gather across all docs, loop per doc (straightforward & clear)
        a_list, b_list, g_list = [], [], []
        for n in range(N):
            ridx = rev_idx[n]
            a_list.append(partial_index(P0, ridx))
            b_list.append(partial_index(P1, ridx))
            g_list.append(partial_index(Gd, ridx))
        param0 = torch.stack(a_list, dim=0)
        param1 = torch.stack(b_list, dim=0)
        Gdoc   = torch.stack(g_list, dim=0)

        doc_values["P"] = [param0, param1]
        doc_values["Prior"] = [param0.clone(), param1.clone()]
        doc_values["G"] = Gdoc
        return z_gen, z_reg, local_category_assignments, doc_values

    def init_markov_chain(self):
        mc = {
            "generation_components": [],
            "regression_mu": [],
            "regression_sigma": [],
        }
        for depth in range(len(self.param_dims)):
            mc[f"G{depth}"] = []
        return mc

    def update_markov_chain(self, mc):
        mc["generation_components"].append(self.mixture_components["generation"].clone())
        mc["regression_mu"].append(self.mixture_components["regression_mu"].clone())
        mc["regression_sigma"].append(self.mixture_components["regression_sigma"].clone())
        for depth in range(len(self.param_dims)):
            mc[f"G{depth}"].append(self.struct_values[f"G{depth}"].clone())
        # keep last 20
        for k in list(mc.keys()):
            if len(mc[k]) > 20:
                mc[k].pop(0)
        return mc

    # ---------- Best snapshots ----------
    def set_struct_to_best(self):
        self.struct_values = copy.deepcopy(self.best_struct_values)
        self.mixture_components = copy.deepcopy(self.best_mixture_components)

    def update_best_struct(self, log_prob: Tensor, **kwargs):
        val = float(log_prob)
        if val > self.best_log_prob:
            self.best_log_prob = val
            self.best_struct_values = copy.deepcopy(self.struct_values)
            self.best_mixture_components = copy.deepcopy(self.mixture_components)
            self.best_z_gen = copy.deepcopy(kwargs.get("z_gen"))
            self.best_z_reg = copy.deepcopy(kwargs.get("z_reg"))
            self.best_local_category_assignments = copy.deepcopy(kwargs.get("local_category_assignments"))
            self.best_doc_values = copy.deepcopy(kwargs.get("doc_values"))

    def update_best_latent(self, **kwargs):
        self.best_z_gen = copy.deepcopy(kwargs.get("z_gen"))
        self.best_z_reg = copy.deepcopy(kwargs.get("z_reg"))
        self.best_local_category_assignments = copy.deepcopy(kwargs.get("local_category_assignments"))
        self.best_doc_values = copy.deepcopy(kwargs.get("doc_values"))

    # ---------- Posterior blending ----------
    def update_struct_posterior(self, lr: float):
        # Move "best" posteriors towards current priors via EMA
        for parent_level in range(len(self.struct_upbd)):
            for i in (0, 1):
                self.best_struct_values[f"Posterior{parent_level}"][i] = \
                    (1 - lr) * self.best_struct_values[f"Posterior{parent_level}"][i] + \
                    lr * self.best_struct_values[f"P{parent_level}"][i]
            if parent_level < len(self.struct_upbd) - 1:
                for i in (0, 1):
                    self.best_struct_values[f"LPosterior{parent_level}"][i] = \
                        (1 - lr) * self.best_struct_values[f"LPosterior{parent_level}"][i] + \
                        lr * self.best_struct_values[f"LP{parent_level}"][i]

        for k in ("generation", "regression_mu", "regression_sigma"):
            self.mixture_components_posterior[k] = (1 - lr) * self.mixture_components_posterior[k] + \
                                                   lr * self.best_mixture_components[k]

    def update_struct_prior(self, seed: int = 1234):
        device = next(self.parameters()).device
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        for parent_level in range(len(self.struct_upbd)):
            self.struct_values[f"Prior{parent_level}"][0] = self.struct_values[f"Posterior{parent_level}"][0].clone()
            self.struct_values[f"Prior{parent_level}"][1] = self.struct_values[f"Posterior{parent_level}"][1].clone()
            self.struct_values[f"P{parent_level}"][0]     = self.struct_values[f"Prior{parent_level}"][0].clone()
            self.struct_values[f"P{parent_level}"][1]     = self.struct_values[f"Prior{parent_level}"][1].clone()
            beta = Beta(self.struct_values[f"P{parent_level}"][0],
                        self.struct_values[f"P{parent_level}"][1]).sample(rng=gen)
            beta[..., -1] = 1.0
            self.struct_values[f"G{parent_level}"] = mix_weights(beta)

            if parent_level < len(self.struct_upbd) - 1:
                self.struct_values[f"LPrior{parent_level}"][0] = self.best_struct_values[f"LPosterior{parent_level}"][0].clone()
                self.struct_values[f"LPrior{parent_level}"][1] = self.best_struct_values[f"LPosterior{parent_level}"][1].clone()
                self.struct_values[f"LP{parent_level}"][0]     = self.struct_values[f"LPrior{parent_level}"][0].clone()
                self.struct_values[f"LP{parent_level}"][1]     = self.struct_values[f"LPrior{parent_level}"][1].clone()
                lbeta = Beta(self.struct_values[f"LP{parent_level}"][0],
                             self.struct_values[f"LP{parent_level}"][1]).sample(rng=gen)
                lbeta[..., -1] = 1.0
                self.struct_values[f"LG{parent_level}"] = mix_weights(lbeta)

        self.mixture_components = copy.deepcopy(self.mixture_components_posterior)

    # ---------- Likelihood ----------
    def compute_log_likelihood(self, obs: Tensor, z_gen: Tensor, z_reg: Tensor, reg: Optional[Tensor], predict: bool = False) -> Tensor:
        gen_param = self.mixture_components["generation"][z_gen]  # (N, M, V)
        gen_param = torch.clamp(gen_param, min=1e-12)
        gen_param = gen_param / gen_param.sum(dim=-1, keepdim=True)
        word_ll = Multinomial(total_count=1.0, probs=gen_param).log_prob(obs).sum()

        total = word_ll
        if not predict and reg is not None:
            mu = self.mixture_components["regression_mu"][z_reg]
            sigma = self.mixture_components["regression_sigma"][z_reg]
            reg_ll = Normal(mu, sigma).log_prob(reg).sum()
            total = total + reg_ll
        return total

    # ---------- Single-site Gibbs helpers ----------
    def gen_mix_gibbs(self, rng: torch.Generator, obs_k: Tensor, k: int, scale: float):
        prior = self._pos(self.model_dir_alpha).expand(self.vocab_size)
        sample = dirichlet_posterior(rng, obs_k, prior, scale)
        self.mixture_components["generation"][k] = sample

    def reg_mix_gibbs(self, rng: torch.Generator, reg_k: Tensor, k: int, scale: float):
        mu0 = self.model_nig_mu
        kappa0 = self._pos(self.model_nig_kappa)
        alpha0 = self._pos(self.model_nig_alpha)
        beta0  = self._pos(self.model_nig_beta)
        new_mu, new_sigma = nig_posterior(rng, reg_k, (mu0, kappa0, alpha0, beta0), scale)
        self.mixture_components["regression_mu"][k] = new_mu
        self.mixture_components["regression_sigma"][k] = new_sigma

    def word_cat_gibbs(self, rng: torch.Generator, word: Tensor, weight: Tensor):
        return topic_mixture_posterior(rng, word, weight, self.mixture_components["generation"])

    def reg_cat_gibbs(self, rng: torch.Generator, score: Tensor, weight: Tensor):
        comps = (self.mixture_components["regression_mu"], self.mixture_components["regression_sigma"])
        return gaussian_mixture_posterior(rng, score, weight, comps)

    def doc_weight_conditional(self, params: List[Tensor], word_cats: Tensor, reg_cats: Tensor, scale: float, predict: bool = False):
        K = self.K
        cat_count = torch.bincount(word_cats.view(-1), minlength=K)
        if not predict and reg_cats.numel() > 0:
            cat_count = cat_count + torch.bincount(reg_cats.view(-1), minlength=K)
        alpha_bias = torch.zeros(K, dtype=torch.int32, device=cat_count.device)
        alpha_bias = alpha_bias.scatter(0, torch.arange(K, device=cat_count.device), cat_count)
        beta_bias = suffix_sum(alpha_bias.to(params[0].dtype))
        new_a = params[0] + alpha_bias.to(params[0].dtype) * scale
        new_b = params[1] + beta_bias * scale
        return [new_a, new_b]

    def doc_weight_gibbs(self, rng: torch.Generator, params: List[Tensor], z_gen: Tensor, z_reg: Tensor, scale: float, predict: bool = False):
        new_params = self.doc_weight_conditional(params, z_gen, z_reg, scale, predict)
        beta = Beta(new_params[0], new_params[1]).sample(rng=rng)
        beta[..., -1] = 1.0
        return new_params, beta

    # ---------- Vectorized Gibbs ----------
    def vectorized_word_cat_gibbs(self, rng: torch.Generator, obs: Tensor, doc_weights: Tensor) -> Tensor:
        N, M, _ = obs.shape
        z = torch.empty((N, M), dtype=torch.long, device=obs.device)
        # Loop over docs & words (keeps correctness & clarity; still vectorized in dist)
        for n in range(N):
            w_n = doc_weights[n]
            for m in range(M):
                z[n, m] = self.word_cat_gibbs(rng, obs[n, m], w_n)
        return z

    def vectorized_reg_cat_gibbs(self, rng: torch.Generator, reg: Tensor, doc_weights: Tensor) -> Tensor:
        N = reg.shape[0]
        z = torch.empty((N,), dtype=torch.long, device=reg.device)
        for n in range(N):
            z[n] = self.reg_cat_gibbs(rng, reg[n], doc_weights[n])
        return z

    def vectorized_doc_weight_gibbs(self, rng: torch.Generator, doc_values: Dict, z_gen: Tensor, z_reg: Tensor, scale: float, predict: bool = False):
        N = z_gen.shape[0]
        Prior0, Prior1 = doc_values["Prior"][0], doc_values["Prior"][1]

        alpha_new = torch.empty_like(Prior0)
        beta_new  = torch.empty_like(Prior1)
        B_new     = torch.empty_like(Prior0)
        G_new     = torch.empty_like(Prior0)

        for n in range(N):
            params = [Prior0[n], Prior1[n]]
            new_params, new_beta = self.doc_weight_gibbs(rng, params, z_gen[n], z_reg[n:n+1], scale, predict)
            alpha_new[n] = new_params[0]
            beta_new[n]  = new_params[1]
            B_new[n]     = new_beta
            G_new[n]     = mix_weights(new_beta)

        return {
            **doc_values,
            "P": [alpha_new, beta_new],
            "B": B_new,
            "G": G_new,
        }

    # ---------- Collapsed category updates ----------
    def collapsed_doc_cats_gibbs(self, rng: torch.Generator, depth: int, obs: Tensor, reg: Tensor, z_gen: Tensor, z_reg: Tensor, parent_cats: Tensor, predict: bool = False):
        if depth == 0:
            weight = self.struct_values[f"G{depth+1}"]              # (C, K)
            cluster_weight = self.struct_values[f"LG{depth}"].flatten()  # (C,)
        else:
            rev_idx = torch.flip(parent_cats, dims=[0])
            weight = partial_index(self.struct_values[f"G{depth+1}"], tuple(rev_idx.tolist()))
            cluster_weight = partial_index(self.struct_values[f"LG{depth}"], tuple(parent_cats.tolist()))
        C = weight.shape[0]
        assert C == self.cluster_dims[depth]

        # Word counts per component
        cat_counts = torch.bincount(z_gen.view(-1), minlength=self.K)
        if not predict:
            cat_counts = cat_counts + torch.bincount(z_reg.view(-1), minlength=self.K)

        # log P(words, reg | cat=c) = sum_k count_k * log weight[c,k] (+ reg term if not predict)
        log_prob = torch.log(weight + 1e-12) * cat_counts.unsqueeze(0).expand_as(weight)
        log_prob = log_prob.sum(dim=1)  # (C,)
        # add cluster prior
        unnorm = log_prob + torch.log(cluster_weight + 1e-12)
        prob = F.softmax(unnorm, dim=-1)
        cat = Categorical(probs=prob).sample(rng=rng)
        return cat, prob

    def collapsed_doc_cats_gibbs_batch(self, rng: torch.Generator, depth: int, obs: Tensor, reg: Tensor, z_gen: Tensor, z_reg: Tensor, local_category_assignments: Tensor, predict: bool = False):
        N = obs.shape[0]
        cats = torch.empty((N,), dtype=torch.long, device=obs.device)
        probs = torch.empty((N, (self.cluster_dims[depth] if depth < len(self.cluster_dims) else self.K)), device=obs.device)

        for i in range(N):
            parents = local_category_assignments[i, :depth] if depth > 0 else torch.zeros(0, dtype=torch.long, device=obs.device)
            c, p = self.collapsed_doc_cats_gibbs(rng, depth, obs[i], reg[i:i+1] if reg is not None else None, z_gen[i], z_reg[i:i+1], parents, predict)
            cats[i] = c
            probs[i, :p.numel()] = p
        return cats, probs

    # ---------- Conditionals for structure ----------
    def _cat_weight_conditional(self, depth: int, rev_cat, word_cats: Tensor, reg_cats: Tensor, scale: float):
        if depth == 0:
            params = [self.struct_values["Prior0"][0], self.struct_values["Prior0"][1]]
        else:
            params = [partial_index(self.struct_values[f"Prior{depth}"][0], tuple(rev_cat.tolist())),
                      partial_index(self.struct_values[f"Prior{depth}"][1], tuple(rev_cat.tolist()))]
        K = self.K
        cat_count = torch.bincount(word_cats.view(-1), minlength=K) + torch.bincount(reg_cats.view(-1), minlength=K)
        alpha_bias = torch.zeros(K, dtype=torch.int32, device=cat_count.device).scatter(0, torch.arange(K, device=cat_count.device), cat_count)
        beta_bias = suffix_sum(alpha_bias.to(params[0].dtype))
        new_a = params[0] + alpha_bias.to(params[0].dtype) * scale
        new_b = params[1] + beta_bias * scale
        return [new_a, new_b]

    def _cluster_weight_conditional(self, depth: int, cats, local_cluster_cats: Tensor, scale: float):
        params = [partial_index(self.struct_values[f"LPrior{depth}"][0], tuple(cats.tolist())),
                  partial_index(self.struct_values[f"LPrior{depth}"][1], tuple(cats.tolist()))]
        C = self.cluster_dims[depth]
        cat_count = torch.bincount(local_cluster_cats.view(-1), minlength=C)
        alpha_bias = torch.zeros(C, dtype=torch.int32, device=cat_count.device).scatter(0, torch.arange(C, device=cat_count.device), cat_count)
        beta_bias = suffix_sum(alpha_bias.to(params[0].dtype))
        new_a = params[0] + alpha_bias.to(params[0].dtype) * scale
        new_b = params[1] + beta_bias * scale
        return [new_a, new_b]

    def struct_weights_gibbs(self, rng: torch.Generator, depth: int, rev_cat, matching_z_gen: Tensor, matching_z_reg: Tensor, scale: float):
        new_params = self._cat_weight_conditional(depth, rev_cat, matching_z_gen, matching_z_reg, scale)
        beta = Beta(new_params[0], new_params[1]).sample(rng=rng)
        beta[..., -1] = 1.0

        self.struct_values[f"P{depth}"][0] = set_by_multi_index(self.struct_values[f"P{depth}"][0], tuple(rev_cat.tolist()), new_params[0])
        self.struct_values[f"P{depth}"][1] = set_by_multi_index(self.struct_values[f"P{depth}"][1], tuple(rev_cat.tolist()), new_params[1])
        self.struct_values[f"G{depth}"]    = set_by_multi_index(self.struct_values[f"G{depth}"], tuple(rev_cat.tolist()), mix_weights(beta))

    def struct_cluster_gibbs(self, rng: torch.Generator, depth: int, row_idx, cats, local_category_assignments: Tensor, scale: float):
        new_params = self._cluster_weight_conditional(depth, cats, local_category_assignments[:, depth][row_idx], scale)
        beta = Beta(new_params[0], new_params[1]).sample(rng=rng)
        beta[..., -1] = 1.0
        self.struct_values[f"LP{depth}"][0] = set_by_multi_index(self.struct_values[f"LP{depth}"][0], tuple(cats.tolist()), new_params[0])
        self.struct_values[f"LP{depth}"][1] = set_by_multi_index(self.struct_values[f"LP{depth}"][1], tuple(cats.tolist()), new_params[1])
        self.struct_values[f"LG{depth}"]    = set_by_multi_index(self.struct_values[f"LG{depth}"],    tuple(cats.tolist()), mix_weights(beta))

    # ---------- High-level API ----------
    def forward(self, obs: Tensor, *args, **kwargs) -> Tensor:
        z_gen, z_reg, local_category_assignments, mc, doc_values, log_prob = self.infer(obs, *args, **kwargs)
        return -log_prob  # for optimizers that minimize

    @torch.no_grad()
    def predict(self, obs: Tensor, *args, **kwargs):
        num_iters = kwargs.get("num_iters", 100)
        device = obs.device
        reg = args[0] if len(args) > 0 else None

        self.set_struct_to_best()

        gen = torch.Generator(device=device)
        gen.manual_seed(kwargs.get("seed", 3))

        N, M, _ = obs.shape
        log_probs: List[Tensor] = []

        z_gen, z_reg, local_category_assignments, doc_values = self.init_latent_variables(obs, *args, **kwargs)
        self.update_best_latent(z_gen=z_gen, z_reg=z_reg, local_category_assignments=local_category_assignments, doc_values=doc_values)

        for it in range(num_iters):
            z_gen = self.vectorized_word_cat_gibbs(gen, obs, doc_values["G"])
            z_reg = self.vectorized_reg_cat_gibbs(gen, reg, doc_values["G"])

            doc_values = self.vectorized_doc_weight_gibbs(gen, doc_values, z_gen, z_reg, scale=1.0, predict=True)

            for depth in range(len(self.cluster_dims)):
                cats, _ = self.collapsed_doc_cats_gibbs_batch(gen, depth, obs, reg, z_gen, z_reg, local_category_assignments, predict=True)
                local_category_assignments[:, depth] = cats

            # Update priors for next iter
            doc_values = self.update_doc_prior_batch(doc_values, local_category_assignments)

            lp = self.compute_log_likelihood(obs, z_gen, z_reg, reg, predict=True)
            if len(log_probs) == 0 or float(lp) > float(torch.stack(log_probs).max()):
                self.update_best_latent(z_gen=z_gen, z_reg=z_reg, local_category_assignments=local_category_assignments, doc_values=doc_values)
            log_probs.append(lp)

        return z_gen, z_reg, local_category_assignments, doc_values, torch.stack(log_probs).cpu().numpy()

    @torch.no_grad()
    def infer(self, obs: Tensor, *args, **kwargs):
        lr = kwargs.get("lr", 0.1)
        self.update_struct_posterior(lr)
        self.set_struct_to_best()

        num_iters = kwargs.get("num_iters", 100)
        device = obs.device
        reg = args[0] if len(args) > 0 else None
        known_cats  = kwargs.get("known_cats", None)    # dict depth -> cat indices (N,)
        known_mixes = kwargs.get("known_mixtures", None)
        known_struct= kwargs.get("known_struct", None)
        known_words = kwargs.get("known_words", None)
        datasize    = kwargs.get("datasize", obs.shape[0])
        epoch       = kwargs.get("epoch", 0)

        if epoch > 0:
            self.update_struct_prior(seed=kwargs.get("prior_seed", 1234))

        N, M, _ = obs.shape
        scale_constant = float(datasize) / float(N)

        gen_rng = torch.Generator(device=device)
        gen_rng.manual_seed(kwargs.get("seed", 4))

        log_probs: List[Tensor] = []
        mc = self.init_markov_chain()

        z_gen, z_reg, local_category_assignments, doc_values = self.init_latent_variables(obs, *args, **kwargs)
        self.update_best_latent(z_gen=z_gen, z_reg=z_reg, local_category_assignments=local_category_assignments, doc_values=doc_values)

        if known_words is not None:
            z_gen = known_words.clone()

        skip_depth = []
        if known_cats is not None:
            for depth, cats in known_cats.items():
                local_category_assignments[:, depth] = cats.to(device)
                skip_depth.append(depth)

        if known_mixes is not None:
            self.mixture_components["generation"] = known_mixes["generation"].to(device)

        for it in range(num_iters):
            if known_words is None:
                z_gen = self.vectorized_word_cat_gibbs(gen_rng, obs, doc_values["G"])
            z_reg = self.vectorized_reg_cat_gibbs(gen_rng, reg, doc_values["G"])

            doc_values = self.vectorized_doc_weight_gibbs(gen_rng, doc_values, z_gen, z_reg, scale_constant)

            for depth in range(len(self.cluster_dims)):
                if depth in skip_depth:
                    continue
                cats, _ = self.collapsed_doc_cats_gibbs_batch(gen_rng, depth, obs, reg, z_gen, z_reg, local_category_assignments)
                local_category_assignments[:, depth] = cats

            doc_values = self.update_doc_prior_batch(doc_values, local_category_assignments)

            # Mixture components
            if known_mixes is None:
                for k in range(self.K):
                    idx = (z_gen == k).nonzero(as_tuple=False)
                    if idx.numel() > 0:
                        obs_k = obs[idx[:, 0], idx[:, 1]]
                        self.gen_mix_gibbs(gen_rng, obs_k, k, scale_constant)

            for k in range(self.K):
                idx = (z_reg == k).nonzero(as_tuple=False).flatten()
                if idx.numel() > 0:
                    self.reg_mix_gibbs(gen_rng, reg[idx], k, scale_constant)

            # Structural weights
            if known_struct is not None:
                for depth, struct_val in known_struct.items():
                    self.struct_values[f"G{depth+1}"] = struct_val.to(device)
                    if depth > 0:
                        unique_rows, positions = get_unique_rows_and_positions(local_category_assignments[:, :depth])
                    else:
                        unique_rows, positions = torch.zeros(1, 0, dtype=torch.long, device=device), [torch.arange(N, device=device)]
                    for row, row_idx in zip(unique_rows, positions):
                        if depth < len(self.cluster_dims):
                            self.struct_cluster_gibbs(gen_rng, depth, row_idx, row, local_category_assignments, scale_constant)
            else:
                for depth in range(len(self.param_dims)):
                    if depth > 0:
                        unique_rows, positions = get_unique_rows_and_positions(local_category_assignments[:, :depth])
                    else:
                        unique_rows, positions = torch.zeros(1, 0, dtype=torch.long, device=device), [torch.arange(N, device=device)]
                    for row, row_idx in zip(unique_rows, positions):
                        rev_cat = torch.flip(row, dims=[0]) if depth > 0 else row
                        self.struct_weights_gibbs(gen_rng, depth, rev_cat, z_gen[row_idx], z_reg[row_idx], scale_constant)
                        if depth < len(self.cluster_dims):
                            self.struct_cluster_gibbs(gen_rng, depth, row_idx, row, local_category_assignments, scale_constant)

            lp = self.compute_log_likelihood(obs, z_gen, z_reg, reg)
            self.update_best_struct(lp, z_gen=z_gen, z_reg=z_reg, local_category_assignments=local_category_assignments, doc_values=doc_values)
            log_probs.append(lp)

            mc = self.update_markov_chain(mc)

        return z_gen, z_reg, local_category_assignments, mc, doc_values, torch.stack(log_probs).cpu().numpy()

    # ---------- Doc-prior batch update ----------
    def update_doc_prior_batch(self, doc_values: Dict, local_category_assignments: Tensor):
        depth = len(self.cluster_dims)
        N = local_category_assignments.shape[0]
        A = torch.empty_like(doc_values["P"][0])
        B = torch.empty_like(doc_values["P"][1])

        for n in range(N):
            rev_cat = torch.flip(local_category_assignments[n], dims=[0]).tolist()
            a, b = gen_next_level_prior(
                partial_index(self.struct_values[f"G{depth}"], tuple(rev_cat)),
                partial_index(self.struct_values[f"alpha{depth}"] if False else self.struct_values[f"P{depth}"][0], tuple(rev_cat))  # fallback to P-depth alpha if alpha tensor not persisted
            )
            # a, b already vectors over K
            A[n] = a
            B[n] = b

        doc_values["Prior"] = (A, B)
        return doc_values


# -------------------------
# Quick smoke test
# -------------------------
if __name__ == "__main__":
    # toy run similar to your __main__
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    toy_struct = {"G0": 5, "G1": 3, "G2": 2}
    model = HDMM(toy_struct, vocab_size=11).to(device)
    print("Model initialized.")
    torch.manual_seed(0)

    N, M, V = 7, 17, 11
    obs = torch.randint(0, 2, (N, M, V), device=device, dtype=torch.float32)
    # make sure rows are one-hot (Multinomial total_count=1.0 assumption)
    obs = obs.argmax(dim=-1)
    obs = F.one_hot(obs, num_classes=V).float()

    reg = torch.randn(N, device=device)

    z_gen, z_reg, local_category_assignments, mc, doc_values, log_prob = model.infer(obs, reg, num_iters=50)
    print("Inference completed. Last log_prob:", log_prob[-1])
    # If you have your own visualization, call it here.
    # likelihood_visualization(log_prob, np.zeros_like(log_prob), epoch=0)
