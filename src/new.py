import numpyro
from numpyro.distributions import constraints
from numpyro import distributions as dist
import jax
import jax.numpy as jnp
import random
import copy
from tqdm import trange

from hdmm_utils import mix_weights, suffix_sum, dirichlet_posterior, nig_posterior, topic_mixture_posterior, gaussian_mixture_posterior, gen_next_level_prior, get_unique_rows_and_positions, beta_mixture_posterior


class HDMM:
    def __init__(self, struct_upbd, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.struct_upbd = struct_upbd
        self.K = int(struct_upbd["G0"])
        self.init_tunable_hyperparameters()
        self.init_mixture_components()
        self.init_structure()

    def init_tunable_hyperparameters(self):
        # Initialize tunable hyperparameters here
        self.struct_params = {}
        self.struct_params["gamma"]      = numpyro.param("model_gamma",      jnp.asarray([1.]), constraint=constraints.positive)
        self.struct_params["dir_alpha"]  = numpyro.param("model_dir_alpha",  jnp.asarray([1.]), constraint=constraints.positive)
        self.struct_params["nig_mu"]     = numpyro.param("model_nig_mu",     jnp.asarray([0.0]))
        self.struct_params["nig_kappa"]  = numpyro.param("model_nig_kappa",  jnp.asarray([1.0]), constraint=constraints.positive)
        self.struct_params["nig_alpha"]  = numpyro.param("model_nig_alpha",  jnp.asarray([1.0]), constraint=constraints.positive)
        self.struct_params["nig_beta"]   = numpyro.param("model_nig_beta",   jnp.asarray([1.0]), constraint=constraints.positive)

        self.param_dims = list(self.struct_upbd.values())
        self.param_dims.reverse()
        self.cluster_dims = self.param_dims[:-1]  # no G0
        self.cluster_dims.reverse()
        # alpha/eta tensors across hierarchy
        for parent_level in range(len(self.struct_upbd) - 1):
            child_level = parent_level + 1

            base = numpyro.param(
                f"model_alpha{parent_level}",
                jnp.ones(tuple(self.param_dims[-child_level:-1])),
                constraint=constraints.positive,
            )
            self.struct_params[f"alpha{parent_level}"] = jnp.expand_dims(base, -1) * jnp.ones(tuple(self.param_dims[-child_level:]))
            assert self.struct_params[f"alpha{parent_level}"].shape == tuple(self.param_dims[-child_level:])

            self.struct_params[f"eta{parent_level}"] = numpyro.param(
                f"model_eta{parent_level}",
                jnp.ones(tuple(self.cluster_dims[:child_level])),
                constraint=constraints.positive,
            )

        # Last level alpha (no eta)
        last_idx = len(self.struct_upbd) - 1
        base_last = numpyro.param(
            f"model_alpha{last_idx}",
            jnp.ones(tuple(self.param_dims[:-1])),
            constraint=constraints.positive,
        )
        self.struct_params[f"alpha{last_idx}"] = jnp.expand_dims(base_last, -1) * jnp.ones(tuple(self.param_dims))
        assert self.struct_params[f"alpha{last_idx}"].shape == tuple(self.param_dims)

    def init_structure(self):
        # ---------------
        # Stick-breaking
        # ---------------
        self.struct_values = {}

        # Top-level Beta sticks B0 -> G0 weights

        B0_a = jnp.ones((self.K,))  # shape (K0,)
        B0_b = jnp.broadcast_to(self.struct_params["gamma"], (self.K,))
        key, sub = random.split(key)
        beta_0 = dist.Beta(B0_a, B0_b).sample(sub)
        beta_0 = beta_0.at[-1].set(1.0)  # last stick is always 1
        self.struct_values["P0"] = [B0_a, B0_b]
        self.struct_values["B0"] = beta_0
        self.struct_values["G0"] = mix_weights(beta_0)  # (K0,)
        self.struct_values["S0"] = (self.K,)
        assert self.struct_values["G0"].shape == (self.K,)


        # Lower levels
        for parent_level in range(len(self.struct_upbd) - 1):
            child_level = parent_level + 1
            full_dim = child_level + 1 

            G_parent = self.struct_values[f"G{parent_level}"]  # shape param_dims[-(parent_level+1):]
            alpha_param = self.struct_params[f"alpha{parent_level}"]  # shape param_dims[-child_level:]
            shape_needed = tuple(self.param_dims[-full_dim:])

            param_alpha = alpha_param * G_parent
            param_beta = suffix_sum(param_alpha)

            a = jnp.broadcast_to(jnp.expand_dims(param_alpha, 0), shape_needed)
            b = jnp.broadcast_to(jnp.expand_dims(param_beta, 0), shape_needed)

            key, sub = random.split(key)
            beta = dist.Beta(a, b).sample(sub)
            beta = beta.at[..., -1].set(1.0)  # last stick is always 1
            self.struct_values[f"P{child_level}"] = [a, b]
            self.struct_values[f"S{child_level}"] = shape_needed
            self.struct_values[f"B{child_level}"] = beta
            self.struct_values[f"G{child_level}"] = mix_weights(beta)
            assert self.struct_values[f"G{child_level}"].shape == tuple(self.param_dims[-full_dim:])
            assert self.struct_values[f"P{child_level}"][0].shape == tuple(self.param_dims[-full_dim:])
            assert self.struct_values[f"P{child_level}"][1].shape == tuple(self.param_dims[-full_dim:])

        # ---------------
        # Cluster weights
        # ---------------
        for parent_level in range(len(self.struct_upbd) - 1):
            child_level = parent_level + 1
            full_dim = child_level + 1
            eta = self.struct_params[f"eta{parent_level}"]  # shape param_dims[-full_dim:-1]

            key, sub = random.split(key)
            a = jnp.ones_like(eta)
            b = eta
            beta = dist.Beta(a, b).sample(sub)
            beta = beta.at[-1].set(1.0)  # last stick is always 1
            assert beta.shape == tuple(self.cluster_dims[:child_level])
            self.struct_values[f"LP{parent_level}"] = [a, b]
            self.struct_values[f"LS{parent_level}"] = tuple(self.cluster_dims[:child_level])
            self.struct_values[f"LB{parent_level}"] = beta
            self.struct_values[f"LG{parent_level}"] = mix_weights(beta)  # categorical probs over next level
            assert self.struct_values[f"LG{parent_level}"].shape == tuple(self.cluster_dims[:child_level])

    def init_mixture_components(self):
        # -----------------------
        # Mixture components
        # -----------------------
        # Topics over vocab
        self.vocab_size = self.kwargs.get("vocab_size", 10000)

        self.mixture_components = {}

        key, sub = random.split(key)
        self.mixture_components["generation"] = dist.Dirichlet(
                self.struct_params["dir_alpha"]
                * jnp.ones((self.vocab_size))
            ).sample(sub, sample_shape=(self.K,))
        assert self.mixture_components["generation"].shape == (self.K, self.vocab_size)
        # Regression components via NIG prior
        key, sub = random.split(key)
        sigma = dist.InverseGamma(
                jnp.broadcast_to(self.struct_params["nig_alpha"], (self.K,)),
                jnp.broadcast_to(self.struct_params["nig_beta"],  (self.K,))
            ).sample(sub)
        assert sigma.shape == (self.K,)
        key, sub = random.split(key)
        mu = dist.Normal(
                jnp.broadcast_to(self.struct_params["nig_mu"], (self.K,)),
                jnp.sqrt(sigma / jnp.broadcast_to(self.struct_params["nig_kappa"], (self.K,)))
            ).sample(sub)
        assert mu.shape == (self.K,)
        self.mixture_components["regression_sigma"] = sigma
        self.mixture_components["regression_mu"] = mu

    def init_latent_variables(self, obs, *args, **kwargs):
        N, M, _ = obs.shape
        doc_values = {}
        z_gen = jnp.zeros_like((N, M), dtype=jnp.int32)  # (N, M)
        z_reg = jnp.zeros((N,), dtype=jnp.int32)  # (N,)
        local_category_assignments = jnp.zeros((N, len(self.param_dims) -1), dtype=jnp.int32)  # (N, L-1)
        return z_gen, z_reg, local_category_assignments, doc_values
    
    def init_markov_chain(self):
        mc = {}
        mc["generation_components"] = []
        mc["regression_mu"] = []
        mc["regression_sigma"] = []
        for depth in range(len(self.param_dims)):
            mc[f"G{depth}"] = []
            mc[f"B{depth}"] = []
        return mc
    
    def update_markov_chain(self, mc):
        mc["generation_components"].append(self.struct_values["G1"])
        mc["regression_mu"].append(self.struct_values["regression_mu"])
        mc["regression_sigma"].append(self.struct_values["regression_sigma"])
        for depth in range(len(self.param_dims)):
            mc[f"G{depth}"].append(self.struct_values[f"G{depth}"])
            mc[f"B{depth}"].append(self.struct_values[f"B{depth}"])

        if (len(mc["generation_components"]) > 20):
            mc["generation_components"].pop(0)
            mc["regression_mu"].pop(0)
            mc["regression_sigma"].pop(0)
            for depth in range(len(self.param_dims)):
                mc[f"G{depth}"].pop(0)
                mc[f"B{depth}"].pop(0)

        return mc
    
    def update_struct_prior(self):
        # ---------------
        # Stick-breaking
        # ---------------
        self.struct_values["Prior0"] = copy.deepcopy(self.struct_values["P0"])

        # Lower levels
        for parent_level in range(len(self.struct_upbd) - 1):
            self.struct_values[f"Prior{parent_level + 1}"] = copy.deepcopy(self.struct_values[f"P{parent_level + 1}"])

        # ---------------
        # Cluster weights
        # ---------------
        for parent_level in range(len(self.struct_upbd) - 1):
            self.struct_values[f"LPrior{parent_level}"] = copy.deepcopy(self.struct_values[f"LP{parent_level}"])

    def forward(self, obs, *args, **kwargs):
        z_gen, z_reg, local_category_assignments, mc, log_prob = self.gibbs_update(obs, *args, **kwargs)
        return -log_prob

    def gibbs_update(self, obs, *args, **kwargs):
        self.update_struct_prior()
        num_iters = kwargs.get("num_iters", 100)
        key = kwargs.get("key", random.PRNGKey(0))

        N, M, _ = obs.shape

        reg = args[0] if len(args) > 0 else None
        
        log_prob = []

        mc = self.init_markov_chain()   
        z_gen, z_reg, local_category_assignments, doc_values = self.init_latent_variables(obs, *args, **kwargs)

        pbar = trange(num_iters + 1, desc="Gibbs Sampling")
        for it in pbar:
            # ------------------------
            # Sample generation components
            # ------------------------
            for k in range(self.K):
                word_idx = jnp.where(z_gen == k)
                if word_idx[0].size > 0:
                    key, sub = random.split(key)
                    obs_k = obs[word_idx]
                    self.gen_mix_gibbs(sub, obs_k, k)

            # ------------------------
            # Sample regression components
            # ------------------------
            for k in range(self.K):
                reg_idx = jnp.where(z_reg == k)
                if reg_idx[0].size > 0:
                    key, sub = random.split(key)
                    reg_k = reg[reg_idx]
                    self.reg_mix_gibbs(sub, reg_k, k)

            # ------------------------
            # Sample document-level weights and word/regression categories
            # ------------------------
            for n in range(N):

                # Sample word-level categories
                for m in range(M):
                    key, sub = random.split(key)
                    z_gen = self.word_cat_gibbs(sub, n, m, obs, doc_values, z_gen)

                # Sample regression category
                key, sub = random.split(key)
                z_reg = self.reg_cat_gibbs(sub, n, reg, doc_values, z_reg)

                # Sample doc-level weights
                key, sub = random.split(key)   
                doc_values = self.doc_weight_gibbs(sub, n, doc_values, z_gen, z_reg, N)

                # Sample local category assignments
                for depth in range(1, len(self.param_dims), 1):
                    key, sub = random.split(key)
                    local_category_assignments, new_cat, cat_idx = self.doc_cats_gibbs(depth, sub, n, doc_values)
                    # update doc-level prior
                    if (depth == len(self.param_dims) - 1):
                         doc_values = self.update_prior(doc_values, n, depth, tuple(int(new_cat), cat_idx))

            # ------------------------
            # Sample structural weights
            # ------------------------
            for depth in range(0, len(self.param_dims), 1):
                if depth > 0:
                    unique_rows, positions = get_unique_rows_and_positions(local_category_assignments[:, :depth])
                else:
                    unique_rows = [(slice(None),)]
                    positions = [jnp.arange(N)]
                for row, row_idx in zip(unique_rows, positions):
                    if row_idx.size == 0 and depth > 0:
                        continue
                    else:
                        key, sub = random.split(key)
                        if (depth > 0):
                            rev_cat = tuple(row.tolist()[::-1])
                        else:
                            rev_cat = row
                        self.struct_weights_gibbs(depth, sub, row_idx, rev_cat, z_gen, z_reg)
                        self.update_prior(doc_values, row_idx, depth, rev_cat)
                        key, sub = random.split(key)
                        self.struct_cluster_gibbs(depth, sub, row_idx, rev_cat, local_category_assignments)

            log_prob.append(self.compute_log_likelihood(obs, z_gen, z_reg, reg))

            mc = self.update_markov_chain(mc)
            pbar.set_description(f"Gibbs Sampling (Iter {it}) LogProb {log_prob[-1]:.2f}")

        return z_gen, z_reg, local_category_assignments, mc, log_prob

    @jax.jit
    def compute_log_likelihood(self, obs, z_gen, z_reg, reg):
        """
        Gibbs sampler for HDMM with proper JAX key handling.
        """
        log_prob = 0.0

        gen_param = self.mixture_components["generation"][z_gen]  # (N, M, V)
        gen_param = jnp.clip(gen_param, 1e-12, 1.0)
        gen_param = gen_param / gen_param.sum(-1, keepdims=True)  #
        word_prob = dist.Multinomial(total_count=1, probs=gen_param).log_prob(obs)
        log_prob += jnp.sum(word_prob)
    
        reg_prob = dist.Normal(loc=self.mixture_components["regression_mu"][z_reg], scale=self.mixture_components["regression_sigma"][z_reg]).log_prob(reg)
        log_prob += jnp.sum(reg_prob)

        return log_prob
    
    def update_prior(self, doc_values, row_idx, depth, rev_cat):
        a, b = gen_next_level_prior(self.struct_values[f"G{depth}"][rev_cat], self.struct_params[f"alpha{depth}"][rev_cat])
        if (depth < len(self.param_dims) - 1):
            self.struct_values[f"Prior{depth + 1}"][0] = self.struct_values[f"Prior{depth + 1}"][0].at[(slice(None),) + rev_cat].set(a)
            self.struct_values[f"Prior{depth + 1}"][1] = self.struct_values[f"Prior{depth + 1}"][1].at[(slice(None),) + rev_cat].set(b)
        else:
            doc_values["Prior"][0] = doc_values["Prior"][0].at[row_idx].set(a)
            doc_values["Prior"][1] = doc_values["Prior"][1].at[row_idx].set(b)
        return doc_values

    def gen_mix_gibbs(self, sub, obs_k, k):
        generation_components_k = dirichlet_posterior(sub, obs_k, self.struct_params["dir_alpha"] * jnp.ones((self.vocab_size,)))
        self.mixture_components["generation"] = self.mixture_components["generation"].at[k].set(generation_components_k)

    def reg_mix_gibbs(self, sub, reg_k, k):
        (new_mu, new_sigma)= nig_posterior(
            sub, reg_k, (
                self.struct_params["nig_mu"],
                self.struct_params["nig_kappa"],
                self.struct_params["nig_alpha"],
                self.struct_params["nig_beta"]
            )
        )
        self.mixture_components["regression_mu"] = self.mixture_components["regression_mu"].at[k].set(new_mu)
        self.mixture_components["regression_sigma"] = self.mixture_components["regression_sigma"].at[k].set(new_sigma)

    def word_cat_gibbs(self, sub, n, m, obs, doc_values, z_gen):
        sample = topic_mixture_posterior(sub, obs[n, m], doc_values["G"][n], self.mixture_components["generation"])
        z_gen = z_gen.at[n, m].set(sample)
        return z_gen
    
    def reg_cat_gibbs(self, sub, n, reg, doc_values, z_reg):
        sample = gaussian_mixture_posterior(sub, reg[n], doc_values["G"][n], (self.mixture_components["regression_mu"], self.mixture_components["regression_sigma"]))
        z_reg = z_reg.at[n].set(sample)
        return z_reg
    
    def doc_weight_gibbs(self, sub, n, doc_values, z_gen, z_reg, N):
        new_params = self._doc_weight_conditional(
            doc_values["B"][n],
            [doc_values["Prior"][0][n], doc_values["Prior"][1][n]],
            z_gen[n], 
            z_reg[n]
        )

        doc_values["P"][0] = doc_values["P"][0].at[n].set(new_params[0])
        doc_values["P"][1] = doc_values["P"][1].at[n].set(new_params[1])

        beta = dist.Beta(doc_values["P"][0][n], doc_values["P"][1][n]).sample(sub)
        beta = beta.at[..., -1].set(1.0)  # last entry is always 1

        doc_values["B"] = doc_values["B"].at[n].set(beta)
        assert doc_values["B"].shape == (N, self.K)
        doc_values["G"] = doc_values["G"].at[n].set(mix_weights(doc_values["B"][n]))  
        return doc_values 
    
    def doc_cats_gibbs(self, depth, sub, n, doc_values):
        cat_idx = tuple(local_category_assignments[n, :depth].tolist()[::-1])

        doc_alpha, doc_beta = gen_next_level_prior(self.struct_values[f"G{depth}"][:, cat_idx], self.struct_params[f"alpha{depth}"][:, cat_idx])
        new_cat = beta_mixture_posterior(sub, doc_values["B"][n], [doc_alpha, doc_beta], self.struct_values[f"LG{depth}"][cat_idx])

        local_category_assignments = local_category_assignments.at[(n,) + cat_idx].set(new_cat)
        return local_category_assignments, new_cat, cat_idx
    
    def struct_weights_gibbs(self, depth, key, row_idx, rev_cat, z_gen, z_reg):
        key, sub = random.split(key)
        new_params = self._cat_weight_conditional(sub, depth, rev_cat, z_gen[row_idx], z_reg[row_idx])
        self.struct_values[f"P{depth}"][0] = self.struct_values[f"P{depth}"][0].at[rev_cat].set(new_params[0])
        self.struct_values[f"P{depth}"][1] = self.struct_values[f"P{depth}"][1].at[rev_cat].set(new_params[1])

        key, sub = random.split(key)
        beta = dist.Beta(self.struct_values[f"P{depth}"][0][rev_cat], self.struct_values[f"P{depth}"][1][rev_cat]).sample(sub)
        beta = beta.at[..., -1].set(1.0)  # last entry is always 1
        self.struct_values[f"B{depth}"] = self.struct_values[f"B{depth}"].at[rev_cat].set(beta)
        self.struct_values[f"G{depth}"] = self.struct_values[f"G{depth}"].at[rev_cat].set(mix_weights(self.struct_values[f"B{depth}"][rev_cat]))
    
    def struct_cluster_gibbs(self, depth, key, row_idx, rev_cat, local_category_assignments):
        new_params, key = self._cluster_weight_conditional(depth, rev_cat, local_category_assignments[:, depth][row_idx])

        key, sub = random.split(key)

        beta = dist.Beta(new_params[0], new_params[1]).sample(sub)
        beta = beta.at[..., -1].set(1.0)  # last entry is always 1
        self.struct_values[f"LB{depth}"] = self.struct_values[f"LB{depth}"].at[rev_cat].set(beta)
        self.struct_values[f"LP{depth}"][0] = self.struct_values[f"LP{depth}"][0].at[rev_cat].set(new_params[0])
        self.struct_values[f"LP{depth}"][1] = self.struct_values[f"LP{depth}"][1].at[rev_cat].set(new_params[1])
        self.struct_values[f"LG{depth}"] = self.struct_values[f"LG{depth}"].at[rev_cat].set(mix_weights(self.struct_values[f"LB{depth}"][rev_cat]))

    @jax.jit
    def _cat_weight_conditional(self, key, depth, rev_cat, word_cats, reg_cats):
        """
        Sample category-level stick-breaking weights given category assignments and Beta parameters.
        Args:
            key: JAX PRNGKey
            nu: (K,) current category-level stick-breaking weights
            params: list of two (K,) arrays, Beta parameters [alpha, beta]
            word_cats: (N_word,) array of word category assignments
            reg_cats: (N_reg,) array of regression category assignments
        Returns:
            new_params: list of two (K,) arrays, updated Beta parameters [alpha, beta
            new_key: updated JAX PRNGKey
        """ 
        nu = self.struct_values[f"B{depth}"][rev_cat]
        params = [self.struct_values[f"Prior{depth}"][0][rev_cat], self.struct_values[f"Prior{depth}"][1][rev_cat]]

        cat_count = jnp.bincount(word_cats.ravel(), length=self.K)
        cat_idx = jnp.arange(self.K)
        reg_count = jnp.bincount(reg_cats.ravel(), length=self.K)
        cat_count = cat_count + reg_count

        alpha_bias = jnp.zeros_like(nu, dtype=jnp.int32).at[cat_idx].set(cat_count)
        beta_bias = suffix_sum(alpha_bias)

        new_params = [params[0] + alpha_bias, params[1] + beta_bias]

        assert new_params[0].shape == (self.param_dims[-1],)
        assert new_params[1].shape == (self.param_dims[-1],)
        return new_params, key

    @jax.jit
    def _doc_weight_conditional(self, nu_doc, params, word_cats, reg_cats):
        """
        Sample document-level stick-breaking weights given category assignments and Beta parameters.
        Args:
            key: JAX PRNGKey
            nu_doc: (K,) current document-level stick-breaking weights
            params: list of two (K,) arrays, Beta parameters [alpha, beta]
            word_cats: (N_word,) array of word category assignments
            reg_cats: (N_reg,) array of regression category assignments
        Returns:
            new_params: list of two (K,) arrays, updated Beta parameters [alpha, beta
            new_key: updated JAX PRNGKey
        """
        cat_count = jnp.bincount(word_cats.ravel(), length=self.K)
        cat_idx = jnp.arange(self.K)
        reg_count = jnp.bincount(reg_cats.ravel(), length=self.K)
        cat_count = cat_count + reg_count

        alpha_bias = jnp.zeros_like(nu_doc, dtype=jnp.int32).at[cat_idx].set(cat_count)
        beta_bias = suffix_sum(alpha_bias)

        new_params = [params[0] + alpha_bias, params[1] + beta_bias]
        return new_params
    
    def _cluster_weight_conditional(self, depth, rev_cat, local_cluster_cats):
        """
        Sample super-cluster-level stick-breaking weights given category assignments and Beta parameters.
        Args:
            key: JAX PRNGKey
            nu_cluster: (S,) current super-cluster-level stick-breaking weights
            params: list of two (S,) arrays, Beta parameters [alpha, beta]
            cluster_cats: (N_cluster,) array of super-cluster category assignments
        Returns:
            new_params: list of two (S,) arrays, updated Beta parameters [alpha, beta
            new_key: updated JAX PRNGKey
        """
        nu_cluster = self.struct_values[f"LB{depth}"][rev_cat]
        params = [self.struct_values[f"LPrior{depth}"][0][rev_cat], self.struct_values[f"LPrior{depth}"][1][rev_cat]]
        cat_count = jnp.bincount(local_cluster_cats.ravel(), length=self.param_dims[depth])
        cat_idx = jnp.arange(self.param_dims[depth])

        alpha_bias = jnp.zeros_like(nu_cluster, dtype=jnp.int32).at[cat_idx].set(cat_count)
        beta_bias = suffix_sum(alpha_bias)

        new_params = [params[0] + alpha_bias, params[1] + beta_bias]
        return new_params
    

