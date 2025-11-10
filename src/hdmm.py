import numpyro
from numpyro.distributions import constraints
from numpyro import distributions as dist
import jax
import jax.numpy as jnp
import copy
from tqdm import trange
import numpy as np

from hdmm_utils import mix_weights, suffix_sum, dirichlet_posterior, nig_posterior, topic_mixture_posterior, gaussian_mixture_posterior, gen_next_level_prior, get_unique_rows_and_positions, beta_mixture_posterior, gather_middle_slice, partial_index, set_by_multi_index

from vis import likelihood_visualization


class HDMM:
    def __init__(self, struct_upbd, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.struct_upbd = struct_upbd
        self.K = int(struct_upbd["G0"])
        self.init_tunable_hyperparameters()
        self.init_mixture_components()
        self.init_structure()
        self.best_log_prob = -jnp.inf

    def init_tunable_hyperparameters(self):
        # Initialize tunable hyperparameters here
        key = jax.random.PRNGKey(0)
        self.struct_params = {}
        key, sub = jax.random.split(key)
        self.struct_params["gamma"]      = numpyro.param("model_gamma",      jax.random.uniform(sub, minval=0.0, maxval=100.0), constraint=constraints.positive)
        key, sub = jax.random.split(key)
        self.struct_params["dir_alpha"]  = numpyro.param("model_dir_alpha",  jax.random.uniform(sub, minval=0.0, maxval=1.0), constraint=constraints.positive)
        key, sub = jax.random.split(key)
        self.struct_params["nig_mu"]     = numpyro.param("model_nig_mu",     jax.random.uniform(sub, minval=0.0, maxval=100.0))
        key, sub = jax.random.split(key)
        self.struct_params["nig_kappa"]  = numpyro.param("model_nig_kappa",  jax.random.uniform(sub, minval=0.0, maxval=100.0), constraint=constraints.positive)
        key, sub = jax.random.split(key)
        self.struct_params["nig_alpha"]  = numpyro.param("model_nig_alpha",  jax.random.uniform(sub, minval=0.0, maxval=100.0), constraint=constraints.positive)
        key, sub = jax.random.split(key)
        self.struct_params["nig_beta"]   = numpyro.param("model_nig_beta",   jax.random.uniform(sub, minval=0.0, maxval=100.0), constraint=constraints.positive)

        self.param_dims = list(self.struct_upbd.values())
        self.param_dims.reverse()
        self.cluster_dims = self.param_dims[:-1]  # no G0
        self.cluster_dims.reverse()
        # alpha/eta tensors across hierarchy
        for parent_level in range(len(self.struct_upbd) - 1):
            child_level = parent_level + 1
            key, sub = jax.random.split(key)
            base = numpyro.param(
                f"model_alpha{parent_level}",
                jax.random.uniform(sub, shape=tuple(self.param_dims[-child_level:-1]), minval=0.0, maxval=100.0),
                constraint=constraints.positive,
            )
            self.struct_params[f"alpha{parent_level}"] = jnp.expand_dims(base, -1) * jnp.ones(tuple(self.param_dims[-child_level:]))
            assert self.struct_params[f"alpha{parent_level}"].shape == tuple(self.param_dims[-child_level:])

            key, sub = jax.random.split(key)
            self.struct_params[f"eta{parent_level}"] = numpyro.param(
                f"model_eta{parent_level}",
                jax.random.uniform(sub, shape=tuple(self.cluster_dims[:child_level], ), minval=0.0, maxval=100.0),
                constraint=constraints.positive,
            )

        # Last level alpha (no eta)
        last_idx = len(self.struct_upbd) - 1
        key, sub = jax.random.split(key)
        base_last = numpyro.param(
            f"model_alpha{last_idx}",
            jax.random.uniform(sub, shape=tuple(self.param_dims[:-1]), minval=0.0, maxval=100.0),
            constraint=constraints.positive,
        )
        self.struct_params[f"alpha{last_idx}"] = jnp.expand_dims(base_last, -1) * jnp.ones(tuple(self.param_dims))
        assert self.struct_params[f"alpha{last_idx}"].shape == tuple(self.param_dims)

    def init_structure(self):
        # ---------------
        # Stick-breaking
        # ---------------
        key = jax.random.PRNGKey(1)
        self.struct_values = {}
        self.best_struct_values = {}

        # Top-level Beta sticks B0 -> G0 weights

        B0_a = jnp.ones((self.K,))  # shape (K0,)
        B0_b = jnp.broadcast_to(self.struct_params["gamma"], (self.K,))
        key, sub = jax.random.split(key)
        beta_0 = dist.Beta(B0_a, B0_b).sample(sub)
        beta_0 = beta_0.at[-1].set(1.0)  # last stick is always 1
        self.struct_values["P0"] = [B0_a, B0_b]
        self.struct_values["Prior0"] = copy.deepcopy(self.struct_values["P0"])
        self.struct_values["Posterior0"] = copy.deepcopy(self.struct_values["P0"])
        self.struct_values["G0"] = mix_weights(beta_0)  # (K0,)
        assert self.struct_values["G0"].shape == (self.K,)

        self.best_struct_values["P0"] = copy.deepcopy(self.struct_values["P0"])
        self.best_struct_values["Prior0"] = copy.deepcopy(self.struct_values["Prior0"])
        self.best_struct_values["Posterior0"] = copy.deepcopy(self.struct_values["Posterior0"])
        self.best_struct_values["G0"] = copy.deepcopy(self.struct_values["G0"])
        

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

            key, sub = jax.random.split(key)
            beta = dist.Beta(a, b).sample(sub)

            beta = beta.at[..., -1].set(1.0)  # last stick is always 1
            self.struct_values[f"P{child_level}"] = [a, b]
            self.struct_values[f"Prior{child_level}"] = copy.deepcopy(self.struct_values[f"P{child_level}"])
            self.struct_values[f"Posterior{child_level}"] = copy.deepcopy(self.struct_values[f"P{child_level}"])
            self.struct_values[f"G{child_level}"] = mix_weights(beta)
            assert self.struct_values[f"G{child_level}"].shape == tuple(self.param_dims[-full_dim:])
            assert self.struct_values[f"P{child_level}"][0].shape == tuple(self.param_dims[-full_dim:])
            assert self.struct_values[f"P{child_level}"][1].shape == tuple(self.param_dims[-full_dim:])

            self.best_struct_values[f"P{child_level}"] = copy.deepcopy(self.struct_values[f"P{child_level}"])
            self.best_struct_values[f"Prior{child_level}"] = copy.deepcopy(self.struct_values[f"Prior{child_level}"])
            self.best_struct_values[f"Posterior{child_level}"] = copy.deepcopy(self.struct_values[f"Posterior{child_level}"])
            self.best_struct_values[f"G{child_level}"] = copy.deepcopy(self.struct_values[f"G{child_level}"])

        # ---------------
        # Cluster weights
        # ---------------
        for parent_level in range(len(self.struct_upbd) - 1):
            child_level = parent_level + 1
            full_dim = child_level + 1
            eta = self.struct_params[f"eta{parent_level}"]  # shape param_dims[-full_dim:-1]

            key, sub = jax.random.split(key)
            a = jnp.ones_like(eta)
            b = eta
            beta = dist.Beta(a, b).sample(sub)
            beta = beta.at[-1].set(1.0)  # last stick is always 1
            assert beta.shape == tuple(self.cluster_dims[:child_level])
            self.struct_values[f"LP{parent_level}"] = [a, b]
            self.struct_values[f"LPrior{parent_level}"] = copy.deepcopy(self.struct_values[f"LP{parent_level}"])
            self.struct_values[f"LPosterior{parent_level}"] = copy.deepcopy(self.struct_values[f"LP{parent_level}"])
            self.struct_values[f"LG{parent_level}"] = mix_weights(beta)  # categorical probs over next level
            assert self.struct_values[f"LG{parent_level}"].shape == tuple(self.cluster_dims[:child_level])

            self.best_struct_values[f"LP{parent_level}"] = copy.deepcopy(self.struct_values[f"LP{parent_level}"])
            self.best_struct_values[f"LPrior{parent_level}"] = copy.deepcopy(self.struct_values[f"LPrior{parent_level}"])
            self.best_struct_values[f"LPosterior{parent_level}"] = copy.deepcopy(self.struct_values[f"LPosterior{parent_level}"])
            self.best_struct_values[f"LG{parent_level}"] = copy.deepcopy(self.struct_values[f"LG{parent_level}"])

    def init_mixture_components(self):
        # -----------------------
        # Mixture components
        # -----------------------
        # Topics over vocab
        self.vocab_size = self.kwargs.get("vocab_size", 10000)
        key = jax.random.PRNGKey(2)

        self.mixture_components = {}

        key, sub = jax.random.split(key)
        self.mixture_components["generation"] = dist.Dirichlet(
                self.struct_params["dir_alpha"]
                * jnp.ones((self.vocab_size))
            ).sample(sub, sample_shape=(self.K,))
        assert self.mixture_components["generation"].shape == (self.K, self.vocab_size)

        # Regression components via NIG prior
        key, sub = jax.random.split(key)
        sigma = dist.InverseGamma(
                jnp.broadcast_to(self.struct_params["nig_alpha"], (self.K,)),
                jnp.broadcast_to(self.struct_params["nig_beta"],  (self.K,))
            ).sample(sub)
        assert sigma.shape == (self.K,)
        key, sub = jax.random.split(key)
        mu = dist.Normal(
                jnp.broadcast_to(self.struct_params["nig_mu"], (self.K,)),
                jnp.sqrt(sigma / jnp.broadcast_to(self.struct_params["nig_kappa"], (self.K,)))
            ).sample(sub)
        assert mu.shape == (self.K,)
        self.mixture_components["regression_sigma"] = sigma
        self.mixture_components["regression_mu"] = mu

        self.mixture_components_posterior = copy.deepcopy(self.mixture_components)
        self.best_mixture_components = copy.deepcopy(self.mixture_components)

    def init_latent_variables(self, obs, *args, **kwargs):
        key = jax.random.PRNGKey(5)
        N, M, _ = obs.shape
        key, sub = jax.random.split(key)
        z_gen = jax.random.randint(sub, shape=(N, M), minval=0, maxval=self.K)  # (N, M)
        key, sub = jax.random.split(key)
        z_reg = jax.random.randint(sub, shape=(N,), minval=0, maxval=self.K)  # (N,)
        local_category_assignments = []
        for max_cat in self.cluster_dims:
            key, sub = jax.random.split(key)
            cats = jax.random.randint(sub, shape=(N,), minval=0, maxval=max_cat)  # (N,)
            local_category_assignments.append(cats)
        local_category_assignments = jnp.stack(local_category_assignments, axis=1)  # (N, num_levels)

        doc_values = {}
        rev_idx = jnp.flip(local_category_assignments, axis=1)
        param0 = gather_middle_slice(jnp.broadcast_to(jnp.expand_dims(self.struct_values[f"P{len(self.cluster_dims)}"][0], 0), (N, *self.param_dims)), rev_idx)
        param1 = gather_middle_slice(jnp.broadcast_to(jnp.expand_dims(self.struct_values[f"P{len(self.cluster_dims)}"][1], 0), (N, *self.param_dims)), rev_idx)
        doc_values["P"] = [param0, param1]
        doc_values["Prior"] = copy.deepcopy(doc_values["P"])
        doc_values["G"] = gather_middle_slice(jnp.broadcast_to(jnp.expand_dims(self.struct_values[f"G{len(self.cluster_dims)}"], 0), (N, *self.param_dims)), rev_idx)
        return z_gen, z_reg, local_category_assignments, doc_values
    
    def init_markov_chain(self):
        mc = {}
        mc["generation_components"] = []
        mc["regression_mu"] = []
        mc["regression_sigma"] = []
        for depth in range(len(self.param_dims)):
            mc[f"G{depth}"] = []
        return mc

    def update_markov_chain(self, mc):
        mc["generation_components"].append(self.mixture_components["generation"])
        mc["regression_mu"].append(self.mixture_components["regression_mu"])
        mc["regression_sigma"].append(self.mixture_components["regression_sigma"])
        for depth in range(len(self.param_dims)):
            mc[f"G{depth}"].append(self.struct_values[f"G{depth}"])

        if (len(mc["generation_components"]) > 20):
            mc["generation_components"].pop(0)
            mc["regression_mu"].pop(0)
            mc["regression_sigma"].pop(0)
            for depth in range(len(self.param_dims)):
                mc[f"G{depth}"].pop(0)

        return mc
    
    def update_struct_posterior(self, lr):
        for parent_level in range(len(self.struct_upbd)):
            self.best_struct_values[f"Posterior{parent_level}"][0] = (1-lr)*self.best_struct_values[f"Posterior{parent_level}"][0] + lr*self.best_struct_values[f"P{parent_level}"][0]
            self.best_struct_values[f"Posterior{parent_level}"][1] = (1-lr)*self.best_struct_values[f"Posterior{parent_level}"][1] + lr*self.best_struct_values[f"P{parent_level}"][1]

            if (parent_level < len(self.struct_upbd) - 1):
                self.best_struct_values[f"LPosterior{parent_level}"][0] = (1-lr)*self.best_struct_values[f"LPosterior{parent_level}"][0] + lr*self.best_struct_values[f"LP{parent_level}"][0]
                self.best_struct_values[f"LPosterior{parent_level}"][1] = (1-lr)*self.best_struct_values[f"LPosterior{parent_level}"][1] + lr*self.best_struct_values[f"LP{parent_level}"][1]

        self.mixture_components_posterior["generation"] = (1-lr)*self.mixture_components_posterior["generation"] + lr*self.best_mixture_components["generation"]
        self.mixture_components_posterior["regression_mu"] = (1-lr)*self.mixture_components_posterior["regression_mu"] + lr*self.best_mixture_components["regression_mu"]
        self.mixture_components_posterior["regression_sigma"] = (1-lr)*self.mixture_components_posterior["regression_sigma"] + lr*self.best_mixture_components["regression_sigma"]

    def set_struct_to_best(self):
        self.struct_values = copy.deepcopy(self.best_struct_values)
        self.mixture_components = copy.deepcopy(self.best_mixture_components)

    def update_best_struct(self, log_prob, predict=False, **kwargs):
        if log_prob > self.best_log_prob:
            self.best_log_prob = log_prob
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

    def update_struct_prior(self, key_int):
        key = jax.random.PRNGKey(key_int)
        for parent_level in range(len(self.struct_upbd)):
            self.struct_values[f"Prior{parent_level}"][0] = copy.deepcopy(self.struct_values[f"Posterior{parent_level}"][0])
            self.struct_values[f"Prior{parent_level}"][1] = copy.deepcopy(self.struct_values[f"Posterior{parent_level}"][1])
            self.struct_values[f"P{parent_level}"][0] = copy.deepcopy(self.struct_values[f"Prior{parent_level}"][0])
            self.struct_values[f"P{parent_level}"][1] = copy.deepcopy(self.struct_values[f"Prior{parent_level}"][1])
            key, sub = jax.random.split(key)
            self.struct_values[f"G{parent_level}"] = mix_weights(dist.Beta(self.struct_values[f"P{parent_level}"][0], self.struct_values[f"P{parent_level}"][1]).sample(sub))
            if (parent_level < len(self.struct_upbd) - 1):
                self.struct_values[f"LPrior{parent_level}"][0] = copy.deepcopy(self.best_struct_values[f"LPosterior{parent_level}"][0])
                self.struct_values[f"LPrior{parent_level}"][1] = copy.deepcopy(self.best_struct_values[f"LPosterior{parent_level}"][1])
                self.struct_values[f"LP{parent_level}"][0] = copy.deepcopy(self.struct_values[f"LPrior{parent_level}"][0])
                self.struct_values[f"LP{parent_level}"][1] = copy.deepcopy(self.struct_values[f"LPrior{parent_level}"][1])
                key, sub = jax.random.split(key)
                self.struct_values[f"LG{parent_level}"] = mix_weights(dist.Beta(self.struct_values[f"LP{parent_level}"][0], self.struct_values[f"LP{parent_level}"][1]).sample(sub))
        
        self.mixture_components = copy.deepcopy(self.mixture_components_posterior)

    def forward(self, obs, *args, **kwargs):
        z_gen, z_reg, local_category_assignments, mc, log_prob = self.gibbs_update(obs, *args, **kwargs)
        return -log_prob

    def predict(self, obs, *args, **kwargs):
        num_iters = kwargs.get("num_iters", 100)
        key = kwargs.get("key", jax.random.PRNGKey(3))
        self.set_struct_to_best()

        N, M, _ = obs.shape

        reg = args[0] if len(args) > 0 else None
        
        log_probs = []

        z_gen, z_reg, local_category_assignments, doc_values = self.init_latent_variables(obs, *args, **kwargs)
        self.update_best_latent(z_gen=z_gen, z_reg=z_reg, local_category_assignments=local_category_assignments, doc_values=doc_values)

        pbar = trange(num_iters, desc="Inference Gibbs Sampling")
        for it in pbar:

            # ------------------------
            # Sample document-level weights and word/regression categories
            # ------------------------

            key, sub = jax.random.split(key)
            z_gen = self.vectorized_word_cat_gibbs(sub, obs, doc_values["G"])

            key, sub = jax.random.split(key)
            doc_values = self.vectorized_doc_weight_gibbs(
                sub,
                doc_values,
                z_gen,
                z_reg,
                scale_constant=1.0,
                predict=True
            )

            for depth in range(len(self.cluster_dims)):
                key, sub = jax.random.split(key)
                cats, probs = self.collapsed_doc_cats_gibbs_batch(sub, depth, obs, reg, z_gen, z_reg, local_category_assignments, predict=True)
                local_category_assignments = local_category_assignments.at[:, depth].set(cats)

            doc_values = self.update_doc_prior_batch(doc_values, local_category_assignments)

            log_prob = self.compute_log_likelihood(obs, z_gen, z_reg, reg, predict=True)
            if log_prob > max(log_probs, default=-jnp.inf):
                self.update_best_latent(z_gen=z_gen, z_reg=z_reg, local_category_assignments=local_category_assignments, doc_values=doc_values)
            log_probs.append(log_prob)
            pbar.set_description(f"Inference Gibbs Sampling (Iter {it+1}) LogProb {log_prob[-1]:.2f}")

        return z_gen, z_reg, local_category_assignments, doc_values, np.array(log_probs)

    def infer(self, obs, *args, **kwargs):
        lr = kwargs.get("lr", 0.1)
        self.update_struct_posterior(lr)
        self.set_struct_to_best()
        num_iters = kwargs.get("num_iters", 100)
        key = kwargs.get("key", jax.random.PRNGKey(4))
        known_cats = kwargs.get("known_cats", None)
        known_mixtures = kwargs.get("known_mixtures", None)
        known_struct = kwargs.get("known_struct", None)
        known_words = kwargs.get("known_words", None)
        datasize = kwargs.get("datasize", obs.shape[0])
        epoch = kwargs.get("epoch", 0)
        if (epoch > 0):
            self.update_struct_prior()
        skip_depth = []

        N, M, _ = obs.shape
        scale_constant = datasize / N # scale to full data size

        reg = args[0] if len(args) > 0 else None
        
        log_probs = []

        mc = self.init_markov_chain()   
        z_gen, z_reg, local_category_assignments, doc_values = self.init_latent_variables(obs, *args, **kwargs)
        self.update_best_latent(z_gen=z_gen, z_reg=z_reg, local_category_assignments=local_category_assignments, doc_values=doc_values)
        
        if known_words is not None:
            z_gen = known_words

        if (known_cats is not None):
            for depth, cats in known_cats.items():
                local_category_assignments = local_category_assignments.at[:, depth].set(cats)
                skip_depth.append(depth)
        
        if known_mixtures is not None:
            self.mixture_components["generation"] = known_mixtures["generation"]

        pbar = trange(num_iters, desc="Gibbs Sampling")
        
        for it in pbar:
            # ------------------------
            # Sample document-level weights and word/regression categories
            # ------------------------
            if (known_words is None):
                key, sub = jax.random.split(key)
                
                z_gen = self.vectorized_word_cat_gibbs(sub, obs, doc_values["G"])

            key, sub = jax.random.split(key)

            z_reg = self.vectorized_reg_cat_gibbs(sub, reg, doc_values["G"])

            key, sub = jax.random.split(key)
            doc_values = self.vectorized_doc_weight_gibbs(
                sub,
                doc_values,
                z_gen,
                z_reg,
                scale_constant
            )

            for depth in range(len(self.cluster_dims)):
                if depth in skip_depth:
                    continue
                key, sub = jax.random.split(key)
                cats, probs = self.collapsed_doc_cats_gibbs_batch(
                    sub, depth, obs, reg, z_gen, z_reg, local_category_assignments
                )
                local_category_assignments = local_category_assignments.at[:, depth].set(cats)

            doc_values = self.update_doc_prior_batch(doc_values, local_category_assignments)

            # ------------------------
            # Sample generation components
            # ------------------------
            if (known_mixtures is None):
                for k in range(self.K):
                    word_idx = jnp.where(z_gen == k)
                    # print("word_idx size:", word_idx[0].size, "k:", k)
                    if word_idx[0].size > 0:
                        key, sub = jax.random.split(key)
                        obs_k = obs[word_idx]
                        self.gen_mix_gibbs(sub, obs_k, k, scale_constant)

            # ------------------------
            # Sample regression components
            # ------------------------
            for k in range(self.K):
                reg_idx = jnp.where(z_reg == k)
                if reg_idx[0].size > 0:
                    key, sub = jax.random.split(key)
                    reg_k = reg[reg_idx]
                    self.reg_mix_gibbs(sub, reg_k, k, scale_constant)

            # ------------------------
            # Sample structural weights
            # ------------------------
            if (known_struct is not None):
                for depth, struct_val in known_struct.items():
                    self.struct_values[f"G{depth+1}"] = struct_val
                    
                    if depth > 0:
                        unique_rows, positions = get_unique_rows_and_positions(local_category_assignments[:, :depth])
                    else:
                        unique_rows = [(slice(None),)]
                        positions = [(slice(None),)]
                    for row, row_idx in zip(unique_rows, positions):
                        key, sub = jax.random.split(key)
                        if (depth < len(self.cluster_dims)):
                            key, sub = jax.random.split(key)
                            self.struct_cluster_gibbs(sub, depth, row_idx, row, local_category_assignments, scale_constant)
            else:
                for depth in range(len(self.param_dims)):
                    if depth > 0:
                        unique_rows, positions = get_unique_rows_and_positions(local_category_assignments[:, :depth])
                    else:
                        unique_rows = [(slice(None),)]
                        positions = [(slice(None),)]
                    for row, row_idx in zip(unique_rows, positions):
                        key, sub = jax.random.split(key)
                        if (depth > 0):
                            rev_cat = jnp.flip(row, axis=0)
                        else:
                            rev_cat = row
                        # print("high level rev_cat:", rev_cat)
                        self.struct_weights_gibbs(sub, depth, rev_cat, z_gen[row_idx], z_reg[row_idx], scale_constant)
                        if (depth < len(self.cluster_dims)):
                            key, sub = jax.random.split(key)
                            self.struct_cluster_gibbs(sub, depth, row_idx, row, local_category_assignments, scale_constant)

            log_prob = self.compute_log_likelihood(obs, z_gen, z_reg, reg)
            self.update_best_struct(log_prob, z_gen=z_gen, z_reg=z_reg, local_category_assignments=local_category_assignments, doc_values=doc_values)
            log_probs.append(log_prob)
            if (it > 0 and it % 50 == 0):
                likelihood_visualization(np.array(log_prob), np.zeros_like(np.array(log_prob)), epoch=it, log_dir=None)

            mc = self.update_markov_chain(mc)
            pbar.set_description(f"Gibbs Sampling (Iter {it+1}) LogProb {log_probs[-1]:.2f}")

        return z_gen, z_reg, local_category_assignments, mc, doc_values, np.array(log_probs)

    def compute_log_likelihood(self, obs, z_gen, z_reg, reg, predict=False):
        """
        Gibbs sampler for HDMM with proper JAX key handling.
        """
        log_prob = 0.0

        gen_param = self.mixture_components["generation"][z_gen]  # (N, M, V)
        gen_param = jnp.clip(gen_param, 1e-12, 1.0)
        gen_param = gen_param / gen_param.sum(-1, keepdims=True)  #
        word_prob = dist.Multinomial(total_count=1, probs=gen_param).log_prob(obs)
        log_prob += jnp.sum(word_prob)
        if (not predict):
            reg_prob = dist.Normal(loc=self.mixture_components["regression_mu"][z_reg], scale=self.mixture_components["regression_sigma"][z_reg]).log_prob(reg)
            log_prob += jnp.sum(reg_prob)

        return log_prob
    
    def update_doc_prior(self, rev_cat):
        depth = len(self.cluster_dims)
        a, b = gen_next_level_prior(jnp.atleast_2d(self.struct_values[f"G{depth}"][rev_cat]), jnp.atleast_2d(self.struct_params[f"alpha{depth}"][rev_cat]))    
        return a.flatten(), b.flatten()

    def gen_mix_gibbs(self, sub, obs_k, k, scale_constant):
        generation_components_k = dirichlet_posterior(sub, obs_k, self.struct_params["dir_alpha"] * jnp.ones((self.vocab_size,)), scale_constant)
        self.mixture_components["generation"] = self.mixture_components["generation"].at[k].set(generation_components_k)

    def reg_mix_gibbs(self, sub, reg_k, k, scale_constant):
        (new_mu, new_sigma)= nig_posterior(
            sub, reg_k, (
                self.struct_params["nig_mu"],
                self.struct_params["nig_kappa"],
                self.struct_params["nig_alpha"],
                self.struct_params["nig_beta"]
            ),
            scale_constant
        )
        self.mixture_components["regression_mu"] = self.mixture_components["regression_mu"].at[k].set(new_mu)
        self.mixture_components["regression_sigma"] = self.mixture_components["regression_sigma"].at[k].set(new_sigma)

    def word_cat_gibbs(self, sub, obs, weight):
        sample = topic_mixture_posterior(sub, obs, weight, self.mixture_components["generation"])
        return sample

    def reg_cat_gibbs(self, sub, reg, weight):
        sample = gaussian_mixture_posterior(sub, reg, weight, (self.mixture_components["regression_mu"], self.mixture_components["regression_sigma"]))
        return sample

    def doc_weight_gibbs(self, sub, params, z_gen, z_reg, scale_constant, predict=False):
        new_params = self._doc_weight_conditional(
            params,
            z_gen,
            z_reg,
            scale_constant,
            predict=predict
        )

        beta = dist.Beta(new_params[0], new_params[1]).sample(sub)
        beta = beta.at[..., -1].set(1.0)  # last entry is always 1
        return new_params, beta
    
    def update_doc_values(self, n, new_params, new_beta, doc_values):
        doc_values["P"][0] = doc_values["P"][0].at[n].set(new_params[0])
        doc_values["P"][1] = doc_values["P"][1].at[n].set(new_params[1])
        doc_values["G"] = doc_values["G"].at[n].set(mix_weights(new_beta))
        return doc_values

    def collapsed_doc_cats_gibbs(self, sub, depth, obs, reg, z_gen, z_reg, parent_cats, predict=False):
        if (depth == 0):
            weight = self.struct_values[f"G{depth+1}"]
            cluster_weight = self.struct_values[f"LG{depth}"].flatten()
        else:
            rev_idx = jnp.flip(parent_cats, axis=0)
            weight = gather_middle_slice(self.struct_values[f"G{depth+1}"], rev_idx)
            cluster_weight = partial_index(self.struct_values[f"LG{depth}"], parent_cats)
        assert weight.shape[0] == self.cluster_dims[depth]
        assert cluster_weight.shape[0] == self.cluster_dims[depth]
        log_probs = []
        for cat_idx in range(self.cluster_dims[depth]):
            log_prob = 0.0
            for word, label in zip(obs, z_gen):
                log_prob += jnp.log(weight[cat_idx, label] + 1e-12) + dist.Multinomial(total_count=1, probs=self.mixture_components["generation"][label]).log_prob(word)
            if (not predict):
                log_prob += jnp.log(weight[cat_idx, z_reg] + 1e-12) + dist.Normal(loc=self.mixture_components["regression_mu"][z_reg], scale=self.mixture_components["regression_sigma"][z_reg]).log_prob(reg)
            log_probs.append(log_prob)
        log_prob = jnp.stack(log_probs, axis=0)  # (num_cats,)

        unnormalized_prob = log_prob + jnp.log(cluster_weight + 1e-12)
        prob = jax.nn.softmax(unnormalized_prob)
        cat = dist.Categorical(probs=prob).sample(sub)
        return cat, prob

    def struct_weights_gibbs(self, key, depth, rev_cat, matching_z_gen, matching_z_reg, scale_constant):
        new_params = self._cat_weight_conditional(depth, rev_cat, matching_z_gen, matching_z_reg, scale_constant)

        key, sub = jax.random.split(key)
        beta = dist.Beta(new_params[0], new_params[1]).sample(sub)
        beta = beta.at[..., -1].set(1.0)  # last entry is always 1
   
        self.struct_values[f"P{depth}"][0] = set_by_multi_index(self.struct_values[f"P{depth}"][0], rev_cat, new_params[0])

        self.struct_values[f"P{depth}"][1] = set_by_multi_index(self.struct_values[f"P{depth}"][1], rev_cat, new_params[1])

        self.struct_values[f"G{depth}"] = set_by_multi_index(self.struct_values[f"G{depth}"], rev_cat, mix_weights(beta))

    def struct_cluster_gibbs(self, key, depth, row_idx, cats, local_category_assignments, scale_constant):
        new_params = self._cluster_weight_conditional(depth, cats, local_category_assignments[:, depth][row_idx], scale_constant)

        key, sub = jax.random.split(key)

        beta = dist.Beta(new_params[0], new_params[1]).sample(sub)
        beta = beta.at[..., -1].set(1.0)  # last entry is always 1
        self.struct_values[f"LP{depth}"][0] = set_by_multi_index(self.struct_values[f"LP{depth}"][0], cats, new_params[0])
        self.struct_values[f"LP{depth}"][1] = set_by_multi_index(self.struct_values[f"LP{depth}"][1], cats, new_params[1])
        self.struct_values[f"LG{depth}"] = set_by_multi_index(self.struct_values[f"LG{depth}"], cats, mix_weights(beta))

    def _cat_weight_conditional(self, depth, rev_cat, word_cats, reg_cats, scale_constant):
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
        if (depth == 0):
            params = [self.struct_values["Prior0"][0], self.struct_values["Prior0"][1]]
        else:
            params = [partial_index(self.struct_values[f"Prior{depth}"][0], rev_cat), partial_index(self.struct_values[f"Prior{depth}"][1], rev_cat)]

        cat_count = jnp.bincount(word_cats.ravel(), length=self.K)
        cat_idx = jnp.arange(self.K)
        reg_count = jnp.bincount(reg_cats.ravel(), length=self.K)
        cat_count = cat_count + reg_count

        alpha_bias = jnp.zeros(self.K, dtype=jnp.int32).at[cat_idx].set(cat_count)
        beta_bias = suffix_sum(alpha_bias)

        new_params = [params[0] + alpha_bias*scale_constant, params[1] + beta_bias*scale_constant]

        return new_params

    def _doc_weight_conditional(self, params, word_cats, reg_cats, scale_constant, predict=False):
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
        if not predict:
            reg_count = jnp.bincount(reg_cats.ravel(), length=self.K)
            cat_count = cat_count + reg_count

        alpha_bias = jnp.zeros((self.K,), dtype=jnp.int32).at[cat_idx].set(cat_count)
        beta_bias = suffix_sum(alpha_bias)

        new_params = [params[0] + alpha_bias*scale_constant, params[1] + beta_bias*scale_constant]
        return new_params

    def _cluster_weight_conditional(self, depth, cats, local_cluster_cats, scale_constant):
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

        params = [partial_index(self.struct_values[f"LPrior{depth}"][0], cats), partial_index(self.struct_values[f"LPrior{depth}"][1], cats)]
        cat_count = jnp.bincount(local_cluster_cats.ravel(), length=self.cluster_dims[depth])
        cat_idx = jnp.arange(self.cluster_dims[depth])
        alpha_bias = jnp.zeros(self.cluster_dims[depth], dtype=jnp.int32).at[cat_idx].set(cat_count)
        beta_bias = suffix_sum(alpha_bias)

        new_params = [params[0] + alpha_bias*scale_constant, params[1] + beta_bias*scale_constant]
        return new_params

    def vectorized_word_cat_gibbs(self, key, obs, doc_weights):
        """
        Vectorized Gibbs sampling for all documents and words.
        Args:
            key: PRNGKey
            obs: (N, M, V)
            doc_weights: (N, K)
        Returns:
            z_gen: (N, M)
        """
        N, M, _ = obs.shape

        # Split one key per document
        doc_keys = jax.random.split(key, N)

        def sample_doc(doc_key, obs_doc, doc_weight):
            # Split per word key
            word_keys = jax.random.split(doc_key, M)
            # Vectorize over words
            word_samples = jax.vmap(
                lambda k, w: self.word_cat_gibbs(k, w, doc_weight)
            )(word_keys, obs_doc)
            return word_samples

        # Vectorize across documents
        z_gen = jax.vmap(sample_doc)(doc_keys, obs, doc_weights)
        return z_gen

    def vectorized_reg_cat_gibbs(self, key, reg, doc_weights):
        """
        Vectorized Gibbs sampling for regression categories across all documents.
        Args:
            key: PRNGKey
            reg: (N,) regression scores
            doc_weights: (N, K) document mixture weights
        Returns:
            z_reg: (N,) sampled category indices
        """
        N = reg.shape[0]

        # Split one key per document
        subkeys = jax.random.split(key, N)

        # Vectorize across documents
        z_reg = jax.vmap(
            lambda k, r, w: self.reg_cat_gibbs(k, r, w)
        )(subkeys, reg, doc_weights)

        return z_reg

    def vectorized_doc_weight_gibbs(self, key, doc_values, z_gen, z_reg, scale_constant, predict=False):
        """
        Vectorized Gibbs update of document-level stick-breaking weights for all documents.
        Args:
            key: PRNGKey
            doc_values: dict with fields ["B"], ["Prior"], ["P"], ["G"]
            z_gen: (N, M) word category assignments
            z_reg: (N,) regression category assignments
        Returns:
            Updated doc_values
        """
        import matplotlib.pyplot as plt
        N = z_gen.shape[0]

        # Split one key per document
        subkeys = jax.random.split(key, N)

        # Prepare per-document inputs
        Prior0 = doc_values["Prior"][0]          # (N, K)
        Prior1 = doc_values["Prior"][1]          # (N, K)
        Priors = jnp.stack([Prior0, Prior1], axis=1)  # (N, 2, K)

        # vmap over documents
        def update_doc(k, P, zg, zr):
            params = [P[0], P[1]]
            new_params, new_beta = self.doc_weight_gibbs(
                k, params, zg, zr, scale_constant, predict
            )
            new_G = mix_weights(new_beta)
            return new_params[0], new_params[1], new_beta, new_G

        alpha_new, beta_new, B_new, G_new = jax.vmap(update_doc)(subkeys, Priors, z_gen, z_reg)

        # Update doc_values in a single functional operation
        doc_values = {
            **doc_values,
            "P": [alpha_new, beta_new],
            "B": B_new,
            "G": G_new,
        }
        return doc_values

    def infer_doc_cats(self, key, depth, doc_values, local_category_assignments):
        N = local_category_assignments.shape[0]

        # Split RNG keys for all documents
        keys = jax.random.split(key, N)

        def update_one(sub, doc_nu, parent_cats):
            cat_z, prob = self.doc_cats_gibbs(sub, depth, doc_nu, parent_cats)
            return cat_z, prob

        # Vectorized over documents
        new_cats, probs = jax.vmap(update_one)(
            keys,
            doc_values["B"],                     # shape (N, C, ...)
            local_category_assignments[:, :depth]  # shape (N, depth)
        )

        # Update all at once
        local_category_assignments = local_category_assignments.at[:, depth].set(new_cats)
        print(f"Depth {depth} category assignment probabilities:", probs)
        return local_category_assignments
    
    def update_doc_prior_batch(self, doc_values, local_category_assignments):
        depth = len(self.cluster_dims)
        N = local_category_assignments.shape[0]

        def per_doc(rev_cat):
            rev_cat_tuple = tuple(jnp.flip(rev_cat).astype(int))
            a, b = gen_next_level_prior(
                jnp.atleast_2d(self.struct_values[f"G{depth}"][rev_cat_tuple]),
                jnp.atleast_2d(self.struct_params[f"alpha{depth}"][rev_cat_tuple])
            )
            return a.flatten(), b.flatten()

        # Vectorize over documents
        A, B = jax.vmap(per_doc)(local_category_assignments)

        # Write results back in one go
        doc_values["Prior"] = (
            doc_values["Prior"][0].at[:N].set(A),
            doc_values["Prior"][1].at[:N].set(B),
        )

        return doc_values

    def collapsed_doc_cats_gibbs_batch(self, key, depth, obs, reg, z_gen, z_reg, local_category_assignments, predict=False):
        """
        Vectorized collapsed_doc_cats_gibbs over all documents (and words inside each doc).
        """
        N = obs.shape[0]
        keys = jax.random.split(key, N)

        def single_doc(sub, obs_i, reg_i, z_gen_i, z_reg_i, parent_cats_i):
            parent_cats_i = jnp.asarray(parent_cats_i, jnp.int32)

            # --- Category weight selection ---
            if depth == 0:
                weight = self.struct_values[f"G{depth+1}"]
                cluster_weight = self.struct_values[f"LG{depth}"].flatten()
            else:
                rev_idx = jnp.flip(parent_cats_i, axis=0)
                weight = gather_middle_slice(self.struct_values[f"G{depth+1}"], rev_idx)
                cluster_weight = partial_index(self.struct_values[f"LG{depth}"], parent_cats_i)

            assert weight.shape[0] == self.cluster_dims[depth], "Weight shape mismatch."
            # --- Word-level likelihoods (vectorized over words) ---

            cat_counts = jnp.bincount(z_gen_i, length=self.K)
            if not predict:
                cat_counts = cat_counts + jnp.bincount(jnp.atleast_1d(z_reg_i), length=self.K)
            log_prob = jnp.log(weight + 1e-12) * jnp.broadcast_to(jnp.expand_dims(cat_counts, axis=0), weight.shape)
            log_prob = jnp.sum(log_prob, axis=1)
                        
            # --- Category sampling ---
            unnorm = log_prob + jnp.log(cluster_weight + 1e-12)
            prob = jax.nn.softmax(unnorm)
            cat = dist.Categorical(probs=prob).sample(sub)
            return cat, prob

        # --- Prepare parents safely (empty when depth == 0) ---
        parents = (
            local_category_assignments[:, :depth].astype(jnp.int32)
            if depth > 0 else jnp.zeros((N, 0), dtype=jnp.int32)
        )

        cats, probs = jax.vmap(single_doc)(
            keys, obs, reg, z_gen, z_reg, parents
        )
        return cats, probs


if __name__ == "__main__":
    toy_struct = {"G0": 5, "G1": 3, "G2": 2}
    model = HDMM(toy_struct, vocab_size=11)
    print("Model initialized.")
    N = 7
    M = 17
    V = 11
    key = jax.random.PRNGKey(0)
    key, sub = jax.random.split(key)
    obs = jax.random.randint(sub, (N, M, V), 0, 2)
    key, sub = jax.random.split(key)
    reg = jax.random.normal(sub, (N,))
    z_gen, z_reg, local_category_assignments, mc, log_prob = model.infer(obs, reg, num_iters=200, key=key)
    print("Inference completed.")
    likelihood_visualization(log_prob, np.zeros_like(log_prob), epoch=0)
