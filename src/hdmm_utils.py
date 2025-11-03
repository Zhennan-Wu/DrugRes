import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist


# @jax.jit
def mix_weights(beta, axis=-1):
    """
    Compute mixture weights from stick-breaking proportions beta.
    beta: (..., K) where K is the number of sticks/components
    axis: axis along which to perform stick-breaking
    Returns: weights of same shape as beta
    """
    # Compute cumulative product of (1-v)
    remaining = jnp.cumprod(1 - beta + 1e-10, axis=axis)

    # Shift along axis: prepend ones, drop last element
    ones_shape = list(remaining.shape)
    ones_shape[axis] = 1
    ones = jnp.ones(ones_shape, dtype=remaining.dtype)
    remaining = jnp.concatenate([ones, remaining[..., :-1]], axis=axis)

    # Stick weights
    weights = beta * remaining
    return weights


# @jax.jit
def suffix_sum(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute suffix sums along the last dimension of a tensor.
    Each entry is the sum of all elements to its right.
    The last element along that dimension is always 0.
    
    Example:
        x = jnp.array([1,2,3])
        suffix_sum(x) -> [5,3,0]
        
        x = jnp.array([[1,2,3],[4,5,6]])
        suffix_sum(x) -> [[5,3,0],
                           [11,6,0]]
    """
    # Flip along the last dimension
    rev = jnp.flip(x, axis=-1)
    # Cumulative sum on the flipped tensor
    rev_cumsum = jnp.cumsum(rev, axis=-1)
    # Flip back
    suffix = jnp.flip(rev_cumsum, axis=-1)
    # Subtract original to exclude current element
    suffix = suffix - x
    return jnp.clip(suffix, a_min=1e-10)


def dirichlet_posterior(sub, obs, params, scaling_constant):
    """
    Sample generation component parameters given assigned observations and Dirichlet prior.
    Args:
        key: JAX PRNGKey
        obs: (N_obs, V) array of assigned one-hot word observations
        params: (V,) array of Dirichlet prior parameters
    Returns:
        sample: (V,) array of sampled generation component parameters
        new_key: updated JAX PRNGKey
    """
    value = jnp.sum(obs, axis=0)

    new_params = params + value*scaling_constant

    sample = dist.Dirichlet(new_params).sample(sub)
    return sample


def nig_posterior(key, obs, params, scale_constant):
    """
    Sample regression component parameters given assigned observations and Normal-Inverse-Gamma prior.
    Args:
        key: JAX PRNGKey
        obs: (N_obs,) array of assigned regression observations
        params: list of four scalars [mu, kappa, alpha, beta] for the NIG prior
    Returns:
        new_params: list of two scalars [new_mu, new_sigma] for the sampled regression component
        new_key: updated JAX PRNGKey
    """
    count = float(obs.size)
    mean = jnp.mean(obs)
    sum_var = jnp.sum((obs - mean) ** 2, keepdims=True)
    kappa = params[1] + count * scale_constant
    mu = (params[1] * params[0] + count * mean * scale_constant) / kappa
    alpha = params[2] + count * scale_constant / 2
    beta = params[3] + 0.5 * sum_var * scale_constant + (params[1] * count * scale_constant * (mean - params[0]) ** 2) / (2 * kappa)

    key, sub = random.split(key)
    new_sigma = dist.InverseGamma(alpha, beta).sample(sub)
    key, sub = random.split(key)
    new_mu = dist.Normal(mu, jnp.sqrt(new_sigma / kappa)).sample(sub)
    return [jnp.squeeze(new_mu), jnp.squeeze(new_sigma)]


def nig_posterior_batch(key, count, mean, sum_var, params, scale_constant):
    """
    Vectorized Normal-Inverse-Gamma posterior update across K components.
    Args:
        key: PRNGKey
        count, mean, sum_var: (K,) arrays with sufficient statistics
        params: (4,) prior parameters (mu0, kappa0, alpha0, beta0)
    Returns:
        new_mu, new_sigma: (K,) arrays
    """
    mu0, kappa0, alpha0, beta0 = params
    kappa = kappa0 + count * scale_constant
    mu = (kappa0 * mu0 + count * mean * scale_constant) / kappa
    alpha = alpha0 + count * scale_constant / 2.0
    beta = beta0 + 0.5 * sum_var * scale_constant + (kappa0 * count * scale_constant * (mean - mu0) ** 2) / (2.0 * kappa)

    # Split PRNG for both InverseGamma and Normal sampling
    keys = random.split(key, 2)
    sigma = dist.InverseGamma(alpha, beta).sample(keys[0])
    mu_samp = dist.Normal(mu, jnp.sqrt(sigma / kappa)).sample(keys[1])
    return mu_samp, sigma


def gaussian_mixture_posterior(sub, score, weight, components, unknown_latent=False):
    """
    Sample category assignment for a single regression score given mixture weights and component distributions.
    Args:
        key: JAX PRNGKey
        score: float, regression score
        weight: (K,) mixture weights for the document
        components: (K, 2) component regression parameters (mean, variance)
    Returns:
        sample: int, sampled category index from 0 to K-1
        new_key: updated JAX PRNGKey
    """
    reg_dist = dist.Normal(loc=components[0], scale=jnp.sqrt(components[1]))
    log_probs = reg_dist.log_prob(score)
    if unknown_latent:
        un_normalized = log_probs
    else:
        un_normalized = log_probs + jnp.log(weight + 1e-12)

    cat_prob = jax.nn.softmax(un_normalized, axis=-1)
    sample = dist.Categorical(probs=cat_prob).sample(sub)
    return sample


def topic_mixture_posterior(sub, word, weight, components, unknown_latent=False):
    """
    Sample category assignment for a single word given mixture weights and component distributions.
    Args:
        key: JAX PRNGKey
        word: (V,) one-hot vector of the word
        weight: (K,) mixture weights for the document
        components: (K, V) component-word distributions 
    Returns:
        sample: int, sampled category index from 0 to K-1
        new_key: updated JAX PRNGKey
    """

    gen_dist = dist.Multinomial(total_count=1, probs=components)
    log_probs = gen_dist.log_prob(word)
    # print("log_probs:", log_probs.shape)
    # print("word:", word.shape)

    # print("weight:", weight.shape)
    if unknown_latent:
        un_normalized = log_probs
    else:
        un_normalized = log_probs + jnp.log(weight + 1e-12)
    cat_prob = jax.nn.softmax(un_normalized, axis=-1)
    # print("cat_prob:", cat_prob.shape)

    sample = dist.Categorical(probs=cat_prob).sample(sub)
    # print("sample:", sample.shape)
    return sample


def gen_next_level_prior(G_parent, alpha_param):
    param_alpha = alpha_param * G_parent
    param_beta = suffix_sum(param_alpha)

    return [param_alpha, param_beta]


def get_unique_rows_and_positions(x: jnp.ndarray):
    """
    Get all unique rows in data and their positions.
    Args:
        data: (N, D) array of data points
        category_assignments: (N,) array of category assignments
        category_index: int, target category index
    Returns:
        selected_data: (N_selected, D) array of data points assigned to category_index
    """

    # Step 1: Get unique rows and inverse mapping
    unique_rows, inv_idx = jnp.unique(x, axis=0, return_inverse=True)

    # Step 2: Group positions for each unique value
    def gather_positions(i):
        return jnp.nonzero(inv_idx == i, size=None)[0]

    positions = [gather_positions(i) for i in range(unique_rows.shape[0])]

    # print("unique_rows:", unique_rows)
    # print("positions:", positions)
    # print("x", x)
    return unique_rows, positions


def beta_mixture_posterior(sub, doc_nu, cat_param, cluster_prob):
    """
    Sample document base-cluster category assignment given stick-breaking weights and cluster probabilities.
    Args:
        sub: JAX PRNGKey
        doc_nu: (C,) document-level stick-breaking weights
        cat_param: list of two (C,) arrays [alpha, beta] for the Beta prior
        cluster_prob: (C,) array of cluster probabilities
    Returns:
        new_cat: int, sampled category index from 0 to C-1
    """
    non_trivial_thres = 1e-2
    doc_alpha, doc_beta = cat_param

    # clip to numerical stability range
    doc_alpha = jnp.clip(doc_alpha, a_min=1e-8)
    doc_beta  = jnp.clip(doc_beta,  a_min=1e-8)
    doc_nu    = jnp.clip(doc_nu,    a_min=1e-8, a_max=1-1e-8)

    # compute log probs for Beta on all components except last
    # print("doc_alpha:", doc_alpha.shape)
    # print("truncated to:", doc_alpha[..., :-1].shape)
    # print("doc_beta:", doc_beta.shape)
    # print("truncated to:", doc_beta[..., :-1].shape)
    # print("doc_nu:", doc_nu.shape)
    # print("truncated to:", doc_nu[..., :-1].shape)
    log_prob_all = dist.Beta(doc_alpha[..., :-1], doc_beta[..., :-1]).log_prob(doc_nu[..., :-1])
    # print("log_prob_all:", log_prob_all.shape)
    # mask small entries instead of using jnp.where (JIT-safe)
    lbd_mask = (doc_nu[..., :-1] > non_trivial_thres)
    masked_log_prob = jnp.where(lbd_mask, log_prob_all, 0.0)
    upbd_mask = (doc_nu[..., :-1] < 1 - non_trivial_thres)
    masked_log_prob = jnp.where(upbd_mask, masked_log_prob, 0.0)
    # print("masked_log_prob:", masked_log_prob.shape)

    # sum over categories
    doc_nu_cat_log_prob = jnp.sum(masked_log_prob)
    # print("doc_nu_cat_log_prob:", doc_nu_cat_log_prob.shape)
    # add cluster log probs
    un_normalized = doc_nu_cat_log_prob + jnp.log(cluster_prob + 1e-12)
    # print("cluster_prob:", cluster_prob.shape)
    prob = jax.nn.softmax(un_normalized)
    # categorical sample
    new_cat = dist.Categorical(probs=prob).sample(sub)
    # print("new_cat:", new_cat.shape)
    return new_cat, jnp.atleast_1d(prob)


def gather_middle_slice(x, idx):
    """
    x: (D0, D1, ..., D{k-1})
    idx: (k-2,) selecting D1..D{k-2}
    Returns: (D0, D{k-1}) subarray
    """
    shape = jnp.array(x.shape[1:-1], dtype=jnp.int32)         # D1..D{k-2}
    strides = jnp.cumprod(jnp.concatenate([jnp.array([1]), shape[:-1]]))
    flat_index = jnp.sum(idx * strides)                       # scalar offset
    flat_x = x.reshape(x.shape[0], -1, x.shape[-1])           # (D0, prod, D{k-1})
    return flat_x[:, flat_index, :]                            # (D0, D{k-1})


# def partial_index(a, idx):
#     if isinstance(idx, tuple) and any(isinstance(i, slice) for i in idx):
#         return a
#     # a.shape = (D1, D2, ..., Dn)
#     # idx.shape = (k,) where k < n
#     k = idx.shape[0]
#     prefix_shape = a.shape[:k]
#     flat_idx = jnp.ravel_multi_index(idx, prefix_shape)
#     sub = a.reshape((-1,) + a.shape[k:])
#     return sub[flat_idx]
def partial_index(a, idx, mode="clip"):
    """
    JIT-safe partial index: returns a[ idx[0], idx[1], ..., idx[k-1], :... ]
    where len(idx) = k < a.ndim.
    """
    if isinstance(idx, tuple) and any(isinstance(i, slice) for i in idx):
        return a

    idx = jnp.atleast_1d(jnp.asarray(idx, jnp.int32))
    k = idx.shape[0]
    prefix_shape = a.shape[:int(k)]   # static tuple at trace time
    tail_shape   = a.shape[int(k):]

    # Compute flat index manually (no ravel_multi_index)
    sizes = jnp.array(prefix_shape)
    if mode == "clip":
        idx = jnp.clip(idx, 0, sizes - 1)
    elif mode == "wrap":
        idx = jnp.mod(idx, sizes)

    strides = jnp.concatenate([jnp.cumprod(sizes[::-1])[::-1][1:], jnp.array([1])])
    flat_idx = jnp.sum(idx * strides)

    # Flatten prefix and index
    sub = a.reshape((-1,) + tail_shape)
    return sub[flat_idx]

def set_by_multi_index(a, idx, value):
    """
    Set a[ idx[0], idx[1], ..., idx[k-1], ... ] = value
    where idx is a prefix index (k <= a.ndim).
    If idx includes slice(None), we defer to .at[idx].set(value).
    value must be broadcastable to a.shape[k:].
    """
    # Handle "select all" like (slice(None),) or any slice in the tuple
    if isinstance(idx, tuple) and any(isinstance(i, slice) for i in idx):
        return a.at[idx].set(value)

    idx = jnp.atleast_1d(jnp.asarray(idx))
    k = int(idx.shape[0])
    assert 1 <= k <= a.ndim, f"prefix length k={k} must be in [1, {a.ndim}]"

    # Flatten the first k dims into one, index that, then reshape back
    prefix_shape = a.shape[:k]
    tail_shape   = a.shape[k:]
    flat_idx = jnp.ravel_multi_index(idx, prefix_shape)

    sub = a.reshape((-1,) + tail_shape)
    # value can be scalar or shaped like tail_shape (or broadcastable)
    return sub.at[flat_idx].set(value).reshape(a.shape)
