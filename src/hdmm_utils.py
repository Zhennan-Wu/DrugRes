import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist


@jax.jit
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


@jax.jit
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


@jax.jit
def dirichlet_posterior(sub, obs, params):
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

    new_params = params + value

    sample = dist.Dirichlet(new_params).sample(sub)
    return sample


@jax.jit
def nig_posterior(key, obs, params):
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
    kappa = params[1] + count
    mu = (params[1] * params[0] + count * mean) / kappa
    alpha = params[2] + count / 2
    beta = params[3] + 0.5 * sum_var + (params[1] * count * (mean - params[0]) ** 2) / (2 * kappa)

    key, sub = random.split(key)
    new_sigma = dist.InverseGamma(alpha, beta).sample(sub)
    key, sub = random.split(key)
    new_mu = dist.Normal(mu, jnp.sqrt(new_sigma / kappa)).sample(sub)
    return [jnp.squeeze(new_mu), jnp.squeeze(new_sigma)]

@jax.jit
def gaussian_mixture_posterior(key, score, weight, components):
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
    un_normalized = log_probs + jnp.log(weight + 1e-12)

    cat_prob = jax.nn.softmax(un_normalized, axis=-1)
    key, sub = random.split(key)
    sample = dist.Categorical(probs=cat_prob).sample(sub)
    return sample, key

@jax.jit
def topic_mixture_posterior(sub, word, weight, components):
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
    un_normalized = log_probs + jnp.log(weight + 1e-12)
    cat_prob = jax.nn.softmax(un_normalized, axis=-1)

    sample = dist.Categorical(probs=cat_prob).sample(sub)
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

    return unique_rows, positions

@jax.jit
def beta_mixture_posterior(key, doc_nu, cat_param, cluster_prob):
    """
    Sample document base-cluster category assignment given stick-breaking weights and cluster probabilities.
    Args:
        key: JAX PRNGKey
        doc_nu: (C,) document-level stick-breaking weights
        cat_param: list of two (C,) arrays [alpha, beta] for the Beta prior
        cluster_prob: (C,) array of cluster probabilities
    Returns:
        new_cat: int, sampled category index from 0 to C-1
        new_key: updated JAX PRNGKey
    """
    non_trivial_thres = 1e-2
    doc_non_trivial_indices = jnp.where(doc_nu[..., :-1] > non_trivial_thres)[0]
    doc_alpha = cat_param[0]
    doc_beta = cat_param[1]
    doc_alpha = jnp.clip(doc_alpha, a_min=1e-8)
    doc_beta = jnp.clip(doc_beta, a_min=1e-8)
    doc_nu = jnp.clip(doc_nu, a_min=1e-8, a_max=1-1e-8)
    doc_nu_cat_log_prob = dist.Beta(doc_alpha[..., :-1], doc_beta[..., :-1]).log_prob(doc_nu[..., :-1])

    adjusted_doc_nu_cat_log_prob = doc_nu_cat_log_prob[..., doc_non_trivial_indices]
    doc_nu_cat_log_prob = jnp.sum(adjusted_doc_nu_cat_log_prob, axis=-1)  # (C,)

    un_normalized = doc_nu_cat_log_prob + jnp.log(cluster_prob + 1e-12)
    prob = jax.nn.softmax(un_normalized.reshape(-1))
    key, sub = random.split(key)
    new_cat = dist.Categorical(probs=prob).sample(sub)
    return new_cat, key