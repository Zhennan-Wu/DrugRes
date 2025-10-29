# dbm_jax.py
from typing import List, Optional, Tuple
import functools

import jax
import jax.numpy as jnp
import equinox as eqx
import optax


# -------------------------
# Utilities: float <-> bits
# -------------------------
def float2bit(x: jnp.ndarray, bits: int) -> jnp.ndarray:
    """
    Convert a floating tensor x in [0,1] to bit representation with `bits` LSBs.
    Input shape: (...,) or (N, ...). Output last-dim becomes `bits`.
    This mirrors typical quantization: round(x * (2^bits - 1)) then to bits.
    """
    if bits <= 0:
        raise ValueError("bits must be positive")
    maxv = (1 << bits) - 1
    xi = jnp.clip(x, 0.0, 1.0)
    xi = jnp.round(xi * maxv).astype(jnp.uint32)
    # produce bits along last axis (LSB last)
    def to_bits(n):
        # returns array (bits,)
        b = [(n >> i) & 1 for i in range(bits)]
        return jnp.array(b, dtype=jnp.uint8)
    # vectorize over elements
    flat = xi.ravel()
    bits_arr = jax.vmap(to_bits)(flat)
    bits_arr = bits_arr.reshape(*xi.shape, bits)
    return bits_arr


def bit2float(bits: jnp.ndarray, bits_num: int) -> jnp.ndarray:
    """
    Convert bit representation back to float in [0,1].
    bits shape: (..., bits_num), values {0,1}
    """
    if bits_num <= 0:
        raise ValueError("bits_num must be positive")
    # assume LSB order used in float2bit (index 0 is bit0)
    idx = jnp.arange(bits_num, dtype=jnp.uint32)
    pow2 = (1 << idx).astype(jnp.uint32)
    flat = bits.astype(jnp.uint32)
    val = jnp.tensordot(flat, pow2, axes=[-1, 0])
    maxv = (1 << bits_num) - 1
    return val.astype(jnp.float32) / jnp.float32(maxv)


# -------------------------
# Helper inits
# -------------------------
def orthogonal_init(key, shape: Tuple[int, int]) -> jnp.ndarray:
    """
    Orthogonal initialization (as in torch.nn.init.orthogonal_).
    shape = (out_features, in_features)
    """
    akey, skey = jax.random.split(key)
    flat_shape = (shape[0], shape[1])
    mat = jax.random.normal(akey, flat_shape)
    # compute QR
    q, r = jnp.linalg.qr(mat)
    # make uniform orthonormal like torch: adjust signs with diag(r)
    d = jnp.sign(jnp.diag(r))
    q = q * d.reshape((-1, 1))
    return q.astype(jnp.float32)


def zeros_init(shape: Tuple[int, ...]) -> jnp.ndarray:
    return jnp.zeros(shape, dtype=jnp.float32)


# -------------------------
# DBM in Equinox
# -------------------------
class DBM(eqx.Module):
    # parameters
    weight: List[jnp.ndarray]
    bias: List[jnp.ndarray]

    # hyperparams stayed as attributes
    nv: int
    nh: int
    size: int
    nc: int
    bits: int
    L: int

    # options (these are static flags, not parameters)
    marginal: bool = False

    def __init__(
        self,
        size: int,
        nc: int,
        nh: Optional[int] = None,
        bits: int = 8,
        L: int = 2,
        *,
        key: Optional[jnp.ndarray] = None
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        else:
            key = jax.random.PRNGKey(int(key[0]))  # ensure reproducibility if given array
        keys = jax.random.split(key, L + L + 3)  # enough randomness for inits

        total_bit = nc * bits
        nv = nc * bits * size * size
        if nh is None:
            nh = nv

        # weights: weight[0]: (nh, nv); weight[1..L-1]: (nh, nh)
        w0 = orthogonal_init(keys[0], (nh, nv))
        w_list = [w0]
        for i in range(1, L):
            w_list.append(orthogonal_init(keys[i], (nh, nh)))

        # biases: bias[0] -> (nv,), bias[1..L] -> (nh,)
        b0 = zeros_init((nv,))
        b_list = [b0] + [zeros_init((nh,)) for _ in range(L)]

        super().__init__(
            weight=w_list,
            bias=b_list,
            nv=nv,
            nh=nh,
            size=size,
            nc=nc,
            bits=bits,
            L=L,
            marginal=False,
        )

    # -------------------------
    # Core - energy and linear
    # -------------------------
    def linear(self, x: jnp.ndarray, W: jnp.ndarray, b: Optional[jnp.ndarray]):
        # x: (N, in_features)
        # W: (out_features, in_features)
        y = x @ W.T
        if b is not None:
            y = y + b
        return y

    def energy(self, v: jnp.ndarray, h: List[jnp.ndarray]) -> jnp.ndarray:
        # v shape (N, nv)
        energy = - jnp.sum(v * self.bias[0][None, :], axis=1)
        for i in range(self.L):
            if i == 0:
                inp = v
            else:
                inp = h[i - 1]
            logits = self.linear(inp, self.weight[i], self.bias[i + 1])
            energy = energy - jnp.sum(h[i] * logits, axis=1)
        return energy

    # -------------------------
    # Gibbs step (non-jitted high-level)
    # -------------------------
    def gibbs_step(
        self,
        v: jnp.ndarray,
        h: List[jnp.ndarray],
        key: jnp.ndarray
,
        fix_v: bool = False,
        rand_v: Optional[jnp.ndarray] = None,
        rand_h: Optional[List[jnp.ndarray]] = None,
        rand_u: Optional[jnp.ndarray] = None,
        T: float = 1.0,
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Performs one Gibbs-like synchronous update split into even/odd batches (mirrors original).
        This returns new v_, h_ (note: original returned old and new; here we return updated).
        """
        N = v.shape[0]
        keys = jax.random.split(key, 4)
        rand_u_local = rand_u if rand_u is not None else jax.random.uniform(keys[0], (N,))
        even = rand_u_local < 0.5
        odd = ~even

        # copy
        v_ = v.copy()
        h_ = [h_i.copy() for h_i in h]

        def sample_bernoulli_from_logits(key, logits, T, rand=None):
            if T == 0:
                return (logits >= 0.0).astype(jnp.float32)
            logits_T = logits / T
            probs = jax.nn.sigmoid(logits_T)
            if rand is None:
                r = jax.random.uniform(key, probs.shape)
            else:
                r = rand
            return (r < probs).astype(jnp.float32)

        # helper to handle even/odd indices using boolean masks
        # process 'even' block
        if jnp.any(even):
            if not fix_v:
                logits_v = self.linear(h[0][even], self.weight[0].T, self.bias[0])
                # logits shape (M, nv)
                if T == 0:
                    v_ = v_.at[even].set((logits_v >= 0.0).astype(jnp.float32))
                else:
                    key_v = jax.random.fold_in(keys[1], 0)
                    if rand_v is None:
                        samp = jax.random.bernoulli(key_v, p=jax.nn.sigmoid(logits_v)).astype(jnp.float32)
                    else:
                        samp = (rand_v[even] < jax.nn.sigmoid(logits_v)).astype(jnp.float32)
                    v_ = v_.at[even].set(samp)

            # update odd-indexed hidden layers using even mask (i odd)
            for i in range(1, len(h), 2):
                logits = self.linear(h[i - 1][even], self.weight[i], self.bias[i + 1])
                if i + 1 < len(h):
                    logits = logits + self.linear(h[i + 1][even], self.weight[i + 1].T, None)
                if T == 0:
                    hnew = (logits >= 0.0).astype(jnp.float32)
                else:
                    if rand_h is None:
                        key_h = jax.random.fold_in(keys[2], i)
                        hnew = jax.random.bernoulli(key_h, p=jax.nn.sigmoid(logits)).astype(jnp.float32)
                    else:
                        hnew = (rand_h[i][even] < jax.nn.sigmoid(logits)).astype(jnp.float32)
                h_ = [h_.at[i].set(hnew) if idx == i else h_ for idx in range(len(h_))]

            # update even-indexed hidden layers (i even)
            for i in range(0, len(h), 2):
                inp = v_[even] if i == 0 else h_[i - 1][even]
                logits = self.linear(inp, self.weight[i], self.bias[i + 1])
                if i + 1 < len(h):
                    logits = logits + self.linear(h_[i + 1][even], self.weight[i + 1].T, None)
                if T == 0:
                    hnew = (logits >= 0.0).astype(jnp.float32)
                else:
                    if rand_h is None:
                        key_h2 = jax.random.fold_in(keys[3], i + 10)
                        hnew = jax.random.bernoulli(key_h2, p=jax.nn.sigmoid(logits)).astype(jnp.float32)
                    else:
                        hnew = (rand_h[i][even] < jax.nn.sigmoid(logits)).astype(jnp.float32)
                h_ = [h_.at[i].set(hnew) if idx == i else h_ for idx in range(len(h_))]

        # process 'odd' block
        if jnp.any(odd):
            for i in range(0, len(h), 2):
                inp = v_[odd] if i == 0 else h_[i - 1][odd]
                logits = self.linear(inp, self.weight[i], self.bias[i + 1])
                if i + 1 < len(h):
                    logits = logits + self.linear(h_[i + 1][odd], self.weight[i + 1].T, None)
                if T == 0:
                    hnew = (logits >= 0.0).astype(jnp.float32)
                else:
                    if rand_h is None:
                        key_h3 = jax.random.fold_in(keys[1], i + 20)
                        hnew = jax.random.bernoulli(key_h3, p=jax.nn.sigmoid(logits)).astype(jnp.float32)
                    else:
                        hnew = (rand_h[i][odd] < jax.nn.sigmoid(logits)).astype(jnp.float32)
                h_ = [h_.at[i].set(hnew) if idx == i else h_ for idx in range(len(h_))]

            if not fix_v:
                logits_v = self.linear(h_[0][odd], self.weight[0].T, self.bias[0])
                if T == 0:
                    v_ = v_.at[odd].set((logits_v >= 0.0).astype(jnp.float32))
                else:
                    key_v2 = jax.random.fold_in(keys[2], 99)
                    if rand_v is None:
                        samp2 = jax.random.bernoulli(key_v2, p=jax.nn.sigmoid(logits_v)).astype(jnp.float32)
                    else:
                        samp2 = (rand_v[odd] < jax.nn.sigmoid(logits_v)).astype(jnp.float32)
                    v_ = v_.at[odd].set(samp2)

            for i in range(1, len(h), 2):
                logits = self.linear(h_[i - 1][odd], self.weight[i], self.bias[i + 1])
                if i + 1 < len(h):
                    logits = logits + self.linear(h_[i + 1][odd], self.weight[i + 1].T, None)
                if T == 0:
                    hnew = (logits >= 0.0).astype(jnp.float32)
                else:
                    if rand_h is None:
                        key_h4 = jax.random.fold_in(keys[3], i + 30)
                        hnew = jax.random.bernoulli(key_h4, p=jax.nn.sigmoid(logits)).astype(jnp.float32)
                    else:
                        hnew = (rand_h[i][odd] < jax.nn.sigmoid(logits)).astype(jnp.float32)
                h_ = [h_.at[i].set(hnew) if idx == i else h_ for idx in range(len(h_))]

        return v_, h_

    # -------------------------
    # MH step (Metropolis-Hastings)
    # -------------------------
    def mh_step(
        self,
        v: jnp.ndarray,
        h: List[jnp.ndarray],
        key: jnp.ndarray
,
        fix_v: bool = False,
        rand_v: Optional[jnp.ndarray] = None,
        rand_h: Optional[List[jnp.ndarray]] = None,
        rand_u: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        N = v.shape[0]
        keys = jax.random.split(key, 3)
        if fix_v:
            v_prop = v
        else:
            if rand_v is None:
                v_prop = jax.random.bernoulli(keys[0], p=0.5, shape=v.shape).astype(jnp.float32)
            else:
                v_prop = (rand_v < 0.5).astype(jnp.float32)

        if rand_h is None:
            h_prop = [jax.random.bernoulli(jax.random.fold_in(keys[1], i), p=0.5, shape=h[i].shape).astype(jnp.float32) for i in range(self.L)]
        else:
            h_prop = [(rand_h[i] < 0.5).astype(jnp.float32) for i in range(self.L)]

        # log_ratio = energy(v, h) - energy(v_prop, h_prop)
        log_ratio = self.energy(v, h) - self.energy(v_prop, h_prop)

        if rand_u is None:
            # accept with min(1, exp(log_ratio))
            u = jax.random.uniform(keys[2], shape=log_ratio.shape)
            accepted = u < jnp.clip(jnp.exp(log_ratio), a_min=0.0, a_max=1.0)
        else:
            accepted = rand_u < jnp.exp(log_ratio)

        accepted = accepted.astype(jnp.bool_)

        # where accepted, replace
        v_new = jnp.where(accepted[:, None], v_prop, v)
        h_new = [jnp.where(accepted[:, None], h_prop[i], h[i]) for i in range(self.L)]
        return v_new, h_new

    # -------------------------
    # Local search and coupling logic
    # -------------------------
    def local_search(self, v: jnp.ndarray, h: List[jnp.ndarray], key: jnp.ndarray
, fix_v: bool = False) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        """
        Iteratively run Gibbs step with T=0 until convergence (component-wise equality).
        """
        keys = jax.random.split(key, 128)
        kiter = 0

        def cond_fn(state):
            v_curr, h_curr, converged, keyi = state
            return jnp.logical_not(jnp.all(converged))

        def body_fn(state):
            v_curr, h_curr, converged, keyi = state
            key1, key2 = jax.random.split(keyi)
            v_new, h_new = self.gibbs_step(v_curr, h_curr, key1, fix_v=fix_v, T=0)
            if fix_v:
                converged_new = jnp.ones(v_curr.shape[0], dtype=jnp.bool_)
            else:
                converged_v = jnp.all(v_new == v_curr, axis=1)
                converged_h = jnp.ones_like(converged_v)
                for i in range(self.L):
                    converged_h = jnp.logical_and(converged_h, jnp.all(h_new[i] == h_curr[i], axis=1))
                converged_new = jnp.logical_and(converged_v, converged_h)
            return v_new, h_new, converged_new, key2

        # initial
        converged_init = jnp.ones(v.shape[0], dtype=jnp.bool_) if fix_v else jnp.zeros(v.shape[0], dtype=jnp.bool_)
        state = (v, h, converged_init, keys[0])
        # run while not converged (bounded iteration to avoid infinite loops)
        state = jax.lax.while_loop(cond_fn, body_fn, state)
        v_final, h_final, _, _ = state
        return v_final, h_final

    def coupling(self, v: jnp.ndarray, h: List[jnp.ndarray], key: jnp.ndarray
, fix_v: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray]]:
        """
        Runs MH steps until convergence, accumulating energy differences as in original.
        Returns (energy, v_final, h_final)
        """
        # initial
        v_curr = v
        h_curr = h
        energy_curr = self.energy(v_curr, h_curr)
        keys = jax.random.split(key, 128)

        def cond_fn(state):
            v_c, h_c, energy_c, converged, _key = state
            return jnp.logical_not(jnp.all(converged))

        def body_fn(state):
            v_c, h_c, energy_c, converged, keyi = state
            # propose on not converged subset (here we operate on full batch in JAX)
            key1, key2 = jax.random.split(keyi)
            # generate proposals
            rand_v = None if fix_v else jax.random.bernoulli(key1, 0.5, shape=v_c.shape).astype(jnp.float32)
            rand_h = [jax.random.bernoulli(jax.random.fold_in(key2, i), 0.5, shape=h_c[i].shape).astype(jnp.float32) for i in range(self.L)]
            rand_u = jax.random.uniform(key2, shape=(v_c.shape[0],))
            v_prop, h_prop = self.mh_step(v_c, h_c, key2, fix_v=fix_v, rand_v=rand_v, rand_h=rand_h, rand_u=rand_u)
            energy_prop = self.energy(v_prop, h_prop)
            # accumulate energy difference
            energy_new = energy_c + (energy_prop - energy_c)
            # check convergence: same strategy
            if fix_v:
                converged_new = jnp.ones(v_c.shape[0], dtype=jnp.bool_)
            else:
                converged_v = jnp.all(v_prop == v_c, axis=1)
                converged_h = jnp.ones_like(converged_v)
                for i in range(self.L):
                    converged_h = jnp.logical_and(converged_h, jnp.all(h_prop[i] == h_c[i], axis=1))
                converged_new = jnp.logical_and(converged_v, converged_h)
            return v_prop, h_prop, energy_new, converged_new, key2

        converged_init = jnp.ones(v.shape[0], dtype=jnp.bool_) if fix_v else jnp.zeros(v.shape[0], dtype=jnp.bool_)
        state = (v_curr, h_curr, energy_curr, converged_init, keys[0])
        state = jax.lax.while_loop(cond_fn, body_fn, state)
        v_final, h_final, energy_final, _, _ = state
        return energy_final, v_final, h_final

    # -------------------------
    # Forward: returns loss = energy_pos - energy_neg
    # -------------------------
    def forward(self, x: jnp.ndarray, key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray
]:
        """
        x: input floats shape (N, nc, size, size) in [0,1]
        returns: loss (N,) and new key
        """
        key, subkey = jax.random.split(key)
        N = x.shape[0]
        # to bits and flatten
        v = float2bit(x, self.bits).reshape(N, -1).astype(jnp.float32)

        if self.L == 1:
            if self.marginal:
                # Not implemented marginal_energy - fallback
                raise NotImplementedError("marginal energy mode not implemented in this conversion")
            else:
                key1, subkey = jax.random.split(subkey)
                v_pos, h_pos = self.gibbs_step(v, [jnp.zeros((N, self.nh))], key1, fix_v=True, T=1.0)
                energy_pos = self.energy(v_pos, h_pos)
        else:
            # initialize random hidden states
            keys = jax.random.split(subkey, self.L + 1)
            h = [jax.random.bernoulli(keys[i], 0.5, (N, self.nh)).astype(jnp.float32) for i in range(self.L)]
            # local search (positive)
            key_ls, key_gib = jax.random.split(keys[-1])
            v_pos, h_pos = self.local_search(v, h, key_ls, fix_v=True)
            v_pos, h_pos = self.gibbs_step(v_pos, h_pos, key_gib, fix_v=True)
            # coupling
            key_coup = jax.random.split(key_gib, 1)[0]
            energy_pos, v_pos, h_pos = self.coupling(v_pos, h_pos, key_coup, fix_v=True)

        # Negative phase
        key_neg, key_sub = jax.random.split(key)
        v_init = jax.random.bernoulli(key_neg, p=0.5, shape=v.shape).astype(jnp.float32)
        keys_neg = jax.random.split(key_sub, self.L + 1)
        h_init = [jax.random.bernoulli(keys_neg[i], p=0.5, shape=(N, self.nh)).astype(jnp.float32) for i in range(self.L)]
        v_neg, h_neg = self.local_search(v_init, h_init, keys_neg[-1], fix_v=False)
        key_gib_neg = jax.random.split(keys_neg[-1], 1)[0]
        v_neg, h_neg = self.gibbs_step(v_neg, h_neg, key_gib_neg)
        key_coup_neg = jax.random.split(key_gib_neg, 1)[0]
        energy_neg, v_neg, h_neg = self.coupling(v_neg, h_neg, key_coup_neg, fix_v=False)

        loss = energy_pos - energy_neg
        return loss, key

    # -------------------------
    # sample & reconstruct
    # -------------------------
    def sample(self, N: int, key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
        key, subkey = jax.random.split(key)
        v = jax.random.bernoulli(subkey, 0.5, (N, self.nv)).astype(jnp.float32)
        keys = jax.random.split(key, self.L + 1)
        h = [jax.random.bernoulli(keys[i], 0.5, (N, self.nh)).astype(jnp.float32) for i in range(self.L)]
        key_ls = keys[-1]
        v_mode, h_mode = self.local_search(v, h, key_ls)
        key_gib = jax.random.split(key_ls, 1)[0]
        v_rand, h_rand = self.gibbs_step(v_mode, h_mode, key_gib)

        # reshape back to (N, nc, size, size, bits)
        v_mode_img = v_mode.reshape(N, self.nc, self.size, self.size, self.bits)
        v_mode_float = bit2float(v_mode_img.astype(jnp.uint8), self.bits)
        v_rand_img = v_rand.reshape(N, self.nc, self.size, self.size, self.bits)
        v_rand_float = bit2float(v_rand_img.astype(jnp.uint8), self.bits)
        return v_mode_float, v_rand_float, key

    def reconstruct(self, x: jnp.ndarray, key: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
        """
        Given input x (N, nc, size, size) -> reconstruct deterministic (T=0) and random versions.
        """
        key, subkey = jax.random.split(key)
        N = x.shape[0]
        v = float2bit(x, self.bits).reshape(N, -1).astype(jnp.float32)
        keys = jax.random.split(subkey, self.L + 1)
        h = [jax.random.bernoulli(keys[i], p=0.5, shape=(N, self.nh)).astype(jnp.float32) for i in range(self.L)]
        # local search with v fixed True
        v_ls, h_ls = self.local_search(v, h, keys[-1], fix_v=True)
        key_mode = jax.random.split(keys[-1], 1)[0]
        v_mode, h_mode = self.gibbs_step(v_ls, h_ls, key_mode, T=0)
        key_rand = jax.random.split(key_mode, 1)[0]
        v_rand, h_rand = self.gibbs_step(v_ls, h_ls, key_rand, T=1.0)

        v_mode_img = v_mode.reshape(N, self.nc, self.size, self.size, self.bits)
        v_mode_float = bit2float(v_mode_img.astype(jnp.uint8), self.bits)
        v_rand_img = v_rand.reshape(N, self.nc, self.size, self.size, self.bits)
        v_rand_float = bit2float(v_rand_img.astype(jnp.uint8), self.bits)
        return v_mode_float, v_rand_float, key


# -------------------------
# Training helper
# -------------------------
@functools.partial(jax.jit, static_argnames=("model",))
def compute_loss_and_grads(model: DBM, x: jnp.ndarray, key: jnp.ndarray
):
    def loss_fn(params_model, x_in, k):
        # `params_model` is a pytree (model) for grad calculation
        loss, _ = params_model.forward(x_in, k)
        # return scalar loss (mean)
        return jnp.mean(loss)
    loss_value, grads = jax.value_and_grad(loss_fn)(model, x, key)
    return loss_value, grads


def train_step(
    model: DBM,
    opt_state,
    optimizer: optax.GradientTransformation,
    x: jnp.ndarray,
    key: jnp.ndarray

) -> Tuple[DBM, optax.OptState, float, jnp.ndarray
]:
    """
    Single train step: compute grads, apply optimizer, return updated model & opt state.
    """
    loss_value, grads = compute_loss_and_grads(model, x, key)
    updates, opt_state = optimizer.update(grads, opt_state, params=model)
    model = eqx.apply_updates(model, updates)
    # advance RNG
    key = jax.random.split(key, 1)[0]
    return model, opt_state, float(loss_value), key


# -------------------------
# Example usage snippet
# -------------------------
if __name__ == "__main__":
    # quick smoke test (tiny sizes)
    rng = jax.random.PRNGKey(42)
    rng, k1 = jax.random.split(rng)
    size = 4
    nc = 1
    bits = 2
    L = 2
    nv = nc * bits * size * size

    model = DBM(size=size, nc=nc, nh=nv, bits=bits, L=L, key=k1)

    # create tiny batch of images in [0,1]
    rng, k2 = jax.random.split(rng)
    batch = jax.random.uniform(k2, (2, nc, size, size))
    rng, k3 = jax.random.split(rng)
    loss, _ = model.forward(batch, k3)
    print("loss:", loss)

    # optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(model)

    # do one train step
    rng, k4 = jax.random.split(rng)
    model, opt_state, loss_val, rng = train_step(model, opt_state, optimizer, batch, k4)
    print("after step loss:", loss_val)
