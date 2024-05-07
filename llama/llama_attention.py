import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy     as np
from typing import Tuple 
from jaxtyping import Array, Float32, PRNGKeyArray

from .llama_config import LLaMAConfig
from .normalization import RMSLayerNorm

def compute_attention_scores(
    q: Float32[Array, " head_dim"],
    ks: Float32[Array, " seqlen head_dim"],
    mask: Float32[Array, " seqlen"],
) -> Float32[Array, " seqlen"]:
    head_dim = q.shape[0]
    unnormalized_scores = jnp.inner(q, ks) / jnp.sqrt(head_dim) + mask
    return jax.nn.softmax(unnormalized_scores)

def compute_self_attention(
    qs: Float32[Array, " seqlen head_dim"],
    ks: Float32[Array, " seqlen head_dim"],
    vs: Float32[Array, " seqlen head_dim"],
) -> Float32[Array, " seqlen head_dim"]:
    """Computes the full self-attention.

    Uses vmap rather than einsums or messy transpositions / reshapes...
    """
    mask = jnp.triu(
        jnp.full((qs.shape[0], qs.shape[0]), float("-inf")), k=1
    )  # Uses -inf for numerical stability.
    scores = jax.vmap(compute_attention_scores, in_axes=(0, None, 0))(qs, ks, mask)
    return scores @ vs

def precompute_freqs_cis(dim: int, end: int, theta: float=50000.0, dtype: jnp.dtype=jnp.float32) -> jnp.ndarray:
    freqs     = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t         = np.arange(end)  # type: ignore
    freqs     = np.outer(t, freqs).astype(dtype)  # type: ignore
    sin, cos  = np.sin(freqs), np.cos(freqs)
    freqs_cis = np.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)

def apply_rotary_emb(
    xq: jnp.ndarray, 
    xk: jnp.ndarray, 
    freqs_cis: jnp.ndarray, 
    dtype: jnp.dtype=jnp.float32, 
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # print('applying rotary embeddings')
    # Note that there will be no batch dim from our point of view (within vmap)
    # print('in')
    # print(xq.shape)
    # print(xk.shape)
    
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
    # print('reshape')
    # print(reshape_xq)

    # print('reshape')
    # print(reshape_xq.shape)
    # print(reshape_xk.shape)
    
    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    # print('complex')
    # print(xq_)
    # print(xq_.shape)

    # print('complex')
    # print(xq_.shape)
    # print(xk_.shape)

    # print('freqs_cis')
    # print(freqs_cis.shape)

    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:1], 1, *freqs_cis.shape[1:]))
    # print('freqs cis head dim')
    # print(freqs_cis)
    # print(freqs_cis.shape)
    # print('freqs_cis reshape')
    # print(freqs_cis.shape)
    
    xq_out = xq_ * freqs_cis
    # print('xq_out')
    # print(xq_out)
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)

    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)

    return xq_out.astype(dtype), xk_out.astype(dtype)

def repeat_kv(
    hidden_states: jnp.ndarray,
    n_rep: int,
) -> jnp.ndarray:
    slen, num_key_value_heads, head_dim = hidden_states.shape # No batch dim
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :] # No batch dim
    hidden_states = jnp.repeat(hidden_states, n_rep, axis=2) # Correct for batch dim missing
    return hidden_states.reshape(slen, num_key_value_heads * n_rep, head_dim) # No batch dim

class AttentionModule(eqx.Module):
    norm: RMSLayerNorm
    linear_q: eqx.nn.Linear
    linear_k: eqx.nn.Linear
    linear_v: eqx.nn.Linear
    linear_o: eqx.nn.Linear

    n_rep  :    int = eqx.field(static=True)
    n_heads:    int = eqx.field(static=True)
    n_kv_heads: int = eqx.field(static=True)
    head_dim:   int = eqx.field(static=True)

    freqs_cis : jnp.ndarray = eqx.field(static=True) # Do not update this!

    def __init__(self, config: LLaMAConfig, *, key: PRNGKeyArray):
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep      = config.n_heads // config.n_kv_heads
        self.head_dim   = config.dim // config.n_heads

        self.norm = RMSLayerNorm(config.dim)

        Lin = lambda d, o, k : eqx.nn.Linear(d, o, use_bias=False, key=k)
        kq, kk, kv, ko = jr.split(key, 4)

        self.linear_q = Lin(config.dim, config.n_heads    * self.head_dim, kq)
        self.linear_k = Lin(config.dim, config.n_kv_heads * self.head_dim, kk)
        self.linear_v = Lin(config.dim, config.n_kv_heads * self.head_dim, kv)
        self.linear_o = Lin(config.dim, config.dim, ko)

        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.max_sequence_length * 2,
                                              theta=config.rope_theta,
                                              dtype=jnp.float32) # TODO Generalize types

        # TODO implement caching/autoregression
        # self.cache_k = jnp.zeros((config.max_batch_size, config.max_sequence_length,
        #                           self.n_kv_heads, self.head_dim))
        # self.cache_v = jnp.zeros((config.max_batch_size, config.max_sequence_length,
        #                           self.n_kv_heads, self.head_dim))

    def _split_heads(self, h, num_heads):
        return h.reshape(h.shape[:1] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:1] + (self.embed_dim,))

    def __call__(self, hidden):
        hidden = jax.vmap(self.norm)(hidden)

        # print(jax.vmap(self.linear_q)(hidden).shape)

        xq = self._split_heads(jax.vmap(self.linear_q)(hidden), self.n_heads)
        xk = self._split_heads(jax.vmap(self.linear_k)(hidden), self.n_kv_heads)
        xv = self._split_heads(jax.vmap(self.linear_v)(hidden), self.n_kv_heads)
        print('xq')
        print(xq)
        print('xk')
        print(xk)
        print('xv')
        print(xv)

        start_pos = 0 # TODO generalize for autoregression
        seq_len   = hidden.shape[0] # There is no batch dimension!
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len] # TODO Check freqs_cis shape

        print('freqs_cis')
        print(freqs_cis)
        print(freqs_cis.shape)

        qs, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, dtype=jnp.float32) # self.dtype) # TODO

        print('qs rotary')
        print(qs)
        print('xk rotary')
        print(xk)

        ks = repeat_kv(xk, self.n_rep)
        vs = repeat_kv(xv, self.n_rep)

        print('keys')
        print(ks)
        print('values')
        print(vs)
        print('queries')
        print(qs)

        # TODO This pattern mimics the official torch release cache implementation
        # TODO If we'd like autoregressive caching, use this
        # self.cache_k = self.cache_k.at[:bs, start_pos : start_pos + seqlen].set(xk)
        # self.cache_v = self.cache_v.at[:bs, start_pos : start_pos + seqlen].set(xv)
        # keys   = self.cache_k[:bs, start_pos + seqlen]
        # values = self.cache_v[:bs, start_pos + seqlen]
        # TODO Consider using bits (e.g. masking) from the flax implementation
        # TODO calculate the attention mask if needed

        attention_out = jax.vmap(compute_self_attention, in_axes=(1, 1, 1), out_axes=1)(
             qs, ks, vs
        )
        print('attention out')
        print(attention_out)
        result = jax.vmap(self.linear_o)(jax.lax.collapse(attention_out, 1, 3))
        print('result')
        print(result)
        return result

