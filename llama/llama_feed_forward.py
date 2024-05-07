import equinox as eqx
import jax
from jaxtyping import Array, Float32, PRNGKeyArray, jaxtyped
import jax.random as jr

from .llama_config import LLaMAConfig
from .normalization import RMSLayerNorm


class FeedForwardModule(eqx.Module):
    norm: RMSLayerNorm
    linear_in_1: eqx.nn.Linear
    linear_in_2: eqx.nn.Linear
    linear_out: eqx.nn.Linear

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.norm = RMSLayerNorm(config.dim)

        Lin = lambda d, o, k : eqx.nn.Linear(d, o, use_bias=False, key=k)
        k0, k1, k2 = jr.split(key, 3)

        self.linear_in_1 = Lin(config.dim, config.size_hidden, k0)
        self.linear_in_2 = Lin(config.dim, config.size_hidden, k1)
        self.linear_out  = Lin(config.size_hidden, config.dim, k2)

    def __call__(
        self,
        xs: Float32[Array, " seq_len size_layer"],
        enable_dropout: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> Float32[Array, " seq_len size_layer"]:
        xs_normalized = jax.vmap(self.norm)(xs)
        hidden_1 = jax.vmap(self.linear_in_1)(xs_normalized)
        hidden_2 = jax.vmap(self.linear_in_2)(xs_normalized)
        hidden_after_swiglu = jax.nn.silu(hidden_1) * hidden_2
        return jax.vmap(self.linear_out)(hidden_after_swiglu)
