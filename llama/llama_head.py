import equinox as eqx
import jax
from jaxtyping import Array, Float16, PRNGKeyArray, jaxtyped

from .llama_config import LLaMAConfig
from .normalization import RMSLayerNorm


class LLaMAHead(eqx.Module):
    norm: RMSLayerNorm
    linear: eqx.nn.Linear

    def __init__(
        self,
        config: LLaMAConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.norm = RMSLayerNorm(config.dim)

        key_linear, key = jax.random.split(key)
        self.linear = eqx.nn.Linear(
            config.dim,
            config.size_vocab,
            use_bias=False,
            key=key_linear,
            dtype=config.dtype
        )

    def __call__(
        self,
        x: Float16[Array, " size_layer"],
        enable_dropout: bool = False,
        key: PRNGKeyArray | None = None,
    ) -> Float16[Array, " size_vocab"]:
        x_normalized = self.norm(x)
        return self.linear(x_normalized)
