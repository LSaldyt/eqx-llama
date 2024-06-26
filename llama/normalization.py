import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float16, jaxtyped


class RMSLayerNorm(eqx.Module):
    """Similar to layer normalization, without the mean estimate.

    Known to give similar results to layer norm, with reduced compute.
    """

    weight: Float16[Array, " dim"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.weight = jnp.ones(shape=(dim,), dtype=jnp.bfloat16)
        self.eps = eps

    def __call__(self, x: Float16[Array, " dim"]) -> Float16[Array, " dim"]:
        moment_2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x_normed = x * jax.lax.rsqrt(moment_2 + self.eps)
        return self.weight * x_normed
