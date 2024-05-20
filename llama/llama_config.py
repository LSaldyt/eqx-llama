from typing import NamedTuple

from numpy import dtype

import jax.numpy as jnp

class LLaMAConfig(NamedTuple):
    num_layers: int
    size_vocab: int
    dim: int
    n_heads: int
    n_kv_heads: int
    size_hidden: int
    # TODO
    max_batch_size : int
    max_sequence_length : int
    rope_theta : int
    dtype : dtype

def llama_config_from_dict(d):
    return LLaMAConfig(
        d['n_layers'],
        d['vocab_size'],
        d['dim'],
        d['n_heads'],
        d['n_kv_heads'],
        d['size_hidden'],
        d['max_batch_size'],
        d['max_sequence_length'],
        d['rope_theta'],
        getattr(jnp, d['dtype']))
