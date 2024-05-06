from typing import NamedTuple


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
