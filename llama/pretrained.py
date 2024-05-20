import jax.tree_util as jtu
from rich.pretty import pprint
import torch
import jax.numpy as jnp
import equinox   as eqx

def load_llama_pretrained(model, checkpoint_path, n_layers, load_head=False):
    empty_model = jtu.tree_map(lambda x : None, model)
    pprint(model)
    del model
    model = empty_model
    pprint(model)

    checkpoint      = torch.load(checkpoint_path, map_location='cpu')
    print(checkpoint)

    def load(filt_lambda, key, model):
        print(f'Loading {key}..', flush=True)
        value = checkpoint.pop(key)
        value = jnp.array(value.float(), dtype=jnp.bfloat16)
        model = eqx.tree_at(filt_lambda, model, value, is_leaf=lambda x: x is None)
        del value
        return model

    model = load(lambda m : m.embeddings.weight, 'tok_embeddings.weight', model)

    for i in range(n_layers):
        model = load(lambda m : m.layers[i].attention_module.norm.weight, f'layers.{i}.attention_norm.weight',   model)
        model = load(lambda m : m.layers[i].attention_module.linear_q.weight, f'layers.{i}.attention.wq.weight', model)
        model = load(lambda m : m.layers[i].attention_module.linear_k.weight, f'layers.{i}.attention.wk.weight', model)
        model = load(lambda m : m.layers[i].attention_module.linear_v.weight, f'layers.{i}.attention.wv.weight', model)
        model = load(lambda m : m.layers[i].attention_module.linear_o.weight, f'layers.{i}.attention.wo.weight', model)

        model = load(lambda m : m.layers[i].feed_forward_module.norm.weight, f'layers.{i}.ffn_norm.weight',        model)
        model = load(lambda m : m.layers[i].feed_forward_module.linear_in_1.weight, f'layers.{i}.feed_forward.w1.weight', model)
        model = load(lambda m : m.layers[i].feed_forward_module.linear_in_2.weight, f'layers.{i}.feed_forward.w3.weight', model) # Intentionally swapped
        model = load(lambda m : m.layers[i].feed_forward_module.linear_out.weight,  f'layers.{i}.feed_forward.w2.weight', model) # Intentionally swapped

    model = load(lambda m : m.head.norm.weight,   'norm.weight',   model)
    if load_head:
        model = load(lambda m : m.head.linear.weight, 'output.weight', model)

    pprint(model)

    return model
