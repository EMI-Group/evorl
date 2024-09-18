import jax.tree_util as jtu


def is_layer_norm_layer(path: tuple[jtu.DictKey]):
    for p in path:
        if "LayerNorm" in p.key:
            return True

    return False
