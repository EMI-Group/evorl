from .comm import (
    get_global_ranks,
    get_process_id,
    is_dist_initialized,
    pmax,
    pmean,
    pmin,
    psum,
    all_gather,
)
from .gradients import (
    gradient_update,
    pmean_nnx_loss_and_grad,
)
from .sharding import shmap_vmap, shmap_map, tree_scatter_set

DP_AXIS_NAME = "DP"

POP_AXIS_NAME = "POP"

# Population of independent workflows, such as the outer PBT population.
META_POP_AXIS_NAME = "META_POP"


__all__ = [
    "get_global_ranks",
    "get_process_id",
    "is_dist_initialized",
    "pmax",
    "pmean",
    "pmin",
    "psum",
    "all_gather",
    "gradient_update",
    "pmean_nnx_loss_and_grad",
    "shmap_vmap",
    "shmap_map",
    "tree_scatter_set",
    "DP_AXIS_NAME",
    "POP_AXIS_NAME",
    "META_POP_AXIS_NAME",
]
