from .comm import (
    get_global_ranks,
    get_process_id,
    is_dist_initialized,
    pmax,
    pmean,
    pmin,
    psum,
    split_key_to_devices,
    tree_pmean,
    tree_unpmap,
    unpmap,
)
from .gradients import agent_gradient_update, gradient_update
from .sharding import parallel_map, tree_device_get, tree_device_put

PMAP_AXIS_NAME = "P"

POP_AXIS_NAME = "POP"
