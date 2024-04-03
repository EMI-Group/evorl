from .comm import psum, pmean, pmin, pmax, unpmap, tree_unpmap, split_key_to_devices

from .gradients import agent_gradient_update


PMAP_AXIS_NAME = "P"