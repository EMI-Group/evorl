import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import PyTreeDict

from ..td3 import TD3TrainMetric


DUMMY_TD3_TRAINMETRIC = TD3TrainMetric(
    critic_loss=jnp.zeros(()),
    actor_loss=jnp.zeros(()),
    raw_loss_dict=PyTreeDict(
        critic_loss=jnp.zeros(()),
        q_value=jnp.zeros(()),
        actor_loss=jnp.zeros(()),
    ),
)


def create_dummy_td3_trainmetric(num: int) -> TD3TrainMetric:
    if num == 1:
        return DUMMY_TD3_TRAINMETRIC
    elif num > 1:
        return DUMMY_TD3_TRAINMETRIC.replace(
            raw_loss_dict=jtu.tree_map(
                lambda x: jnp.broadcast_to(x, (num, *x.shape)),
                DUMMY_TD3_TRAINMETRIC.raw_loss_dict,
            )
        )
    else:
        raise ValueError(f"num should be positive, got {num}")
