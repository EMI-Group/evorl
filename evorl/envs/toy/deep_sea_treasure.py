"""Native JAX Deep Sea Treasure, using Yang et al.'s convex treasure map."""

import jax.numpy as jnp

from ..env import Env, EnvState
from ..space import Box, Discrete
from ..wrappers import (
    EpisodeWrapper,
    OneEpisodeWrapper,
    VmapAutoResetWrapper,
    VmapWrapper,
    FastVmapAutoResetWrapper,
    VmapEnvPoolAutoResetWrapper,
)
from ..wrappers.training_wrapper import AutoresetMode


class DeepSeaTreasure(Env):
    """Trade treasure value against travel time in the classic two-objective grid.

    Actions are up, down, left and right (0..3). Observation coordinates are
    `(row, column)`. Rocks and grid boundaries block movement. Each step returns
    `[treasure_value, -1]`; reaching a treasure terminates the episode. Time
    truncation is handled exclusively by EvoRL's ordinary episode wrappers.

    The map and movement convention match MO-Gymnasium's default convex task:
    https://mo-gymnasium.farama.org/environments/deep-sea-treasure/
    """

    reward_keys = ("treasure", "time")

    def __init__(self):
        self.sea_map = jnp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-10, 8.2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-10, -10, 11.5, 0, 0, 0, 0, 0, 0, 0, 0],
                [-10, -10, -10, 14.0, 15.1, 16.1, 0, 0, 0, 0, 0],
                [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
                [-10, -10, -10, -10, -10, -10, 0, 0, 0, 0, 0],
                [-10, -10, -10, -10, -10, -10, 19.6, 20.3, 0, 0, 0],
                [-10, -10, -10, -10, -10, -10, -10, -10, 0, 0, 0],
                [-10, -10, -10, -10, -10, -10, -10, -10, 22.4, 0, 0],
                [-10, -10, -10, -10, -10, -10, -10, -10, -10, 23.7, 0],
            ],
            jnp.float32,
        )

    @property
    def action_space(self):
        return Discrete(n=4)

    @property
    def obs_space(self):
        return Box(low=jnp.zeros(2, jnp.int32), high=jnp.full((2,), 10, jnp.int32))

    def reset(self, key):
        position = jnp.zeros(2, jnp.int32)
        return EnvState(
            env_state=position,
            obs=position,
            reward=jnp.zeros(2, jnp.float32),
            done=jnp.float32(0),
        )

    def step(self, state, action):
        directions = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], jnp.int32)
        proposed = state.env_state + directions[action]
        clipped = jnp.clip(proposed, 0, 10)
        valid = jnp.all(proposed == clipped) & (
            self.sea_map[clipped[0], clipped[1]] >= 0
        )
        position = jnp.where(valid, proposed, state.env_state)
        treasure = self.sea_map[position[0], position[1]]
        return state.replace(
            env_state=position,
            obs=position,
            reward=jnp.array([treasure, -1], jnp.float32),
            done=(treasure > 0).astype(jnp.float32),
        )


def create_wrapped_deep_sea_treasure(
    env_name,
    episode_length=100,
    parallel=1,
    autoreset_mode=AutoresetMode.NORMAL,
    discount=1.0,
    record_ori_obs=False,
):
    """Create the native benchmark with standard EvoRL episode/autoreset behavior."""
    if env_name != "deep-sea-treasure-v0":
        raise ValueError(f"Unsupported toy environment: {env_name}")
    env = DeepSeaTreasure()
    if autoreset_mode == AutoresetMode.DISABLED:
        return VmapWrapper(
            OneEpisodeWrapper(env, episode_length, record_ori_obs=record_ori_obs),
            parallel,
            vmap_step=True,
        )
    env = EpisodeWrapper(
        env,
        episode_length,
        record_ori_obs=record_ori_obs,
        discount=discount,
    )
    wrappers = {
        AutoresetMode.NORMAL: VmapAutoResetWrapper,
        AutoresetMode.FAST: FastVmapAutoResetWrapper,
        AutoresetMode.ENVPOOL: VmapEnvPoolAutoResetWrapper,
    }
    return wrappers[autoreset_mode](env, parallel)
