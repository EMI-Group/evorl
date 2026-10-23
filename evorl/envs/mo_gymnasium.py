"""MO-Gymnasium environments through the ordinary ordered host callback adapter."""

from functools import partial
import warnings

import gymnasium
import mo_gymnasium
from mo_gymnasium.wrappers.vector import MOAsyncVectorEnv

from evorl.types import Action

from .env import BaseEnv
from .gymnasium import GymnasiumAdapter, OneEpisodeWrapper
from .space import Space
from .wrappers import AutoresetMode


class MOGymnasiumAdapter(GymnasiumAdapter):
    """Preserve each objective as an array coordinate, including a single objective.

    The host environment is stateful and steps through `io_callback(ordered=True)`.
    Nested `vmap` merges population and environment batches into one native pool,
    as in `GymnasiumAdapter`. Native environment states cannot be rewound.
    """

    def _create_vector_env(self, num_envs):
        create_env = partial(
            mo_gymnasium.make,
            gymnasium.spec(self.env_name),
            max_episode_steps=self.max_episode_steps,
            **self.env_specs,
        )
        return MOAsyncVectorEnv([create_env] * num_envs, **(self.vecenv_kwargs or {}))


def create_mo_gymnasium_env(
    env_name: str,
    episode_length: int = 1000,
    parallel: int = 1,
    autoreset_mode: AutoresetMode = AutoresetMode.ENVPOOL,
    discount: float | None = 1.0,
    record_ori_obs: bool = False,
    **kwargs,
) -> BaseEnv[Action, Space]:
    """Create vector-reward environments with EvoRL's host autoreset behavior.

    Args:
        env_name: Registered MO-Gymnasium task name.
        episode_length: Maximum number of steps per episode.
        parallel: Number of environments in each unmapped batch.
        autoreset_mode: Same-step, next-step, or one-episode behavior. FAST
            falls back to same-step autoreset, matching Gymnasium.
        discount: Episode-return discount, or `None` to disable return recording.
        record_ori_obs: Preserve terminal observations in `info.ori_obs`.
        **kwargs: Native task arguments and optional `vecenv_kwargs`.

    Returns:
        Environment with rewards of shape `(parallel, num_objectives)` and
        done signals of shape `(parallel,)`. Objective axes are never squeezed.
    """
    match autoreset_mode:
        case AutoresetMode.FAST:
            warnings.warn(
                f"{autoreset_mode} is not supported for MO-Gymnasium Envs. Fallback to AutoresetMode.NORMAL.",
            )
            gymnasium_autoreset_mode = gymnasium.vector.AutoresetMode.SAME_STEP
        case AutoresetMode.NORMAL:
            gymnasium_autoreset_mode = gymnasium.vector.AutoresetMode.SAME_STEP
        case AutoresetMode.ENVPOOL:
            gymnasium_autoreset_mode = gymnasium.vector.AutoresetMode.NEXT_STEP
            if record_ori_obs:
                warnings.warn(
                    f"{autoreset_mode} does not need record_ori_obs. Fallback to False.",
                )
                record_ori_obs = False
        case AutoresetMode.DISABLED:
            gymnasium_autoreset_mode = gymnasium.vector.AutoresetMode.NEXT_STEP
            discount = None

    vecenv_kwargs = dict(
        autoreset_mode=gymnasium_autoreset_mode,
        context="spawn",
    )
    if "vecenv_kwargs" in kwargs:
        vecenv_kwargs.update(kwargs.pop("vecenv_kwargs"))

    env = MOGymnasiumAdapter(
        env_name=env_name,
        max_episode_steps=episode_length,
        num_envs=parallel,
        record_ori_obs=record_ori_obs,
        discount=discount,
        vecenv_kwargs=vecenv_kwargs,
        **kwargs,
    )

    if autoreset_mode == AutoresetMode.DISABLED:
        env = OneEpisodeWrapper(env)

    return env
