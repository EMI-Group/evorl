from enum import Enum

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from evorl.utils.jax_utils import ensure_batched_key, rng_split

from ..env import BaseEnv, EnvState, get_episode_done
from .wrapper import Wrapper


class EpisodeWrapper[ActionT, SpaceT](Wrapper[ActionT, SpaceT]):
    """Maintains episode step count and sets done at episode end.

    This is the same as brax's EpisodeWrapper, and add some new fields in transition.info.
    Including:
    - steps: the current step count of the episode
    - trunction: whether the episode is truncated
    - termination: whether the episode is terminated
    - ori_obs: the next observation without autoreset
    - episode_return: the current sum of dicounted reward of the episode

    Environments that distinguish native termination from truncation can supply
    `termination` and `truncation` in `EnvState._internal`. These describe the
    current transition and must be updated by the environment on every step.
    Otherwise, an environment's `done` is treated as termination.
    """

    def __init__(
        self,
        env: BaseEnv[ActionT, SpaceT],
        episode_length: int,
        record_ori_obs: bool = True,
        discount: float | None = None,
    ):
        """Initializes the env wrapper.

        Args:
            env: the wrapped env should be a single un-vectorized environment.
            episode_length: the maxiumum length of each episode for truncation
            action_repeat: the number of times to repeat each action
            record_ori_obs: whether to record the real next observation of each episode
            discount: the discount factor for computing the return. Default is None, which means do not reacord the episode_return.
        """
        super().__init__(env)
        self.episode_length = episode_length
        self.record_ori_obs = record_ori_obs
        self.record_episode_return = discount is not None
        self.discount = discount

    def reset(self, key: jax.Array) -> EnvState:
        state = self.env.reset(key)

        info = state.info.replace(
            steps=jnp.zeros((), dtype=jnp.int32),
            termination=jtu.tree_map(jnp.zeros_like, state.done),
            truncation=jtu.tree_map(jnp.zeros_like, state.done),
        )

        if self.record_ori_obs:
            info.ori_obs = jtu.tree_map(jnp.zeros_like, state.obs)
            if "global_obs" in state.info:
                info.ori_global_obs = jtu.tree_map(
                    jnp.zeros_like, state.info.global_obs
                )
        if self.record_episode_return:
            info.episode_return = jtu.tree_map(jnp.zeros_like, state.reward)

        return state.replace(info=info)

    def step(self, state: EnvState, action: ActionT) -> EnvState:
        return self._step(state, action)

    def _step(self, state: EnvState, action: ActionT) -> EnvState:
        prev_done = get_episode_done(state.done)
        # reset steps when prev episode is done(truncation or termination)
        steps = state.info.steps * (1 - prev_done).astype(jnp.int32)

        if self.record_episode_return:
            # reset the episode_return when the episode is done
            episode_return = jtu.tree_map(
                lambda value: value * (1 - prev_done), state.info.episode_return
            )

        # ============== pre update ==============
        state = self.env.step(state, action)

        # ============== post update ==============
        steps = steps + 1

        done = jtu.tree_map(
            lambda flag: jnp.where(
                steps >= self.episode_length, jnp.ones_like(flag), flag
            ),
            state.done,
        )
        termination = state._internal.get("termination", state.done)
        truncation = state._internal.get(
            "truncation", jtu.tree_map(jnp.zeros_like, state.done)
        )

        info = state.info.replace(
            steps=steps,
            termination=termination,
            # Note: here we also consider the case:
            # when termination and truncation are both happened
            # at the last step, we set truncation=0
            truncation=jtu.tree_map(
                lambda terminated, truncated: jnp.where(
                    steps >= self.episode_length, 1 - terminated, truncated
                ),
                termination,
                truncation,
            ),
        )

        if self.record_ori_obs:
            # the real next_obs at the end of episodes, where
            # state.obs could be changed to the next episode's inital state
            # by VmapAutoResetWrapper
            info.ori_obs = state.obs  # i.e. obs at t+1
            if "global_obs" in state.info:
                info.ori_global_obs = state.info.global_obs

        if self.record_episode_return:
            if self.discount == 1.0:  # a shortcut for discount=1.0
                episode_return = jtu.tree_map(jnp.add, episode_return, state.reward)
            else:
                episode_return = jtu.tree_map(
                    lambda total, reward: (
                        total + jnp.power(self.discount, steps - 1) * reward
                    ),
                    episode_return,
                    state.reward,
                )
            info.episode_return = episode_return

        return state.replace(done=done, info=info)


class OneEpisodeWrapper[ActionT, SpaceT](EpisodeWrapper[ActionT, SpaceT]):
    """Maintains episode step count and sets done at episode end.

    When call step() after the env is done, stop simulation and
    directly return previous state.
    """

    def step(self, state: EnvState, action: ActionT) -> EnvState:
        new_state = self._step(state, action)
        # Select old state when already done (via jnp.where on leaves).
        # We avoid lax.cond here because under jax.vmap it traces both
        # branches, which breaks warp backend's custom_vmap for mjx.step.
        return jtu.tree_map(
            lambda old, new: jnp.where(get_episode_done(state.done), old, new),
            state,
            new_state,
        )


class VmapWrapper[ActionT, SpaceT](Wrapper[ActionT, SpaceT]):
    """Vectorize env."""

    def __init__(
        self, env: BaseEnv[ActionT, SpaceT], num_envs: int = 1, vmap_step: bool = False
    ):
        """Initialize the env wrapper.

        Args:
            env: the original env
            num_envs: number of envs to vectorize
            vmap_step: whether to vectorize the step function by `vmap`, or use `lax.map`
        """
        super().__init__(env)
        self.num_envs = num_envs
        self.vmap_step = vmap_step

    def reset(self, key: jax.Array) -> EnvState:
        """Reset the vmapped env.

        Args:
            key: scalar typed key or a batch of B typed keys
        """
        key = ensure_batched_key(key, self.num_envs)

        return jax.vmap(self.env.reset)(key)

    def step(self, state: EnvState, action: ActionT) -> EnvState:
        if self.vmap_step:
            return jax.vmap(self.env.step)(state, action)
        else:
            return jax.lax.map(lambda x: self.env.step(*x), (state, action))


class VmapAutoResetWrapper[ActionT, SpaceT](Wrapper[ActionT, SpaceT]):
    """Vectorize env and Autoreset."""

    def __init__(self, env: BaseEnv[ActionT, SpaceT], num_envs: int = 1):
        """Initialize the env wrapper.

        Args:
            env: the original env
            num_envs: number of parallel envs.
        """
        super().__init__(env)
        self.num_envs = num_envs

    def reset(self, key: jax.Array) -> EnvState:
        """Reset the vmapped env.

        Args:
            key: scalar typed key or a batch of B typed keys
        """
        key = ensure_batched_key(key, self.num_envs)

        reset_key, key = rng_split(key)
        state = jax.vmap(self.env.reset)(key)
        state._internal.reset_key = reset_key  # for autoreset

        return state

    def step(self, state: EnvState, action: ActionT) -> EnvState:
        state = jax.vmap(self.env.step)(state, action)

        # Map heterogeneous computation (non-parallelizable).
        # This avoids lax.cond becoming lax.select in vmap
        state = jax.lax.map(self._maybe_reset, state)

        return state

    def _auto_reset(self, state: EnvState) -> EnvState:
        """AutoReset the state of one Env.

        Reset the state and overwrite `timestep.observation` with the reset observation if the episode has terminated.
        """
        # Make sure that the random key in the environment changes at each call to reset.
        new_key, reset_key = jax.random.split(state._internal.reset_key)
        reset_state = self.env.reset(reset_key)

        state = state.replace(
            env_state=reset_state.env_state,
            obs=reset_state.obs,
            _internal=state._internal.replace(reset_key=new_key),
        )
        if "global_obs" in state.info:
            state = state.replace(
                info=state.info.replace(global_obs=reset_state.info.global_obs)
            )

        return state

    def _maybe_reset(self, state: EnvState) -> EnvState:
        """Overwrite the state and timestep appropriately if the episode terminates."""
        return jax.lax.cond(
            get_episode_done(state.done),
            self._auto_reset,
            lambda state: state.replace(),
            state,
        )


class FastVmapAutoResetWrapper[ActionT, SpaceT](Wrapper[ActionT, SpaceT]):
    """Brax-style AutoReset: no randomness in reset.

    This wrapper reuses the state in the return of `env.reset()`. When the episodes have short length or the `env.reset()` is expensive, This wrapper is more efficient than `VmapAutoResetWrapper`.
    """

    def __init__(self, env: BaseEnv[ActionT, SpaceT], num_envs: int = 1):
        """Initialize the env wrapper.

        Args:
            env: the original env
            num_envs: number of parallel envs.
        """
        super().__init__(env)
        self.num_envs = num_envs

    def reset(self, key: jax.Array) -> EnvState:
        """Reset the vmapped env.

        Args:
        key: scalar typed key or a batch of B typed keys
        """
        key = ensure_batched_key(key, self.num_envs)

        state = jax.vmap(self.env.reset)(key)
        state._internal.first_env_state = state.env_state
        state._internal.first_obs = state.obs
        if "global_obs" in state.info:
            state._internal.first_global_obs = state.info.global_obs

        return state

    def step(self, state: EnvState, action: ActionT) -> EnvState:
        state = jax.vmap(self.env.step)(state, action)

        def where_done(x, y):
            done = get_episode_done(state.done)
            if done.ndim > 0:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        env_state = jtu.tree_map(
            where_done, state._internal.first_env_state, state.env_state
        )
        obs = jtu.tree_map(where_done, state._internal.first_obs, state.obs)

        state = state.replace(env_state=env_state, obs=obs)
        if "global_obs" in state.info:
            state = state.replace(
                info=state.info.replace(
                    global_obs=jtu.tree_map(
                        where_done,
                        state._internal.first_global_obs,
                        state.info.global_obs,
                    )
                )
            )
        return state


class VmapEnvPoolAutoResetWrapper[ActionT, SpaceT](Wrapper[ActionT, SpaceT]):
    """EnvPool style AutoReset.

    When the episode ends, an additional reset step is performed.
    See EnvPool: https://envpool.readthedocs.io/en/latest/content/python_interface.html#auto-reset, and the Next-Step Mode in gymnasium: https://farama.org/Vector-Autoreset-Mode.
    This is helpful for algorithms that require n-step TD or GAE with Partial episode bootstrapping (PEB) support on time-limited environments. When using this wrapper, remember to skip the invalid transitions via the mask `autoreset`.
    """

    def __init__(self, env: BaseEnv[ActionT, SpaceT], num_envs: int = 1):
        """Initialize the env wrapper.

        Args:
            env: the original env
            num_envs: number of parallel envs.
        """
        super().__init__(env)
        self.num_envs = num_envs

    def reset(self, key: jax.Array) -> EnvState:
        """Reset the vmapped env.

        Args:
            key: scalar typed key or a batch of B typed keys
        """
        key = ensure_batched_key(key, self.num_envs)

        reset_key, key = rng_split(key)
        state = jax.vmap(self.env.reset)(key)
        state.info.autoreset = jnp.zeros_like(get_episode_done(state.done))
        state._internal.reset_key = reset_key  # for autoreset

        return state

    def step(self, state: EnvState, action: ActionT) -> EnvState:
        autoreset = get_episode_done(state.done)  # i.e. prev_done

        def _where_autoreset(x, y):
            # where prev_done
            if autoreset.ndim > 0:
                cond = jnp.reshape(autoreset, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(cond, x, y)

        reset_state = self.reset(state._internal.reset_key)
        new_state = jax.vmap(self.env.step)(state, action)

        state = jtu.tree_map(
            _where_autoreset,
            reset_state,
            new_state,
        )
        state.info.autoreset = autoreset

        return state


class AutoresetMode(Enum):
    """Autoreset mode."""

    NORMAL = "normal"
    FAST = "fast"
    DISABLED = "disabled"
    ENVPOOL = "envpool"
