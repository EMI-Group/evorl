import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import Action

from ..env import Env, EnvState
from ..space import Box, Space
from .wrapper import Wrapper


class ActionSquashWrapper(Wrapper):
    """Convert continuous action space from [-1, 1] to [low, high]."""

    def __init__(self, env: Env):
        super().__init__(env)

        # TODO: support pytree action space
        action_space = self.env.action_space
        assert isinstance(action_space, Box), "Only support Box action_space"

        self.scale = (action_space.high - action_space.low) * 0.5
        self.bias = (action_space.high + action_space.low) * 0.5

    def step(self, state: EnvState, action: Action) -> EnvState:
        squashed_action = self.scale * action + self.bias
        return self.env.step(state, squashed_action)

    @property
    def action_space(self) -> Space:
        return Box(low=-jnp.ones_like(self.scale), high=jnp.ones_like(self.scale))


class ActionRepeatWrapper(Wrapper):
    """Repeat action for a number of steps.

    :::{note}
    This wrapper only accumulates `state.reward` and `state.info.ori_reward`. It is safe to use `ActionRepeatWrapper(RewardScaleWrapper(EpisodeWrapper(env)))`. However, if you want accumulate other metrics, inherit this class and add your own logic.
    :::
    :::{caution}
    When using rollout functions like `rollout`, `eval_rollout_episode` with `rollout_length` argument, users should use `math.ceil(env.max_episode_steps/action_repeat)` to match the real rollout_length.
    :::
    """

    def __init__(self, env: Env, action_repeat: int):
        super().__init__(env)

        self.action_repeat = action_repeat

    def step(self, state: EnvState, action: Action) -> EnvState:
        def f(state, _):
            nstate = self.env.step(state, action)

            def where_done(x, y):
                done = state.done  # prev_done
                if done.ndim > 0:
                    done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
                return jnp.where(done, x, y)

            # when prev_done=True, keep the previous state and set reward=0
            nstate = jtu.tree_map(where_done, state, nstate)

            reward = nstate.reward
            reward = jtu.tree_map(where_done, jnp.zeros_like(reward), reward)

            if "ori_reward" in nstate.info:
                ori_reward = nstate.info.ori_reward
                ori_reward = jtu.tree_map(
                    where_done, jnp.zeros_like(ori_reward), ori_reward
                )
            else:
                ori_reward = None

            return nstate, (reward, ori_reward)

        state, (rewards, ori_rewards) = jax.lax.scan(
            f, state, (), length=self.action_repeat
        )

        state = state.replace(
            reward=jtu.tree_map(jnp.sum, rewards),
        )

        if ori_rewards is not None:
            state = state.replace(
                info=state.info.replace(
                    ori_reward=jtu.tree_map(jnp.sum, ori_rewards),
                ),
            )

        return state
