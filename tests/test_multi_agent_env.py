"""Episode and autoreset semantics shared by multi-agent adapters."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evorl.envs import BaseEnv, Box, Discrete, Env, MultiAgentEnv, SpaceContainer
from evorl.envs.env import EnvState
from evorl.envs.wrappers.training_wrapper import (
    EpisodeWrapper,
    FastVmapAutoResetWrapper,
    OneEpisodeWrapper,
    VmapAutoResetWrapper,
    VmapEnvPoolAutoResetWrapper,
    VmapWrapper,
)
from evorl.envs.wrappers.wrapper import Wrapper
from evorl.envs.wrappers import RewardScaleWrapper, get_wrapper
from evorl.types import PyTreeDict
from tests.helpers.envs import ToyEnv


class EarlyFinishEnv(MultiAgentEnv):
    """One agent finishes on step one; the environment finishes on step four."""

    @property
    def agents(self):
        return ["agent_0", "agent_1"]

    @property
    def action_space(self):
        return {agent: Discrete(n=2) for agent in self.agents}

    @property
    def obs_space(self):
        return {
            agent: SpaceContainer(
                spaces=PyTreeDict(value=Box(low=jnp.array(0), high=jnp.array(4)))
            )
            for agent in self.agents
        }

    def reset(self, key):
        flags = PyTreeDict(agent_0=jnp.array(0.0), agent_1=jnp.array(0.0))
        done = flags.replace(__all__=jnp.array(0.0))
        return EnvState(
            env_state=jnp.array(0),
            obs=PyTreeDict(
                {agent: PyTreeDict(value=jnp.array(0)) for agent in self.agents}
            ),
            reward=flags,
            done=done,
            info=PyTreeDict(global_obs=PyTreeDict(value=jnp.array(0))),
            _internal=PyTreeDict(termination=done, truncation=done),
        )

    def step(self, state, action):
        count = state.env_state + 1
        done = PyTreeDict(
            agent_0=(count >= 1).astype(jnp.float32),
            agent_1=(count >= 4).astype(jnp.float32),
            __all__=(count >= 4).astype(jnp.float32),
        )
        return state.replace(
            env_state=count,
            obs=PyTreeDict({agent: PyTreeDict(value=count) for agent in self.agents}),
            reward=PyTreeDict(agent_0=jnp.array(1.0), agent_1=jnp.array(2.0)),
            done=done,
            info=state.info.replace(global_obs=PyTreeDict(value=count)),
            _internal=state._internal.replace(
                termination=done, truncation=jax.tree.map(jnp.zeros_like, done)
            ),
        )


@pytest.mark.parametrize("mode", ["normal", "fast", "envpool", "disabled"])
def test_agent_done_does_not_reset_episode_under_nested_vmap(mode):
    native = EarlyFinishEnv()
    if mode == "disabled":
        env = VmapWrapper(OneEpisodeWrapper(native, 2, discount=0.5), 2, vmap_step=True)
    else:
        cls = {
            "normal": VmapAutoResetWrapper,
            "fast": FastVmapAutoResetWrapper,
            "envpool": VmapEnvPoolAutoResetWrapper,
        }[mode]
        env = cls(EpisodeWrapper(native, 2, discount=0.5), 2)
    reset = jax.jit(jax.vmap(jax.vmap(env.reset)))
    step = jax.jit(jax.vmap(jax.vmap(env.step)))
    state = reset(jax.random.split(jax.random.key(0), (2, 2)))
    action = {agent: jnp.zeros((2, 2, 2), dtype=jnp.int32) for agent in native.agents}
    first = step(state, action)
    np.testing.assert_array_equal(first.done.agent_0, 1)
    np.testing.assert_array_equal(first.done.agent_1, 0)
    np.testing.assert_array_equal(first.done["__all__"], 0)
    np.testing.assert_array_equal(first.obs.agent_1.value, 1)
    np.testing.assert_array_equal(first.info.steps, 1)
    terminal = step(first, action)
    for flag in terminal.done.values():
        np.testing.assert_array_equal(flag, 1)
    np.testing.assert_array_equal(terminal.info.termination.agent_0, 1)
    np.testing.assert_array_equal(terminal.info.truncation.agent_0, 0)
    np.testing.assert_array_equal(terminal.info.termination.agent_1, 0)
    np.testing.assert_array_equal(terminal.info.truncation.agent_1, 1)
    np.testing.assert_array_equal(terminal.info.truncation["__all__"], 1)
    np.testing.assert_allclose(terminal.info.episode_return.agent_0, 1.5)
    np.testing.assert_allclose(terminal.info.episode_return.agent_1, 3.0)
    np.testing.assert_array_equal(terminal.info.ori_obs.agent_1.value, 2)
    np.testing.assert_array_equal(terminal.info.ori_global_obs.value, 2)
    expected_value = 0 if mode in ("normal", "fast") else 2
    np.testing.assert_array_equal(terminal.obs.agent_1.value, expected_value)
    np.testing.assert_array_equal(terminal.info.global_obs.value, expected_value)

    following = step(terminal, action)
    if mode == "disabled":
        assert jax.tree.all(jax.tree.map(jnp.array_equal, terminal, following))
    elif mode == "envpool":
        np.testing.assert_array_equal(following.info.autoreset, 1)
        assert following.info.autoreset.shape == (2, 2, 2)
        np.testing.assert_array_equal(following.info.global_obs.value, 0)
        for reward in following.reward.values():
            np.testing.assert_array_equal(reward, 0)
    else:
        np.testing.assert_array_equal(following.info.steps, 1)
        np.testing.assert_allclose(following.info.episode_return.agent_1, 2.0)


def test_plain_wrapper_passes_complete_state_and_dict_action():
    env = Wrapper(EarlyFinishEnv())
    assert isinstance(env, BaseEnv)
    assert isinstance(env.unwrapped, MultiAgentEnv)
    assert not isinstance(env.unwrapped, Env)
    state = env.reset(jax.random.key(0))
    result = jax.jit(env.step)(state, {agent: jnp.array(0) for agent in env.agents})
    assert result.env_state == 1
    assert result.done["__all__"] == 0


@pytest.mark.parametrize("multi_agent", [False, True])
def test_wrapper_chain_preserves_spaces_attributes_and_transitions(multi_agent):
    native = EarlyFinishEnv() if multi_agent else ToyEnv(2, discrete=True)
    inner = Wrapper(native)
    scaled = RewardScaleWrapper(inner, reward_scale=0.5)
    env = Wrapper(scaled)
    assert isinstance(env, BaseEnv)
    assert env.env is scaled
    assert env.unwrapped is native
    assert get_wrapper(env, RewardScaleWrapper) is scaled
    assert get_wrapper(env, EpisodeWrapper) is None
    if multi_agent:
        assert env.agents == native.agents
        actions = {agent: jnp.array(0) for agent in env.agents}
        for agent in env.agents:
            assert env.action_space[agent].contains(actions[agent])
    else:
        assert env.num_envs == native.num_envs
        actions = jnp.zeros(env.num_envs, dtype=jnp.int32)
        assert env.action_space.n == native.action_space.n

    key = jax.random.key(0)
    initial = jax.jit(env.reset)(key)
    result = jax.jit(env.step)(initial, actions)
    expected = jax.jit(native.step)(jax.jit(native.reset)(key), actions)
    for actual, reference in [
        (result.env_state, expected.env_state),
        (result.obs, expected.obs),
        (result.done, expected.done),
        (result.info.ori_reward, expected.reward),
        (result.reward, jax.tree.map(lambda r: r * 0.5, expected.reward)),
    ]:
        assert jax.tree.all(jax.tree.map(jnp.array_equal, actual, reference))
    if multi_agent:
        for agent in env.agents:
            assert env.obs_space[agent].contains(result.obs[agent])
    else:
        np.testing.assert_array_equal(env.obs_space.low, native.obs_space.low)
        np.testing.assert_array_equal(env.obs_space.high, native.obs_space.high)
        assert jnp.all(jax.vmap(env.obs_space.contains)(initial.obs))
