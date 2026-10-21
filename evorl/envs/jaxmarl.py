from collections.abc import Mapping

import jax
import jax.numpy as jnp
import jaxmarl
from jaxmarl.environments import Hanabi, MultiAgentEnv as JaxMARLEnv, SMAX
from jaxmarl.environments import spaces as native_spaces
from jaxmarl.environments.smax.heuristic_enemy_smax_env import EnemySMAX

from evorl.types import Action, AgentID, AgentMapping, PyTreeDict

from .env import BaseEnv, EnvState
from .multi_agent_env import MultiAgentEnvAdapter
from .space import Box, Discrete, MultiDiscrete, Space, SpaceContainer
from .wrappers.training_wrapper import (
    AutoresetMode,
    EpisodeWrapper,
    FastVmapAutoResetWrapper,
    OneEpisodeWrapper,
    VmapAutoResetWrapper,
    VmapEnvPoolAutoResetWrapper,
    VmapWrapper,
)


class JaxMARLAdapter(MultiAgentEnvAdapter):
    """Adapt JaxMARL's agent-keyed API without its automatic episode reset.

    Native rewards and agent IDs are preserved. Global observations, when
    provided by the environment, are exposed in `info.global_obs`. SMAX and
    Hanabi observations include `observation` and boolean `action_mask` fields
    so masks follow current and terminal observations through EvoRL autoreset.
    """

    def __init__(self, env: JaxMARLEnv):
        super().__init__(env)
        self._action_mask_fn = None
        if isinstance(env, Hanabi):
            self._action_mask_fn = env.get_legal_moves
        elif isinstance(env, (SMAX, EnemySMAX)) and env.action_type == "discrete":
            self._action_mask_fn = env.get_avail_actions
        self._action_spaces = {
            agent: jaxmarl_space_to_evorl_space(env.action_space(agent))
            for agent in self.agents
        }
        observations, _ = jax.eval_shape(env.reset, jax.random.key(0))
        self._obs_spaces = {}
        for agent in self.agents:
            spec = env.observation_space(agent)
            # FACMAC 6a/9a retain the default 3a shape in their native spaces;
            # OvercookedV2 also reports the first agent's shape for mixed views.
            if isinstance(spec, native_spaces.Box):
                spec = native_spaces.Box(
                    spec.low, spec.high, observations[agent].shape, spec.dtype
                )
            space = jaxmarl_space_to_evorl_space(spec)
            if self._action_mask_fn is not None:
                space = SpaceContainer(
                    spaces=PyTreeDict(
                        observation=space,
                        action_mask=Box(
                            low=jnp.zeros(env.action_space(agent).n, jnp.bool_),
                            high=jnp.ones(env.action_space(agent).n, jnp.bool_),
                        ),
                    )
                )
            self._obs_spaces[agent] = space

    def _observations(self, obs, env_state, info):
        if "world_state" in obs:
            info.global_obs = obs["world_state"]
        elif hasattr(self.env, "get_world_state"):
            info.global_obs = self.env.get_world_state(env_state)
        observations = PyTreeDict({agent: obs[agent] for agent in self.agents})
        if self._action_mask_fn is not None:
            masks = self._action_mask_fn(env_state)
            observations = PyTreeDict(
                {
                    agent: PyTreeDict(
                        observation=observations[agent],
                        action_mask=masks[agent].astype(jnp.bool_),
                    )
                    for agent in self.agents
                }
            )
        return observations

    def reset(self, key: jax.Array) -> EnvState:
        key, reset_key = jax.random.split(key)
        obs, env_state = self.env.reset(reset_key)
        # Trace the transition structure without advancing the simulator. Info
        # fields must have the same pytree structure in reset and step for scans.
        actions = {}
        for agent in self.agents:
            space = self.env.action_space(agent)
            actions[agent] = jnp.zeros(space.shape, dtype=space.dtype)
        _, _, rewards, dones, infos = jax.eval_shape(
            self.env.step_env, key, env_state, actions
        )
        info = PyTreeDict(jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), infos))
        observations = self._observations(obs, env_state, info)
        return EnvState(
            env_state=env_state,
            obs=observations,
            reward=PyTreeDict(
                {
                    name: jnp.zeros(spec.shape, jnp.float32)
                    for name, spec in rewards.items()
                }
            ),
            done=PyTreeDict(
                {
                    agent: jnp.zeros(dones[agent].shape, jnp.float32)
                    for agent in (*self.agents, "__all__")
                }
            ),
            info=info,
            _internal=PyTreeDict(step_key=key),
        )

    def step(self, state: EnvState, action: AgentMapping[Action]) -> EnvState:
        if not isinstance(action, Mapping) or set(action) != set(self.agents):
            raise ValueError("Actions must contain exactly one entry per agent.")
        key, step_key = jax.random.split(state._internal.step_key)
        obs, env_state, reward, done, infos = self.env.step_env(
            step_key, state.env_state, action
        )
        info = state.info.replace(**infos)
        observations = self._observations(obs, env_state, info)
        return state.replace(
            env_state=env_state,
            obs=observations,
            reward=PyTreeDict(jax.tree.map(lambda x: x.astype(jnp.float32), reward)),
            done=PyTreeDict(jax.tree.map(lambda x: x.astype(jnp.float32), done)),
            info=info,
            _internal=state._internal.replace(step_key=key),
        )

    @property
    def action_space(self) -> AgentMapping[Space]:
        return self._action_spaces.copy()

    @property
    def obs_space(self) -> AgentMapping[Space]:
        return self._obs_spaces.copy()

    @property
    def agents(self) -> list[AgentID]:
        return list(self.env.agents)


def jaxmarl_space_to_evorl_space(space: native_spaces.Space) -> Space:
    """Convert JaxMARL spaces, preserving nested structures and bounds."""
    if isinstance(space, native_spaces.Box):
        low = jnp.broadcast_to(jnp.asarray(space.low, dtype=space.dtype), space.shape)
        high = jnp.broadcast_to(jnp.asarray(space.high, dtype=space.dtype), space.shape)
        return Box(
            low=jnp.nan_to_num(low, neginf=-1e10, posinf=1e10),
            high=jnp.nan_to_num(high, neginf=-1e10, posinf=1e10),
        )
    if isinstance(space, native_spaces.Discrete):
        return Discrete(n=space.n)
    if isinstance(space, native_spaces.MultiDiscrete):
        return MultiDiscrete(nvec=space.num_categories)
    if isinstance(space, native_spaces.Dict):
        return SpaceContainer(
            spaces=PyTreeDict(
                {
                    name: jaxmarl_space_to_evorl_space(child)
                    for name, child in space.spaces.items()
                }
            )
        )
    if isinstance(space, native_spaces.Tuple):
        return SpaceContainer(
            spaces=tuple(jaxmarl_space_to_evorl_space(child) for child in space.spaces)
        )
    raise NotImplementedError(f"Unsupported JaxMARL space: {type(space)}")


supported_jaxmarl_env_list = (
    "MPE_simple_v3",
    "MPE_simple_tag_v3",
    "MPE_simple_world_comm_v3",
    "MPE_simple_spread_v3",
    "MPE_simple_crypto_v3",
    "MPE_simple_speaker_listener_v4",
    "MPE_simple_push_v3",
    "MPE_simple_adversary_v3",
    "MPE_simple_reference_v3",
    "MPE_simple_facmac_v1",
    "MPE_simple_facmac_3a_v1",
    "MPE_simple_facmac_6a_v1",
    "MPE_simple_facmac_9a_v1",
    "switch_riddle",
    "SMAX",
    "HeuristicEnemySMAX",
    "LearnedPolicyEnemySMAX",
    "hanabi",
    "overcooked",
    "overcooked_v2",
    "coin_game",
    "jaxnav",
)


def create_jaxmarl_env(env_name: str, **kwargs) -> JaxMARLAdapter:
    """Create a JaxMARL environment with agent-keyed observations and rewards.

    Args:
        env_name: Supported JaxMARL registry name. STORM's heterogeneous native
            APIs, optional JaxRobotarium tasks and deprecated MABrax are not
            included. Prefer `overcooked_v2` over deprecated `overcooked`.
        **kwargs: Arguments for `jaxmarl.make`. `LearnedPolicyEnemySMAX` requires
            the caller to supply its opponent `policy` and `params`.

    Returns:
        An environment following the EvoRL `MultiAgentEnv` interface.
    """
    if env_name not in supported_jaxmarl_env_list:
        raise ValueError(f"Unsupported JaxMARL environment: {env_name}")
    return JaxMARLAdapter(jaxmarl.make(env_name, **kwargs))


def create_wrapped_jaxmarl_env(
    env_name: str,
    episode_length: int = 1000,
    parallel: int = 1,
    autoreset_mode: AutoresetMode = AutoresetMode.NORMAL,
    discount: float | None = 1.0,
    record_ori_obs: bool = False,
    **kwargs,
) -> BaseEnv[AgentMapping[Action], AgentMapping[Space]]:
    """Create a JaxMARL environment with EvoRL episode and autoreset wrappers.

    Args:
        env_name: Supported JaxMARL registry name.
        episode_length: Maximum episode length enforced by EvoRL.
        parallel: Number of parallel worlds, each containing all agents.
        autoreset_mode: EvoRL reset mode; native `step` autoreset is bypassed.
        discount: Discount for per-agent episode returns, or `None` to disable.
        record_ori_obs: Preserve terminal observations for same-step autoreset.
        **kwargs: Native environment constructor arguments for `jaxmarl.make`.

    Returns:
        A vectorized environment preserving per-agent rewards and observations.
    """
    env = create_jaxmarl_env(env_name, **kwargs)
    if autoreset_mode == AutoresetMode.ENVPOOL:
        record_ori_obs = False
    if autoreset_mode == AutoresetMode.DISABLED:
        env = OneEpisodeWrapper(env, episode_length, record_ori_obs=record_ori_obs)
        return VmapWrapper(env, num_envs=parallel, vmap_step=True)
    env = EpisodeWrapper(
        env, episode_length, record_ori_obs=record_ori_obs, discount=discount
    )
    wrapper = {
        AutoresetMode.NORMAL: VmapAutoResetWrapper,
        AutoresetMode.FAST: FastVmapAutoResetWrapper,
        AutoresetMode.ENVPOOL: VmapEnvPoolAutoResetWrapper,
    }[autoreset_mode]
    return wrapper(env, num_envs=parallel)
