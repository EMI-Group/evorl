from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import jax
import jax.numpy as jnp
import jumanji
import numpy as np
from jumanji.env import Environment as JumanjiEnv
from jumanji.environments import (
    Cleaner,
    Connector,
    LevelBasedForaging,
    MMST,
    MultiCVRP,
    RobotWarehouse,
    SearchAndRescue,
    TSP,
)
from jumanji.specs import Array, BoundedArray, DiscreteArray, MultiDiscreteArray, Spec

from evorl.types import Action, AgentMapping, PyTreeDict

from .env import BaseEnv, EnvAdapter, EnvState
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


class JumanjiAdapter(EnvAdapter):
    """Adapt Jumanji states and structured observations to EvoRL.

    Observation records become nested `PyTreeDict` objects, preserving array
    shapes, dtypes and action masks. Multi-agent tasks use the separate
    `JumanjiMultiAgentAdapter` with agent-keyed data.

    Base `done` flags indicate native termination, derived from zero native
    discounts. Use `EpisodeWrapper` with the same limit as the native task to
    add time truncation. Native discounts are consumed during conversion and
    are not stored in `EnvState`.
    """

    def reset(self, key: jax.Array) -> EnvState:
        return _reset(self.env, key)

    def step(self, state: EnvState, action: Action) -> EnvState:
        return _step(self.env, state, action)

    @property
    def action_space(self) -> Space:
        return _action_spec_to_space(self.env.action_spec)

    @property
    def obs_space(self) -> Space:
        space = jumanji_specs_to_evorl_space(self.env.observation_spec)
        if isinstance(self.env, TSP):
            # Native reset uses -1 for "no city selected", outside its Discrete spec.
            dtype = self.env.observation_spec.position.dtype
            space.spaces.position = Box(
                low=jnp.asarray(-1, dtype=dtype),
                high=jnp.asarray(self.env.num_cities - 1, dtype=dtype),
            )
        return space


# Native task constructors that accept an episode time limit.
_TIME_LIMIT_ENVS = {
    "Cleaner",
    "Connector",
    "LevelBasedForaging",
    "Maze",
    "MMST",
    "PacMan",
    "RobotWarehouse",
    "RubiksCube",
    "SearchAndRescue",
    "SlidingTilePuzzle",
    "Snake",
    "Sokoban",
    "Tetris",
}


# Only these fields describe individual agents' observations. Shared fields,
# including Cleaner/MMST positions, stay visible to all agents. Array size alone
# cannot distinguish agent observations from global data.
_AGENT_OBSERVATION_FIELDS = {
    Cleaner: ("action_mask",),
    Connector: ("action_mask",),
    LevelBasedForaging: ("agents_view", "action_mask"),
    MMST: ("action_mask",),
    MultiCVRP: ("vehicles", "action_mask"),
    RobotWarehouse: ("agents_view", "action_mask"),
    SearchAndRescue: ("searcher_views", "positions"),
}


class JumanjiMultiAgentAdapter(MultiAgentEnvAdapter):
    """Expose Jumanji's joint-action tasks through `MultiAgentEnv`.

    Agent-specific observation fields are sliced explicitly; shared fields
    retain their native values. Each observation includes a numeric `agent_id`
    identifying its agent in shared grids and graphs. The unmodified joint
    observation is available as `info.global_obs`.

    Vector rewards and native discounts are indexed by agent. Scalar team
    rewards are copied unchanged to every agent and retained under
    `reward["__all__"]`. Zero native discount marks an agent's termination;
    scalar discounts give all agents the same flag. Base `done["__all__"]`
    indicates that all agents have terminated. Native discounts are consumed
    during conversion and are not stored in `EnvState`.

    Use `EpisodeWrapper` with the native time limit to add truncation for
    unfinished agents and the whole episode. No upstream autoreset or reward
    aggregation wrapper is installed.
    """

    def __init__(self, env: JumanjiEnv):
        super().__init__(env)
        self._agent_observation_fields = _AGENT_OBSERVATION_FIELDS[type(env)]
        self._agents = [f"agent_{i}" for i in range(env.action_spec.shape[0])]
        # Spaces are static metadata. Build integer category counts before JIT
        # tracing so querying action_space inside a policy does not concretize.
        space = _action_spec_to_space(env.action_spec)
        if isinstance(space, MultiDiscrete):
            self._action_space = {
                agent: Discrete(n=int(space.nvec[i]))
                for i, agent in enumerate(self._agents)
            }
        else:
            self._action_space = {
                agent: Box(low=space.low[i], high=space.high[i])
                for i, agent in enumerate(self._agents)
            }

    @property
    def agents(self) -> list[str]:
        return self._agents.copy()

    def reset(self, key: jax.Array) -> EnvState:
        return self._multi_agent_state(_reset(self.env, key))

    def step(self, state: EnvState, action: AgentMapping[Action]) -> EnvState:
        if not isinstance(action, Mapping) or set(action) != set(self._agents):
            raise ValueError("Actions must contain exactly one entry per agent.")
        joint_action = jnp.stack([action[agent] for agent in self._agents])
        return self._multi_agent_state(_step(self.env, state, joint_action))

    def _multi_agent_state(self, state: EnvState) -> EnvState:
        observation = state.obs
        reward = state.reward
        terminated = state.done
        rewards, dones = PyTreeDict(), PyTreeDict()
        for i, agent in enumerate(self._agents):
            rewards[agent] = reward if reward.ndim == 0 else reward[i]
            dones[agent] = terminated if terminated.ndim == 0 else terminated[i]
        if reward.ndim == 0:
            rewards["__all__"] = reward
        # This function runs before vectorization, so only native agent flags
        # are reduced; environment and population axes are added by wrappers.
        dones["__all__"] = jnp.all(terminated).astype(jnp.float32)
        return state.replace(
            obs=PyTreeDict(
                {
                    agent: observation.replace(
                        **{
                            name: jax.tree.map(lambda x, i=i: x[i], observation[name])
                            for name in self._agent_observation_fields
                        },
                        agent_id=jnp.asarray(i, dtype=jnp.int32),
                    )
                    for i, agent in enumerate(self._agents)
                }
            ),
            reward=rewards,
            done=dones,
            info=state.info.replace(global_obs=observation),
        )

    @property
    def action_space(self) -> AgentMapping[Space]:
        return self._action_space.copy()

    @property
    def obs_space(self) -> AgentMapping[Space]:
        spaces = jumanji_specs_to_evorl_space(self.env.observation_spec).spaces
        return {
            agent: SpaceContainer(
                spaces=spaces.replace(
                    **{
                        name: jax.tree.map(lambda x, i=i: x[i], spaces[name])
                        for name in self._agent_observation_fields
                    },
                    agent_id=Box(
                        low=jnp.asarray(0, dtype=jnp.int32),
                        high=jnp.asarray(len(self._agents) - 1, dtype=jnp.int32),
                    ),
                )
            )
            for i, agent in enumerate(self._agents)
        }


def _reset(env: JumanjiEnv, key: jax.Array) -> EnvState:
    """Convert a native reset result shared by both Jumanji adapters."""
    env_state, timestep = env.reset(key)
    return EnvState(
        env_state=env_state,
        obs=_observation_to_dict(timestep.observation),
        reward=jnp.asarray(timestep.reward, dtype=jnp.float32),
        done=jnp.asarray(timestep.discount == 0, dtype=jnp.float32),
        info=PyTreeDict(timestep.extras),
    )


def _step(env: JumanjiEnv, state: EnvState, action: jax.Array) -> EnvState:
    """Convert a native transition shared by both Jumanji adapters."""
    action = jnp.asarray(action, dtype=env.action_spec.dtype)
    env_state, timestep = env.step(state.env_state, action)
    return state.replace(
        env_state=env_state,
        obs=_observation_to_dict(timestep.observation),
        reward=jnp.asarray(timestep.reward, dtype=jnp.float32),
        done=jnp.asarray(timestep.discount == 0, dtype=jnp.float32),
        info=state.info.replace(**timestep.extras),
    )


def _action_spec_to_space(spec: Spec) -> Space:
    """Convert action specs, including zero-based bounded integer actions."""
    if (
        isinstance(spec, BoundedArray)
        and not isinstance(spec, (DiscreteArray, MultiDiscreteArray))
        and jnp.issubdtype(spec.dtype, jnp.integer)
    ):
        if np.any(np.asarray(spec.minimum) != 0):
            raise NotImplementedError("Integer actions must start at zero.")
        return MultiDiscrete(
            nvec=jnp.broadcast_to(spec.maximum + 1, spec.shape).astype(spec.dtype)
        )
    return jumanji_specs_to_evorl_space(spec)


def _observation_to_dict(observation):
    if hasattr(observation, "_asdict"):
        observation = observation._asdict()
    elif is_dataclass(observation):
        observation = {
            field.name: getattr(observation, field.name)
            for field in fields(observation)
        }
    if isinstance(observation, Mapping):
        return PyTreeDict(
            {name: _observation_to_dict(value) for name, value in observation.items()}
        )
    return observation


def jumanji_specs_to_evorl_space(spec: Spec) -> Space:
    """Convert Jumanji array and nested record specs to EvoRL spaces."""
    if isinstance(spec, DiscreteArray):
        return Discrete(n=spec.num_values)
    elif isinstance(spec, MultiDiscreteArray):
        return MultiDiscrete(nvec=jnp.asarray(spec.num_values, dtype=spec.dtype))
    elif isinstance(spec, BoundedArray):
        low = jnp.broadcast_to(spec.minimum, spec.shape)
        high = jnp.broadcast_to(spec.maximum, spec.shape)
        return Box(low=low, high=high)
    elif isinstance(spec, Array):
        if jnp.issubdtype(spec.dtype, jnp.bool_):
            low, high = False, True
        elif jnp.issubdtype(spec.dtype, jnp.integer):
            limits = jnp.iinfo(spec.dtype)
            low, high = limits.min, limits.max
        else:
            low, high = -1e10, 1e10
        return Box(
            low=jnp.full(spec.shape, low, dtype=spec.dtype),
            high=jnp.full(spec.shape, high, dtype=spec.dtype),
        )
    elif isinstance(spec, Spec):
        # Jumanji exposes nested spec fields as attributes, as its own
        # Gymnasium conversion does. Match the dictionary observation tree.
        return SpaceContainer(
            spaces=PyTreeDict(
                {
                    name: jumanji_specs_to_evorl_space(child)
                    for name, child in vars(spec).items()
                    if isinstance(child, Spec)
                }
            )
        )
    else:
        raise NotImplementedError(f"Unsupported space type: {type(spec)}")


def create_jumanji_env(
    env_name: str,
    **kwargs,
) -> BaseEnv:
    """Create an adapted Jumanji environment without episode wrappers.

    Base `done` flags report native termination. Use
    `create_wrapped_jumanji_env` to apply the shared time limit and autoreset,
    including native Connector and LevelBasedForaging time truncations.

    Args:
        env_name: Registered Jumanji environment name.
        **kwargs: Arguments passed to the Jumanji environment constructor.

    Returns:
        An `Env` for single-agent tasks or a `MultiAgentEnv` for joint-action
        tasks, preserving structured observations, action masks and rewards.
        Continuous actions retain their native bounds; SearchAndRescue uses
        `[-1, 1]` natively.

    """
    env = jumanji.make(env_name, **kwargs)
    if isinstance(env, MultiCVRP) and type(env.action_spec) is BoundedArray:
        # Jumanji 1.1.2 reports an inclusive upper bound one past the last node.
        # Match the public action mask so sampling cannot select that node.
        env.action_spec = env.action_spec.replace(
            maximum=env.observation_spec.action_mask.shape[-1] - 1
        )
    if type(env) in _AGENT_OBSERVATION_FIELDS:
        return JumanjiMultiAgentAdapter(env)
    return JumanjiAdapter(env)


def create_wrapped_jumanji_env(
    env_name: str,
    episode_length: int = 1000,
    parallel: int = 1,
    autoreset_mode: AutoresetMode = AutoresetMode.NORMAL,
    discount: float | None = 1.0,
    record_ori_obs: bool = False,
    **kwargs,
) -> BaseEnv:
    """Create a vectorized single-agent or multi-agent Jumanji environment.

    Native tasks supporting `time_limit` receive `episode_length` at
    construction. EvoRL's episode wrapper uses the same limit and preserves
    native termination; tasks without a native limit are externally truncated.
    Native rewards and action masks are preserved. Multi-agent tasks
    use agent-keyed observations, actions, rewards and done flags, with
    `done["__all__"]` controlling episode limits and autoreset.

    Args:
        env_name: Registered Jumanji environment name.
        episode_length: Shared native and EvoRL episode limit.
        parallel: Number of parallel environments.
        autoreset_mode: Episode autoreset behavior.
        discount: Discount for recording episode returns, or `None` to disable.
        record_ori_obs: Whether to record observations before autoreset.
        **kwargs: Arguments passed to the Jumanji environment constructor.
            An explicit `time_limit` must equal `episode_length`.

    Returns:
        Environment with the requested vectorization and autoreset behavior.

    Raises:
        ValueError: A native `time_limit` conflicts with `episode_length`.
    """
    if env_name.partition("-")[0] in _TIME_LIMIT_ENVS:
        if "time_limit" in kwargs and kwargs["time_limit"] != episode_length:
            raise ValueError(
                "Configure the shared Jumanji time limit with episode_length "
                "(max_episode_steps in YAML); time_limit must match it."
            )
        kwargs["time_limit"] = episode_length
    env = create_jumanji_env(env_name, **kwargs)

    if autoreset_mode == AutoresetMode.ENVPOOL:
        record_ori_obs = False

    if autoreset_mode != AutoresetMode.DISABLED:
        env = EpisodeWrapper(
            env, episode_length, record_ori_obs=record_ori_obs, discount=discount
        )
        if autoreset_mode == AutoresetMode.FAST:
            env = FastVmapAutoResetWrapper(env, num_envs=parallel)
        elif autoreset_mode == AutoresetMode.NORMAL:
            env = VmapAutoResetWrapper(env, num_envs=parallel)
        elif autoreset_mode == AutoresetMode.ENVPOOL:
            env = VmapEnvPoolAutoResetWrapper(env, num_envs=parallel)
    else:
        env = OneEpisodeWrapper(env, episode_length, record_ori_obs=record_ori_obs)
        env = VmapWrapper(env, num_envs=parallel, vmap_step=True)

    return env
