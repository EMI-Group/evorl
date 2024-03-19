import jax


import chex
from typing import Sequence, Tuple, Callable, Union, Protocol
from evorl.agents import Agent, AgentState
from evorl.types import EnvState, Episode, SampleBatch
from evorl.envs import Env


def shuffle_sample_batch(sample_batch: SampleBatch, key: chex.PRNGKey):
    return jax.tree_util.tree_map(
        lambda x: jax.random.permutation(key, x),
        sample_batch)


def actor_step(
    env: Env,
    env_state: EnvState,
    agent: Agent,
    agent_state: AgentState,  # readonly
    sample_batch: SampleBatch,
    key: chex.PRNGKey,
    env_extra_fields: Sequence[str] = (),
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect data.
    """

    actions, policy_extras = agent.compute_actions(
        agent_state, sample_batch, key)
    env_nstate = env.step(env_state, actions)

    info = env_nstate.info
    env_extras = {x: info[x] for x in env_extra_fields if x in info}

    transition = SampleBatch(
        obs=env_state.obs,
        action=actions,
        reward=env_nstate.reward,
        done=env_nstate.done,
        next_obs=env_nstate.obs,
        extras=dict(
            policy_extras=policy_extras,
            env_extras=env_extras
        ))

    return env_nstate, transition


def rollout(
    env: Env,
    env_state: EnvState,
    agent: Agent,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
    extra_fields: Sequence[str] = ('last_obs',)
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect given rollout_length trajectory.

        Args:
            env: vampped env w/ autoreset

        Returns:
            env_state: last env_state after rollout
            trajectory: SampleBatch [T, B, ...], T=rollout_length, B=#envs
    """

    def fn(carry, unused_t):
        """
            sample_batch: one-step obs
            transition: one-step full info
        """
        env_state, sample_batch, current_key = carry
        next_key, current_key = jax.random.split(current_key, 2)

        # Note: will XLA optimize repeated calls?
        # transition: [#envs, ...]
        env_nstate, transition = actor_step(
            env, env_state,
            agent, agent_state,
            sample_batch, current_key, extra_fields
        )

        # sample_batch: [#envs, ...]
        sample_batch = SampleBatch(
            obs=env_nstate.obs,
            # extras=transition.extras # for RNN hidden_state?
        )

        return (env_nstate, sample_batch, next_key), transition

    init_sample_batch = SampleBatch(
        obs=env_state.obs,
    )

    # trajectory: [T, #envs, ...]
    (env_state, _, _), trajectory = jax.lax.scan(
        fn, (env_state, init_sample_batch, key), (),
        length=rollout_length,
        # unroll=16 # unroll optimization
    )

    return env_state, trajectory


def rollout_episode(
    env: Env,
    env_state: EnvState,
    agent: Agent,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: Union[int, None] = None,
    extra_fields: Sequence[str] = ()
) -> Tuple[EnvState, Episode]:
    """
        Collect given rollout_length trajectory.
        Args:
            env: vampped env w/o autoreset
    """

    if rollout_length is None:
        rollout_length = env.episode_length

    def fn(carry, unused_t):
        """
            sample_batch: one-step obs
            transition: one-step full info
        """
        env_state, sample_batch, current_key = carry

        next_key, current_key = jax.random.split(current_key, 2)

        # Note: will XLA optimize repeated calls?
        # transition: [#envs, ...]
        env_nstate, transition = actor_step(
            env, env_state,
            agent, agent_state,
            sample_batch, current_key, extra_fields
        )

        # sample_batch: [#envs, ...]
        sample_batch = SampleBatch(
            obs=env_nstate.obs,
            # extras=transition.extras # RNN-state?
        )

        return (env_nstate, sample_batch, next_key), transition

    init_sample_batch = SampleBatch(
        obs=env_state.obs,
    )

    # trajectory: [rollout_length, #envs, ...]
    (env_state, _, _), trajectory = jax.lax.scan(
        fn, (env_state, init_sample_batch, key), (), length=rollout_length, unroll=16)

    episodes = Episode(trajectory=trajectory, last_obs=env_state.obs)

    return env_state, episodes


def rollout_episode_mod(
    env,
    env_state: EnvState,
    agent: Agent,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: Union[int, None] = None,
    extra_fields: Sequence[str] = ()
) -> Tuple[EnvState, Episode]:
    """
        Collect given rollout_length trajectory.
        Avoid unnecessary action_step()
        Args:
            env: vampped env w/o autoreset
    """

    if rollout_length is None:
        rollout_length = env.episode_length

    def fn(carry, unused_t):
        """
            sample_batch: one-step obs
            transition: one-step full info
        """
        env_state, sample_batch, current_key, last_transition = carry
        next_key, current_key = jax.random.split(current_key, 2)

        def dummy_actor_step(*args, **kwargs) -> Tuple[EnvState, SampleBatch]:
            # env_state.done.all() == True
            return env_state.replace(), last_transition.replace()

        env_nstate, transition = jax.lax.cond(
            env_state.done.all(),
            dummy_actor_step,
            actor_step,
            env, env_state, agent, agent_state, sample_batch, current_key, extra_fields
        )

        # sample_batch for get actions
        sample_batch = SampleBatch(
            obs=transition.next_obs,
            # extras=transition.extras
        )

        return (env_nstate, sample_batch, next_key, transition), transition

    (env_state, _, _), trajectory = jax.lax.scan(
        fn, (env_state, key), (), length=rollout_length, unroll=16)

    # valid_mask is still ensured
    episodes = Episode(trajectory=trajectory, last_obs=env_state.obs)

    return env_state, episodes


def rollout_episode_nojit(
    env,
    env_state: EnvState,
    agent: Agent,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: Union[int, None] = None,
    extra_fields: Sequence[str] = ()
) -> Tuple[EnvState, Episode]:
    """
        Collect given rollout_length trajectory.
        Args:
            env: vampped env w/o autoreset
    """

    if rollout_length is None:
        rollout_length = env.episode_length

    trainsitions = []

    while ~ env_state.done.all():
        next_key, current_key = jax.random.split(current_key, 2)
        # sample_batch for get actions

        sample_batch = SampleBatch(
            obs=env_state.obs
        )

        env_state, transition = actor_step(
            env, env_state,
            agent, agent_state,
            sample_batch, current_key, extra_fields
        )

        current_key = next_key

    trajectory = transition[0]

    if len(trainsitions) > 1:
        trajectory = trajectory.concatenate(*trainsitions[1:], axis=0)

    episodes = Episode(trajectory=trajectory, last_obs=env_state.obs)

    return env_state, episodes
