import jax

from evox import Problem, State
import chex
from typing import Tuple, Union

from evorl.agents import Agent, AgentState
from evorl.envs import create_wrapped_brax_env, Env, EnvState
from evorl.rollout import SampleBatch
from evorl.types import Reward, RewardDict, Callable, Action, PolicyExtraInfo

from functools import partial

#TODO: use dataclass after evox update
class MultiObjectiveBraxProblem(Problem):
    def __init__(
        self,
        agent: Agent,
        env_name: str,
        parallel: int = 1,
        max_episode_steps: int = 1000,
        num_episodes: int = 10,
        **kwargs
    ):
        """
            Args:
                agent: agent model that defined the weights
                env_name: name of the environment
                parallel: number of parallel environments
                max_episode_steps: maximum number of steps in an episode
                num_episodes: number of episodes to evaluate
        """
        self.agent = agent
        self.max_episode_steps = max_episode_steps
        self.env = create_wrapped_brax_env(
            env_name,
            episode_length=max_episode_steps,
            parallel=parallel,
            autoreset=False,
            fast_reset=True,
            **kwargs
        )
        self.agent_eval_actions = jax.vmap(self.agent.evaluate_actions, axis_name='pop')
        # Note: there are two vmap: [#pop, #envs, ...]
        self.env_reset = jax.vmap(self.env.reset, axis_name='pop')
        self.env_step = jax.vmap(self.env.step, axis_name='pop')

    def setup(self, key: chex.PRNGKey):
        return State(
            key=key,
        )
    

    def evaluate(self, state: State, pop_agent_states: chex.ArrayTree) -> Tuple[chex.ArrayTree, State]:
        pass



def eval_env_step(
    env_fn: Callable[[EnvState, Action], EnvState],
    action_fn: Callable[[AgentState, SampleBatch, chex.PRNGKey], Tuple[Action, PolicyExtraInfo]],
    env_state: EnvState,
    agent_state: AgentState,  # readonly
    sample_batch: SampleBatch,
    key: chex.PRNGKey
) -> Tuple[EnvState, SampleBatch]:

    actions, policy_extras = action_fn(agent_state, sample_batch, key)
    env_nstate = env_fn(env_state, actions)

    # info = env_nstate.info
    # env_extras = {x: info[x] for x in env_extra_fields if x in info}

    metrics = env_nstate.info.metrics.copy()
    metrics.reward = env_nstate.reward

    transition = SampleBatch(
        rewards=metrics,
        dones=env_nstate.done,
    )

    return env_nstate, transition



def eval_rollout_episode(
    env: Env,
    agent: Agent,
    env_state: EnvState,
    agent_state: AgentState,
    key: chex.PRNGKey,
    rollout_length: int,
) -> Tuple[EnvState, SampleBatch]:
    """
        Collect given rollout_length trajectory.
        Avoid unnecessary env_step()
        Args:
            env: vmapped env w/o autoreset
    """

    _eval_env_step = partial(eval_env_step, env, agent)

    def _one_step_rollout(carry, unused_t):
        """
            sample_batch: one-step obs
            transition: one-step full info
        """
        env_state, current_key, prev_transition = carry
        next_key, current_key = jax.random.split(current_key, 2)

        sample_batch = SampleBatch(
            obs=env_state.obs,
        )

        env_nstate, transition = jax.lax.cond(
            env_state.done.all(),
            lambda *x: (env_state.replace(), prev_transition.replace()),
            _eval_env_step,
            env_state, agent_state,
            sample_batch, current_key
        )

        return (env_nstate, next_key, transition), transition

    # run one-step rollout first to get bootstrap transition
    # it will not include in the trajectory when env_state is from env.reset()
    # this is manually controlled by user.
    _, transition = _eval_env_step(
        env_state, agent_state,
        SampleBatch(obs=env_state.obs), key
    )

    (env_state, _, _), trajectory = jax.lax.scan(
        _one_step_rollout, (env_state, key, transition),
        (), length=rollout_length
    )

    return env_state, trajectory