import jax
import chex
from omegaconf import DictConfig
from typing_extensions import Self  # pytype: disable=not-supported-yet

from evorl.distributed import split_key_to_devices, unpmap
from evorl.workflows import RLWorkflow
from evorl.agent import RandomAgent, Agent
from evorl.metrics import MetricBase, EvaluateMetric
from evorl.types import State
from evorl.envs import create_env, AutoresetMode
from evorl.evaluators import Evaluator
from evorl.recorders import add_prefix


class RandomTrainMetric(MetricBase):
    pass


class RandomAgentWorkflow(RLWorkflow):
    def __init__(
        self,
        agent: Agent,
        evaluator: Evaluator,
        config: DictConfig,
    ):
        super().__init__(config)

        self.agent = agent
        self.evaluator = evaluator

    @classmethod
    def name(cls):
        return "Random"

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        max_episode_steps = config.env.max_episode_steps

        agent = RandomAgent()

        eval_env = create_env(
            config.env,
            episode_length=max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=max_episode_steps,
        )

        return cls(agent, evaluator, config)

    def setup(self, key: chex.PRNGKey) -> State:
        key, agent_key = jax.random.split(key)

        env = self.evaluator.env

        agent_state = self.agent.init(env.obs_space, env.action_space, agent_key)

        if self.enable_multi_devices:
            agent_state = jax.device_put_replicated(agent_state, self.devices)

            # key and env_state should be different over devices
            key = split_key_to_devices(key, self.devices)

        return State(
            key=key,
            agent_state=agent_state,
        )

    def step(self, state: State) -> tuple[MetricBase, State]:
        """Dummy step function for random agent."""
        return RandomTrainMetric(), state.replace()

    def learn(self, state: State) -> State:
        """Dummy learn function for random agent."""
        eval_metrics, state = self.evaluate(state)
        eval_metrics = unpmap(eval_metrics, self.pmap_axis_name)
        self.recorder.write(add_prefix(eval_metrics.to_local_dict(), "eval"), 0)
        return state.replace()

    def evaluate(self, state: State) -> tuple[MetricBase, State]:
        key, eval_key = jax.random.split(state.key, num=2)

        # [#episodes]
        raw_eval_metrics = self.evaluator.evaluate(
            state.agent_state, eval_key, num_episodes=self.config.eval_episodes
        )

        eval_metrics = EvaluateMetric(
            episode_returns=raw_eval_metrics.episode_returns.mean(),
            episode_lengths=raw_eval_metrics.episode_lengths.mean(),
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        state = state.replace(key=key)
        return eval_metrics, state
