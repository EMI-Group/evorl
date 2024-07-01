import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import evox.algorithms
import evox

from evorl.utils.ec_utils import ParamVectorSpec
from evorl.utils.jax_utils import jit_method
from evorl.workflows import ECWorkflow
from evorl.envs import create_wrapped_brax_env
from evorl.ec import GeneralRLProblem
from evorl.metrics import EvaluateMetric
from evorl.distributed import tree_unpmap
from evorl.evaluator import Evaluator
from evorl.types import State
from ..ec import DeterministicECAgent


class CMAESWorkflow(ECWorkflow):
    @classmethod
    def name(cls):
        return "CMAES"

    def __init__(self, config: DictConfig):
        env = create_wrapped_brax_env(
            config.env.env_name,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset=False,
        )

        agent = DeterministicECAgent(
            action_space=env.action_space,
            obs_space=env.obs_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,  # use linear model
            normalize_obs=False
        )

        problem = GeneralRLProblem(
            agent=agent,
            env=env,
            num_episodes=config.episodes,
            max_episode_steps=config.env.max_episode_steps,
            discount=config.discount,
        )

        # dummy agent_state
        agent_key = jax.random.PRNGKey(config.seed)
        agent_state = agent.init(agent_key)
        param_vec_spec = ParamVectorSpec(agent_state.params.policy_params)

        algorithm = evox.algorithms.CMAES(
            center_init=param_vec_spec.to_vector(
                agent_state.params.policy_params),
            init_stdev=config.init_stdev,
            pop_size=config.pop_size,
        )

        def _candidate_transform(flat_cand):
            cand = param_vec_spec.to_tree(flat_cand)
            params = agent_state.params.replace(policy_params=cand)
            return agent_state.replace(params=params)

        super().__init__(
            config=config,
            algorithm=algorithm,
            problem=problem,
            opt_direction='max',
            candidate_transforms=[jax.vmap(_candidate_transform)],
        )

        self._candidate_transform = _candidate_transform

        eval_env = create_wrapped_brax_env(
            config.env.env_name,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset=False,
        )
        evaluator = Evaluator(
            env=eval_env,
            agent=agent,
            max_episode_steps=config.env.max_episode_steps
        )

    def evaluate(self, state: State) -> Tuple[EvaluateMetric, State]:
        """Evaluate the policy with the mean of CMAES
        """
        key, eval_key = jax.random.split(state.key, num=2)

        flat_pop_mean = state.evox_state.query_state('algorithm').mean
        agent_state = self._candidate_transform(flat_pop_mean)

        # [#episodes]
        raw_eval_metrics = self.evaluator.evaluate(
            agent_state,
            num_episodes=self.config.eval_episodes,
            key=eval_key
        )

        eval_metrics = EvaluateMetric(
            episode_returns=raw_eval_metrics.episode_returns.mean(),
            episode_lengths=raw_eval_metrics.episode_lengths.mean()
        ).all_reduce(self.pmap_axis_name)

        state = state.replace(key=key)

        return eval_metrics, state

    def setup_multiple_device(self, state: State) -> State:
        state = super().setup_multiple_device(state)

        self._evaluate = jax.pmap(self._evaluate,)

        return state

    def learn(self, state: State) -> State:
        start_iteration = tree_unpmap(
            state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, self.config.num_iters):
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            train_metrics = tree_unpmap(
                train_metrics, axis_name=self.pmap_axis_name)
            workflow_metrics = tree_unpmap(
                workflow_metrics, self.pmap_axis_name)

            self.recorder.write(workflow_metrics.to_local_dict(), i)
            self.recorder.write(train_metrics.to_local_dict(), i)

            eval_metrics, state = self.evaluate(state)
            eval_metrics = tree_unpmap(eval_metrics, self.pmap_axis_name)
            self.recorder.write(
                {'eval_pop_mean': eval_metrics.to_local_dict()}, i)
