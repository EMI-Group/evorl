import jax
import jax.numpy as jnp

from omegaconf import DictConfig
import logging
import evox.algorithms

from evorl.utils.ec_utils import ParamVectorSpec
from evorl.envs import create_wrapped_brax_env
from evorl.ec import GeneralRLProblem
from evorl.metrics import EvaluateMetric
from evorl.distributed import tree_unpmap
from evorl.evaluator import Evaluator
from evorl.types import State
from ..ec import DeterministicECAgent
from .es_base import ESBaseWorkflow

logger = logging.getLogger(__name__)


class OpenESWorkflow(ESBaseWorkflow):
    @classmethod
    def name(cls):
        return "OpenES"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
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

        # TODO: impl complete version of OpenES
        algorithm = evox.algorithms.OpenES(
            center_init=param_vec_spec.to_vector(
                agent_state.params.policy_params),
            pop_size=config.pop_size,
            learning_rate=config.optimizer.lr,
            noise_stdev=config.noise_stdev,
            optimizer='adam',
            mirrored_sampling=True
        )

        def _candidate_transform(flat_cand):
            cand = param_vec_spec.to_tree(flat_cand)
            params = agent_state.params.replace(policy_params=cand)
            return agent_state.replace(params=params)

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

        workflow = cls(
            config=config,
            agent=agent,
            evaluator=evaluator,
            algorithm=algorithm,
            problem=problem,
            opt_direction='max',
            candidate_transforms=(jax.vmap(_candidate_transform),)
        )
        workflow._candidate_transform = _candidate_transform

        return workflow

    @staticmethod
    def _rescale_config(config: DictConfig) -> None:
        num_devices = jax.device_count()

        if config.num_envs % num_devices != 0:
            logging.warning(
                f"num_envs ({config.num_envs}) must be divisible by the number of devices ({num_devices}), "
                f"rescale eval_episodes to {config.num_envs // num_devices * num_devices}")

        config.eval_episodes = config.eval_episodes // num_devices

    def evaluate(self, state: State) -> tuple[EvaluateMetric, State]:
        """Evaluate the policy with the mean of CMAES
        """
        key, eval_key = jax.random.split(state.key, num=2)

        flat_pop_center = state.evox_state.query_state('algorithm').center
        agent_state = self._candidate_transform(flat_pop_center)

        # [#episodes]
        raw_eval_metrics = self.evaluator.evaluate(
            agent_state,
            num_episodes=self.config.eval_episodes,
            key=eval_key
        )

        eval_metrics = EvaluateMetric(
            episode_returns=raw_eval_metrics.episode_returns.mean(),
            episode_lengths=raw_eval_metrics.episode_lengths.mean()
        ).all_reduce(pmap_axis_name=self.pmap_axis_name)

        return eval_metrics, state.replace(key=key)

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
                {'eval_pop_center': eval_metrics.to_local_dict()}, i)
