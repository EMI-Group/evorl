import logging
import numpy as np
from omegaconf import DictConfig

from evox import State as EvoXState
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import State
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators import Evaluator
from evorl.utils.ec_utils import ParamVectorSpec
from evorl.agent import AgentState
from evorl.ec.evox_algorithm import CMAES, SepCMAES
from evorl.ec.evox_problems import GeneralRLProblem
from evorl.workflows import EvoXESWorkflowTemplate


from ..ec.ec_agent import make_deterministic_ec_agent

logger = logging.getLogger(__name__)


class CMAESWorkflow(EvoXESWorkflowTemplate):
    @classmethod
    def name(cls):
        return "CMAES"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        agent = make_deterministic_ec_agent(
            action_space=env.action_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,  # use linear model
            normalize_obs=False,
            norm_layer_type=config.agent_network.norm_layer_type,
        )

        problem = GeneralRLProblem(
            agent=agent,
            env=env,
            num_episodes=config.episodes_for_fitness,
            max_episode_steps=config.env.max_episode_steps,
            discount=config.discount,
            explore=config.explore,
        )

        # dummy agent_state
        agent_key = jax.random.PRNGKey(config.seed)
        agent_state = agent.init(env.obs_space, env.action_space, agent_key)
        param_vec_spec = ParamVectorSpec(agent_state.params.policy_params)

        algorithm = CMAES(
            center_init=param_vec_spec.to_vector(agent_state.params.policy_params),
            init_stdev=config.init_stdev,
            pop_size=config.pop_size,
            mu=config.num_elites,
        )

        def _candidate_transform(flat_cand):
            cand = param_vec_spec.to_tree(flat_cand)
            params = agent_state.params.replace(policy_params=cand)
            return agent_state.replace(params=params)

        eval_env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )
        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        workflow = cls(
            config=config,
            agent=agent,
            evaluator=evaluator,
            algorithm=algorithm,
            problem=problem,
            opt_direction="max",
            candidate_transforms=(jax.vmap(_candidate_transform),),
        )
        workflow._candidate_transform = _candidate_transform
        workflow._param_vec_spec = param_vec_spec

        return workflow

    def _get_pop_center(self, state: State) -> AgentState:
        flat_pop_center = state.evox_state.query_state("algorithm").mean
        agent_state = self._candidate_transform(flat_pop_center)
        return agent_state

    def _record_callback(
        self,
        evox_state: EvoXState,
        iters: int = 0,
    ):
        algo_state = evox_state.query_state("algorithm")
        cov = algo_state.C
        std = jnp.sqrt(jnp.diagonal(cov)) * algo_state.sigma

        # recover to the network shapes
        std = self._param_vec_spec.to_tree(std)
        std_statistics = _get_std_statistics(std)
        self.recorder.write({"ec/std": std_statistics}, iters)
        self.recorder.write({"ec/sigma": algo_state.sigma.tolist()}, iters)


class SepCMAESWorkflow(EvoXESWorkflowTemplate):
    @classmethod
    def name(cls):
        return "SepCMAES"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        agent = make_deterministic_ec_agent(
            action_space=env.action_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,  # use linear model
            normalize_obs=False,
            norm_layer_type=config.agent_network.norm_layer_type,
        )

        problem = GeneralRLProblem(
            agent=agent,
            env=env,
            num_episodes=config.episodes_for_fitness,
            max_episode_steps=config.env.max_episode_steps,
            discount=config.discount,
            explore=config.explore,
        )

        # dummy agent_state
        agent_key = jax.random.PRNGKey(config.seed)
        agent_state = agent.init(env.obs_space, env.action_space, agent_key)
        param_vec_spec = ParamVectorSpec(agent_state.params.policy_params)

        algorithm = SepCMAES(
            center_init=param_vec_spec.to_vector(agent_state.params.policy_params),
            init_stdev=config.init_stdev,
            pop_size=config.pop_size,
            mu=config.num_elites,
        )

        def _candidate_transform(flat_cand):
            cand = param_vec_spec.to_tree(flat_cand)
            params = agent_state.params.replace(policy_params=cand)
            return agent_state.replace(params=params)

        eval_env = create_env(
            config.env.env_name,
            config.env.env_type,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )
        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        workflow = cls(
            config=config,
            agent=agent,
            evaluator=evaluator,
            algorithm=algorithm,
            problem=problem,
            opt_direction="max",
            candidate_transforms=(jax.vmap(_candidate_transform),),
        )
        workflow._candidate_transform = _candidate_transform
        workflow._param_vec_spec = param_vec_spec

        return workflow

    def _get_pop_center(self, state: State) -> AgentState:
        flat_pop_center = state.evox_state.query_state("algorithm").mean
        agent_state = self._candidate_transform(flat_pop_center)
        return agent_state

    def _record_callback(
        self,
        evox_state: EvoXState,
        iters: int = 0,
    ) -> None:
        algo_state = evox_state.query_state("algorithm")
        cov = algo_state.C
        std = jnp.sqrt(cov) * algo_state.sigma

        # recover to the network shapes
        std = self._param_vec_spec.to_tree(std)
        std_statistics = _get_std_statistics(std)
        self.recorder.write({"ec/std": std_statistics}, iters)
        self.recorder.write({"ec/sigma": algo_state.sigma.tolist()}, iters)


def _get_std_statistics(variance):
    def _get_stats(x):
        x = np.asarray(x)
        return dict(
            min=np.min(x).tolist(),
            max=np.max(x).tolist(),
            mean=np.mean(x).tolist(),
        )

    return jtu.tree_map(_get_stats, variance)
