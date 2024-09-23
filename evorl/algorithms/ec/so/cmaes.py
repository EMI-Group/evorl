import logging
import numpy as np

import evox.algorithms
from evox import State as EvoXState
import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import State
from evorl.envs import AutoresetMode, create_env
from evorl.evaluator import Evaluator
from evorl.utils.ec_utils import ParamVectorSpec
from evorl.agent import AgentState
from omegaconf import DictConfig

from ..problems import GeneralRLProblem
from ..ec_agent import DeterministicECAgent
from .es_base import ESWorkflowTemplate

logger = logging.getLogger(__name__)


class CMAESWorkflow(ESWorkflowTemplate):
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

        agent = DeterministicECAgent(
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,  # use linear model
            normalize_obs=False,
            norm_layer_type=config.agent_network.norm_layer_type,
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
        agent_state = agent.init(env.obs_space, env.action_space, agent_key)
        param_vec_spec = ParamVectorSpec(agent_state.params.policy_params)

        num_elites = config.num_elites
        recombination_weights = jnp.log(num_elites + 0.5) - jnp.log(
            jnp.arange(1, num_elites + 1)
        )
        recombination_weights /= jnp.sum(recombination_weights)

        algorithm = evox.algorithms.CMAES(
            center_init=param_vec_spec.to_vector(agent_state.params.policy_params),
            init_stdev=config.init_stdev,
            pop_size=config.pop_size,
            recombination_weights=recombination_weights,
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
            env=eval_env, agent=agent, max_episode_steps=config.env.max_episode_steps
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
        diag_cov = jnp.diagonal(cov)

        # recover to the network shapes
        diag_cov = self._param_vec_spec.to_tree(diag_cov)
        std_statistics = _get_std_statistics(diag_cov)
        self.recorder.write({"ec/std": std_statistics}, iters)

        self.recorder.write({"ec/sigma": algo_state.sigma.tolist()}, iters)


def _get_std_statistics(variance):
    def _get_stats(x):
        x = np.asarray(x)
        x = np.sqrt(x)
        return dict(
            min=np.min(x).tolist(),
            max=np.max(x).tolist(),
            mean=np.mean(x).tolist(),
        )

    return jtu.tree_map(_get_stats, variance)
