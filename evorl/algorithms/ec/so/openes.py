import logging

import evox.algorithms
import jax

from evorl.ec import GeneralRLProblem
from evorl.envs import AutoresetMode, create_wrapped_brax_env
from evorl.evaluator import Evaluator
from evorl.utils.ec_utils import ParamVectorSpec
from omegaconf import DictConfig

from ..ec_agent import DeterministicECAgent
from .es_base import ESWorkflowTemplate

logger = logging.getLogger(__name__)


class OpenESWorkflow(ESWorkflowTemplate):
    @classmethod
    def name(cls):
        return "OpenES"

    @classmethod
    def _build_from_config(cls, config: DictConfig):
        env = create_wrapped_brax_env(
            config.env.env_name,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        agent = DeterministicECAgent(
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,  # use linear model
            normalize_obs=False,
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

        # TODO: impl complete version of OpenES
        algorithm = evox.algorithms.OpenES(
            center_init=param_vec_spec.to_vector(agent_state.params.policy_params),
            pop_size=config.pop_size,
            learning_rate=config.optimizer.lr,
            noise_stdev=config.noise_stdev,
            optimizer="adam",
            mirrored_sampling=True,
        )

        def _candidate_transform(flat_cand):
            cand = param_vec_spec.to_tree(flat_cand)
            params = agent_state.params.replace(policy_params=cand)
            return agent_state.replace(params=params)

        eval_env = create_wrapped_brax_env(
            config.env.env_name,
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

        return workflow
