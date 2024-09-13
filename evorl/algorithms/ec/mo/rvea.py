from omegaconf import DictConfig

import evox.algorithms
import jax
import jax.numpy as jnp

from evorl.envs import AutoresetMode, create_wrapped_brax_env
from evorl.utils.ec_utils import ParamVectorSpec

from ..problems import MultiObjectiveBraxProblem
from .mo_base import MOECWorkflowTemplate
from ..ec_agent import DeterministicECAgent


class RVEAWorkflow(MOECWorkflowTemplate):
    @classmethod
    def name(cls):
        return "RVEA"

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
            norm_layer_type=config.agent_network.norm_layer_type,
        )

        problem = MultiObjectiveBraxProblem(
            agent=agent,
            env=env,
            num_episodes=config.episodes,
            max_episode_steps=config.env.max_episode_steps,
            discount=config.discount,
            metric_names=config.obj_names,
            flatten_objectives=True,
        )

        # dummy agent_state
        agent_key = jax.random.PRNGKey(config.seed)
        agent_state = agent.init(env.obs_space, env.action_space, agent_key)
        param_vec_spec = ParamVectorSpec(agent_state.params.policy_params)

        algorithm = evox.algorithms.RVEA(
            lb=jnp.full((param_vec_spec.vec_size,), fill_value=config.agent_network.lb),
            ub=jnp.full((param_vec_spec.vec_size,), fill_value=config.agent_network.ub),
            n_objs=len(config.obj_names),
            pop_size=config.pop_size,
            alpha=config.alpha,
            fr=config.fr,
            max_gen=config.num_iters,
        )

        def _candidate_transform(flat_cand):
            cand = param_vec_spec.to_tree(flat_cand)
            params = agent_state.params.replace(policy_params=cand)
            return agent_state.replace(params=params)

        return cls(
            config=config,
            agent=agent,
            algorithm=algorithm,
            problem=problem,
            opt_direction=config.opt_directions,
            candidate_transforms=(jax.vmap(_candidate_transform),),
        )
