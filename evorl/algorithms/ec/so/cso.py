from omegaconf import DictConfig
from functools import partial

import evox.algorithms
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import orbax.checkpoint as ocp

from evorl.distributed import tree_unpmap
from evorl.envs import AutoresetMode, create_env
from evorl.types import State
from evorl.utils.ec_utils import ParamVectorSpec
from evorl.recorders import get_1d_array_statistics
from evorl.workflows import EvoXWorkflowWrapper

from ..evox_problems import GeneralRLProblem
from ..ec_agent import make_deterministic_ec_agent


class CSOWorkflow(EvoXWorkflowWrapper):
    @classmethod
    def name(cls):
        return "CSO"

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

        # TODO: impl complete version of OpenES
        algorithm = evox.algorithms.CSO(
            lb=jnp.full((param_vec_spec.vec_size,), fill_value=config.agent_network.lb),
            ub=jnp.full((param_vec_spec.vec_size,), fill_value=config.agent_network.ub),
            pop_size=config.pop_size,
            phi=config.phi,
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
            opt_direction="max",
            candidate_transforms=(jax.vmap(_candidate_transform),),
        )

    def learn(self, state: State) -> State:
        start_iteration = tree_unpmap(state.metrics.iterations, self.pmap_axis_name)

        for i in range(start_iteration, self.config.num_iters):
            iters = i + 1
            train_metrics, state = self.step(state)
            workflow_metrics = state.metrics

            workflow_metrics = tree_unpmap(workflow_metrics, self.pmap_axis_name)
            self.recorder.write(workflow_metrics.to_local_dict(), iters)

            train_metrics = tree_unpmap(train_metrics, self.pmap_axis_name)
            train_metrics_dict = train_metrics.to_local_dict()
            train_metrics_dict = jtu.tree_map(
                partial(get_1d_array_statistics, histogram=True),
                train_metrics.to_local_dict(),
            )
            self.recorder.write(train_metrics_dict, iters)

            self.checkpoint_manager.save(
                iters,
                args=ocp.args.StandardSave(
                    tree_unpmap(state, self.pmap_axis_name),
                ),
            )
