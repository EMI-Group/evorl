from omegaconf import DictConfig
from typing_extensions import Self  # pytype: disable=not-supported-yet]

import jax

from evorl.types import State, Params
from evorl.envs import AutoresetMode, create_env
from evorl.evaluator import Evaluator
from evorl.agent import AgentState
from evorl.ec.optimizers import SepCEM, ExponentialScheduleSpec, ECState

from .es_base import ESWorkflowTemplate
from ..ec_agent import make_deterministic_ec_agent


class SepCEMWorkflow(ESWorkflowTemplate):
    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
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

        ec_optimizer = SepCEM(
            pop_size=config.pop_size,
            num_elites=config.num_elites,
            diagonal_variance=ExponentialScheduleSpec(**config.diagonal_variance),
            weighted_update=config.weighted_update,
            rank_weight_shift=config.rank_weight_shift,
            mirror_sampling=config.mirror_sampling,
        )

        if config.explore:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        ec_evaluator = Evaluator(
            env=env,
            action_fn=action_fn,
            max_episode_steps=config.env.max_episode_steps,
            discount=config.discount,
        )

        # to evaluate the pop-mean actor
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

        return cls(
            config=config,
            env=env,
            agent=agent,
            ec_optimizer=ec_optimizer,
            ec_evaluator=ec_evaluator,
            evaluator=evaluator,
        )

    def _setup_agent_and_optimizer(self, key: jax.Array) -> tuple[AgentState, ECState]:
        agent_key, opt_key = jax.random.split(key)
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )

        init_actor_params = agent_state.params.policy_params
        ec_opt_state = self.ec_optimizer.init(init_actor_params)

        agent_state = self._replace_actor_params(agent_state, params=None)

        return agent_state, ec_opt_state

    def _replace_actor_params(
        self, agent_state: AgentState, params: Params
    ) -> AgentState:
        return agent_state.replace(
            params=agent_state.params.replace(policy_params=params)
        )

    def _get_pop_center(self, state: State) -> AgentState:
        pop_center = state.ec_opt_state.mean

        return self._replace_actor_params(state.agent_state, pop_center)
