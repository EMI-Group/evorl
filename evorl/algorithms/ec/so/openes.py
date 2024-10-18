import logging
from omegaconf import DictConfig
from typing_extensions import Self  # pytype: disable=not-supported-yet]

import jax

from evorl.types import State, Params
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators import (
    Evaluator,
    EpisodeCollector,
    init_obs_preprocessor_with_random_timesteps,
)
from evorl.agent import AgentState
from evorl.ec.optimizers import OpenES, ExponentialScheduleSpec, ECState

from .es_base import ESWorkflowTemplate
from ..ec_agent import make_deterministic_ec_agent

logger = logging.getLogger(__name__)


class OpenESWorkflow(ESWorkflowTemplate):
    @classmethod
    def name(cls):
        return "OpenES"

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        cls._rescale_config(config)

        num_devices = jax.device_count()
        if config.random_timesteps % num_devices != 0:
            logging.warning(
                f"When enable_multi_devices=True, pop_size ({config.random_timesteps}) should be divisible by num_devices ({num_devices}),"
            )

        config.random_timesteps = (config.random_timesteps // num_devices) * num_devices

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
            normalize_obs=config.normalize_obs,
            norm_layer_type=config.agent_network.norm_layer_type,
        )

        ec_optimizer = OpenES(
            pop_size=config.pop_size,
            lr_schedule=ExponentialScheduleSpec(**config.ec_lr),
            noise_std_schedule=ExponentialScheduleSpec(**config.ec_noise_std),
            mirror_sampling=config.mirror_sampling,
        )

        if config.explore:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        ec_evaluator = EpisodeCollector(
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
        agent_key, obs_key = jax.random.split(key)
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )

        init_actor_params = agent_state.params.policy_params
        ec_opt_state = self.ec_optimizer.init(init_actor_params)

        # steup obs_preprocessor_state
        if self.config.normalize_obs:
            env = create_env(
                self.config.env.env_name,
                self.config.env.env_type,
                episode_length=self.config.env.max_episode_steps,
                parallel=self.config.num_envs,
                autoreset_mode=AutoresetMode.NORMAL,
            )

            obs_preprocessor_state = init_obs_preprocessor_with_random_timesteps(
                agent_state.obs_preprocessor_state,
                self.config.random_timesteps,
                env,
                obs_key,
                pmap_axis_name=self.pmap_axis_name,
            )

            agent_state = agent_state.replace(
                obs_preprocessor_state=obs_preprocessor_state
            )

        # remove params
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
