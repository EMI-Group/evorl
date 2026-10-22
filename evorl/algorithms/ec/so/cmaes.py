from evorl.utils import running_statistics
from flax import nnx
import numpy as np
from omegaconf import DictConfig
from typing import Self

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from evorl.types import State, Params
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators import Evaluator
from evorl.agent import AgentState
from evorl.ec.optimizers import CMAES, SepCMAES, ECState
from evorl.utils.ec_utils import ParamVectorSpec

from .es_workflow import ESWorkflowTemplate
from ..obs_utils import init_obs_preprocessor
from ..ec_agent import DeterministicECAgent


class CMAESWorkflow(ESWorkflowTemplate):
    @classmethod
    def name(cls):
        return "CMAES"

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        env = create_env(
            config.env,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        agent = DeterministicECAgent(
            agent_network_cfg=config.agent_network,
            obs_preprocessor=running_statistics.normalize
            if config.normalize_obs
            else None,
        )

        # dummy agent_state
        agent_key = jax.random.key(config.seed)
        agent_state = agent.init(env.obs_space, env.action_space, agent_key)
        param_vec_spec = ParamVectorSpec(
            nnx.state(nnx.merge(agent_state.model_state, copy=True).policy, nnx.Param)
        )

        ec_optimizer = cls._build_optimizer(config, param_vec_spec)

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
            config.env,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        agent_state_vmap_axes = AgentState(
            model_state=nnx.StateAxes({nnx.Param: 0, ...: None}),
            obs_preprocessor_state=None,
        )

        return cls(
            config=config,
            env=env,
            agent=agent,
            ec_optimizer=ec_optimizer,
            ec_evaluator=ec_evaluator,
            evaluator=evaluator,
            agent_state_vmap_axes=agent_state_vmap_axes,
        )

    @classmethod
    def _build_optimizer(cls, config: DictConfig, param_vec_spec: ParamVectorSpec):
        return CMAES(
            pop_size=config.pop_size,
            num_elites=config.num_elites,
            init_stdev=config.init_stdev,
            param_vec_spec=param_vec_spec,
        )

    def _setup_agent_and_optimizer(self, key: jax.Array) -> tuple[AgentState, ECState]:
        agent_key, ec_key = jax.random.split(key)
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )

        ec_opt_state = self.ec_optimizer.init(
            nnx.state(nnx.merge(agent_state.model_state, copy=True).policy, nnx.Param),
            ec_key,
        )

        # remove params
        agent_state = self._replace_actor_params(agent_state, params=None)

        return agent_state, ec_opt_state

    def _postsetup(self, state: State) -> State:
        # setup obs_preprocessor_state
        if self.config.normalize_obs:
            key, obs_key = jax.random.split(state.key, 2)
            agent_state = init_obs_preprocessor(
                agent_state=state.agent_state,
                config=self.config,
                key=obs_key,
                axis_name=self.dp_axis_name,
            )

            # Note: we don't count these random timesteps in state.metrics
            return state.replace(
                agent_state=agent_state,
                key=key,
            )
        else:
            return state

    def _replace_actor_params(
        self, agent_state: AgentState, params: Params
    ) -> AgentState:
        models = nnx.merge(agent_state.model_state, copy=True)
        if params is None:
            params = jax.tree.map(lambda _: None, nnx.state(models.policy, nnx.Param))
        nnx.update(models.policy, params)
        return agent_state.replace(model_state=nnx.unpack(models))

    def _get_pop_center(self, state: State) -> AgentState:
        flat_pop_center = state.ec_opt_state.mean

        pop_center = self.ec_optimizer.param_vec_spec.to_tree(flat_pop_center)

        return self._replace_actor_params(state.agent_state, pop_center)

    def _record_callback(self, state: State, iters: int) -> None:
        algo_state = state.ec_opt_state
        C = algo_state.C
        std = jnp.sqrt(jnp.diagonal(C)) * algo_state.sigma

        std = self.ec_optimizer.param_vec_spec.to_tree(std)
        std_statistics = _get_std_statistics(std)
        self.recorder.write({"ec/std": std_statistics}, iters)
        self.recorder.write({"ec/sigma": algo_state.sigma.tolist()}, iters)


class SepCMAESWorkflow(CMAESWorkflow):
    @classmethod
    def name(cls):
        return "SepCMAES"

    @classmethod
    def _build_optimizer(cls, config: DictConfig, param_vec_spec: ParamVectorSpec):
        return SepCMAES(
            pop_size=config.pop_size,
            num_elites=config.num_elites,
            init_stdev=config.init_stdev,
            param_vec_spec=param_vec_spec,
        )

    def _record_callback(self, state: State, iters: int) -> None:
        algo_state = state.ec_opt_state
        C = algo_state.C
        std = jnp.sqrt(C) * algo_state.sigma

        std = self.ec_optimizer.param_vec_spec.to_tree(std)
        std_statistics = _get_std_statistics(std)
        self.recorder.write({"ec/std": std_statistics}, iters)
        self.recorder.write({"ec/sigma": algo_state.sigma.tolist()}, iters)


def _get_std_statistics(variance):
    if isinstance(variance, nnx.State):
        variance = nnx.to_pure_dict(variance)

    def _get_stats(x):
        x = np.asarray(x)
        return dict(
            min=np.min(x).tolist(),
            max=np.max(x).tolist(),
            mean=np.mean(x).tolist(),
        )

    return jtu.tree_map(_get_stats, variance)
