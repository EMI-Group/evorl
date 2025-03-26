import logging
from omegaconf import DictConfig
from collections.abc import Sequence
from typing_extensions import Self  # pytype: disable=not-supported-yet]

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import flax.linen as nn

from evorl.types import State, Params
from evorl.envs import AutoresetMode, create_env, Space, Box
from evorl.evaluators import Evaluator, EpisodeObsCollector
from evorl.sample_batch import SampleBatch
from evorl.agent import AgentState
from evorl.ec.optimizers import ARS, ECState
from evorl.utils import running_statistics
from evorl.networks import make_mlp, ActivationFn

from evorl.algorithms.ec.so.es_workflow import ESWorkflowTemplate
from evorl.algorithms.ec.obs_utils import init_obs_preprocessor
from evorl.algorithms.ec.ec_agent import DeterministicECAgent


logger = logging.getLogger(__name__)


def make_policy_network(
    action_size: int,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    use_bias: bool = True,
    activation: ActivationFn = nn.relu,
    activation_final: ActivationFn | None = None,
    norm_layer_type: str = "none",
) -> nn.Module:
    """Creates a policy network."""

    class Policy(nn.Module):
        @nn.compact
        def __call__(self, x):
            policy_model = make_mlp(
                layer_sizes=tuple(hidden_layer_sizes) + (action_size,),
                activation=activation,
                # kernel_init=jax.nn.initializers.lecun_uniform(),
                kernel_init=jax.nn.initializers.zeros,
                activation_final=activation_final,
                use_bias=use_bias,
                norm_layer_type=norm_layer_type,
            )
            actions = policy_model(x)
            return jnp.clip(actions, -1.0, 1.0)

    return Policy()


def make_deterministic_ec_agent(
    action_space: Space,
    actor_hidden_layer_sizes: tuple[int] = (256, 256),
    use_bias: bool = True,
    norm_layer_type: str = "none",
    normalize_obs: bool = False,
):
    assert isinstance(action_space, Box), "Only continue action space is supported."

    action_size = action_space.shape[0]

    policy_network = make_policy_network(
        action_size=action_size,
        hidden_layer_sizes=actor_hidden_layer_sizes,
        use_bias=use_bias,
        activation_final=None,
        norm_layer_type=norm_layer_type,
    )

    if normalize_obs:
        obs_preprocessor = running_statistics.normalize
    else:
        obs_preprocessor = None

    return DeterministicECAgent(
        policy_network=policy_network,
        obs_preprocessor=obs_preprocessor,
    )


class ARSWorkflow(ESWorkflowTemplate):
    @classmethod
    def name(cls):
        return "ARS"

    @classmethod
    def _rescale_config(cls, config: DictConfig) -> None:
        super()._rescale_config(config)

        num_devices = jax.device_count()
        if config.random_timesteps % num_devices != 0:
            logger.warning(
                f"When enable_multi_devices=True, pop_size ({config.random_timesteps}) should be divisible by num_devices ({num_devices}),"
            )

        config.random_timesteps = (config.random_timesteps // num_devices) * num_devices

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        env = create_env(
            config.env,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        agent = make_deterministic_ec_agent(
            action_space=env.action_space,
            actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,
            use_bias=config.agent_network.use_bias,
            normalize_obs=config.normalize_obs,
            norm_layer_type=config.agent_network.norm_layer_type,
        )

        ec_optimizer = ARS(
            pop_size=config.pop_size,
            num_elites=config.num_elites,
            lr=config.lr,
            noise_std=config.noise_std,
            optimizer_name=config.optimizer_name,
        )

        if config.explore:
            action_fn = agent.compute_actions
        else:
            action_fn = agent.evaluate_actions

        assert config.normalize_obs_mode in ["VBN", "RS", "Global"]
        if config.normalize_obs_mode == "VBN":
            ec_evaluator = Evaluator(
                env=env,
                action_fn=action_fn,
                max_episode_steps=config.env.max_episode_steps,
                discount=config.discount,
            )
        else:
            ec_evaluator = EpisodeObsCollector(
                env=env,
                action_fn=action_fn,
                max_episode_steps=config.env.max_episode_steps,
                discount=config.discount,
            )

        # to evaluate the pop-mean actor
        eval_env = create_env(
            config.env,
            episode_length=config.env.max_episode_steps,
            parallel=config.num_eval_envs,
            autoreset_mode=AutoresetMode.DISABLED,
        )

        evaluator = Evaluator(
            env=eval_env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
        )

        agent_state_vmap_axes = AgentState(
            params=0,
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

    def _setup_agent_and_optimizer(self, key: jax.Array) -> tuple[AgentState, ECState]:
        agent_key, ec_key = jax.random.split(key)
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )

        init_actor_params = agent_state.params.policy_params
        ec_opt_state = self.ec_optimizer.init(init_actor_params, ec_key)

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
                pmap_axis_name=self.pmap_axis_name,
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
        return agent_state.replace(
            params=agent_state.params.replace(policy_params=params)
        )

    def _get_pop_center(self, state: State) -> AgentState:
        pop_center = state.ec_opt_state.mean

        return self._replace_actor_params(state.agent_state, pop_center)

    def _update_obs_preprocessor(
        self, agent_state: AgentState, trajectory: SampleBatch
    ) -> AgentState:
        if self.config.normalize_obs_mode == "Global":
            obs_preprocessor_state = running_statistics.update(
                agent_state.obs_preprocessor_state,
                trajectory.obs,
                weights=1 - trajectory.dones,
                pmap_axis_name=self.pmap_axis_name,
            )

        elif self.config.normalize_obs_mode == "RS":
            dummy_obs = jtu.tree_map(lambda x: x[0, 0], trajectory.obs)
            obs_preprocessor_state = running_statistics.init_state(dummy_obs)
            obs_preprocessor_state = running_statistics.update(
                obs_preprocessor_state,
                trajectory.obs,
                weights=1 - trajectory.dones,
                pmap_axis_name=self.pmap_axis_name,
            )
        else:
            obs_preprocessor_state = agent_state.obs_preprocessor_state

        return agent_state.replace(obs_preprocessor_state=obs_preprocessor_state)
