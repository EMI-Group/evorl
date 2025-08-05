import logging
from omegaconf import DictConfig
from typing_extensions import Self  # pytype: disable=not-supported-yet]
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from evorl.types import Params, pytree_field, PyTreeDict
from evorl.envs import AutoresetMode, create_env
from evorl.evaluators import Evaluator
from evorl.agent import AgentState
from evorl.utils.jax_utils import rng_split_like_tree
from evorl.ec.optimizers import EvoOptimizer, ECState
from evorl.ec.optimizers.openes import compute_centered_ranks, OpenESState
from evorl.ec.optimizers.utils import weight_sum

from evorl.algorithms.ec.so.openes import OpenESWorkflow
from evorl.algorithms.ec.ec_agent import make_deterministic_ec_agent


logger = logging.getLogger(__name__)


class OpenES(EvoOptimizer):
    pop_size: int
    lr: float
    noise_std: float
    mirror_sampling: bool = True
    weight_decay: float | None = None

    fitness_shaping_fn: Callable[[chex.Array], chex.Array] = pytree_field(
        static=True, default=compute_centered_ranks
    )
    optimizer: optax.GradientTransformation = pytree_field(static=True, init=False)

    def __post_init__(self):
        assert self.pop_size > 0, "pop_size must be positive"
        if self.mirror_sampling:
            assert self.pop_size % 2 == 0, "pop_size must be even for mirror sampling"

        self.optimizer = optax.inject_hyperparams(
            optax.adam, static_args=("b1", "b2", "eps", "eps_root")
        )(learning_rate=self.lr)

    def init(self, mean: Params, key: chex.PRNGKey) -> ECState:
        return OpenESState(
            mean=mean,
            opt_state=self.optimizer.init(mean),
            noise_std=jnp.float32(self.noise_std),
            key=key,
        )

    def ask(self, state: ECState) -> tuple[chex.ArrayTree, ECState]:
        """Generate new candidate solutions."""
        key, sample_key = jax.random.split(state.key)
        sample_keys = rng_split_like_tree(sample_key, state.mean)

        if self.mirror_sampling:
            noise = jtu.tree_map(
                lambda x, k: jax.random.normal(k, shape=(self.pop_size // 2, *x.shape)),
                state.mean,
                sample_keys,
            )
            noise = jtu.tree_map(lambda z: jnp.concatenate([z, -z], axis=0), noise)
        else:
            noise = jtu.tree_map(
                lambda x, k: jax.random.normal(k, shape=(self.pop_size, *x.shape)),
                state.mean,
                sample_keys,
            )

        pop = jtu.tree_map(
            lambda m, z: m + state.noise_std * z,
            state.mean,
            noise,
        )
        state = state.replace(key=key, noise=noise)

        return pop, state

    def tell(
        self, state: ECState, fitnesses: chex.Array
    ) -> tuple[PyTreeDict, OpenESState]:
        transformed_fitnesses = self.fitness_shaping_fn(fitnesses)

        # grad = 1/(N*sigma^2) * sum(F_i*(x_i-m))
        grad = jtu.tree_map(
            # Note: we need additional "-1.0" since we are maximizing the fitness
            lambda z: (
                -weight_sum(z, transformed_fitnesses)
                / (self.pop_size * state.noise_std)
            ),
            state.noise,
        )

        # add L2 weight decay
        if self.weight_decay is not None:
            grad = jtu.tree_map(
                lambda g, x: g + self.weight_decay * x,
                grad,
                state.mean,
            )

        update, opt_state = self.optimizer.update(grad, state.opt_state)
        mean = optax.apply_updates(state.mean, update)

        return PyTreeDict(), state.replace(mean=mean, opt_state=opt_state, noise=None)


class ParamOpenESWorkflow(OpenESWorkflow):
    @classmethod
    def name(cls):
        return "ParamOpenES"

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

        ec_optimizer = OpenES(
            pop_size=config.pop_size,
            lr=config.ec_lr,
            noise_std=config.ec_noise_std,
            mirror_sampling=config.mirror_sampling,
            weight_decay=config.weight_decay,
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
