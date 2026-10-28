"""Single-device EvoX optimization with EvoRL agents and environments."""

from copy import deepcopy
from collections.abc import Callable
from typing import Self

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from evorl.agent import Agent, AgentState
from evorl.algorithms.ec.ec_agent import DeterministicECAgent
from evorl.algorithms.ec.obs_utils import init_obs_preprocessor
from evorl.envs import AutoresetMode, BaseEnv, create_env
from evorl.evaluators import Evaluator
from evorl.metrics import ECTrainMetric, MultiObjectiveECWorkflowMetric
from evorl.recorders import get_1d_array_statistics
from evorl.types import PyTreeDict, State
from evorl.utils.ec_utils import ParamVectorSpec
from evorl.utils import running_statistics
from evorl.networks.utils import model_state_axes
from evorl.utils.torch_utils import jax_to_torch_device
from evorl.workflows import Workflow

from .evox_algorithm_wrapper import EvoXAlgorithmWrapper


class EvoXWorkflow(Workflow):
    """Use a PyTorch EvoX algorithm on scalar or vector single-agent rewards.

    Python orchestrates one algorithm. `build_from_config(..., jit=True)`
    compiles JAX population evaluation through the `enable_jit` class hook.
    Observation warmup runs without an extra JIT wrapper.
    `config.torch_compile` independently controls PyTorch compilation. DLPack
    keeps populations and fitness on the shared device; evaluation calls stay
    outside the PyTorch graph.
    Policies use the existing deterministic, continuous-action EC agent.
    Subclasses implement `_build_algorithm` to supply algorithm-specific
    constructor arguments explicitly.
    """

    def __init__(
        self,
        *,
        env: BaseEnv,
        agent: Agent,
        evaluator: Evaluator,
        param_vec_spec: ParamVectorSpec,
        num_objectives: int | None,
        device: torch.device,
        config: DictConfig,
    ):
        """Initialize the workflow with its static components.

        Args:
            env: Vectorized environment used for policy evaluation.
            agent: Policy and observation preprocessing functions.
            evaluator: Episode evaluator used for candidate fitness.
            param_vec_spec: Conversion between policy trees and vectors.
            num_objectives: `None` for scalar rewards, otherwise vector length.
            device: PyTorch device shared with JAX policy evaluation.
            config: Workflow configuration.
        """
        super().__init__(config)
        self.env = env
        self.agent = agent
        self.evaluator = evaluator
        self.param_vec_spec = param_vec_spec
        self.vector_reward = num_objectives is not None
        try:
            self.algorithm = self._build_algorithm(
                config,
                param_dim=param_vec_spec.vec_size,
                num_objectives=num_objectives,
                device=device,
                evaluate=self._evaluate,
            )
        except Exception:
            self.checkpoint_manager.close()
            raise

    @classmethod
    def name(cls):
        return "EvoX"

    @classmethod
    def build_from_config(
        cls, config: DictConfig, multi_device: bool = False, jit: bool = True
    ) -> Self:
        """Build an optional EvoX workflow on the default JAX device.

        Call on a fresh execution subclass, as in `scripts/train.py`, because
        `enable_jit` transforms methods on that subclass.

        Args:
            config: Environment, agent, evaluation and EvoX configuration.
            multi_device: Must be false; this integration uses one device.
            jit: Compile JAX population evaluation through `enable_jit`.
                PyTorch compilation is controlled by `config.torch_compile`.

        Returns:
            A workflow ready for `init(key)`.

        Raises:
            NotImplementedError: If `multi_device` is true.
        """
        if multi_device:
            raise NotImplementedError("The EvoX workflow currently supports one device")
        config = deepcopy(config)
        OmegaConf.set_readonly(config, True)
        # Bind the compiled callback when constructing the algorithm wrapper.
        if jit:
            cls.enable_jit()
        return cls._build_from_config(config)

    @classmethod
    def _build_from_config(cls, config: DictConfig) -> Self:
        """Construct the environment, policy, EvoX algorithm and evaluator.

        Environment setup, policy evaluation and training are shared. The
        concrete subclass constructs its algorithm through `_build_algorithm`.

        Args:
            config: Independent, read-only workflow configuration.

        Returns:
            A single-device workflow with its configured evaluation callback.
        """
        if config.num_iters < 0:
            raise ValueError("num_iters must be nonnegative")
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
        key = jax.random.key(config.seed)
        reward = jax.eval_shape(env.reset, key).reward
        if not isinstance(reward, jax.ShapeDtypeStruct) or reward.ndim not in (1, 2):
            raise ValueError("Expected batched scalar or vector single-agent rewards")
        vector_reward = reward.ndim == 2
        num_objectives = reward.shape[1] if vector_reward else None
        if num_objectives is not None and num_objectives < 1:
            raise ValueError("Vector rewards must have at least one objective")
        dummy_agent_state = agent.init(env.obs_space, env.action_space, key)
        params = nnx.state(
            nnx.merge(dummy_agent_state.model_state, copy=True).policy, nnx.Param
        )
        param_vec_spec = ParamVectorSpec(params)
        torch_device = jax_to_torch_device(jax.tree.leaves(params)[0].device)
        evaluator = Evaluator(
            env=env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=config.env.max_episode_steps,
            discount=config.discount,
        )
        return cls(
            config=config,
            env=env,
            agent=agent,
            evaluator=evaluator,
            param_vec_spec=param_vec_spec,
            num_objectives=num_objectives,
            device=torch_device,
        )

    @classmethod
    def _build_algorithm(
        cls,
        config: DictConfig,
        *,
        param_dim: int,
        num_objectives: int | None,
        device: torch.device,
        evaluate: Callable,
    ) -> EvoXAlgorithmWrapper:
        """Construct the concrete algorithm and its functional adapter.

        Args:
            config: Prepared workflow configuration.
            param_dim: Number of flattened policy parameters.
            num_objectives: `None` for scalar rewards, otherwise the vector
                length, including one for a single-objective vector.
            device: Single PyTorch device shared with JAX evaluation.
            evaluate: Fixed reward callback with explicit evaluation state.

        Returns:
            An algorithm adapter bound to the supplied reward callback.
        """
        raise NotImplementedError

    @classmethod
    def enable_jit(cls) -> None:
        """Compile population evaluation, leaving Python/Torch orchestration eager.

        Apply on a fresh execution subclass, as in `scripts/train.py`.
        Observation warmup and the small training-statistics updates remain
        outside this compilation boundary.
        """
        cls._evaluate = jax.jit(cls._evaluate, static_argnums=0)

    def setup(self, key: jax.Array) -> State:
        """Create complete state and account for native initialization sampling.

        EvoX's native `init_step` belongs to initialization. It may evaluate
        candidates or perform an initial update. `iterations` starts at zero;
        `num_iters` counts subsequent calls to `step`. Recording and checkpoint
        saving are handled exclusively by `learn`.
        """
        key, agent_key, algorithm_key, obs_key = jax.random.split(key, 4)
        agent_state = self.agent.init(
            self.env.obs_space, self.env.action_space, agent_key
        )
        if self.config.normalize_obs:
            agent_state = init_obs_preprocessor(agent_state, self.config, obs_key)
        models = nnx.merge(agent_state.model_state, copy=True)
        nnx.update(
            models.policy,
            jax.tree.map(lambda _: None, nnx.state(models.policy, nnx.Param)),
        )
        agent_state = agent_state.replace(model_state=nnx.unpack(models))
        evaluation_metrics, algorithm_state = self.algorithm.init(
            algorithm_key, PyTreeDict(key=key, agent_state=agent_state)
        )
        workflow_metrics = MultiObjectiveECWorkflowMetric(
            sampled_episodes=jnp.asarray(
                sum(batch.episode_length.size for batch in evaluation_metrics),
                dtype=jnp.uint32,
            ),
            sampled_timesteps_m=sum(
                batch.episode_length.sum() for batch in evaluation_metrics
            )
            / 1e6,
        )
        return State(algorithm_state=algorithm_state, metrics=workflow_metrics)

    def _evaluate(self, population, evaluation_state):
        """Evaluate policies, returning rewards, batch metrics and updated state.

        This pure JAX function covers policy reconstruction, episode rollout
        and reward reduction. `enable_jit` compiles it before the algorithm
        wrapper binds the callback. Per-batch outputs stay outside persistent
        state, including when EvoX requests several batches of different sizes.
        """
        key, rollout_key = jax.random.split(evaluation_state.key)
        params = self.param_vec_spec.to_tree(population)
        agent_state = evaluation_state.agent_state
        models = nnx.merge(agent_state.model_state, copy=True)
        nnx.update(models.policy, params)
        population_state = agent_state.replace(model_state=nnx.unpack(models))
        metrics = jax.vmap(
            self.evaluator.evaluate,
            in_axes=(
                AgentState(
                    model_state=model_state_axes(
                        population_state.model_state,
                        nnx.StateAxes({nnx.Param: 0, ...: None}),
                    ),
                    obs_preprocessor_state=None,
                ),
                0,
                None,
            ),
        )(
            population_state,
            jax.random.split(rollout_key, population.shape[0]),
            self.config.episodes_for_fitness,
        )
        # Axis 1 is episodes for both scalar and vector rewards.
        fitness = metrics.episode_return.mean(axis=1)
        return (
            fitness,
            PyTreeDict(objectives=fitness, episode_length=metrics.episode_length),
            evaluation_state.replace(key=key),
        )

    def step(self, state: State) -> tuple[ECTrainMetric, State]:
        """Optimize once and count every actual candidate/episode evaluation."""
        evaluation_metrics, algorithm_state = self.algorithm.step(state.algorithm_state)
        if not evaluation_metrics:
            raise ValueError("The EvoX iteration did not evaluate any candidates")
        train_metrics = ECTrainMetric(
            objectives=jnp.concatenate(
                [batch.objectives for batch in evaluation_metrics]
            ),
            ec_metrics=PyTreeDict(),
        )
        workflow_metrics = state.metrics.replace(
            iterations=state.metrics.iterations + 1,
            sampled_episodes=state.metrics.sampled_episodes
            + sum(batch.episode_length.size for batch in evaluation_metrics),
            sampled_timesteps_m=state.metrics.sampled_timesteps_m
            + sum(batch.episode_length.sum() for batch in evaluation_metrics) / 1e6,
        )
        return train_metrics, state.replace(
            algorithm_state=algorithm_state, metrics=workflow_metrics
        )

    def learn(self, state: State) -> State:
        """Run remaining updates, record their metrics and save checkpoints."""
        for i in range(int(state.metrics.iterations), self.config.num_iters):
            iteration = i + 1
            train_metrics, state = self.step(state)
            self.recorder.write(state.metrics.to_local_dict(), iteration)
            objectives = np.asarray(train_metrics.objectives)
            if self.vector_reward:
                names = self.config.env.get("reward_keys") or [
                    f"objective_{i}" for i in range(objectives.shape[1])
                ]
                statistics = {
                    name: get_1d_array_statistics(objectives[:, i], histogram=True)
                    for i, name in enumerate(names)
                }
            else:
                statistics = get_1d_array_statistics(objectives, histogram=True)
            self.recorder.write({"objectives": statistics}, iteration)
            self.checkpoint_manager.save(
                iteration, state, force=iteration == self.config.num_iters
            )
        return state
