"""Shared EvoX constructors and independent state/RNG assertions."""

from pathlib import Path

import jax
import jax.numpy as jnp
import torch
from omegaconf import OmegaConf
from evox.algorithms import NSGA2

from evorl.algorithms.contrib.evox.evox_algorithm_wrapper import EvoXAlgorithmWrapper
from evorl.algorithms.contrib.evox.evox_workflow import EvoXWorkflow
from evorl.utils.torch_utils import jax_to_torch_device


def torch_device():
    return jax_to_torch_device(jax.local_devices()[0])


@jax.jit
def evaluate_rewards(population, evaluation_state):
    return -jnp.square(population).sum(-1), (), evaluation_state


def make_wrapper(
    algorithm_cls,
    algorithm_kwargs,
    *,
    device,
    compile=False,
    evaluate=evaluate_rewards,
    static_tensors=None,
):
    return EvoXAlgorithmWrapper(
        algorithm_cls,
        algorithm_kwargs,
        device=device,
        compile=compile,
        evaluate=evaluate,
        static_tensors=static_tensors,
    )


def rng_states():
    states = [torch.random.get_rng_state().clone()]
    device = torch_device()
    if device.type != "cpu":
        states.append(
            torch.get_device_module(device).get_rng_state(device.index).clone()
        )
    return states


def assert_rng_states_equal(expected):
    assert all(torch.equal(left, right) for left, right in zip(expected, rng_states()))


class NSGA2TestWorkflow(EvoXWorkflow):
    """Exercise shared vector-reward handling, including a size-one vector."""

    @classmethod
    def _build_algorithm(cls, config, *, param_dim, num_objectives, device, evaluate):
        return EvoXAlgorithmWrapper(
            NSGA2,
            dict(
                pop_size=config.pop_size,
                n_objs=num_objectives,
                lb=torch.full((param_dim,), config.agent_network.lb, device=device),
                ub=torch.full((param_dim,), config.agent_network.ub, device=device),
                device=device,
            ),
            device=device,
            compile=config.torch_compile,
            evaluate=evaluate,
        )


def workflow_config(algorithm="cso"):
    root = Path(__file__).resolve().parents[2]
    return OmegaConf.merge(
        OmegaConf.load(root / "configs/config.yaml"),
        OmegaConf.load(root / f"configs/agent/contrib/{algorithm}-evox.yaml"),
        {
            "env": {"max_episode_steps": 2},
            "num_iters": 2,
            "pop_size": 4,
            "num_envs": 2,
            "episodes_for_fitness": 3,
            "normalize_obs": False,
            "agent_network": {"actor_hidden_layer_sizes": [2]},
        },
    )


def workflow_state_with_key_data(state):
    evaluation_state = state.algorithm_state.evaluation_state
    return state.replace(
        algorithm_state=state.algorithm_state.replace(
            evaluation_state=evaluation_state.replace(
                key=jax.random.key_data(evaluation_state.key)
            )
        )
    )
