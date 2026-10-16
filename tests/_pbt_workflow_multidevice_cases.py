"""Exercise real PBT targets on the independently named meta-population axis."""

from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import get_class
from omegaconf import OmegaConf

from evorl.distributed import META_POP_AXIS_NAME, POP_AXIS_NAME
from evorl.rollout import rollout, eval_rollout_episode, fast_eval_rollout_episode
from evorl.types import PyTreeDict

from _workflow_multidevice_cases import ToyEnv


def check_rollout_context():
    class VariableLengthEnv(ToyEnv):
        def step(self, state, action):
            next_state = super().step(state, action)
            # Input-dependent termination keeps the evaluation loop dynamic;
            # uniform lengths can hide automatic cross-device loop reductions.
            threshold = 1 + jnp.floor(state.obs[:, 0] * 4).astype(jnp.int32)
            return next_state.replace(
                done=(next_state.env_state >= threshold).astype(jnp.float32)
            )

    env = VariableLengthEnv(2)

    def action_fn(agent_state, batch, key):
        return batch.obs[:, :1], PyTreeDict()

    def instance(key):
        state = env.reset(key)
        train, _ = rollout(env.step, action_fn, state, None, key, 2)
        episode, _ = eval_rollout_episode(env.step, action_fn, state, None, key, 2)
        fast, _ = fast_eval_rollout_episode(env.step, action_fn, state, None, key, 2)
        return jnp.stack(
            (train.rewards.sum(), episode.rewards.sum(), fast.episode_returns.sum())
        )

    keys = jax.random.split(jax.random.key(3), 4)
    expected = jax.jit(instance)(keys[0])
    mesh = jax.make_mesh((2,), (META_POP_AXIS_NAME,))
    with jax.set_mesh(mesh):
        mapped = jax.shard_map(
            jax.vmap(instance),
            in_specs=jax.P(META_POP_AXIS_NAME),
            out_specs=jax.P(META_POP_AXIS_NAME),
        )
        inputs = jax.device_put(keys, jax.P(META_POP_AXIS_NAME))
        executable = jax.jit(mapped).lower(inputs).compile()
        hlo = executable.as_text()
        assert "all-reduce" not in hlo and "all-gather" not in hlo
        actual = executable(inputs)
        np.testing.assert_allclose(np.asarray(actual)[0], expected, rtol=1e-6)
    print(
        "All rollout variants infer META_POP under jit/vmap, matching local execution"
    )


def run_case(target):
    root = Path(__file__).resolve().parents[1]
    common = OmegaConf.load(root / "configs/config.yaml")
    target_config = OmegaConf.merge(
        common,
        OmegaConf.load(root / f"configs/agent/meta/{target}.yaml"),
        dict(
            num_envs=2,
            num_eval_envs=2,
            eval_episodes=2,
            normalize_obs=True,
            random_timesteps=4,
            agent_network=dict(actor_hidden_layer_sizes=[4]),
        ),
    )
    if target == "paramppo":
        target_config.rollout_length = 2
        target_config.minibatch_size = 4
        target_config.reuse_rollout_epochs = 1
        target_config.agent_network.critic_hidden_layer_sizes = [4]
    else:
        target_config.num_envs = 1
        target_config.pop_size = 4
        target_config.episodes_for_fitness = 1
    config = OmegaConf.merge(
        common,
        OmegaConf.load(root / f"configs/agent/meta/pbt-{target}.yaml"),
        dict(
            target_workflow=target_config,
            pop_size=4,
            num_eval_envs=2,
            eval_episodes=2,
            workflow_steps_per_iter=1,
            warmup_steps=100,
            bottom_ratio=0.25,
            top_ratio=0.25,
            env=dict(max_episode_steps=2),
        ),
    )
    target_cls = get_class(target_config.workflow_cls)
    base = get_class(config.workflow_cls)
    cls = type(f"AxisProbe{base.__name__}", (base,), {})

    def create_env(*args, **kwargs):
        return ToyEnv(kwargs["parallel"])

    with (
        patch(f"{target_cls.__module__}.create_env", side_effect=create_env),
        patch("evorl.algorithms.meta.pbt_workflow.create_env", side_effect=create_env),
    ):
        workflow = cls.build_from_config(config, jit=True)
    try:
        assert workflow.mesh.axis_names == (META_POP_AXIS_NAME,)
        assert META_POP_AXIS_NAME != POP_AXIS_NAME
        assert workflow.workflow.dp_axis_name is None
        assert not hasattr(workflow, "rollout_axis_name")
        assert not hasattr(workflow.workflow, "rollout_axis_name")
        with (
            patch("evorl.algorithms.ec.obs_utils.create_env", side_effect=create_env),
            jax.set_mesh(workflow.mesh),
        ):
            state = jax.jit(workflow.setup)(jax.random.key(0))
            if target == "paramopenes":
                # VMA inference must not reduce statistics over META_POP.
                np.testing.assert_array_equal(
                    state.pop_workflow_state.agent_state.obs_preprocessor_state.count,
                    [4] * config.pop_size,
                )

            @jax.shard_map(
                in_specs=jax.P(META_POP_AXIS_NAME),
                out_specs=jax.P(META_POP_AXIS_NAME),
            )
            def train_local_members(states):
                return jax.vmap(workflow.workflow.step)(states)

            executable = (
                jax.jit(train_local_members).lower(state.pop_workflow_state).compile()
            )
            hlo = executable.as_text()
            assert "all-reduce" not in hlo and "all-gather" not in hlo
            _, trained = executable(state.pop_workflow_state)
            if target == "paramppo":
                # One population member must match its independent local update.
                member_index = jnp.array([0])
                member = jax.tree.map(
                    lambda x: x.at[member_index].get(out_sharding=jax.P())[0],
                    state.pop_workflow_state,
                )
                _, expected = workflow.workflow.step(member)
                for actual, reference in zip(
                    jax.tree.leaves(trained.agent_state.params),
                    jax.tree.leaves(expected.agent_state.params),
                    strict=True,
                ):
                    np.testing.assert_allclose(
                        actual.at[member_index].get(out_sharding=jax.P())[0],
                        reference,
                        rtol=1e-5,
                        atol=1e-6,
                    )
            _, state = workflow.step(state)
            metrics, state = workflow.evaluate(state)
            jax.block_until_ready(state)
            assert np.isfinite(metrics.pop_episode_returns).all()
            assert state.pop_workflow_state.key.sharding.spec == jax.P(
                META_POP_AXIS_NAME
            )
        print(f"{target}: META_POP setup/step/evaluate, independent updates PASS")
    finally:
        workflow.close()


if __name__ == "__main__":
    assert jax.device_count() == 2
    check_rollout_context()
    for target in ("paramppo", "paramopenes"):
        run_case(target)
