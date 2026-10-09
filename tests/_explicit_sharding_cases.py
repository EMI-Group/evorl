"""Small two-device workflow cases executed in a fresh CPU process."""


def main():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from evorl.types import State, PyTreeDict
    from evorl.sample_batch import SampleBatch
    from evorl.distributed import tree_scatter_set
    from evorl.utils.jax_utils import tree_get
    from evorl.agent import AgentState
    from evorl.metrics import WorkflowMetric, EvaluateMetric
    from evorl.workflows.rl_workflow import OnPolicyWorkflow, OffPolicyWorkflow
    from evorl.workflows.ec_workflow import ECWorkflowTemplate
    from evorl.algorithms.random_agent import RandomAgentWorkflow
    from evorl.algorithms.meta.pbt_workflow import (
        PBTWorkflowTemplate,
        PBTOffpolicyWorkflowTemplate,
        PBTOptState,
    )
    from evorl.algorithms.meta.pbt_ppo.pbt_cso_param_ppo import PBTCSOParamPPOWorkflow

    assert jax.device_count() == 2

    class ToyEnv:
        obs_space = None
        action_space = None

        def reset(self, key):
            return PyTreeDict(key=key, obs=jax.random.normal(key, (3,)))

    env = ToyEnv()
    dp_mesh = jax.make_mesh((2,), ("DP",), axis_types=(jax.sharding.AxisType.Explicit,))
    jax.set_mesh(dp_mesh)
    common = dict(
        enable_multi_devices=True,
        mesh=dp_mesh,
        dp_axis_name="DP",
        env=env,
        _setup_agent_and_optimizer=lambda key: (
            AgentState(params=PyTreeDict(w=jnp.ones(3))),
            PyTreeDict(m=jnp.zeros(3)),
        ),
        _setup_workflow_metrics=WorkflowMetric,
    )
    self = SimpleNamespace(**common)
    state = OnPolicyWorkflow.setup(self, jax.random.key(0))
    assert state.key.sharding.spec == jax.P("DP")
    assert state.env_state.obs.shape == (2, 3)
    assert state.agent_state.params.w.shape == (3,)
    print("on-policy setup PASS", flush=True)
    self = SimpleNamespace(
        **common,
        _setup_replaybuffer=lambda key: PyTreeDict(
            data=jnp.zeros((4, 3)), count=jnp.array(0)
        ),
        _postsetup_replaybuffer=lambda state: state.replace(
            replay_buffer_state=state.replay_buffer_state.replace(
                count=state.replay_buffer_state.count + 1
            ),
            metrics=state.metrics.replace(
                sampled_timesteps=jax.lax.psum(jnp.uint32(1), "DP")
            ),
        ),
    )
    state = OffPolicyWorkflow.setup(self, jax.random.key(1))
    np.testing.assert_array_equal(state.replay_buffer_state.count, np.ones(2))
    assert state.replay_buffer_state.data.shape == (2, 4, 3)
    assert state.agent_state.params.w.shape == (3,)
    assert state.metrics.sampled_timesteps.shape == ()
    assert int(state.metrics.sampled_timesteps) == 2
    print("off-policy setup PASS", flush=True)
    self = SimpleNamespace(**common, _postsetup=lambda state: state)
    state = ECWorkflowTemplate.setup(self, jax.random.key(2))
    assert state.key.sharding.spec == jax.P("DP")
    np.testing.assert_array_equal(state.distributed_info.rank, [0, 1])
    print("EC setup PASS", flush=True)
    self = SimpleNamespace(
        enable_multi_devices=True,
        mesh=dp_mesh,
        dp_axis_name="DP",
        evaluator=SimpleNamespace(env=env),
        agent=SimpleNamespace(
            init=lambda *args: AgentState(params=PyTreeDict(w=jnp.ones(3)))
        ),
    )
    state = RandomAgentWorkflow.setup(self, jax.random.key(3))
    assert state.key.sharding.spec == jax.P("DP")
    print("random-agent setup PASS", flush=True)

    class Target:
        def setup(self, key):
            return State(
                key=key,
                agent_state=AgentState(
                    params=PyTreeDict(w=jax.random.normal(key, (3,)))
                ),
                metrics=WorkflowMetric(),
            )

        def step(self, state):
            return PyTreeDict(loss=state.agent_state.params.w.sum()), state.replace(
                agent_state=state.agent_state.replace(
                    params=PyTreeDict(w=state.agent_state.params.w + 1)
                ),
                metrics=state.metrics.replace(
                    sampled_timesteps=state.metrics.sampled_timesteps + 1
                ),
            )

        def evaluate(self, state):
            return EvaluateMetric(
                episode_returns=state.agent_state.params.w.sum(),
                episode_lengths=jnp.float32(1),
            ), state

    class Harness(PBTWorkflowTemplate):
        def _customize_optimizer(self):
            pass

        def _setup_pop_and_pbt_optimizer(self, key):
            return PyTreeDict(
                actor_loss_weight=jnp.arange(4, dtype=jnp.float32) + 1
            ), PBTOptState()

        def apply_hyperparams_to_workflow_state(self, state, hp):
            return state.replace(
                agent_state=state.agent_state.replace(
                    params=PyTreeDict(
                        w=state.agent_state.params.w + 0 * hp.actor_loss_weight
                    )
                )
            )

    class CSOHarness(PBTCSOParamPPOWorkflow):
        _customize_optimizer = Harness._customize_optimizer
        apply_hyperparams_to_workflow_state = (
            Harness.apply_hyperparams_to_workflow_state
        )

    cfg = OmegaConf.create(
        dict(
            checkpoint=dict(enable=False),
            pop_size=4,
            workflow_steps_per_iter=1,
            warmup_steps=0,
            bottom_ratio=0.25,
            top_ratio=0.25,
            perturb_factor=dict(actor_loss_weight=0.1),
            search_space=dict(actor_loss_weight=dict(low=0.1, high=5)),
            eval_episodes=1,
        )
    )
    mesh = jax.make_mesh((2,), ("POP",), axis_types=(jax.sharding.AxisType.Explicit,))

    class Buffer:
        def add(self, state, trajectory):
            return state.replace(
                total=state.total + trajectory.obs.sum(),
                buffer_size=state.buffer_size + trajectory.obs.shape[0],
            )

    class OffTarget(Target):
        replay_buffer = Buffer()

        def setup(self, key):
            return super().setup(key).replace(replay_buffer_state=None)

        def _setup_replaybuffer(self, key):
            return PyTreeDict(total=jnp.float32(0), buffer_size=jnp.int32(0))

        def step(self, state):
            metrics, state = super().step(state)
            return metrics.replace(
                trajectory=SampleBatch(obs=state.agent_state.params.w[None])
            ), state

    class OffHarness(PBTOffpolicyWorkflowTemplate):
        _customize_optimizer = Harness._customize_optimizer
        _setup_pop_and_pbt_optimizer = Harness._setup_pop_and_pbt_optimizer
        apply_hyperparams_to_workflow_state = (
            Harness.apply_hyperparams_to_workflow_state
        )

        def _postsetup_replaybuffer(self, state):
            return state

    evaluator = SimpleNamespace(
        evaluate=lambda agent_state, key, num_episodes: EvaluateMetric(
            episode_returns=agent_state.params.w.sum()[None],
            episode_lengths=jnp.ones(1),
        )
    )
    for cls in (Harness, CSOHarness, OffHarness):
        wf = cls(OffTarget() if cls is OffHarness else Target(), evaluator, cfg)
        wf.mesh = mesh
        jax.set_mesh(wf.mesh)
        state = jax.jit(wf.setup)(jax.random.key(7))
        for _ in range(2):
            metrics, state = jax.jit(wf.step)(state)
        metrics, state = jax.jit(wf.evaluate)(state)
        assert state.pop.actor_loss_weight.sharding.spec == jax.P("POP")
        assert state.pop_workflow_state.agent_state.params.w.sharding.spec == jax.P(
            "POP", None
        )
        assert metrics.pop_episode_returns.shape == (4,)
        jax.block_until_ready(state)
        if cls is OffHarness:
            assert int(state.replay_buffer_state.buffer_size) == 8
            assert state.replay_buffer_state.total.sharding.spec == jax.P()

        # A one-device run is the numerical reference for distributed selection,
        # mutation, state transfer and shared-buffer accumulation.
        reference = cls(OffTarget() if cls is OffHarness else Target(), evaluator, cfg)
        reference.mesh = jax.make_mesh(
            (1,),
            ("POP",),
            axis_types=(jax.sharding.AxisType.Explicit,),
            devices=jax.devices()[:1],
        )
        with jax.set_mesh(reference.mesh):
            expected = jax.jit(reference.setup)(jax.random.key(7))
            for _ in range(2):
                _, expected = jax.jit(reference.step)(expected)
            _, expected = jax.jit(reference.evaluate)(expected)
        assert jax.tree.structure(state) == jax.tree.structure(expected)
        for actual_leaf, expected_leaf in zip(
            jax.tree.leaves(state), jax.tree.leaves(expected), strict=True
        ):
            if jax.dtypes.issubdtype(actual_leaf.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(
                    jax.random.key_data(actual_leaf), jax.random.key_data(expected_leaf)
                )
            else:
                np.testing.assert_allclose(
                    actual_leaf, expected_leaf, rtol=1e-5, atol=1e-6
                )
        print(cls.__name__, "one-/two-device equivalence PASS", flush=True)

    source = jax.device_put(jnp.array([10.0, 20.0, 30.0, 40.0]), jax.P("POP"))
    selected = tree_get(source, jnp.array([3, 0, 3]), out_sharding=jax.P())
    np.testing.assert_array_equal(selected, [40, 10, 40])
    result = jax.jit(
        lambda x: tree_scatter_set(
            x,
            jnp.array([100.0, 400.0]),
            jnp.array([1, 3]),
            jax.P("POP"),
        )
    )(source)
    np.testing.assert_array_equal(result, [10, 100, 30, 400])
    print("cross-device parent gather/scatter PASS", flush=True)

    from evorl.distributed import shmap_vmap
    from evorl.utils import running_statistics

    norm = running_statistics.init_state(jnp.zeros(3))
    norm = jax.tree.map(lambda x: jnp.broadcast_to(x, (4, *x.shape)), norm)
    norm = jax.device_put(norm, jax.P("POP"))
    data = jax.device_put(
        jnp.arange(18, dtype=jnp.float32).reshape(6, 3),
        jax.P(),
    )
    update = shmap_vmap(
        running_statistics.update,
        in_specs=(jax.P("POP"), jax.P()),
        out_specs=jax.P("POP"),
        vmap_in_axes=(0, None),
    )
    norm = jax.jit(update)(norm, data)
    np.testing.assert_allclose(norm.mean, np.tile([7.5, 8.5, 9.5], (4, 1)))
    np.testing.assert_array_equal(norm.count, np.full(4, 6))
    print("shared-data normalization PASS", flush=True)


if __name__ == "__main__":
    main()
