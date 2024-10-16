import jax
from evorl.algorithms.ec.so.evox_cmaes import CMAESWorkflow
from hydra import compose, initialize


def setup_workflow():
    with initialize(config_path="../configs"):
        cfg = compose(
            config_name="config", overrides=["agent=ec/cmaes", "env=brax/ant"]
        )

    cfg.num_iters = 10

    workflow = CMAESWorkflow.build_from_config(cfg, enable_jit=True)

    return workflow


def test_ec_workflow():
    workflow = setup_workflow()

    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    for i in range(100):
        state = workflow.step(state)


def test_ec_workflow_learn():
    workflow = setup_workflow()

    key = jax.random.PRNGKey(42)
    state = workflow.init(key)
    state = workflow.learn(state)
