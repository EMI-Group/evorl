import jax
import jax.numpy as jnp
import chex
from evorl.workflows import ECWorkflow

from evox import algorithms, problems, monitors


def test_ec_workflow():
    pso = algorithms.PSO(
        lb=jnp.full(shape=(2,), fill_value=-32),
        ub=jnp.full(shape=(2,), fill_value=32),
        pop_size=100,
    )
    ackley = problems.numerical.Ackley()
    # monitor = monitors.EvalMonitor()

    workflow = ECWorkflow(
        algorithm=pso,
        problem=ackley,
    )

    key = jax.random.PRNGKey(42)
    state = workflow.init(key)

    for i in range(100):
        state = workflow.step(state)