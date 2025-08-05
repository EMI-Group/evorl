from evorl.ec.operators import mlp_crossover, mlp_mutate, MLPCrossover, MLPMutation
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from evorl.networks import make_policy_network


def test_mutation():
    action_size, obs_size = 3, 5
    model = make_policy_network(action_size)

    def init_fn(key):
        return model.init(key, jnp.ones((1, obs_size)))

    key1, key2, key3, key4, key5 = jax.random.split(jax.random.PRNGKey(0), num=5)

    state1 = init_fn(key1)
    state2 = init_fn(key2)
    state3 = init_fn(key3)
    state4 = init_fn(key4)
    state = jtu.tree_map(lambda *x: jnp.stack(x), state1, state2, state3, state4)

    mlp_mutate(state1, key5)

    jax.jit(
        mlp_mutate,
        static_argnames=(
            "weight_max_magnitude",
            "mut_strength",
            "vector_num_mutation_frac",
            "matrix_num_mutation_frac",
        ),
    )(state1, key5)

    MLPMutation()(state, key5)


def test_crossover():
    action_size, obs_size = 3, 5
    model = make_policy_network(action_size)

    def init_fn(key):
        return model.init(key, jnp.ones((1, obs_size)))

    key1, key2, key3, key4, key5 = jax.random.split(jax.random.PRNGKey(0), num=5)

    state1 = init_fn(key1)
    state2 = init_fn(key2)
    state3 = init_fn(key3)
    state4 = init_fn(key4)
    state = jtu.tree_map(lambda *x: jnp.stack(x), state1, state2, state3, state4)

    mlp_crossover(state1, state2, key5)
    jax.jit(mlp_crossover, static_argnames=("num_crossover_frac"))(state1, state2, key5)

    MLPCrossover()(state, key5)
