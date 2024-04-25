import math
from functools import partial
from evorl.ec.operations import mlp_crossover, mlp_mutate, MLPCrossover, MLPMutation
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from evorl.networks import make_policy_network



def test_mutation():
    model, init_fn = make_policy_network(3, 5)

    key1, key2, key3, key4, key5 = jax.random.split(jax.random.PRNGKey(0), num=5)

    state1 = init_fn(key1)
    state2 = init_fn(key2)
    state3 = init_fn(key3)
    state4 = init_fn(key4)
    state=jtu.tree_map(lambda *x: jnp.stack(x), state1, state2, state3, state4)

    mlp_mutate(key5, state1)
    jax.jit(mlp_mutate, static_argnames=('weight_max_magnitude', 'mut_strength', 'num_mutation_frac',
        'super_mut_strength', 'super_mut_prob', 'reset_prob', 'vec_relative_prob'))(key5, state1)
    
    MLPMutation()(key5, state)


def test_crossover():
    model, init_fn = make_policy_network(3, 5)

    key1, key2, key3, key4, key5 = jax.random.split(jax.random.PRNGKey(0), num=5)

    state1 = init_fn(key1)
    state2 = init_fn(key2)
    state3 = init_fn(key3)
    state4 = init_fn(key4)
    state=jtu.tree_map(lambda *x: jnp.stack(x), state1, state2, state3, state4)

    mlp_crossover(key5, state1, state2)
    jax.jit(mlp_crossover)(key5, state1, state2)

    MLPCrossover()(key5, state)