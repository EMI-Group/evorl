import copy
import pandas as pd

import chex
import jax
import jax.numpy as jnp
from optax.schedules import InjectStatefulHyperparamsState


def convert_pop_to_df(pop):
    df = pd.DataFrame.from_dict(pop)
    df.insert(0, "pop_id", range(len(df)))
    return df


def deepcopy_opt_state(state: InjectStatefulHyperparamsState):
    assert isinstance(state, InjectStatefulHyperparamsState)

    return InjectStatefulHyperparamsState(
        count=state.count,
        hyperparams=copy.deepcopy(state.hyperparams),
        hyperparams_states=state.hyperparams_states,
        inner_state=state.inner_state,
    )


def uniform_init(search_space, key: chex.PRNGKey, num: int) -> chex.Array:
    assert search_space.low <= search_space.high
    return jax.random.uniform(
        key,
        (num,),
        minval=search_space.low,
        maxval=search_space.high,
    )


def truncated_normal_init(
    search_space, key: chex.PRNGKey, num: int, m=0.95
) -> chex.Array:
    assert search_space.low <= search_space.high

    # Note: 1.96 is the z-score for 95% confidence interval,
    # meaning a value sampled from this distribution will be within

    z = jax.scipy.stats.norm.ppf(1 - (1 - m) / 2)

    mu = (search_space.high + search_space.low) / 2

    std = (search_space.high - search_space.low) / (2 * z)

    samples = jax.random.truncated_normal(
        key,
        lower=(search_space.low - mu) / std,
        upper=(search_space.high - mu) / std,
        shape=(num,),
    )
    samples = samples * std + mu

    return


def log_uniform_init(search_space, key: chex.PRNGKey, num: int) -> chex.Array:
    """Suitable for hyperparameters that need explore different magnitudes. eg: [1e-3, 100]."""
    assert (
        search_space.low > 0
        and search_space.high > 0
        and search_space.low <= search_space.high
    )

    return jnp.exp(
        jax.random.uniform(
            key,
            (num,),
            minval=jnp.log(search_space.low),
            maxval=jnp.log(search_space.high),
        )
    )


def exp_uniform_init(search_space, key: chex.PRNGKey, num: int) -> chex.Array:
    """exp(-x)"""
    assert (
        search_space.low > 0
        and search_space.high > 0
        and search_space.low <= search_space.high
    )

    return jnp.exp(
        -jax.random.uniform(
            key,
            (num,),
            minval=search_space.low,
            maxval=search_space.high,
        )
    )
