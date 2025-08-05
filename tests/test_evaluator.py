import jax
import jax.numpy as jnp
import chex

from evorl.evaluators import Evaluator
from evorl.envs import create_wrapped_brax_env, AutoresetMode
from evorl.utils.rl_toolkits import compute_discount_return, compute_episode_length
from evorl.types import PyTreeDict

from .utils import DebugRandomAgent


def test_eval_rollout_epsiode():
    env_name = "hopper"

    env = create_wrapped_brax_env(
        env_name,
        parallel=7,
        autoreset_mode=AutoresetMode.DISABLED,
    )

    agent = DebugRandomAgent()

    for discount in [0.99, 1.0]:
        evaluator = Evaluator(
            env=env,
            action_fn=agent.evaluate_actions,
            max_episode_steps=1000,
            discount=discount,
        )

        key = jax.random.PRNGKey(42)

        key, rollout_key, agent_key = jax.random.split(key, 3)

        agent_state = agent.init(env.obs_space, env.action_space, agent_key)

        metric = evaluator.evaluate(agent_state, rollout_key, 7 * 3)

        assert metric.episode_returns.shape == (7 * 3,)
        assert metric.episode_lengths.shape == (7 * 3,)


def _normal_eval(rewards, dones, max_length):
    rewards = rewards[:max_length]
    dones = dones[:max_length]
    ep_return = compute_discount_return(rewards, dones)
    ep_len = compute_episode_length(dones)
    return ep_return, ep_len


def _fast_eval(rewards, dones, rollout_length):
    def _terminate_cond(carry):
        env_state, prev_metrics = carry
        return (prev_metrics.episode_lengths < rollout_length).all() & (
            ~env_state.done.all()
        )

    def _one_step_rollout(carry):
        env_state, prev_metrics = carry

        prev_dones = env_state.done

        reward = rewards[env_state.i]
        done = dones[env_state.i]

        env_nstate = env_state.replace(
            done=done,
            i=env_state.i + 1,
        )

        metrics = PyTreeDict(
            episode_returns=prev_metrics.episode_returns + (1 - prev_dones) * reward,
            episode_lengths=prev_metrics.episode_lengths
            + (1 - prev_dones).astype(
                jnp.int32
            ),  # this will miss counting the last step
        )

        return env_nstate, metrics

    batch_shape = rewards.shape[1]

    env_state = PyTreeDict(
        done=jnp.zeros(batch_shape),
        i=0,
    )

    env_state, metrics = jax.lax.while_loop(
        _terminate_cond,
        _one_step_rollout,
        (
            env_state,
            PyTreeDict(
                episode_returns=jnp.zeros(batch_shape),
                episode_lengths=jnp.zeros(batch_shape, dtype=jnp.int32),
            ),
        ),
    )

    return metrics.episode_returns, metrics.episode_lengths


def _setup_trajectory1(max_length=17):
    num_envs = 4

    dones = jnp.zeros((num_envs, max_length))
    dones = dones.at[:, -2:].set(1)
    dones = dones.T

    rewards = jnp.arange(num_envs * max_length, dtype=jnp.float32).reshape(
        (max_length, num_envs)
    )

    return rewards, dones


def _setup_trajectory2(max_length=13):
    num_envs = 4

    dones = jnp.zeros((num_envs, max_length))
    dones = dones.at[0, -2:].set(1)
    dones = dones.at[1, -3:].set(1)
    dones = dones.at[2, -3:].set(1)
    dones = dones.at[3, -4:].set(1)
    dones = dones.T

    rewards = jnp.arange(num_envs * max_length, dtype=jnp.float32).reshape(
        (max_length, num_envs)
    )

    return rewards, dones


def _setup_trajectory3(max_length=1000):
    num_envs = 4

    dones = jnp.zeros((num_envs, max_length))
    dones = dones.at[0, -1:].set(1)
    dones = dones.at[1, -3:].set(1)
    dones = dones.at[2, -3:].set(1)
    dones = dones.at[3, -200:].set(1)
    dones = dones.T

    key = jax.random.PRNGKey(42)
    rewards = jax.random.normal(key, (max_length, num_envs))

    return rewards, dones


def test_fast_evaluation():
    def _test(rewards, dones, max_length):
        print("rewards", rewards)
        print("dones", dones)

        ep_return, ep_len = _normal_eval(rewards, dones, max_length)

        fast_ep_return, fast_ep_len = _fast_eval(rewards, dones, max_length)

        print("ep_return", ep_return)
        print("ep_len", ep_len)
        print("fast_ep_return", fast_ep_return)
        print("fast_ep_len", fast_ep_len)
        print("max diff", jnp.max(jnp.abs(ep_return - fast_ep_return)))
        print("+" * 20)

        chex.assert_trees_all_close(ep_return, fast_ep_return, rtol=1e-05)

    rewards, dones = _setup_trajectory1(17)
    _test(rewards, dones, 17)
    _test(rewards, dones, 10)

    rewards, dones = _setup_trajectory2(13)
    _test(rewards, dones, 13)
    _test(rewards, dones, 7)

    rewards, dones = _setup_trajectory3(1000)
    _test(rewards, dones, 1000)
    _test(rewards, dones, 900)
    _test(rewards, dones, 500)
