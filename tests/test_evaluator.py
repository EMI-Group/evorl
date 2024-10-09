import jax
import jax.numpy as jnp
import chex

from evorl.evaluator import Evaluator
from evorl.envs import create_wrapped_brax_env, AutoresetMode
from evorl.utils.rl_toolkits import compute_discount_return, compute_episode_length
from evorl.utils.jax_utils import rng_split_by_shape
from evorl.types import PyTreeDict

from .utils import DebugRandomAgent, FakeVmapEnv


def test_eval_rollout_epsiode():
    env_name = "hopper"

    env = create_wrapped_brax_env(
        env_name,
        parallel=7,
        autoreset_mode=AutoresetMode.DISABLED,
    )

    agent = DebugRandomAgent()

    evaluator = Evaluator(env, agent.evaluate_actions, 1000, discount=0.99)

    key = jax.random.PRNGKey(42)

    key, rollout_key, agent_key = jax.random.split(key, 3)

    agent_state = agent.init(env.obs_space, env.action_space, agent_key)

    metric = evaluator.evaluate(agent_state, rollout_key, 7 * 3)

    assert metric.episode_returns.shape == (7 * 3,)
    assert metric.episode_lengths.shape == (7 * 3,)


def test_batched_eval_rollout_epsiode():
    env_name = "hopper"

    env = create_wrapped_brax_env(
        env_name,
        parallel=7,
        autoreset_mode=AutoresetMode.DISABLED,
    )
    agent = DebugRandomAgent()
    evaluator = Evaluator(env, agent.evaluate_actions, 1000, discount=0.99)

    num_agents = 5
    key = jax.random.PRNGKey(42)
    key, rollout_key, agent_key = jax.random.split(key, 3)

    agent_keys = jax.random.split(agent_key, num_agents)
    rollout_keys = jax.random.split(rollout_key, num_agents)
    agent_init = jax.vmap(agent.init, in_axes=(None, None, 0))
    agent_state = agent_init(env.obs_space, env.action_space, agent_keys)

    metric = evaluator.evaluate(agent_state, rollout_keys, 7 * 3)

    assert metric.episode_returns.shape == (num_agents, 7 * 3)
    assert metric.episode_lengths.shape == (num_agents, 7 * 3)


def test_multi_batched_eval_rollout_epsiode():
    env_name = "hopper"

    env = create_wrapped_brax_env(
        env_name,
        parallel=7,
        autoreset_mode=AutoresetMode.DISABLED,
    )

    agent = DebugRandomAgent()

    evaluator = Evaluator(env, agent.evaluate_actions, 1000, discount=0.99)

    key = jax.random.PRNGKey(42)
    key, rollout_key, agent_key = jax.random.split(key, 3)
    num_agents = (11, 13)

    agent_keys = rng_split_by_shape(agent_key, num_agents)
    rollout_keys = rng_split_by_shape(rollout_key, num_agents)

    agent_init = jax.vmap(agent.init, in_axes=(None, None, 0))
    agent_init = jax.vmap(agent_init, in_axes=(None, None, 0))
    agent_state = agent_init(env.obs_space, env.action_space, agent_keys)

    metric = evaluator.evaluate(agent_state, rollout_keys, 7 * 3)

    assert metric.episode_returns.shape == (num_agents) + (7 * 3,)
    assert metric.episode_lengths.shape == (num_agents) + (7 * 3,)


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
        """
        sample_batch: one-step obs
        transition: one-step full info
        """
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


def test_fast_eval_rollout_epsiode():
    # due to impl, the two methods could generate different trajectories,
    # so we disable this test
    # for env_name in ["hopper", "ant", "walker2d", "halfcheetah"]:
    #     print("Testing", env_name)
    #     env = create_wrapped_brax_env(
    #         env_name,
    #         parallel=7,
    #         autoreset_mode=AutoresetMode.DISABLED,
    #     )
    # for env_name in ["Pendulum-v1", "Breakout-MinAtar", "Swimmer-misc", "Acrobot-v1"]:
    #     print("Testing", env_name)
    #     env = create_wrapped_gymnax_env(
    #         env_name,
    #         parallel=7,
    #         autoreset_mode=AutoresetMode.DISABLED,
    #     )
    env1 = FakeVmapEnv(*_setup_trajectory1(17))
    env2 = FakeVmapEnv(*_setup_trajectory2(13))
    env3 = FakeVmapEnv(*_setup_trajectory3(1000))

    test_cases = {
        env1: (17, 10, 100),
        env2: (13, 7, 100),
        env3: (1000, 900, 500),
    }

    for env, max_length_list in test_cases.items():
        agent = DebugRandomAgent()

        for max_length in max_length_list:
            evaluator = Evaluator(env, agent.evaluate_actions, max_length)

            key = jax.random.PRNGKey(42)

            jit_eval = jax.jit(evaluator._evaluate, static_argnums=(1,))
            jit_fast_eval = jax.jit(evaluator._fast_evaluate, static_argnums=(1,))

            num_iters = 7

            for _ in range(10):
                key, rollout_key, agent_key = jax.random.split(key, 3)

                agent_state = agent.init(env.obs_space, env.action_space, agent_key)

                metric = jit_eval(agent_state, num_iters, rollout_key)
                fast_metric = jit_fast_eval(agent_state, num_iters, rollout_key)

                print(metric)
                print(fast_metric)

                print(
                    jnp.max(
                        jnp.abs(metric.episode_returns - fast_metric.episode_returns)
                    )
                )
                print("+" * 20)

                chex.assert_trees_all_close(metric, fast_metric, rtol=1e-05)
