import jax
import chex

from evorl.evaluator import Evaluator
from evorl.agents.random_agent import DebugRandomAgent
from evorl.envs import create_wrapped_brax_env, AutoresetMode


def test_eval_rollout_epsiode():
    env_name = "hopper"

    env = create_wrapped_brax_env(
        env_name,
        parallel=7,
        autoreset_mode=AutoresetMode.NORMAL,
    )

    agent = DebugRandomAgent()

    evaluator = Evaluator(env, agent, 1000, discount=0.99)

    key = jax.random.PRNGKey(42)

    key, rollout_key, env_key, agent_key = jax.random.split(key, 4)

    agent_state = agent.init(env.obs_space, env.action_space, agent_key)

    metric = evaluator.evaluate(agent_state, 7 * 3, rollout_key)

    assert metric.episode_returns.shape == (7 * 3,)
    assert metric.episode_lengths.shape == (7 * 3,)


def test_fast_eval_rollout_epsiode():
    env_name = "hopper"

    env = create_wrapped_brax_env(
        env_name,
        parallel=7,
        autoreset_mode=AutoresetMode.NORMAL,
    )

    agent = DebugRandomAgent()

    evaluator = Evaluator(env, agent, 1000)

    key = jax.random.PRNGKey(42)

    key, rollout_key, env_key, agent_key = jax.random.split(key, 4)

    agent_state = agent.init(env.obs_space, env.action_space, agent_key)

    metric = evaluator.evaluate(agent_state, 7 * 3, rollout_key)

    assert metric.episode_returns.shape == (7 * 3,)
    assert metric.episode_lengths.shape == (7 * 3,)
