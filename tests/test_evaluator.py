import jax
import chex

from evorl.evaluator import Evaluator
from evorl.agents.random_agent import RandomAgent
from evorl.envs import create_brax_env


def test_eval_rollout_epsiode():
    env_name = "hopper"

    env = create_brax_env(
        env_name,
        parallel=7,
        autoreset=False,
    )

    agent = RandomAgent(action_space=env.action_space, obs_space=env.obs_space)
    evaluator = Evaluator(env, agent, 1000)

    key = jax.random.PRNGKey(42)

    key, rollout_key, env_key, agent_key = jax.random.split(key, 4)

    agent_state = agent.init(agent_key)

    metric = evaluator.evaluate(agent_state, 7 * 3, rollout_key)
