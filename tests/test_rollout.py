import jax

from evorl.rollout import (
    rollout,
)

from .utils import DebugRandomAgent
from evorl.envs import create_env


def test_rollout():
    env = create_env("ant", "brax", parallel=7)

    agent = DebugRandomAgent()

    key = jax.random.PRNGKey(42)

    rollout_key, env_key, agent_key = jax.random.split(key, 3)

    env_state = env.reset(env_key)

    agent_state = agent.init(env.obs_space, env.action_space, agent_key)

    env_extra_fields = (
        "termination",
        "truncation",
        "last_obs",
        "steps",
        "episode_return",
    )

    trajectory, env_nstate = rollout(
        env.step,
        agent.compute_actions,
        env_state,
        agent_state,
        rollout_key,
        rollout_length=1000,
        env_extra_fields=env_extra_fields,
    )

    env_extras = trajectory.extras["env_extras"]
    for key in env_extra_fields:
        assert key in env_extras, f"{key} not in rollout trjectory"
