import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig
from evox import Stateful, State

from evorl.agents import Agent
from evorl.envs import Env
from evorl.types import RolloutMetric

"""
    Single Agent Off-Policy Reinforcement Learning Workflow
"""


class OnPolicyRLWorkflow(Stateful):
    def __init__(
        self,
        config: DictConfig,
        env: Env,
        agent: Agent,
        optimizer: optax.GradientTransformation,
    ):
        super(OnPolicyRLWorkflow, self).__init__()
        self.config = config
        self.agent = agent
        self.env = env # batched env
        self.optimizer = optimizer
        


    def setup(self, key):
        key, agent_key, env_key = jax.random.split(key, 3)
        agent_state = self.agent.init(agent_key)
        return State(
            key=key,
            rollout_metric = RolloutMetric(),
            agent_state=agent_state,
            env_state=self.env.reset(env_key),
            opt_state=self.optimizer.init(agent_state.params)
        )

    def step(self, state):
        return state.update()
