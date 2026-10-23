"""MAPPO: decentralized PPO actors with agent-specific centralized values."""

from .ippo import IPPOWorkflow


class MAPPOWorkflow(IPPOWorkflow):
    """Use native global state plus local observations for each value function.

    Without native global state, concatenate the agents' observations. Actor
    inputs stay local. This is the separate-policy, feedforward MAPPO variant;
    network parameters and optimizer state are independent across agents.
    """

    centralized_critic = True

    @classmethod
    def name(cls):
        return "MAPPO"
