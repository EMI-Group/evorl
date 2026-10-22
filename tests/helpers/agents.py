"""Build small agent fixtures using the same YAML defaults as workflows."""

from dataclasses import fields
from pathlib import Path

from omegaconf import OmegaConf

from evorl.algorithms.sac import SACDiscreteAgent
from evorl.envs import Box, Discrete
from evorl.utils import running_statistics

_CONFIGS = {
    "TD3Agent": "td3",
    "TD7Agent": "td7",
    "DDPGAgent": "ddpg",
    "DQNAgent": "dqn",
    "SACAgent": "sac",
    "SACDiscreteAgent": "sac",
    "PPOAgent": "ppo",
    "A2CAgent": "a2c",
    "IMPALAAgent": "impala",
    "ParamPPOAgent": "meta/paramppo",
    "ParamSACAgent": "meta/paramsac",
    "DeterministicECAgent": "ec/openes",
    "StochasticECAgent": "ec/openes",
}


def build_agent(agent_cls, obs_space, action_space, **overrides):
    """Load production defaults and apply the dimensions needed by a test."""
    root = Path(__file__).resolve().parents[2] / "configs"
    cfg = OmegaConf.merge(
        OmegaConf.load(root / "config.yaml"),
        OmegaConf.load(root / f"agent/{_CONFIGS[agent_cls.__name__]}.yaml"),
    )
    network_cfg = overrides.pop("agent_network_cfg", cfg.agent_network)
    aliases = {"z_s_dim": "zs_dim", "z_sa_dim": "zsa_dim"}
    for key in list(overrides):
        name = aliases.get(key, key)
        if name in network_cfg or key in aliases:
            network_cfg[name] = overrides.pop(key)
    normalize = overrides.pop("normalize_obs", cfg.get("normalize_obs", False))
    if agent_cls.__name__ == "SACAgent" and isinstance(action_space, Discrete):
        agent_cls = SACDiscreteAgent
    kwargs = {f.name: cfg[f.name] for f in fields(agent_cls) if f.name in cfg}
    kwargs.update(
        agent_network_cfg=network_cfg,
        obs_preprocessor=running_statistics.normalize if normalize else None,
    )
    names = {f.name for f in fields(agent_cls)}
    if "continuous_action" in names:
        kwargs["continuous_action"] = isinstance(action_space, Box)
    if "init_alpha" in names:
        kwargs["init_alpha"] = cfg.get("alpha", 1.0)
    kwargs.update(overrides)
    return agent_cls(**kwargs)
