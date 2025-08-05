import os
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from evorl.algorithms.impala import compute_vtrace, make_mlp_impala_agent
from evorl.rollout import rollout
from evorl.envs import create_wrapped_brax_env, AutoresetMode
from evorl.utils.rl_toolkits import compute_gae
from evorl.distribution import get_categorical_dist, get_tanh_norm_dist


def setup_trajectory():
    os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
    env = create_wrapped_brax_env(
        "ant", parallel=5, autoreset_mode=AutoresetMode.ENVPOOL
    )
    agent = make_mlp_impala_agent(action_space=env.action_space)

    key, rollout_key = jax.random.split(jax.random.PRNGKey(42))

    env_state = env.reset(key)
    agent_state = agent.init(env.obs_space, env.action_space, key)

    trajectory, env_state = rollout(
        env.step,
        agent.compute_actions,
        env_state,
        agent_state,
        rollout_key,
        rollout_length=1000,
        env_extra_fields=("autoreset", "episode_return", "termination"),
    )

    # _obs = jnp.concatenate([trajectory.obs, trajectory.next_obs[-1:]], axis=0)
    _obs = jtu.tree_map(
        lambda obs, next_obs: jnp.concatenate([obs, next_obs[-1:]], axis=0),
        trajectory.obs,
        trajectory.next_obs,
    )

    vs = agent.value_network.apply(agent_state.params.value_params, _obs)

    sampled_actions_logp = trajectory.extras.policy_extras.logp
    sampled_actions = trajectory.actions

    # [T, B, A]
    raw_actions = agent.policy_network.apply(
        agent_state.params.policy_params, trajectory.obs
    )

    if agent.continuous_action:
        actions_dist = get_tanh_norm_dist(*jnp.split(raw_actions, 2, axis=-1))
    else:
        actions_dist = get_categorical_dist(raw_actions)

    # [T, B]
    actions_logp = actions_dist.log_prob(sampled_actions)
    rho = jnp.exp(actions_logp - sampled_actions_logp)

    return trajectory, vs, rho


def test_vtrace():
    trajectory, vs, rho = setup_trajectory()

    lambda_ = 0.95
    discount = 0.99

    # Note: JAX cannot ensure rho=1, so we manually set it
    rho = jnp.ones_like(rho)

    vtrace = compute_vtrace(
        rho_t=rho,
        v_t=vs[:-1],
        v_t_plus_1=vs[1:],
        rewards=trajectory.rewards,
        dones=trajectory.dones,
        discount=discount,
        lambda_=lambda_,
        clip_c_threshold=1.0,
        clip_rho_threshold=1.0,
    )

    gae_v, gae_adv = compute_gae(
        rewards=trajectory.rewards,
        values=vs,
        dones=trajectory.dones,
        terminations=trajectory.extras.env_extras.termination,
        gae_lambda=lambda_,
        discount=discount,
    )

    assert jnp.allclose(vtrace, gae_v, rtol=0, atol=1e-4), f"{vtrace} != {gae_v}"
    assert jnp.allclose(vtrace - vs[:-1], gae_adv, rtol=0, atol=1e-4).all(), (
        f"{vtrace - vs[:-1]} != {gae_adv}"
    )
