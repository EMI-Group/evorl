import jax
import jax.numpy as jnp

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from . import GeneralRLProblem, EvoXAlgorithmWrapper

from evorl.utils.ec_utils import ParamVectorSpec
from evorl.workflows import ECWorkflow
from evorl.envs import create_wrapped_brax_env
from evorl.agents.ec import DeterministicECAgent
from evox import algorithms




def train():
    with initialize(config_path="../../configs", version_base=None):
        config = compose(config_name="config", overrides=["agent=ec-so", "env=brax/hopper"])
    print(OmegaConf.to_yaml(config))

    key = jax.random.PRNGKey(config.seed)

    agent_key, workflow_key = jax.random.split(key)

    env = create_wrapped_brax_env(
        config.env.env_name,
        episode_length=config.env.max_episode_steps,
        parallel=config.parallel_envs,
        autoreset=False,
    )

    agent = DeterministicECAgent(
        action_space=env.action_space,
        obs_space=env.obs_space,
        actor_hidden_layer_sizes=config.agent_network.actor_hidden_layer_sizes,  # use linear model
        normalize_obs=False
    )

    problem = GeneralRLProblem(
        agent=agent,
        env=env,
        num_episodes=config.eval_episodes,
        max_episode_steps=config.env.max_episode_steps,
        discount=1.0,
    )

    # dummy_agent
    agent_state = agent.init(agent_key)
    param_vec_spec = ParamVectorSpec(agent_state.params.policy_params)

    cmaes = algorithms.CMAES(
        center_init=param_vec_spec.to_vector(agent_state.params.policy_params), 
        init_stdev=config.init_stdev,
        pop_size=config.pop_size,
    )

    # TODO: add op for MLP
    cmaes_rl = EvoXAlgorithmWrapper(
        algo=cmaes,
        param_vec_spec=param_vec_spec
    )

    def _sol_transform(cand):
        params = agent_state.params.replace(policy_params=cand)
        return agent_state.replace(params=params)


    workflow = ECWorkflow(
        algorithm=cmaes_rl,
        problem=problem,
        opt_direction='max',
        sol_transforms=[jax.vmap(_sol_transform)],
    )

    jnp.set_printoptions(precision=4, suppress=True)

    state = workflow.init(workflow_key)
    for i in range(config.num_iters):
        train_metrics, state = workflow.step(state)
        jax.block_until_ready(train_metrics)
        print(f"iteration: {i}")
        objective = train_metrics.objective

        print(jax.lax.top_k(objective, k=5)[0].tolist())


if __name__ == "__main__":
    train()
