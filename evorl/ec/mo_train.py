import jax
import jax.numpy as jnp

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from evorl.ec import MOAlgorithmWrapper, MultiObjectiveBraxProblem

from evorl.utils.ec_utils import ParamVectorSpec
from evorl.workflows import ECWorkflow
from evorl.envs import create_wrapped_brax_env
from evorl.agents.ec import DeterministicECAgent
from evox import algorithms, monitors
from evox.operators import non_dominated_sort




def train():
    with initialize(config_path="../../configs", version_base=None):
        config = compose(config_name="config", overrides=["agent=ec", "env=brax/hopper"])
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

    problem = MultiObjectiveBraxProblem(
        agent=agent,
        env=env,
        num_episodes=config.eval_episodes,
        max_episode_steps=config.env.max_episode_steps,
        discount=1.0,
        metric_names=config.metric_names,
        flatten_objectives=True
    )

    # dummy_agent
    agent_state = agent.init(agent_key)
    param_vec_spec = ParamVectorSpec(agent_state.params.policy_params)

    nsga2 = algorithms.NSGA2(
        lb=jnp.full(shape=(param_vec_spec.vec_size,), fill_value=config.agent_network.lb),
        ub=jnp.full(shape=(param_vec_spec.vec_size,), fill_value=config.agent_network.ub),
        n_objs=len(config.metric_names),
        pop_size=config.pop_size,
    )

    # TODO: add op for MLP
    nsga2_rl = MOAlgorithmWrapper(
        algo=nsga2,
        param_vec_spec=param_vec_spec
    )

    def _sol_transform(cand):
        params = agent_state.params.replace(policy_params=cand)
        return agent_state.replace(params=params)

    # monitor = monitors.EvalMonitor()

    workflow = ECWorkflow(
        algorithm=nsga2_rl,
        problem=problem,
        opt_direction=['max', 'max'],
        sol_transforms=[jax.vmap(_sol_transform)],
        # monitors=[monitor]
    )

    jnp.set_printoptions(precision=4, suppress=True)

    state = workflow.init(workflow_key)
    for i in range(config.num_iters):
        state = workflow.step(state)
        jax.block_until_ready(state)
        print(f"iteration: {state.generation}")
        # fitness = monitor.get_latest_fitness()
        fitness = state.get_child_state('algorithm').get_child_state('algo').fitness

        rank = non_dominated_sort(-fitness)
        pf = rank == 0
        pf_fitness = fitness[pf]
        pf_fitness = pf_fitness[pf_fitness[:, 0].argsort()]
        print(f"#pf={pf_fitness.shape[0]}")
        print(pf_fitness)


if __name__ == "__main__":
    train()
