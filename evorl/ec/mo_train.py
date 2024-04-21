import jax
import jax.numpy as jnp

from evorl.ec import MOAlgorithmWrapper, MultiObjectiveBraxProblem

from evorl.utils.ec_utils import ParamVectorSpec
from evorl.workflows import ECWorkflow
from evorl.envs import create_wrapped_brax_env
from evorl.agents.ec import DeterministicECAgent
from evox import algorithms, monitors
from evox.operators import non_dominated_sort

def train(seed=42):

    pop_size = 1000

    parallel_envs: int = 5
    max_episode_steps: int = 1000

    env_name = 'ant'
    metric_names = ('reward_forward', 'reward_ctrl')

    key = jax.random.PRNGKey(seed)

    agent_key, workflow_key = jax.random.split(key)

    env = create_wrapped_brax_env(
        env_name,
        episode_length=max_episode_steps,
        parallel=parallel_envs,
        autoreset=False,
    )

    agent = DeterministicECAgent(
        action_space=env.action_space,
        obs_space=env.obs_space,
        actor_hidden_layer_sizes=(32, 32),
        normalize_obs=False
    )

    problem = MultiObjectiveBraxProblem(
        agent=agent,
        env=env,
        num_episodes=5,
        max_episode_steps=17,
        discount=1.0,
        metric_names=metric_names,
        flatten_objectives=True
    )

    # dummy_agent
    agent_state = agent.init(agent_key)
    param_vec_spec = ParamVectorSpec(agent_state.params.policy_params)

    nsga2 = algorithms.NSGA2(
        lb=jnp.full(shape=(param_vec_spec.vec_size,), fill_value=-10),
        ub=jnp.full(shape=(param_vec_spec.vec_size,), fill_value=10),
        n_objs=len(metric_names),
        pop_size=pop_size,
    )

    # TODO: add op for MLP
    nsga2_rl = MOAlgorithmWrapper(
        algo=nsga2,
        param_vec_spec=param_vec_spec
    )

    def _sol_transform(cand):
        params = agent_state.params.replace(policy_params=cand)
        return agent_state.replace(params=params)

    monitor = monitors.EvalMonitor()

    workflow = ECWorkflow(
        algorithm=nsga2_rl,
        problem=problem,
        opt_direction='max',
        sol_transforms=[jax.vmap(_sol_transform)],
        monitors=[monitor]
    )

    state = workflow.init(workflow_key)
    for i in range(1000):
        state = workflow.step(state)
        jax.block_until_ready(state)
        print(state.generation)
        fitness = monitor.get_latest_fitness()

        rank = non_dominated_sort(-fitness)
        pf = rank==0
        pf_fitness = fitness[pf]

        print(pf_fitness)


if __name__ == "__main__":
    train()
