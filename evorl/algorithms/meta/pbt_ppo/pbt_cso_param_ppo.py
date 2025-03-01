from omegaconf import OmegaConf

import chex
import jax
import jax.numpy as jnp

from evorl.types import PyTreeDict, State
from evorl.utils.jax_utils import tree_zeros_like, tree_set, tree_get

from ..pbt_workflow import PBTOptState
from ..pbt_utils import uniform_init, log_uniform_init
from .pbt_param_ppo import PBTParamPPOWorkflow


class PBTCSOOptState(PBTOptState):
    velocity: chex.ArrayTree


class PBTCSOParamPPOWorkflow(PBTParamPPOWorkflow):
    @classmethod
    def name(cls):
        return "PBT-CSO-ParamPPO"

    def _setup_pop_and_pbt_optimizer(
        self, key: chex.PRNGKey
    ) -> tuple[chex.ArrayTree, PBTOptState]:
        search_space = self.config.search_space
        pop_size = self.config.pop_size

        assert pop_size % 2 == 0, "pop_size must be even"

        def _init(hp, key):
            match hp:
                case "actor_loss_weight" | "critic_loss_weight" | "clip_epsilon":
                    return log_uniform_init(search_space[hp], key, pop_size)
                case "entropy_loss_weight":
                    return -log_uniform_init(
                        OmegaConf.create(
                            dict(low=-search_space[hp].high, high=-search_space[hp].low)
                        ),
                        key,
                        pop_size,
                    )
                case "discount_g" | "gae_lambda_g":
                    return uniform_init(search_space[hp], key, pop_size)

        pop = PyTreeDict(
            {
                hp: _init(hp, key)
                for hp, key in zip(
                    search_space.keys(), jax.random.split(key, len(search_space))
                )
            }
        )

        pbt_opt_state = PBTCSOOptState(velocity=tree_zeros_like(pop))

        return pop, pbt_opt_state

    def exploit_and_explore(
        self,
        pbt_opt_state: PBTOptState,  # shared
        pop: chex.ArrayTree,  # sharding
        pop_workflow_state: State,  # sharding
        pop_metrics: chex.ArrayTree,  # sharding
        key: chex.PRNGKey,  # shared
    ) -> tuple[chex.ArrayTree, State, PBTOptState]:
        pairing_key, rand_key = jax.random.split(key)
        velocity = pbt_opt_state.velocity  # PyTreeDict
        pop_size = self.config.pop_size
        search_space = self.config.search_space

        randperm = jax.random.permutation(pairing_key, pop_size).reshape(2, -1)

        mask = pop_metrics[randperm[0]] > pop_metrics[randperm[1]]
        teacher_indices = jnp.where(mask, randperm[0], randperm[1])  # fast learner
        student_indices = jnp.where(mask, randperm[1], randperm[0])  # slow learner

        students_velocity = PyTreeDict()
        offsprings = PyTreeDict()
        for hp, key in zip(
            search_space.keys(), jax.random.split(rand_key, len(search_space))
        ):
            v = velocity[hp]
            x = pop[hp]

            match hp:
                case "actor_loss_weight" | "critic_loss_weight" | "clip_epsilon":
                    # compute the velocity in log space
                    x = jnp.log(x)
                case "entropy_loss_weight":
                    x = jnp.log(-x)
                case "discount_g" | "gae_lambda_g":
                    pass

            r1_key, r2_key = jax.random.split(key)
            chex.assert_equal_shape((v, x))
            r1 = jax.random.uniform(r1_key, shape=(pop_size // 2, *v.shape[1:]))
            r2 = jax.random.uniform(r2_key, shape=(pop_size // 2, *v.shape[1:]))
            v_stu = r1 * v[student_indices] + r2 * (
                x[teacher_indices] - x[student_indices]
            )

            x_stu = x[student_indices] + v_stu

            # turn back to original space
            match hp:
                case "actor_loss_weight" | "critic_loss_weight" | "clip_epsilon":
                    x_stu = jnp.exp(x_stu)
                case "entropy_loss_weight":
                    x_stu = -jnp.exp(x_stu)
                case "discount_g" | "gae_lambda_g":
                    pass

            x_stu = jnp.clip(
                x_stu, min=search_space[hp]["low"], max=search_space[hp]["high"]
            )

            students_velocity[hp] = v_stu
            offsprings[hp] = x_stu

        # Note: no need to deepcopy teachers_wf_state here, since it should be
        # ensured immutable in apply_hyperparams_to_workflow_state()
        teachers_wf_state = tree_get(pop_workflow_state, teacher_indices)
        offsprings_workflow_state = jax.vmap(self.apply_hyperparams_to_workflow_state)(
            teachers_wf_state, offsprings
        )

        velocity = tree_set(
            velocity, students_velocity, student_indices, unique_indices=True
        )
        pbt_opt_state = pbt_opt_state.replace(velocity=velocity)

        # ==== survival | merge population ====
        pop = tree_set(pop, offsprings, student_indices, unique_indices=True)
        pop_workflow_state = tree_set(
            pop_workflow_state,
            offsprings_workflow_state,
            student_indices,
            unique_indices=True,
        )

        return pop, pop_workflow_state, pbt_opt_state
