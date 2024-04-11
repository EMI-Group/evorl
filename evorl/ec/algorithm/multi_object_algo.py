import jax
import jax.numpy as jnp
from evox import Algorithm
from evorl.agents import Agent
from evorl.utils.ec_utils import ParamVectorSpec

class MOAlgorithmWrapper(Algorithm):
    """
        Wrapper for EvoX basic MO Algorithms:
            - Use flattten params
            - Use flatten objectives
    """

    def __init__(self, algo: Algorithm, agent: Agent, param_vec_spec: ParamVectorSpec):
        self.algo = algo
        self.agent = agent
        self.param_vec_spec = param_vec_spec

    def setup(self, key):
        state = self.algo.setup(key)
        return state

    def init_ask(self, state):
        flat_pop, state = self.algo.init_ask(state)
        return self.param_vec_spec.to_tree(flat_pop), state

    def init_tell(self, state, fitness):
        # flatten pytree fitness: -> [pop_size, #obj]
        fitness = jnp.stack([f for f in fitness.values()])
        return self.algo.init_tell(state, fitness)

