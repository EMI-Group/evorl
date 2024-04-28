import jax
import jax.numpy as jnp
from evox import Algorithm, State, use_state
from evorl.agents import Agent
from evorl.utils.ec_utils import ParamVectorSpec

class EvoXAlgorithmWrapper(Algorithm):
    """
        Wrapper for EvoX basic Algorithms:
            - Use flattten params
            - Use flatten objectives
    """

    def __init__(self, algo: Algorithm, param_vec_spec: ParamVectorSpec):
        self.algo = algo
        self.param_vec_spec = param_vec_spec

    # def setup(self, key):
    #     #TODO: fix duplicate state in here and in child 'algo'
    #     state = self.algo.setup(key)
    #     return state

    def init_ask(self, state):
        flat_pop, state = use_state(self.algo.init_ask)(state)
        return self._postprocess_pop(flat_pop), state

    def init_tell(self, state, fitness):
        # fitness = self._preprocess_fitness(fitness)
        return use_state(self.algo.init_tell)(state, fitness)
    
    def ask(self, state):
        flat_pop, state = use_state(self.algo.ask)(state)
        return self._postprocess_pop(flat_pop), state
    
    def tell(self, state, fitness):
        # fitness = self._preprocess_fitness(fitness)
        return use_state(self.algo.tell)(state, fitness)
    
    def _postprocess_pop(self, flat_pop):
        if flat_pop is None:
            return None
        else:
            return self.param_vec_spec.to_tree(flat_pop)
    
    # def _preprocess_fitness(self, fitness):
    #     # flatten pytree fitness: -> [pop_size, #obj]
    #     return jnp.stack([f for f in fitness.values()])

