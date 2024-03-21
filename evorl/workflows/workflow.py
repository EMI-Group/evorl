import jax
import jax.numpy as jnp
import chex
from omegaconf import DictConfig
from evox import Stateful, State
from abc import ABC

class Workflow(Stateful):
    def step(self, state: State) -> State:
        raise NotImplementedError

    def learn(self, state: State) -> State:
        """
            run the complete learning process.
            Note: this is designed for the non pure function. Don't wrap it with jit.
        """
        raise NotImplementedError

    def jit(self) -> None:
        self.step = jax.jit(self.step, donate_argnums=(0,))