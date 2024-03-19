#%%
%load_ext autoreload
%autoreload 2
#%%
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import jax
from evorl.agents.a2c import A2CWorkflow
from evorl.utils.toolkits import disable_gpu_preallocation
from hydra import compose, initialize
from omegaconf import OmegaConf
with initialize(config_path="./configs", version_base=None):
    cfg = compose(config_name="a2c")


learner = A2CWorkflow(cfg)
state = learner.init(jax.random.PRNGKey(42))
state
#%%
learner.step(state)
# %%

jit_step = jax.jit(learner.step)

for _ in range(100):
    state = jit_step(state)

state
# from evorl.workflows import EAWorkflow
# %%

# %%
state._state_id
# %%
learner.agent.policy_network
# %%
import jax
import jax.numpy as jnp

x= jnp.zeros((7, 8))
x1, x2=jnp.split(x, 2, axis=1)
print(x1.shape)
print(x2.shape)
# %%
import distrax
import jax
import jax.numpy as jnp
loc = jnp.zeros((3,7))
scale = jnp.ones((3,7))

dist= distrax.Normal(loc, scale)

# %%
import jax
import jax.numpy as jnp
import chex
from evorl.utils.distribution import TanhNormal

T=11
B=7
A=3

loc = jnp.zeros((T, B, A))
scale = jnp.ones((T, B, A))
actions = jax.random.uniform(jax.random.PRNGKey(42), shape=(T, B, A), minval=-0.8, maxval=0.8)

def loss_fn(loc, scale, actions):
    actions_dist = TanhNormal(loc, scale)
    logp = actions_dist.log_prob(actions)
    print(logp.shape)

    return -logp.mean()

loss, (g_loc, g_scale) = jax.value_and_grad(loss_fn, argnums=(0,1))(loc, scale, actions)
#%%
print(loss)
#%%
print(g_loc)
#%%
print(g_scale)
# %%
import distrax


# %%
# %%
import jax
import jax.numpy as jnp
import chex
from evorl.utils.distribution import TanhNormal

T=11
B=7
A=3

loc = jnp.zeros((T, B, A))
scale = jnp.ones((T, B, A))
actions_dist = TanhNormal(loc, scale)
# %%
import distrax
dist = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale)
# %%
bi=distrax.Tanh()
bib= distrax.Block(bi, 1)
# %%
dist = distrax.Transformed(
    distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale),
    distrax.Block(distrax.Tanh(), ndims=1)
)
# %%
dist.mode()
# %%
import jax
import jax.numpy as jnp
import chex
from evorl.agents.a2c import A2CWorkflow, A2CAgent, rollout

from hydra import compose, initialize
from evorl.envs import create_brax_env
from omegaconf import OmegaConf

import time
import contextlib

@contextlib.contextmanager
def timeit(name=""):
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{name} Time: {time.perf_counter()-start:.2f}s")

config = OmegaConf.create()
config.env = "ant"
config.num_envs = 4
config.rollout_length = 11
config.continuous_action = True

env = create_brax_env(
    config.env, parallel=config.num_envs, autoset=True)
agent = A2CAgent(
    action_space=env.action_space,
    obs_space=env.obs_space,
    continuous_action=config.continuous_action,
)

with timeit("init"):
    env_key, agent_key, rollout_key = jax.random.split(
        jax.random.PRNGKey(42), 3)
    env_state = env.reset(env_key)
    agent_state = agent.init(agent_key)

with timeit("rollout"):
    env_nstate, trajectory = rollout(
        env,
        env_state,
        agent,
        agent_state,
        rollout_key,
        rollout_length=config.rollout_length,
        extra_fields=('last_obs',)
    )
# %%
for k,v in trajectory.extras.items():
    print(k)
    for k1,v1 in v.items():
        print(f"{k1}: {v1.shape}")

# %%
env_nstate.reward.shape
# %%
