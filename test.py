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
