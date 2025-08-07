# Key Concepts

## Object-oriented functional programming model

EvoRL uses an object-oriented functional programming model, where classes define the static execution logic and their running states are stored externally. This is different from the concepts for the commonly used object-oriented programming model, where the class's states are stored inside the class as its properties.

This object-oriented functional programming model is to support JAX's functional programming style while taking advantage of object-oriented programming's modularity and composability. Below is a toy example to demonstrate what the codes look like.

```python
import jax
import jax.numpy as jnp


class Foo:
    def __init__(self, n, i):
        # n and i should be treated as static variables,
        # we should not change them after initialization.
        self.n = n
        self.i = i

    def init(self):
        return dict(
            a=jnp.ones((self.n,)),
            b=jnp.zeros((self.n,)),
        )

    @partial(jax.jit, static_argnums=0)
    def increment(self, state):
        res = state["a"] * state["b"]

        new_state = dict(
            a=state["a"] + self.i,
            b=state["b"] + self.i,
        )

        return res, new_state

foo = Foo(n=3, i=1)
state = foo.init()
for _ in range(10):
    res, state = foo.increment(state)
    print(res)
```

Functional programming requires that the functions are [Pure function](https://en.wikipedia.org/wiki/Pure_function), which have no side effects, i.e., no mutation of external variables out of the function. In this example, after creating the `foo` object, we should not change `foo.n` & `foo.i` and should view them as read-only variables. The `init()` function defines the initial values of `foo`. These initial values represent the object's **state** and are stored outside the object. Then, the state is utilized to execute the static logic defined in `foo.increment()`.

## Basic PyTree Data Containers

We provide some basic data containers to support the object-oriented functional programming model. They simplify the procedures of writing corresponding codes and increase the flexibility.

In package [`evorl.types`](#evorl.types), there are three basic data containers, as listed in the table. They are registered as [JAX PyTree](https://docs.jax.dev/en/latest/pytrees.html). With JAX PyTree API support, we can define which part of the data in the container is **static**.

```{note}
The term of **static** cames from `jax.jit`. Jitted functions only allow PyTree or `jax.Array` types as inputs. The static part of an input pytree object will be deemed constants during the compilation and determine the computation graph. When the static part of the input is changed in the following calls, the jitted function will be re-compiled.

Conversely, the `jax.Array` objects are viewed as **pure data**. When the data in these objects are changed (`dtype` and `shape` are still the same), the jitted function will not be compiled again.
```

| Type                                    | Description                     | Usage             |
| --------------------------------------- | ------------------------------- | ----------------- |
| [`PyTreeDict`](#evorl.types.PyTreeDict) | An easydict with pytree support | Store pure data   |
| [`PyTreeData`](#evorl.types.PyTreeData) | A pytree dataclass for Data     | Store data        |
| [`PyTreeNode`](#evorl.types.PyTreeNode) | A pytree dataclass for Node     | Build logic class |

- [`PyTreeDict`](#evorl.types.PyTreeDict) provides an [easydict](https://github.com/makinacorpus/easydict)-like API for general storage of pure data (`jax.Array`).

    ```python
    from evorl.types import PyTreeDict

    d = PyTreeDict({"a": jnp.ones((3,)), "b": jnp.zeros((3,))})
    print(d.a, d["b"])
    d.c = jnp.zeros((5,))
    ```

- [`PyTreeData`](#evorl.types.PyTreeData) provides python [dataclasses](https://docs.python.org/3/library/dataclasses.html) API. New data classes can inherit this class and explicitly define each field. Compared to `PyTreeDict`, it allows defining static data via [`pytree_field`](#evorl.types.pytree_field), and it ensures that all fields cannot be modified after the creation.

    ```python
    from evorl.types import PyTreeData

    class SampleBatch(PyTreeData):
        obs: jax.Array | None = None
        actions: jax.Array | None = None
        rewards: jax.Array | None = None
        next_obs: jax.Array | None = None
        dones: jax.Array | None = None

    sample_batch = SampleBatch(obs=jnp.ones((3, 4)))
    ```

    ```python
    class Bar(PyTreeData):
        a: jax.Array
        b: int = pytree_field(static=True, default=1) # b is static

    bar = Bar(a=jnp.ones((3, 4)), b=5)

    # The PyTreeData object is immutable
    # bar.a = jnp.zeros((3, 4)) # raise FrozenInstanceError

    # To change the field, use the `replace` method,
    # which will return a new data instance.
    new_bar = bar.replace(a=jnp.zeros((3, 4)))
    ```

- [`PyTreeNode`](#evorl.types.PyTreeNode) is similar to `PyTreeData`. However, it allows setting or changing some fields in `__post_init__` after creation. This feature makes it suitable for general classes. For example, `Agent`, `Evaluator`, `EvoOptimizer`, etc., are all from `PyTreeNode`.

    ```python
    class OpenES(EvoOptimizer):
        """OpenAI ES."""

        pop_size: int
        lr_schedule: ExponentialScheduleSpec
        noise_std_schedule: ExponentialScheduleSpec
        mirror_sampling: bool = True
        optimizer_name: str = "adam"
        weight_decay: float | None = None

        fitness_shaping_fn: Callable[[chex.Array], chex.Array] = pytree_field(
            static=True, default=compute_centered_ranks
        )
        optimizer: optax.GradientTransformation = pytree_field(static=True, init=False)

        def __post_init__(self):
            self.optimizer = optax.adam(
                learning_rate=self.lr_schedule.init
            )
    ```

## Agent

`Agent` encapsulates the learning agent and defines its actions for both training
and evaluation. It manages the networks for the learning agent, including the policy network, which determines the agent’s decisions for actions, and an optional value network used for estimating state or
state-action values. This class also specifies optional loss functions required for gradient-based updates.

In summary, it has two public methods:

- [`compute_actions(agent_state, sample_batch, key)`](#evorl.agent.Agent.compute_actions) defines the decision of actions for training, which computes the action from the policy model and adds some exploration noise.
- [`evaluate_actions(agent_state, sample_batch, key)`](#evorl.agent.Agent.evaluate_actions) defines the decision of actions for evaluation, which are usually the most confident actions from the policy model without additional exploration.

Besides, most RL-based Agents also include one or multiple loss functions, which will be called in the corresponding `workflow.step()`.

## RL Environments

We provide a unified environment API in [`Env`](#evorl.envs.env.Env) to adapt multiple env libraries.

An example about how to interact with the environment:

```python
from evorl.envs.brax import BraxAdapter
from brax.envs import get_environment

brax_env = get_environment("hopper")
env = BraxAdapter(brax_env)

# reset the environment
env_key = jax.random.PRNGKey(42)
env_state = env.reset(env_key)

# apply one step
actions = jnp.zeros((3,))
env_nstate = env.step(env_state, actions)
```

We provide multiple [`Wrapper`](#evorl.envs.wrappers.Wrapper) classes for `Env`, they are defined in [`evorl.envs.wrappers`](#evorl.envs.wrappers). For instance, [`ActionSquashWrapper`](#evorl.envs.wrappers.ActionSquashWrapper) converts the action space from [-1,1] to [low, high], [`VmapAutoResetWrapper`](#evorl.envs.wrappers.VmapAutoResetWrapper) converts a single env to k parallel envs.

Based on the top of them, we provide environment creation functions for different libraries.

```python
from evorl.envs import create_wrapped_brax_env, AutoresetMode

train_vec_env = create_wrapped_brax_env(
    "hopper", parallel=16, autoreset_mode=AutoresetMode.NORMAL
)
eval_vec_env = create_wrapped_brax_env(
    "hopper", parallel=16, autoreset_mode=AutoresetMode.DISABLED
)
```

### Timestep subscript notation

The environment timestep subscript notation follows the convention of other common RL libraries. For the env_state at timestep t:

```mermaid
graph LR
  pre:::hidden --> A[$$s_t$$] -- "$$A_t,R_t,D_t,T_t$$" --> B["$$s_{t+1}$$"] --> post:::hidden
  classDef hidden display: none;
```

* action: $A_t$
* done & termination: $D_t, T_t$
* reward: $R_t$
* obs: $O_{t+1}$



## Trajectory Data & Rollout

[`SampleBatch`](#evorl.sample_batch.SampleBatch) is a data container for trajectory data from the rollout between the agent and the environment. It is a subclass of `PyTreeData` and has 6 fields:

- `obs: chex.ArrayTree | None = None`
- `actions: chex.ArrayTree | None = None`
- `rewards: Reward | RewardDict | None = None`
- `next_obs: chex.Array | None = None`
- `dones: chex.Array | None = None`
- `extras: ExtraInfo | None = None`: Other trajectory information.

Some fields can be empty for various use cases. For example, `SampleBatch` can be used as a obs-only batch for computing actions, or used for storing trajectory data from the rollout.

[`evorl.rollout`](#evorl.rollout) provides various function to execute the rollout for a given agent and environment object. The example below demonstrates how to collect training data from a vectorized environment:

```python
from evorl import RandomAgent
from evorl.envs import create_wrapped_brax_env, AutoresetMode
from evorl.rollout import rollout

env = create_wrapped_brax_env(
    "hopper", parallel=16, autoreset_mode=AutoresetMode.NORMAL
)
agent = RandomAgent()

key = jax.random.PRNGKey(42)
rollout_key, env_key, agent_key = jax.random.split(key, 3)
env_state = env.reset(env_key)
agent_state = agent.init(env.obs_space, env.action_space, agent_key)

# trajectory data shape [128, 16, ...]
trajectory, env_nstate = rollout(
    env.step,
    agent.compute_actions,
    env_state,
    agent_state,
    rollout_key,
    rollout_length=128,
    env_extra_fields=("termination", "truncation", "steps"),
)
```

Besides collecting trajectory data, algorithms also need to evaluate the agent by complete episodes. We provide various evaluators in [`evorl.evaluators`](#evorl.evaluators).

```python
from evorl import RandomAgent
from evorl.envs import create_wrapped_brax_env, AutoresetMode
from evorl.evaluators import Evaluator

env = create_wrapped_brax_env(
    "hopper", parallel=16, autoreset_mode=AutoresetMode.NORMAL
)
agent = RandomAgent()

key = jax.random.PRNGKey(42)
eval_key, env_key, agent_key = jax.random.split(key, 3)
env_state = env.reset(env_key)
agent_state = agent.init(env.obs_space, env.action_space, agent_key)

evaluator = Evaluator(
    env=env,
    agent_fn=agent.evaluate_actions,
    max_episode_steps=1000,
    discount=1,
)

eval_metrics = evaluator.evaluate(agent_state, eval_key, num_episodes=10)
print("Avg Return:", eval_metrics.episode_returns.mean())
print("Avg Length:", eval_metrics.episode_lengths.mean())
```

## Workflow

A complete algorithm consists of three components:

1. a config file,
2. a `Workflow` subclass
3. a corresponding `Agent` subclass.

`Workflow` defines the entire training logic for a given algorithm. Its `step()` method encapsulates a single training iteration. Meanwhile, `learn()` method orchestrates the training loop. Besides calling `step()` repeatedly, it also manages other tasks such as termination condition checks, performance evaluation, periodic logging, and model checkpointing.

Algorithms are defined in `evorl.algorithm`. Most algorithms are defined in a single `*.py` file, containing their `Agent` class and `Workflow` class. A workflow receives a config object during the creation, which is linked to a `*.yaml` config file in path `configs/agent`.

An example of how to use `Workflow`:

```{include} ../_static/workflow_api.py
:literal:
:lang: python
```
