# Contributing to EvoRL

## Install Developer Version

```shell
# Step 1: Create a virtual env from conda, venv, pyenv, ...:
# For example, using pyenv-virtualenv:
pyenv virtualenv 3.10 evorl-dev
pyenv activate evorl-dev

# Step 2: Install JAX in advance
# Note: the installation command depends on your platform.
# See https://docs.jax.dev/en/latest/installation.html
pip install -U "jax[cuda12]"

# Step 3: Install the developer version
pip install -e ".[dev]"

# Step 4: post-setup
# This repo is configured with `pre-commit`.
pre-commit install
```

### Debugging

We provide some examples in `.vscode/launch.json` for debugging in Visual Studio Code.

## Documentation

Documentations are written under the `docs/` directory. We use [`Sphinx`](https://www.sphinx-doc.org/) to construct the documentation. Since we use the [`MyST-Parser`](https://myst-parser.readthedocs.io/) extension, Markdown documents (`*.md`) all allowed. API References are automatically generated by [`autodoc2`](https://sphinx-autodoc2.readthedocs.io/).

If you want to build the documentations in local machine, run the script `./docs/build.sh`. Then, the documentation Web files are generated in `docs/_build`.

## Code Style

In general, we follow the [Black code style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) for Python code. For Python documentation, we follow the [Google pydoc format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

The following code snippets demonstrate the code style in EvoRL:

```python
class Workflow(AbstractWorkflow):
    """The base class for all Workflows.

    All workflow classes are inherit from this class, and customize by implementing
    """

    def __init__(self, config: DictConfig):
        """Initialize a RLWorkflow instance.

        Args:
            config: the config object.
        """
        self.config = config
        self.recorder = ChainRecorder([])
        self.checkpoint_manager = setup_checkpoint_manager(config)
```

```python
class ReplayBufferState(PyTreeData):
    """Contains data related to a replay buffer.

    Attributes:
        data: the stored replay buffer data.
        current_index: the pointer used for adding data.
        buffer_size: the current size of the replay buffer.
    """

    data: chex.ArrayTree
    current_index: chex.Array = jnp.zeros((), jnp.int32)
    buffer_size: chex.Array = jnp.zeros((), jnp.int32)
```

Developers do not need to worry about whether the submitted code satisfy the above code style. Before submitting any commit, the `pre-commit` will raise warning or errors to remind you what is wrong. Besides, developers can manually execute this process by running

```shell
pre-commit run -a
```
