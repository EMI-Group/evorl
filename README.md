# EvoRL

EvoRL is an efficient Evolutionary Reinforcement Learning Framework integrated with [EvoX](https://github.com/EMI-Group/evox) (A JAX-accelerated EC Framework)

## Setup

For developer:

```shell
conda|mamba env create -f requirements/xuanwu.yaml
# install jax (in various way)
pip install "jax[cuda12]"
# install pip packages
pip install -r requirements/requirements-conda.txt
conda activate xuanwu
```

## Usage

EvoRL uses [hydra](https://hydra.cc/) to manage configs and run algorithms.

Train agents from cli:

```shell
python -m evorl.train agent=exp/ppo/brax/ant env=brax/ant

python -m evorl.train agent=exp/ppo/brax/ant env=gymnax/CartPole-v1 agent_network.continuous_action=false

# sweep over multiple config values (seed=114 or seed=514):
python -m evorl.train -m agent=exp/ppo/brax/ant env=brax/ant seed=114,514
```

## Acknowledgement

- brax
- acme
- evox
- gymnax
- jumanji
- jaxmarl
