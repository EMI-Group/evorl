# EvoRL
EvoRL is an efficient Evolutionary Reinforcement Learning Framework based on [EvoX](https://github.com/EMI-Group/evox)


# Setup
```
conda|mamba env create -f xuanwu.yaml
conda activate xuanwu
```

# Usage

EvoRL uses [hydra](https://hydra.cc/) to run algorithms.

Train agents from cli:
```shell
python -m evorl.train agent=a2c env=brax/ant

python -m evorl.train agent=a2c env=gymnax/CartPole-v1 agent_network.continuous_action=false

# sweep over multiple config values (seed=114 or seed=514):
python -m evorl.train -m  agent=a2c env=brax/ant seed=114,514
```

# Acknowledgement

- brax
- acme
- evox
- gymnax
- jumanji
- jaxmarl
