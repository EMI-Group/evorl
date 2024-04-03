# EvoRL
EvoRL is an efficient Evolutionary Reinforcement Learning Framework based on [EvoX](https://github.com/EMI-Group/evox)


# Setup
```
conda|mamba env create -f xuanwu.yaml
conda activate xuanwu
```

# Usage

Train agents from cli:
```shell
python -m evorl.train agent=a2c env=brax env_name=ant

python -m evorl.train agent=a2c env=gymnax/CartPole-v1 agent_network.continuous_action=false
```

# Acknowledgement

- brax
- acme
- evox
- gymnax
- jumanji
- jaxmarl
