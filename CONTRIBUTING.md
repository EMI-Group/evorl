# Setup

```shell
# step1: create a virtual env from conda, venv, pyenv, ...:
# for example, using pyenv:
pyenv virtualenv 3.10 evorl-dev
pyenv activate evorl-dev

# install jax first (in various way)
pip install -U "jax[cuda12]"
# install pip packages
pip install -r requirements/requirements-contrib.txt
```

This repo is configured with pre-commit, to setup:

```shell
pre-commit install
```
