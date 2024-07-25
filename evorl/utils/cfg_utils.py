import re
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


def set_omegaconf_resolvers():
    OmegaConf.register_new_resolver(
        "sanitize_dirname", lambda path: re.sub(r"/", "_", path)
    )


def get_output_dir(default_path: str = "./debug"):
    if HydraConfig.initialized():
        output_dir = Path(HydraConfig.get().runtime.output_dir).absolute()
    else:
        output_dir = Path(default_path).absolute()

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

    return output_dir
