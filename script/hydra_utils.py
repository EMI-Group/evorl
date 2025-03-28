import re
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from absl import logging


def set_omegaconf_resolvers():
    """Senitize the dirname."""
    OmegaConf.register_new_resolver(
        "sanitize_dirname", lambda path: re.sub(r"/", "_", path)
    )


def get_output_dir(default_path: str = "./debug"):
    """Return the output directory of hydra."""
    if HydraConfig.initialized():
        output_dir = HydraConfig.get().runtime.output_dir
    else:
        output_dir = default_path

    output_dir = Path(output_dir).expanduser().resolve()

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    return output_dir


_absl_log_level_map = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.FATAL,
}


def set_absl_log_level(level: str = "warning"):
    """Set the absl log level."""
    logging.set_verbosity(_absl_log_level_map[level])
