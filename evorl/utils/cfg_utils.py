from hydra.core.hydra_config import HydraConfig
from pathlib import Path

def get_output_dir(default_path: str='./debug'):
    if HydraConfig.initialized():
        output_dir = Path(HydraConfig.get().run.dir).absolute()
    else:
        output_dir = Path(default_path).absolute()

    return output_dir