from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(config_path="configs", config_name="base")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    train()