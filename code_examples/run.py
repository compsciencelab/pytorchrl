from omegaconf import DictConfig, OmegaConf
import hydra

from pytorchrl.custom_environments import my_custom_environment_factory

@hydra.main(config_name="conf", config_path="./cfg")
def run_training(cfg: DictConfig) -> None:
    config = OmegaConf.to_yaml(cfg)
    
    trainer = Trainer(config, my_custom_environment_factory)
    trainer.train()


if __name__ == "__main__":
    run_training()