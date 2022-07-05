from omegaconf import DictConfig, OmegaConf
import hydra
from pytorchrl.trainer import Trainer
import wandb

# from pytorchrl.custom_environments import my_custom_environment_factory

@hydra.main(config_name="conf", config_path="./cfg", version_base=None)
def run_training(cfg: DictConfig) -> None:
    config = OmegaConf.to_yaml(cfg)
    print("Start Training\n")
    print("Training config: ", config, "\n")
    
    # Handle wandb init
    if cfg.wandb_key:
        mode = "online"
        wandb.login(key=str(cfg.wandb_key))
    else:
        mode = "disabled"

    with wandb.init(project=cfg.experiment_name, name=cfg.agent_name, config=cfg, mode=mode):
        trainer = Trainer(cfg, custom_environment_factory=None)
        trainer.train(wandb)


if __name__ == "__main__":
    run_training()