import os
import wandb
from omegaconf import DictConfig, OmegaConf
from datetime import datetime


def setup_wandb(cfg: DictConfig, run_id: str = None) -> None:
    if run_id:
        run = wandb.init(
            project=cfg.logger.project,
            entity=cfg.logger.entity,
            id=run_id,
            resume="must",
        )
    else:
        name = get_wandb_name(cfg)
        datestamp = datetime.now().strftime("%Y-%m-%d")
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        run = wandb.init(
            project=cfg.logger.project,
            entity=cfg.logger.entity,
            name=name,
            group=cfg.logger.group,
            tags=[
                cfg.algorithm._target_.split(".")[-1],
                cfg.environment._target_.split(".")[-1],
                f"seed_{cfg.train.seed}",
                datestamp,
                cfg.experiment_name,
            ],
            config=cfg_dict,
        )
        cfg.logger.run_id = run.id
    return run


def get_wandb_name(cfg):
    """
    Get the name of the wandb run based on algorithm, environment, and seed and a 5 digit random number.
    """
    pid = os.getpid()
    algorithm = cfg.algorithm._target_.split(".")[-1]
    env_name = cfg.environment._target_.split(".")[-1]
    experiment_name = cfg.experiment_name
    wandb_name = f"({pid})_{algorithm}_{env_name}_{experiment_name}_{cfg.train.seed}"
    return wandb_name
