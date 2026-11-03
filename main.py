import os
import logging
from pathlib import Path
from datetime import datetime

import hydra
import numpy as np
from omegaconf import DictConfig

from spectral_alignment.utils.logger_utils import setup_wandb
from spectral_alignment.utils.common_utils import (
    seed_everything,
    save_model_config,
    load_model_config,
)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # --- Set up logging and model saving ---
    seed_everything(cfg.train.seed)
    pid = os.getpid()
    env_name = cfg.environment._target_.split(".")[-1]
    algo_name = cfg.algorithm._target_.split(".")[-1]
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = cfg.experiment_name
    model_path = (
        Path(cfg.saved_models_dir)
        / algo_name
        / env_name
        / experiment_name
        / f"seed_{cfg.train.seed}"
        / f"{timestamp}__{pid}"
    )
    # ---------------------------------------
    if cfg.mode == "train":
        logging.info(f"Training: {algo_name} on {env_name}")
        setup_wandb(cfg)
        save_model_config(cfg, model_path)
        env = hydra.utils.instantiate(cfg.environment)
        eval_env = hydra.utils.instantiate(cfg.environment, dataset_config=None)
        algo = hydra.utils.instantiate(cfg.algorithm, env=env)
        algo.train(
            total_timesteps=cfg.train.train_steps,
            batch_size=cfg.train.batch_size,
            log_interval=cfg.train.log_interval,
            eval_interval=cfg.train.eval_interval,
            eval_episodes=cfg.train.eval_episodes,
            save_top_k=2,
            model_path=model_path,
            eval_env=eval_env,
            global_step_start=0,
        )

    elif cfg.mode == "resume":
        logging.info(f"Resuming training: {algo_name} on {env_name}")
        saved_config = load_model_config(Path(cfg.load_model_path).parent)
        if saved_config is not None:
            restart_step = int(Path(cfg.load_model_path).stem.split("_")[-1])
            setup_wandb(saved_config, run_id=saved_config.logger.run_id)
            logging.info("Loaded saved model configuration for resuming training.")
            env = hydra.utils.instantiate(saved_config.environment)
            eval_env = hydra.utils.instantiate(saved_config.environment, dataset_config=None)
            algo = hydra.utils.instantiate(saved_config.algorithm, env=env)
        else:
            raise FileNotFoundError("Saved model configuration not found. Cannot resume training.")
        algo.load(model_path=cfg.load_model_path)
        algo.train(
            total_timesteps=saved_config.train.train_steps,
            batch_size=saved_config.train.batch_size,
            log_interval=saved_config.train.log_interval,
            eval_interval=saved_config.train.eval_interval,
            eval_episodes=saved_config.train.eval_episodes,
            save_top_k=2,
            model_path=Path(cfg.load_model_path).parent,
            eval_env=eval_env,
            global_step_start=restart_step + 1,
        )

    elif cfg.mode == "evaluate":
        N_EVAL_EPISODES = 50
        logging.info(f"Evaluating for {N_EVAL_EPISODES} episodes")
        saved_config = load_model_config(Path(cfg.load_model_path).parent)
        if saved_config is not None:
            logging.info("Loaded saved model configuration for evaluation.")
            env = hydra.utils.instantiate(saved_config.environment, dataset_config=None)
            algo = hydra.utils.instantiate(saved_config.algorithm, env=env)
        else:
            logging.warning("Could not find saved model configuration. Using current configuration for evaluation.")
            env = hydra.utils.instantiate(cfg.environment, dataset_config=None)
            algo = hydra.utils.instantiate(cfg.algorithm, env=env)
        algo.load(model_path=cfg.load_model_path)
        # eval_task_id=null evaluates on random OGBench tasks; eval_task_id=i evaluates only on task i
        task_id = cfg.eval_task_id
        _, _, returns, eps_lens = algo.test(env=env, n_episodes=N_EVAL_EPISODES, verbose=True, task_id=task_id)
        task_prefix = "" if task_id is None else f"task{task_id}__"
        np.savez(
            Path(cfg.load_model_path).parent / f"test_evaluations__{task_prefix}{Path(cfg.load_model_path).stem}.npz",
            mean_return=np.mean(returns),
            std_return=np.std(returns),
            returns=np.array(returns),
            eps_lens=np.array(eps_lens),
            n_eval_episodes=len(returns),
        )

    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    main()
