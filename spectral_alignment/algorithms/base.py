import os
from abc import ABC, abstractmethod
from typing import Tuple, Union, Dict
from pathlib import Path
import logging
import time

import gymnasium as gym
import numpy as np
import torch
import wandb


class BaseAlgorithm(ABC):
    """
    Base class for all algorithms in the project.
    """

    @abstractmethod
    def train_step(self, obs: np.ndarray, batch_size: int, global_step: int) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Train step for the algorithm.
        This is where the training logic should be implemented.
        Args:
            obs: The observation from the environment. (Not used in offline algorithms.)
            batch_size: The batch size for training.
            global_step: The current global step in the training process.
        Returns:
            train_metrics: A dictionary of training metrics to log.
            next_obs: The next observation from the environment (None for offline algorithms).
        """
        pass

    @abstractmethod
    def test(self, *args, **kwargs) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Returns the mean reward, mean episode length, returns, and episode lengths.
        """
        pass

    @abstractmethod
    def set_to_train_mode(self):
        """
        Set the networks to training mode.
        """
        pass

    @abstractmethod
    def set_to_eval_mode(self):
        """
        Set the networks to evaluation mode.
        """
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, observation: torch.Tensor, *args, **kwargs) -> Union[np.ndarray, int]:
        """
        Predict method for the algorithm.
        This is where the actions should be clipped / scaled to fit the environment.

        Args:
            observation: The observation from the environment.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        Returns:
            action: The action to take in the environment.
        """
        pass

    def train(
        self,
        total_timesteps: int,
        batch_size: int,
        log_interval: int,
        eval_interval: int,
        eval_episodes: int,
        save_top_k: int,
        model_path: str,
        eval_env: gym.Env = None,
        global_step_start: int = 0,
    ) -> None:
        """
        The training loop for all algorithms.
        This is the main function that will be called to train the algorithm.
        It will call the train_step() method to perform the training step and
        the validate_step() method to perform the validation step.

        Args:
            total_timesteps: Total timesteps (gradient steps) to train the algorithm.
            batch_size: Batch size for training.
            log_interval: How often to log the training stats.
            eval_interval: How often to evaluate (validate) the model.
            eval_episodes: Number of episodes to evaluate the model.
            save_top_k: Save how many top k models.
            model_path: Path to save the models.
            eval_env: Environment to evaluate the model on. If None, use the training environment (resets the environment).
            global_step_start: The global step to start from (useful for resuming training).
        """
        loggables = {}
        if global_step_start > 0:
            logging.info(f"Resuming training from global step {global_step_start}")
            # load previous evaluations.npz
            evals_path = Path(model_path) / f"evaluations.npz"
            if evals_path.exists():
                data = np.load(evals_path, allow_pickle=True)
                evaluations_timesteps = data["timesteps"].tolist()
                evaluations_returns = data["returns"].tolist()
                evaluations_lengths = data["lengths"].tolist()
        else:
            evaluations_timesteps: list[int] = []
            evaluations_returns: list[list[float]] = []
            evaluations_lengths: list[list[int]] = []
        top_k_returns = [-float("inf")] * save_top_k
        self.set_to_train_mode()
        obs, _ = self.env.reset()
        start_time = time.time()
        logging.info("Starting training...")
        for global_step in range(global_step_start, total_timesteps+1):
            # --- Training Step ---
            train_metrics, obs = self.train_step(obs=obs, batch_size=batch_size, global_step=global_step)

            # --- Log Training Metrics ---
            if global_step % log_interval == 0:
                loggables.update(train_metrics)
                loggables["train/SPS"] = int(global_step / (time.time() - start_time))
                wandb.log(loggables, step=global_step)

            # --- Validation Step ---
            if global_step % eval_interval == 0:
                logging.info(f"Validating at step {global_step}")
                self.validate_step(
                    model_path=model_path,
                    global_step=global_step,
                    eval_env=eval_env,
                    eval_episodes=eval_episodes,
                    top_k_returns=top_k_returns,
                    evaluations_timesteps=evaluations_timesteps,
                    evaluations_returns=evaluations_returns,
                    evaluations_lengths=evaluations_lengths,
                )

    def validate_step(
        self,
        model_path: str,
        global_step: int,
        eval_episodes: int,
        top_k_returns: list,
        evaluations_timesteps: list[int],
        evaluations_returns: list[list[float]],
        evaluations_lengths: list[list[int]],
        eval_env: gym.Env = None,
    ):
        """
        Validate the model during training by evaluating it on the given environment and logging the results.
        This is called within the train() method.
        """
        if eval_env is None:
            eval_env = self.env
        mean_return, mean_eps_len, returns, lens = self.test(eval_env, n_episodes=eval_episodes)
        wandb.log(
            {
                "eval/mean_return": mean_return,
                "eval/mean_eps_len": mean_eps_len,
            },
            step=global_step,
        )
        # save results in evaluations.npz
        evaluations_timesteps.append(global_step)
        evaluations_returns.append(returns)
        evaluations_lengths.append(lens)
        os.makedirs(Path(model_path), exist_ok=True)
        np.savez(
            Path(model_path) / f"evaluations.npz",
            timesteps=evaluations_timesteps,
            returns=evaluations_returns,
            lengths=evaluations_lengths,
        )
        # Save the top k models based on mean return
        top_k_returns.sort(reverse=True)  # decreasing order
        for i, top_return in enumerate(top_k_returns):
            if mean_return >= top_return:
                top_k_returns.pop(i)
                top_k_returns.insert(i, mean_return)
                model_name = Path(model_path) / f"model_top_{i+1}.zip"
                self.save(model_path=model_name, verbose=False)
                logging.info(f"{model_name} saved with mean return {mean_return}")
                break
        # remove the latest model to save space (only keep top k models + the latest model)
        for file in Path(model_path).glob("last_*.zip"):
            file.unlink()
        # save the latest model (overwrites the previous latest model)
        save_latest_path = Path(model_path) / f"last_{global_step}.zip"
        self.save(model_path=save_latest_path, verbose=False)
