import logging
import math
import os
from typing import Any
from pathlib import Path

import coloredlogs
from cpprb import ReplayBuffer
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update

from spectral_alignment.algorithms.base import BaseAlgorithm
from spectral_alignment.utils.dataset_utils import add_dataset_to_buffer
from spectral_alignment.utils.common_utils import (
    cpprb_samples_to_torch,
    evaluate_multi_task,
    combine_batches,
    grad_norm,
)

logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG", logger=logger)


class FBRL(BaseAlgorithm):

    def __init__(
        self,
        env,
        features_extractor: DictConfig,
        goal_features_extractor: DictConfig,
        obs_normalizer: DictConfig,
        actor: DictConfig,
        forwardnet: DictConfig,
        backwardnet: DictConfig,
        forwardnet_optimizer: DictConfig,
        backwardnet_optimizer: DictConfig,
        actor_optimizer: DictConfig,
        buffer_size: int,
        gamma,
        tau,
        z_dim,
        orthogonal_loss_coef,
        z_sampling_goal_ratio,
        learning_starts,
        batch_size,
        batch_offline_ratio,
        device,
        z_hold_steps,
        update_to_data_ratio,
    ):
        self.env = env
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.z_dim = z_dim
        self.orthogonal_loss_coef = orthogonal_loss_coef
        self.z_sampling_goal_ratio = z_sampling_goal_ratio
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.batch_offline_ratio = batch_offline_ratio
        self.device = device
        self.features_extractor = features_extractor
        self.goal_features_extractor = goal_features_extractor
        self.obs_normalizer_config = obs_normalizer
        self.actor_config = actor
        self.forwardnet_config = forwardnet
        self.backwardnet_config = backwardnet
        self.forwardnet_optimizer_config = forwardnet_optimizer
        self.backwardnet_optimizer_config = backwardnet_optimizer
        self.actor_optimizer_config = actor_optimizer
        self.z_hold_steps = z_hold_steps
        self.update_to_data_ratio = update_to_data_ratio
        self._setup()

    def _setup(self):
        self._setup_models()
        self._setup_data()
        self.diag = torch.eye(self.batch_size, device=self.device)
        self.offdiag = 1.0 - self.diag
        self.offdiag_sum = self.offdiag.sum()
        self.current_rollout_z = self.sample_z(1, device=self.device)

    def _setup_models(self):
        """
        Sets up the models (e.g., neural networks) used in the algorithm.
        """
        # Networks
        self.features_extractor.input_dim = self.env.observation_space.shape[-1]  # state_dim or channel_dim
        if OmegaConf.is_missing(self.goal_features_extractor, "input_dim"):
            self.goal_features_extractor.input_dim = self.env.observation_space.shape[-1]  # state_dim or channel_dim,
        self.obs_normalizer = instantiate(
            self.obs_normalizer_config,
            input_dim=self.env.observation_space.shape[-1],
            dtype=torch.float32,
            device=self.device,
        )
        self.actor = instantiate(
            self.actor_config,
            features_extractor=self.features_extractor,
            action_dim=self.env.action_space.shape[0],
            z_dim=self.z_dim,
        ).to(self.device)
        self.forwardnet = instantiate(
            self.forwardnet_config,
            features_extractor=self.features_extractor,
            action_dim=self.env.action_space.shape[0],
            z_dim=self.z_dim,
        ).to(self.device)
        self.backwardnet = instantiate(
            self.backwardnet_config,
            features_extractor=self.goal_features_extractor,
            z_dim=self.z_dim,
        ).to(self.device)
        self.target_forwardnet = instantiate(
            self.forwardnet_config,
            features_extractor=self.features_extractor,
            action_dim=self.env.action_space.shape[0],
            z_dim=self.z_dim,
        ).to(self.device)
        self.target_backwardnet = instantiate(
            self.backwardnet_config,
            features_extractor=self.goal_features_extractor,
            z_dim=self.z_dim,
        ).to(self.device)
        # Copy weights to target networks
        self.target_forwardnet.load_state_dict(self.forwardnet.state_dict())
        self.target_backwardnet.load_state_dict(self.backwardnet.state_dict())
        # Optimizers
        self.forwardnet_optimizer = instantiate(
            self.forwardnet_optimizer_config,
            params=self.forwardnet.parameters(),
        )
        self.backwardnet_optimizer = instantiate(
            self.backwardnet_optimizer_config,
            params=self.backwardnet.parameters(),
        )
        self.actor_optimizer = instantiate(self.actor_optimizer_config, params=list(self.actor.parameters()))

    def _setup_data(self):
        """
        Sets up datasets, replay buffers, ... etc.
        """
        self.online_buffer = ReplayBuffer(
            size=self.buffer_size,
            env_dict={
                "obs": {
                    "shape": self.env.observation_space.shape,
                    "dtype": self.env.observation_space.dtype,
                },
                "next_obs": {
                    "shape": self.env.observation_space.shape,
                    "dtype": self.env.observation_space.dtype,
                },
                "act": {
                    "shape": self.env.action_space.shape,
                    "dtype": self.env.action_space.dtype,
                },
                "rew": {},
                "done": {},
            },
            default_dtype=np.float32,
        )

        dataset, dataset_info = self.env.load_dataset()
        if dataset is not None:
            # create offline buffer
            offline_buffer_size = dataset_info["size"]
            self.offline_buffer = ReplayBuffer(
                size=offline_buffer_size,
                env_dict={
                    "obs": {
                        "shape": self.env.observation_space.shape,
                        "dtype": self.env.observation_space.dtype,
                    },
                    "next_obs": {
                        "shape": self.env.observation_space.shape,
                        "dtype": self.env.observation_space.dtype,
                    },
                    "act": {
                        "shape": self.env.action_space.shape,
                        "dtype": self.env.action_space.dtype,
                    },
                    "rew": {},
                    "done": {},
                },
                default_dtype=np.float32,
            )
            add_dataset_to_buffer(dataset, dataset_info, self.offline_buffer)
            del dataset  # free up memory
            logger.info("Dataset loaded into offline replay buffer.")

            # warmup the observation normalizer
            for _ in range(10):
                observations = torch.from_numpy(self.offline_buffer.sample(100)["obs"]).to(self.device)
                self.obs_normalizer.update(observations)
            logger.info("Observation normalizer warmed up with dataset observations.")

    def train_step(self, obs, batch_size, global_step):
        self.set_to_train_mode()
        # --- ONLINE INTERACTION ---
        if global_step % self.update_to_data_ratio == 0:
            if global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                if global_step % self.z_hold_steps == 0:
                    self.current_rollout_z = self.sample_z(1, device=self.device)
                with torch.no_grad():
                    normalized_obs = self.obs_normalizer.normalize(torch.from_numpy(obs[np.newaxis, :]))
                    action = self.actor.get_action(
                        normalized_obs,
                        self.current_rollout_z,
                    )[0]
                    action = action.flatten().cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            self.online_buffer.add(
                obs=obs,
                next_obs=next_obs,
                act=action,
                rew=reward,
                done=False,
            )

            if terminated or truncated:
                obs, _ = self.env.reset()
                self.online_buffer.on_episode_end()
            else:
                obs = next_obs

            # possibly render
            if self.env.render_mode == "human":
                self.env.render()

        # --- TRAINING LOGIC ---
        training_metrics = {}
        if global_step >= self.learning_starts:
            # if there is an offline buffer, sample a mixed batch from the online and offline buffers
            if hasattr(self, "offline_buffer") and self.offline_buffer.get_buffer_size() > 0:
                n_offline_samples = int(batch_size * self.batch_offline_ratio)
                n_online_samples = batch_size - n_offline_samples
                online_samples = self.online_buffer.sample(n_online_samples)
                offline_samples = self.offline_buffer.sample(n_offline_samples)
                samples = combine_batches(online_samples, offline_samples)
            else:
                samples = self.online_buffer.sample(batch_size)
            samples = cpprb_samples_to_torch(samples, self.device)
            observations = samples["obs"]
            next_observations = samples["next_obs"]
            actions = samples["act"]

            # Normalize observations
            self.obs_normalizer.update(observations)
            self.obs_normalizer.update(next_observations)
            observations = self.obs_normalizer.normalize(observations)
            next_observations = self.obs_normalizer.normalize(next_observations)

            # Sample z
            z = self.sample_mixed_z(batch_size, train_goal=next_observations)

            # FB UPDATE
            with torch.no_grad():
                next_actions, _, _ = self.actor.get_action(next_observations, z)
                F_target = self.target_forwardnet(next_observations, next_actions, z)  # (N, batch_size, z_dim)
                B_target = self.target_backwardnet(next_observations)  # (batch_size, z_dim)
                Ms_target = torch.matmul(F_target, B_target.T)  # (N, batch_size, batch_size)
                M_target = Ms_target.mean(dim=0)  # (batch_size, batch_size)

            F_current = self.forwardnet(observations, actions, z)  # (N, batch_size, z_dim)
            B_current = self.backwardnet(next_observations)  # (batch_size, z_dim)
            Ms_current = torch.matmul(F_current, B_current.T)  # (N, batch_size, batch_size)

            # FB loss (1) - Bellman loss
            diff = Ms_current - self.gamma * M_target  # (N, batch_size, batch_size)
            FB_loss_offdiag = 0.5 * (diff * self.offdiag).pow(2).sum() / self.offdiag_sum
            FB_loss_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms_current.shape[0]
            FB_loss_bellman = FB_loss_offdiag + FB_loss_diag

            # FB loss (2) - Orthonormality's regularisation loss
            Cov = torch.matmul(B_current, B_current.T)  # (batch_size, batch_size)
            orth_loss_diag = -Cov.diag().mean()
            # orth_loss_diag = (self.diag.diag() - Cov.diag()).pow(2).mean()
            orth_loss_offdiag = 0.5 * (Cov * self.offdiag).pow(2).sum() / self.offdiag_sum
            FB_loss_orth = orth_loss_diag + orth_loss_offdiag
            FB_loss = FB_loss_bellman + self.orthogonal_loss_coef * FB_loss_orth

            # FB optimization step
            self.forwardnet_optimizer.zero_grad(set_to_none=True)
            self.backwardnet_optimizer.zero_grad(set_to_none=True)
            FB_loss.backward()
            forward_grad_norm = grad_norm(self.forwardnet.parameters())
            backward_grad_norm = grad_norm(self.backwardnet.parameters())
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.forwardnet.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(self.backwardnet.parameters(), 5.0)
            self.forwardnet_optimizer.step()
            self.backwardnet_optimizer.step()

            # Actor loss
            actions_pred, _, _ = self.actor.get_action(observations, z)
            F_pred = self.forwardnet(observations, actions_pred, z)  # (N, batch_size, z_dim)
            Q_pred = (F_pred * z).sum(dim=-1)  # (batch_size,)
            actor_loss = -Q_pred.mean()

            # Actor optimization step
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = grad_norm(self.actor.parameters())
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_optimizer.step()

            # Update target networks
            polyak_update(
                self.forwardnet.parameters(),
                self.target_forwardnet.parameters(),
                tau=self.tau,
            )
            polyak_update(
                self.backwardnet.parameters(),
                self.target_backwardnet.parameters(),
                tau=self.tau,
            )
            training_metrics["train/fb_loss"] = FB_loss.item()
            training_metrics["train/fb_loss_bellman"] = FB_loss_bellman.item()
            training_metrics["train/fb_loss_bellman_offdiag"] = FB_loss_offdiag.item()
            training_metrics["train/fb_loss_bellman_diag"] = FB_loss_diag.item()
            training_metrics["train/fb_loss_orth"] = FB_loss_orth.item()
            training_metrics["train/fb_loss_orth_offdiag"] = orth_loss_offdiag.item()
            training_metrics["train/fb_loss_orth_diag"] = orth_loss_diag.item()
            training_metrics["train/actor_loss"] = actor_loss.item()
            training_metrics["train/forward_grad_norm"] = forward_grad_norm
            training_metrics["train/backward_grad_norm"] = backward_grad_norm
            training_metrics["train/actor_grad_norm"] = actor_grad_norm

        # NOTE: MAKE SURE YOU ARE RETURNING THE NEXT OBSERVATION (NOT THE CURRENT ONE, NOT NORMALIZED)
        return training_metrics, obs

    def test(self, env: Any, n_episodes: int = 10, verbose: bool = False, task_id: int | None = None):
        mean_return, mean_eps_len, returns, eps_lens = evaluate_multi_task(
            env, self, n_episodes, self.device, task_id=task_id
        )
        if verbose:
            logger.info(
                f"return: {mean_return:.3f}+/-{np.std(returns):.3f}, episode length: {mean_eps_len:.3f}+/-{np.std(eps_lens):.3f}"
            )
        return mean_return, mean_eps_len, returns, eps_lens

    def set_to_train_mode(self):
        self.obs_normalizer.train()
        self.forwardnet.train()
        self.backwardnet.train()
        self.actor.train()

    def set_to_eval_mode(self):
        self.obs_normalizer.eval()
        self.forwardnet.eval()
        self.backwardnet.eval()
        self.actor.eval()

    def save(self, model_path: str = "model.zip", verbose: bool = False):
        os.makedirs(Path(model_path).parent, exist_ok=True)
        to_save_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "forwardnet_state_dict": self.forwardnet.state_dict(),
            "backwardnet_state_dict": self.backwardnet.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "forwardnet_optimizer_state_dict": self.forwardnet_optimizer.state_dict(),
            "backwardnet_optimizer_state_dict": self.backwardnet_optimizer.state_dict(),
            "observation_normalizer": self.obs_normalizer.state_dict(),
        }

        # save the model
        torch.save(
            to_save_dict,
            model_path,
        )
        if verbose:
            logger.info(f"model saved to {model_path}")

    def load(self, model_path: str, verbose=False):
        checkpoint = torch.load(
            model_path,
            weights_only=True,
            map_location=self.device,
        )
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.forwardnet.load_state_dict(checkpoint["forwardnet_state_dict"])
        self.backwardnet.load_state_dict(checkpoint["backwardnet_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.forwardnet_optimizer.load_state_dict(checkpoint["forwardnet_optimizer_state_dict"])
        self.backwardnet_optimizer.load_state_dict(checkpoint["backwardnet_optimizer_state_dict"])
        self.obs_normalizer.load_state_dict(checkpoint["observation_normalizer"])

        # Update target networks
        self.target_forwardnet.load_state_dict(self.forwardnet.state_dict())
        self.target_backwardnet.load_state_dict(self.backwardnet.state_dict())
        # move to device
        self.actor.to(self.device)
        self.forwardnet.to(self.device)
        self.target_forwardnet.to(self.device)
        self.backwardnet.to(self.device)
        self.target_backwardnet.to(self.device)
        if verbose:
            logger.info(f"model loaded from {model_path}")

    def predict(self, observation, goal):
        self.set_to_eval_mode()
        normalized_observation = self.obs_normalizer.normalize(observation)
        normalized_goal = self.obs_normalizer.normalize(goal)
        # get z from goal
        with torch.no_grad():
            z = self.backwardnet(normalized_goal)
            z = self.project_z(z)
            action = self.actor.predict(normalized_observation, z)
        action = action.flatten().cpu().numpy()
        return action

    @torch.no_grad()
    def sample_mixed_z(self, batch_size: int, train_goal: torch.Tensor | None = None):
        # samples a batch from the z distribution used to update the networks
        z = self.sample_z(batch_size, device=self.device)
        if train_goal is not None:
            perm = torch.randperm(batch_size, device=self.device)
            goals = self.backwardnet(train_goal[perm])
            goals = self.project_z(goals)
            mask = torch.rand((batch_size, 1), device=self.device) < self.z_sampling_goal_ratio
            z = torch.where(mask, goals, z)
        return z

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z
