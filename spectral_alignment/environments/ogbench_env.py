from typing import Any, Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium.wrappers.transform_observation import DtypeObservation
import ogbench

from spectral_alignment.environments.wrappers import (
    OGBenchRenderingWrapper,
    ActionRepeatWrapper,
    SparseRewardWrapper,
    ActionChunkWrapper,
)
from spectral_alignment.utils.dataset_utils import relabel_with_action_repeat, relabel_with_action_chunk


class OGBenchEnv(gym.Wrapper):
    def __init__(
        self,
        seed,
        action_repeat_count=1,
        action_chunk_count=1,
        dataset_config=None,
        sparse_reward=False,
        *args,
        **kwargs,
    ):
        if action_repeat_count > 1 and action_chunk_count > 1:
            raise ValueError("Cannot use both action_repeat_count and action_chunk_count > 1 at the same time.")
        env = ogbench.make_env_and_datasets(*args, env_only=True, **kwargs)
        # --- Add env wrappers here if needed ---
        if env.observation_space.dtype == np.float64:
            env = DtypeObservation(env, dtype=np.float32)
        if action_repeat_count > 1:
            env = ActionRepeatWrapper(env, repeat_count=action_repeat_count)
        if action_chunk_count > 1:
            env = ActionChunkWrapper(env, chunk_count=action_chunk_count)
        env = OGBenchRenderingWrapper(env, render_mode=kwargs.get("render_mode", None))
        if sparse_reward:
            env = SparseRewardWrapper(env, reward_indicator_fn=self.is_successful)
        # --- call super constructor ---
        super().__init__(env)
        # --- seeding the environment ---
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        # --- save environment and dataset config ---
        self.dataset_config = dataset_config
        self.action_repeat_count = action_repeat_count
        self.action_chunk_count = action_chunk_count

    def is_successful(self, reward, terminated, truncated, info) -> bool:
        raise NotImplementedError(
            "is_successful method should be implemented by subclasses to determine a binary success."
        )

    def load_dataset(self) -> Tuple[Any | None, Dict | None]:
        """
        Loads the dataset using the OGBench API.
        NOTE: Override this method if you want to load the dataset differently (e.g., with custom relabeling or preprocessing).
        """
        if (
            self.dataset_config is None
            or self.dataset_config.get("dataset_type", None) is None
            or self.dataset_config.get("dataset_name", None) is None
        ):
            return None, None

        if self.dataset_config["dataset_type"] == "ogbench":
            return self._load_ogbench_dataset()
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_config['dataset_type']}")

    def _load_ogbench_dataset(self) -> Tuple[Any, Dict]:
        dataset_name = self.dataset_config.get("dataset_name")
        dataset_dir = self.dataset_config.get("dataset_dir", "~/.ogbench/datasets")
        train_dataset, _ = ogbench.make_env_and_datasets(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            dataset_only=True,
            cur_env=self.env,
        )
        if self.dataset_config.get("dataset_fraction", 1.0) < 1.0:
            n = len(train_dataset["actions"])
            end_idx = int(n * self.dataset_config["dataset_fraction"])
            train_dataset = {k: v[:end_idx] for k, v in train_dataset.items()}
        if self.action_repeat_count > 1:
            train_dataset = relabel_with_action_repeat(train_dataset, self.action_repeat_count)
        if self.action_chunk_count > 1:
            train_dataset = relabel_with_action_chunk(train_dataset, self.action_chunk_count)
        return train_dataset, {"type": "ogbench", "size": len(train_dataset["actions"])}


class ArenaEnv(OGBenchEnv):
    def __init__(self, *args, visual=False, single_task=False, sparse_reward=False, **kwargs):
        dataset_name = "pointmaze-arena-navigate-singletask-v0" if single_task else "pointmaze-arena-navigate-v0"
        if visual:
            dataset_name = f"visual-{dataset_name}"
        super().__init__(*args, dataset_name=dataset_name, sparse_reward=sparse_reward, **kwargs)


class RoomsEnv(OGBenchEnv):
    def __init__(self, *args, visual=False, single_task=False, sparse_reward=False, **kwargs):
        dataset_name = "pointmaze-rooms-navigate-singletask-v0" if single_task else "pointmaze-rooms-navigate-v0"
        if visual:
            dataset_name = f"visual-{dataset_name}"
        super().__init__(*args, dataset_name=dataset_name, sparse_reward=sparse_reward, **kwargs)


class MazeEnv(OGBenchEnv):
    def __init__(self, *args, visual=False, single_task=False, sparse_reward=False, **kwargs):
        dataset_name = "pointmaze-medium-navigate-singletask-v0" if single_task else "pointmaze-medium-navigate-v0"
        if visual:
            dataset_name = f"visual-{dataset_name}"
        super().__init__(*args, dataset_name=dataset_name, sparse_reward=sparse_reward, **kwargs)

    def is_successful(self, reward, terminated, truncated, info) -> bool:
        return bool(info["success"])

    @staticmethod
    def obs_subset_for_density(obs: np.ndarray) -> np.ndarray:
        """
        Extracts a subset of the observation for density estimation.
        For Maze we use the agent position (x, y)
        """
        return obs


class LargeMazeEnv(OGBenchEnv):
    def __init__(self, *args, visual=False, single_task=False, sparse_reward=False, **kwargs):
        dataset_name = "pointmaze-large-navigate-singletask-v0" if single_task else "pointmaze-large-navigate-v0"
        if visual:
            dataset_name = f"visual-{dataset_name}"
        super().__init__(*args, dataset_name=dataset_name, sparse_reward=sparse_reward, **kwargs)

    def is_successful(self, reward, terminated, truncated, info) -> bool:
        return bool(info["success"])

    @staticmethod
    def obs_subset_for_density(obs: np.ndarray) -> np.ndarray:
        """
        Extracts a subset of the observation for density estimation.
        For Maze we use the agent position (x, y)
        """
        return obs


class GiantMazeEnv(OGBenchEnv):
    def __init__(self, *args, visual=False, single_task=False, sparse_reward=False, **kwargs):
        dataset_name = "pointmaze-giant-navigate-singletask-v0" if single_task else "pointmaze-giant-navigate-v0"
        if visual:
            dataset_name = f"visual-{dataset_name}"
        super().__init__(*args, dataset_name=dataset_name, sparse_reward=sparse_reward, **kwargs)

    def is_successful(self, reward, terminated, truncated, info) -> bool:
        return bool(info["success"])

    @staticmethod
    def obs_subset_for_density(obs: np.ndarray) -> np.ndarray:
        """
        Extracts a subset of the observation for density estimation.
        For Maze we use the agent position (x, y)
        """
        return obs


class CubeEnv(OGBenchEnv):
    def __init__(self, *args, visual=False, single_task=False, sparse_reward=False, **kwargs):
        dataset_name = "cube-single-play-singletask-v0" if single_task else "cube-single-play-v0"
        if visual:
            dataset_name = f"visual-{dataset_name}"
        super().__init__(*args, dataset_name=dataset_name, sparse_reward=sparse_reward, **kwargs)

    def is_successful(self, reward, terminated, truncated, info) -> bool:
        return bool(info["success"])

    @staticmethod
    def obs_subset_for_density(obs: np.ndarray) -> np.ndarray:
        """
        Extracts a subset of the observation for density estimation.
        For Cube we use the end-effector position (x, y, z)
        """
        return obs[12:15]


class CubeDoubleEnv(OGBenchEnv):
    def __init__(self, *args, visual=False, single_task=False, sparse_reward=False, **kwargs):
        dataset_name = "cube-double-play-singletask-v0" if single_task else "cube-double-play-v0"
        if visual:
            dataset_name = f"visual-{dataset_name}"
        super().__init__(*args, dataset_name=dataset_name, sparse_reward=sparse_reward, **kwargs)


class SceneEnv(OGBenchEnv):
    def __init__(self, *args, visual=False, single_task=False, sparse_reward=False, **kwargs):
        dataset_name = "scene-play-singletask-v0" if single_task else "scene-play-v0"
        if visual:
            dataset_name = f"visual-{dataset_name}"
        super().__init__(*args, dataset_name=dataset_name, sparse_reward=sparse_reward, **kwargs)
