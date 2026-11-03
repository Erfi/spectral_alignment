from typing import Callable
import numpy as np
import gymnasium as gym
import mujoco.viewer


class SparseRewardWrapper(gym.Wrapper):
    """
    Changes the behavior of the reward function to be sparse.
    """

    def __init__(self, env, reward_indicator_fn: Callable):
        """
        reward_indicator_fn: a function that takes in the reward, terminated, truncated, and info and returns a boolean value indicating whether the we should give the reward or not.
        """
        super().__init__(env)
        self.reward_indicator_fn = reward_indicator_fn

    def reward(self, reward, terminal, truncated, info):
        if self.reward_indicator_fn(reward, terminal, truncated, info):
            return 1.0
        return 0.0

    def step(self, action):
        obs, reward, terminal, truncated, info = self.env.step(action)
        reward = self.reward(reward, terminal, truncated, info)
        return obs, reward, terminal, truncated, info


class OGBenchRenderingWrapper(gym.Wrapper):
    """
    Wrapper to set a separate viewer for OGBench environments.
    This is because OGBench does not have a "human" render mode by default.
    """

    def __init__(self, env, render_mode=None):
        super().__init__(env)
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(
                self.env.unwrapped.model,
                self.env.unwrapped.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            # Try setting the camera angle the same as rendering in ogbench
            camera = self._get_camera()
            if camera is not None and isinstance(camera, mujoco._structs.MjvCamera):
                self.viewer.cam.distance = camera.distance
                self.viewer.cam.elevation = camera.elevation
                self.viewer.cam.azimuth = camera.azimuth
                self.viewer.cam.lookat[:] = camera.lookat
        else:
            self.viewer = None

    def render(self, *args, **kwargs):
        if self.viewer is not None:
            self.viewer.sync()
        return self.env.render(*args, **kwargs)

    def _get_camera(self):
        if hasattr(self.env.unwrapped, "custom_camera"):
            return self.env.unwrapped.custom_camera
        elif hasattr(self.env.unwrapped, "_camera"):
            return self.env.unwrapped._camera
        else:
            return None


class ActionRepeatWrapper(gym.Wrapper):
    """
    Wrapper to repeat the same action for a specified number of steps.
    """

    def __init__(self, env, repeat_count: int):
        super().__init__(env)
        self.action_repeat_count = repeat_count

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.action_repeat_count):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ActionChunkWrapper(gym.Wrapper):
    """
    Wrapper to chunk the action space into a specified number of steps.
    This wrapper modifies the action space to accept a sequence of actions, which are then executed in order over multiple steps.
    The action space will be modified as follows:
    - Original action space: Box(low, high, shape=(action_dim,))
    - Modified action space: Box(low, high, shape=(chunk_count * action_dim))
    """

    def __init__(self, env, chunk_count: int):
        super().__init__(env)
        self.action_chunk_count = chunk_count
        self.original_action_space = env.action_space
        self.action_space = gym.spaces.Box(
            low=np.repeat(
                self.original_action_space.low,
                self.action_chunk_count,
                axis=0,
            ),
            high=np.repeat(
                self.original_action_space.high,
                self.action_chunk_count,
                axis=0,
            ),
            shape=(self.action_chunk_count * self.original_action_space.shape[0],),
            dtype=self.original_action_space.dtype,
        )

    def step(self, action):
        assert (
            action.shape[0] == self.action_space.shape[0]
        ), "Action shape does not match the expected chunked action space."
        single_actions = action.reshape(self.action_chunk_count, -1)
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for i in range(self.action_chunk_count):
            obs, reward, terminated, truncated, info = self.env.step(single_actions[i])
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
