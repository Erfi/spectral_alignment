from typing import Dict, Any
from cpprb import ReplayBuffer
import numpy as np


def add_dataset_to_buffer(dataset: Any, dataset_info: Dict, buffer: ReplayBuffer) -> None:
    """
    Adds a dataset to a cpprb replay buffer.

    Args:
        dataset (Any): The dataset to add to the buffer (ogbench format).
        dataset_info (Dict): The information about the dataset (e.g., type, size).
        buffer (ReplayBuffer): The replay buffer to add the dataset to.
    """
    datset_type = dataset_info.get("type", None)
    if dataset_info["type"] == "ogbench":
        # NOTE: in ogbench datasets, "terminals" mark the end of the episide no matter whether the episode is successful or not.
        # If there exist a "masks" key it should be used to mark success. Masks are 0 only when the episode is successful, and 1 otherwise.
        # "masks" are only available for single-task datasets.
        buffer.add(
            obs=dataset["observations"],
            next_obs=dataset["next_observations"],
            act=dataset["actions"],
            rew=np.zeros_like(dataset["terminals"]) if "rewards" not in dataset else (dataset["rewards"]),
            done=np.zeros_like(dataset["terminals"]) if "masks" not in dataset else (1.0 - dataset["masks"]),
        )
    else:
        raise ValueError(f"Unsupported dataset type: {datset_type}")


def relabel_with_action_repeat(dataset: Dict, action_repeat: int) -> Dict:
    """
    Used to relabel the OGBench manipulation datasets with action repeat.
    This is needed because the original datasets are collected without action repeat,
    and we want to use them with action repeat if the online interaction uses action repeat.

    Args:
        dataset (Dict): The original ogbench dataset collected without action repeat.
        action_repeat (int): The action repeat to relabel the dataset with.
    """
    if action_repeat <= 1:
        return dataset

    observations = dataset["observations"]
    actions = dataset["actions"]
    next_observations = dataset["next_observations"]
    terminals = dataset["terminals"]
    rewards = dataset.get("rewards", None)
    masks = dataset.get("masks", None)
    if not np.any(terminals == 1):
        raise ValueError("Input dataset has no terminal==1 entries; cannot split into episodes.")

    repeated_obs = []
    repeated_actions = []
    repeated_next_obs = []
    repeated_terminals = []
    repeated_rewards = [] if rewards is not None else None
    repeated_masks = [] if masks is not None else None

    episode_start = 0
    n = len(terminals)

    for idx in range(n):
        is_episode_end = bool(terminals[idx])

        if not is_episode_end:
            continue

        episode_end = idx + 1
        ep_obs = observations[episode_start:episode_end]
        ep_actions = actions[episode_start:episode_end]
        ep_next_obs = next_observations[episode_start:episode_end]
        ep_terminals = terminals[episode_start:episode_end]
        ep_rewards = rewards[episode_start:episode_end] if rewards is not None else None
        ep_masks = masks[episode_start:episode_end] if masks is not None else None

        ep_len = len(ep_actions)
        num_chunks = ep_len // action_repeat

        for chunk_idx in range(num_chunks):
            start = chunk_idx * action_repeat
            end = start + action_repeat

            repeated_obs.append(ep_obs[start])
            repeated_actions.append(ep_actions[start])
            repeated_next_obs.append(ep_next_obs[end - 1])
            repeated_terminals.append(ep_terminals[start:end].max())
            if ep_rewards is not None:
                repeated_rewards.append(ep_rewards[start:end].sum())
            if ep_masks is not None:
                repeated_masks.append(ep_masks[start:end].min())
        episode_start = episode_end

    dataset["observations"] = np.asarray(repeated_obs, dtype=observations.dtype)
    dataset["actions"] = np.asarray(repeated_actions, dtype=actions.dtype)
    dataset["next_observations"] = np.asarray(repeated_next_obs, dtype=next_observations.dtype)
    dataset["terminals"] = np.asarray(repeated_terminals, dtype=terminals.dtype)
    if rewards is not None:
        dataset["rewards"] = np.asarray(repeated_rewards, dtype=rewards.dtype)
    if masks is not None:
        dataset["masks"] = np.asarray(repeated_masks, dtype=masks.dtype)

    return dataset


def relabel_with_action_chunk(dataset: Dict, action_chunk: int) -> Dict:
    """
    Used to relabel the OGBench manipulation datasets with action chunking.
    This is needed because the original datasets are collected without action chunking,
    and we want to use them with action chunking if the online interaction uses action chunking.
    It works by concatenating the actions togeter until the action chunk size is reached,
    and then using the first observation of the chunk as the observation, and the last observation of the chunk as the next observation.

    Args:
        dataset (Dict): The original ogbench dataset collected without action chunking.
        action_chunk (int): The action chunk to relabel the dataset with.
    """
    if action_chunk <= 1:
        return dataset

    observations = dataset["observations"]
    actions = dataset["actions"]
    next_observations = dataset["next_observations"]
    terminals = dataset["terminals"]
    rewards = dataset.get("rewards", None)
    masks = dataset.get("masks", None)
    if not np.any(terminals == 1):
        raise ValueError("Input dataset has no terminal==1 entries; cannot split into episodes.")

    chunked_obs = []
    chunked_actions = []
    chunked_next_obs = []
    chunked_terminals = []
    chunked_rewards = [] if rewards is not None else None
    chunked_masks = [] if masks is not None else None

    episode_start = 0
    n = len(terminals)

    for idx in range(n):
        is_episode_end = bool(terminals[idx])

        if not is_episode_end:
            continue

        episode_end = idx + 1
        ep_obs = observations[episode_start:episode_end]
        ep_actions = actions[episode_start:episode_end]
        ep_next_obs = next_observations[episode_start:episode_end]
        ep_terminals = terminals[episode_start:episode_end]
        ep_rewards = rewards[episode_start:episode_end] if rewards is not None else None
        ep_masks = masks[episode_start:episode_end] if masks is not None else None

        ep_len = len(ep_actions)
        num_chunks = ep_len // action_chunk

        for chunk_idx in range(num_chunks):
            start = chunk_idx * action_chunk
            end = start + action_chunk

            chunked_obs.append(ep_obs[start])
            chunked_actions.append(np.concatenate(ep_actions[start:end], axis=-1))
            chunked_next_obs.append(ep_next_obs[end - 1])
            chunked_terminals.append(ep_terminals[start:end].max())
            if ep_rewards is not None:
                chunked_rewards.append(ep_rewards[start:end].sum())
            if ep_masks is not None:
                chunked_masks.append(ep_masks[start:end].min())
        episode_start = episode_end

    dataset["observations"] = np.asarray(chunked_obs, dtype=observations.dtype)
    dataset["actions"] = np.asarray(chunked_actions, dtype=actions.dtype)
    dataset["next_observations"] = np.asarray(chunked_next_obs, dtype=next_observations.dtype)
    dataset["terminals"] = np.asarray(chunked_terminals, dtype=terminals.dtype)
    if rewards is not None:
        dataset["rewards"] = np.asarray(chunked_rewards, dtype=rewards.dtype)
    if masks is not None:
        dataset["masks"] = np.asarray(chunked_masks, dtype=masks.dtype)

    return dataset
