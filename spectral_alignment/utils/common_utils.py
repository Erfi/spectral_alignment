import random
from pathlib import Path
from typing import Dict

import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch.nn as nn
import torch



def grad_norm(parameters) -> float:
    """
    Computes the L2 norm of the gradients of the given parameters of a PyTorch model.
    """
    total_norm_sq = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.norm(2).item()
        total_norm_sq += param_norm * param_norm
    return total_norm_sq**0.5


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model_config(config: DictConfig, save_path: str | Path):
    """
    Save the OmegaConf DictConfig configuration dictionary to a YAML file.
    Args:
        config (DictConfig): The configuration dictionary to save.
        save_path (str | Path): The path where the YAML file will be saved.
    """
    save_path = Path(save_path) / "config.yaml"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        OmegaConf.save(config, f, resolve=True)


def load_model_config(load_path: str | Path) -> DictConfig:
    """
    Load the OmegaConf DictConfig configuration dictionary from a YAML file.
    Args:
        load_path (str | Path): The path to the YAML file to load.
    Returns:
        DictConfig: The loaded configuration dictionary.
    """
    load_path = Path(load_path) / "config.yaml"
    if not load_path.exists():
        return None
    return OmegaConf.load(load_path)


def recompose_config(config_path: str, overrides: list[str]):
    """
    recompose the config with the overrides
    we can use this to get the config for the model evaluation
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path=config_path, job_name="hydra_recompose")
    config = compose(config_name="config", overrides=overrides)
    return config


def stable_rank(M, tol=1e-8, top_k=None):
    """
    Compute the stable rank of matrix M.
    Args:
        M: input matrix (n x n)
    Returns:
        stable_rank: stable rank of M
    """
    s = np.linalg.svd(M, compute_uv=False)
    if top_k is not None:
        s = s[:top_k]
    s = s[s > tol]
    return np.sum(s**2) / np.max(s**2)


def participation_ratio(M, tol=1e-8, top_k=None):
    """
    Participation Ratio of matrix M.

    Parameters
    ----------
    M : np.ndarray
        SR matrix (or approximation)
    tol : float
        Singular values below this are ignored

    Returns
    -------
    float
        Participation ratio
    """
    s = np.linalg.svd(M, compute_uv=False)
    if top_k is not None:
        s = s[:top_k]
    s = s[s > tol]
    s2 = s**2

    numerator = np.sum(s2) ** 2
    denominator = np.sum(s2**2)

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def spectral_entropy(M, tol=1e-8, top_k=None):
    """
    Entropy-based effective rank of matrix M.

    Parameters
    ----------
    M : np.ndarray
        SR matrix (or approximation)
    tol : float
        Singular values below this are ignored

    Returns
    -------
    float
        Entropy effective rank
    """
    s = np.linalg.svd(M, compute_uv=False)
    if top_k is not None:
        s = s[:top_k]
    s = s[s > tol]
    s2 = s**2
    p = s2 / np.sum(s2)

    # Numerical safety: avoid log(0)
    entropy = -np.sum(p * np.log(p + 1e-20))
    # Normalize by log(B) where B is the number of components
    # This keeps the metric in the range [0, 1]
    normalization_factor = np.log(len(s))

    return entropy / normalization_factor if normalization_factor > 0 else 0.0


def evaluate_multi_task(env, model, n_episodes, device, task_id=None):
    """
    Evaluates the model (using the predict() method) in the given environment for n_episodes.
    The environment is assumed to be multi-task, i.e., each episode has a different goal.
    NOTE: Goals are taken from the info of the env.reset() method for each episode.

    Args:
        env: The environment to evaluate the actor in.
        model: The model to evaluate (needs to have a predict() method).
        n_episodes: The number of episodes to evaluate the actor for.
        task_id: If given, every episode uses this OGBench evaluation task (1-5); otherwise a random task per episode.

    Returns: The mean returns, mean episode length, list of returns and list of episode lengths.
    """
    eps_returns = []
    eps_lens = []
    reset_options = None if task_id is None else {"task_id": task_id}
    for _ in range(n_episodes):
        current_return = 0
        current_ep_len = 0
        state, reset_info = env.reset(options=reset_options)
        goal = reset_info.get("goal", None)
        if goal is None:
            raise ValueError(
                "Environment does not provide goal information in reset info. Is this a single-task environment?"
            )
        goal = torch.FloatTensor(goal).unsqueeze(0).to(device)
        done = False
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = model.predict(state, goal)
            next_state, reward, terminated, truncated, _ = env.step(action)
            current_return += reward
            current_ep_len += 1
            state = next_state
            done = terminated or truncated
            if env.render_mode == "human":
                env.render()
        eps_returns.append(current_return)
        eps_lens.append(current_ep_len)
    return np.mean(eps_returns), np.mean(eps_lens), eps_returns, eps_lens


def combine_batches(
    batch1: Dict[str, np.ndarray],
    batch2: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Combines two batches of samples into one.
    Args:
        batch1 (Dict[str, np.ndarray]): First batch of samples.
        batch2 (Dict[str, np.ndarray]): Second batch of samples.
    Returns:
        Dict[str, np.ndarray]: Combined batch of samples.
    """
    combined_batch = {}
    for key in batch1.keys():
        assert key in batch2, f"Key {key} not found in both batches."
        combined_batch[key] = np.concatenate([batch1[key], batch2[key]], axis=0)
    return combined_batch


def cpprb_samples_to_torch(
    samples: Dict,
    device: str = "cpu",
) -> dict:
    """
    Converts a batch of samples from cpprb (NumPy arrays) to PyTorch tensors.
    Args:
        samples (Dict): A dictionary containing the batch of samples with NumPy arrays.
        device (str): The device to move the tensors to ("cpu" or "cuda").
    Returns:
        dict: A dictionary containing the batch of samples as PyTorch tensors.
    """
    torch_samples = {}
    for key, value in samples.items():
        torch_samples[key] = torch.from_numpy(value).to(device)
    return torch_samples


class BaseObsNormalizer(nn.Module):
    """
    Base class for observation normalizers.
    """

    def __init__(self, dtype=torch.float32, device="cpu"):
        super(BaseObsNormalizer, self).__init__()
        self.device = device
        self.dtype = dtype

    def update(self, batch, *args, **kwargs):
        raise NotImplementedError

    def normalize(self, x):
        raise NotImplementedError

    def to_torch(self, x: np.ndarray | torch.Tensor, channels_first: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input data (NumPy array or PyTorch tensor).
                Must have shape (batch_size, obs_dim) for 1d
                or (batch_size, channels, height, width) for images.
                If single observation/image is provided,
                it will be unsqueezed to add batch dimension.
            channels_first: Whether the input images are in channels-first format.
        Returns:
            A PyTorch tensor on the correct device and with the correct dtype.
            If input is 1D, output shape will be (1, obs_dim).
            If input is 2D (images), output shape will be (1, channels, height, width).
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension for single observation
        elif x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension for single image
        elif x.dim() != 2 and x.dim() != 4:
            assert x.dim() == 2 or x.dim() == 4, "Input must be 1D or 2D (images)."

        if x.dim() == 4 and not channels_first:
            x = x.permute(0, 3, 1, 2)  # Convert to channels-first if needed

        return x.to(self.device, dtype=self.dtype)

    def forward(self, x: np.ndarray | torch.Tensor):
        """
        Forward pass for PyTorch compatibility.
        """
        return self.normalize(x)

    def __call__(self, x: np.ndarray | torch.Tensor):
        """
        Call method for PyTorch compatibility.
        """
        return self.normalize(x)


class IdentityNormalizer(BaseObsNormalizer):
    """
    Identity normalizer that returns the input as is.
    Useful as a placeholder when no normalization is desired.
    """

    def __init__(self, *args, channels_first: bool = False, dtype=torch.float32, device="cpu", **kwargs):
        super(IdentityNormalizer, self).__init__(dtype=dtype, device=device)
        self.device = device
        self.dtype = dtype
        self.channels_first = channels_first

    def update(self, *args, **kwargs) -> None:
        pass

    def normalize(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Returns the input as is.
        Args:
            x: Data to "normalize" (NumPy array or PyTorch tensor).
        Returns:
            The same data as a PyTorch tensor.
            If input is 1D, output shape will be (1, obs_dim).
            If input is 2D (images), output shape will be (1, channels, height, width).
        """
        x = self.to_torch(x, channels_first=self.channels_first)
        return x
