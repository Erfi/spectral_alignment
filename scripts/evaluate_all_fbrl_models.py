"""
Evaluate all saved FBRL models matching a glob pattern (50 episodes each by default) and write
test_evaluations__[task{i}__]model_top_1.npz next to each checkpoint. These files are read by visualize_FBRL.py.

Examples:
    python scripts/evaluate_all_fbrl_models.py --pattern "models/FBRL/RoomsEnv/**/model_top_1.zip"
    # per-task evaluation (OGBench tasks 1-5), e.g. for the Scene per-task figure
    python scripts/evaluate_all_fbrl_models.py --pattern "models/FBRL/SceneEnv/**/model_top_1.zip" --task_ids 1 2 3 4 5
"""

import argparse
import logging
from pathlib import Path
import glob
import subprocess
from multiprocessing import Pool

import hydra
import numpy as np
import torch
from tqdm import tqdm

from spectral_alignment.utils.common_utils import (
    seed_everything,
    load_model_config,
)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def get_available_devices():
    """Get list of available (unutilized) GPUs as torch device strings, or ["cpu"] if there is no GPU."""
    if not torch.cuda.is_available():
        return ["cpu"]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        available_gpus = []
        for line in result.stdout.strip().split("\n"):
            gpu_id, utilization = line.split(",")
            # Consider GPU available if utilization < 10%
            if int(utilization.strip()) < 10:
                available_gpus.append(f"cuda:{gpu_id.strip()}")
        return available_gpus if available_gpus else ["cuda:0"]  # Default to GPU 0 if none available
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("Could not query GPUs, defaulting to single GPU (0)")
        return ["cuda:0"]


def evaluate_model(path, device, task_id=None, n_eval_episodes=50, override=False):
    """Evaluate a single model on a specific device. task_id=None evaluates on random OGBench tasks."""
    task_prefix = "" if task_id is None else f"task{task_id}__"
    save_path = Path(path).parent / f"test_evaluations__{task_prefix}{Path(path).stem}.npz"
    if not override and save_path.exists():
        logging.info(f"Skipping evaluation for model at: {path} (already exists)")
        return

    saved_config = load_model_config(Path(path).parent)
    if saved_config is None:
        raise ValueError(f"Could not find saved model configuration for model at: {path}")

    logging.info(f"Evaluating {path} (task: {task_id}) on {device}")
    env = hydra.utils.instantiate(saved_config.environment, dataset_config=None)
    algo = hydra.utils.instantiate(saved_config.algorithm, env=env, device=device)
    algo.load(model_path=Path(path))
    _, _, returns, eps_lens = algo.test(env=env, n_episodes=n_eval_episodes, verbose=True, task_id=task_id)

    np.savez(
        save_path,
        mean_return=np.mean(returns),
        std_return=np.std(returns),
        returns=np.array(returns),
        eps_lens=np.array(eps_lens),
        n_eval_episodes=len(returns),
    )
    logging.info(f"Completed evaluation for model at: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", default="models/FBRL/**/model_top_1.zip", help="glob pattern for checkpoints")
    parser.add_argument("--task_ids", type=int, nargs="*", default=None, help="OGBench task ids (default: random)")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--override", action="store_true", help="re-evaluate even if the result file exists")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    devices = get_available_devices()
    logging.info(f"Using device(s): {devices}")

    model_paths = sorted(glob.glob(args.pattern, recursive=True))
    task_ids = args.task_ids if args.task_ids else [None]
    jobs = [(path, task_id) for path in model_paths for task_id in task_ids]
    logging.info(f"Found {len(model_paths)} models, {len(jobs)} evaluations to run")

    # Assign jobs to devices in round-robin fashion; one worker per device
    jobs = [(path, devices[i % len(devices)], task_id, args.n_episodes, args.override) for i, (path, task_id) in enumerate(jobs)]
    if len(devices) == 1:
        for job in tqdm(jobs, desc="Evaluating models"):
            evaluate_model(*job)
    else:
        with Pool(processes=len(devices)) as pool:
            list(tqdm(pool.starmap(evaluate_model, jobs), total=len(jobs), desc="Evaluating models in parallel"))


if __name__ == "__main__":
    main()
