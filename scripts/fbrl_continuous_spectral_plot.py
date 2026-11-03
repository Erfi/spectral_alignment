"""
The purpose of this plot is to plot the spectrum of the M matrix for different values of K from saved models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import glob
import hydra
from omegaconf import DictConfig
import torch

from spectral_alignment.utils.common_utils import (
    seed_everything,
    load_model_config,
    cpprb_samples_to_torch,
    stable_rank,
    spectral_entropy,
)
from spectral_alignment.utils.visualize_utils_FBRL import beautify_axis


def collect_data(model_path, save_path, n_eval_episodes=100, random_actions=False):
    """
    Collect replay buffer data from a saved model and save it to a file.
    """
    config = load_model_config(Path(model_path).parent)
    env = hydra.utils.instantiate(config.environment)
    algo = hydra.utils.instantiate(config.algorithm, env=env, device="cuda:0" if torch.cuda.is_available() else "cpu")
    algo.load(Path(model_path))

    for _ in range(n_eval_episodes):
        state, reset_info = env.reset()
        goal = reset_info.get("goal", None)
        goal = torch.FloatTensor(goal).unsqueeze(0).to(algo.device)
        done = False
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(algo.device)
            action = env.action_space.sample() if random_actions else algo.predict(state, goal)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            algo.online_buffer.add(
                obs=state.detach().cpu().numpy(), next_obs=next_state, act=action, rew=reward, done=done
            )
            state = next_state

    # save the replay buffer to a file
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    algo.online_buffer.save_transitions(save_path)
    logging.info(f"Saved {algo.online_buffer.get_stored_size()} transitions to {save_path}")


def calculate_M(algo, batch_size, gamma=None):
    # --- Sample a batch and compute the M matrix ---
    samples = algo.online_buffer.sample(batch_size)
    samples = cpprb_samples_to_torch(samples, algo.device)
    observations = samples["obs"]
    next_observations = samples["next_obs"]
    actions = samples["act"]

    # Normalize observations
    observations = algo.obs_normalizer.normalize(observations)
    next_observations = algo.obs_normalizer.normalize(next_observations)

    # Sample z
    # z = algo.sample_mixed_z(batch_size, train_goal=next_observations)
    z = algo.sample_mixed_z(1, train_goal=next_observations)
    z = z.repeat(batch_size, 1)  # (batch_size, z_dim)
    with torch.no_grad():
        F_current = algo.forwardnet(observations, actions, z)  # (N, batch_size, z_dim)
        B_current = algo.backwardnet(next_observations)  # (batch_size, z_dim)
        Ms_current = torch.matmul(F_current, B_current.T)  # (N, batch_size, batch_size)
        Ms_current = torch.mean(Ms_current, dim=0)  # (batch_size, batch_size)
        # Sinkhorn Normalization (rows to sum to 1/(1-gamma))
        Ms_current = torch.softmax(Ms_current / 100.0, dim=1) * (1.0 / (1.0 - gamma))
    return Ms_current.detach().cpu().numpy()


def get_model_paths(base_dir, pattern="rbf__act-rep*/**/model_top_1.zip"):
    """
    Get all model paths matching the pattern.
    Outputs a dictionary mapping from (act-rep, gamma, zdim) to a list of model paths (seeds).
    """
    model_paths = glob.glob(os.path.join(base_dir, pattern), recursive=True)
    logging.info(f"Found {len(model_paths)} models to evaluate")
    model_paths_dict = {}
    for path in model_paths:
        parts = Path(path).parts
        # experiment folder: rbf__act-rep-<k>__gamma-<gamma>__zdim-<zdim>
        if parts[-4].startswith("rbf__act-rep-"):
            sub_parts = parts[-4].split("__")
            act_rep = int(sub_parts[1].split("-")[-1])
            gamma = float(sub_parts[2].split("-")[-1])
            zdim = int(sub_parts[3].split("-")[-1])
            key = (act_rep, gamma, zdim)
            if key not in model_paths_dict:
                model_paths_dict[key] = []
            model_paths_dict[key].append(path)
    return model_paths_dict


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    seed_everything(0)

    MODEL_PATHS = get_model_paths(base_dir="./models/FBRL/RoomsEnv")
    BATCH_SIZE = 500
    N_EVAL_EPISODES = 100
    TOP_K_SVALS = 100  # , BATCH_SIZE
    SAVE_RB_PATH = "models/FBRL/RoomsEnv/misc/replay_buffer.npz"
    COLLECT_DATA = not Path(SAVE_RB_PATH).exists()  # collect once with random actions, then reuse
    PLOT_SAVE_PATH = Path("plots/continuous_spectral_analysis")
    PLOT_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    title_size = 24
    label_size = 24
    tick_size = 18
    legend_size = 16

    if COLLECT_DATA:
        collect_data(MODEL_PATHS[(1, 0.95, 100)][0], SAVE_RB_PATH, n_eval_episodes=N_EVAL_EPISODES, random_actions=True)

    ks = [1, 3, 5, 10, 20]
    gammas = [0.90, 0.95, 0.99, 0.999]
    zdims = [100]
    svals = {}
    sranks = {}
    entropies = {}

    for k in ks:
        for gamma in gammas:
            for zdim in zdims:
                key = (k, gamma, zdim)
                logging.info(f"Processing model with act-rep={k}, gamma={gamma}, zdim={zdim}")
                if key not in MODEL_PATHS:
                    logging.warning(f"No model found for act-rep={k}, gamma={gamma}, zdim={zdim}. Skipping.")
                    continue
                # average over seeds
                model_Ms = []
                for model_path in MODEL_PATHS[key]:
                    config = load_model_config(Path(model_path).parent)
                    env = hydra.utils.instantiate(config.environment)
                    algo = hydra.utils.instantiate(config.algorithm, env=env, device="cuda:0" if torch.cuda.is_available() else "cpu")
                    algo.load(Path(model_path))

                    logging.info(f"Loading replay buffer data from {SAVE_RB_PATH}")
                    algo.online_buffer.load_transitions(SAVE_RB_PATH)

                    logging.info(f"Calculating M matrix with batch size {BATCH_SIZE}")
                    M = calculate_M(algo, batch_size=BATCH_SIZE, gamma=gamma)
                    model_Ms.append(M)

                # average over seeds
                sv = [np.linalg.svd(M, compute_uv=False) for M in model_Ms]
                sv = np.mean(sv, axis=0)
                srank = [stable_rank(M) for M in model_Ms]
                srank = np.mean(srank)
                entropy = [spectral_entropy(M) for M in model_Ms]
                entropy = np.mean(entropy)
                svals[key] = sv
                sranks[key] = srank
                entropies[key] = entropy

    # ------------------------------------------------------------
    # --- Plot the singular values (gamma=0.95, z=100, vary k) ---
    # ------------------------------------------------------------
    sing_vals = [svals[(k, 0.9, 100)] for k in ks]
    labels = [f"k={k}" for k in ks]
    color_list = plt.get_cmap("cool", len(labels))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    for i, (label, sval) in enumerate(zip(labels, sing_vals)):
        ax.plot(
            sval[0:TOP_K_SVALS],
            marker="o",
            label=label,
            markersize=3,
            linewidth=1,
            color=color_list(i),
        )
    beautify_axis(
        ax,
        x_limits=(0, TOP_K_SVALS),
        xticks_int=True,
        decimals=2,
    )
    ax.set_title("Singular Values", fontsize=title_size)
    ax.set_xlabel("Index", fontsize=label_size)
    ax.set_ylabel("Value", fontsize=label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    ax.legend(loc="upper right", fontsize=legend_size)
    fig.tight_layout()
    fig.savefig(PLOT_SAVE_PATH / "singular_values_vary_k.pdf")

    # --------------------------------------------------------------------
    # --- Scatter-line Plot of Spectral Entropy gamma=0.9, z=100, vary k ---
    # --------------------------------------------------------------------
    ents = [entropies[(k, 0.9, 100)] for k in ks]
    labels = [f"k={k}" for k in ks]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    # as scatter + line plots
    # ax.plot(range(1, len(ks) + 1), ents, marker="o", markersize=10, color="slateblue", linewidth=2)
    ax.plot(ks, ents, marker="o", markersize=10, color="slateblue", linewidth=2)
    beautify_axis(ax, x_limits=(1, max(ks)), y_limits=(0, 1), xticks_int=True, decimals=1)
    # ax.set_xticks(range(1, len(ks) + 1))
    # ax.set_xticklabels(ks)
    ax.set_title("Spectral Entropy", fontsize=title_size)
    ax.set_ylabel("Entropy", fontsize=label_size)
    ax.set_xlabel("k", fontsize=label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    fig.tight_layout()
    fig.savefig(PLOT_SAVE_PATH / "spectral_entropy_vary_k.pdf")

    # --------------------------------------------------------------------
    # --- Scatter-line Plot of Stable Rank gamma=0.9, z=100, vary k ---
    # --------------------------------------------------------------------
    srs = [sranks[(k, 0.9, 100)] for k in ks]
    labels = [f"k={k}" for k in ks]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    # as scatter + line plots
    # ax.plot(range(1, len(ks) + 1), srs, marker="o", markersize=10, color="slateblue", linewidth=2)
    ax.plot(ks, srs, marker="o", markersize=10, color="slateblue", linewidth=2)
    beautify_axis(ax, x_limits=(1, max(ks)), y_limits=(0, 15), xticks_int=True, decimals=1)
    # ax.set_xticks(range(1, len(ks) + 1))
    # ax.set_xticklabels(ks)
    ax.set_title("Stable Rank", fontsize=title_size)
    ax.set_ylabel("Rank", fontsize=label_size)
    ax.set_xlabel("k", fontsize=label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    fig.tight_layout()
    fig.savefig(PLOT_SAVE_PATH / "stable_rank_vary_k.pdf")

    # ---------------------------------------------------------
    # --- Plot the singular values (k=1, z=100, vary gamma) ---
    # ---------------------------------------------------------
    sing_vals = [svals[(1, gamma, 100)] for gamma in gammas]
    labels = [f"$\gamma$={gamma}" for gamma in gammas]
    color_list = plt.get_cmap("cool", len(labels))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    for i, (label, sval) in enumerate(zip(labels, sing_vals)):
        ax.plot(
            sval[0:TOP_K_SVALS],
            marker="o",
            label=label,
            markersize=3,
            linewidth=1,
            color=color_list(i),
        )
    beautify_axis(
        ax,
        x_limits=(0, TOP_K_SVALS),
        xticks_int=True,
        decimals=0,
    )
    ax.set_title("Singular Values", fontsize=title_size)
    ax.set_xlabel("Index", fontsize=label_size)
    ax.set_ylabel("Value", fontsize=label_size)
    ax.set_ylim(bottom=1e-3)  # set a minimum y limit to avoid issues with log scale
    ax.set_yscale("symlog", base=10)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    ax.legend(loc="upper right", fontsize=legend_size)
    fig.tight_layout()
    fig.savefig(PLOT_SAVE_PATH / "singular_values_vary_gamma.pdf")

    # --------------------------------------------------------------------
    # --- Scatter-line Plot of Spectral Entropy k=1, z=100, vary gamma ---
    # --------------------------------------------------------------------
    ents = [entropies[(1, gamma, 100)] for gamma in gammas]
    labels = [f"$\gamma$={gamma}" for gamma in gammas]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    # as scatter + line plots
    # ax.plot(range(1, len(gammas) + 1), ents, marker="o", markersize=10, color="slateblue", linewidth=2)
    ax.plot(gammas, ents, marker="o", markersize=10, color="slateblue", linewidth=2)
    beautify_axis(ax, x_limits=(min(gammas), max(gammas)), y_limits=(0, 1), xticks_int=False, decimals=1)
    ax.set_title("Spectral Entropy", fontsize=title_size)
    ax.set_ylabel("Entropy", fontsize=label_size)
    ax.set_xlabel("$\gamma$", fontsize=label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    fig.tight_layout()
    fig.savefig(PLOT_SAVE_PATH / "spectral_entropy_vary_gamma.pdf")

    # --------------------------------------------------------------------
    # --- Scatter-line Plot of Stable Rank k=1, z=100, vary gamma ---
    # --------------------------------------------------------------------
    srs = [sranks[(1, gamma, 100)] for gamma in gammas]
    labels = [f"$\gamma$={gamma}" for gamma in gammas]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    # as scatter + line plots
    ax.plot(gammas, srs, marker="o", markersize=10, color="slateblue", linewidth=2)
    beautify_axis(ax, x_limits=(min(gammas), max(gammas)), y_limits=(0, 15), xticks_int=False, decimals=1)
    ax.set_title("Stable Rank", fontsize=title_size)
    ax.set_ylabel("Rank", fontsize=label_size)
    ax.set_xlabel("$\gamma$", fontsize=label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    fig.tight_layout()
    fig.savefig(PLOT_SAVE_PATH / "stable_rank_vary_gamma.pdf")

    # ---------------------------------------------------
    # --- Plot the Heatmap of spectral Entropy z=100, vary gamma and k ---
    # ---------------------------------------------------
    heatmap_data = np.zeros((len(gammas), len(ks)))
    for i, gamma in enumerate(gammas):
        for j, k in enumerate(ks):
            heatmap_data[i, j] = entropies.get((k, gamma, 100), np.nan)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    img = ax.imshow(heatmap_data, cmap="cool", origin="lower", aspect="auto")
    x_tick_labels = [str(k) for k in ks]
    y_tick_labels = [str(gamma) for gamma in gammas]
    # beautify_axis(ax, xticks_int=True, decimals=2)
    ax.set_xticks(np.arange(len(x_tick_labels)))
    ax.set_yticks(np.arange(len(y_tick_labels)))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)
    ax.spines[["right", "left", "top", "bottom"]].set_visible(False)
    ax.set_title("Spectral Entropy", fontsize=title_size)
    ax.set_xlabel("K", fontsize=label_size)
    ax.set_ylabel("$\gamma$", fontsize=label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.locator_params(nbins=4, tight=True)
    fig.tight_layout()
    fig.savefig(PLOT_SAVE_PATH / "spectral_entropy_heatmap.pdf")

    # ---------------------------------------------------
    # --- Plot the Heatmap of Stable Rank z=100, vary gamma and k ---
    # ---------------------------------------------------
    heatmap_data = np.zeros((len(gammas), len(ks)))
    for i, gamma in enumerate(gammas):
        for j, k in enumerate(ks):
            heatmap_data[i, j] = sranks.get((k, gamma, 100), np.nan)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))
    img = ax.imshow(heatmap_data, cmap="cool", origin="lower", aspect="auto")
    x_tick_labels = [str(k) for k in ks]
    y_tick_labels = [str(gamma) for gamma in gammas]
    # beautify_axis(ax, xticks_int=True, decimals=2)
    ax.set_xticks(np.arange(len(x_tick_labels)))
    ax.set_yticks(np.arange(len(y_tick_labels)))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)
    ax.spines[["right", "left", "top", "bottom"]].set_visible(False)
    ax.set_title("Stable Rank", fontsize=title_size)
    ax.set_xlabel("K", fontsize=label_size)
    ax.set_ylabel("$\gamma$", fontsize=label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.locator_params(nbins=5, tight=True)
    fig.tight_layout()
    fig.savefig(PLOT_SAVE_PATH / "stable_rank_heatmap.pdf")


if __name__ == "__main__":
    main()
