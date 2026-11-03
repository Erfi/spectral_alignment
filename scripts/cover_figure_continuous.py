"""
We want to create the necessary functions for:
1. Loading a models trained on four room environment
2. Calculating SR over a grid of goal states for a given start state (averaged over cardinal actions)
3. Calculating Value function over a grid of start states for a given goal state (averaged over cardinal actions)
4. Average the SR and V values over seeds
5. Plot the resulting SR and V heatmaps with matplotlib and include the walls, start state (SR) and goal state (V) on the plot.
"""

import logging
from pathlib import Path
from glob import glob

import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

from spectral_alignment.utils.common_utils import load_model_config


logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("matplotlib.pyplot").disabled = True
logging.getLogger("PIL.PngImagePlugin").disabled = True


def calculate_SR_heatmap(env, algo, resolution=100, start_ij=(1, 1)):
    action_up = torch.tensor([0.0, 1.0], dtype=torch.float32, device=algo.device).unsqueeze(0)
    action_down = torch.tensor([0.0, -1.0], dtype=torch.float32, device=algo.device).unsqueeze(0)
    action_left = torch.tensor([-1.0, 0.0], dtype=torch.float32, device=algo.device).unsqueeze(0)
    action_right = torch.tensor([1.0, 0.0], dtype=torch.float32, device=algo.device).unsqueeze(0)
    actions = [action_up, action_down, action_left, action_right]
    env_map = env.unwrapped.maze_map  # 9 x 9 map of zeros and ones, wall = 1, free space = 0
    min_x, min_y = env.unwrapped.ij_to_xy((0, 0))  # top-left corner
    max_x, max_y = env.unwrapped.ij_to_xy((env_map.shape[0] - 1, env_map.shape[1] - 1))  # bottom-right corner
    xs = np.linspace(min_x, max_x, resolution, dtype=np.float32)
    ys = np.linspace(min_y, max_y, resolution, dtype=np.float32)

    coords = []  # (x, y)
    for x in xs:  # columns ( x-axis)
        for y in ys:  # rows (y-axis)
            coords.append([x, y])
    coords = np.array(coords, dtype=np.float32)

    SR = np.zeros((resolution, resolution, len(actions)), dtype=np.float32)
    START_COORD = env.unwrapped.ij_to_xy(start_ij)

    # Move to torch and normalize observations in batch
    states = torch.tensor(coords, dtype=torch.float32, device=algo.device)
    states = algo.obs_normalizer.normalize(states)
    start = torch.tensor(START_COORD, dtype=torch.float32, device=algo.device).unsqueeze(0)
    start = algo.obs_normalizer.normalize(start)

    with torch.inference_mode():
        zs = algo.backwardnet(states)  # shape (num_states, latent_dim)
        zs = algo.project_z(zs)  # shape (num_states, latent_dim)
        start_expanded = start.expand(len(zs), -1)  # shape (num_states, obs_dim)
        for k, action in enumerate(actions):
            action_expanded = action.expand(len(zs), -1)  # shape (num_states, action_dim)
            # compute F for (START, action, zs)
            fs = algo.forwardnet(start_expanded, action_expanded, zs)  # shape ((ensembles), num_states, latent_dim)
            if fs.ndim == 3 and fs.shape[0] == algo.forwardnet.ensemble_size:
                fs = fs.mean(dim=0)  # shape (num_states, latent_dim)
            # compute SR for this action
            sr_values = torch.matmul(fs, zs.T)  # shape (num_states, num_states)
            # We want the SR(START, action) which is the diagonal elements
            sr_action = torch.diagonal(sr_values)  # shape (num_states,)
            # Store the SR values
            SR[:, :, k] = sr_action.cpu().numpy().reshape(resolution, resolution)
        SR = SR.mean(axis=2)  # average over actions to get SR(START) -> shape (resolution, resolution)
    return SR


def calculate_Q_heatmap(env, algo, resolution=100, goal_ij=(1, 1)):
    action_up = torch.tensor([0.0, 1.0], dtype=torch.float32, device=algo.device).unsqueeze(0)
    action_down = torch.tensor([0.0, -1.0], dtype=torch.float32, device=algo.device).unsqueeze(0)
    action_left = torch.tensor([-1.0, 0.0], dtype=torch.float32, device=algo.device).unsqueeze(0)
    action_right = torch.tensor([1.0, 0.0], dtype=torch.float32, device=algo.device).unsqueeze(0)
    actions = [action_up, action_down, action_left, action_right]
    env_map = env.unwrapped.maze_map  # wall = 1, free space = 0
    min_x, min_y = env.unwrapped.ij_to_xy((0, 0))  # top-left corner
    max_x, max_y = env.unwrapped.ij_to_xy((env_map.shape[0] - 1, env_map.shape[1] - 1))  # bottom-right corner
    xs = np.linspace(min_x, max_x, resolution, dtype=np.float32)
    ys = np.linspace(min_y, max_y, resolution, dtype=np.float32)

    coords = []  # (x, y)
    for x in xs:  # columns ( x-axis)
        for y in ys:  # rows (y-axis)
            coords.append([x, y])
    coords = np.array(coords, dtype=np.float32)
    Q = np.zeros((resolution, resolution, len(actions)), dtype=np.float32)
    GOAL_COORD = env.unwrapped.ij_to_xy(goal_ij)

    coords = np.array(coords, dtype=np.float32)
    # Move to torch and normalize observations in batch
    states = torch.tensor(coords, dtype=torch.float32, device=algo.device)
    states = algo.obs_normalizer.normalize(states)
    goal = torch.tensor(GOAL_COORD, dtype=torch.float32, device=algo.device).unsqueeze(0)
    goal = algo.obs_normalizer.normalize(goal)

    with torch.inference_mode():
        z = algo.backwardnet(goal)  # shape (1, latent_dim)
        z = algo.project_z(z)  # shape (1, latent_dim)
        z_expanded = z.expand(len(states), -1)  # shape (num_states, latent_dim)
        for k, action in enumerate(actions):
            action_expanded = action.expand(len(states), -1)  # shape (num_states, action_dim)
            # compute F for (ALL_STATES, action, zs)
            fs = algo.forwardnet(states, action_expanded, z_expanded)  # shape ((ensembles), num_states, latent_dim)
            if fs.ndim == 3 and fs.shape[0] == algo.forwardnet.ensemble_size:
                fs = fs.mean(dim=0)  # shape (num_states, latent_dim)
            # compute Q values for this action
            q_values = torch.sum(fs * z_expanded, dim=1)  # shape (num_states,)
            # Store the Q values
            Q[:, :, k] = q_values.cpu().numpy().reshape(resolution, resolution)
        Q = Q.mean(axis=2)  # average over actions to get Q(GOAL) -> shape (resolution, resolution)
    return Q


def calculate_wall_mask(env, resolution: int):
    env_map = env.unwrapped.maze_map  # wall = 1, free space = 0
    min_x, min_y = env.unwrapped.ij_to_xy((0, 0))  # top-left corner
    max_x, max_y = env.unwrapped.ij_to_xy((env_map.shape[0] - 1, env_map.shape[1] - 1))  # bottom-right corner
    xs = np.linspace(min_x, max_x, resolution, dtype=np.float32)
    ys = np.linspace(min_y, max_y, resolution, dtype=np.float32)
    wall_mask = np.zeros((resolution, resolution), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            x_coord, y_coord = x, y
            # Check if this coordinate is a wall by converting back to ij and checking the maze map
            ij = env.unwrapped.xy_to_ij((x_coord, y_coord))
            if env_map[ij] == 1:  # it's a wall
                wall_mask[j, i] = True  # note the order of indices for imshow (row, col)
    return wall_mask


def load_algo_and_env(model_path: str):
    saved_config = load_model_config(Path(model_path).parent)
    env = hydra.utils.instantiate(saved_config.environment, render_mode="rgb_array")
    algo = hydra.utils.instantiate(saved_config.algorithm, env=env, device="cuda:0" if torch.cuda.is_available() else "cpu")
    algo.load(model_path=model_path)
    return algo, env


def add_star_marker(ax, start, env, resolution, size=100, offset=[0.0, 0.0], marker="*"):
    # Mark the start or goal state
    start_i, start_j = start
    start_i_plot = resolution - 1 - int(start_i * (resolution - 1) / (env.unwrapped.maze_map.shape[0] - 1))
    start_j_plot = int(start_j * (resolution - 1) / (env.unwrapped.maze_map.shape[1] - 1))
    ax.scatter(
        start_j_plot + offset[0],
        start_i_plot + offset[1],
        marker=marker,
        s=size,
        c="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=10,
    )


def calculate_mean_SR_over_seeds(experiment_path: Path | str, resolution=100, start_ij=(1, 1)):
    path = glob(str(Path(experiment_path) / "seed_*" / "**" / "model_top_1.zip"), recursive=True)
    SR_list = []
    wall_mask = None
    for p in path:
        algo, env = load_algo_and_env(p)
        SR = calculate_SR_heatmap(env, algo, resolution=resolution, start_ij=start_ij)
        SR_list.append(SR)
    SR_avg = np.mean(np.array(SR_list), axis=0)
    return SR_avg


def calculate_mean_Q_over_seeds(experiment_path: Path | str, resolution=100, goal_ij=(1, 1)):
    path = glob(str(Path(experiment_path) / "seed_*" / "**" / "model_top_1.zip"), recursive=True)
    Q_list = []
    wall_mask = None
    for p in path:
        algo, env = load_algo_and_env(p)
        Q = calculate_Q_heatmap(env, algo, resolution=resolution, goal_ij=goal_ij)
        Q_list.append(Q)
    Q_avg = np.mean(np.array(Q_list), axis=0)
    return Q_avg


def main():
    RESOLUTION = 100
    START = GOAL = (2, 2)
    STAR_OFFSET = [2.0, 0.0]
    PLOT_SAVE_PATH = Path("plots/cover_figure")
    PLOT_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_PATH = Path("models/FBRL/RoomsEnv/rbf__act-rep-1__gamma-0.95__zdim-200")
    MODEL_PATHS = glob(str(EXPERIMENT_PATH / "seed_*" / "**" / "model_top_1.zip"), recursive=True)

    _, env = load_algo_and_env(MODEL_PATHS[0])
    wall_mask = calculate_wall_mask(env, resolution=RESOLUTION)
    SR = calculate_mean_SR_over_seeds(EXPERIMENT_PATH, resolution=RESOLUTION, start_ij=START)
    Q = calculate_mean_Q_over_seeds(EXPERIMENT_PATH, resolution=RESOLUTION, goal_ij=GOAL)

    # Plotting
    ncolors = 16
    quantile_low = 0.2
    quantile_high = 0.95

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot SR heatmap
    vals = SR.ravel()
    vmin, vmax = np.quantile(vals, [quantile_low, quantile_high])
    edges = np.linspace(vmin, vmax, ncolors + 1)
    cmap = colors.ListedColormap(cm.pink(np.linspace(0, 1, ncolors)))
    norm = colors.BoundaryNorm(edges, ncolors=ncolors, clip=True)

    axs[0].set_facecolor((0.9, 0.9, 0.9))  # set background color for walls
    SR_masked = np.ma.array(np.flipud(SR), mask=np.flipud(wall_mask))  # mask walls so they appear as light gray
    axs[0].pcolor(SR_masked, cmap=cmap, norm=norm, edgecolors="lightgray", linewidth=0.3)
    axs[0].set_title("Successor Representation (SR)")
    axs[0].spines[["right", "top", "left", "bottom"]].set_visible(False)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    add_star_marker(axs[0], START, env, RESOLUTION, offset=STAR_OFFSET)

    # Plot Q heatmap
    vals = Q.ravel()
    vmin, vmax = np.quantile(vals, [quantile_low, quantile_high])
    edges = np.linspace(vmin, vmax, ncolors + 1)
    cmap = colors.ListedColormap(cm.pink(np.linspace(0, 1, ncolors)))
    norm = colors.BoundaryNorm(edges, ncolors=ncolors, clip=True)

    axs[1].set_facecolor((0.9, 0.9, 0.9))  # set background color for walls
    Q_masked = np.ma.array(np.flipud(Q), mask=np.flipud(wall_mask))  # mask walls so they appear as light gray
    axs[1].pcolor(Q_masked, cmap=cmap, norm=norm, edgecolors="lightgray", linewidth=0.3)
    axs[1].set_title("Value Function (Q)")
    axs[1].spines[["right", "top", "left", "bottom"]].set_visible(False)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    add_star_marker(axs[1], GOAL, env, RESOLUTION, offset=STAR_OFFSET)

    fig.savefig(PLOT_SAVE_PATH / f"{EXPERIMENT_PATH.name}.png")


if __name__ == "__main__":
    main()
