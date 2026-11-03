"""
This module is for FBRL experiments and environments.
This module is responsible for visualizing data in various formats.
It includes functions to create plots, charts, and other visual representations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from spectral_alignment.utils.visualize_utils_FBRL import (
    get_validation_data,
    get_evaluation_data,
    plot_validation_data,
    plot_evaluation_data,
    plot_bellman_error,
    plot_ablation_grid,
    plot_parallel_coordinate_plot,
    plot_effective_vs_nominal_gamma,
)

# common plot settings
labelsize = 20
tick_labelsize = 16
legend_labelsize = 14
figsize = (6, 4)

# YOU CAN ADD MORE ENVIRONMENTS HERE
environments = ["RoomsEnv", "MazeEnv", "LargeMazeEnv", "CubeEnv", "SceneEnv"]
# YOU CAN ADD MORE ALGORITHMS HERE
algorithms = ["FBRL"]
# YOU CAN ADD MORE EXPERIMENTS HERE
experiments = [
    "act-rep-1__gamma-0.9__zdim-100",
    "act-rep-1__gamma-0.95__zdim-100",
    "act-rep-1__gamma-0.98__zdim-100",
    "act-rep-1__gamma-0.99__zdim-100",
    "act-rep-1__gamma-0.999__zdim-100",
    "act-rep-2__gamma-0.95__zdim-100",
    "act-rep-2__gamma-0.98__zdim-100",
    "act-rep-3__gamma-0.9__zdim-100",
    "act-rep-3__gamma-0.95__zdim-100",
    "act-rep-3__gamma-0.98__zdim-100",
    "act-rep-3__gamma-0.99__zdim-100",
    "act-rep-3__gamma-0.999__zdim-100",
    "act-rep-4__gamma-0.95__zdim-100",
    "act-rep-4__gamma-0.98__zdim-100",
    "act-rep-5__gamma-0.9__zdim-100",
    "act-rep-5__gamma-0.95__zdim-100",
    "act-rep-5__gamma-0.98__zdim-100",
    "act-rep-5__gamma-0.99__zdim-100",
    "act-rep-5__gamma-0.999__zdim-100",
    "act-rep-10__gamma-0.9__zdim-100",
    "act-rep-10__gamma-0.95__zdim-100",
    "act-rep-10__gamma-0.99__zdim-100",
    "act-rep-10__gamma-0.999__zdim-100",
    "act-rep-20__gamma-0.9__zdim-100",
    "act-rep-20__gamma-0.95__zdim-100",
    "act-rep-20__gamma-0.99__zdim-100",
    "act-rep-20__gamma-0.999__zdim-100",
    "act-rep-50__gamma-0.9__zdim-100",
    "act-rep-50__gamma-0.95__zdim-100",
    "act-rep-50__gamma-0.99__zdim-100",
    "act-rep-50__gamma-0.999__zdim-100",
    "act-rep-1__gamma-0.95__zdim-25",
    "act-rep-1__gamma-0.95__zdim-50",
    "act-rep-1__gamma-0.95__zdim-200",
    "act-rep-1__gamma-0.95__zdim-400",
    "act-rep-3__gamma-0.95__zdim-25",
    "act-rep-3__gamma-0.95__zdim-50",
    "act-rep-3__gamma-0.95__zdim-200",
    "act-rep-3__gamma-0.95__zdim-400",
    "act-rep-5__gamma-0.95__zdim-25",
    "act-rep-5__gamma-0.95__zdim-50",
    "act-rep-5__gamma-0.95__zdim-200",
    "act-rep-5__gamma-0.95__zdim-400",
    "act-rep-10__gamma-0.95__zdim-25",
    "act-rep-10__gamma-0.95__zdim-50",
    "act-rep-10__gamma-0.95__zdim-200",
    "act-rep-10__gamma-0.95__zdim-400",
    "act-rep-20__gamma-0.95__zdim-25",
    "act-rep-20__gamma-0.95__zdim-50",
    "act-rep-20__gamma-0.95__zdim-200",
    "act-rep-20__gamma-0.95__zdim-400",
    "act-rep-50__gamma-0.95__zdim-25",
    "act-rep-50__gamma-0.95__zdim-50",
    "act-rep-50__gamma-0.95__zdim-200",
    "act-rep-50__gamma-0.95__zdim-400",
    "act-rep-3__gamma-0.95__zdim-100__no-demo",
    "visual__act-rep-1__gamma-0.95__zdim-100",
    "visual__act-rep-10__gamma-0.95__zdim-100",
    "act-chunk-1__gamma-0.95__zdim-100",
    "act-chunk-2__gamma-0.95__zdim-100",
    "act-chunk-3__gamma-0.95__zdim-100",
    "act-chunk-4__gamma-0.95__zdim-100",
    "act-chunk-5__gamma-0.95__zdim-100",
    "act-chunk-10__gamma-0.95__zdim-100",
    "act-chunk-20__gamma-0.95__zdim-100",
    "act-chunk-50__gamma-0.95__zdim-100",
]
# Navigation envs (rooms / maze / largemaze) name runs by their state encoder: rbf__..., mlp__..., visual__... (CNN).
# Manipulation envs (cube / scene) always use an MLP encoder and keep the plain names above.
experiments += [f"rbf__{exp}" for exp in experiments if not exp.startswith("visual__")]

validation_data = get_validation_data(
    models_folder="./models", environments=environments, algorithms=algorithms, experiments=experiments, min_steps=1e5
)
evaluation_data = get_evaluation_data(
    models_folder="./models", environments=environments, algorithms=algorithms, experiments=experiments
)
# --------------------------------------------
# --- line plot - z25 - gamma0.95 - vary K ---
# --------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 6))
experiments_plot_info = {
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-25": {"label": "k=1", "color": color_list[0]},
    "roomsenv__fbrl__rbf__act-rep-3__gamma-0.95__zdim-25": {"label": "k=3", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-5__gamma-0.95__zdim-25": {"label": "k=5", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-25": {"label": "k=10", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-20__gamma-0.95__zdim-25": {"label": "k=20", "color": color_list[4]},
    "roomsenv__fbrl__rbf__act-rep-50__gamma-0.95__zdim-25": {"label": "k=50", "color": color_list[5]},
}
plot_validation_data(
    validation_data=validation_data,
    save_path="./plots/RoomsEnv_z25_gamma0.95_vary_k",
    experiments=experiments_plot_info,
    normalized=False,
    title="",
    validation_frequency=10000,
    legend_loc="upper left",
    figsize=figsize,
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
)

# --------------------------------------------
# --- line plot - k=1 - gamma0.95 - vary zdim ---
# --------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 5))
experiments_plot_info = {
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-25": {"label": "d=25", "color": color_list[0]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-50": {"label": "d=50", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-100": {"label": "d=100", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-200": {"label": "d=200", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-400": {"label": "d=400", "color": color_list[4]},
}
plot_validation_data(
    validation_data=validation_data,
    save_path="./plots/RoomsEnv_k1_gamma0.95_vary_zdim",
    experiments=experiments_plot_info,
    normalized=False,
    title="",
    validation_frequency=10000,
    legend_loc="upper left",
    figsize=figsize,
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
)

# --------------------------------------------
# --- line plot - k=1 - zdim=100 - vary gamma ---
# --------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 4))
experiments_plot_info = {
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.9__zdim-100": {"label": "$\gamma$=0.9", "color": color_list[0]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-100": {"label": "$\gamma$=0.95", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.99__zdim-100": {"label": "$\gamma$=0.99", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.999__zdim-100": {"label": "$\gamma$=0.999", "color": color_list[3]},
}
plot_validation_data(
    validation_data=validation_data,
    save_path="./plots/RoomsEnv_k1_zdim100_vary_gamma",
    experiments=experiments_plot_info,
    normalized=False,
    title="",
    validation_frequency=10000,
    legend_loc="upper left",
    figsize=figsize,
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
)

# ---------------------------------------------
# --- line plot - z400 - gamma0.95 - vary K ---
# ---------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 6))
experiments_plot_info = {
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-400": {"label": "k=1", "color": color_list[0]},
    "roomsenv__fbrl__rbf__act-rep-3__gamma-0.95__zdim-400": {"label": "k=3", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-5__gamma-0.95__zdim-400": {"label": "k=5", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-400": {"label": "k=10", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-20__gamma-0.95__zdim-400": {"label": "k=20", "color": color_list[4]},
    "roomsenv__fbrl__rbf__act-rep-50__gamma-0.95__zdim-400": {"label": "k=50", "color": color_list[5]},
}
plot_validation_data(
    validation_data=validation_data,
    save_path="./plots/RoomsEnv_z400_gamma0.95_vary_k",
    experiments=experiments_plot_info,
    normalized=False,
    title="",
    validation_frequency=10000,
    legend_loc="upper left",
    figsize=figsize,
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
)

# --------------------------------------------
# --- line plot - z100 - gamma0.9 - vary K ---
# --------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 6))
experiments_plot_info = {
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.9__zdim-100": {"label": "k=1", "color": color_list[0]},
    "roomsenv__fbrl__rbf__act-rep-3__gamma-0.9__zdim-100": {"label": "k=3", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-5__gamma-0.9__zdim-100": {"label": "k=5", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.9__zdim-100": {"label": "k=10", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-20__gamma-0.9__zdim-100": {"label": "k=20", "color": color_list[4]},
    "roomsenv__fbrl__rbf__act-rep-50__gamma-0.9__zdim-100": {"label": "k=50", "color": color_list[5]},
}
plot_validation_data(
    validation_data=validation_data,
    save_path="./plots/RoomsEnv_z100_gamma0.9_vary_k",
    experiments=experiments_plot_info,
    normalized=False,
    title="",
    validation_frequency=10000,
    legend_loc="upper left",
    figsize=figsize,
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
)


# ---------------------------------------------------------------------------------------
# --- Bellman error boxplots (RoomsEnv) ---
# Training metrics are fetched from wandb with:
#   python scripts/fetch_wandb_metrics.py --entity <your_entity> --env RoomsEnv
# which writes one column per run named "<experiment_name>_<seed> - train/<metric>".
# ---------------------------------------------------------------------------------------
TRAIN_METRICS_PATH = Path("models/FBRL/RoomsEnv/misc/train_metrics.csv")
if TRAIN_METRICS_PATH.exists():
    train_metrics = pd.read_csv(TRAIN_METRICS_PATH)
else:
    print(f"WARNING: {TRAIN_METRICS_PATH} not found, Bellman error plots will be empty (see README).")
    train_metrics = pd.DataFrame()
SEEDS = [0, 1, 2, 3, 4]


def metric_values(experiment, metric="fb_loss_bellman_offdiag", normalize_by=None):
    """All logged values of a train metric over the seeds of an experiment, optionally divided by |normalize_by|."""
    cols = [f"{experiment}_{seed} - train/{metric}" for seed in SEEDS]
    cols = [col for col in cols if col in train_metrics]
    values = train_metrics[cols].values
    if normalize_by is not None:
        values = values / np.abs(train_metrics[[col.replace(metric, normalize_by) for col in cols]].values)
    values = values.ravel()
    return values[~np.isnan(values)]


# --- Boxplot - Bellman error distribution - z100 - k1 - gammas[0.9, 0.95, 0.99,0.999] ---
gammas = ["0.9", "0.95", "0.99", "0.999"]
plot_bellman_error(
    data=[metric_values(f"rbf__act-rep-1__gamma-{gamma}__zdim-100") for gamma in gammas],
    labels=gammas,
    x_label="Discount Factor ($\\gamma$)",
    y_label="Bellman Error",
    title="",
    save_path="./plots/RoomsEnv_bellman_error_boxplot",
    figsize=figsize,
    labelsize=labelsize + 2,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
    y_limits=(0.0, 1200.0),
)

# --- Normalized Bellman error ---
plot_bellman_error(
    data=[metric_values(f"rbf__act-rep-1__gamma-{gamma}__zdim-100", normalize_by="actor_loss") for gamma in gammas],
    labels=gammas,
    x_label="Discount Factor ($\\gamma$)",
    y_label="Normalized \nBellman Error",
    title="",
    save_path="./plots/RoomsEnv_bellman_error_normalized_boxplot",
    figsize=figsize,
    labelsize=labelsize + 2,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
    y_limits=(0.0, 6.0),
)


# --- Boxplot - Bellman error distribution - z100 - gamma0.95 - k=[1,3,5,10,20,50] ---
ks = ["1", "3", "5", "10", "20", "50"]
plot_bellman_error(
    data=[metric_values(f"rbf__act-rep-{k}__gamma-0.95__zdim-100") for k in ks],
    labels=ks,
    x_label="Temporal Abstraction (k)",
    y_label="Bellman Error",
    title="",
    save_path="./plots/RoomsEnv_bellman_error_vary_k_boxplot",
    figsize=figsize,
    labelsize=labelsize + 2,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
    y_limits=(0.0, 300.0),
)

# --- Normalized Bellman error ---
plot_bellman_error(
    data=[metric_values(f"rbf__act-rep-{k}__gamma-0.95__zdim-100", normalize_by="actor_loss") for k in ks],
    labels=ks,
    x_label="Temporal Abstraction (k)",
    y_label="Normalized \nBellman Error",
    title="",
    save_path="./plots/RoomsEnv_bellman_error_normalized_vary_k_boxplot",
    figsize=figsize,
    labelsize=labelsize + 2,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
    y_limits=(0.0, 3.0),
)

# --- Boxplot - Bellman error distribution - gamma0.95 - k=1 - vary z ---
zdims = ["25", "50", "100", "200", "400"]
plot_bellman_error(
    data=[metric_values(f"rbf__act-rep-1__gamma-0.95__zdim-{z}") for z in zdims],
    labels=zdims,
    x_label="Embedding Dimension (d)",
    y_label="Bellman Error",
    title="",
    save_path="./plots/RoomsEnv_bellman_error_vary_z_boxplot",
    figsize=figsize,
    labelsize=labelsize + 2,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
    y_limits=(0.0, 350.0),
)

# --- Normalized Bellman error ---
plot_bellman_error(
    data=[metric_values(f"rbf__act-rep-1__gamma-0.95__zdim-{z}", normalize_by="actor_loss") for z in zdims],
    labels=zdims,
    x_label="Embedding Dimension (d)",
    y_label="Normalized \nBellman Error",
    title="",
    save_path="./plots/RoomsEnv_bellman_error_normalized_vary_z_boxplot",
    figsize=figsize,
    labelsize=labelsize + 2,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
    y_limits=(0.0, 4.0),
)

# -------------------------------------------
# --- Grid Plot - z100 - vary K and gamma ---
# -------------------------------------------
ks = [1, 3, 5, 10, 20, 50]
gammas = [0.9, 0.95, 0.99, 0.999]
grid = np.zeros((len(gammas), len(ks)))
for i, gamma in enumerate(gammas):
    for j, k in enumerate(ks):
        exp_key = f"roomsenv__fbrl__rbf__act-rep-{k}__gamma-{gamma}__zdim-100"
        if exp_key in evaluation_data:
            grid[i, j] = evaluation_data[exp_key]["mean"]
        else:
            print(f"WARNING: Experiment {exp_key} not found in evaluation data.")
            grid[i, j] = np.nan
plot_ablation_grid(
    grid=grid,
    x_ticklabels=[str(k) for k in ks],
    y_ticklabels=[str(gamma) for gamma in gammas],
    x_label="Temporal Abstraction (k)",
    y_label="Discount Factor ($\\gamma$)",
    title="",
    save_path="./plots/RoomsEnv_k_gamma_ablation_grid",
    figsize=(8, 4),
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
)

# --------------------------------------------
# --- Grid Plot - gamma0.95 - vary K and z ---
# --------------------------------------------
ks = [1, 3, 5, 10, 20, 50]
zdims = [25, 50, 100, 200, 400]
grid = np.zeros((len(zdims), len(ks)))
for i, zdim in enumerate(zdims):
    for j, k in enumerate(ks):
        exp_key = f"roomsenv__fbrl__rbf__act-rep-{k}__gamma-0.95__zdim-{zdim}"
        if exp_key in evaluation_data:
            grid[i, j] = evaluation_data[exp_key]["mean"]
        else:
            print(f"WARNING: Experiment {exp_key} not found in evaluation data.")
            grid[i, j] = np.nan
plot_ablation_grid(
    grid=grid,
    x_ticklabels=[str(k) for k in ks],
    y_ticklabels=[str(zdim) for zdim in zdims],
    x_label="Temporal Abstraction (k)",
    y_label="Embed Dimension (d)",
    title="",
    save_path="./plots/RoomsEnv_k_zdim_ablation_grid",
    figsize=figsize,
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
)


# ------------------------------------------------------
# --- Bar Plot - CNN vs RBF - gamma0.95 - k10 - z100 ---
# ------------------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 4))
experiments_plot_info = {
    "roomsenv__fbrl__visual__act-rep-10__gamma-0.95__zdim-100": {
        "label": "FourRooms",
        "color": color_list[1],
        "legend": "Image",
    },
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-100": {
        "label": "FourRooms",
        "color": color_list[2],
        "legend": "State",
    },
    "mazeenv__fbrl__visual__act-rep-10__gamma-0.95__zdim-100": {"label": "Maze", "color": color_list[1]},
    "mazeenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-100": {"label": "Maze", "color": color_list[2]},
    "largemazeenv__fbrl__visual__act-rep-10__gamma-0.95__zdim-100": {"label": "LargeMaze", "color": color_list[1]},
    "largemazeenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-100": {"label": "LargeMaze", "color": color_list[2]},
}
plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/RoomsEnv_cnn_vs_rbf_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=2,
    title="",
    x_label="",
    y_label="Mean Return",
    figsize=figsize,
    labelsize=labelsize + 2,
    tick_labelsize=tick_labelsize + 4,
    legend_labelsize=legend_labelsize + 2,
    legend_cols=2,
)

# --------------------------------------------------------
# --- Bar Plot - CNN sweep - gamma0.95 - z100 - k-[1,10]---
# --------------------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 4))
experiments_plot_info = {
    "mazeenv__fbrl__visual__act-rep-1__gamma-0.95__zdim-100": {"label": "Maze", "color": color_list[1]},
    "mazeenv__fbrl__visual__act-rep-10__gamma-0.95__zdim-100": {"label": "Maze", "color": color_list[2]},
    "largemazeenv__fbrl__visual__act-rep-1__gamma-0.95__zdim-100": {"label": "LargeMaze", "color": color_list[1]},
    "largemazeenv__fbrl__visual__act-rep-10__gamma-0.95__zdim-100": {"label": "LargeMaze", "color": color_list[2]},
}
plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/cnn_sweep_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=2,
    title="",
    x_label="",
    y_label="Mean Return",
    figsize=figsize,
    labelsize=labelsize + 2,
    tick_labelsize=tick_labelsize + 4,
    legend_labelsize=legend_labelsize + 2,
    legend_cols=2,
)

# ------------------------------------------------------------------------
# --- Parallel Coordinate Plot - mapping of k, gamma, z to mean return ---
# ------------------------------------------------------------------------
plot_parallel_coordinate_plot(
    evaluation_data=evaluation_data,
    save_path="./plots/RoomsEnv_parallel_coordinates_plot",
    figsize=(6, 4),
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
)


# -------------------------------------------------------------------------------------------------------
# --- Line Plot - Two extremes of temporal abstraction (k=1 vs k=10) grouping over z-dim - gamma=0.95 ---
# -------------------------------------------------------------------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 12))
experiments_plot_info = {
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-25": {"label": "k=1, d=25", "color": color_list[0]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-50": {"label": "k=1, d=50", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-100": {"label": "k=1, d=100", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-200": {"label": "k=1, d=200", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-400": {"label": "k=1, d=400", "color": color_list[4]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-25": {"label": "k=10, d=25", "color": color_list[-5]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-50": {"label": "k=10, d=50", "color": color_list[-4]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-100": {"label": "k=10, d=100", "color": color_list[-3]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-200": {"label": "k=10, d=200", "color": color_list[-2]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-400": {"label": "k=10, d=400", "color": color_list[-1]},
}
plot_validation_data(
    validation_data=validation_data,
    save_path="./plots/RoomsEnv_gamma0.95_k_high_low_vary_zdim",
    experiments=experiments_plot_info,
    normalized=False,
    title="",
    validation_frequency=10000,
    legend_loc="upper left",
    figsize=(10, 7),
    labelsize=24,
    tick_labelsize=20,
    legend_labelsize=17,
)

# -------------------------------------------------------------------------------------------------------
# --- Line Plot - Two extremes of temporal abstraction (k=1 vs k=10) grouping over gamma - zdim=100   ---
# -------------------------------------------------------------------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 10))
experiments_plot_info = {
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.9__zdim-100": {"label": "k=1, $\gamma$=0.9", "color": color_list[0]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-100": {"label": "k=1, $\gamma$=0.95", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.99__zdim-100": {"label": "k=1, $\gamma$=0.99", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.999__zdim-100": {"label": "k=1, $\gamma$=0.999", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.9__zdim-100": {"label": "k=10, $\gamma$=0.9", "color": color_list[-4]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-100": {"label": "k=10, $\gamma$=0.95", "color": color_list[-3]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.99__zdim-100": {"label": "k=10, $\gamma$=0.99", "color": color_list[-2]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.999__zdim-100": {"label": "k=10, $\gamma$=0.999", "color": color_list[-1]},
}
plot_validation_data(
    validation_data=validation_data,
    save_path="./plots/RoomsEnv_z100_k_high_low_vary_gamma",
    experiments=experiments_plot_info,
    normalized=False,
    title="",
    validation_frequency=10000,
    legend_loc="upper left",
    figsize=(10, 7),
    labelsize=24,
    tick_labelsize=20,
    legend_labelsize=17,
)

# ---------------------------------------------------------------------------------
# --- Bar Plot - Diminishing return of increasing z-dim - gamma0.95 - k1 vs k10 ---
# ---------------------------------------------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 4))
experiments_plot_info = {
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-25": {"label": "25", "color": color_list[1], "legend": "k=1"},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-25": {"label": "25", "color": color_list[2], "legend": "k=10"},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-50": {"label": "50", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-50": {"label": "50", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-100": {"label": "100", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-100": {"label": "100", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-200": {"label": "200", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-200": {"label": "200", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-400": {"label": "400", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-400": {"label": "400", "color": color_list[2]},
}
plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/Roomsenv_diminishing_returns_capacity_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=2,
    title="",
    x_label="Embedding Dimension (d)",
    y_label="Mean Return",
    figsize=figsize,
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
)

# ----------------------------------------------------------------------------
# --- Bar Plot - Diminishing return of increasing gamma - k1 vs k10 - z100 ---
# ----------------------------------------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 4))
experiments_plot_info = {
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.9__zdim-100": {"label": "0.9", "color": color_list[1], "legend": "k=1"},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.9__zdim-100": {"label": "0.9", "color": color_list[2], "legend": "k=10"},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-100": {"label": "0.95", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-100": {"label": "0.95", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.99__zdim-100": {"label": "0.99", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.99__zdim-100": {"label": "0.99", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.999__zdim-100": {"label": "0.999", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.999__zdim-100": {"label": "0.999", "color": color_list[2]},
}
plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/Roomsenv_diminishing_return_gamma_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=2,
    title="",
    x_label="Discount Factor ($\\gamma$)",
    y_label="Mean Return",
    figsize=figsize,
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
)

# ----------------------------------------------------------------------------
# --- scatter-Plot - nominal vs effective gamma conversion and performance ---
# ----------------------------------------------------------------------------
plot_effective_vs_nominal_gamma(
    evaluation_data=evaluation_data,
    save_path="./plots/RoomsEnv_effective_vs_nominal_gamma_scatter",
    figsize=figsize,
    labelsize=labelsize + 2,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize + 1,
)


# --------------------------------------------
# --- Bar Plot - gamma0.95 - z100 - vary K ---
# --------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 3))
experiments_plot_info = {
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-100": {"label": "1", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-3__gamma-0.95__zdim-100": {"label": "3", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-5__gamma-0.95__zdim-100": {"label": "5", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-100": {"label": "10", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-20__gamma-0.95__zdim-100": {"label": "20", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-50__gamma-0.95__zdim-100": {"label": "50", "color": color_list[1]},
}
plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/RoomsEnv_gamma0.95_z100_vary_k_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=1,
    title="",
    x_label="Temporal Abstraction (k)",
    y_label="Mean Return",
    figsize=(5, 4),
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize - 0.5,
)

# --------------------------------------------
# --- Bar Plot - gamma0.95 - vary z - vary K ---
# --------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 5))
experiments_plot_info = {
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-25": {"label": "1", "color": color_list[1], "legend": "d=25"},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-100": {"label": "1", "color": color_list[2], "legend": "d=100"},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-400": {"label": "1", "color": color_list[3], "legend": "d=400"},
    "roomsenv__fbrl__rbf__act-rep-3__gamma-0.95__zdim-25": {"label": "3", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-3__gamma-0.95__zdim-100": {"label": "3", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-3__gamma-0.95__zdim-400": {"label": "3", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-5__gamma-0.95__zdim-25": {"label": "5", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-5__gamma-0.95__zdim-100": {"label": "5", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-5__gamma-0.95__zdim-400": {"label": "5", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-25": {"label": "10", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-100": {"label": "10", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-400": {"label": "10", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-20__gamma-0.95__zdim-25": {"label": "20", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-20__gamma-0.95__zdim-100": {"label": "20", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-20__gamma-0.95__zdim-400": {"label": "20", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-50__gamma-0.95__zdim-25": {"label": "50", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-50__gamma-0.95__zdim-100": {"label": "50", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-50__gamma-0.95__zdim-400": {"label": "50", "color": color_list[3]},
}
plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/RoomsEnv_gamma0.95_vary_z_vary_k_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=3,
    title="",
    x_label="Temporal Abstraction (k)",
    y_label="Mean Return",
    figsize=(7, 4),
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize - 0.5,
    legend_loc="upper left",
)

# --------------------------------------------
# --- Bar Plot - z100 - vary gamma - vary K ---
# --------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 5))
experiments_plot_info = {
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.9__zdim-100": {"label": "1", "color": color_list[1], "legend": "$\gamma$=0.9"},
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.99__zdim-100": {
        "label": "1",
        "color": color_list[2],
        "legend": "$\gamma$=0.99",
    },
    "roomsenv__fbrl__rbf__act-rep-1__gamma-0.999__zdim-100": {
        "label": "1",
        "color": color_list[3],
        "legend": "$\gamma$=0.999",
    },
    "roomsenv__fbrl__rbf__act-rep-3__gamma-0.9__zdim-100": {"label": "3", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-3__gamma-0.99__zdim-100": {"label": "3", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-3__gamma-0.999__zdim-100": {"label": "3", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-5__gamma-0.9__zdim-100": {"label": "5", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-5__gamma-0.99__zdim-100": {"label": "5", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-5__gamma-0.999__zdim-100": {"label": "5", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.9__zdim-100": {"label": "10", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.99__zdim-100": {"label": "10", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-10__gamma-0.999__zdim-100": {"label": "10", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-20__gamma-0.9__zdim-100": {"label": "20", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-20__gamma-0.99__zdim-100": {"label": "20", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-20__gamma-0.999__zdim-100": {"label": "20", "color": color_list[3]},
    "roomsenv__fbrl__rbf__act-rep-50__gamma-0.9__zdim-100": {"label": "50", "color": color_list[1]},
    "roomsenv__fbrl__rbf__act-rep-50__gamma-0.99__zdim-100": {"label": "50", "color": color_list[2]},
    "roomsenv__fbrl__rbf__act-rep-50__gamma-0.999__zdim-100": {"label": "50", "color": color_list[3]},
}

plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/RoomsEnv_z100_vary_gamma_vary_k_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=3,
    title="",
    x_label="Temporal Abstraction (k)",
    y_label="Mean Return",
    figsize=(7, 4),
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize - 0.5,
    legend_loc="upper left",
)


# ---------------------------------------------------------------
# --- Bar Plot - CubeEnv - gamma0.95 - z100 - vary K (k=1,2,3,4,5)
# ---------------------------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 5))
experiments_plot_info = {
    "cubeenv__fbrl__act-rep-1__gamma-0.95__zdim-100": {"label": "1", "color": color_list[0]},
    "cubeenv__fbrl__act-rep-2__gamma-0.95__zdim-100": {"label": "2", "color": color_list[1]},
    "cubeenv__fbrl__act-rep-3__gamma-0.95__zdim-100": {"label": "3", "color": color_list[2]},
    "cubeenv__fbrl__act-rep-4__gamma-0.95__zdim-100": {"label": "4", "color": color_list[3]},
    "cubeenv__fbrl__act-rep-5__gamma-0.95__zdim-100": {"label": "5", "color": color_list[4]},
}
plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/CubeEnv_gamma0.95_z100_vary_k_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=1,
    title="",
    x_label="Temporal Abstraction (k)",
    y_label="Mean Return",
    figsize=(5, 4),
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize - 0.5,
)

# ---------------------------------------------------------------------------
# --- Bar Plot - CubeEnv - Demo/No Demo - gamma0.95 - z100 - k3
# ---------------------------------------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 2))
experiments_plot_info = {
    "cubeenv__fbrl__act-rep-3__gamma-0.95__zdim-100__no-demo": {"label": "No Demo", "color": color_list[0]},
    "cubeenv__fbrl__act-rep-3__gamma-0.95__zdim-100": {"label": "Demo", "color": color_list[1]},
}
plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/CubeEnv_no-demo_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=1,
    title="",
    x_label="",
    y_label="Mean Return",
    figsize=(5, 4),
    labelsize=labelsize,
    tick_labelsize=tick_labelsize - 2,
    legend_labelsize=legend_labelsize - 0.5,
)

# ---------------------------------------------------------------
# --- Bar Plot - SceneEnv - gamma0.98 - z100 - vary K (k=1,2,3,4,5)
# ---------------------------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 5))
experiments_plot_info = {
    "sceneenv__fbrl__act-rep-1__gamma-0.99__zdim-100": {"label": "1", "color": color_list[0]},
    "sceneenv__fbrl__act-rep-3__gamma-0.99__zdim-100": {"label": "3", "color": color_list[1]},
    "sceneenv__fbrl__act-rep-5__gamma-0.99__zdim-100": {"label": "5", "color": color_list[2]},
    "sceneenv__fbrl__act-rep-10__gamma-0.99__zdim-100": {"label": "10", "color": color_list[3]},
    "sceneenv__fbrl__act-rep-20__gamma-0.99__zdim-100": {"label": "20", "color": color_list[4]},
}

plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/SceneEnv_gamma0.99_z100_vary_k_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=1,
    title="",
    x_label="Temporal Abstraction (k)",
    y_label="Mean Return",
    figsize=(6, 4),
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize - 0.5,
    y_limit_high=0.4,
)

# ----------------------------------------------------------------------------------------
# --- Bar Plot - SceneEnv seperated by Tasks [1-5] -gamma0.99 - z100 - vary K (k=1,3,5,10)
# ----------------------------------------------------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 5))
experiments_plot_info = {
    "sceneenv__fbrl__act-rep-1__gamma-0.99__zdim-100__task1": {
        "label": "Task 1",
        "color": color_list[0],
        "legend": "k=1",
    },
    "sceneenv__fbrl__act-rep-3__gamma-0.99__zdim-100__task1": {
        "label": "Task 1",
        "color": color_list[1],
        "legend": "k=3",
    },
    "sceneenv__fbrl__act-rep-5__gamma-0.99__zdim-100__task1": {
        "label": "Task 1",
        "color": color_list[2],
        "legend": "k=5",
    },
    "sceneenv__fbrl__act-rep-10__gamma-0.99__zdim-100__task1": {
        "label": "Task 1",
        "color": color_list[3],
        "legend": "k=10",
    },
    "sceneenv__fbrl__act-rep-20__gamma-0.99__zdim-100__task1": {
        "label": "Task 1",
        "color": color_list[4],
        "legend": "k=20",
    },
    "sceneenv__fbrl__act-rep-1__gamma-0.99__zdim-100__task2": {"label": "Task 2", "color": color_list[0]},
    "sceneenv__fbrl__act-rep-3__gamma-0.99__zdim-100__task2": {"label": "Task 2", "color": color_list[1]},
    "sceneenv__fbrl__act-rep-5__gamma-0.99__zdim-100__task2": {"label": "Task 2", "color": color_list[2]},
    "sceneenv__fbrl__act-rep-10__gamma-0.99__zdim-100__task2": {"label": "Task 2", "color": color_list[3]},
    "sceneenv__fbrl__act-rep-20__gamma-0.99__zdim-100__task2": {"label": "Task 2", "color": color_list[4]},
    "sceneenv__fbrl__act-rep-1__gamma-0.99__zdim-100__task3": {"label": "Task 3", "color": color_list[0]},
    "sceneenv__fbrl__act-rep-3__gamma-0.99__zdim-100__task3": {"label": "Task 3", "color": color_list[1]},
    "sceneenv__fbrl__act-rep-5__gamma-0.99__zdim-100__task3": {"label": "Task 3", "color": color_list[2]},
    "sceneenv__fbrl__act-rep-10__gamma-0.99__zdim-100__task3": {"label": "Task 3", "color": color_list[3]},
    "sceneenv__fbrl__act-rep-20__gamma-0.99__zdim-100__task3": {"label": "Task 3", "color": color_list[4]},
    "sceneenv__fbrl__act-rep-1__gamma-0.99__zdim-100__task4": {"label": "Task 4", "color": color_list[0]},
    "sceneenv__fbrl__act-rep-3__gamma-0.99__zdim-100__task4": {"label": "Task 4", "color": color_list[1]},
    "sceneenv__fbrl__act-rep-5__gamma-0.99__zdim-100__task4": {"label": "Task 4", "color": color_list[2]},
    "sceneenv__fbrl__act-rep-10__gamma-0.99__zdim-100__task4": {"label": "Task 4", "color": color_list[3]},
    "sceneenv__fbrl__act-rep-20__gamma-0.99__zdim-100__task4": {"label": "Task 4", "color": color_list[4]},
    "sceneenv__fbrl__act-rep-1__gamma-0.99__zdim-100__task5": {"label": "Task 5", "color": color_list[0]},
    "sceneenv__fbrl__act-rep-3__gamma-0.99__zdim-100__task5": {"label": "Task 5", "color": color_list[1]},
    "sceneenv__fbrl__act-rep-5__gamma-0.99__zdim-100__task5": {"label": "Task 5", "color": color_list[2]},
    "sceneenv__fbrl__act-rep-10__gamma-0.99__zdim-100__task5": {"label": "Task 5", "color": color_list[3]},
    "sceneenv__fbrl__act-rep-20__gamma-0.99__zdim-100__task5": {"label": "Task 5", "color": color_list[4]},
}

plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/SceneEnv_tasks_gamma0.99_z100_vary_k_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=5,
    title="",
    x_label="",
    y_label="Mean Return",
    figsize=(6, 4),
    labelsize=labelsize - 3,
    tick_labelsize=tick_labelsize - 3,
    legend_labelsize=legend_labelsize - 3,
    y_limit_high=0.0,
    y_limit_low=-3000,
    legend_loc="lower right",
)


# ----------------------------------------------------------------------------------------------------------------
# --- Bar Plot - Action Repeat vs Action Chunk - LargeMazeEnv - gamma0.95 - z100 - vary K (k=1, 3, 5, 10, 20, 50)
# ----------------------------------------------------------------------------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 5))
experiments_plot_info = {
    "largemazeenv__fbrl__rbf__act-rep-1__gamma-0.95__zdim-100": {
        "label": "k=1",
        "color": color_list[1],
        "legend": "Action Repeat",
    },
    "largemazeenv__fbrl__rbf__act-chunk-1__gamma-0.95__zdim-100": {
        "label": "k=1",
        "color": color_list[3],
        "legend": "Action Chunk",
    },
    "largemazeenv__fbrl__rbf__act-rep-3__gamma-0.95__zdim-100": {"label": "k=3", "color": color_list[1]},
    "largemazeenv__fbrl__rbf__act-chunk-3__gamma-0.95__zdim-100": {"label": "k=3", "color": color_list[3]},
    "largemazeenv__fbrl__rbf__act-rep-5__gamma-0.95__zdim-100": {"label": "k=5", "color": color_list[1]},
    "largemazeenv__fbrl__rbf__act-chunk-5__gamma-0.95__zdim-100": {"label": "k=5", "color": color_list[3]},
    "largemazeenv__fbrl__rbf__act-rep-10__gamma-0.95__zdim-100": {"label": "k=10", "color": color_list[1]},
    "largemazeenv__fbrl__rbf__act-chunk-10__gamma-0.95__zdim-100": {"label": "k=10", "color": color_list[3]},
    "largemazeenv__fbrl__rbf__act-rep-20__gamma-0.95__zdim-100": {"label": "k=20", "color": color_list[1]},
    "largemazeenv__fbrl__rbf__act-chunk-20__gamma-0.95__zdim-100": {"label": "k=20", "color": color_list[3]},
    "largemazeenv__fbrl__rbf__act-rep-50__gamma-0.95__zdim-100": {"label": "k=50", "color": color_list[1]},
    "largemazeenv__fbrl__rbf__act-chunk-50__gamma-0.95__zdim-100": {"label": "k=50", "color": color_list[3]},
}
plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/LargeMazeEnv_repeat_vs_chunk_gamma0.95_z100_vary_k_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=2,
    title="",
    x_label="",
    y_label="Mean Return",
    figsize=(6, 4),
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
)

# ----------------------------------------------------------------------------------------------------------------
# --- Bar Plot - Action Repeat vs Action Chunk - CubeEnv - gamma0.95 - z100 - vary K (k=1,2,3,4,5)
# ----------------------------------------------------------------------------------------------------------------
color_list = plt.cm.cool(np.linspace(0, 1, 5))
experiments_plot_info = {
    "cubeenv__fbrl__act-rep-1__gamma-0.95__zdim-100": {
        "label": "k=1",
        "color": color_list[1],
        "legend": "Action Repeat",
    },
    "cubeenv__fbrl__act-chunk-1__gamma-0.95__zdim-100": {
        "label": "k=1",
        "color": color_list[3],
        "legend": "Action Chunk",
    },
    "cubeenv__fbrl__act-rep-2__gamma-0.95__zdim-100": {"label": "k=2", "color": color_list[1]},
    "cubeenv__fbrl__act-chunk-2__gamma-0.95__zdim-100": {"label": "k=2", "color": color_list[3]},
    "cubeenv__fbrl__act-rep-3__gamma-0.95__zdim-100": {"label": "k=3", "color": color_list[1]},
    "cubeenv__fbrl__act-chunk-3__gamma-0.95__zdim-100": {"label": "k=3", "color": color_list[3]},
    "cubeenv__fbrl__act-rep-4__gamma-0.95__zdim-100": {"label": "k=4", "color": color_list[1]},
    "cubeenv__fbrl__act-chunk-4__gamma-0.95__zdim-100": {"label": "k=4", "color": color_list[3]},
    "cubeenv__fbrl__act-rep-5__gamma-0.95__zdim-100": {"label": "k=5", "color": color_list[1]},
    "cubeenv__fbrl__act-chunk-5__gamma-0.95__zdim-100": {"label": "k=5", "color": color_list[3]},
}
plot_evaluation_data(
    evaluation_data=evaluation_data,
    save_path="./plots/CubeEnv_repeat_vs_chunk_gamma0.95_z100_vary_k_barplot",
    experiments=experiments_plot_info,
    normalized=False,
    group_size=2,
    title="",
    x_label="",
    y_label="Mean Return",
    figsize=(6, 4),
    labelsize=labelsize,
    tick_labelsize=tick_labelsize,
    legend_labelsize=legend_labelsize,
    legend_loc="lower right",
)
