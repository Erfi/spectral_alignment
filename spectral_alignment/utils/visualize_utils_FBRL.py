"""
This module provides utility functions for visualizing data in the FBRL project.
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from spectral_alignment.utils.pcp_utils import pcp



# ===========================================================
# =========== DATA GATHERING/PROCESSING FUNCTIONS ===========
# ===========================================================


def normalize_returns(returns: np.ndarray, min_score: float, max_score: float) -> np.ndarray:
    return 100 * (returns - min_score) / (max_score - min_score)


def get_validation_stats(validation_filenames: List[str], min_score=None, max_score=None) -> Dict:
    """
    The validation filenames are evaluation.npz files.
    They should contain a "returns" key with shape (N, M) where N is the number of validations steps
    and M is the number of rollouts per validation step.

    At each validation step we compute the mean over the rollouts (M) to get an array of shape (N) per file.
    If there are multiple files (S seeds), we append the results to get an array of (S, N).
    We then compute the mean and std over the seeds (S) to get an array of shape (N).
    """

    all_means = []
    for validation_eval_path in validation_filenames:
        data = np.load(validation_eval_path)
        rewards = data["returns"]  # shape (N, M)
        mean = np.mean(rewards, axis=1)  # shape (N)
        all_means.append(mean)
    try:
        all_means = np.array(all_means).reshape(len(validation_filenames), -1)  # shape (S, N)
    except ValueError:  # handle uneven lengths by filling with mean
        logging.warning(f"Uneven lengths in validation files. Filling with mean. {validation_filenames}")
        max_length = max(len(x) for x in all_means)
        all_means = np.array(
            [np.concatenate([x, np.full(max_length - len(x), np.mean(x))]) for x in all_means]
        )  # shape (S, N)
    # MEDIAN and INTER-QUANTILE RANGE
    median = np.median(all_means, axis=0)  # shape (N)
    if len(all_means) == 0:
        interquartile_range = 0
    else:
        interquartile_range = np.percentile(all_means, 75, axis=0) - np.percentile(all_means, 25, axis=0)
    # MEAN and STD
    mean = np.mean(all_means, axis=0)  # shape (N)
    std = np.std(all_means, axis=0)  # shape (N)
    ci_95 = 1.96 * std / np.sqrt(len(all_means)) if len(all_means) > 0 else 0

    if (min_score is not None) and (max_score is not None):
        normalized_all_means = normalize_returns(all_means, min_score, max_score)
        # NORMALIZED --- MEDIAN and INTER-QUANTILE RANGE
        median_normalized = np.median(normalized_all_means, axis=0)  # shape (N)
        if len(normalized_all_means) == 0:
            interquartile_normalized_range = 0
        else:
            interquartile_normalized_range = np.percentile(normalized_all_means, 75, axis=0) - np.percentile(
                normalized_all_means, 25, axis=0
            )
        # NORMALIZED --- MEAN and STD
        mean_normalized = np.mean(normalized_all_means, axis=0)  # shape (N)
        std_normalized = np.std(normalized_all_means, axis=0)  # shape (N)
        ci_95_normalized = (
            1.96 * std_normalized / np.sqrt(len(normalized_all_means)) if len(normalized_all_means) > 0 else 0
        )
    else:
        normalized_all_means = None
        median_normalized = None
        interquartile_normalized_range = None
        mean_normalized = None
        std_normalized = None
        ci_95_normalized = None
    return {
        "median_normalized": median_normalized,
        "interquartile_normalized_range": interquartile_normalized_range,
        "mean_normalized": mean_normalized,
        "std_normalized": std_normalized,
        "ci_95_normalized": ci_95_normalized,
        "normalized_all_means": normalized_all_means,
        "median": median,
        "interquartile_range": interquartile_range,
        "mean": mean,
        "std": std,
        "ci_95": ci_95,
        "all_means": all_means,
    }


def get_evaluation_stats(evaluation_filenames: List[str], min_score=None, max_score=None) -> Dict:
    """
    The evaluation filenames are test_evaluations__model_top_1.npz files.
    They should contain a "returns" key with shape (M,) where M is the number of rollouts.

    For each file, we compute the mean over the rollouts to get a scalar per file.
    If there are multiple files (S seeds), we collect the results to get an array of shape (S,).
    We then compute the mean and std over the seeds (S) to get scalars.
    """

    all_means = []
    for evaluation_eval_path in evaluation_filenames:
        data = np.load(evaluation_eval_path)
        rewards = data["returns"]  # shape (M,)
        mean = np.mean(rewards)  # scalar
        all_means.append(mean)
    all_means = np.array(all_means).reshape(-1)  # shape (S,)

    mean = np.mean(all_means)  # scalar
    std = np.std(all_means)  # scalar
    ci_95 = 1.96 * std / np.sqrt(len(all_means)) if len(all_means) > 0 else 0

    median = np.median(all_means)  # scalar
    if len(all_means) == 0:
        interquartile_range = 0
    else:
        interquartile_range = np.percentile(all_means, 75) - np.percentile(all_means, 25)

    if (min_score is not None) and (max_score is not None):
        normalized_all_means = normalize_returns(all_means, min_score, max_score)
        mean_normalized = np.mean(normalized_all_means)  # scalar
        std_normalized = np.std(normalized_all_means)  # scalar
        ci_95_normalized = (
            1.96 * std_normalized / np.sqrt(len(normalized_all_means)) if len(normalized_all_means) > 0 else 0
        )
        median_normalized = np.median(normalized_all_means)  # scalar
        if len(normalized_all_means) == 0:
            interquartile_normalized_range = 0
        else:
            interquartile_normalized_range = np.percentile(normalized_all_means, 75) - np.percentile(
                normalized_all_means, 25
            )
    else:
        normalized_all_means = None
        mean_normalized = None
        std_normalized = None
        ci_95_normalized = None
        median_normalized = None
        interquartile_normalized_range = None

    return {
        "median_normalized": median_normalized,
        "interquartile_normalized_range": interquartile_normalized_range,
        "mean_normalized": mean_normalized,
        "std_normalized": std_normalized,
        "ci_95_normalized": ci_95_normalized,
        "normalized_all_means": normalized_all_means,
        "median": median,
        "interquartile_range": interquartile_range,
        "mean": mean,
        "std": std,
        "ci_95": ci_95,
        "all_means": all_means,
    }


def get_validation_data(
    models_folder: str, environments: list, algorithms: list, experiments: list, min_steps: int = 10e6
) -> Dict:
    """
    This function retrieves all validation (during training) data from the specified models folder.
    The data is gathered by looking into the "evaluations.npz" files in the folder.
    """
    all_data = {}
    for env in environments:
        for algo in algorithms:
            for exp in experiments:
                filenames = glob.glob(
                    f"{models_folder}/{algo}/{env}/{exp}/**/evaluations.npz",
                    recursive=True,
                )
                # Filter files based on the number of steps
                valid_filenames = []
                for valid_file in filenames:
                    # open the file and check the number of steps
                    data = np.load(valid_file)
                    timesteps = data["timesteps"]
                    if timesteps[-1] >= min_steps:
                        valid_filenames.append(valid_file)
                key = f"{env.lower()}__{algo.lower()}__{exp.lower()}"
                if len(valid_filenames) > 0:
                    stats = get_validation_stats(valid_filenames, min_score=None, max_score=None)
                    all_data[key] = stats
    return all_data


def get_evaluation_data(models_folder: str, environments: list, algorithms: list, experiments: list) -> Dict:
    """
    This function retrieves all final evaluation (test) data from the specified models folder.
    The data is gathered by looking into the "test_evaluations__model_top_1.npz" files in the folder.
    """
    all_data = {}
    for env in environments:
        for algo in algorithms:
            for exp in experiments:
                filenames = glob.glob(
                    f"{models_folder}/{algo}/{env}/{exp}/**/test_evaluations__model_top_1.npz",
                    recursive=True,
                )
                key = f"{env.lower()}__{algo.lower()}__{exp.lower()}"
                if len(filenames) > 0:
                    stats = get_evaluation_stats(filenames, min_score=None, max_score=None)
                    all_data[key] = stats
                # --- special case for OGBenchEnvs evaluated on individual tasks ---
                for task_id in range(1, 6):
                    task_filenames = glob.glob(
                        f"{models_folder}/{algo}/{env}/{exp}/**/test_evaluations__task{task_id}__model_top_1.npz",
                        recursive=True,
                    )
                    key = f"{env.lower()}__{algo.lower()}__{exp.lower()}__task{task_id}"
                    if len(task_filenames) > 0:
                        stats = get_evaluation_stats(task_filenames, min_score=None, max_score=None)
                        all_data[key] = stats

    return all_data


# ==========================================
# =========== PLOTTING FUNCTIONS ===========
# ==========================================


def beautify_axis_x(ax, x_limits, ax_offset_scale=0.05, xticks_int=False, decimals: int = 1):
    x_range = abs(x_limits[1] - x_limits[0])
    ax.set_xlim(x_limits[0] - ax_offset_scale * x_range, x_limits[1] + ax_offset_scale * x_range)
    # round to 1 decimal place
    x_limits = np.round(x_limits, decimals)
    xticks = np.linspace(*x_limits, 3)
    if xticks_int:
        xticks = xticks.astype(int)
    ax.set_xticks(xticks)
    ax.spines["bottom"].set_bounds(*x_limits)


def beautify_axis_y(ax, y_limits, ax_offset_scale=0.05, decimals: int = 1):
    y_range = abs(y_limits[1] - y_limits[0])
    ax.set_ylim(y_limits[0] - ax_offset_scale * y_range, y_limits[1] + ax_offset_scale * y_range)
    # round to 1 decimal place
    y_limits = np.round(y_limits, decimals)
    yticks = np.linspace(*y_limits, 3)
    ax.set_yticks(yticks)
    ax.spines["left"].set_bounds(*y_limits)


def beautify_axis(ax, x_limits=None, y_limits=None, ax_offset_scale=0.05, xticks_int=False, decimals: int = 1):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if x_limits:
        beautify_axis_x(
            ax, x_limits=x_limits, ax_offset_scale=ax_offset_scale, xticks_int=xticks_int, decimals=decimals
        )
    if y_limits:
        beautify_axis_y(ax, y_limits=y_limits, ax_offset_scale=ax_offset_scale, decimals=decimals)


def plot_validation_data(
    validation_data: Dict,
    save_path: str,
    experiments: Dict[str, str],
    normalized: bool = False,
    title: str = None,
    validation_frequency: int = 10000,
    legend_loc: str = "upper right",
    figsize: tuple = (6, 4),
    labelsize: int = 14,
    tick_labelsize: int = 12,
    legend_labelsize: int = 10,
):
    """

    This function plots (Line plot) the validation data for the specified experiments.
    It saves the plots to the given save path.
    Args:
        validation_data (dict): The validation data to plot.
        save_path (str): The path to save the plots.
        experiments (dict): A dictionary mapping experiment names (keys in validation_data) to labels and colors.
        normalized (bool): Whether to plot normalized returns or not.
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (exp_key, exp_value) in enumerate(experiments.items()):
        if exp_key not in validation_data:
            logging.warning(f"Experiment {exp_key} not found in validation data.")
            continue
        env_name, algo_name, exp_name = exp_key.split("__", maxsplit=2)
        data = validation_data[exp_key]
        mean = data["mean_normalized"] if normalized else data["mean"]
        # std = data["std_normalized"] if normalized else data["std"]
        ci_95 = data["ci_95_normalized"] if normalized else data["ci_95"]
        x = np.arange(len(mean)) * validation_frequency
        ax.plot(x, mean, color=exp_value["color"], label=exp_value["label"], linewidth=2)
        ax.fill_between(
            x,
            mean - ci_95,
            mean + ci_95,
            alpha=0.1,
            color=exp_value["color"],
        )
    y_label = "Mean Normalized Return" if normalized else "Mean Return"
    title = title if title is not None else env_name.title()
    beautify_axis(ax=ax, x_limits=(0, 1e6), y_limits=(0.0, 1.0), ax_offset_scale=0.01, xticks_int=True, decimals=1)
    ax.set_xlabel("Training Steps", fontsize=labelsize)
    ax.set_ylabel(y_label, fontsize=labelsize)
    ax.set_title(title, fontsize=16)
    # set tick label size for both axes
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    ax.legend(fontsize=legend_labelsize, loc=legend_loc)
    fig.tight_layout()
    fig.savefig(f"{save_path}.pdf")
    plt.close(fig)
    logging.info(f"Plot saved to {save_path}.pdf")


def plot_evaluation_data(
    evaluation_data: Dict,
    save_path: str,
    experiments: Dict[str, str],
    normalized: bool = False,
    group_size: int = 1,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    figsize: tuple = (6, 4),
    labelsize: int = 14,
    tick_labelsize: int = 12,
    legend_labelsize: int = 10,
    legend_loc: str = "upper right",
    legend_cols: int = 1,
    y_limit_low: float = 0.0,
    y_limit_high: float = 1.0,
):
    """
    This function plots (Bar plot) the evaluation data for the specified experiments.
    It saves the plots to the given save path.
    Args:
        evaluation_data (dict): The evaluation data to plot.
        save_path (str): The path to save the plots.
        experiments (dict): A dictionary mapping experiment names (keys in evaluation_data) to labels and colors.
        normalized (bool): Whether to plot normalized returns or not.
        title (str): The title of the plot.
        group_size (int): The number of experiments to group together in the bar plot. This is useful to contrast
            different groups of experiments (e.g. different values of k vs different values of z).
            This is based on the order of the experiments in the "experiments" dictionary.
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)
    width = 0.4  # width of each bar
    offset = 0.4  # offset between groups of bars
    padding = 0.05  # padding within each group

    fig, ax = plt.subplots(figsize=figsize)
    x_labels = []
    x_ticks = []
    for i, (exp_key, exp_value) in enumerate(experiments.items()):
        if exp_key not in evaluation_data:
            # keep the bar's slot (empty) so the layout of the remaining bars is unchanged
            logging.warning(f"Experiment {exp_key} not found in evaluation data.")
            mean, ci_95 = 0.0, 0.0
        else:
            data = evaluation_data[exp_key]
            mean = data["mean_normalized"] if normalized else data["mean"]
            # std = data["std_normalized"] if normalized else data["std"]
            ci_95 = data["ci_95_normalized"] if normalized else data["ci_95"]

        x = width * i + offset * (i // group_size) + padding * (i % group_size)  # position of the bar
        if i % group_size == 0:
            x_labels.append(exp_value["label"])
            x_ticks.append(x + (width + padding) * (group_size - 1) / 2)  # center the tick label on the group of bars
        ax.bar(
            x,
            mean,
            yerr=ci_95,
            color=exp_value["color"],
            ecolor="dimgray",
            alpha=0.7,
            capsize=5,
            width=width,
            label=exp_value.get("legend", None),
        )
    beautify_axis(
        ax=ax, x_limits=(-offset, x + width), y_limits=(y_limit_low, y_limit_high), ax_offset_scale=0.01, decimals=1
    )
    ax.set_xticks(ticks=x_ticks, labels=x_labels, fontsize=tick_labelsize)
    ax.tick_params(axis="y", which="major", labelsize=tick_labelsize)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=labelsize)
    ax.set_ylabel(y_label, fontsize=labelsize)
    (
        ax.legend(fontsize=legend_labelsize, loc=legend_loc, ncol=legend_cols)
        if any(exp_value.get("legend", None) for exp_value in experiments.values())
        else None
    )
    fig.tight_layout()
    fig.savefig(f"{save_path}.pdf")
    plt.close(fig)
    logging.info(f"Plot saved to {save_path}.pdf")


def plot_bellman_error(
    data: list,
    labels: list,
    x_label: str,
    y_label: str,
    title: str,
    save_path: str,
    figsize: tuple = (6, 4),
    labelsize: int = 14,
    tick_labelsize: int = 12,
    legend_labelsize: int = 10,
    y_limits: tuple = (0.0, 1.0),
):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        notch=False,
        orientation="vertical",
        boxprops=dict(facecolor="mediumpurple", color="indigo", linewidth=2),
        medianprops=dict(color="darkorange", linewidth=1),
        whiskerprops=dict(color="indigo", linewidth=2),
        capprops=dict(color="indigo", linewidth=2),
    )

    beautify_axis(ax=ax, y_limits=y_limits, ax_offset_scale=0.01, decimals=1)
    ax.set_xlabel(x_label, fontsize=labelsize)
    ax.set_ylabel(y_label, fontsize=labelsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    ax.set_title(title, fontsize=16)

    fig.tight_layout()
    fig.savefig(f"{save_path}.pdf")
    plt.close(fig)
    logging.info(f"Plot saved to {save_path}.pdf")


def plot_ablation_grid(
    grid: np.ndarray,
    x_ticklabels: list,
    y_ticklabels: list,
    x_label: str,
    y_label: str,
    title: str,
    save_path: str,
    figsize: tuple = (6, 4),
    labelsize: int = 14,
    tick_labelsize: int = 12,
    legend_labelsize: int = 10,
):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, cmap="cool", origin="lower", vmin=0, vmax=1)
    beautify_axis(
        ax=ax, x_limits=(-0.5, len(x_ticklabels) - 0.5), y_limits=(-0.5, len(y_ticklabels) - 0.5), ax_offset_scale=0.01
    )
    ax.spines[["right", "left", "top", "bottom"]].set_visible(False)
    # Set ticks and labels
    ax.set_xticks(np.arange(len(x_ticklabels)))
    ax.set_yticks(np.arange(len(y_ticklabels)))
    ax.set_xticklabels(x_ticklabels, fontsize=tick_labelsize)
    ax.set_yticklabels(y_ticklabels, fontsize=tick_labelsize)
    ax.set_xlabel(x_label, fontsize=labelsize)
    ax.set_ylabel(y_label, fontsize=labelsize)
    ax.set_title(title, fontsize=16)
    # colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Mean Return", fontsize=labelsize)
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.tick_params(labelsize=tick_labelsize)
    fig.tight_layout()
    fig.savefig(f"{save_path}.pdf")
    plt.close(fig)
    logging.info(f"Plot saved to {save_path}.pdf")


def plot_parallel_coordinate_plot(
    evaluation_data: dict,
    save_path: str,
    figsize: tuple,
    labelsize: int,
    tick_labelsize: int,
    linewidth: float = 1.5,
):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    labels = ["d", "$\gamma$", "k", "Return"]
    data = []
    for exp_key, exp_info in evaluation_data.items():
        if exp_key.startswith("roomsenv__fbrl__rbf__act-rep"):
            parts = exp_key.split("__")
            # key format: roomsenv__fbrl__rbf__act-rep-<k>__gamma-<gamma>__zdim-<zdim>
            k = int(parts[3].split("-")[-1])
            gamma = float(parts[4].split("-")[-1])
            zdim = int(parts[5].split("-")[-1])
            mean_return = exp_info["mean"]
            data.append([zdim, gamma, k, mean_return])
    if not data:
        logging.warning(f"No RoomsEnv evaluation data found, skipping {save_path}.")
        return
    fig = pcp(
        data=data,
        labels=labels,
        colorbar=True,
        figsize=figsize,
        ylim=[[25, 400], [0.9, 1.0], [1, 50], [0, 1]],
        alpha=0.99,
        xtick_fontsize=labelsize,
        ytick_fontsize=tick_labelsize,
        linewidth=linewidth,
    )
    fig.savefig(f"{save_path}.pdf")
    plt.close(fig)
    logging.info(f"Plot saved to {save_path}.pdf")


def plot_effective_vs_nominal_gamma(
    evaluation_data: dict,
    save_path: str,
    figsize: tuple,
    labelsize: int,
    tick_labelsize: int,
    legend_labelsize: int,
):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    gamma_eff = {}
    horizon = {}
    returns = {}
    colorlist = plt.cm.cool(np.linspace(0, 1, 6))
    ks = [1, 3, 5, 10, 20, 50]
    gammas = [0.9, 0.95, 0.99, 0.999]
    for i, k in enumerate(ks):
        for j, gamma in enumerate(gammas):
            gamma_eff[(k, gamma)] = gamma ** (1 / k)
            horizon[(k, gamma)] = 1 / (1 - gamma_eff[(k, gamma)])
            returns[(k, gamma)] = evaluation_data.get(f"roomsenv__fbrl__rbf__act-rep-{k}__gamma-{gamma}__zdim-100", {}).get(
                "mean", 0
            )
            ax.scatter(
                (j + 1),
                horizon[(k, gamma)],
                label=f"k={k}" if j == 0 else None,
                s=20,
                color=colorlist[i],
                alpha=0.8,
                zorder=1,
            )
            ax.scatter(
                (j + 1),
                horizon[(k, gamma)],
                label="",
                s=returns[(k, gamma)] * 250,
                color=colorlist[i],
                edgecolors="black",
                alpha=1.0,
            )
    ax.legend(fontsize=legend_labelsize, ncols=2, loc="upper left")
    beautify_axis(ax, x_limits=(1, len(gammas)), ax_offset_scale=0.05)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(1, len(gammas) + 1), labels=[str(g) for g in gammas])
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    ax.set_xlabel("Nominal Discount Factor ($\\gamma$)", fontsize=labelsize)
    ax.set_ylabel("Effective Horizon", fontsize=labelsize)
    fig.tight_layout()
    fig.savefig(f"{save_path}.pdf")
    plt.close(fig)
    logging.info(f"Plot saved to {save_path}.pdf")
