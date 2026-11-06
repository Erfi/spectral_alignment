"""Plot the Single-Room and Four-Rooms layouts with six goal regions.

Run:
    uv run --with numpy --with scipy --with matplotlib \
        code/plot_layouts_and_matrices.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from matrix_fb_stagewise_experiment import (
    DEFAULT_LAYOUT_SIZE,
    GOAL_RADIUS,
    action_transitions,
    layout_from_name,
    spatial_goal_regions,
)


def plot_layouts(output: Path, goals: int = 6) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })
    wall_color = np.array([0.16, 0.16, 0.18])
    free_color = np.array([1.0, 1.0, 1.0])
    goal_colors = np.array([
        [0.00, 0.86, 0.89],
        [0.00, 0.72, 0.97],
        [0.30, 0.52, 0.98],
        [0.55, 0.37, 0.98],
        [0.92, 0.02, 0.86],
        [0.76, 0.24, 0.96],
    ])[:goals]

    figure, axes = plt.subplots(
        1, 2, figsize=(4.0, 2.15), gridspec_kw={"wspace": 0.22},
    )
    for ax, layout in zip(axes, ["Single Room", "Four Rooms"]):
        free = layout_from_name(layout, DEFAULT_LAYOUT_SIZE)
        _, coordinates = action_transitions(free)
        _, rewards = spatial_goal_regions(coordinates, goals, GOAL_RADIUS)

        image = np.where(free[..., None], free_color, wall_color)
        for g in range(goals):
            for j, (r, c) in enumerate(coordinates):
                if rewards[g, j] > 0:
                    image[r, c] = 0.72 * goal_colors[g] + 0.28 * free_color

        ax.imshow(image, origin="upper", interpolation="nearest")
        n = DEFAULT_LAYOUT_SIZE
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="#c9c9c9", linewidth=0.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(which="both", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
        ax.set_title(layout, fontsize=10)
        ax.set_aspect("equal")

    figure.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    out = Path("assets/bound_tightness")
    plot_layouts(out / "layout_matrices.pdf", goals=6)
