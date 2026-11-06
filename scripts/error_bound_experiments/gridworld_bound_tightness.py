"""Compare the SR spectral bound with the exact truncation error on grids.

Run with:
    uv run --with numpy code/gridworld_bound_tightness.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def gridworld_transition_matrix_with_walls(
    rows: int, cols: int, free: np.ndarray, stay_prob: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(free)
    state_of = {tuple(cell): i for i, cell in enumerate(coordinates)}
    transition = np.zeros((len(coordinates), len(coordinates)))
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i, (row, col) in enumerate(coordinates):
        neighbors = []
        for d_row, d_col in actions:
            candidate = (int(row + d_row), int(col + d_col))
            if candidate in state_of:
                neighbors.append(state_of[candidate])

        if neighbors:
            transition[i, neighbors] = (1.0 - stay_prob) / len(neighbors)
            transition[i, i] += stay_prob
        else:
            transition[i, i] = 1.0

    if not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("Transition matrix is not row-stochastic")
    return transition, coordinates


def bordered_four_rooms_layout(
    rows: int,
    cols: int,
    wall_thickness: int = 1,
    door_size: int = 1,
    border: int = 1,
) -> np.ndarray:
    free = np.ones((rows, cols), dtype=bool)
    if border > 0:
        free[:border, :] = False
        free[-border:, :] = False
        free[:, :border] = False
        free[:, -border:] = False

    row_start, row_end = border, rows - border
    col_start, col_end = border, cols - border
    row_middle = (row_start + row_end - 1) // 2
    col_middle = (col_start + col_end - 1) // 2
    free[row_middle : row_middle + wall_thickness, col_start:col_end] = False
    free[row_start:row_end, col_middle : col_middle + wall_thickness] = False

    row_span = row_end - row_start
    col_span = col_end - col_start
    door_columns = [col_start + col_span // 3, col_start + 2 * col_span // 3]
    door_rows = [row_start + row_span // 3, row_start + 2 * row_span // 3]
    for column in door_columns:
        column = int(np.clip(column, col_start, col_end - door_size))
        free[
            row_middle : row_middle + wall_thickness,
            column : column + door_size,
        ] = True
    for row in door_rows:
        row = int(np.clip(row, row_start, row_end - door_size))
        free[
            row : row + door_size,
            col_middle : col_middle + wall_thickness,
        ] = True
    return free


def bordered_single_room_layout(rows: int, cols: int, border: int = 1) -> np.ndarray:
    free = np.ones((rows, cols), dtype=bool)
    if border > 0:
        free[:border, :] = False
        free[-border:, :] = False
        free[:, :border] = False
        free[:, -border:] = False
    return free


def reversible_eigendecomposition(
    transition: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return eigenvalues and a valid eigenbasis condition bound for a grid walk."""
    off_diagonal = transition.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    degree = np.count_nonzero(off_diagonal, axis=1).astype(float)
    degree[degree == 0] = 1.0

    sqrt_degree = np.sqrt(degree)
    symmetric = (
        sqrt_degree[:, None]
        * transition
        / sqrt_degree[None, :]
    )
    if not np.allclose(symmetric, symmetric.T, atol=1e-12):
        raise ValueError("Expected reversible random-walk dynamics")

    eigenvalues = np.linalg.eigvalsh(symmetric)
    eigenvalues = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
    kappa = np.sqrt(degree.max() / degree.min())
    return eigenvalues, kappa


def evaluate_layout(
    name: str,
    free: np.ndarray,
    gamma: float,
    stay_prob: float,
    ranks: list[int],
    repeats: list[int],
) -> list[dict[str, float | int | str]]:
    transition, _ = gridworld_transition_matrix_with_walls(
        *free.shape, free, stay_prob=stay_prob
    )
    eigenvalues, kappa = reversible_eigendecomposition(transition)
    records = []

    for repeat in repeats:
        repeated_transition = np.linalg.matrix_power(transition, repeat)
        effective_discount = gamma**repeat
        successor = np.linalg.inv(
            np.eye(len(transition)) - effective_discount * repeated_transition
        )
        singular_values = np.linalg.svd(successor, compute_uv=False)

        for rank in ranks:
            if rank >= len(transition):
                continue
            discarded_eigenvalue = abs(eigenvalues[rank]) ** repeat
            theoretical_bound = kappa / (
                1.0 - effective_discount * discarded_eigenvalue
            )
            actual_error = singular_values[rank]
            records.append(
                {
                    "layout": name,
                    "gamma": gamma,
                    "states": len(transition),
                    "rank": rank,
                    "repeat": repeat,
                    "exact_truncation_error": actual_error,
                    "theoretical_bound": theoretical_bound,
                    "bound_over_exact": theoretical_bound / actual_error,
                    "kappa": kappa,
                    "discarded_eigenvalue": discarded_eigenvalue,
                }
            )
    return records


def write_csv(records: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=[0.9, 0.95, 0.99, 0.999]
    )
    parser.add_argument("--stay-prob", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, nargs="+", default=[1, 5, 9, 11])
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("assets/bound_tightness")
    )
    args = parser.parse_args()

    layouts = {
        "Single Room": bordered_single_room_layout(20, 20, border=1),
        "Four Rooms": bordered_four_rooms_layout(20, 20, border=1),
    }
    records = []
    for gamma in args.gammas:
        for name, free in layouts.items():
            records.extend(
                evaluate_layout(
                    name,
                    free,
                    gamma,
                    args.stay_prob,
                    args.ranks,
                    args.repeats,
                )
            )
    csv_output = args.output_dir / "gridworld_bound_tightness.csv"
    write_csv(records, csv_output)
    print(f"Wrote {csv_output}")
    for gamma in args.gammas:
        for layout in layouts:
            selected = [
                record
                for record in records
                if record["layout"] == layout and record["gamma"] == gamma
            ]
            ratios = np.array(
                [record["bound_over_exact"] for record in selected], dtype=float
            )
            print(
                f"gamma={gamma}, {layout}: bound/exact min={ratios.min():.3f}, "
                f"median={np.median(ratios):.3f}, max={ratios.max():.3f}"
            )


if __name__ == "__main__":
    main()
