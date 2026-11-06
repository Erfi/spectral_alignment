"""Six-goal Boltzmann shared-subspace sweep (temperature 0.3).

Default run:
    uv run --with numpy --with scipy code/soft_policy_tau03_sweep.py --workers 4
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_fb_stagewise_experiment import (
    NUM_ACTIONS,
    action_transitions,
    layout_from_name,
    macro_mdp,
    spatial_goal_regions,
    transition_sa,
)
from soft_policy_subspace_experiment import (
    boltzmann_optimal_q,
    train_rank,
    validate_stochastic_successor,
)


DEFAULT_LAYOUTS = ["Single Room", "Four Rooms"]
DEFAULT_GAMMAS = [0.95, 0.99]
DEFAULT_REPEATS = [1, 2, 3, 5, 9]
DEFAULT_RANKS = [16, 32, 64]


@dataclass(frozen=True)
class SweepConfiguration:
    layout: str
    size: int
    gamma: float
    repeat: int
    ranks: tuple[int, ...]
    goals: int
    goal_radius: float
    temperature: float
    max_iterations: int
    policy_tolerance: float
    policy_step_size: float
    spectral_tolerance: float
    spectral_max_iterations: int
    seed: int


def evaluate_configuration(configuration: SweepConfiguration) -> tuple[list[dict], list[dict]]:
    free = layout_from_name(configuration.layout, configuration.size)
    primitive_transitions, coordinates = action_transitions(free)
    centers, state_rewards = spatial_goal_regions(
        coordinates, configuration.goals, configuration.goal_radius
    )
    primitive_gamma = configuration.gamma ** (1.0 / configuration.repeat)
    primitive_rewards = np.repeat(state_rewards[:, :, None], NUM_ACTIONS, axis=2)
    repeated_transitions, repeated_rewards = macro_mdp(
        primitive_transitions, state_rewards, primitive_gamma, configuration.repeat
    )
    original_q, _, original_iterations, original_residual = boltzmann_optimal_q(
        primitive_transitions, primitive_rewards, primitive_gamma,
        configuration.temperature, 1e-10, 100_000,
    )
    repeated_q, repeated_policies, repeated_iterations, repeated_residual = boltzmann_optimal_q(
        repeated_transitions, repeated_rewards, configuration.gamma,
        configuration.temperature, 1e-10, 100_000,
    )
    transition = transition_sa(repeated_transitions)

    records, histories = [], []
    for rank in configuration.ranks:
        rank_records, rank_history = train_rank(
            transition, repeated_transitions, repeated_rewards, repeated_policies,
            repeated_q, configuration.gamma, configuration.temperature, rank,
            configuration.max_iterations, configuration.policy_tolerance,
            configuration.spectral_tolerance, configuration.spectral_max_iterations,
            configuration.seed + rank, comparison_q=original_q,
            policy_step_size=configuration.policy_step_size,
        )
        for record in rank_records:
            goal = record["goal"]
            original_scale = max(
                float(np.max(np.abs(original_q[goal]))), np.finfo(float).tiny
            )
            repeated_scale = max(
                float(np.max(np.abs(repeated_q[goal]))), np.finfo(float).tiny
            )
            repeat_gap = float(np.max(np.abs(original_q[goal] - repeated_q[goal])))
            record.update({
                "layout": configuration.layout,
                "states": len(coordinates),
                "state_actions": len(coordinates) * NUM_ACTIONS,
                "nominal_gamma": configuration.gamma,
                "primitive_gamma": primitive_gamma,
                "repeat": configuration.repeat,
                "goals_jointly_fitted": configuration.goals,
                "policy_step_size": configuration.policy_step_size,
                "goal_state": int(centers[goal]),
                "q_original_scale": original_scale,
                "q_repeated_scale": repeated_scale,
                "reward_l2": float(np.linalg.norm(repeated_rewards[goal].reshape(-1))),
                "repeat_q_inf": repeat_gap,
                "repeat_q_normalized": repeat_gap / original_scale,
                "q_model_original_normalized": record["q_model_comparison_inf"] / original_scale,
                "q_policy_original_normalized": record["q_policy_comparison_inf"] / original_scale,
                "q_model_repeated_normalized": record["q_model_reference_inf"] / repeated_scale,
                "q_policy_repeated_normalized": record["q_policy_reference_inf"] / repeated_scale,
                "original_dp_iterations": original_iterations,
                "repeated_dp_iterations": repeated_iterations,
                "original_dp_residual": original_residual,
                "repeated_dp_residual": repeated_residual,
            })
            records.append(record)
        for row in rank_history:
            row.update({
                "layout": configuration.layout,
                "nominal_gamma": configuration.gamma,
                "repeat": configuration.repeat,
            })
            histories.append(row)
    return records, histories


def add_soft_lemma41_bounds(records: list[dict]) -> None:
    """Add a soft-policy diagnostic using Lemma 4.1's direct spectral step only."""
    groups: dict[tuple[str, float, int], list[dict]] = {}
    for record in records:
        groups.setdefault(
            (record["layout"], record["nominal_gamma"], record["rank"]), []
        ).append(record)
    for group in groups.values():
        c_sf = max(
            record["sigma_d_plus_1"]
            * (1.0 - record["nominal_gamma"] * record["joint_lambda_d_plus_1"])
            for record in group
        )
        for record in group:
            spectral_bound = c_sf / (
                1.0 - record["nominal_gamma"] * record["joint_lambda_d_plus_1"]
            )
            exact_sigma_bound = (
                record["repeat_q_inf"]
                + record["q_policy_reference_inf"]
                + record["reward_l2"] * record["model_error_2"]
            )
            lemma41_bound = (
                record["repeat_q_inf"]
                + record["q_policy_reference_inf"]
                + record["reward_l2"]
                * (record["realization_error"] + spectral_bound)
            )
            tolerance = 1e-6 * max(1.0, lemma41_bound)
            if exact_sigma_bound + tolerance < record["q_model_comparison_inf"]:
                raise AssertionError("Exact-sigma soft bound failed")
            if lemma41_bound + tolerance < exact_sigma_bound:
                raise AssertionError("Lemma 4.1 soft bound failed")
            record.update({
                "empirical_c_sf": c_sf,
                "lemma41_sigma_bound": spectral_bound,
                "soft_exact_sigma_bound_original_inf": exact_sigma_bound,
                "soft_lemma41_bound_original_inf": lemma41_bound,
                "soft_exact_sigma_bound_original_normalized": (
                    exact_sigma_bound / record["q_original_scale"]
                ),
                "soft_lemma41_bound_original_normalized": (
                    lemma41_bound / record["q_original_scale"]
                ),
            })


def write_csv(rows: list[dict], output: Path) -> None:
    with output.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", nargs="+", choices=DEFAULT_LAYOUTS, default=DEFAULT_LAYOUTS)
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    parser.add_argument("--repeats", type=int, nargs="+", default=DEFAULT_REPEATS)
    parser.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    parser.add_argument("--goals", type=int, default=6)
    parser.add_argument("--goal-radius", type=float, default=1.75)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--policy-tolerance", type=float, default=1e-5)
    parser.add_argument("--policy-step-size", type=float, default=0.5)
    parser.add_argument("--spectral-tolerance", type=float, default=2e-5)
    parser.add_argument("--spectral-max-iterations", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("assets/bound_tightness/soft_policy_tau0p3_alpha0p5_bounds"),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.layouts, args.gammas, args.repeats = ["Four Rooms"], [0.99], [3]
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_stochastic_successor(args.seed)
    print(f"Stochastic successor validation max error={validation:.3g}")
    configurations = [
        SweepConfiguration(
            layout, args.size, gamma, repeat, tuple(args.ranks), args.goals,
            args.goal_radius, args.temperature, args.max_iterations,
            args.policy_tolerance, args.policy_step_size, args.spectral_tolerance,
            args.spectral_max_iterations,
            args.seed + 100_000 * li + 10_000 * gi + 100 * repeat,
        )
        for li, layout in enumerate(args.layouts)
        for gi, gamma in enumerate(args.gammas)
        for repeat in args.repeats
    ]
    start = time.perf_counter()
    records, histories = [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(configurations))) as executor:
        futures = {executor.submit(evaluate_configuration, config): config for config in configurations}
        for future in as_completed(futures):
            config = futures[future]
            config_records, config_histories = future.result()
            records.extend(config_records)
            histories.extend(config_histories)
            print(
                f"Completed {config.layout}, gamma={config.gamma:g}, k={config.repeat}",
                flush=True,
            )
    records.sort(key=lambda record: (
        args.layouts.index(record["layout"]),
        args.gammas.index(record["nominal_gamma"]),
        args.repeats.index(record["repeat"]),
        args.ranks.index(record["rank"]), record["goal"],
    ))
    histories.sort(key=lambda record: (
        args.layouts.index(record["layout"]),
        args.gammas.index(record["nominal_gamma"]),
        args.repeats.index(record["repeat"]),
        args.ranks.index(record["rank"]), record["iteration"],
    ))
    add_soft_lemma41_bounds(records)

    max_residual = max(
        max(record["policy_bellman_residual"], record["original_dp_residual"], record["repeated_dp_residual"])
        for record in records
    )
    decomposition_error = max(
        abs(record["model_error_2"] - record["sigma_d_plus_1"] - record["realization_error"])
        for record in records
    )
    if max_residual > 1e-7 or decomposition_error > 1e-6:
        raise AssertionError(
            f"Validation failed: Bellman={max_residual:g}, decomposition={decomposition_error:g}"
        )
    write_csv(records, args.output_dir / "soft_policy_tau0p3_results.csv")
    write_csv(histories, args.output_dir / "soft_policy_tau0p3_convergence.csv")
    configurations_converged = {
        (record["layout"], record["nominal_gamma"], record["repeat"], record["rank"])
        for record in records if record["converged"]
    }
    print(
        f"Wrote {len(records)} rows in {time.perf_counter() - start:.1f}s; "
        f"fixed points={len(configurations_converged)}/{len(configurations) * len(args.ranks)}"
    )
    for layout in args.layouts:
        for rank in args.ranks:
            selected = [r for r in records if r["layout"] == layout and r["rank"] == rank]
            print(
                f"{layout}, d={rank}: median realization={np.median([r['realization_error'] for r in selected]):.3g}, "
                f"spectral floor={np.median([r['sigma_d_plus_1'] for r in selected]):.3g}"
            )


if __name__ == "__main__":
    main()
