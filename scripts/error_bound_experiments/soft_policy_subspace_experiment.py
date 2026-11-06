"""Boltzmann shared-subspace fit; V(s)=sum_a pi(a|s)Q(s,a), without entropy rewards.

Default run:
    uv run --with numpy --with scipy \
        code/soft_policy_subspace_experiment.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.sparse.linalg import LinearOperator

from matrix_fb_stagewise_experiment import (
    NUM_ACTIONS,
    action_transitions,
    difference,
    fit_backward,
    layout_from_name,
    macro_mdp,
    model_operator,
    singular_values,
    spatial_goal_regions,
    transition_sa,
)


def boltzmann(q_values: np.ndarray, temperature: float) -> np.ndarray:
    logits = (q_values - q_values.max(axis=-1, keepdims=True)) / temperature
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities


def boltzmann_optimal_q(
    transitions: list,
    rewards: np.ndarray,
    discount: float,
    temperature: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Solve Q=r+gamma P E_{softmax(Q/tau)}Q by fixed-point iteration."""
    q_values = np.zeros_like(rewards)
    for iteration in range(1, max_iterations + 1):
        policies = boltzmann(q_values, temperature)
        values = np.sum(policies * q_values, axis=2)
        continuation = np.stack(
            [(transition @ values.T).T for transition in transitions], axis=2
        )
        updated = rewards + discount * continuation
        residual = float(np.max(np.abs(updated - q_values)))
        q_values = updated
        if residual <= tolerance:
            return q_values, boltzmann(q_values, temperature), iteration, residual
    raise RuntimeError(f"Boltzmann DP failed to converge: residual={residual:g}")


@dataclass
class StochasticSuccessor:
    """Apply (I - discount * P_sa^pi)^-1 via a smaller state-space LU solve."""

    transition: object
    policy: np.ndarray
    discount: float

    def __post_init__(self) -> None:
        self.n, self.s = self.transition.shape
        if self.policy.shape != (self.s, NUM_ACTIONS):
            raise ValueError(f"Invalid policy shape: {self.policy.shape}")
        if np.min(self.policy) < 0 or not np.allclose(self.policy.sum(axis=1), 1.0):
            raise ValueError("Policy must contain row-stochastic action probabilities")
        rows = np.arange(self.s)[:, None] * NUM_ACTIONS + np.arange(NUM_ACTIONS)
        self.chain = sum(
            self.policy[:, action, None] * self.transition[rows[:, action]].toarray()
            for action in range(NUM_ACTIONS)
        )
        self.lu, self.pivots = scipy.linalg.lu_factor(
            np.eye(self.s) - self.discount * self.chain, check_finite=False
        )
        self.operator = LinearOperator(
            (self.n, self.n), matvec=self.matvec, rmatvec=self.rmatvec,
            matmat=self.matmat, rmatmat=self.rmatmat, dtype=float,
        )

    def project_policy(self, values: np.ndarray) -> np.ndarray:
        trailing = values.shape[1:]
        reshaped = values.reshape(self.s, NUM_ACTIONS, *trailing)
        weights = self.policy.reshape(self.s, NUM_ACTIONS, *((1,) * len(trailing)))
        return np.sum(weights * reshaped, axis=1)

    def lift_policy(self, values: np.ndarray) -> np.ndarray:
        trailing = values.shape[1:]
        weights = self.policy.reshape(self.s, NUM_ACTIONS, *((1,) * len(trailing)))
        return (weights * values.reshape(self.s, 1, *trailing)).reshape(self.n, *trailing)

    def matmat(self, values: np.ndarray) -> np.ndarray:
        correction = scipy.linalg.lu_solve(
            (self.lu, self.pivots), self.project_policy(values), check_finite=False
        )
        return values + self.discount * (self.transition @ correction)

    def rmatmat(self, values: np.ndarray) -> np.ndarray:
        correction = scipy.linalg.lu_solve(
            (self.lu, self.pivots), self.transition.T @ values,
            trans=1, check_finite=False,
        )
        return values + self.discount * self.lift_policy(correction)

    def matvec(self, values: np.ndarray) -> np.ndarray:
        return self.matmat(values[:, None]).ravel()

    def rmatvec(self, values: np.ndarray) -> np.ndarray:
        return self.rmatmat(values[:, None]).ravel()


def validate_stochastic_successor(seed: int) -> float:
    free = layout_from_name("Single Room", 5)
    transitions, coordinates = action_transitions(free)
    transition = transition_sa(transitions)
    rng = np.random.default_rng(seed)
    policy = rng.dirichlet(np.ones(NUM_ACTIONS), size=len(coordinates))
    discount = 0.9
    successor = StochasticSuccessor(transition, policy, discount)
    policy_lift = np.zeros((len(coordinates), len(coordinates) * NUM_ACTIONS))
    for state in range(len(coordinates)):
        policy_lift[state, state * NUM_ACTIONS:(state + 1) * NUM_ACTIONS] = policy[state]
    explicit = np.linalg.inv(
        np.eye(transition.shape[0]) - discount * transition.toarray() @ policy_lift
    )
    operator_matrix = successor.matmat(np.eye(transition.shape[0]))
    error = float(np.max(np.abs(operator_matrix - explicit)))
    if error > 1e-10:
        raise AssertionError(f"Stochastic successor validation failed: {error:g}")
    return error


def train_rank(
    transition: object,
    action_transitions: list,
    rewards: np.ndarray,
    initial_policies: np.ndarray,
    reference_q: np.ndarray,
    discount: float,
    temperature: float,
    rank: int,
    max_iterations: int,
    policy_tolerance: float,
    spectral_tolerance: float,
    spectral_max_iterations: int,
    seed: int,
    comparison_q: np.ndarray | None = None,
    policy_step_size: float = 1.0,
) -> tuple[list[dict], list[dict]]:
    if not 0.0 < policy_step_size <= 1.0:
        raise ValueError("Policy step size must be in (0, 1]")
    policies = initial_policies.copy()
    snapshots = []
    history = []
    for iteration in range(1, max_iterations + 1):
        targets = [StochasticSuccessor(transition, policy, discount) for policy in policies]
        backward, fit_residual = fit_backward(
            targets, rank, spectral_tolerance, spectral_max_iterations, seed + iteration
        )
        models = [model_operator(target, backward) for target in targets]
        q_hat = np.stack(
            [model @ rewards[goal].reshape(-1) for goal, model in enumerate(models)]
        ).reshape(rewards.shape)
        updated_policies = boltzmann(q_hat, temperature)
        probability_change = float(np.max(np.abs(updated_policies - policies)))
        action_change = float(
            np.mean(updated_policies.argmax(axis=2) != policies.argmax(axis=2))
        )
        history.append({
            "rank": rank, "iteration": iteration,
            "policy_probability_change": probability_change,
            "policy_action_change": action_change,
        })
        snapshots.append((
            probability_change, models, q_hat, updated_policies, fit_residual, iteration,
        ))
        if probability_change <= policy_tolerance:
            break
        policies = (1.0 - policy_step_size) * policies + policy_step_size * updated_policies

    final = min(snapshots, key=lambda snapshot: snapshot[0])
    change, models, q_hat, induced_policies, fit_residual, iteration = final
    induced_again = boltzmann(q_hat, temperature)
    if not np.allclose(induced_again, induced_policies, atol=1e-12, rtol=1e-12):
        raise AssertionError("Stored model does not induce the stored Boltzmann policy")

    records = []
    for goal, policy in enumerate(induced_policies):
        induced = StochasticSuccessor(transition, policy, discount)
        reward = rewards[goal].reshape(-1)
        policy_q = induced.matvec(reward)
        model_diff = difference(models[goal], induced.operator)
        model_sv, model_residual = singular_values(
            model_diff, 1, spectral_tolerance, spectral_max_iterations,
            seed + 100_000 + goal,
        )
        successor_sv, sigma_residual = singular_values(
            induced.operator, rank + 1, spectral_tolerance,
            spectral_max_iterations, seed + 200_000 + goal,
        )
        model_error = float(model_sv[0])
        sigma = float(successor_sv[rank])
        realization = max(model_error - sigma, 0.0)
        eigenvalues = np.sort(np.abs(np.linalg.eigvals(induced.chain)))[::-1]
        joint_lambda = float(eigenvalues[rank]) if rank < len(eigenvalues) else 0.0
        q_shape = reference_q[goal].shape
        policy_values = np.sum(policy * policy_q.reshape(q_shape), axis=1)
        residual = float(np.max(np.abs(
            policy_q.reshape(q_shape)
            - rewards[goal]
            - discount * np.stack(
                [kernel @ policy_values for kernel in action_transitions], axis=1
            )
        )))
        if residual > 1e-7 or model_error + 1e-6 < sigma:
            raise AssertionError("Soft-policy successor validation failed")
        record = {
            "goal": goal,
            "rank": rank,
            "temperature": temperature,
            "converged": change <= policy_tolerance,
            "iterations": iteration,
            "policy_probability_change": change,
            "policy_action_change": history[iteration - 1]["policy_action_change"],
            "policy_step_size": policy_step_size,
            "q_model_reference_inf": float(np.max(np.abs(q_hat[goal].reshape(-1) - reference_q[goal].reshape(-1)))),
            "q_policy_reference_inf": float(np.max(np.abs(policy_q - reference_q[goal].reshape(-1)))),
            "q_model_policy_inf": float(np.max(np.abs(q_hat[goal].reshape(-1) - policy_q))),
            "model_error_2": model_error,
            "sigma_d_plus_1": sigma,
            "realization_error": realization,
            "joint_lambda_d_plus_1": joint_lambda,
            "fit_relative_eigen_residual": fit_residual,
            "model_normal_relative_eigen_residual": model_residual,
            "sigma_normal_relative_eigen_residual": sigma_residual,
            "policy_bellman_residual": residual,
        }
        if comparison_q is not None:
            comparison = comparison_q[goal].reshape(-1)
            record.update({
                "q_model_comparison_inf": float(np.max(np.abs(
                    q_hat[goal].reshape(-1) - comparison
                ))),
                "q_policy_comparison_inf": float(np.max(np.abs(policy_q - comparison))),
            })
        records.append(record)
    return records, history


def write_csv(rows: list[dict], output: Path) -> None:
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict], ranks: list[int]) -> list[dict]:
    summary = []
    for rank in ranks:
        rank_records = [record for record in records if record["rank"] == rank]
        summary.append({
            "rank": rank,
            "converged": all(record["converged"] for record in rank_records),
            "iterations": max(record["iterations"] for record in rank_records),
            "policy_probability_change": max(
                record["policy_probability_change"] for record in rank_records
            ),
            "model_error_2_median": float(np.median([
                record["model_error_2"] for record in rank_records
            ])),
            "sigma_d_plus_1_median": float(np.median([
                record["sigma_d_plus_1"] for record in rank_records
            ])),
            "realization_error_median": float(np.median([
                record["realization_error"] for record in rank_records
            ])),
            "q_model_reference_inf_median": float(np.median([
                record["q_model_reference_inf"] for record in rank_records
            ])),
            "q_policy_reference_inf_median": float(np.median([
                record["q_policy_reference_inf"] for record in rank_records
            ])),
        })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=["Single Room", "Four Rooms"], default="Four Rooms")
    parser.add_argument("--size", type=int, default=11)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--goals", type=int, default=6)
    parser.add_argument("--goal-radius", type=float, default=1.75)
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--policy-tolerance", type=float, default=1e-5)
    parser.add_argument("--policy-step-size", type=float, default=1.0)
    parser.add_argument("--spectral-tolerance", type=float, default=2e-5)
    parser.add_argument("--spectral-max-iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("assets/bound_tightness/soft_policy_representative_tau0p3"),
    )
    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("Temperature must be positive")
    validation_error = validate_stochastic_successor(args.seed)
    print(f"Stochastic successor validation max error={validation_error:.3g}")

    free = layout_from_name(args.layout, args.size)
    primitive_transitions, coordinates = action_transitions(free)
    _, state_rewards = spatial_goal_regions(coordinates, args.goals, args.goal_radius)
    primitive_gamma = args.gamma ** (1.0 / args.repeat)
    repeated_transitions, repeated_rewards = macro_mdp(
        primitive_transitions, state_rewards, primitive_gamma, args.repeat
    )
    reference_q, initial_policies, dp_iterations, dp_residual = boltzmann_optimal_q(
        repeated_transitions, repeated_rewards, args.gamma, args.temperature,
        1e-10, 100_000,
    )
    transition = transition_sa(repeated_transitions)
    all_records, all_history = [], []
    for rank in args.ranks:
        records, history = train_rank(
            transition, repeated_transitions, repeated_rewards, initial_policies, reference_q,
            args.gamma, args.temperature, rank, args.max_iterations,
            args.policy_tolerance, args.spectral_tolerance,
            args.spectral_max_iterations, args.seed + rank,
            policy_step_size=args.policy_step_size,
        )
        all_records.extend(records)
        all_history.extend(history)
        print(
            f"d={rank}: converged={records[0]['converged']}, "
            f"iterations={records[0]['iterations']}, "
            f"policy_change={records[0]['policy_probability_change']:.3g}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(all_records, args.output_dir / "soft_policy_results.csv")
    write_csv(all_history, args.output_dir / "soft_policy_convergence.csv")
    write_csv(summarize(all_records, args.ranks), args.output_dir / "soft_policy_summary.csv")
    print(
        f"Boltzmann DP: iterations={dp_iterations}, residual={dp_residual:.3g}; "
        f"wrote outputs to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
