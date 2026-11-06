"""Hard-greedy shared-subspace diagnostic and grid utilities.

The replay measure is absorbed into reward coordinates: B is Euclidean-orthonormal,
z=B.T r, and F=M B. The fit uses exact successor operators, not learned FB training.

Default run (separate shared-B fit for every rank):
    uv run --with numpy --with scipy code/matrix_fb_stagewise_experiment.py --workers 4

Fast checks:
    uv run --with numpy --with scipy code/matrix_fb_stagewise_experiment.py --validate-only
    uv run --with numpy --with scipy code/matrix_fb_stagewise_experiment.py --smoke
"""

from __future__ import annotations

import os

# Set these before importing NumPy/SciPy: independent processes provide parallelism.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import argparse
import csv
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh

from gridworld_bound_tightness import (
    bordered_four_rooms_layout,
    bordered_single_room_layout,
)


ACTIONS = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (0, 0),
)
NUM_ACTIONS = len(ACTIONS)
DEFAULT_GAMMAS = [0.95, 0.99]
DEFAULT_REPEATS = [1, 2, 3, 5, 9]
DEFAULT_RANKS = [16, 32, 64]
DEFAULT_LAYOUT_SIZE = 11
DEFAULT_GOALS = 12
INTENDED_PROBABILITY = 0.8
GOAL_RADIUS = 1.75
TINY = np.finfo(float).tiny


def layout_from_name(name: str, size: int) -> np.ndarray:
    if name == "Single Room":
        return bordered_single_room_layout(size, size, border=1)
    if name == "Four Rooms":
        return bordered_four_rooms_layout(size, size, border=1)
    raise ValueError(f"Unknown layout: {name}")


def action_transitions(
    free: np.ndarray, intended_probability: float = INTENDED_PROBABILITY
) -> tuple[list[sp.csr_matrix], np.ndarray]:
    coordinates = np.argwhere(free)
    state_of = {tuple(cell): state for state, cell in enumerate(coordinates)}
    transitions: list[sp.csr_matrix] = []
    for intended, intended_delta in enumerate(ACTIONS):
        probabilities = np.full(NUM_ACTIONS, (1.0 - intended_probability) / 8.0)
        probabilities[intended] = intended_probability
        rows, columns, data = [], [], []
        for state, (row, column) in enumerate(coordinates):
            destinations: dict[int, float] = {}
            for delta, probability in zip(ACTIONS, probabilities):
                destination = state_of.get(
                    (int(row + delta[0]), int(column + delta[1])), state
                )
                destinations[destination] = destinations.get(destination, 0.0) + probability
            for destination, probability in destinations.items():
                rows.append(state)
                columns.append(destination)
                data.append(float(probability))
        transition = sp.csr_matrix(
            (data, (rows, columns)), shape=(len(coordinates), len(coordinates))
        )
        if not np.allclose(np.asarray(transition.sum(axis=1)).ravel(), 1.0):
            raise AssertionError("Action kernel is not row stochastic")
        transitions.append(transition)
    return transitions, coordinates


def spatial_goal_regions(
    coordinates: np.ndarray, count: int, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    if count > len(coordinates):
        raise ValueError("More goals than free states")
    center = coordinates.mean(axis=0)
    selected = [int(np.argmax(np.sum((coordinates - center) ** 2, axis=1)))]
    distances = np.sum((coordinates - coordinates[selected[0]]) ** 2, axis=1)
    while len(selected) < count:
        distances[selected] = -1.0
        candidate = int(np.argmax(distances))
        selected.append(candidate)
        distances = np.minimum(
            distances,
            np.sum((coordinates - coordinates[candidate]) ** 2, axis=1),
        )
    centers = np.asarray(selected)
    rewards = np.stack(
        [
            (np.linalg.norm(coordinates - coordinates[index], axis=1) <= radius).astype(float)
            for index in centers
        ]
    )
    return centers, rewards


def macro_mdp(
    transitions: list[sp.csr_matrix],
    state_rewards: np.ndarray,
    primitive_gamma: float,
    repeat: int,
) -> tuple[list[sp.csr_matrix], np.ndarray]:
    """Return P_a^k and sum_{t<k} primitive_gamma^t P_a^t r."""
    num_goals, num_states = state_rewards.shape
    repeated: list[sp.csr_matrix] = []
    rewards = np.empty((num_goals, num_states, NUM_ACTIONS))
    identity = sp.eye(num_states, format="csr")
    for action, transition in enumerate(transitions):
        power = identity
        accumulated = np.zeros((num_goals, num_states))
        for step in range(repeat):
            accumulated += primitive_gamma**step * (power @ state_rewards.T).T
            power = power @ transition
        repeated.append(power.tocsr())
        rewards[:, :, action] = accumulated
    return repeated, rewards


def optimal_q(
    transitions: list[sp.csr_matrix],
    rewards: np.ndarray,
    discount: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, float]:
    q = np.zeros_like(rewards)
    for iteration in range(1, max_iterations + 1):
        values = q.max(axis=2)
        continuation = np.stack(
            [(transition @ values.T).T for transition in transitions], axis=2
        )
        updated = rewards + discount * continuation
        residual = float(np.max(np.abs(updated - q)))
        q = updated
        if residual <= tolerance:
            return q, iteration, residual
    raise RuntimeError(f"Value iteration failed: residual={residual:g}")


def transition_sa(transitions: list[sp.csr_matrix]) -> sp.csr_matrix:
    """State-major/action-minor map from SA rows to next-state distributions."""
    return sp.vstack(
        [sp.vstack([transition[state] for transition in transitions]) for state in range(transitions[0].shape[0])],
        format="csr",
    )


@dataclass
class Successor:
    transition: sp.csr_matrix
    policy: np.ndarray
    discount: float

    def __post_init__(self) -> None:
        self.n, self.s = self.transition.shape
        self.selected = np.arange(self.s) * NUM_ACTIONS + self.policy
        chain = self.transition[self.selected].toarray()
        self.lu, self.pivots = scipy.linalg.lu_factor(
            np.eye(self.s) - self.discount * chain, check_finite=False
        )
        self.operator = LinearOperator(
            (self.n, self.n), matvec=self.matvec, rmatvec=self.rmatvec,
            matmat=self.matmat, rmatmat=self.rmatmat, dtype=float
        )

    def matmat(self, values: np.ndarray) -> np.ndarray:
        correction = scipy.linalg.lu_solve(
            (self.lu, self.pivots), values[self.selected], check_finite=False
        )
        return values + self.discount * (self.transition @ correction)

    def rmatmat(self, values: np.ndarray) -> np.ndarray:
        correction = scipy.linalg.lu_solve(
            (self.lu, self.pivots), self.transition.T @ values,
            trans=1, check_finite=False
        )
        result = values.copy()
        result[self.selected] += self.discount * correction
        return result

    def matvec(self, value: np.ndarray) -> np.ndarray:
        return self.matmat(value[:, None]).ravel()

    def rmatvec(self, value: np.ndarray) -> np.ndarray:
        return self.rmatmat(value[:, None]).ravel()


def top_eigenpairs(
    operator: LinearOperator, count: int, tolerance: float, max_iterations: int, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    if count >= operator.shape[0]:
        matrix = np.column_stack([operator @ np.eye(operator.shape[0])[:, i] for i in range(operator.shape[0])])
        values, vectors = np.linalg.eigh(matrix)
    else:
        try:
            values, vectors = eigsh(
                operator, k=count, which="LA", tol=tolerance, maxiter=max_iterations,
                v0=np.random.default_rng(seed).standard_normal(operator.shape[0])
            )
        except ArpackNoConvergence as error:
            if len(error.eigenvalues) < count:
                raise RuntimeError(f"ARPACK returned {len(error.eigenvalues)}/{count}") from error
            values, vectors = error.eigenvalues, error.eigenvectors
    order = np.argsort(values)[::-1][:count]
    values, vectors = np.maximum(values[order], 0.0), vectors[:, order]
    residual = max(
        np.linalg.norm(operator @ vectors[:, i] - values[i] * vectors[:, i])
        / max(abs(values[i]), TINY)
        for i in range(count)
    )
    return values, vectors, float(residual)


def fit_backward(
    successors: list[Successor], rank: int, tolerance: float, max_iterations: int, seed: int
) -> tuple[np.ndarray, float]:
    n = successors[0].n
    covariance = LinearOperator(
        (n, n),
        matvec=lambda x: sum((m.rmatvec(m.matvec(x)) for m in successors), start=np.zeros(n)),
        dtype=float,
    )
    _, backward, residual = top_eigenpairs(covariance, rank, tolerance, max_iterations, seed)
    orthogonality = np.linalg.norm(backward.T @ backward - np.eye(rank), ord=2)
    if backward.shape != (n, rank) or orthogonality > 1e-6:
        raise AssertionError(f"Invalid B: shape={backward.shape}, orthogonality={orthogonality:g}")
    return backward, residual


def model_operator(target: Successor, backward: np.ndarray) -> LinearOperator:
    return LinearOperator(
        (target.n, target.n),
        matvec=lambda x: target.matvec(backward @ (backward.T @ x)),
        rmatvec=lambda x: backward @ (backward.T @ target.rmatvec(x)),
        dtype=float,
    )


def difference(left: LinearOperator, right: LinearOperator) -> LinearOperator:
    return LinearOperator(
        left.shape,
        matvec=lambda x: left @ x - right @ x,
        rmatvec=lambda x: left.rmatvec(x) - right.rmatvec(x),
        dtype=float,
    )


def singular_values(
    operator: LinearOperator, count: int, tolerance: float, max_iterations: int, seed: int
) -> tuple[np.ndarray, float]:
    normal = LinearOperator(
        operator.shape, matvec=lambda x: operator.rmatvec(operator @ x), dtype=float
    )
    values, _, residual = top_eigenpairs(normal, count, tolerance, max_iterations, seed)
    return np.sqrt(values), residual


@dataclass(frozen=True)
class Configuration:
    layout: str
    size: int
    nominal_gamma: float
    repeat: int
    rank: int
    goals: int
    radius: float
    max_fb_iterations: int
    vi_tolerance: float
    vi_max_iterations: int
    spectral_tolerance: float
    spectral_max_iterations: int
    seed: int


def terminal_repeat_diagnostics(
    transitions: list[sp.csr_matrix], region: np.ndarray, gamma: float, repeat: int,
    tolerance: float, max_iterations: int
) -> tuple[float, float, float, float]:
    """Absolute and normalized V gaps for separately labeled first-hit diagnostics."""
    terminal = np.flatnonzero(region)
    live = np.flatnonzero(~region.astype(bool))

    def solve(k: int) -> np.ndarray:
        kernels, rewards = [], np.zeros((len(live), NUM_ACTIONS))
        for action, transition in enumerate(transitions):
            survive = transition[live][:, live].tocsr()
            hit = np.asarray(transition[live][:, terminal].sum(axis=1)).ravel()
            power = sp.eye(len(live), format="csr")
            for step in range(k):
                rewards[:, action] += gamma**step * (power @ hit)
                power = power @ survive
            kernels.append(power)
        q = np.zeros_like(rewards)
        discount = gamma**k
        for _ in range(max_iterations):
            updated = rewards + discount * np.column_stack([kernel @ q.max(axis=1) for kernel in kernels])
            if np.max(np.abs(updated - q)) <= tolerance:
                return updated.max(axis=1)
            q = updated
        raise RuntimeError("Terminal diagnostic did not converge")

    primitive_v, repeated_v = solve(1), solve(repeat)
    gap = np.abs(primitive_v - repeated_v)
    worst = float(gap.max(initial=0.0))
    mean = float(gap.mean() if len(gap) else 0.0)
    scale = max(float(np.max(np.abs(primitive_v), initial=0.0)), TINY)
    return worst, mean, worst / scale, mean / scale


def evaluate(configuration: Configuration) -> list[dict]:
    free = layout_from_name(configuration.layout, configuration.size)
    transitions, coordinates = action_transitions(free)
    centers, state_rewards = spatial_goal_regions(coordinates, configuration.goals, configuration.radius)
    primitive_gamma = configuration.nominal_gamma ** (1.0 / configuration.repeat)
    repeated_transitions, repeated_rewards = macro_mdp(
        transitions, state_rewards, primitive_gamma, configuration.repeat
    )
    original_rewards = np.repeat(state_rewards[:, :, None], NUM_ACTIONS, axis=2)
    original_q, original_iterations, original_residual = optimal_q(
        transitions, original_rewards, primitive_gamma,
        configuration.vi_tolerance, configuration.vi_max_iterations
    )
    repeated_q, repeated_iterations, repeated_residual = optimal_q(
        repeated_transitions, repeated_rewards, configuration.nominal_gamma,
        configuration.vi_tolerance, configuration.vi_max_iterations
    )
    transition = transition_sa(repeated_transitions)
    policies = repeated_q.argmax(axis=2)
    seen: dict[bytes, int] = {}
    history: list[float] = []
    snapshots = []
    converged = False
    cycle_length = 0
    final = None

    for iteration in range(1, configuration.max_fb_iterations + 1):
        key = policies.astype(np.uint8).tobytes()
        if key in seen:
            cycle_length = iteration - seen[key]
            break
        seen[key] = iteration
        targets = [Successor(transition, policy, configuration.nominal_gamma) for policy in policies]
        backward, fit_residual = fit_backward(
            targets, configuration.rank, configuration.spectral_tolerance,
            configuration.spectral_max_iterations, configuration.seed + iteration
        )
        models = [model_operator(target, backward) for target in targets]
        approximate_q = np.stack(
            [models[g] @ repeated_rewards[g].reshape(-1) for g in range(configuration.goals)]
        ).reshape(configuration.goals, len(coordinates), NUM_ACTIONS)
        greedy = approximate_q.argmax(axis=2)  # first-index tie break
        mismatch = float(np.mean(greedy != policies))
        history.append(mismatch)
        final = (backward, targets, models, approximate_q, greedy, fit_residual, iteration)
        snapshots.append((mismatch, final))
        if np.array_equal(greedy, policies):
            converged = True
            break
        policies = greedy

    if final is None:
        raise RuntimeError("FB alternation produced no snapshot")
    selected_mismatch, final = min(snapshots, key=lambda item: item[0])
    backward, targets, models, approximate_q, stored_policies, fit_residual, iterations = final
    induced_again = np.stack(
        [(models[g] @ repeated_rewards[g].reshape(-1)).reshape(len(coordinates), NUM_ACTIONS).argmax(axis=1)
         for g in range(configuration.goals)]
    )
    if not np.array_equal(induced_again, stored_policies):
        raise AssertionError("Final stored model does not induce final stored policy")

    block_spectrum = []
    for primitive in transitions:
        block_spectrum.extend(np.abs(np.linalg.eigvals(primitive.toarray())))
    block_spectrum = np.sort(np.asarray(block_spectrum))[::-1]
    block_lambda = float(block_spectrum[configuration.rank]) if configuration.rank < len(block_spectrum) else 0.0
    terminal = [
        terminal_repeat_diagnostics(
            transitions, state_rewards[g], primitive_gamma, configuration.repeat,
            configuration.vi_tolerance, configuration.vi_max_iterations
        )
        for g in range(configuration.goals)
    ]

    records = []
    n = len(coordinates) * NUM_ACTIONS
    beta = configuration.nominal_gamma
    for goal in range(configuration.goals):
        reward = repeated_rewards[goal].reshape(-1)
        qhat = approximate_q[goal].reshape(-1)
        induced = Successor(transition, stored_policies[goal], beta)
        q_policy = induced.matvec(reward)
        model = models[goal]
        model_diff = difference(model, induced.operator)
        model_sv, model_residual = singular_values(
            model_diff, 1, configuration.spectral_tolerance,
            configuration.spectral_max_iterations, configuration.seed + 100_000 + goal
        )
        successor_sv, sigma_residual = singular_values(
            induced.operator, configuration.rank + 1, configuration.spectral_tolerance,
            configuration.spectral_max_iterations, configuration.seed + 200_000 + goal
        )
        model_error = float(model_sv[0])
        sigma = float(successor_sv[configuration.rank])
        eym_tolerance = max(1e-7, 20 * configuration.spectral_tolerance * max(model_error, sigma))
        if model_error + eym_tolerance < sigma:
            raise AssertionError("EYM nonnegativity violated")
        realization = max(model_error - sigma, 0.0)
        if abs(model_error - (realization + sigma)) > eym_tolerance:
            raise AssertionError("model_error != realization + sigma")

        chain = transition[induced.selected].toarray()
        lambdas = np.sort(np.abs(np.linalg.eigvals(chain)))[::-1]
        joint_lambda = float(lambdas[configuration.rank]) if configuration.rank < len(lambdas) else 0.0
        original = original_q[goal].reshape(-1)
        repeated = repeated_q[goal].reshape(-1)
        epsilon_repeat = float(np.max(np.abs(original - repeated)))
        observed = float(np.max(np.abs(qhat - original)))
        repeat_q_error = float(np.max(np.abs(qhat - repeated)))
        qhat_policy_error = float(np.max(np.abs(qhat - q_policy)))
        policy_optimality_gap = float(np.max(np.abs(q_policy - repeated)))
        direct_reward_error = float(np.max(np.abs(model_diff @ reward)))
        reward_l2, reward_inf = float(np.linalg.norm(reward)), float(np.max(np.abs(reward)))
        stage = [
            observed,
            epsilon_repeat + repeat_q_error,
            epsilon_repeat + qhat_policy_error + policy_optimality_gap,
            epsilon_repeat + 2.0 / (1.0 - beta) * direct_reward_error,
            epsilon_repeat + 2.0 / (1.0 - beta) * reward_l2 * model_error,
            epsilon_repeat + 2.0 * np.sqrt(n) * reward_inf / (1.0 - beta) * model_error,
            epsilon_repeat + 2.0 * np.sqrt(n) / (1.0 - primitive_gamma)
            * (realization + sigma),
        ]
        exact_greedy_factor = epsilon_repeat + (1.0 + beta) / (1.0 - beta) * direct_reward_error
        bellman_policy = float(np.max(np.abs(
            q_policy.reshape(len(coordinates), NUM_ACTIONS)
            - repeated_rewards[goal]
            - beta * np.stack([p @ q_policy.reshape(len(coordinates), NUM_ACTIONS).max(axis=1) for p in repeated_transitions], axis=1)
        )))
        policy_values = q_policy.reshape(len(coordinates), NUM_ACTIONS)[
            np.arange(len(coordinates)), stored_policies[goal]
        ]
        bellman_policy_exact = float(np.max(np.abs(
            q_policy.reshape(len(coordinates), NUM_ACTIONS)
            - repeated_rewards[goal]
            - beta * np.stack([p @ policy_values for p in repeated_transitions], axis=1)
        )))
        tolerance = max(1e-7, 50 * configuration.vi_tolerance / (1.0 - beta))
        dominance = [True] + [value + tolerance >= observed for value in stage[1:]]
        if not all(dominance):
            raise AssertionError(f"Claimed stage failed to dominate: {dominance}")
        if bellman_policy_exact > 1e-7 or max(original_residual, repeated_residual) > configuration.vi_tolerance * 1.01:
            raise AssertionError("Bellman residual assertion failed")
        z = backward.T @ reward
        f = targets[goal].matmat(backward)
        if not np.allclose(f @ z, qhat, atol=1e-8, rtol=1e-8):
            raise AssertionError("F z != Mhat r")

        record = {
            "layout": configuration.layout, "layout_size": configuration.size,
            "states": len(coordinates), "state_actions": n, "actions": NUM_ACTIONS,
            "nominal_gamma": beta, "primitive_gamma": primitive_gamma,
            "repeat": configuration.repeat, "rank": configuration.rank,
            "goal": goal, "goal_state": int(centers[goal]),
            "goal_row": int(coordinates[centers[goal], 0]), "goal_col": int(coordinates[centers[goal], 1]),
            "goal_radius": configuration.radius, "goal_region_states": int(state_rewards[goal].sum()),
            "criterion": "continuing binary region reward on common reward-free MDP",
            "nu_convention": "nu absorbed; Euclidean orthonormal B and SA reward coordinates",
            "primitive_action_intended_probability": INTENDED_PROBABILITY,
            "noop_is_deterministic_stay": False,
            "noop_intended_stay_probability": INTENDED_PROBABILITY,
            "q_error_original_inf": observed, "epsilon_repeat_q_inf": epsilon_repeat,
            "epsilon_repeat_q_normalized": epsilon_repeat / max(float(np.max(np.abs(original))), TINY),
            "qhat_repeat_qstar_inf": repeat_q_error, "qhat_qpolicy_inf": qhat_policy_error,
            "qpolicy_repeat_qstar_inf": policy_optimality_gap,
            "direct_reward_projected_model_error_inf": direct_reward_error,
            "reward_l2": reward_l2, "reward_inf": reward_inf,
            "model_error_2": model_error, "sigma_d_plus_1": sigma,
            "realization_error": realization, "model_equals_realization_plus_sigma": True,
            "joint_lambda_d_plus_1": joint_lambda, "primitive_block_lambda_d_plus_1": block_lambda,
            "stage0_observed": stage[0], "stage1_triangle": stage[1],
            "stage2_policy_decomposition": stage[2], "stage3_direct_model": stage[3],
            "stage3_exact_one_plus_beta_diagnostic": exact_greedy_factor,
            "stage4_l2_matrix": stage[4], "stage5_infinity_conversion": stage[5],
            "stage6_exact_sigma_paper": stage[6],
            "stage1_dominates": dominance[1], "stage2_dominates": dominance[2],
            "stage3_dominates": dominance[3], "stage4_dominates": dominance[4],
            "stage5_dominates": dominance[5], "stage6_dominates": dominance[6],
            "fb_converged": converged, "fb_iterations": iterations,
            "fb_cycle_length": cycle_length, "final_policy_mismatch_fraction": selected_mismatch,
            "minimum_policy_mismatch_fraction": min(history),
            "induced_policy_verified": True,
            "induced_policy_sha256": hashlib.sha256(stored_policies[goal].astype(np.uint8).tobytes()).hexdigest(),
            "fit_relative_eigen_residual": fit_residual,
            "model_normal_relative_eigen_residual": model_residual,
            "sigma_normal_relative_eigen_residual": sigma_residual,
            "original_vi_iterations": original_iterations, "repeat_vi_iterations": repeated_iterations,
            "original_vi_residual": original_residual, "repeat_vi_residual": repeated_residual,
            "qpolicy_bellman_optimality_residual_diagnostic": bellman_policy,
            "qpolicy_fixed_policy_bellman_residual": bellman_policy_exact,
            "terminal_first_hit_worst_v_gap": terminal[goal][0],
            "terminal_first_hit_mean_state_v_gap": terminal[goal][1],
            "terminal_first_hit_worst_v_gap_normalized": terminal[goal][2],
            "terminal_first_hit_mean_state_v_gap_normalized": terminal[goal][3],
        }
        records.append(record)
    return records


def add_empirical_lemma_stages(records: list[dict]) -> None:
    """Calibrate constants over displayed samples; these are diagnostics, not theory."""
    groups: dict[tuple[str, float, int], list[dict]] = {}
    for record in records:
        groups.setdefault((record["layout"], record["nominal_gamma"], record["rank"]), []).append(record)
    for group in groups.values():
        c_sf = max(r["sigma_d_plus_1"] * (1.0 - r["nominal_gamma"] * r["joint_lambda_d_plus_1"]) for r in group)
        c_rep = max(
            r["joint_lambda_d_plus_1"] / max(r["primitive_block_lambda_d_plus_1"] ** r["repeat"], TINY)
            for r in group
        )
        for r in group:
            beta = r["nominal_gamma"]
            prefactor = 2.0 * np.sqrt(r["state_actions"]) / (1.0 - r["primitive_gamma"])
            sigma_41 = c_sf / (1.0 - beta * r["joint_lambda_d_plus_1"])
            empirical_lambda = min(1.0, c_rep * r["primitive_block_lambda_d_plus_1"] ** r["repeat"])
            sigma_42 = c_sf / (1.0 - beta * empirical_lambda)
            floor = (2.0 * r["state_actions"] - 1.0) * np.sqrt(NUM_ACTIONS)
            floor_lambda = min(1.0, floor * r["primitive_block_lambda_d_plus_1"] ** r["repeat"])
            floor_sigma = c_sf / (1.0 - beta * floor_lambda)
            stage7 = r["epsilon_repeat_q_inf"] + prefactor * (r["realization_error"] + sigma_41)
            stage8 = r["epsilon_repeat_q_inf"] + prefactor * (r["realization_error"] + sigma_42)
            floor_value = r["epsilon_repeat_q_inf"] + prefactor * (r["realization_error"] + floor_sigma)
            r.update({
                "empirical_c_sf": c_sf, "empirical_c_rep": c_rep,
                "empirical_calibration_scope": "max over sampled goals and k at fixed layout,gamma,d",
                "stage7_empirical_lemma41_diagnostic": stage7,
                "stage8_empirical_lemma42_diagnostic": stage8,
                "stage7_dominates": stage7 + 1e-7 >= r["q_error_original_inf"],
                "stage8_dominates": stage8 + 1e-7 >= r["q_error_original_inf"],
                "theoretical_c_rep_floor_kappa_one": floor,
                "optimistic_theoretical_floor_diagnostic": floor_value,
                "optimistic_floor_dominates": floor_value + 1e-7 >= r["q_error_original_inf"],
                "theoretical_kappa_verified": False,
            })
    if not all(r["stage7_dominates"] and r["stage8_dominates"] for r in records):
        raise AssertionError("Empirically calibrated lemma diagnostic failed sample dominance")


def validate_full_rank() -> dict[str, float | bool]:
    free = bordered_single_room_layout(5, 5, border=1)
    transitions, coordinates = action_transitions(free)
    _, rewards = spatial_goal_regions(coordinates, 3, 1.5)
    gamma = 0.9
    reward_sa = np.repeat(rewards[:, :, None], NUM_ACTIONS, axis=2)
    qstar, _, residual = optimal_q(transitions, reward_sa, gamma, 1e-12, 100_000)
    transition = transition_sa(transitions)
    policies = qstar.argmax(axis=2)
    successors = [Successor(transition, p, gamma) for p in policies]
    n = len(coordinates) * NUM_ACTIONS
    backward = np.eye(n)
    qhat = np.stack(
        [successors[g].matvec(reward_sa[g].reshape(-1)) for g in range(3)]
    ).reshape(qstar.shape)
    induced = qhat.argmax(axis=2)
    q_error = float(np.max(np.abs(qhat - qstar)))
    policy_match = float(np.mean(induced == policies))
    # Exact Q values can have action ties; independent linear/iterative solves may
    # choose different tied first indices at roundoff scale.
    if q_error > 2e-9 or policy_match < 0.9 or residual > 1e-12 * 1.01:
        raise AssertionError("Full-rank validation failed")
    if np.linalg.matrix_rank(backward) != n:
        raise AssertionError("Full-rank B is rank deficient")
    return {"passed": True, "states": len(coordinates), "state_actions": n,
            "rank": n, "q_error": q_error, "policy_match": policy_match, "bellman_residual": residual}


def reserve_outputs(output_dir: Path) -> dict[str, Path]:
    outputs = {name: output_dir / name for name in [
        "matrix_fb_stagewise_results.csv", "validation.csv"
    ]}
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing outputs: " + ", ".join(existing))
    return outputs


def write_csv(records: list[dict], output: Path) -> None:
    with output.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", nargs="+", choices=["Single Room", "Four Rooms"], default=["Single Room", "Four Rooms"])
    parser.add_argument("--size", type=int, default=DEFAULT_LAYOUT_SIZE)
    parser.add_argument("--gammas", type=float, nargs="+", default=DEFAULT_GAMMAS)
    parser.add_argument("--repeats", type=int, nargs="+", default=DEFAULT_REPEATS)
    parser.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    parser.add_argument("--goals", type=int, default=DEFAULT_GOALS)
    parser.add_argument("--goal-radius", type=float, default=GOAL_RADIUS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-fb-iterations", type=int, default=20)
    parser.add_argument("--vi-tolerance", type=float, default=1e-10)
    parser.add_argument("--vi-max-iterations", type=int, default=100_000)
    parser.add_argument("--spectral-tolerance", type=float, default=2e-5)
    parser.add_argument("--spectral-max-iterations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("assets/bound_tightness/matrix_fb"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use one layout/gamma/repeat/rank and four goals")
    args = parser.parse_args()
    validation = validate_full_rank()
    print("Validation:", validation, flush=True)
    if args.validate_only:
        return
    if args.smoke:
        args.layouts, args.gammas, args.repeats, args.ranks, args.goals = ["Single Room"], [0.95], [2], [16], 4
    if any(rank < 1 for rank in args.ranks) or args.workers < 1:
        raise ValueError("Ranks and workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = reserve_outputs(args.output_dir)
    configurations = [
        Configuration(layout, args.size, gamma, repeat, rank, args.goals, args.goal_radius,
                      args.max_fb_iterations, args.vi_tolerance, args.vi_max_iterations,
                      args.spectral_tolerance, args.spectral_max_iterations,
                      args.seed + 100_000 * li + 10_000 * gi + 100 * repeat + rank)
        for li, layout in enumerate(args.layouts) for gi, gamma in enumerate(args.gammas)
        for repeat in args.repeats for rank in args.ranks
    ]
    start = time.perf_counter(); records: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(configurations))) as executor:
        futures = {executor.submit(evaluate, config): config for config in configurations}
        for future in as_completed(futures):
            config = futures[future]; records.extend(future.result())
            print(f"Completed {config.layout}, gamma={config.nominal_gamma}, k={config.repeat}, d={config.rank}", flush=True)
    records.sort(key=lambda r: (args.layouts.index(r["layout"]), args.gammas.index(r["nominal_gamma"]), args.repeats.index(r["repeat"]), args.ranks.index(r["rank"]), r["goal"]))
    add_empirical_lemma_stages(records)
    write_csv(records, outputs["matrix_fb_stagewise_results.csv"])
    write_csv([validation], outputs["validation.csv"])
    runtime = time.perf_counter() - start
    configs = {(r["layout"], r["nominal_gamma"], r["repeat"], r["rank"]): r for r in records}
    converged = sum(bool(r["fb_converged"]) for r in configs.values())
    mismatch = np.asarray([r["final_policy_mismatch_fraction"] for r in configs.values()])
    ratios = np.asarray([r["stage6_exact_sigma_paper"] / max(r["q_error_original_inf"], TINY) for r in records])
    print(f"Wrote {len(records)} rows to {args.output_dir}")
    print(f"Runtime: {runtime:.1f} seconds")
    print(f"Fixed points: {converged}/{len(configs)}; final mismatch median/max={np.median(mismatch):.3%}/{mismatch.max():.3%}")
    print(f"Exact-sigma stage/observed median/min={np.median(ratios):.3g}/{ratios.min():.3g}")


if __name__ == "__main__":
    main()
