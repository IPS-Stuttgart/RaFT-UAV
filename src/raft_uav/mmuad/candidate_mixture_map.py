"""Robust candidate-mixture trajectory smoothing for MMUAD.

Hard per-frame candidate selection performed poorly in the MMUAD experiments,
while an uncertainty-aware candidate mixture substantially reduced pose error.
This module promotes that idea into an inference-safe, reusable implementation:

* keep the top-K candidates at each timestamp;
* combine ranker scores with learned per-candidate uncertainty;
* form robust Huber responsibilities around the current trajectory;
* solve a quadratic, irregular-time acceleration-regularized trajectory;
* iterate until the candidate responsibilities and trajectory stabilize.

Validation/test inference does not require truth. Truth is accepted only as an
optional diagnostic input for local score summaries.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns
from raft_uav.mmuad.tracker import add_truth_errors, compute_metrics

LOSS_CHOICES = ("huber", "squared")
SCORE_NORMALIZATION_CHOICES = ("minmax", "rank", "none")
INITIALIZATION_CHOICES = ("uncertainty-top1", "score-top1")


@dataclass(frozen=True)
class CandidateMixtureMapConfig:
    """Configuration for robust candidate-mixture trajectory inference."""

    top_k: int = 20
    score_column: str = "candidate_reservoir_grid_score"
    fallback_score_columns: tuple[str, ...] = ("ranker_score", "confidence")
    sigma_column: str = "predicted_sigma_m"
    default_sigma_m: float = 10.0
    sigma_min_m: float = 1.0
    sigma_max_m: float = 30.0
    score_normalization: str = "minmax"
    score_weight: float = 1.0
    temperature: float = 1.0
    sigma_log_weight: float = 3.0
    loss: str = "huber"
    huber_delta: float = 1.0
    smoothness_weight: float = 7200.0
    iterations: int = 5
    tolerance_m: float = 1.0e-3
    uniform_weight_floor: float = 0.0
    min_measurement_precision: float = 1.0e-6
    max_measurement_precision: float = 1.0e6
    initialization: str = "uncertainty-top1"


@dataclass(frozen=True)
class CandidateMixtureMapResult:
    """Trajectory estimates plus candidate-assignment diagnostics."""

    estimates: pd.DataFrame
    assignments: pd.DataFrame
    iteration_summary: pd.DataFrame
    summary: dict[str, Any]


def run_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    config: CandidateMixtureMapConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> CandidateMixtureMapResult:
    """Estimate one smooth trajectory per sequence from candidate mixtures."""

    config = config or CandidateMixtureMapConfig()
    _validate_config(config)
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        empty = pd.DataFrame()
        return CandidateMixtureMapResult(
            estimates=empty,
            assignments=empty,
            iteration_summary=empty,
            summary={"candidate_rows": 0, "estimate_rows": 0, "config": asdict(config)},
        )
    rows = rows.copy().reset_index(drop=True)
    rows["_mixture_input_row"] = np.arange(len(rows), dtype=int)
    truth_rows = None
    if truth is not None:
        truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    initial_rows = _normalize_initial_estimates(initial_estimates)

    estimate_parts: list[pd.DataFrame] = []
    assignment_parts: list[pd.DataFrame] = []
    iteration_parts: list[pd.DataFrame] = []
    metrics_by_sequence: dict[str, Any] = {}
    convergence_rows: list[dict[str, Any]] = []
    for sequence_id, sequence_rows in rows.groupby("sequence_id", sort=True):
        frames = _prepare_candidate_frames(sequence_rows, config=config)
        if not frames:
            continue
        times = np.asarray([frame["time_s"] for frame in frames], dtype=float)
        state = _initial_trajectory(
            frames,
            times=times,
            sequence_id=str(sequence_id),
            initial_estimates=initial_rows,
            config=config,
        )
        iteration_records: list[dict[str, Any]] = []
        converged = False
        final_response: list[dict[str, Any]] = []
        for iteration in range(1, int(config.iterations) + 1):
            response = _mixture_response(frames, state, config=config)
            updated = _solve_smooth_trajectory(
                times,
                pseudo_positions=np.asarray([item["pseudo_position"] for item in response]),
                measurement_precision=np.asarray(
                    [item["measurement_precision"] for item in response]
                ),
                smoothness_weight=float(config.smoothness_weight),
            )
            displacement = np.linalg.norm(updated - state, axis=1)
            objective = _quadratic_objective(
                times,
                updated,
                pseudo_positions=np.asarray([item["pseudo_position"] for item in response]),
                measurement_precision=np.asarray(
                    [item["measurement_precision"] for item in response]
                ),
                smoothness_weight=float(config.smoothness_weight),
            )
            iteration_records.append(
                {
                    "sequence_id": str(sequence_id),
                    "iteration": int(iteration),
                    "max_displacement_m": float(np.max(displacement)),
                    "mean_displacement_m": float(np.mean(displacement)),
                    "mean_assignment_entropy": float(
                        np.mean([item["entropy"] for item in response])
                    ),
                    "mean_effective_candidate_count": float(
                        np.mean([item["effective_candidate_count"] for item in response])
                    ),
                    "quadratic_objective": float(objective),
                }
            )
            state = updated
            final_response = response
            if float(np.max(displacement)) <= float(config.tolerance_m):
                converged = True
                break
        final_response = _mixture_response(frames, state, config=config)
        estimates = _estimate_rows(
            sequence_id=str(sequence_id),
            times=times,
            state=state,
            response=final_response,
            iterations=len(iteration_records),
            converged=converged,
        )
        sequence_truth = None
        if truth_rows is not None:
            sequence_truth = truth_rows.loc[
                truth_rows["sequence_id"].astype(str) == str(sequence_id)
            ]
            if not sequence_truth.empty:
                estimates = add_truth_errors(estimates, sequence_truth)
        estimate_parts.append(estimates)
        assignment_parts.append(
            _assignment_rows(
                sequence_id=str(sequence_id),
                state=state,
                frames=frames,
                response=final_response,
            )
        )
        iteration_parts.append(pd.DataFrame.from_records(iteration_records))
        metrics_by_sequence[str(sequence_id)] = compute_metrics(estimates, sequence_truth)
        convergence_rows.append(
            {
                "sequence_id": str(sequence_id),
                "frame_count": int(len(times)),
                "iteration_count": int(len(iteration_records)),
                "converged": bool(converged),
            }
        )

    estimates_all = _concat(estimate_parts)
    assignments_all = _concat(assignment_parts)
    iteration_all = _concat(iteration_parts)
    pooled_metrics = compute_metrics(estimates_all, truth_rows)
    summary = {
        "candidate_rows": int(len(rows)),
        "sequence_count": int(estimates_all["sequence_id"].nunique())
        if not estimates_all.empty
        else 0,
        "estimate_rows": int(len(estimates_all)),
        "assignment_rows": int(len(assignments_all)),
        "converged_sequence_count": int(
            sum(bool(row["converged"]) for row in convergence_rows)
        ),
        "config": asdict(config),
        "metrics": {
            "pooled": pooled_metrics,
            "sequences": metrics_by_sequence,
        },
        "convergence": convergence_rows,
    }
    return CandidateMixtureMapResult(
        estimates=estimates_all,
        assignments=assignments_all,
        iteration_summary=iteration_all,
        summary=_jsonable(summary),
    )


def write_candidate_mixture_map_outputs(
    result: CandidateMixtureMapResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write mixture estimates, assignments, iteration rows, and summary JSON."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "estimates_csv": output / "mmuad_candidate_mixture_estimates.csv",
        "assignments_csv": output / "mmuad_candidate_mixture_assignments.csv",
        "iterations_csv": output / "mmuad_candidate_mixture_iterations.csv",
        "summary_json": output / "mmuad_candidate_mixture_summary.json",
    }
    result.estimates.to_csv(paths["estimates_csv"], index=False)
    result.assignments.to_csv(paths["assignments_csv"], index=False)
    result.iteration_summary.to_csv(paths["iterations_csv"], index=False)
    paths["summary_json"].write_text(
        json.dumps(_jsonable(result.summary), indent=2),
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-mixture-map",
        description="run robust uncertainty-aware MMUAD candidate-mixture smoothing",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--fallback-score-column", action="append", default=[])
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument(
        "--score-normalization",
        choices=SCORE_NORMALIZATION_CHOICES,
        default="minmax",
    )
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=3.0)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tolerance-m", type=float, default=1.0e-3)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_CHOICES,
        default="uncertainty-top1",
    )
    args = parser.parse_args(argv)

    fallback_columns = tuple(args.fallback_score_column) or ("ranker_score", "confidence")
    candidates = load_candidate_file(args.candidates_csv).rows
    initial_estimates = (
        None if args.initial_estimates_csv is None else pd.read_csv(args.initial_estimates_csv)
    )
    truth = None
    if args.truth_csv is not None:
        truth = load_evaluation_truth_file(args.truth_csv).rows
    result = run_candidate_mixture_map(
        candidates,
        config=CandidateMixtureMapConfig(
            top_k=args.top_k,
            score_column=args.score_column,
            fallback_score_columns=fallback_columns,
            sigma_column=args.sigma_column,
            default_sigma_m=args.default_sigma_m,
            sigma_min_m=args.sigma_min_m,
            sigma_max_m=args.sigma_max_m,
            score_normalization=args.score_normalization,
            score_weight=args.score_weight,
            temperature=args.temperature,
            sigma_log_weight=args.sigma_log_weight,
            loss=args.loss,
            huber_delta=args.huber_delta,
            smoothness_weight=args.smoothness_weight,
            iterations=args.iterations,
            tolerance_m=args.tolerance_m,
            uniform_weight_floor=args.uniform_weight_floor,
            initialization=args.initialization,
        ),
        initial_estimates=initial_estimates,
        truth=truth,
    )
    paths = write_candidate_mixture_map_outputs(result, args.output_dir)
    print("mmuad_candidate_mixture_map=ok")
    print(f"estimate_rows={len(result.estimates)}")
    pooled = result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _prepare_candidate_frames(
    sequence_rows: pd.DataFrame,
    *,
    config: CandidateMixtureMapConfig,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for time_s, frame in sequence_rows.groupby("time_s", sort=True):
        group = frame.copy()
        group["_mixture_raw_score"] = _candidate_scores(group, config=config)
        group["_mixture_sigma_m"] = _candidate_sigmas(group, config=config)
        group = group.sort_values(
            ["_mixture_raw_score", "_mixture_sigma_m", "_mixture_input_row"],
            ascending=[False, True, True],
        ).head(int(config.top_k))
        if group.empty:
            continue
        group = group.reset_index(drop=True)
        group["_mixture_candidate_rank"] = np.arange(1, len(group) + 1, dtype=int)
        normalized_score = _normalize_scores(
            group["_mixture_raw_score"].to_numpy(float),
            mode=config.score_normalization,
        )
        group["_mixture_normalized_score"] = normalized_score
        frames.append(
            {
                "time_s": float(time_s),
                "rows": group,
                "positions": group[["x_m", "y_m", "z_m"]].to_numpy(float),
                "raw_scores": group["_mixture_raw_score"].to_numpy(float),
                "normalized_scores": normalized_score,
                "sigmas": group["_mixture_sigma_m"].to_numpy(float),
            }
        )
    return frames


def _initial_trajectory(
    frames: Sequence[dict[str, Any]],
    *,
    times: np.ndarray,
    sequence_id: str,
    initial_estimates: pd.DataFrame | None,
    config: CandidateMixtureMapConfig,
) -> np.ndarray:
    fallback = []
    for frame in frames:
        scores = np.asarray(frame["normalized_scores"], dtype=float)
        sigmas = np.asarray(frame["sigmas"], dtype=float)
        if config.initialization == "score-top1":
            index = int(np.argmax(scores))
        else:
            initial_log_score = (
                float(config.score_weight) * scores / float(config.temperature)
                - float(config.sigma_log_weight) * np.log(sigmas)
            )
            index = int(np.argmax(initial_log_score))
        fallback.append(np.asarray(frame["positions"][index], dtype=float))
    fallback_state = np.asarray(fallback, dtype=float)
    if initial_estimates is None or initial_estimates.empty:
        return fallback_state
    sequence_initial = initial_estimates.loc[
        initial_estimates["sequence_id"].astype(str) == str(sequence_id)
    ]
    if sequence_initial.empty:
        return fallback_state
    initial_times = sequence_initial["time_s"].to_numpy(float)
    order = np.argsort(initial_times)
    initial_times = initial_times[order]
    initial_xyz = sequence_initial[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)[
        order
    ]
    if len(initial_times) == 1:
        interpolated = np.repeat(initial_xyz, len(times), axis=0)
    else:
        interpolated = np.column_stack(
            [np.interp(times, initial_times, initial_xyz[:, axis]) for axis in range(3)]
        )
    finite = np.isfinite(interpolated).all(axis=1)
    return np.where(finite[:, None], interpolated, fallback_state)


def _mixture_response(
    frames: Sequence[dict[str, Any]],
    state: np.ndarray,
    *,
    config: CandidateMixtureMapConfig,
) -> list[dict[str, Any]]:
    response: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        positions = np.asarray(frame["positions"], dtype=float)
        sigmas = np.asarray(frame["sigmas"], dtype=float)
        scores = np.asarray(frame["normalized_scores"], dtype=float)
        distances = np.linalg.norm(positions - state[frame_index], axis=1)
        normalized_residual = distances / sigmas
        robust_cost = _robust_cost(normalized_residual, config=config)
        log_weight = (
            float(config.score_weight) * scores / float(config.temperature)
            - robust_cost
            - float(config.sigma_log_weight) * np.log(sigmas)
        )
        weights = _stable_softmax(log_weight)
        floor = float(config.uniform_weight_floor)
        if floor > 0.0:
            weights = (1.0 - floor) * weights + floor / len(weights)
        pseudo = np.sum(weights[:, None] * positions, axis=0)
        spread_variance = np.sum(
            weights * np.sum((positions - pseudo) ** 2, axis=1) / 3.0
        )
        noise_variance = np.sum(weights * sigmas**2)
        effective_variance = max(float(noise_variance + spread_variance), 1.0e-12)
        precision = float(
            np.clip(
                1.0 / effective_variance,
                float(config.min_measurement_precision),
                float(config.max_measurement_precision),
            )
        )
        entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1.0e-300))))
        response.append(
            {
                "weights": weights,
                "distances": distances,
                "normalized_residual": normalized_residual,
                "robust_cost": robust_cost,
                "log_weight": log_weight,
                "pseudo_position": pseudo,
                "effective_sigma_m": float(np.sqrt(effective_variance)),
                "measurement_precision": precision,
                "entropy": entropy,
                "effective_candidate_count": float(np.exp(entropy)),
                "dominant_index": int(np.argmax(weights)),
            }
        )
    return response


def _solve_smooth_trajectory(
    times: np.ndarray,
    *,
    pseudo_positions: np.ndarray,
    measurement_precision: np.ndarray,
    smoothness_weight: float,
) -> np.ndarray:
    count = len(times)
    precision = np.asarray(measurement_precision, dtype=float)
    system = np.diag(precision)
    second_derivative = _second_derivative_matrix(times)
    if second_derivative.size and smoothness_weight > 0.0:
        system = system + float(smoothness_weight) * (
            second_derivative.T @ second_derivative
        )
    ridge = max(float(np.mean(np.diag(system))), 1.0) * 1.0e-10
    system = system + ridge * np.eye(count)
    estimate = np.zeros_like(pseudo_positions, dtype=float)
    for axis in range(3):
        rhs = precision * pseudo_positions[:, axis]
        try:
            estimate[:, axis] = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            estimate[:, axis] = np.linalg.pinv(system) @ rhs
    return estimate


def _second_derivative_matrix(times: np.ndarray) -> np.ndarray:
    count = len(times)
    if count < 3:
        return np.zeros((0, count), dtype=float)
    matrix = np.zeros((count - 2, count), dtype=float)
    for row, center in enumerate(range(1, count - 1)):
        left_dt = max(float(times[center] - times[center - 1]), 1.0e-6)
        right_dt = max(float(times[center + 1] - times[center]), 1.0e-6)
        scale = 2.0 / (left_dt + right_dt)
        matrix[row, center - 1] = scale / left_dt
        matrix[row, center] = -scale * (1.0 / left_dt + 1.0 / right_dt)
        matrix[row, center + 1] = scale / right_dt
    return matrix


def _quadratic_objective(
    times: np.ndarray,
    state: np.ndarray,
    *,
    pseudo_positions: np.ndarray,
    measurement_precision: np.ndarray,
    smoothness_weight: float,
) -> float:
    measurement = float(
        np.sum(measurement_precision[:, None] * (state - pseudo_positions) ** 2)
    )
    second_derivative = _second_derivative_matrix(times)
    smoothness = 0.0
    if second_derivative.size and smoothness_weight > 0.0:
        smoothness = float(smoothness_weight) * float(
            np.sum((second_derivative @ state) ** 2)
        )
    return measurement + smoothness


def _estimate_rows(
    *,
    sequence_id: str,
    times: np.ndarray,
    state: np.ndarray,
    response: Sequence[dict[str, Any]],
    iterations: int,
    converged: bool,
) -> pd.DataFrame:
    velocity = _trajectory_velocity(times, state)
    records: list[dict[str, Any]] = []
    for index, time_s in enumerate(times):
        item = response[index]
        records.append(
            {
                "sequence_id": sequence_id,
                "time_s": float(time_s),
                "source": "candidate-mixture-map",
                "track_id": "candidate-mixture-map",
                "update_action": "candidate_mixture_map",
                "selected_path_update": True,
                "state_x_m": float(state[index, 0]),
                "state_y_m": float(state[index, 1]),
                "state_z_m": float(state[index, 2]),
                "v_x_mps": float(velocity[index, 0]),
                "v_y_mps": float(velocity[index, 1]),
                "v_z_mps": float(velocity[index, 2]),
                "mixture_candidate_count": int(len(item["weights"])),
                "mixture_effective_candidate_count": float(
                    item["effective_candidate_count"]
                ),
                "mixture_assignment_entropy": float(item["entropy"]),
                "mixture_dominant_weight": float(
                    item["weights"][int(item["dominant_index"])]
                ),
                "mixture_effective_sigma_m": float(item["effective_sigma_m"]),
                "mixture_iteration_count": int(iterations),
                "mixture_converged": bool(converged),
            }
        )
    return pd.DataFrame.from_records(records)


def _assignment_rows(
    *,
    sequence_id: str,
    state: np.ndarray,
    frames: Sequence[dict[str, Any]],
    response: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        rows = frame["rows"]
        item = response[frame_index]
        dominant = int(item["dominant_index"])
        for candidate_index, (_, candidate) in enumerate(rows.iterrows()):
            records.append(
                {
                    "sequence_id": sequence_id,
                    "time_s": float(frame["time_s"]),
                    "candidate_input_row": int(candidate["_mixture_input_row"]),
                    "candidate_rank": int(candidate["_mixture_candidate_rank"]),
                    "source": candidate.get("source"),
                    "track_id": candidate.get("track_id"),
                    "candidate_branch": candidate.get("candidate_branch"),
                    "x_m": float(candidate["x_m"]),
                    "y_m": float(candidate["y_m"]),
                    "z_m": float(candidate["z_m"]),
                    "mixture_raw_score": float(candidate["_mixture_raw_score"]),
                    "mixture_normalized_score": float(
                        candidate["_mixture_normalized_score"]
                    ),
                    "mixture_sigma_m": float(candidate["_mixture_sigma_m"]),
                    "mixture_final_weight": float(item["weights"][candidate_index]),
                    "mixture_distance_to_state_m": float(
                        item["distances"][candidate_index]
                    ),
                    "mixture_normalized_residual": float(
                        item["normalized_residual"][candidate_index]
                    ),
                    "mixture_robust_cost": float(item["robust_cost"][candidate_index]),
                    "mixture_log_weight": float(item["log_weight"][candidate_index]),
                    "mixture_dominant": bool(candidate_index == dominant),
                    "state_x_m": float(state[frame_index, 0]),
                    "state_y_m": float(state[frame_index, 1]),
                    "state_z_m": float(state[frame_index, 2]),
                }
            )
    return pd.DataFrame.from_records(records)


def _candidate_scores(
    rows: pd.DataFrame,
    *,
    config: CandidateMixtureMapConfig,
) -> pd.Series:
    columns = (config.score_column, *config.fallback_score_columns)
    result = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in columns:
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        result = result.where(result.notna(), values)
    return result.fillna(0.0).astype(float)


def _candidate_sigmas(
    rows: pd.DataFrame,
    *,
    config: CandidateMixtureMapConfig,
) -> pd.Series:
    if config.sigma_column in rows.columns:
        sigma = pd.to_numeric(rows[config.sigma_column], errors="coerce")
    else:
        sigma = pd.Series(np.nan, index=rows.index, dtype=float)
    if "std_xy_m" in rows.columns:
        fallback = pd.to_numeric(rows["std_xy_m"], errors="coerce")
        sigma = sigma.where(sigma.notna(), fallback)
    sigma = sigma.fillna(float(config.default_sigma_m))
    sigma = sigma.where(sigma > 0.0, float(config.default_sigma_m))
    return sigma.clip(lower=float(config.sigma_min_m), upper=float(config.sigma_max_m))


def _normalize_scores(values: np.ndarray, *, mode: str) -> np.ndarray:
    score = np.asarray(values, dtype=float)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "none":
        return score
    if mode == "rank":
        if len(score) <= 1:
            return np.full(len(score), 0.5, dtype=float)
        order = np.argsort(np.argsort(score, kind="stable"), kind="stable")
        return order.astype(float) / float(len(score) - 1)
    minimum = float(np.min(score))
    maximum = float(np.max(score))
    if maximum <= minimum:
        return np.full(len(score), 0.5, dtype=float)
    return (score - minimum) / (maximum - minimum)


def _robust_cost(
    normalized_residual: np.ndarray,
    *,
    config: CandidateMixtureMapConfig,
) -> np.ndarray:
    residual = np.abs(np.asarray(normalized_residual, dtype=float))
    if config.loss == "squared":
        return 0.5 * residual**2
    delta = float(config.huber_delta)
    return np.where(
        residual <= delta,
        0.5 * residual**2,
        delta * (residual - 0.5 * delta),
    )


def _stable_softmax(log_weight: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weight, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 1.0 / max(len(values), 1), dtype=float)
    floor = float(np.min(values[finite])) - 100.0
    values = np.where(finite, values, floor)
    shifted = np.clip(values - float(np.max(values)), -700.0, 0.0)
    exp_values = np.exp(shifted)
    total = float(np.sum(exp_values))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(values), 1.0 / max(len(values), 1), dtype=float)
    return exp_values / total


def _trajectory_velocity(times: np.ndarray, state: np.ndarray) -> np.ndarray:
    if len(times) <= 1:
        return np.zeros_like(state)
    return np.column_stack(
        [np.gradient(state[:, axis], times, edge_order=1) for axis in range(3)]
    )


def _normalize_initial_estimates(estimates: pd.DataFrame | None) -> pd.DataFrame | None:
    if estimates is None:
        return None
    rows = pd.DataFrame(estimates).copy()
    if rows.empty:
        return rows
    if "sequence_id" not in rows.columns:
        rows["sequence_id"] = "default"
    aliases = {
        "x_m": "state_x_m",
        "y_m": "state_y_m",
        "z_m": "state_z_m",
    }
    for source, target in aliases.items():
        if target not in rows.columns and source in rows.columns:
            rows[target] = rows[source]
    required = ["time_s", "state_x_m", "state_y_m", "state_z_m"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"initial estimates missing required columns: {missing}")
    for column in required:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[required].to_numpy(float)).all(axis=1)
    return rows.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _validate_config(config: CandidateMixtureMapConfig) -> None:
    if int(config.top_k) <= 0:
        raise ValueError("top_k must be positive")
    if not (0.0 < float(config.sigma_min_m) <= float(config.sigma_max_m)):
        raise ValueError("sigma bounds must satisfy 0 < sigma_min_m <= sigma_max_m")
    if float(config.default_sigma_m) <= 0.0:
        raise ValueError("default_sigma_m must be positive")
    if config.score_normalization not in SCORE_NORMALIZATION_CHOICES:
        raise ValueError(f"unsupported score normalization {config.score_normalization!r}")
    if float(config.temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if config.loss not in LOSS_CHOICES:
        raise ValueError(f"unsupported loss {config.loss!r}")
    if float(config.huber_delta) <= 0.0:
        raise ValueError("huber_delta must be positive")
    if float(config.smoothness_weight) < 0.0:
        raise ValueError("smoothness_weight must be non-negative")
    if int(config.iterations) <= 0:
        raise ValueError("iterations must be positive")
    if float(config.tolerance_m) < 0.0:
        raise ValueError("tolerance_m must be non-negative")
    if not (0.0 <= float(config.uniform_weight_floor) < 1.0):
        raise ValueError("uniform_weight_floor must be in [0, 1)")
    if config.initialization not in INITIALIZATION_CHOICES:
        raise ValueError(f"unsupported initialization {config.initialization!r}")


def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
