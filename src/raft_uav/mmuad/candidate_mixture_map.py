"""Robust branch-balanced candidate-mixture trajectory inference for MMUAD.

The MMUAD experiments show that hard top-1 selection can discard useful raw,
dynamic, or calibrated hypotheses before trajectory smoothing.  This module
keeps a bounded candidate reservoir alive during inference, attaches learned
candidate uncertainty when available, and alternates between robust candidate
responsibilities and an acceleration-regularized trajectory solve.

The implementation is truth-free at inference time.  It consumes only candidate
positions, scores, source/branch labels, and optional predicted uncertainty.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns
from raft_uav.mmuad.submission import (
    load_sequence_class_map,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)

_REQUIRED_COLUMNS = ("sequence_id", "time_s", "x_m", "y_m", "z_m")
_INITIALIZATION_MODES = ("weighted-mean", "best-score")
_MEASUREMENT_WEIGHT_MODES = ("precision", "uniform")


@dataclass(frozen=True)
class CandidateMixtureConfig:
    """Configuration for robust candidate-mixture trajectory inference."""

    top_k: int = 20
    score_column: str = "candidate_reservoir_score"
    fallback_score_columns: tuple[str, ...] = ("ranker_score", "confidence")
    sigma_column: str = "predicted_sigma_m"
    fallback_sigma_m: float = 10.0
    sigma_min_m: float = 1.0
    sigma_max_m: float = 50.0
    temperature: float = 128.0
    smoothness_weight: float = 7200.0
    huber_scale: float = 1.0
    iterations: int = 5
    branch_balance: float = 0.25
    source_balance: float = 0.0
    responsibility_floor: float = 0.01
    initialization: Literal["weighted-mean", "best-score"] = "weighted-mean"
    measurement_weight_mode: Literal["precision", "uniform"] = "precision"
    normalize_measurement_weights: bool = True
    linear_solve_ridge: float = 1.0e-8


@dataclass(frozen=True)
class CandidateMixtureResult:
    """Outputs from candidate-mixture trajectory inference."""

    estimates: pd.DataFrame
    frame_diagnostics: pd.DataFrame
    candidate_assignments: pd.DataFrame
    iteration_summary: pd.DataFrame
    summary: dict[str, Any]


def run_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    config: CandidateMixtureConfig | None = None,
) -> CandidateMixtureResult:
    """Estimate one trajectory per sequence from a soft candidate mixture."""

    config = config or CandidateMixtureConfig()
    _validate_config(config)
    rows = _prepare_candidate_rows(candidates, config=config)
    if rows.empty:
        empty = pd.DataFrame()
        return CandidateMixtureResult(
            estimates=empty,
            frame_diagnostics=empty,
            candidate_assignments=empty,
            iteration_summary=empty,
            summary={
                "config": asdict(config),
                "input_candidate_rows": int(len(candidates)),
                "retained_candidate_rows": 0,
                "sequence_count": 0,
                "estimate_rows": 0,
            },
        )

    estimate_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    assignment_parts: list[pd.DataFrame] = []
    iteration_parts: list[pd.DataFrame] = []
    for sequence_id, sequence_rows in rows.groupby("sequence_id", sort=True):
        estimates, diagnostics, assignments, iterations = _solve_sequence(
            str(sequence_id),
            sequence_rows,
            config=config,
        )
        estimate_parts.append(estimates)
        diagnostic_parts.append(diagnostics)
        assignment_parts.append(assignments)
        iteration_parts.append(iterations)

    estimates = pd.concat(estimate_parts, ignore_index=True)
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True)
    assignments = pd.concat(assignment_parts, ignore_index=True)
    iterations = pd.concat(iteration_parts, ignore_index=True)
    summary = build_candidate_mixture_summary(
        input_candidates=pd.DataFrame(candidates),
        retained_candidates=rows,
        estimates=estimates,
        frame_diagnostics=diagnostics,
        iteration_summary=iterations,
        config=config,
    )
    return CandidateMixtureResult(
        estimates=estimates,
        frame_diagnostics=diagnostics,
        candidate_assignments=assignments,
        iteration_summary=iterations,
        summary=summary,
    )


def compute_candidate_responsibilities(
    candidates: pd.DataFrame,
    state_xyz: np.ndarray,
    *,
    config: CandidateMixtureConfig | None = None,
) -> pd.DataFrame:
    """Return robust responsibilities for one frame, useful for diagnostics/tests."""

    config = config or CandidateMixtureConfig()
    _validate_config(config)
    rows = _prepare_candidate_rows(candidates, config=config)
    if rows.empty:
        return rows.assign(mixture_responsibility=pd.Series(dtype=float))
    if rows[["sequence_id", "time_s"]].drop_duplicates().shape[0] != 1:
        raise ValueError("compute_candidate_responsibilities expects exactly one frame")
    responsibility, residual = _responsibility_arrays(
        rows,
        np.asarray(state_xyz, dtype=float).reshape(3),
        config=config,
    )
    out = rows.copy()
    out["mixture_responsibility"] = responsibility
    out["mixture_residual_m"] = residual
    return out.drop(columns=["_mixture_row_id"], errors="ignore")


def build_candidate_mixture_summary(
    *,
    input_candidates: pd.DataFrame,
    retained_candidates: pd.DataFrame,
    estimates: pd.DataFrame,
    frame_diagnostics: pd.DataFrame,
    iteration_summary: pd.DataFrame,
    config: CandidateMixtureConfig,
) -> dict[str, Any]:
    """Build a compact JSON-serializable inference summary."""

    entropy = pd.to_numeric(
        frame_diagnostics.get("assignment_entropy"),
        errors="coerce",
    )
    effective_count = pd.to_numeric(
        frame_diagnostics.get("effective_candidate_count"),
        errors="coerce",
    )
    sigma = pd.to_numeric(frame_diagnostics.get("effective_sigma_m"), errors="coerce")
    state_change = pd.to_numeric(iteration_summary.get("mean_state_change_m"), errors="coerce")
    return {
        "schema": "raft-uav-mmuad-candidate-mixture-map-v1",
        "config": asdict(config),
        "input_candidate_rows": int(len(input_candidates)),
        "retained_candidate_rows": int(len(retained_candidates)),
        "sequence_count": int(estimates["sequence_id"].nunique()) if not estimates.empty else 0,
        "estimate_rows": int(len(estimates)),
        "assignment_entropy_mean": _safe_mean(entropy),
        "effective_candidate_count_mean": _safe_mean(effective_count),
        "effective_sigma_mean_m": _safe_mean(sigma),
        "final_iteration_mean_state_change_m": _last_iteration_mean(state_change),
        "dominant_branch_counts": _value_counts(frame_diagnostics, "dominant_candidate_branch"),
        "dominant_source_counts": _value_counts(frame_diagnostics, "dominant_candidate_source"),
    }


def _solve_sequence(
    sequence_id: str,
    sequence_rows: pd.DataFrame,
    *,
    config: CandidateMixtureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = sequence_rows.sort_values(["time_s", "_mixture_score"], ascending=[True, False])
    grouped = [group.copy() for _, group in rows.groupby("time_s", sort=True)]
    times = np.asarray([float(group["time_s"].iloc[0]) for group in grouped], dtype=float)
    states = np.vstack([_initial_frame_state(group, config=config) for group in grouped])
    iteration_records: list[dict[str, Any]] = []

    for iteration in range(int(config.iterations)):
        pseudo, effective_sigma, frame_entropy = _mixture_pseudo_measurements(
            grouped,
            states,
            config=config,
        )
        measurement_weights = _measurement_weights(effective_sigma, config=config)
        new_states = _smooth_trajectory(
            times,
            pseudo,
            measurement_weights,
            smoothness_weight=float(config.smoothness_weight),
            ridge=float(config.linear_solve_ridge),
        )
        change = np.linalg.norm(new_states - states, axis=1)
        iteration_records.append(
            {
                "sequence_id": sequence_id,
                "iteration": int(iteration + 1),
                "mean_state_change_m": float(np.mean(change)),
                "max_state_change_m": float(np.max(change)),
                "mean_assignment_entropy": float(np.mean(frame_entropy)),
                "mean_effective_sigma_m": float(np.mean(effective_sigma)),
            }
        )
        states = new_states

    estimates_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    assignment_parts: list[pd.DataFrame] = []
    for frame_index, (group, state) in enumerate(zip(grouped, states, strict=True)):
        responsibility, residual = _responsibility_arrays(group, state, config=config)
        xyz = group[["x_m", "y_m", "z_m"]].to_numpy(float)
        pseudo = np.sum(responsibility[:, None] * xyz, axis=0)
        effective_sigma = _effective_sigma(group, xyz, pseudo, responsibility, config=config)
        entropy = _entropy(responsibility)
        dominant_index = int(np.argmax(responsibility))
        dominant = group.iloc[dominant_index]
        branch_mass = _label_mass(responsibility, group["candidate_branch"])
        source_mass = _label_mass(responsibility, group["source"])
        estimates_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": float(times[frame_index]),
                "state_x_m": float(state[0]),
                "state_y_m": float(state[1]),
                "state_z_m": float(state[2]),
                "x_m": float(state[0]),
                "y_m": float(state[1]),
                "z_m": float(state[2]),
                "score": float(np.max(responsibility)),
                "mixture_effective_sigma_m": float(effective_sigma),
                "mixture_effective_candidate_count": float(np.exp(entropy)),
            }
        )
        diagnostic_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": float(times[frame_index]),
                "candidate_count": int(len(group)),
                "pseudo_x_m": float(pseudo[0]),
                "pseudo_y_m": float(pseudo[1]),
                "pseudo_z_m": float(pseudo[2]),
                "state_x_m": float(state[0]),
                "state_y_m": float(state[1]),
                "state_z_m": float(state[2]),
                "effective_sigma_m": float(effective_sigma),
                "assignment_entropy": float(entropy),
                "effective_candidate_count": float(np.exp(entropy)),
                "dominant_responsibility": float(responsibility[dominant_index]),
                "dominant_candidate_branch": str(dominant["candidate_branch"]),
                "dominant_candidate_source": str(dominant["source"]),
                "dominant_candidate_track_id": str(dominant.get("track_id", "")),
                "branch_mass_json": json.dumps(branch_mass, sort_keys=True),
                "source_mass_json": json.dumps(source_mass, sort_keys=True),
            }
        )
        assignment = group.copy()
        assignment["mixture_responsibility"] = responsibility
        assignment["mixture_residual_m"] = residual
        assignment["mixture_frame_index"] = int(frame_index)
        assignment_parts.append(assignment)

    estimates = pd.DataFrame.from_records(estimates_records)
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    assignments = pd.concat(assignment_parts, ignore_index=True).drop(
        columns=["_mixture_row_id"],
        errors="ignore",
    )
    iterations = pd.DataFrame.from_records(iteration_records)
    return estimates, diagnostics, assignments, iterations


def _prepare_candidate_rows(
    candidates: pd.DataFrame,
    *,
    config: CandidateMixtureConfig,
) -> pd.DataFrame:
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    missing = set(_REQUIRED_COLUMNS).difference(rows.columns)
    if missing:
        raise ValueError(f"candidate mixture rows missing columns: {sorted(missing)}")
    if rows.empty:
        return rows
    rows = rows.copy().reset_index(drop=True)
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    rows = rows.loc[finite].copy().reset_index(drop=True)
    if rows.empty:
        return rows
    rows["source"] = rows.get("source", pd.Series("unknown", index=rows.index))
    rows["source"] = rows["source"].fillna("unknown").astype(str)
    rows["candidate_branch"] = rows.get(
        "candidate_branch",
        rows["source"],
    )
    rows["candidate_branch"] = rows["candidate_branch"].fillna(rows["source"]).astype(str)
    rows["_mixture_score"] = _candidate_scores(rows, config=config)
    rows["_mixture_sigma_m"] = _candidate_sigmas(rows, config=config)
    rows["_mixture_row_id"] = np.arange(len(rows), dtype=int)
    if int(config.top_k) > 0:
        rows = (
            rows.sort_values(
                ["sequence_id", "time_s", "_mixture_score"],
                ascending=[True, True, False],
            )
            .groupby(["sequence_id", "time_s"], sort=False, as_index=False)
            .head(int(config.top_k))
            .reset_index(drop=True)
        )
    return rows


def _candidate_scores(rows: pd.DataFrame, *, config: CandidateMixtureConfig) -> pd.Series:
    score = pd.to_numeric(rows.get(config.score_column), errors="coerce")
    if not isinstance(score, pd.Series):
        score = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in config.fallback_score_columns:
        fallback = pd.to_numeric(rows.get(column), errors="coerce")
        if isinstance(fallback, pd.Series):
            score = score.fillna(fallback)
    return score.fillna(0.0).astype(float)


def _candidate_sigmas(rows: pd.DataFrame, *, config: CandidateMixtureConfig) -> pd.Series:
    sigma = pd.to_numeric(rows.get(config.sigma_column), errors="coerce")
    if not isinstance(sigma, pd.Series):
        sigma = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in ("std_xy_m", "std_z_m"):
        fallback = pd.to_numeric(rows.get(column), errors="coerce")
        if isinstance(fallback, pd.Series):
            sigma = sigma.fillna(fallback)
    sigma = sigma.fillna(float(config.fallback_sigma_m)).astype(float)
    return sigma.clip(lower=float(config.sigma_min_m), upper=float(config.sigma_max_m))


def _initial_frame_state(group: pd.DataFrame, *, config: CandidateMixtureConfig) -> np.ndarray:
    xyz = group[["x_m", "y_m", "z_m"]].to_numpy(float)
    if config.initialization == "best-score":
        return xyz[int(np.argmax(group["_mixture_score"].to_numpy(float)))].copy()
    sigma = group["_mixture_sigma_m"].to_numpy(float)
    prior = _score_prior(group["_mixture_score"].to_numpy(float), config=config)
    weight = prior / np.maximum(sigma**2, 1.0e-12)
    weight = _normalize(weight)
    return np.sum(weight[:, None] * xyz, axis=0)


def _mixture_pseudo_measurements(
    grouped: list[pd.DataFrame],
    states: np.ndarray,
    *,
    config: CandidateMixtureConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pseudo_rows: list[np.ndarray] = []
    sigma_rows: list[float] = []
    entropy_rows: list[float] = []
    for group, state in zip(grouped, states, strict=True):
        responsibility, _ = _responsibility_arrays(group, state, config=config)
        xyz = group[["x_m", "y_m", "z_m"]].to_numpy(float)
        pseudo = np.sum(responsibility[:, None] * xyz, axis=0)
        pseudo_rows.append(pseudo)
        sigma_rows.append(_effective_sigma(group, xyz, pseudo, responsibility, config=config))
        entropy_rows.append(_entropy(responsibility))
    return (
        np.vstack(pseudo_rows),
        np.asarray(sigma_rows, dtype=float),
        np.asarray(entropy_rows, dtype=float),
    )


def _responsibility_arrays(
    group: pd.DataFrame,
    state_xyz: np.ndarray,
    *,
    config: CandidateMixtureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = group[["x_m", "y_m", "z_m"]].to_numpy(float)
    sigma = group["_mixture_sigma_m"].to_numpy(float)
    residual = np.linalg.norm(xyz - np.asarray(state_xyz, dtype=float).reshape(1, 3), axis=1)
    normalized_residual = residual / np.maximum(sigma, 1.0e-12)
    robust_weight = _huber_irls_weight(normalized_residual, scale=float(config.huber_scale))
    prior = _score_prior(group["_mixture_score"].to_numpy(float), config=config)
    raw = prior * robust_weight / np.maximum(sigma**2, 1.0e-12)
    global_distribution = _normalize(raw)
    distribution = global_distribution
    if float(config.branch_balance) > 0.0:
        branch_distribution = _balanced_label_distribution(
            raw,
            group["candidate_branch"].astype(str).to_numpy(),
        )
        distribution = (
            (1.0 - float(config.branch_balance)) * distribution
            + float(config.branch_balance) * branch_distribution
        )
    if float(config.source_balance) > 0.0:
        source_distribution = _balanced_label_distribution(
            raw,
            group["source"].astype(str).to_numpy(),
        )
        distribution = (
            (1.0 - float(config.source_balance)) * distribution
            + float(config.source_balance) * source_distribution
        )
    if float(config.responsibility_floor) > 0.0:
        uniform = np.full(len(distribution), 1.0 / len(distribution), dtype=float)
        distribution = (
            (1.0 - float(config.responsibility_floor)) * distribution
            + float(config.responsibility_floor) * uniform
        )
    return _normalize(distribution), residual


def _score_prior(scores: np.ndarray, *, config: CandidateMixtureConfig) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).any():
        return np.full(len(scores), 1.0 / max(len(scores), 1), dtype=float)
    scores = np.nan_to_num(scores, nan=float(np.nanmin(scores[np.isfinite(scores)])))
    temperature = max(float(config.temperature), 1.0e-9)
    logits = np.clip((scores - float(np.max(scores))) / temperature, -60.0, 0.0)
    return _normalize(np.exp(logits))


def _balanced_label_distribution(raw: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=str)
    unique = sorted(set(labels.tolist()))
    if not unique:
        return _normalize(raw)
    out = np.zeros(len(raw), dtype=float)
    label_mass = 1.0 / len(unique)
    for label in unique:
        mask = labels == label
        within = _normalize(np.asarray(raw, dtype=float)[mask])
        out[mask] = label_mass * within
    return _normalize(out)


def _huber_irls_weight(residual: np.ndarray, *, scale: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=float)
    if scale <= 0.0:
        return np.ones_like(residual)
    absolute = np.abs(residual)
    return np.where(absolute <= scale, 1.0, scale / np.maximum(absolute, 1.0e-12))


def _effective_sigma(
    group: pd.DataFrame,
    xyz: np.ndarray,
    pseudo: np.ndarray,
    responsibility: np.ndarray,
    *,
    config: CandidateMixtureConfig,
) -> float:
    sigma = group["_mixture_sigma_m"].to_numpy(float)
    measurement_variance = float(np.sum(responsibility * sigma**2))
    spatial_variance = float(
        np.sum(responsibility * np.sum((xyz - pseudo.reshape(1, 3)) ** 2, axis=1)) / 3.0
    )
    value = np.sqrt(max(measurement_variance + spatial_variance, config.sigma_min_m**2))
    return float(np.clip(value, config.sigma_min_m, config.sigma_max_m))


def _measurement_weights(
    effective_sigma: np.ndarray,
    *,
    config: CandidateMixtureConfig,
) -> np.ndarray:
    if config.measurement_weight_mode == "uniform":
        return np.ones(len(effective_sigma), dtype=float)
    weights = 1.0 / np.maximum(np.asarray(effective_sigma, dtype=float) ** 2, 1.0e-12)
    if config.normalize_measurement_weights:
        finite = weights[np.isfinite(weights) & (weights > 0.0)]
        if len(finite):
            weights = weights / float(np.median(finite))
    return np.clip(weights, 1.0e-6, 1.0e6)


def _smooth_trajectory(
    times: np.ndarray,
    pseudo: np.ndarray,
    measurement_weights: np.ndarray,
    *,
    smoothness_weight: float,
    ridge: float,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    pseudo = np.asarray(pseudo, dtype=float)
    weights = np.asarray(measurement_weights, dtype=float)
    count = len(times)
    if count < 3 or smoothness_weight <= 0.0:
        return pseudo.copy()
    difference = _acceleration_difference_matrix(times)
    system = np.diag(weights) + float(smoothness_weight) * (difference.T @ difference)
    system += float(max(ridge, 0.0)) * np.eye(count)
    target = weights[:, None] * pseudo
    try:
        return np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, target, rcond=None)[0]


def _acceleration_difference_matrix(times: np.ndarray) -> np.ndarray:
    count = len(times)
    matrix = np.zeros((max(count - 2, 0), count), dtype=float)
    for index in range(1, count - 1):
        dt_before = max(float(times[index] - times[index - 1]), 1.0e-6)
        dt_after = max(float(times[index + 1] - times[index]), 1.0e-6)
        span = dt_before + dt_after
        row = index - 1
        matrix[row, index - 1] = 2.0 / (dt_before * span)
        matrix[row, index] = -2.0 * (1.0 / dt_before + 1.0 / dt_after) / span
        matrix[row, index + 1] = 2.0 / (dt_after * span)
    return matrix


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    total = float(np.sum(values))
    if total <= 0.0:
        return np.full(len(values), 1.0 / max(len(values), 1), dtype=float)
    return values / total


def _entropy(values: np.ndarray) -> float:
    probability = np.clip(_normalize(values), 1.0e-15, 1.0)
    return float(-np.sum(probability * np.log(probability)))


def _label_mass(values: np.ndarray, labels: pd.Series) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, value in zip(labels.astype(str), values, strict=True):
        out[str(label)] = out.get(str(label), 0.0) + float(value)
    return out


def _validate_config(config: CandidateMixtureConfig) -> None:
    if config.top_k < 0:
        raise ValueError("top_k must be non-negative; use 0 for all candidates")
    if config.temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.sigma_min_m <= 0.0 or config.sigma_max_m < config.sigma_min_m:
        raise ValueError("sigma bounds must satisfy 0 < sigma_min_m <= sigma_max_m")
    for name, value in (
        ("branch_balance", config.branch_balance),
        ("source_balance", config.source_balance),
        ("responsibility_floor", config.responsibility_floor),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be within [0, 1]")
    if config.initialization not in _INITIALIZATION_MODES:
        raise ValueError(f"unsupported initialization={config.initialization!r}")
    if config.measurement_weight_mode not in _MEASUREMENT_WEIGHT_MODES:
        raise ValueError(
            f"unsupported measurement_weight_mode={config.measurement_weight_mode!r}"
        )


def _safe_mean(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else None


def _last_iteration_mean(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite)]
    return float(finite.iloc[-1]) if len(finite) else None


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in rows[column].fillna("unknown").astype(str).value_counts().items()
    }


def _write_frame(frame: pd.DataFrame, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-mixture-map",
        description="run robust branch-balanced candidate-mixture MMUAD smoothing",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-estimates-csv", type=Path, required=True)
    parser.add_argument("--frame-diagnostics-csv", type=Path)
    parser.add_argument("--candidate-assignments-csv", type=Path)
    parser.add_argument("--iteration-summary-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--score-column", default="candidate_reservoir_score")
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--fallback-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=50.0)
    parser.add_argument("--temperature", type=float, default=128.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--huber-scale", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--branch-balance", type=float, default=0.25)
    parser.add_argument("--source-balance", type=float, default=0.0)
    parser.add_argument("--responsibility-floor", type=float, default=0.01)
    parser.add_argument("--initialization", choices=_INITIALIZATION_MODES, default="weighted-mean")
    parser.add_argument(
        "--measurement-weight-mode",
        choices=_MEASUREMENT_WEIGHT_MODES,
        default="precision",
    )
    parser.add_argument("--no-normalize-measurement-weights", action="store_true")
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="2")
    parser.add_argument("--official-results-csv", type=Path)
    parser.add_argument("--official-zip", type=Path)
    args = parser.parse_args(argv)

    config = CandidateMixtureConfig(
        top_k=args.top_k,
        score_column=args.score_column,
        sigma_column=args.sigma_column,
        fallback_sigma_m=args.fallback_sigma_m,
        sigma_min_m=args.sigma_min_m,
        sigma_max_m=args.sigma_max_m,
        temperature=args.temperature,
        smoothness_weight=args.smoothness_weight,
        huber_scale=args.huber_scale,
        iterations=args.iterations,
        branch_balance=args.branch_balance,
        source_balance=args.source_balance,
        responsibility_floor=args.responsibility_floor,
        initialization=args.initialization,
        measurement_weight_mode=args.measurement_weight_mode,
        normalize_measurement_weights=not args.no_normalize_measurement_weights,
    )
    candidates = load_candidate_file(args.candidates_csv).rows
    result = run_candidate_mixture_map(candidates, config=config)
    estimates = result.estimates.copy()
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    estimates["classification"] = [
        class_map.get(str(sequence_id), str(args.default_classification))
        for sequence_id in estimates.get("sequence_id", pd.Series(dtype=str))
    ]

    _write_frame(estimates, args.output_estimates_csv)
    _write_frame(result.frame_diagnostics, args.frame_diagnostics_csv)
    _write_frame(result.candidate_assignments, args.candidate_assignments_csv)
    _write_frame(result.iteration_summary, args.iteration_summary_csv)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    if args.official_results_csv is not None:
        write_official_mmaud_results_csv(
            estimates,
            args.official_results_csv,
            classification=args.default_classification,
            class_map=class_map,
        )
    if args.official_zip is not None:
        write_official_ug2_codabench_zip(
            estimates,
            args.official_zip,
            classification=args.default_classification,
            class_map=class_map,
        )

    print("mmuad_candidate_mixture_map=ok")
    print(f"output_estimates_csv={args.output_estimates_csv}")
    print(f"estimate_rows={len(estimates)}")
    print(f"sequence_count={estimates['sequence_id'].nunique() if not estimates.empty else 0}")
    if args.official_results_csv is not None:
        print(f"official_results_csv={args.official_results_csv}")
    if args.official_zip is not None:
        print(f"official_zip={args.official_zip}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
