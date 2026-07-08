"""Template-grid trajectory regularization for MMUAD Track 5 submissions.

This module is a final-stage, inference-safe post-processor for Codabench/UG2+
Track 5 pose trajectories.  It resamples estimates onto the official template,
then applies an acceleration-regularized robust smoother per sequence before
writing an upload-ready ``mmaud_results.csv`` and ZIP.  The smoother uses only the
predicted trajectory and requested Sequence/Timestamp rows; truth is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    load_sequence_class_map,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_template_resample import RESAMPLE_METHODS
from raft_uav.mmuad.track5_template_resample import ResampleMethod
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template
from raft_uav.mmuad.track5_template_resample import summarize_template_resample_diagnostics

REGULARIZED_ESTIMATES_CSV = "mmuad_track5_regularized_estimates.csv"
REGULARIZER_DIAGNOSTICS_CSV = "mmuad_track5_regularizer_diagnostics.csv"
REGULARIZER_RESAMPLE_SUMMARY_CSV = "mmuad_track5_regularizer_resample_summary.csv"
REGULARIZER_MANIFEST_JSON = "mmuad_track5_regularizer_manifest.json"
VALIDATION_JSON = "mmuad_track5_regularizer_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_regularizer_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"


def regularize_track5_estimates(
    estimates: pd.DataFrame,
    *,
    smoothness_weight: float = 10.0,
    huber_delta_m: float = 25.0,
    iterations: int = 5,
    observation_sigma_m: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return acceleration-smoothed Track 5 estimates and diagnostics."""

    if smoothness_weight < 0.0:
        raise ValueError("smoothness_weight must be non-negative")
    if huber_delta_m <= 0.0:
        raise ValueError("huber_delta_m must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if observation_sigma_m <= 0.0:
        raise ValueError("observation_sigma_m must be positive")

    rows = _normalize_estimate_rows(estimates)
    if rows.empty:
        return rows, pd.DataFrame()
    output_parts: list[pd.DataFrame] = []
    diagnostic_records: list[dict[str, Any]] = []
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        sequence = group.sort_values("time_s").reset_index(drop=True)
        times = sequence["time_s"].to_numpy(float)
        observed = sequence[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
        finite = np.isfinite(observed).all(axis=1) & np.isfinite(times)
        if finite.sum() == 0:
            smoothed = observed.copy()
            residual = np.full(len(sequence), np.nan)
            robust_weights = np.zeros(len(sequence), dtype=float)
        elif finite.sum() < 3 or smoothness_weight == 0.0:
            smoothed = observed.copy()
            residual = np.linalg.norm(smoothed - observed, axis=1)
            robust_weights = finite.astype(float)
        else:
            smoothed, residual, robust_weights = _robust_smooth_sequence(
                times,
                observed,
                finite=finite,
                smoothness_weight=float(smoothness_weight),
                huber_delta_m=float(huber_delta_m),
                iterations=int(iterations),
                observation_sigma_m=float(observation_sigma_m),
            )
        out = sequence.copy()
        out["regularizer_input_x_m"] = out["state_x_m"]
        out["regularizer_input_y_m"] = out["state_y_m"]
        out["regularizer_input_z_m"] = out["state_z_m"]
        out["state_x_m"] = smoothed[:, 0]
        out["state_y_m"] = smoothed[:, 1]
        out["state_z_m"] = smoothed[:, 2]
        out["track5_regularized"] = True
        out["regularizer_residual_m"] = residual
        out["regularizer_robust_weight"] = robust_weights
        output_parts.append(out)
        diagnostic_records.append(
            {
                "sequence_id": str(sequence_id),
                "row_count": int(len(sequence)),
                "finite_input_count": int(finite.sum()),
                "smoothness_weight": float(smoothness_weight),
                "huber_delta_m": float(huber_delta_m),
                "iterations": int(iterations),
                "observation_sigma_m": float(observation_sigma_m),
                "mean_regularizer_residual_m": _safe_mean(pd.Series(residual)),
                "p95_regularizer_residual_m": _safe_percentile(pd.Series(residual), 95),
                "mean_robust_weight": _safe_mean(pd.Series(robust_weights)),
            }
        )
    return (
        pd.concat(output_parts, ignore_index=True, sort=False),
        pd.DataFrame.from_records(diagnostic_records),
    )


def run_track5_trajectory_regularizer(
    *,
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
    resample_method: ResampleMethod = "linear",
    max_interpolation_gap_s: float | None = None,
    smoothness_weight: float = 10.0,
    huber_delta_m: float = 25.0,
    iterations: int = 5,
    observation_sigma_m: float = 10.0,
) -> dict[str, Path]:
    """Resample, regularize, package, and validate Track 5 estimates."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resampled, resample_diagnostics = resample_estimates_to_track5_template(
        estimates,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        resample_method=resample_method,
        max_interpolation_gap_s=max_interpolation_gap_s,
    )
    regularized, regularizer_diagnostics = regularize_track5_estimates(
        resampled,
        smoothness_weight=smoothness_weight,
        huber_delta_m=huber_delta_m,
        iterations=iterations,
        observation_sigma_m=observation_sigma_m,
    )
    paths = {
        "regularized_estimates_csv": output / REGULARIZED_ESTIMATES_CSV,
        "diagnostics_csv": output / REGULARIZER_DIAGNOSTICS_CSV,
        "resample_summary_csv": output / REGULARIZER_RESAMPLE_SUMMARY_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / VALIDATION_JSON,
        "validation_rows_csv": output / VALIDATION_ROWS_CSV,
        "manifest_json": output / REGULARIZER_MANIFEST_JSON,
    }
    regularized.to_csv(paths["regularized_estimates_csv"], index=False)
    regularizer_diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    summarize_template_resample_diagnostics(resample_diagnostics).to_csv(
        paths["resample_summary_csv"],
        index=False,
    )
    class_map = class_map or {}
    write_official_mmaud_results_csv(
        regularized,
        paths["official_results_csv"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        regularized,
        paths["official_zip"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    validation = validate_official_track5_submission(
        paths["official_zip"],
        template=template,
        require_zip=True,
    )
    paths["validation_json"].write_text(
        json.dumps(_jsonable(validation.summary), indent=2),
        encoding="utf-8",
    )
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    manifest = {
        "schema": "raft-uav-mmuad-track5-trajectory-regularizer-v1",
        "input_estimate_rows": int(len(estimates)),
        "template_rows": int(len(template)),
        "regularized_rows": int(len(regularized)),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "resample_method": resample_method,
        "max_interpolation_gap_s": max_interpolation_gap_s,
        "smoothness_weight": float(smoothness_weight),
        "huber_delta_m": float(huber_delta_m),
        "iterations": int(iterations),
        "observation_sigma_m": float(observation_sigma_m),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-trajectory-regularizer",
        description="regularize a Track 5 estimate trajectory and write an upload-ready ZIP",
    )
    parser.add_argument("--estimates-csv", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--resample-method", choices=RESAMPLE_METHODS, default="linear")
    parser.add_argument("--max-interpolation-gap-s", type=float)
    parser.add_argument("--smoothness-weight", type=float, default=10.0)
    parser.add_argument("--huber-delta-m", type=float, default=25.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--observation-sigma-m", type=float, default=10.0)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    estimates = read_estimate_csv(args.estimates_csv)
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = run_track5_trajectory_regularizer(
        estimates=estimates,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        resample_method=args.resample_method,
        max_interpolation_gap_s=args.max_interpolation_gap_s,
        smoothness_weight=float(args.smoothness_weight),
        huber_delta_m=float(args.huber_delta_m),
        iterations=int(args.iterations),
        observation_sigma_m=float(args.observation_sigma_m),
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_trajectory_regularizer=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"regularized upload is not leaderboard-ready: {reasons}")
    return 0


def _robust_smooth_sequence(
    times: np.ndarray,
    observed: np.ndarray,
    *,
    finite: np.ndarray,
    smoothness_weight: float,
    huber_delta_m: float,
    iterations: int,
    observation_sigma_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = observed.copy()
    for axis in range(3):
        state[:, axis] = _fill_axis(times, observed[:, axis], finite)
    robust_weights = finite.astype(float)
    base_precision = 1.0 / float(observation_sigma_m) ** 2
    for _ in range(iterations):
        residual = np.linalg.norm(state - observed, axis=1)
        robust_weights = np.where(
            finite,
            np.minimum(1.0, float(huber_delta_m) / np.maximum(residual, 1.0e-12)),
            0.0,
        )
        precision = base_precision * robust_weights
        state = _solve_smooth_trajectory(times, observed, precision, smoothness_weight)
    residual = np.linalg.norm(state - observed, axis=1)
    return state, residual, robust_weights


def _solve_smooth_trajectory(
    times: np.ndarray,
    observed: np.ndarray,
    precision: np.ndarray,
    smoothness_weight: float,
) -> np.ndarray:
    count = len(times)
    if count == 0:
        return np.empty((0, 3), dtype=float)
    lhs = np.diag(np.asarray(precision, dtype=float))
    second = _second_derivative_matrix(times)
    if second.size and smoothness_weight > 0.0:
        lhs = lhs + float(smoothness_weight) * (second.T @ second)
    # Gentle ridge keeps sequences with sparse valid rows solvable.
    lhs = lhs + np.eye(count) * 1.0e-9
    rhs = precision[:, None] * observed
    filled = observed.copy()
    for axis in range(3):
        filled[:, axis] = _fill_axis(times, observed[:, axis], np.isfinite(observed[:, axis]))
    rhs = np.where(np.isfinite(rhs), rhs, 0.0)
    solved = np.empty_like(filled, dtype=float)
    for axis in range(3):
        try:
            solved[:, axis] = np.linalg.solve(lhs, rhs[:, axis])
        except np.linalg.LinAlgError:
            solved[:, axis] = filled[:, axis]
    return solved


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


def _fill_axis(times: np.ndarray, values: np.ndarray, finite: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.asarray(finite, dtype=bool) & np.isfinite(values)
    if finite.sum() == 0:
        return np.zeros_like(values, dtype=float)
    if finite.sum() == 1:
        return np.full_like(values, float(values[finite][0]), dtype=float)
    return np.interp(np.asarray(times, dtype=float), times[finite], values[finite])


def _normalize_estimate_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(estimates).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"])
    if "sequence_id" not in rows.columns or "time_s" not in rows.columns:
        raise ValueError("estimates must contain sequence_id and time_s columns")
    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows = rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)
    return rows


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(numeric.mean())


def _safe_percentile(values: pd.Series, percentile: float) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.percentile(numeric.to_numpy(float), percentile))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
