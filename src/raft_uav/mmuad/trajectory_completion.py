"""MMUAD trajectory smoothing and short-gap completion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import TruthFrame, normalize_truth_columns
from raft_uav.mmuad.tracker import add_truth_errors, compute_metrics


TrajectoryCompletionMode = Literal[
    "none",
    "gap-interpolation",
    "fixed-lag",
    "constant-velocity",
    "constant-acceleration",
]

_ABLATION_MODES: tuple[TrajectoryCompletionMode, ...] = (
    "raw",
    "gap-interpolation",
    "fixed-lag",
    "constant-velocity",
    "constant-acceleration",
)  # type: ignore[assignment]


@dataclass(frozen=True)
class TrajectoryCompletionConfig:
    """Configuration for tracker-output trajectory completion."""

    mode: TrajectoryCompletionMode = "none"
    max_gap_s: float = 1.0
    fixed_lag_s: float = 1.0
    smoothing_blend: float = 1.0
    include_truth_timestamps: bool = True
    infer_missing_grid: bool = True


@dataclass(frozen=True)
class TrajectoryCompletionResult:
    """Completed estimates plus diagnostics."""

    estimates: pd.DataFrame
    gap_summary: pd.DataFrame
    smoothing_ablation: pd.DataFrame
    sequence_error_summary: pd.DataFrame


def complete_and_smooth_estimates(
    estimates: pd.DataFrame,
    truth: TruthFrame | pd.DataFrame | None = None,
    *,
    config: TrajectoryCompletionConfig | None = None,
) -> TrajectoryCompletionResult:
    """Fill short gaps and smooth per-sequence MMUAD tracker estimates."""

    config = config or TrajectoryCompletionConfig()
    rows = _estimate_rows(estimates)
    truth_rows = _truth_rows(truth)
    if rows.empty:
        empty = pd.DataFrame()
        return TrajectoryCompletionResult(empty, empty, empty, empty)

    final_groups: list[pd.DataFrame] = []
    gap_rows: list[dict[str, Any]] = []
    ablation_frames: dict[str, list[pd.DataFrame]] = {mode: [] for mode in _ABLATION_MODES}
    for sequence_id, trajectory_id, group in _trajectory_groups(rows):
        sequence_truth = _truth_for_sequence(truth_rows, sequence_id)
        target_times = _target_times(group, sequence_truth, config=config)
        gap_rows.extend(
            _gap_summary_rows(
                group,
                target_times,
                sequence_id=sequence_id,
                trajectory_id=trajectory_id,
                config=config,
            )
        )
        for mode in _ABLATION_MODES:
            ablation_frames[mode].append(
                _complete_group(
                    group,
                    target_times if mode != "raw" else _unique_times(group),
                    mode=mode,
                    config=config,
                    sequence_id=sequence_id,
                    trajectory_id=trajectory_id,
                )
            )
        final_mode = config.mode if config.mode != "none" else "raw"
        final_groups.append(
            _complete_group(
                group,
                target_times if final_mode != "raw" else _unique_times(group),
                mode=final_mode,
                config=config,
                sequence_id=sequence_id,
                trajectory_id=trajectory_id,
            )
        )

    final_estimates = _concat_frames(final_groups)
    final_estimates = _attach_truth_errors_by_sequence(final_estimates, truth_rows)
    ablation_rows = [
        row
        for mode, frames in ablation_frames.items()
        for row in _ablation_rows(
            _attach_truth_errors_by_sequence(_concat_frames(frames), truth_rows),
            truth_rows,
            mode=mode,
        )
    ]
    return TrajectoryCompletionResult(
        estimates=final_estimates.sort_values(_sort_columns(final_estimates)).reset_index(drop=True),
        gap_summary=pd.DataFrame.from_records(gap_rows),
        smoothing_ablation=pd.DataFrame.from_records(ablation_rows),
        sequence_error_summary=_sequence_error_summary(
            raw=_attach_truth_errors_by_sequence(_concat_frames(ablation_frames["raw"]), truth_rows),
            final=final_estimates,
            truth_rows=truth_rows,
            mode=config.mode,
        ),
    )


def write_trajectory_completion_diagnostics(
    result: TrajectoryCompletionResult,
    output_dir,
) -> dict[str, str]:
    """Write requested MMUAD trajectory-completion diagnostics."""

    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "gap_summary_csv": output_dir / "mmuad_gap_summary.csv",
        "smoothing_ablation_csv": output_dir / "mmuad_smoothing_ablation.csv",
        "sequence_error_summary_csv": output_dir / "mmuad_sequence_error_summary.csv",
    }
    result.gap_summary.to_csv(paths["gap_summary_csv"], index=False)
    result.smoothing_ablation.to_csv(paths["smoothing_ablation_csv"], index=False)
    result.sequence_error_summary.to_csv(paths["sequence_error_summary_csv"], index=False)
    return {key: str(value) for key, value in paths.items()}


def _estimate_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(estimates).copy()
    if rows.empty:
        return rows
    if "sequence_id" not in rows.columns:
        rows["sequence_id"] = "default"
    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m"):
        if column not in rows.columns:
            raise ValueError(f"estimate rows missing required column {column!r}")
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)).all(axis=1)
    rows = rows.loc[finite].copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    if "selected_path_update" in rows.columns:
        rows["selected_path_update"] = rows["selected_path_update"].astype(bool)
    else:
        rows["selected_path_update"] = True
    return rows.sort_values(_sort_columns(rows)).reset_index(drop=True)


def _truth_rows(truth: TruthFrame | pd.DataFrame | None) -> pd.DataFrame | None:
    if truth is None:
        return None
    rows = truth.rows.copy() if isinstance(truth, TruthFrame) else normalize_truth_columns(pd.DataFrame(truth))
    if rows.empty:
        return rows
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _trajectory_groups(rows: pd.DataFrame):
    if "output_track_id" in rows.columns:
        group_columns = ["sequence_id", "output_track_id"]
        work = rows.copy()
        work["output_track_id"] = work["output_track_id"].fillna("single").astype(str)
    else:
        group_columns = ["sequence_id", "_trajectory_id"]
        work = rows.copy()
        work["_trajectory_id"] = "single"
    for keys, group in work.groupby(group_columns, sort=True):
        sequence_id, trajectory_id = keys
        yield str(sequence_id), str(trajectory_id), group.drop(columns=["_trajectory_id"], errors="ignore")


def _target_times(
    group: pd.DataFrame,
    truth_rows: pd.DataFrame | None,
    *,
    config: TrajectoryCompletionConfig,
) -> np.ndarray:
    original = _unique_times(group)
    targets = set(float(value) for value in original)
    if config.include_truth_timestamps and truth_rows is not None and not truth_rows.empty:
        for timestamp in pd.to_numeric(truth_rows["time_s"], errors="coerce").to_numpy(float):
            if np.isfinite(timestamp) and _time_supported_by_short_gap(timestamp, original, config.max_gap_s):
                targets.add(float(timestamp))
    elif config.infer_missing_grid:
        step = _typical_step_s(original)
        if np.isfinite(step) and step > 0.0:
            for left, right in zip(original[:-1], original[1:], strict=False):
                gap_s = float(right - left)
                if gap_s <= max(float(config.max_gap_s), step) and gap_s > 1.5 * step:
                    count = int(np.floor(gap_s / step))
                    for index in range(1, count):
                        value = float(left + index * step)
                        if value < right - 1.0e-9:
                            targets.add(value)
    return np.asarray(sorted(targets), dtype=float)


def _time_supported_by_short_gap(timestamp: float, times: np.ndarray, max_gap_s: float) -> bool:
    if times.size == 0:
        return False
    if np.min(np.abs(times - timestamp)) <= 1.0e-9:
        return True
    insert = int(np.searchsorted(times, timestamp))
    if insert <= 0 or insert >= len(times):
        return False
    return float(times[insert] - times[insert - 1]) <= float(max_gap_s)


def _complete_group(
    group: pd.DataFrame,
    target_times: np.ndarray,
    *,
    mode: str,
    config: TrajectoryCompletionConfig,
    sequence_id: str,
    trajectory_id: str,
) -> pd.DataFrame:
    source = _dedupe_by_time(group)
    source_times = source["time_s"].to_numpy(float)
    source_xyz = source[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    if source.empty or target_times.size == 0:
        return source.iloc[0:0].copy()
    base_xyz, methods, source_indices = _interpolate_base_positions(
        source_times,
        source_xyz,
        target_times,
        max_gap_s=config.max_gap_s,
    )
    valid = np.isfinite(base_xyz).all(axis=1)
    if not valid.any():
        return source.iloc[0:0].copy()
    target_times = target_times[valid]
    base_xyz = base_xyz[valid]
    methods = [method for method, keep in zip(methods, valid, strict=False) if keep]
    source_indices = source_indices[valid]
    smoothed = _smoothed_positions(
        source,
        target_times,
        base_xyz,
        mode=mode,
        config=config,
    )
    velocities = _finite_difference_velocities(target_times, smoothed)
    records = []
    for row_idx, timestamp in enumerate(target_times):
        source_row = source.iloc[int(source_indices[row_idx])].to_dict()
        source_row.update(
            {
                "sequence_id": sequence_id,
                "time_s": float(timestamp),
                "state_x_m": float(smoothed[row_idx, 0]),
                "state_y_m": float(smoothed[row_idx, 1]),
                "state_z_m": float(smoothed[row_idx, 2]),
                "v_x_mps": float(velocities[row_idx, 0]),
                "v_y_mps": float(velocities[row_idx, 1]),
                "v_z_mps": float(velocities[row_idx, 2]),
                "trajectory_completion_mode": mode,
                "trajectory_completion_method": _method_for_mode(methods[row_idx], mode),
                "trajectory_completion_filled": bool(methods[row_idx] != "exact"),
            }
        )
        if "output_track_id" in group.columns:
            source_row["output_track_id"] = trajectory_id
        records.append(source_row)
    return pd.DataFrame.from_records(records)


def _dedupe_by_time(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("time_s").copy()
    if not group["time_s"].duplicated().any():
        return group.reset_index(drop=True)
    rows = []
    for _, same_time in group.groupby("time_s", sort=True):
        row = same_time.iloc[-1].copy()
        for column in ("state_x_m", "state_y_m", "state_z_m", "v_x_mps", "v_y_mps", "v_z_mps"):
            if column in same_time.columns:
                row[column] = pd.to_numeric(same_time[column], errors="coerce").median()
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _interpolate_base_positions(
    times: np.ndarray,
    xyz: np.ndarray,
    target_times: np.ndarray,
    *,
    max_gap_s: float,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    out = np.full((len(target_times), 3), np.nan, dtype=float)
    methods: list[str] = []
    source_indices = np.zeros(len(target_times), dtype=int)
    for idx, timestamp in enumerate(target_times):
        exact = np.flatnonzero(np.abs(times - timestamp) <= 1.0e-9)
        if exact.size:
            source_idx = int(exact[-1])
            out[idx] = xyz[source_idx]
            methods.append("exact")
            source_indices[idx] = source_idx
            continue
        insert = int(np.searchsorted(times, timestamp))
        if insert <= 0 or insert >= len(times):
            methods.append("unsupported")
            source_indices[idx] = int(np.clip(insert, 0, len(times) - 1))
            continue
        left_idx = insert - 1
        right_idx = insert
        gap_s = float(times[right_idx] - times[left_idx])
        if gap_s > float(max_gap_s):
            methods.append("unsupported")
            source_indices[idx] = left_idx
            continue
        alpha = float((timestamp - times[left_idx]) / max(gap_s, 1.0e-9))
        out[idx] = xyz[left_idx] + alpha * (xyz[right_idx] - xyz[left_idx])
        methods.append("interpolated_short_gap")
        source_indices[idx] = left_idx
    return out, methods, source_indices


def _smoothed_positions(
    source: pd.DataFrame,
    target_times: np.ndarray,
    base_xyz: np.ndarray,
    *,
    mode: str,
    config: TrajectoryCompletionConfig,
) -> np.ndarray:
    if mode in {"raw", "gap-interpolation"}:
        return base_xyz
    measurements = _selected_measurements(source)
    measurement_times = measurements["time_s"].to_numpy(float)
    measurement_xyz = measurements[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    if mode == "fixed-lag":
        fitted = _local_polynomial_positions(
            target_times,
            measurement_times,
            measurement_xyz,
            degree=1,
            lag_s=config.fixed_lag_s,
        )
    elif mode == "constant-velocity":
        fitted = _global_polynomial_positions(
            target_times,
            measurement_times,
            measurement_xyz,
            degree=1,
        )
    elif mode == "constant-acceleration":
        fitted = _global_polynomial_positions(
            target_times,
            measurement_times,
            measurement_xyz,
            degree=2,
        )
    else:
        raise ValueError(f"unknown trajectory completion mode {mode!r}")
    finite_fit = np.isfinite(fitted).all(axis=1)
    blend = float(np.clip(config.smoothing_blend, 0.0, 1.0))
    out = base_xyz.copy()
    out[finite_fit] = (1.0 - blend) * base_xyz[finite_fit] + blend * fitted[finite_fit]
    return out


def _selected_measurements(source: pd.DataFrame) -> pd.DataFrame:
    selected = source.loc[source["selected_path_update"].astype(bool)].copy()
    return selected if len(selected) >= 2 else source


def _local_polynomial_positions(
    target_times: np.ndarray,
    measurement_times: np.ndarray,
    measurement_xyz: np.ndarray,
    *,
    degree: int,
    lag_s: float,
) -> np.ndarray:
    out = np.full((len(target_times), 3), np.nan, dtype=float)
    if len(measurement_times) == 0:
        return out
    nearest_count = min(len(measurement_times), max(int(degree) + 1, 2))
    for idx, timestamp in enumerate(target_times):
        mask = np.abs(measurement_times - timestamp) <= float(lag_s)
        if int(mask.sum()) < nearest_count:
            nearest = np.argsort(np.abs(measurement_times - timestamp))[:nearest_count]
            mask = np.zeros(len(measurement_times), dtype=bool)
            mask[nearest] = True
        out[idx] = _fit_polynomial_at(
            timestamp,
            measurement_times[mask],
            measurement_xyz[mask],
            degree=degree,
        )
    return out


def _global_polynomial_positions(
    target_times: np.ndarray,
    measurement_times: np.ndarray,
    measurement_xyz: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    if len(measurement_times) == 0:
        return np.full((len(target_times), 3), np.nan, dtype=float)
    return np.vstack(
        [
            _fit_polynomial_at(timestamp, measurement_times, measurement_xyz, degree=degree)
            for timestamp in target_times
        ]
    )


def _fit_polynomial_at(
    timestamp: float,
    measurement_times: np.ndarray,
    measurement_xyz: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    if len(measurement_times) == 0:
        return np.full(3, np.nan, dtype=float)
    fit_degree = min(int(degree), len(measurement_times) - 1)
    if fit_degree <= 0:
        return measurement_xyz[-1].astype(float)
    t0 = float(np.mean(measurement_times))
    x = measurement_times - t0
    value = float(timestamp) - t0
    return np.asarray(
        [
            np.polyval(np.polyfit(x, measurement_xyz[:, axis], fit_degree), value)
            for axis in range(3)
        ],
        dtype=float,
    )


def _finite_difference_velocities(times: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    if len(times) <= 1:
        return np.zeros_like(xyz)
    velocities = np.zeros_like(xyz)
    for idx in range(len(times)):
        if idx == 0:
            left, right = 0, 1
        elif idx == len(times) - 1:
            left, right = idx - 1, idx
        else:
            left, right = idx - 1, idx + 1
        dt = max(float(times[right] - times[left]), 1.0e-9)
        velocities[idx] = (xyz[right] - xyz[left]) / dt
    return velocities


def _method_for_mode(base_method: str, mode: str) -> str:
    if mode == "raw":
        return base_method
    if mode == "gap-interpolation":
        return base_method
    if base_method == "exact":
        return f"{mode}_smoothed"
    return f"{mode}_{base_method}"


def _gap_summary_rows(
    group: pd.DataFrame,
    target_times: np.ndarray,
    *,
    sequence_id: str,
    trajectory_id: str,
    config: TrajectoryCompletionConfig,
) -> list[dict[str, Any]]:
    source = _dedupe_by_time(group)
    times = source["time_s"].to_numpy(float)
    step = _typical_step_s(times)
    rows = []
    for gap_index, (left_idx, right_idx) in enumerate(zip(range(len(times) - 1), range(1, len(times)), strict=False)):
        left_t = float(times[left_idx])
        right_t = float(times[right_idx])
        inside = target_times[(target_times > left_t + 1.0e-9) & (target_times < right_t - 1.0e-9)]
        rows.append(
            {
                "sequence_id": sequence_id,
                "trajectory_id": trajectory_id,
                "gap_index": int(gap_index),
                "start_time_s": left_t,
                "end_time_s": right_t,
                "gap_s": float(right_t - left_t),
                "typical_step_s": float(step) if np.isfinite(step) else np.nan,
                "interpolation_eligible": bool((right_t - left_t) <= float(config.max_gap_s)),
                "target_times_inside_gap": int(len(inside)),
                "filled_count": int(len(inside) if (right_t - left_t) <= float(config.max_gap_s) else 0),
                "left_update_action": source.iloc[left_idx].get("update_action"),
                "right_update_action": source.iloc[right_idx].get("update_action"),
            }
        )
    return rows


def _ablation_rows(
    estimates: pd.DataFrame,
    truth_rows: pd.DataFrame | None,
    *,
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    sequence_ids = sorted(estimates["sequence_id"].dropna().astype(str).unique()) if not estimates.empty else []
    for sequence_id in sequence_ids:
        seq_estimates = estimates.loc[estimates["sequence_id"].astype(str) == sequence_id]
        metrics = compute_metrics(seq_estimates, _truth_for_sequence(truth_rows, sequence_id))
        rows.append(_metrics_row(seq_estimates, metrics, mode=mode, sequence_id=sequence_id))
    metrics = compute_metrics(estimates, truth_rows)
    rows.append(_metrics_row(estimates, metrics, mode=mode, sequence_id="__pooled__"))
    return rows


def _sequence_error_summary(
    *,
    raw: pd.DataFrame,
    final: pd.DataFrame,
    truth_rows: pd.DataFrame | None,
    mode: str,
) -> pd.DataFrame:
    sequence_ids = set()
    if not raw.empty:
        sequence_ids.update(raw["sequence_id"].dropna().astype(str))
    if not final.empty:
        sequence_ids.update(final["sequence_id"].dropna().astype(str))
    if truth_rows is not None and not truth_rows.empty:
        sequence_ids.update(truth_rows["sequence_id"].dropna().astype(str))
    records = []
    for sequence_id in sorted(sequence_ids):
        raw_seq = raw.loc[raw["sequence_id"].astype(str) == sequence_id] if not raw.empty else raw
        final_seq = final.loc[final["sequence_id"].astype(str) == sequence_id] if not final.empty else final
        truth_seq = _truth_for_sequence(truth_rows, sequence_id)
        raw_metrics = compute_metrics(raw_seq, truth_seq)
        final_metrics = compute_metrics(final_seq, truth_seq)
        records.append(
            {
                "sequence_id": sequence_id,
                "trajectory_completion_mode": mode,
                "raw_count": int(raw_metrics.get("count", 0)),
                "final_count": int(final_metrics.get("count", 0)),
                "filled_count": int(
                    final_seq.get("trajectory_completion_filled", pd.Series(dtype=bool)).fillna(False).sum()
                ),
                "raw_mean_3d_m": raw_metrics.get("mean_3d_m"),
                "final_mean_3d_m": final_metrics.get("mean_3d_m"),
                "delta_mean_3d_m": _metric_delta(raw_metrics, final_metrics, "mean_3d_m"),
                "raw_rmse_3d_m": raw_metrics.get("rmse_3d_m"),
                "final_rmse_3d_m": final_metrics.get("rmse_3d_m"),
                "delta_rmse_3d_m": _metric_delta(raw_metrics, final_metrics, "rmse_3d_m"),
                "raw_p95_3d_m": raw_metrics.get("p95_3d_m"),
                "final_p95_3d_m": final_metrics.get("p95_3d_m"),
                "delta_p95_3d_m": _metric_delta(raw_metrics, final_metrics, "p95_3d_m"),
                "raw_max_3d_m": raw_metrics.get("max_3d_m"),
                "final_max_3d_m": final_metrics.get("max_3d_m"),
                "delta_max_3d_m": _metric_delta(raw_metrics, final_metrics, "max_3d_m"),
                "final_roughness_mps2": _trajectory_roughness(final_seq),
            }
        )
    return pd.DataFrame.from_records(records)


def _metrics_row(estimates: pd.DataFrame, metrics: dict[str, Any], *, mode: str, sequence_id: str) -> dict[str, Any]:
    return {
        "trajectory_completion_mode": mode,
        "sequence_id": sequence_id,
        "row_count": int(len(estimates)),
        "filled_count": int(
            estimates.get("trajectory_completion_filled", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "mean_3d_m": metrics.get("mean_3d_m"),
        "rmse_3d_m": metrics.get("rmse_3d_m"),
        "p95_3d_m": metrics.get("p95_3d_m"),
        "max_3d_m": metrics.get("max_3d_m"),
        "mean_2d_m": metrics.get("mean_2d_m"),
        "p95_2d_m": metrics.get("p95_2d_m"),
        "max_2d_m": metrics.get("max_2d_m"),
        "roughness_mps2": _trajectory_roughness(estimates),
    }


def _attach_truth_errors_by_sequence(estimates: pd.DataFrame, truth_rows: pd.DataFrame | None) -> pd.DataFrame:
    if estimates.empty or truth_rows is None or truth_rows.empty:
        return estimates.copy()
    frames = []
    for sequence_id, group in estimates.groupby("sequence_id", sort=True):
        sequence_truth = _truth_for_sequence(truth_rows, str(sequence_id))
        frames.append(add_truth_errors(group.copy(), sequence_truth) if sequence_truth is not None and not sequence_truth.empty else group.copy())
    return pd.concat(frames, ignore_index=True) if frames else estimates.copy()


def _truth_for_sequence(truth_rows: pd.DataFrame | None, sequence_id: str) -> pd.DataFrame | None:
    if truth_rows is None or truth_rows.empty:
        return None
    return truth_rows.loc[truth_rows["sequence_id"].astype(str) == str(sequence_id)].copy()


def _metric_delta(raw_metrics: dict[str, Any], final_metrics: dict[str, Any], key: str) -> float | None:
    raw_value = raw_metrics.get(key)
    final_value = final_metrics.get(key)
    if raw_value is None or final_value is None:
        return None
    return float(final_value) - float(raw_value)


def _trajectory_roughness(estimates: pd.DataFrame) -> float | None:
    if estimates.empty or len(estimates) < 3:
        return None
    values = estimates.sort_values("time_s")
    times = pd.to_numeric(values["time_s"], errors="coerce").to_numpy(float)
    xyz = values[["state_x_m", "state_y_m", "state_z_m"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    finite = np.isfinite(times) & np.isfinite(xyz).all(axis=1)
    times = times[finite]
    xyz = xyz[finite]
    if len(times) < 3:
        return None
    velocities = _finite_difference_velocities(times, xyz)
    accelerations = _finite_difference_velocities(times, velocities)
    norms = np.linalg.norm(accelerations[1:-1], axis=1)
    finite_norms = norms[np.isfinite(norms)]
    return float(np.mean(finite_norms)) if finite_norms.size else None


def _unique_times(group: pd.DataFrame) -> np.ndarray:
    return np.asarray(sorted(pd.to_numeric(group["time_s"], errors="coerce").dropna().unique()), dtype=float)


def _typical_step_s(times: np.ndarray) -> float:
    if len(times) < 2:
        return np.nan
    diffs = np.diff(np.sort(times))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1.0e-9)]
    return float(np.median(diffs)) if diffs.size else np.nan


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    rows = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _sort_columns(rows: pd.DataFrame) -> list[str]:
    columns = ["sequence_id", "time_s"]
    if "output_track_id" in rows.columns:
        columns.insert(1, "output_track_id")
    return columns
