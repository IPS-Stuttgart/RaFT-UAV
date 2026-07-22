"""Small factor-graph style smoothing and association-refinement utilities.

This module is not a replacement for a mature factor-graph library.  It gives
RaFT-UAV a dependency-light batch/fixed-lag MAP baseline that can be initialized
from existing tracker outputs and used to diagnose whether remaining error is due
to filtering, timing, or association.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

PositionColumns = ("east_m", "north_m", "up_m")


@dataclass(frozen=True)
class LeastSquaresSmoothingConfig:
    """Configuration for the lightweight position-only smoothing backend."""

    motion_std_mps2: float = 4.0
    measurement_std_m: float = 25.0
    rf_std_m: float = 50.0
    robust_loss: str = "soft_l1"
    max_nfev: int = 200


@dataclass(frozen=True)
class FactorGraphSmoothingResult:
    """Result of a batch/fixed-lag position smoothing solve."""

    estimates: pd.DataFrame
    cost: float
    optimality: float
    iterations: int
    success: bool
    message: str


def _finite_nonnegative_scalar(value: object, *, name: str) -> float:
    """Return a finite non-negative real scalar without accepting pseudo-numbers."""

    message = f"{name} must be a finite non-negative real scalar"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError(message)
    try:
        number = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(message)
    return number


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Return an exact non-negative scalar integer."""

    message = f"{name} must be a non-negative integer"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError(message)
    try:
        number = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number) or number < 0.0 or not number.is_integer():
        raise ValueError(message)
    return int(number)


def smooth_position_trajectory(
    measurements: pd.DataFrame,
    *,
    initial: pd.DataFrame | None = None,
    config: LeastSquaresSmoothingConfig | None = None,
) -> FactorGraphSmoothingResult:
    """Smooth 3D positions with measurement and constant-velocity residuals.

    Measurements must contain ``time_s,east_m,north_m,up_m`` and may contain
    ``source`` plus covariance/std columns. Rows without a finite timestamp and
    complete finite 3D position are ignored. The state is one 3D point per unique
    usable timestamp. A second-difference prior approximates a white-acceleration
    motion model.
    """

    cfg = config or LeastSquaresSmoothingConfig()
    if measurements.empty:
        return FactorGraphSmoothingResult(pd.DataFrame(columns=["time_s", *PositionColumns]), 0.0, 0.0, 0, True, "empty")
    _require_columns(measurements, {"time_s", *PositionColumns}, "measurements")
    ordered = measurements.copy()
    numeric_columns = ["time_s", *PositionColumns]
    for column in numeric_columns:
        ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    finite_rows = np.isfinite(
        ordered.loc[:, numeric_columns].to_numpy(dtype=float)
    ).all(axis=1)
    ordered = ordered.loc[finite_rows].sort_values("time_s").reset_index(drop=True)
    if ordered.empty:
        raise ValueError("measurements contain no finite time/position rows")
    times = np.sort(ordered["time_s"].unique().astype(float))
    index_by_time = {float(t): i for i, t in enumerate(times)}
    x0 = _initial_positions(times, ordered, initial)

    measurement_terms = []
    for _, row in ordered.iterrows():
        time_s = float(row["time_s"])
        if time_s not in index_by_time:
            continue
        vector = row.loc[list(PositionColumns)].to_numpy(dtype=float)
        if not np.isfinite(vector).all():
            continue
        std = _row_position_std(row, cfg)
        measurement_terms.append((index_by_time[time_s], vector, std))

    def residual(flat: np.ndarray) -> np.ndarray:
        points = flat.reshape(-1, 3)
        blocks: list[np.ndarray] = []
        for idx, vector, std in measurement_terms:
            blocks.append((points[idx] - vector) / std)
        if len(times) >= 3 and cfg.motion_std_mps2 > 0.0:
            for i in range(1, len(times) - 1):
                dt0 = max(float(times[i] - times[i - 1]), 1.0e-6)
                dt1 = max(float(times[i + 1] - times[i]), 1.0e-6)
                v0 = (points[i] - points[i - 1]) / dt0
                v1 = (points[i + 1] - points[i]) / dt1
                scale = max(float(cfg.motion_std_mps2) * np.sqrt(0.5 * (dt0 + dt1)), 1.0e-6)
                blocks.append((v1 - v0) / scale)
        return np.concatenate(blocks) if blocks else np.zeros(0)

    result = least_squares(
        residual,
        x0.reshape(-1),
        loss=cfg.robust_loss,
        max_nfev=int(cfg.max_nfev),
    )
    estimates = pd.DataFrame(result.x.reshape(-1, 3), columns=list(PositionColumns))
    estimates.insert(0, "time_s", times)
    return FactorGraphSmoothingResult(
        estimates=estimates,
        cost=float(result.cost),
        optimality=float(result.optimality),
        iterations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


def coordinate_descent_association_and_smoothing(
    radar: pd.DataFrame,
    rf: pd.DataFrame | None = None,
    *,
    iterations: int = 3,
    candidate_gate_m: float = 250.0,
    config: LeastSquaresSmoothingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alternate radar association and trajectory smoothing.

    This offline diagnostic approximates an EM loop: initialize a coarse
    trajectory from RF or high-class-probability radar rows, select the nearest
    plausible radar candidate at each frame, smooth the resulting measurements,
    and repeat.  Truth is not used.
    """

    _require_columns(radar, {"time_s", *PositionColumns}, "radar")
    iteration_count = _nonnegative_integer(iterations, name="iterations")
    candidate_gate = _finite_nonnegative_scalar(
        candidate_gate_m,
        name="candidate_gate_m",
    )
    cfg = config or LeastSquaresSmoothingConfig()
    selected = _initial_radar_selection(radar)
    measurements = _combine_measurements(selected, rf)
    trajectory = smooth_position_trajectory(measurements, config=cfg).estimates
    for _ in range(iteration_count):
        selected = _select_candidates_against_trajectory(
            radar,
            trajectory,
            candidate_gate_m=candidate_gate,
        )
        measurements = _combine_measurements(selected, rf)
        trajectory = smooth_position_trajectory(measurements, initial=trajectory, config=cfg).estimates
    return trajectory, selected


def _initial_positions(
    times: np.ndarray,
    measurements: pd.DataFrame,
    initial: pd.DataFrame | None,
) -> np.ndarray:
    if initial is not None and not initial.empty:
        _require_columns(initial, {"time_s", *PositionColumns}, "initial")
        initial_rows = initial.loc[:, ["time_s", *PositionColumns]].copy()
        for column in ("time_s", *PositionColumns):
            initial_rows[column] = pd.to_numeric(initial_rows[column], errors="coerce")
        finite = np.isfinite(initial_rows.to_numpy(dtype=float)).all(axis=1)
        if not finite.any():
            raise ValueError("initial contains no finite time/position rows")
        initial_rows = (
            initial_rows.loc[finite]
            .groupby("time_s", sort=True, as_index=False)[list(PositionColumns)]
            .median()
        )
        init_times = initial_rows["time_s"].to_numpy(dtype=float)
        init_xyz = initial_rows.loc[:, PositionColumns].to_numpy(dtype=float)
        return np.column_stack([np.interp(times, init_times, init_xyz[:, axis]) for axis in range(3)])
    grouped = measurements.groupby("time_s", sort=True)[list(PositionColumns)].median()
    grouped_times = grouped.index.to_numpy(dtype=float)
    grouped_xyz = grouped.to_numpy(dtype=float)
    return np.column_stack([np.interp(times, grouped_times, grouped_xyz[:, axis]) for axis in range(3)])


def _row_position_std(row: pd.Series, cfg: LeastSquaresSmoothingConfig) -> np.ndarray:
    source = str(row.get("source", "radar"))
    default = float(cfg.rf_std_m if source == "rf" else cfg.measurement_std_m)
    std_cols = ["std_east_m", "std_north_m", "std_up_m"]
    if all(col in row.index for col in std_cols):
        values = np.array([float(row[col]) for col in std_cols], dtype=float)
        if np.isfinite(values).all() and np.all(values > 0.0):
            return values
    cov_cols = ["cov_ee", "cov_nn", "cov_uu"]
    if all(col in row.index for col in cov_cols):
        values = np.sqrt(np.maximum([float(row[col]) for col in cov_cols], 1.0e-9))
        if np.isfinite(values).all() and np.all(values > 0.0):
            return values
    return np.full(3, default, dtype=float)


def _initial_radar_selection(radar: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, frame in _radar_frame_groups(radar):
        if "cat_prob_uav" in frame.columns:
            scores = pd.to_numeric(frame["cat_prob_uav"], errors="coerce").fillna(-np.inf)
            rows.append(frame.loc[scores.idxmax()].copy())
        else:
            rows.append(frame.iloc[0].copy())
    return pd.DataFrame(rows).reset_index(drop=True) if rows else radar.iloc[0:0].copy()


def _select_candidates_against_trajectory(
    radar: pd.DataFrame,
    trajectory: pd.DataFrame,
    *,
    candidate_gate_m: float,
) -> pd.DataFrame:
    if trajectory.empty:
        return _initial_radar_selection(radar)
    trajectory_times = trajectory["time_s"].to_numpy(dtype=float)
    trajectory_xyz = trajectory.loc[:, PositionColumns].to_numpy(dtype=float)
    rows = []
    for _, frame in _radar_frame_groups(radar):
        time_s = float(frame["time_s"].median())
        pred = np.array([np.interp(time_s, trajectory_times, trajectory_xyz[:, axis]) for axis in range(3)])
        if not np.isfinite(pred).all():
            continue
        positions = (
            frame.loc[:, PositionColumns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        finite = np.isfinite(positions).all(axis=1)
        if not finite.any():
            continue
        errors = np.full(len(frame), np.inf, dtype=float)
        errors[finite] = np.linalg.norm(positions[finite] - pred.reshape(1, 3), axis=1)
        best_idx = int(np.argmin(errors))
        if float(errors[best_idx]) <= float(candidate_gate_m):
            row = frame.iloc[best_idx].copy()
            row["association_mode"] = "coordinate-descent-smoothing"
            row["association_score"] = float(errors[best_idx])
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else radar.iloc[0:0].copy()


def _combine_measurements(radar_selected: pd.DataFrame, rf: pd.DataFrame | None) -> pd.DataFrame:
    frames = []
    if rf is not None and not rf.empty:
        rf_frame = rf.copy()
        rf_frame["source"] = "rf"
        frames.append(rf_frame)
    if not radar_selected.empty:
        radar_frame = radar_selected.copy()
        radar_frame["source"] = "radar"
        frames.append(radar_frame)
    if not frames:
        return pd.DataFrame(columns=["time_s", *PositionColumns, "source"])
    return pd.concat(frames, ignore_index=True, sort=False)


def _radar_frame_groups(radar: pd.DataFrame) -> list[tuple[object, pd.DataFrame]]:
    sort_cols = [c for c in ("time_s", "frame_index", "track_id") if c in radar.columns]
    ordered = radar.sort_values(sort_cols).reset_index(drop=True)
    times = pd.to_numeric(ordered["time_s"], errors="coerce")
    if "frame_index" in ordered.columns:
        frame_indices = pd.to_numeric(ordered["frame_index"], errors="coerce")
    else:
        frame_indices = pd.Series(np.nan, index=ordered.index, dtype=float)
    group_keys = pd.Series(
        [
            ("frame_index", float(frame_index))
            if np.isfinite(frame_index)
            else ("time_s", float(time_s))
            if np.isfinite(time_s)
            else None
            for frame_index, time_s in zip(frame_indices, times, strict=True)
        ],
        index=ordered.index,
        dtype=object,
    )
    usable = group_keys.notna()
    ordered = ordered.loc[usable]
    group_keys = group_keys.loc[usable]
    return [(key, group.copy()) for key, group in ordered.groupby(group_keys, sort=False)]


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
