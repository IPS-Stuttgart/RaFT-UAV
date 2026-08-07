"""Compatibility fixes for the factor-graph research utilities.

The maintained implementation lives in the sibling ``factor_graph.py`` module.
This package preserves the public import path while parsing row-level measurement
uncertainties and time/position values defensively.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "factor_graph.py"
_LEGACY_NAME = f"{__name__.rsplit('.', 1)[0]}._factor_graph_legacy"
_SPEC = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError(f"cannot load factor-graph implementation from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_LEGACY_NAME] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)
_ORIGINAL_SMOOTH_POSITION_TRAJECTORY = _LEGACY.smooth_position_trajectory

_TIME_POSITION_COLUMNS = ("time_s", *_LEGACY.PositionColumns)
_VALID_ROBUST_LOSSES = frozenset({"linear", "huber", "soft_l1", "cauchy", "arctan"})


def _real_float(value: object) -> float | None:
    """Return a finite scalar with no nonzero imaginary component."""

    if value is None or np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, np.ndarray):
        if value.ndim > 0:
            return None
        value = value.item()
        if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
            return None
    if isinstance(value, (complex, np.complexfloating)):
        if (
            not np.isfinite(value.real)
            or not np.isfinite(value.imag)
            or value.imag != 0.0
        ):
            return None
        return float(value.real)
    return optional_float(value)


def _validated_robust_loss_config(config: object | None) -> object | None:
    """Reject unsupported least-squares losses before data-dependent returns."""

    if config is None:
        return None
    if not isinstance(config, _LEGACY.LeastSquaresSmoothingConfig):
        raise TypeError("config must be a LeastSquaresSmoothingConfig or None")
    robust_loss = config.robust_loss
    if callable(robust_loss):
        return config
    if not isinstance(robust_loss, str) or robust_loss not in _VALID_ROBUST_LOSSES:
        allowed = ", ".join(sorted(_VALID_ROBUST_LOSSES))
        raise ValueError(f"robust_loss must be one of {allowed} or a callable")
    return config


def _normalized_real_time_positions(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize finite real scalar time/position values without complex truncation."""

    normalized = frame.copy()
    for column in _TIME_POSITION_COLUMNS:
        if column in normalized.columns:
            normalized[column] = [_real_float(value) for value in normalized[column]]
    return normalized


def _real_position_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return real-valued candidate positions, marking malformed scalars as NaN."""

    return np.column_stack(
        [
            np.asarray([_real_float(value) for value in frame[column]], dtype=float)
            for column in _LEGACY.PositionColumns
        ]
    )


def _frame_time_s(frame: pd.DataFrame) -> float | None:
    """Return the finite median timestamp for one grouped radar frame."""

    times = np.asarray(
        [_real_float(value) for value in frame["time_s"]],
        dtype=float,
    )
    finite = times[np.isfinite(times)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def smooth_position_trajectory(
    measurements: pd.DataFrame,
    *,
    initial: pd.DataFrame | None = None,
    config: object | None = None,
):
    """Smooth positions after rejecting non-real scalar time/position values."""

    normalized_initial = (
        None if initial is None else _normalized_real_time_positions(initial)
    )
    return _ORIGINAL_SMOOTH_POSITION_TRAJECTORY(
        _normalized_real_time_positions(measurements),
        initial=normalized_initial,
        config=_validated_robust_loss_config(config),
    )


def _row_position_std(row: pd.Series, cfg: object) -> np.ndarray:
    """Return valid row uncertainty or the configured source-specific default."""

    source = str(row.get("source", "radar")).strip().casefold()
    default = float(cfg.rf_std_m if source == "rf" else cfg.measurement_std_m)

    std_columns = ("std_east_m", "std_north_m", "std_up_m")
    if all(column in row.index for column in std_columns):
        standard_deviations = [optional_float(row[column]) for column in std_columns]
        if all(value is not None and value > 0.0 for value in standard_deviations):
            return np.asarray(standard_deviations, dtype=float)

    covariance_columns = ("cov_ee", "cov_nn", "cov_uu")
    if all(column in row.index for column in covariance_columns):
        variances = [optional_float(row[column]) for column in covariance_columns]
        if all(value is not None and value >= 0.0 for value in variances):
            return np.sqrt(np.maximum(np.asarray(variances, dtype=float), 1.0e-9))

    return np.full(3, default, dtype=float)


def _initial_radar_selection(radar: pd.DataFrame) -> pd.DataFrame:
    """Choose one finite real-position candidate per usable radar frame."""

    rows = []
    for _, frame in _LEGACY._radar_frame_groups(radar):
        positions = _real_position_matrix(frame)
        finite_positions = np.isfinite(positions).all(axis=1)
        if not finite_positions.any():
            continue

        candidates = frame.loc[finite_positions]
        if "cat_prob_uav" in candidates.columns:
            scores = pd.Series(
                [_real_float(value) for value in candidates["cat_prob_uav"]],
                index=candidates.index,
                dtype=float,
            )
            finite_scores = scores.notna()
            if finite_scores.any():
                rows.append(candidates.loc[scores.loc[finite_scores].idxmax()].copy())
                continue
        rows.append(candidates.iloc[0].copy())

    return (
        pd.DataFrame(rows).reset_index(drop=True)
        if rows
        else radar.iloc[0:0].copy()
    )


def _select_candidates_against_trajectory(
    radar: pd.DataFrame,
    trajectory: pd.DataFrame,
    *,
    candidate_gate_m: float,
) -> pd.DataFrame:
    """Select finite real-position candidates nearest to the current trajectory."""

    if trajectory.empty:
        return _initial_radar_selection(radar)
    trajectory_times = trajectory["time_s"].to_numpy(dtype=float)
    trajectory_xyz = trajectory.loc[:, _LEGACY.PositionColumns].to_numpy(dtype=float)
    rows = []
    for _, frame in _LEGACY._radar_frame_groups(radar):
        time_s = _frame_time_s(frame)
        if time_s is None:
            continue
        pred = np.array(
            [
                np.interp(time_s, trajectory_times, trajectory_xyz[:, axis])
                for axis in range(3)
            ]
        )
        if not np.isfinite(pred).all():
            continue
        positions = _real_position_matrix(frame)
        finite = np.isfinite(positions).all(axis=1)
        if not finite.any():
            continue
        errors = np.full(len(frame), np.inf, dtype=float)
        errors[finite] = np.linalg.norm(
            positions[finite] - pred.reshape(1, 3),
            axis=1,
        )
        best_idx = int(np.argmin(errors))
        if float(errors[best_idx]) <= float(candidate_gate_m):
            row = frame.iloc[best_idx].copy()
            row["association_mode"] = "coordinate-descent-smoothing"
            row["association_score"] = float(errors[best_idx])
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else radar.iloc[0:0].copy()


_LEGACY.smooth_position_trajectory = smooth_position_trajectory
_LEGACY._row_position_std = _row_position_std
_LEGACY._initial_radar_selection = _initial_radar_selection
_LEGACY._select_candidates_against_trajectory = _select_candidates_against_trajectory

for _name in dir(_LEGACY):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_LEGACY, _name)
