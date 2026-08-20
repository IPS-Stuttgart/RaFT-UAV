"""Compatibility fixes for MMUAD trajectory-completion parsing and grid inference.

The maintained implementation lives in the sibling ``trajectory_completion.py``
module. This package preserves the public import path while parsing serialized
``selected_path_update`` values explicitly instead of relying on string
truthiness, normalizing serialized Boolean completion controls, avoiding
floating-point undercounting when inferring regular timestamps inside short
gaps, validating completion controls before they can silently disable
processing or corrupt finite trajectories, scoping completion and pooled
kinematic diagnostics by physical flight aliases, and preserving the final
posterior when sequential updates share a timestamp.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "trajectory_completion.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._trajectory_completion_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load MMUAD trajectory completion implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_COMPLETE_AND_SMOOTH_ESTIMATES = _IMPL.complete_and_smooth_estimates
_ORIGINAL_ESTIMATE_ROWS = _IMPL._estimate_rows
_ORIGINAL_SELECTED_MEASUREMENTS = _IMPL._selected_measurements
_ORIGINAL_SPEED_GATE_SUMMARY_ROW = _IMPL._speed_gate_summary_row
_TRUE_TEXT = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_TEXT = frozenset(
    {"0", "false", "f", "no", "n", "off", "", "none", "null", "nan", "<na>", "nat"}
)
_CONFIG_FALSE_TEXT = frozenset({"0", "false", "f", "no", "n", "off"})
_TRAJECTORY_COMPLETION_MODES = frozenset(
    {
        "none",
        "gap-interpolation",
        "fixed-lag",
        "constant-velocity",
        "constant-acceleration",
    }
)
_OUTLIER_REPLACEMENT_MODES = frozenset({"none", "local-linear"})


def _parse_selected_path_update(value: Any) -> bool:
    """Return one explicit path-selection flag without string truthiness."""

    scalar = value
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"cannot parse selected_path_update value: {value!r}")
        scalar = value.item()

    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if scalar is None:
        return False

    try:
        missing = pd.isna(scalar)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return False

    if isinstance(scalar, str):
        text = scalar.strip().lower()
        if text in _TRUE_TEXT:
            return True
        if text in _FALSE_TEXT:
            return False
        raise ValueError(f"cannot parse selected_path_update value: {scalar!r}")

    if isinstance(scalar, (int, float, np.integer, np.floating)):
        numeric = float(scalar)
        if not np.isfinite(numeric):
            return False
        if numeric in {0.0, 1.0}:
            return bool(numeric)
        raise ValueError(f"cannot parse selected_path_update value: {scalar!r}")

    raise ValueError(f"cannot parse selected_path_update value: {scalar!r}")


def _boolean_control(value: Any, *, name: str) -> bool:
    """Return one strict Boolean control, including common serialized forms."""

    message = f"{name} must be a Boolean scalar"
    scalar = value
    seen_array_ids: set[int] = set()
    while isinstance(scalar, np.ndarray):
        if scalar.ndim != 0:
            raise ValueError(message)
        array_id = id(scalar)
        if array_id in seen_array_ids:
            raise ValueError(message)
        seen_array_ids.add(array_id)
        scalar = scalar.item()

    if np.ma.is_masked(scalar):
        raise ValueError(message)
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, str):
        text = scalar.strip().casefold()
        if text in _TRUE_TEXT:
            return True
        if text in _CONFIG_FALSE_TEXT:
            return False
        raise ValueError(message)

    numeric = optional_float(scalar)
    if numeric in {0.0, 1.0}:
        return bool(numeric)
    raise ValueError(message)


def _normalized_selected_path_updates(rows: pd.DataFrame) -> pd.DataFrame:
    """Return rows with any selected-path flag column normalized to Boolean."""

    normalized = pd.DataFrame(rows).copy()
    if "selected_path_update" in normalized.columns:
        normalized["selected_path_update"] = normalized[
            "selected_path_update"
        ].map(_parse_selected_path_update)
    return normalized


def _estimate_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    """Normalize trajectory estimates after parsing serialized selection flags."""

    return _ORIGINAL_ESTIMATE_ROWS(_normalized_selected_path_updates(estimates))


def _selected_measurements(source: pd.DataFrame) -> pd.DataFrame:
    """Select smoothing measurements after normalizing serialized flags."""

    return _ORIGINAL_SELECTED_MEASUREMENTS(_normalized_selected_path_updates(source))


def _trajectory_groups(rows: pd.DataFrame):
    """Yield trajectories without crossing independent physical flights."""

    work = pd.DataFrame(rows).copy()
    group_columns = ["sequence_id"]
    if "flight_id" in work.columns:
        group_columns.append("flight_id")
    if "output_track_id" in work.columns:
        work["output_track_id"] = work["output_track_id"].fillna("single").astype(str)
        group_columns.append("output_track_id")
    else:
        work["_trajectory_id"] = "single"
        group_columns.append("_trajectory_id")

    for keys, group in work.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        sequence_id = key_tuple[0]
        trajectory_id = key_tuple[-1]
        yield (
            str(sequence_id),
            str(trajectory_id),
            group.drop(columns=["_trajectory_id"], errors="ignore"),
        )


def _dedupe_by_time(group: pd.DataFrame) -> pd.DataFrame:
    """Keep the final filter posterior for each duplicate timestamp.

    Same-timestamp tracker rows are sequential posterior snapshots. Prefer the
    explicit selected update, then any selected-path row, and otherwise the last
    snapshot. Coordinate-wise medians create a state the filter never produced.
    """

    rows = pd.DataFrame(group).sort_values("time_s", kind="mergesort").copy()
    if not rows["time_s"].duplicated().any():
        return rows.reset_index(drop=True)

    kept: list[pd.Series] = []
    for _, same_time in rows.groupby("time_s", sort=True):
        normalized = _normalized_selected_path_updates(same_time)
        update_actions = normalized.get(
            "update_action",
            pd.Series("", index=normalized.index, dtype=object),
        ).fillna("").astype(str)
        selected_updates = update_actions.eq("selected_update")
        if bool(selected_updates.any()):
            kept.append(normalized.loc[selected_updates].iloc[-1].copy())
            continue
        if "selected_path_update" in normalized.columns:
            selected_path = normalized["selected_path_update"].astype(bool)
            if bool(selected_path.any()):
                kept.append(normalized.loc[selected_path].iloc[-1].copy())
                continue
        kept.append(normalized.iloc[-1].copy())
    return pd.DataFrame(kept).reset_index(drop=True)


def _finite_nonnegative_control(value: Any, *, name: str) -> float:
    """Return one finite non-negative real trajectory-completion control."""

    parsed = optional_float(value)
    if parsed is None or parsed < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real scalar")
    return parsed


def _unit_interval_control(value: Any, *, name: str) -> float:
    """Return one finite real trajectory-completion control in [0, 1]."""

    parsed = optional_float(value)
    if parsed is None or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be a finite real scalar in [0, 1]")
    return parsed


def _validate_trajectory_completion_config(config: Any) -> None:
    """Reject malformed controls before empty-input or smoothing shortcuts."""

    mode = getattr(config, "mode", None)
    if mode not in _TRAJECTORY_COMPLETION_MODES:
        raise ValueError(f"unknown trajectory completion mode {mode!r}")

    replacement = getattr(config, "outlier_replacement", None)
    if replacement not in _OUTLIER_REPLACEMENT_MODES:
        raise ValueError(f"unknown trajectory outlier replacement {replacement!r}")

    _finite_nonnegative_control(config.max_gap_s, name="max_gap_s")
    _finite_nonnegative_control(config.fixed_lag_s, name="fixed_lag_s")
    _unit_interval_control(config.smoothing_blend, name="smoothing_blend")
    _boolean_control(
        config.include_truth_timestamps,
        name="include_truth_timestamps",
    )
    _boolean_control(config.infer_missing_grid, name="infer_missing_grid")
    _finite_nonnegative_control(config.speed_gate_mps, name="speed_gate_mps")
    if config.outlier_replacement_max_gap_s is not None:
        _finite_nonnegative_control(
            config.outlier_replacement_max_gap_s,
            name="outlier_replacement_max_gap_s",
        )


def _normalized_trajectory_completion_config(config: Any):
    """Return a config whose Boolean controls cannot rely on string truthiness."""

    _validate_trajectory_completion_config(config)
    return replace(
        config,
        include_truth_timestamps=_boolean_control(
            config.include_truth_timestamps,
            name="include_truth_timestamps",
        ),
        infer_missing_grid=_boolean_control(
            config.infer_missing_grid,
            name="infer_missing_grid",
        ),
    )


def _trajectory_diagnostic_groups(estimates: pd.DataFrame):
    """Yield independent trajectories for aggregate kinematic diagnostics."""

    group_columns = [
        column
        for column in ("sequence_id", "flight_id", "output_track_id")
        if column in estimates.columns
    ]
    if not group_columns:
        yield estimates
        return
    for _, group in estimates.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        yield group


def _trajectory_segment_speeds(estimates: pd.DataFrame) -> np.ndarray:
    """Return finite segment speeds without crossing trajectory boundaries."""

    required = {"time_s", "state_x_m", "state_y_m", "state_z_m"}
    if estimates.empty or not required.issubset(estimates.columns):
        return np.asarray([], dtype=float)

    speed_parts: list[np.ndarray] = []
    for group in _trajectory_diagnostic_groups(estimates):
        values = group.sort_values("time_s")
        times = pd.to_numeric(values["time_s"], errors="coerce").to_numpy(float)
        xyz = (
            values[["state_x_m", "state_y_m", "state_z_m"]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )
        finite = np.isfinite(times) & np.isfinite(xyz).all(axis=1)
        speeds = _IMPL._segment_speeds(times[finite], xyz[finite])
        finite_speeds = speeds[np.isfinite(speeds)]
        if finite_speeds.size:
            speed_parts.append(finite_speeds)
    return np.concatenate(speed_parts) if speed_parts else np.asarray([], dtype=float)


def _speed_gate_summary_row(
    source: pd.DataFrame,
    *,
    sequence_id: str,
    trajectory_id: str,
    config: Any,
) -> dict[str, Any]:
    """Summarize segment speeds without connecting independent trajectories."""

    summary = _ORIGINAL_SPEED_GATE_SUMMARY_ROW(
        source,
        sequence_id=sequence_id,
        trajectory_id=trajectory_id,
        config=config,
    )
    finite_speeds = _trajectory_segment_speeds(source)
    gate = float(config.speed_gate_mps or 0.0)
    summary.update(
        {
            "segment_count": int(len(finite_speeds)),
            "segment_over_gate_count": int(np.sum(finite_speeds > gate))
            if gate > 0.0
            else 0,
            "max_segment_speed_mps": float(np.max(finite_speeds))
            if finite_speeds.size
            else np.nan,
            "p95_segment_speed_mps": float(np.percentile(finite_speeds, 95.0))
            if finite_speeds.size
            else np.nan,
            "median_segment_speed_mps": float(np.median(finite_speeds))
            if finite_speeds.size
            else np.nan,
        }
    )
    return summary


def _trajectory_roughness(estimates: pd.DataFrame) -> float | None:
    """Average acceleration magnitude without crossing trajectory boundaries."""

    required = {"time_s", "state_x_m", "state_y_m", "state_z_m"}
    if estimates.empty or not required.issubset(estimates.columns):
        return None

    norm_parts: list[np.ndarray] = []
    for group in _trajectory_diagnostic_groups(estimates):
        values = group.sort_values("time_s")
        times = pd.to_numeric(values["time_s"], errors="coerce").to_numpy(float)
        xyz = (
            values[["state_x_m", "state_y_m", "state_z_m"]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )
        finite = np.isfinite(times) & np.isfinite(xyz).all(axis=1)
        times = times[finite]
        xyz = xyz[finite]
        if len(times) < 3:
            continue
        velocities = _IMPL._finite_difference_velocities(times, xyz)
        accelerations = _IMPL._finite_difference_velocities(times, velocities)
        norms = np.linalg.norm(accelerations[1:-1], axis=1)
        finite_norms = norms[np.isfinite(norms)]
        if finite_norms.size:
            norm_parts.append(finite_norms)
    if not norm_parts:
        return None
    return float(np.mean(np.concatenate(norm_parts)))


def complete_and_smooth_estimates(
    estimates: pd.DataFrame,
    truth: Any = None,
    *,
    config: Any = None,
):
    """Complete trajectories after validating every processing control."""

    resolved_config = (
        _IMPL.TrajectoryCompletionConfig() if config is None else config
    )
    normalized_config = _normalized_trajectory_completion_config(resolved_config)
    return _ORIGINAL_COMPLETE_AND_SMOOTH_ESTIMATES(
        estimates,
        truth,
        config=normalized_config,
    )


def _target_times(
    group: pd.DataFrame,
    truth_rows: pd.DataFrame | None,
    *,
    config: Any,
) -> np.ndarray:
    """Build target times without floor-based floating-point undercounting."""

    original = _IMPL._unique_times(group)
    targets = {float(value) for value in original}
    scoped_truth = truth_rows
    if (
        truth_rows is not None
        and not truth_rows.empty
        and "flight_id" in group.columns
        and "flight_id" in truth_rows.columns
    ):
        flight_values = group["flight_id"].drop_duplicates()
        if len(flight_values) == 1:
            flight_id = flight_values.iloc[0]
            if pd.isna(flight_id):
                scoped_truth = truth_rows.loc[truth_rows["flight_id"].isna()].copy()
            else:
                scoped_truth = truth_rows.loc[
                    truth_rows["flight_id"].astype(str) == str(flight_id)
                ].copy()
    include_truth_timestamps = _boolean_control(
        config.include_truth_timestamps,
        name="include_truth_timestamps",
    )
    infer_missing_grid = _boolean_control(
        config.infer_missing_grid,
        name="infer_missing_grid",
    )
    if (
        include_truth_timestamps
        and scoped_truth is not None
        and not scoped_truth.empty
    ):
        for timestamp in pd.to_numeric(
            scoped_truth["time_s"],
            errors="coerce",
        ).to_numpy(float):
            if np.isfinite(timestamp) and _IMPL._time_supported_by_short_gap(
                timestamp,
                original,
                config.max_gap_s,
            ):
                targets.add(float(timestamp))
    elif infer_missing_grid:
        step = _IMPL._typical_step_s(original)
        if np.isfinite(step) and step > 0.0:
            for left, right in zip(original[:-1], original[1:], strict=False):
                gap_s = float(right - left)
                if (
                    gap_s <= max(float(config.max_gap_s), step)
                    and gap_s > 1.5 * step
                ):
                    index = 1
                    previous = float(left)
                    while True:
                        value = float(left + index * step)
                        if value <= previous or value >= right - 1.0e-9:
                            break
                        targets.add(value)
                        previous = value
                        index += 1
    return np.asarray(sorted(targets), dtype=float)


_IMPL._parse_selected_path_update = _parse_selected_path_update
_IMPL._boolean_control = _boolean_control
_IMPL._normalized_selected_path_updates = _normalized_selected_path_updates
_IMPL._estimate_rows = _estimate_rows
_IMPL._selected_measurements = _selected_measurements
_IMPL._trajectory_groups = _trajectory_groups
_IMPL._dedupe_by_time = _dedupe_by_time
_IMPL._finite_nonnegative_control = _finite_nonnegative_control
_IMPL._unit_interval_control = _unit_interval_control
_IMPL._validate_trajectory_completion_config = _validate_trajectory_completion_config
_IMPL._normalized_trajectory_completion_config = (
    _normalized_trajectory_completion_config
)
_IMPL._speed_gate_summary_row = _speed_gate_summary_row
_IMPL._trajectory_roughness = _trajectory_roughness
_IMPL.complete_and_smooth_estimates = complete_and_smooth_estimates
_IMPL._target_times = _target_times

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_parse_selected_path_update"] = _parse_selected_path_update
globals()["_boolean_control"] = _boolean_control
globals()["_normalized_selected_path_updates"] = _normalized_selected_path_updates
globals()["_estimate_rows"] = _estimate_rows
globals()["_selected_measurements"] = _selected_measurements
globals()["_trajectory_groups"] = _trajectory_groups
globals()["_dedupe_by_time"] = _dedupe_by_time
globals()["_finite_nonnegative_control"] = _finite_nonnegative_control
globals()["_unit_interval_control"] = _unit_interval_control
globals()["_validate_trajectory_completion_config"] = (
    _validate_trajectory_completion_config
)
globals()["_normalized_trajectory_completion_config"] = (
    _normalized_trajectory_completion_config
)
globals()["_speed_gate_summary_row"] = _speed_gate_summary_row
globals()["_trajectory_roughness"] = _trajectory_roughness
globals()["complete_and_smooth_estimates"] = complete_and_smooth_estimates
globals()["_target_times"] = _target_times

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
