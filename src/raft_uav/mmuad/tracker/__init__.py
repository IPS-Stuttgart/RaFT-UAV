"""Compatibility fixes for the basic MMUAD tracker.

The maintained implementation lives in the sibling ``tracker.py`` module. This
package preserves the public import path while normalizing numeric inputs,
ignoring unusable timestamps in mobility scoring, preventing blank track IDs
from becoming false multi-frame identities, ordering same-timestamp updates
deterministically, keeping finite measurement covariances, keeping final
same-time truth samples, and keeping truth interpolation inside the supported
time span.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "tracker.py"
_LEGACY_NAME = f"{__name__.rsplit('.', 1)[0]}._tracker_legacy"
_SPEC = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError(f"cannot load tracker implementation from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_LEGACY_NAME] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)

for _name in dir(_LEGACY):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_LEGACY, _name)

TrackerConfig = _LEGACY.TrackerConfig
_ORIGINAL_CANDIDATE_ROWS_WITH_OPTIONAL_DEFAULTS = (
    _LEGACY._candidate_rows_with_optional_defaults
)
_ORIGINAL_CANDIDATE_MOBILITY = _LEGACY._candidate_mobility
_ORIGINAL_SELECT_TRACKLET_PATH = _LEGACY.select_tracklet_path
_ORIGINAL_RUN_MMUAD_TRACKER = _LEGACY.run_mmuad_tracker
_TRACKER_NUMERIC_COLUMNS = (
    "time_s",
    "x_m",
    "y_m",
    "z_m",
    "std_xy_m",
    "std_z_m",
    "confidence",
)
_FILTER_SORT_COLUMNS = (
    "time_s",
    "_selected_update_order",
    "_source_sort_key",
    "_track_sort_key",
    "x_m",
    "y_m",
    "z_m",
    "_std_xy_sort_key",
    "_std_z_sort_key",
)
_FILTER_SORT_ASCENDING = (
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    False,
)
_FILTER_HELPER_COLUMNS = (
    "_selected_update_order",
    "_source_sort_key",
    "_track_sort_key",
    "_std_xy_sort_key",
    "_std_z_sort_key",
)
_FLOAT_MAX = np.finfo(float).max


def _normalize_covariance_scale(value: object, *, field_name: str) -> float:
    """Return a finite non-negative covariance scale."""

    scale = optional_float(value)
    if scale is None or scale < 0.0:
        raise ValueError(
            f"{field_name} must be a finite non-negative real scalar"
        )
    return scale


def _normalize_soft_anchor_limit(value: object, *, field_name: str) -> float:
    """Return a finite non-negative soft-anchor distance control."""

    limit = optional_float(value)
    if limit is None or limit < 0.0:
        raise ValueError(
            f"{field_name} must be a finite non-negative real scalar"
        )
    return limit


def _normalize_boolean_control(value: object, *, field_name: str) -> bool:
    """Return an actual Python/NumPy Boolean scalar without truthy coercion."""

    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a Boolean scalar")
    return bool(value)


def _finite_scaled_square(value: float, scale: float) -> float:
    """Return ``value**2 * scale`` without overflowing intermediate products."""

    value = float(value)
    scale = float(scale)
    if scale == 0.0:
        return 0.0
    value_mantissa, value_exponent = math.frexp(value)
    scale_mantissa, scale_exponent = math.frexp(scale)
    mantissa = value_mantissa * value_mantissa * scale_mantissa
    exponent = 2 * value_exponent + scale_exponent
    try:
        variance = math.ldexp(mantissa, exponent)
    except OverflowError:
        return _FLOAT_MAX
    return min(float(variance), _FLOAT_MAX)


def _measurement_covariance(
    std_xy: float,
    std_z: float,
    *,
    scale: float,
) -> np.ndarray:
    """Build a finite diagonal measurement covariance for a validated scale."""

    xy_variance = _finite_scaled_square(std_xy, scale)
    z_variance = _finite_scaled_square(std_z, scale)
    return np.diag([xy_variance, xy_variance, z_variance])


def _stable_filter_update(
    filt: object,
    measurement: np.ndarray,
    covariance: np.ndarray,
) -> None:
    """Apply the Kalman update without SVD cutoff on highly anisotropic covariance."""

    h = np.zeros((3, 6), dtype=float)
    h[0, 0] = h[1, 1] = h[2, 2] = 1.0
    innovation = np.asarray(measurement, dtype=float) - h @ filt.state
    innovation_covariance = h @ filt.covariance @ h.T + covariance
    prior_cross_covariance = filt.covariance @ h.T
    try:
        gain = np.linalg.solve(
            innovation_covariance,
            prior_cross_covariance.T,
        ).T
    except np.linalg.LinAlgError:
        gain = prior_cross_covariance @ np.linalg.pinv(innovation_covariance)
    filt.state = filt.state + gain @ innovation
    filt.covariance = (np.eye(6) - gain @ h) @ filt.covariance


def _validated_tracker_config(config: TrackerConfig) -> TrackerConfig:
    """Normalize guarded tracker scalars before they reach filter logic."""

    return replace(
        config,
        acceleration_std_mps2=_normalize_covariance_scale(
            config.acceleration_std_mps2,
            field_name="acceleration_std_mps2",
        ),
        primary_covariance_scale=_normalize_covariance_scale(
            config.primary_covariance_scale,
            field_name="primary_covariance_scale",
        ),
        secondary_covariance_scale=_normalize_covariance_scale(
            config.secondary_covariance_scale,
            field_name="secondary_covariance_scale",
        ),
        soft_anchor_cap_m=_normalize_soft_anchor_limit(
            config.soft_anchor_cap_m,
            field_name="soft_anchor_cap_m",
        ),
        soft_anchor_gate_m=_normalize_soft_anchor_limit(
            config.soft_anchor_gate_m,
            field_name="soft_anchor_gate_m",
        ),
        first_selected_bootstrap=_normalize_boolean_control(
            config.first_selected_bootstrap,
            field_name="first_selected_bootstrap",
        ),
    )


def run_mmuad_tracker(candidates, truth=None, *, config=None):
    """Run the tracker without silently replacing invalid falsy configurations."""

    resolved_config = TrackerConfig() if config is None else config
    if not isinstance(resolved_config, TrackerConfig):
        raise TypeError("config must be a TrackerConfig or None")
    resolved_config = _validated_tracker_config(resolved_config)
    return _ORIGINAL_RUN_MMUAD_TRACKER(
        candidates,
        truth,
        config=resolved_config,
    )


def _candidate_rows_with_optional_defaults(rows: pd.DataFrame) -> pd.DataFrame:
    """Fill defaults and retain only finite real scalar tracker values."""

    out = _ORIGINAL_CANDIDATE_ROWS_WITH_OPTIONAL_DEFAULTS(rows)
    for column in _TRACKER_NUMERIC_COLUMNS:
        if column in out.columns:
            out[column] = out[column].map(optional_float)
    return out


def _candidate_mobility(frame: pd.DataFrame, *, radius_m: float) -> np.ndarray:
    """Ignore unusable timestamp rows when computing the spatial mobility prior."""

    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(float)
    finite_time = np.isfinite(times)
    if finite_time.all():
        return _ORIGINAL_CANDIDATE_MOBILITY(frame, radius_m=radius_m)

    mobility = np.zeros(len(frame), dtype=float)
    if finite_time.any():
        mobility[finite_time] = _ORIGINAL_CANDIDATE_MOBILITY(
            frame.loc[finite_time].copy(),
            radius_m=radius_m,
        )
    return mobility


def _is_blank_track_id(value: object) -> bool:
    """Return whether a string identifier contains no non-whitespace text."""

    return isinstance(value, str) and not value.strip()


def select_tracklet_path(
    candidates: pd.DataFrame,
    *,
    config: TrackerConfig,
) -> pd.DataFrame:
    """Select a path without treating blank IDs as stable track identities."""

    rows = pd.DataFrame(candidates).copy()
    if "track_id" in rows.columns:
        blank = rows["track_id"].map(_is_blank_track_id)
        if bool(blank.any()):
            rows.loc[blank, "track_id"] = np.nan
    return _ORIGINAL_SELECT_TRACKLET_PATH(rows, config=config)


def _ordered_filter_events(
    candidates: pd.DataFrame,
    *,
    selected_keys: set[tuple[object, ...]],
) -> pd.DataFrame:
    """Return a deterministic within-frame update order with selected rows last."""

    events = candidates.loc[_LEGACY._finite_position_mask(candidates)].copy()
    if events.empty:
        return events.reset_index(drop=True)

    events["_selected_update_order"] = [
        int(_LEGACY._candidate_key(row) in selected_keys)
        for _, row in events.iterrows()
    ]
    events["_source_sort_key"] = events.get(
        "source",
        pd.Series("", index=events.index, dtype=object),
    ).fillna("").astype(str)
    events["_track_sort_key"] = events.get(
        "track_id",
        pd.Series("", index=events.index, dtype=object),
    ).fillna("").astype(str)
    std_xy_values = events.get(
        "std_xy_m",
        pd.Series(10.0, index=events.index, dtype=float),
    )
    events["_std_xy_sort_key"] = [
        _LEGACY._positive_float(value, default=10.0) for value in std_xy_values
    ]
    std_z_values = events.get(
        "std_z_m",
        pd.Series(np.nan, index=events.index, dtype=float),
    )
    events["_std_z_sort_key"] = [
        _LEGACY._positive_float(value, default=std_xy)
        for value, std_xy in zip(
            std_z_values,
            events["_std_xy_sort_key"],
            strict=True,
        )
    ]
    # Capped soft-anchor updates are nonlinear, so equal-position measurements
    # with different covariance do not commute. Process less precise rows first
    # and leave the more precise update last within an otherwise identical tie.
    return (
        events.sort_values(
            list(_FILTER_SORT_COLUMNS),
            ascending=list(_FILTER_SORT_ASCENDING),
            kind="mergesort",
        )
        .drop(columns=list(_FILTER_HELPER_COLUMNS))
        .reset_index(drop=True)
    )


def _run_sequence_filter(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    sequence_truth: pd.DataFrame | None,
    config: TrackerConfig,
) -> pd.DataFrame:
    """Filter one sequence using a row-order-independent within-frame schedule."""

    selected = selected.loc[_LEGACY._finite_position_mask(selected)].copy()
    if selected.empty:
        return pd.DataFrame()
    selected_keys = set(_LEGACY._candidate_keys(selected))
    events = _ordered_filter_events(candidates, selected_keys=selected_keys)
    if events.empty:
        return pd.DataFrame()

    ordered_selected = _ordered_filter_events(selected, selected_keys=selected_keys)
    # Disabling selected-measurement bootstrap must also change the initial event.
    # Otherwise rows before the first selected detection are processed from a state
    # initialized in the future, while the legacy predictor only rewinds its clock.
    bootstrap = (
        ordered_selected.iloc[0]
        if config.first_selected_bootstrap
        else events.iloc[0]
    )
    filt = _LEGACY._ConstantVelocityFilter(
        acceleration_std_mps2=config.acceleration_std_mps2,
        initial_time_s=float(bootstrap["time_s"]),
        initial_position=bootstrap[["x_m", "y_m", "z_m"]].to_numpy(float),
    )
    estimate_rows: list[dict[str, object]] = []
    for _, row in events.iterrows():
        time_s = float(row["time_s"])
        if config.first_selected_bootstrap and time_s < float(bootstrap["time_s"]):
            continue
        filt.predict(time_s)
        key = _LEGACY._candidate_key(row)
        is_selected = key in selected_keys
        z = row[["x_m", "y_m", "z_m"]].to_numpy(float)
        std_xy = _LEGACY._positive_float(row.get("std_xy_m", 10.0), default=10.0)
        std_z = _LEGACY._positive_float(row.get("std_z_m", std_xy), default=std_xy)
        if is_selected:
            action = "selected_update"
            _stable_filter_update(
                filt,
                z,
                _measurement_covariance(
                    std_xy,
                    std_z,
                    scale=config.primary_covariance_scale,
                ),
            )
        else:
            predicted = filt.state[:3].copy()
            innovation = z - predicted
            if config.soft_anchor_gate_m > 0 and (
                float(_overflow_stable_norm(innovation, axis=0))
                > config.soft_anchor_gate_m
            ):
                action = "soft_anchor_gated"
            else:
                action = "soft_anchor"
                horizontal_norm = float(_overflow_stable_norm(innovation[:2], axis=0))
                if horizontal_norm > config.soft_anchor_cap_m > 0:
                    innovation[:2] *= config.soft_anchor_cap_m / horizontal_norm
                capped_z = predicted + innovation
                _stable_filter_update(
                    filt,
                    capped_z,
                    _measurement_covariance(
                        std_xy,
                        std_z,
                        scale=config.secondary_covariance_scale,
                    ),
                )
        state = filt.state.copy()
        estimate_rows.append(
            {
                "time_s": time_s,
                "source": row.get("source"),
                "track_id": row.get("track_id"),
                "class_name": row.get("class_name"),
                "update_action": action,
                "selected_path_update": bool(is_selected),
                "state_x_m": state[0],
                "state_y_m": state[1],
                "state_z_m": state[2],
                "v_x_mps": state[3],
                "v_y_mps": state[4],
                "v_z_mps": state[5],
            }
        )
    estimates = pd.DataFrame.from_records(estimate_rows)
    if sequence_truth is not None and not sequence_truth.empty and not estimates.empty:
        estimates = add_truth_errors(estimates, sequence_truth)
    return estimates


def _finite_truth_by_time(truth: pd.DataFrame) -> pd.DataFrame:
    """Return the final finite truth row for every normalized timestamp."""

    columns = ["time_s", "x_m", "y_m", "z_m"]
    work = truth.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(work.to_numpy(dtype=float)).all(axis=1)
    if not finite.any():
        return work.iloc[0:0].copy()
    return (
        work.loc[finite]
        .drop_duplicates(subset=["time_s"], keep="last")
        .sort_values("time_s", kind="mergesort")
        .reset_index(drop=True)
    )


def _overflow_stable_norm(values: np.ndarray, *, axis: int) -> np.ndarray:
    """Preserve ordinary norms and repair only finite-input overflow results."""

    array = np.asarray(values, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.asarray(np.linalg.norm(array, axis=axis), dtype=float)
    finite_inputs = np.isfinite(array).all(axis=axis)
    repair = finite_inputs & ~np.isfinite(norms)
    if np.ndim(norms) == 0:
        if bool(repair):
            return np.asarray(
                np.hypot.reduce(np.abs(array), axis=axis),
                dtype=float,
            )
        return norms
    if bool(np.any(repair)):
        norms = norms.copy()
        norms[repair] = np.hypot.reduce(np.abs(array[repair]), axis=axis)
    return norms


def add_truth_errors(estimates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Attach interpolated truth errors only inside the finite truth time span."""

    out = estimates.copy()
    truth_interp = _LEGACY._finite_truth_by_time(truth)
    estimate_times = pd.to_numeric(out["time_s"], errors="coerce").to_numpy(float)
    interp = np.full((len(out), 3), np.nan, dtype=float)
    if not truth_interp.empty:
        truth_times = truth_interp["time_s"].to_numpy(float)
        truth_xyz = truth_interp[["x_m", "y_m", "z_m"]].to_numpy(float)
        supported = (
            np.isfinite(estimate_times)
            & (estimate_times >= truth_times[0])
            & (estimate_times <= truth_times[-1])
        )
        if supported.any():
            interp[supported] = np.column_stack(
                [
                    np.interp(
                        estimate_times[supported],
                        truth_times,
                        truth_xyz[:, axis],
                    )
                    for axis in range(3)
                ]
            )
    est_xyz = (
        out[["state_x_m", "state_y_m", "state_z_m"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        err = est_xyz - interp
    out["truth_x_m"] = interp[:, 0]
    out["truth_y_m"] = interp[:, 1]
    out["truth_z_m"] = interp[:, 2]
    out["error_2d_m"] = _overflow_stable_norm(err[:, :2], axis=1)
    out["error_3d_m"] = _overflow_stable_norm(err, axis=1)
    return out


def _stable_nonnegative_mean(values: np.ndarray) -> float:
    """Return a mean without overflowing the intermediate sum."""

    array = np.asarray(values, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        ordinary = float(np.mean(array))
    if np.isfinite(ordinary):
        return ordinary
    scale = float(np.max(array))
    if scale == 0.0:
        return 0.0
    return float(scale * np.mean(array / scale))


def _stable_root_mean_square(values: np.ndarray) -> float:
    """Return RMS without squaring unscaled large finite values."""

    array = np.asarray(values, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        ordinary = float(np.sqrt(np.mean(array**2)))
    if np.isfinite(ordinary):
        return ordinary
    scale = float(np.max(np.abs(array)))
    if scale == 0.0:
        return 0.0
    scaled = array / scale
    return float(scale * np.sqrt(np.mean(scaled**2)))


def compute_metrics(estimates: pd.DataFrame, truth: pd.DataFrame | None) -> dict[str, object]:
    """Summarize tracker errors without overflowing representable statistics."""

    del truth
    if estimates is None or estimates.empty or "error_3d_m" not in estimates.columns:
        return {"count": int(len(estimates)) if estimates is not None else 0}
    err3 = pd.to_numeric(estimates["error_3d_m"], errors="coerce").to_numpy(float)
    err2 = (
        pd.to_numeric(estimates["error_2d_m"], errors="coerce").to_numpy(float)
        if "error_2d_m" in estimates.columns
        else np.array([], dtype=float)
    )
    finite3 = err3[np.isfinite(err3)]
    finite2 = err2[np.isfinite(err2)]
    if finite3.size == 0:
        return {"count": 0}
    return {
        "count": int(finite3.size),
        "mean_3d_m": _stable_nonnegative_mean(finite3),
        "rmse_3d_m": _stable_root_mean_square(finite3),
        "p95_3d_m": float(np.percentile(finite3, 95.0)),
        "max_3d_m": float(np.max(finite3)),
        "mean_2d_m": _stable_nonnegative_mean(finite2) if finite2.size else None,
        "p95_2d_m": float(np.percentile(finite2, 95.0)) if finite2.size else None,
        "max_2d_m": float(np.max(finite2)) if finite2.size else None,
    }


_LEGACY._candidate_rows_with_optional_defaults = _candidate_rows_with_optional_defaults
_LEGACY._candidate_mobility = _candidate_mobility
_LEGACY.select_tracklet_path = select_tracklet_path
_LEGACY.run_mmuad_tracker = run_mmuad_tracker
_LEGACY._run_sequence_filter = _run_sequence_filter
_LEGACY._finite_truth_by_time = _finite_truth_by_time
_LEGACY.add_truth_errors = add_truth_errors
_LEGACY.compute_metrics = compute_metrics
