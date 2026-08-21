"""Validate robust-MAP configuration, lag horizons, and record chronology."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps

import numpy as np

from raft_uav.baselines import robust_map as _robust_map
from raft_uav.baselines import smoothing as _smoothing
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.numeric import optional_float as _optional_float
from raft_uav.numeric import optional_int as _optional_int

_ORIGINAL_ROBUST_MAP_SMOOTH_RECORDS = _robust_map.robust_map_smooth_records
_ORIGINAL_SMOOTH_TRACKING_RECORDS = _smoothing.smooth_tracking_records


def _validated_lag_s(value: object) -> float | None:
    """Return a finite nonnegative lag horizon without lossy coercion."""

    if value is None:
        return None
    error = "lag_s must be a finite nonnegative real scalar or None"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        lag_s = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(lag_s) or lag_s < 0.0:
        raise ValueError(error)
    return lag_s


def _finite_float(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> float:
    """Return a finite real scalar satisfying the requested lower bound."""

    number = _optional_float(value)
    qualifier = "positive" if positive else "nonnegative"
    if number is None or (number <= 0.0 if positive else number < 0.0):
        raise ValueError(f"{name} must be a finite {qualifier} real scalar")
    return number


def _positive_integer(value: object, *, name: str) -> int:
    """Return an exact positive integer scalar."""

    number = _optional_int(value)
    if number is None or number < 1:
        raise ValueError(f"{name} must be a positive integer scalar")
    return number


def _boolean(value: object, *, name: str) -> bool:
    """Return an actual Boolean scalar instead of applying truthiness."""

    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a Boolean scalar")
    return bool(value)


def _validate_robust_map_config_fields(
    config: _robust_map.RobustMapSmootherConfig,
) -> _robust_map.RobustMapSmootherConfig:
    """Validate and normalize every robust-MAP configuration field."""

    if not isinstance(config.loss, str) or config.loss not in _robust_map.ROBUST_MAP_LOSSES:
        raise ValueError(f"loss must be one of {_robust_map.ROBUST_MAP_LOSSES}")
    object.__setattr__(
        config,
        "loss_scale",
        _finite_float(config.loss_scale, name="loss_scale", positive=True),
    )
    object.__setattr__(
        config,
        "max_iterations",
        _positive_integer(config.max_iterations, name="max_iterations"),
    )
    object.__setattr__(
        config,
        "relative_tolerance",
        _finite_float(
            config.relative_tolerance,
            name="relative_tolerance",
            positive=True,
        ),
    )
    object.__setattr__(
        config,
        "measurement_time_tolerance_s",
        _finite_float(
            config.measurement_time_tolerance_s,
            name="measurement_time_tolerance_s",
        ),
    )
    object.__setattr__(
        config,
        "process_position_floor_m",
        _finite_float(
            config.process_position_floor_m,
            name="process_position_floor_m",
        ),
    )
    object.__setattr__(
        config,
        "process_velocity_floor_mps",
        _finite_float(
            config.process_velocity_floor_mps,
            name="process_velocity_floor_mps",
        ),
    )
    object.__setattr__(
        config,
        "accepted_measurements_only",
        _boolean(
            config.accepted_measurements_only,
            name="accepted_measurements_only",
        ),
    )
    return config


def _robust_map_config_post_init(
    self: _robust_map.RobustMapSmootherConfig,
) -> None:
    """Validate newly constructed robust-MAP configuration objects."""

    _validate_robust_map_config_fields(self)


def _validated_robust_map_config(
    value: object,
    *,
    name: str,
) -> _robust_map.RobustMapSmootherConfig | None:
    """Reject malformed explicit configuration instead of replacing it by defaults."""

    if value is None:
        return None
    if not isinstance(value, _robust_map.RobustMapSmootherConfig):
        raise TypeError(f"{name} must be a RobustMapSmootherConfig or None")
    return _validate_robust_map_config_fields(value)


def _validate_record_chronology(records: list[dict[str, object]]) -> None:
    """Reject decreasing record times required to be sorted by search-based helpers.

    Equal timestamps remain valid because asynchronous trackers can emit multiple
    sequential sensor updates at the same physical time. Malformed timestamps are
    left to the maintained record parser so its established diagnostics are kept.
    """

    previous_time: float | None = None
    for record in records:
        if "time_s" not in record:
            return
        time_s = _optional_float(record["time_s"])
        if time_s is None:
            return
        if previous_time is not None and time_s < previous_time:
            raise ValueError(
                "records must be ordered by nondecreasing time_s for robust-map smoothing"
            )
        previous_time = time_s


@wraps(_ORIGINAL_ROBUST_MAP_SMOOTH_RECORDS)
def robust_map_smooth_records(
    records: list[dict[str, object]],
    *,
    measurements: Iterable[TrackingMeasurement] | None,
    acceleration_std_mps2: float,
    config: _robust_map.RobustMapSmootherConfig | None = None,
    lag_s: float | None = None,
) -> list[dict[str, object]]:
    """Run robust-MAP smoothing after validating controls and record chronology."""

    normalized_acceleration_std_mps2 = acceleration_std_mps2
    if records:
        normalized_acceleration_std_mps2 = _finite_float(
            acceleration_std_mps2,
            name="acceleration_std_mps2",
        )
    normalized_config = _validated_robust_map_config(config, name="config")
    normalized_lag_s = _validated_lag_s(lag_s)
    _validate_record_chronology(records)
    return _ORIGINAL_ROBUST_MAP_SMOOTH_RECORDS(
        records,
        measurements=measurements,
        acceleration_std_mps2=normalized_acceleration_std_mps2,
        config=normalized_config,
        lag_s=normalized_lag_s,
    )


@wraps(_ORIGINAL_SMOOTH_TRACKING_RECORDS)
def smooth_tracking_records(
    records: list[dict[str, object]],
    *,
    method: str,
    acceleration_std_mps2: float,
    lag_s: float | None = None,
    measurements: Iterable[TrackingMeasurement] | None = None,
    robust_map_config: _robust_map.RobustMapSmootherConfig | None = None,
) -> list[dict[str, object]]:
    """Validate robust-MAP controls before dispatching to either smoother."""

    normalized_acceleration_std_mps2 = acceleration_std_mps2
    if records and method in _smoothing.SMOOTHER_MODES and method != "none":
        normalized_acceleration_std_mps2 = _finite_float(
            acceleration_std_mps2,
            name="acceleration_std_mps2",
        )
    normalized_lag_s = lag_s
    if method in {"fixed-lag", "fixed-lag-map"}:
        normalized_lag_s = _validated_lag_s(lag_s)
        if normalized_lag_s is None:
            raise ValueError(f"{method} smoothing requires a nonnegative lag_s")
    normalized_config = robust_map_config
    if method in {"robust-map", "fixed-lag-map"}:
        normalized_config = _validated_robust_map_config(
            robust_map_config,
            name="robust_map_config",
        )
    return _ORIGINAL_SMOOTH_TRACKING_RECORDS(
        records,
        method=method,
        acceleration_std_mps2=normalized_acceleration_std_mps2,
        lag_s=normalized_lag_s,
        measurements=measurements,
        robust_map_config=normalized_config,
    )


def apply_robust_map_lag_validation_patch() -> None:
    """Install robust-MAP configuration, lag, and chronology validation."""

    _robust_map.RobustMapSmootherConfig.__post_init__ = _robust_map_config_post_init
    _robust_map.robust_map_smooth_records = robust_map_smooth_records
    _smoothing.robust_map_smooth_records = robust_map_smooth_records
    _smoothing.smooth_tracking_records = smooth_tracking_records


__all__ = [
    "apply_robust_map_lag_validation_patch",
    "robust_map_smooth_records",
    "smooth_tracking_records",
]
