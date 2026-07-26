"""Validate smoothing lag horizons before fixed-lag window construction."""

from __future__ import annotations

from collections.abc import Iterable
from functools import wraps

import numpy as np

from raft_uav.baselines import robust_map as _robust_map
from raft_uav.baselines import smoothing as _smoothing
from raft_uav.baselines.kalman import TrackingMeasurement

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


@wraps(_ORIGINAL_ROBUST_MAP_SMOOTH_RECORDS)
def robust_map_smooth_records(
    records: list[dict[str, object]],
    *,
    measurements: Iterable[TrackingMeasurement] | None,
    acceleration_std_mps2: float,
    config: _robust_map.RobustMapSmootherConfig | None = None,
    lag_s: float | None = None,
) -> list[dict[str, object]]:
    """Run robust-MAP smoothing after validating the optional lag horizon."""

    return _ORIGINAL_ROBUST_MAP_SMOOTH_RECORDS(
        records,
        measurements=measurements,
        acceleration_std_mps2=acceleration_std_mps2,
        config=config,
        lag_s=_validated_lag_s(lag_s),
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
    """Validate fixed-lag horizons before dispatching to either smoother."""

    normalized_lag_s = lag_s
    if method in {"fixed-lag", "fixed-lag-map"}:
        normalized_lag_s = _validated_lag_s(lag_s)
        if normalized_lag_s is None:
            raise ValueError(f"{method} smoothing requires a nonnegative lag_s")
    return _ORIGINAL_SMOOTH_TRACKING_RECORDS(
        records,
        method=method,
        acceleration_std_mps2=acceleration_std_mps2,
        lag_s=normalized_lag_s,
        measurements=measurements,
        robust_map_config=robust_map_config,
    )


def apply_robust_map_lag_validation_patch() -> None:
    """Install lag validation at direct and generic smoothing entry points."""

    _robust_map.robust_map_smooth_records = robust_map_smooth_records
    _smoothing.robust_map_smooth_records = robust_map_smooth_records
    _smoothing.smooth_tracking_records = smooth_tracking_records


__all__ = [
    "apply_robust_map_lag_validation_patch",
    "robust_map_smooth_records",
    "smooth_tracking_records",
]
