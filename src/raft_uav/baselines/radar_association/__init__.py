"""Compatibility validation for radar-association covariance controls.

The maintained implementation lives in the sibling ``radar_association.py``
module. This package preserves the public import path while rejecting non-finite
radar covariance parameters before they can create NaN/Inf measurement
covariances.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "radar_association.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._radar_association_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load radar association implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_RUN_ASYNC_CV_BASELINE_WITH_RADAR_ASSOCIATION = (
    _IMPL.run_async_cv_baseline_with_radar_association
)
_RUN_SIGNATURE = inspect.signature(
    _ORIGINAL_RUN_ASYNC_CV_BASELINE_WITH_RADAR_ASSOCIATION
)

_POSITIVE_FINITE_RADAR_COVARIANCE_PARAMETERS = (
    "radar_xy_std_m",
    "radar_z_std_m",
    "radar_range_std_m",
    "radar_crossrange_angle_std_deg",
    "radar_crossrange_min_std_m",
    "radar_crossrange_max_std_m",
)
_NONNEGATIVE_FINITE_RADAR_COVARIANCE_PARAMETERS = (
    "radar_range_std_fraction",
)


def run_async_cv_baseline_with_radar_association(*args: Any, **kwargs: Any) -> Any:
    """Run radar association after validating covariance-scale inputs."""

    bound = _RUN_SIGNATURE.bind(*args, **kwargs)
    bound.apply_defaults()
    _validate_radar_covariance_parameters(bound.arguments)
    return _ORIGINAL_RUN_ASYNC_CV_BASELINE_WITH_RADAR_ASSOCIATION(*args, **kwargs)


def _validate_radar_covariance_parameters(arguments: dict[str, Any]) -> None:
    for name in _POSITIVE_FINITE_RADAR_COVARIANCE_PARAMETERS:
        _require_finite_positive(name, arguments[name])
    for name in _NONNEGATIVE_FINITE_RADAR_COVARIANCE_PARAMETERS:
        _require_finite_nonnegative(name, arguments[name])

    crossrange_min = float(arguments["radar_crossrange_min_std_m"])
    crossrange_max = float(arguments["radar_crossrange_max_std_m"])
    if crossrange_max < crossrange_min:
        raise ValueError("radar_crossrange_max_std_m must be >= radar_crossrange_min_std_m")


def _finite_float(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_finite_positive(name: str, value: Any) -> float:
    number = _finite_float(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _require_finite_nonnegative(name: str, value: Any) -> float:
    number = _finite_float(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


_IMPL.run_async_cv_baseline_with_radar_association = run_async_cv_baseline_with_radar_association

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["run_async_cv_baseline_with_radar_association"] = (
    run_async_cv_baseline_with_radar_association
)
globals()["_validate_radar_covariance_parameters"] = _validate_radar_covariance_parameters
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
