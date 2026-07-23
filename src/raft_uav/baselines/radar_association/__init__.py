"""Compatibility validation for radar-association numeric controls.

The maintained implementation lives in the sibling ``radar_association.py``
module. This package preserves the public import path while rejecting non-finite
numeric parameters and malformed integer controls before they can create NaN/Inf
tracker state, covariance values, or silently truncated association settings, and
initializes track-bank state from supported position-plus-velocity bootstrap
measurements without a shape mismatch.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np

from raft_uav.numeric import optional_int

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
_POSITIVE_FINITE_ASSOCIATION_PARAMETERS = (
    "track_switch_nis_ratio",
    "geometry_velocity_std_mps",
    "rf_anchor_nis_cap",
    "rf_anchor_gate_nis",
    "pda_nis_temperature",
    "track_bank_clutter_intensity",
    "track_bank_prune_log_weight_delta",
    "stable_segment_max_transition_speed_mps",
    "stable_segment_interpolation_std_scale",
    "stable_segment_rf_nis_cap",
)
_NONNEGATIVE_FINITE_ASSOCIATION_PARAMETERS = (
    "geometry_velocity_weight",
    "geometry_switch_penalty",
    "geometry_catprob_weight",
    "rf_anchor_weight",
    "rf_anchor_time_gate_s",
    "pda_catprob_exponent",
    "stable_segment_interpolation_gap_std_mps",
    "stable_segment_rf_score_weight",
    "stable_segment_rf_time_gate_s",
)
_OPTIONAL_POSITIVE_FINITE_ASSOCIATION_PARAMETERS = (
    "stable_segment_range_gate_m",
    "stable_segment_interpolation_max_gap_s",
    "stable_segment_interpolation_max_speed_mps",
)
_OPTIONAL_UNIT_INTERVAL_ASSOCIATION_PARAMETERS = (
    "candidate_catprob_threshold",
    "paper_compatible_catprob_threshold",
)
_POSITIVE_INTEGER_ASSOCIATION_PARAMETERS = (
    "track_bank_max_hypotheses",
    "track_bank_max_assignments",
    "track_bank_max_candidates",
    "stable_segment_min_frames",
)


class _RadarAssociationModule(ModuleType):
    """Module proxy that keeps runtime monkeypatches visible to legacy globals."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "_IMPL":
            return
        implementation = self.__dict__.get("_IMPL")
        if implementation is not None and hasattr(implementation, name):
            setattr(implementation, name, value)


def run_async_cv_baseline_with_radar_association(*args: Any, **kwargs: Any) -> Any:
    """Run radar association after validating and normalizing numeric controls."""

    bound = _RUN_SIGNATURE.bind(*args, **kwargs)
    bound.apply_defaults()
    _validate_radar_association_parameters(bound.arguments)
    return _ORIGINAL_RUN_ASYNC_CV_BASELINE_WITH_RADAR_ASSOCIATION(
        *bound.args,
        **bound.kwargs,
    )


def _initial_mht_tracker(
    initial_measurement: Any,
    *,
    max_global_hypotheses: int,
    max_assignments_per_hypothesis: int,
    max_candidates_per_track: int,
    gate_probability: float,
    detection_probability: float,
    clutter_intensity: float,
    prune_log_weight_delta: float,
) -> Any:
    """Initialize the MHT from supported position or position/velocity samples."""

    measurement = np.asarray(initial_measurement.vector, dtype=float).reshape(-1)
    state = np.zeros(6, dtype=float)
    if measurement.size == 2:
        state[:2] = measurement
    elif measurement.size == 3:
        state[:3] = measurement
    elif measurement.size == 6:
        state[:] = measurement
    else:
        raise ValueError("initial measurement must contain 2, 3, or 6 elements")

    state_covariance = np.diag(
        [50.0**2, 50.0**2, 50.0**2, 15.0**2, 15.0**2, 15.0**2]
    )
    return _IMPL.MultiHypothesisTracker(
        initial_prior=[_IMPL.KalmanFilter((state, state_covariance))],
        association_param={
            "gating_probability": float(gate_probability),
            "detection_probability": float(detection_probability),
            "clutter_intensity": float(clutter_intensity),
            "max_global_hypotheses": int(max_global_hypotheses),
            "max_hypotheses_per_global_hypothesis": int(
                max_assignments_per_hypothesis
            ),
            "max_measurements_per_track": int(max_candidates_per_track),
            "prune_log_weight_delta": float(prune_log_weight_delta),
        },
        log_prior_estimates=False,
        log_posterior_estimates=False,
    )


def _validate_radar_association_parameters(arguments: dict[str, Any]) -> None:
    for name in _POSITIVE_FINITE_RADAR_COVARIANCE_PARAMETERS:
        _require_finite_positive(name, arguments[name])
    for name in _NONNEGATIVE_FINITE_RADAR_COVARIANCE_PARAMETERS:
        _require_finite_nonnegative(name, arguments[name])
    for name in _POSITIVE_FINITE_ASSOCIATION_PARAMETERS:
        _require_finite_positive(name, arguments[name])
    for name in _NONNEGATIVE_FINITE_ASSOCIATION_PARAMETERS:
        _require_finite_nonnegative(name, arguments[name])
    for name in _OPTIONAL_POSITIVE_FINITE_ASSOCIATION_PARAMETERS:
        value = arguments[name]
        if value is not None:
            _require_finite_positive(name, value)
    for name in _OPTIONAL_UNIT_INTERVAL_ASSOCIATION_PARAMETERS:
        value = arguments[name]
        if value is not None:
            _require_finite_unit_interval(name, value)
    for name in _POSITIVE_INTEGER_ASSOCIATION_PARAMETERS:
        arguments[name] = _require_positive_integer(name, arguments[name])

    crossrange_min = float(arguments["radar_crossrange_min_std_m"])
    crossrange_max = float(arguments["radar_crossrange_max_std_m"])
    if crossrange_max < crossrange_min:
        raise ValueError("radar_crossrange_max_std_m must be >= radar_crossrange_min_std_m")


def _validate_radar_covariance_parameters(arguments: dict[str, Any]) -> None:
    """Backward-compatible alias for the expanded numeric validation."""

    _validate_radar_association_parameters(arguments)


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


def _require_finite_unit_interval(name: str, value: Any) -> float:
    number = _finite_float(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _require_positive_integer(name: str, value: Any) -> int:
    number = optional_int(value)
    if number is None or number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return number


_IMPL.run_async_cv_baseline_with_radar_association = (
    run_async_cv_baseline_with_radar_association
)
_IMPL._initial_mht_tracker = _initial_mht_tracker

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
globals()["_initial_mht_tracker"] = _initial_mht_tracker
globals()["_validate_radar_association_parameters"] = (
    _validate_radar_association_parameters
)
globals()["_validate_radar_covariance_parameters"] = _validate_radar_covariance_parameters
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
sys.modules[__name__].__class__ = _RadarAssociationModule
