"""Validate CV tracker times before stateful operations mutate bootstrap state."""

from __future__ import annotations

from functools import wraps
from typing import Any

from raft_uav.numeric import optional_float

_PATCH_MARKER = "_raft_uav_kalman_time_state_validation_patch_applied"


def _finite_time_s(value: Any) -> float:
    """Return a finite real tracker timestamp or raise a stable error."""

    parsed = optional_float(value)
    if parsed is None:
        raise ValueError("time_s must be a finite real scalar")
    return parsed


def _reject_backward_time(tracker: Any, time_s: float) -> None:
    """Reject a timestamp that precedes the current tracker time."""

    if time_s < float(tracker.current_time_s) - 1.0e-9:
        raise ValueError("measurements must be processed in chronological order")


def apply_kalman_time_state_validation_patch(kalman_module: Any) -> None:
    """Fail before bootstrap state changes on malformed or backward timestamps."""

    if getattr(kalman_module, _PATCH_MARKER, False):
        return

    tracker_type = kalman_module.AsyncConstantVelocityKalmanTracker
    original_predict_to = tracker_type.predict_to
    original_coast_to = tracker_type.coast_to
    original_update = tracker_type.update

    @wraps(original_predict_to)
    def predict_to(self: Any, time_s: float) -> None:
        parsed = _finite_time_s(time_s)
        original_predict_to(self, parsed)

    @wraps(original_coast_to)
    def coast_to(self: Any, time_s: float) -> None:
        parsed = _finite_time_s(time_s)
        _reject_backward_time(self, parsed)
        original_coast_to(self, parsed)

    @wraps(original_update)
    def update(self: Any, measurement: Any, *args: Any, **kwargs: Any) -> Any:
        parsed = _finite_time_s(getattr(measurement, "time_s", None))
        _reject_backward_time(self, parsed)
        return original_update(self, measurement, *args, **kwargs)

    tracker_type.predict_to = predict_to
    tracker_type.coast_to = coast_to
    tracker_type.update = update
    setattr(kalman_module, _PATCH_MARKER, True)
