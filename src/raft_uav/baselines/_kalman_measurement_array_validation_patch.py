"""Reject masked and complex arrays at the Kalman measurement boundary."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

_PATCH_MARKER = "_raft_uav_rejects_lossy_measurement_arrays"


def _validate_unmasked_real_array(value: Any, *, name: str) -> None:
    """Reject arrays whose masks or imaginary parts would be lost on coercion."""

    error = f"{name} must contain only unmasked real values"
    if np.ma.is_masked(value):
        raise ValueError(error)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if np.iscomplexobj(array):
        raise ValueError(error)
    if array.dtype == object and any(
        np.ma.is_masked(item) or np.iscomplexobj(item) for item in array.flat
    ):
        raise ValueError(error)


def apply_kalman_measurement_array_validation_patch(module: ModuleType) -> None:
    """Patch ``TrackingMeasurement`` before dependent baseline modules import it."""

    original = module.TrackingMeasurement
    if getattr(original, _PATCH_MARKER, False):
        return

    class TrackingMeasurement(original):
        def __post_init__(self, *args: Any, **kwargs: Any) -> None:
            _validate_unmasked_real_array(self.vector, name="measurement vector")
            _validate_unmasked_real_array(
                self.covariance,
                name="measurement covariance",
            )
            super().__post_init__(*args, **kwargs)
            _validate_unmasked_real_array(self.vector, name="measurement vector")
            _validate_unmasked_real_array(
                self.covariance,
                name="measurement covariance",
            )

    TrackingMeasurement.__name__ = original.__name__
    TrackingMeasurement.__qualname__ = original.__qualname__
    TrackingMeasurement.__module__ = original.__module__
    TrackingMeasurement.__doc__ = original.__doc__
    setattr(TrackingMeasurement, _PATCH_MARKER, True)
    module.TrackingMeasurement = TrackingMeasurement
