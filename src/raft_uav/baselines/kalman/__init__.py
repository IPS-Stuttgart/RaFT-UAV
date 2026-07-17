"""Compatibility package validating Kalman measurement timestamps.

The maintained implementation lives in the sibling ``kalman.py`` module. This
package preserves the public import path while rejecting malformed timestamps
before they can enter asynchronous tracker chronology.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from raft_uav.numeric import optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "kalman.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._kalman_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load Kalman implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TRACKING_MEASUREMENT_POST_INIT = _IMPL.TrackingMeasurement.__post_init__


def _validated_tracking_measurement_post_init(
    self: Any,
    _apply_runtime_calibration: bool,
) -> None:
    """Reject non-finite and non-scalar timestamps before tracker ingestion."""

    time_s = optional_float(self.time_s)
    if time_s is None:
        raise ValueError("measurement time_s must be a finite real scalar")
    object.__setattr__(self, "time_s", time_s)
    _ORIGINAL_TRACKING_MEASUREMENT_POST_INIT(self, _apply_runtime_calibration)


_IMPL.TrackingMeasurement.__post_init__ = _validated_tracking_measurement_post_init

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_tracking_measurement_post_init"] = (
    _validated_tracking_measurement_post_init
)

__doc__ = _IMPL.__doc__
__all__ = list(getattr(_IMPL, "__all__", ()))
