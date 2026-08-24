"""Compatibility package for the legacy :mod:`native_ros` module.

The implementation historically lives in ``native_ros.py``.  This package
loads that module unchanged and installs strict validation for ROS
``sec``/``nanosec`` timestamp pairs so malformed fractional components cannot
silently become plausible timestamps.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any


_LEGACY_MODULE_PATH = Path(__file__).resolve().parent.parent / "native_ros.py"
_LEGACY_SPEC = importlib.util.spec_from_file_location(
    f"{__name__}._legacy",
    _LEGACY_MODULE_PATH,
)
if _LEGACY_SPEC is None or _LEGACY_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"unable to load native_ros implementation at {_LEGACY_MODULE_PATH}")

_IMPL = importlib.util.module_from_spec(_LEGACY_SPEC)
sys.modules[_LEGACY_SPEC.name] = _IMPL
try:
    _LEGACY_SPEC.loader.exec_module(_IMPL)
except Exception:
    sys.modules.pop(_LEGACY_SPEC.name, None)
    raise

_ORIGINAL_TIME_FIELD_TO_S = _IMPL._time_field_to_s
_SECONDS_KEYS = ("sec", "secs", "seconds")
_NANOSECONDS_KEYS = ("nanosec", "nsec", "nsecs", "nanoseconds")


def _finite_float(value: object) -> float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _time_field_to_s(value: Any) -> float | None:
    """Convert a scalar or ROS stamp to finite seconds.

    ROS ``nanosec`` is an integer fractional component, not an unrestricted
    duration.  Reject non-finite, Boolean, fractional, negative, and
    one-billion-or-larger values instead of carrying or borrowing seconds.
    """

    if isinstance(value, bool):
        return None

    seconds_raw = _IMPL._field_value(value, *_SECONDS_KEYS)
    if seconds_raw is None:
        return _ORIGINAL_TIME_FIELD_TO_S(value)

    seconds = _finite_float(seconds_raw)
    if seconds is None:
        return None

    nanoseconds_raw = _IMPL._field_value(value, *_NANOSECONDS_KEYS)
    if nanoseconds_raw in (None, ""):
        nanoseconds = 0.0
    else:
        nanoseconds = _finite_float(nanoseconds_raw)
        if nanoseconds is None or not nanoseconds.is_integer():
            return None
        if not 0.0 <= nanoseconds < 1_000_000_000.0:
            return None

    timestamp = seconds + nanoseconds * 1.0e-9
    return timestamp if math.isfinite(timestamp) else None


_IMPL._time_field_to_s = _time_field_to_s

globals().update(
    {
        name: value
        for name, value in vars(_IMPL).items()
        if not (name.startswith("__") and name not in {"__all__"})
    }
)
globals()["_time_field_to_s"] = _time_field_to_s
