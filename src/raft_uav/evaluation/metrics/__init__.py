"""Compatibility validation for trajectory-metric time gates.

The maintained implementation lives in the sibling ``metrics.py`` module. This
package preserves the public import path while rejecting invalid
``max_time_delta_s`` values before they can silently remove every metric sample
or disable the intended timestamp gate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "metrics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._metrics_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load evaluation metrics implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_POSITION_ERRORS_M = _IMPL.position_errors_m
_ORIGINAL_POSITION_ERRORS_AT_ESTIMATES_M = _IMPL.position_errors_at_estimates_m
_ORIGINAL_INTERPOLATE_POSITIONS_AT_TIMES = _IMPL.interpolate_positions_at_times


def _validate_max_time_delta_s(value: Any) -> float | None:
    """Return a finite non-negative time gate or raise explicitly."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_time_delta_s must be numeric or None") from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError("max_time_delta_s must be finite and non-negative")
    return number


def position_errors_m(
    estimate_times_s: np.ndarray,
    estimate_positions_m: np.ndarray,
    truth_times_s: np.ndarray,
    truth_positions_m: np.ndarray,
    max_time_delta_s: float | None = None,
    dimensions: int = 3,
) -> np.ndarray:
    """Compute truth-grid errors with a validated interpolation-support gate."""

    return _ORIGINAL_POSITION_ERRORS_M(
        estimate_times_s,
        estimate_positions_m,
        truth_times_s,
        truth_positions_m,
        max_time_delta_s=_validate_max_time_delta_s(max_time_delta_s),
        dimensions=dimensions,
    )


def position_errors_at_estimates_m(
    estimate_times_s: np.ndarray,
    estimate_positions_m: np.ndarray,
    truth_times_s: np.ndarray,
    truth_positions_m: np.ndarray,
    max_time_delta_s: float | None = None,
    dimensions: int = 3,
) -> np.ndarray:
    """Compute sample errors with a validated nearest-truth time gate."""

    return _ORIGINAL_POSITION_ERRORS_AT_ESTIMATES_M(
        estimate_times_s,
        estimate_positions_m,
        truth_times_s,
        truth_positions_m,
        max_time_delta_s=_validate_max_time_delta_s(max_time_delta_s),
        dimensions=dimensions,
    )


def interpolate_positions_at_times(
    reference_times_s: np.ndarray,
    reference_positions_m: np.ndarray,
    query_times_s: np.ndarray,
    *,
    max_time_delta_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate positions with a validated nearest-reference time gate."""

    return _ORIGINAL_INTERPOLATE_POSITIONS_AT_TIMES(
        reference_times_s,
        reference_positions_m,
        query_times_s,
        max_time_delta_s=_validate_max_time_delta_s(max_time_delta_s),
    )


_IMPL.position_errors_m = position_errors_m
_IMPL.position_errors_at_estimates_m = position_errors_at_estimates_m
_IMPL.interpolate_positions_at_times = interpolate_positions_at_times

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["position_errors_m"] = position_errors_m
globals()["position_errors_at_estimates_m"] = position_errors_at_estimates_m
globals()["interpolate_positions_at_times"] = interpolate_positions_at_times
globals()["_validate_max_time_delta_s"] = _validate_max_time_delta_s
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
