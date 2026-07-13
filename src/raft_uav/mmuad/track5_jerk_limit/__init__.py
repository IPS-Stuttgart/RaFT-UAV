"""Compatibility wrapper hardening the Track 5 jerk-limit repair.

The maintained implementation lives in the sibling ``track5_jerk_limit.py``
module. This package keeps the public import path while preserving jerk-window
row support and rejecting malformed ``iterations`` values instead of silently
truncating or clamping them.
"""

from __future__ import annotations

import importlib.util
import numbers
from pathlib import Path
import sys

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_jerk_limit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_jerk_limit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 jerk-limit implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_REPAIR = _IMPL.repair_track5_jerk_kinks


def _row_jerk_proxy_with_window_support(
    times: np.ndarray,
    xyz: np.ndarray,
) -> np.ndarray:
    """Map every valid jerk window to the rows in that window."""

    count = len(times)
    row_jerk = np.full(count, np.nan, dtype=float)
    d3 = _IMPL._third_derivative_matrix(times)
    if d3.size == 0:
        return row_jerk
    jerk_windows = d3 @ np.asarray(xyz, dtype=float)
    norms = np.linalg.norm(jerk_windows, axis=1)
    for coefficients, norm in zip(d3, norms, strict=True):
        for row_index in np.flatnonzero(coefficients):
            if np.isnan(row_jerk[row_index]) or norm > row_jerk[row_index]:
                row_jerk[row_index] = float(norm)
    return row_jerk


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer without lossy or Boolean coercion."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, numbers.Integral):
        integer = int(value)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} must be a positive integer")
        integer = int(numeric)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def repair_track5_jerk_kinks(
    submission,
    *,
    max_jerk_mps3: float = 80.0,
    smoothness_weight: float = 10.0,
    min_correction_m: float = 1.0,
    max_correction_m: float | None = None,
    iterations: int = 1,
    repair_blend: float = 1.0,
):
    """Return jerk-limited estimates after validating ``iterations``."""

    validated_iterations = _positive_integer(iterations, name="iterations")
    return _ORIGINAL_REPAIR(
        submission,
        max_jerk_mps3=max_jerk_mps3,
        smoothness_weight=smoothness_weight,
        min_correction_m=min_correction_m,
        max_correction_m=max_correction_m,
        iterations=validated_iterations,
        repair_blend=repair_blend,
    )


_IMPL._row_jerk_proxy = _row_jerk_proxy_with_window_support
_IMPL.repair_track5_jerk_kinks = repair_track5_jerk_kinks

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_row_jerk_proxy_with_window_support"] = _row_jerk_proxy_with_window_support
globals()["_positive_integer"] = _positive_integer
globals()["repair_track5_jerk_kinks"] = repair_track5_jerk_kinks

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
