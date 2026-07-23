"""Compatibility wrapper with symmetric metric timestamp tolerance.

The maintained implementation lives in the sibling ``metrics.py`` module. This
package preserves the public import path while ensuring that timestamps within
the existing 1 ns equality tolerance are accepted at either interpolation
endpoint, regardless of whether a maximum time-delta gate is configured. The
same endpoint rule is applied to both truth-grid metrics and paper-table
interpolation at estimate timestamps. Non-finite nearest-time queries and
masked time-gate controls are rejected instead of being silently assigned or
unwrapped.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "metrics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._metrics_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load metrics implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ENDPOINT_ATOL_S = 1.0e-9
_ORIGINAL_NEAREST_TIME_INDICES = _IMPL.nearest_time_indices
_ORIGINAL_VALIDATE_MAX_TIME_DELTA_S = _IMPL._validate_max_time_delta_s
_ORIGINAL_INTERPOLATE_POSITIONS_AT_TIMES = _IMPL.interpolate_positions_at_times


def _validate_max_time_delta_s_without_masked(value: object) -> float | None:
    """Reject masked scalar gates before NumPy exposes their hidden payload."""

    if value is not None and np.ma.is_masked(value):
        raise ValueError("max_time_delta_s must be a finite, non-negative scalar")
    return _ORIGINAL_VALIDATE_MAX_TIME_DELTA_S(value)


def _nearest_time_indices_with_finite_queries(
    reference_times_s: np.ndarray,
    query_times_s: np.ndarray,
) -> np.ndarray:
    """Reject invalid query timestamps before nearest-neighbor assignment."""

    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    if not np.isfinite(query).all():
        raise ValueError("query_times_s must contain only finite timestamps")
    return _ORIGINAL_NEAREST_TIME_INDICES(reference_times_s, query)


def _truth_grid_with_symmetric_tolerance(
    estimate_times: np.ndarray,
    truth_times: np.ndarray,
    truth_positions: np.ndarray,
    *,
    max_time_delta_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep tolerance-equivalent truth samples at either bracket endpoint."""

    supported = (truth_times >= estimate_times[0]) & (
        truth_times <= estimate_times[-1]
    )
    supported |= np.isclose(
        truth_times,
        estimate_times[0],
        rtol=0.0,
        atol=_ENDPOINT_ATOL_S,
    )
    supported |= np.isclose(
        truth_times,
        estimate_times[-1],
        rtol=0.0,
        atol=_ENDPOINT_ATOL_S,
    )
    query_times = truth_times[supported]
    query_truth_positions = truth_positions[supported]
    if query_times.size == 0 or max_time_delta_s is None:
        return query_times, query_truth_positions

    max_delta = float(max_time_delta_s)
    right = np.searchsorted(estimate_times, query_times, side="left")
    right = np.clip(right, 0, estimate_times.size - 1)
    left = np.clip(right - 1, 0, estimate_times.size - 1)

    left_exact = np.isclose(
        estimate_times[left],
        query_times,
        rtol=0.0,
        atol=_ENDPOINT_ATOL_S,
    )
    right_exact = np.isclose(
        estimate_times[right],
        query_times,
        rtol=0.0,
        atol=_ENDPOINT_ATOL_S,
    )
    left_delta = np.abs(query_times - estimate_times[left])
    right_delta = np.abs(estimate_times[right] - query_times)
    close_to_bracket = (left_delta <= max_delta) & (right_delta <= max_delta)
    keep = left_exact | right_exact | close_to_bracket
    return query_times[keep], query_truth_positions[keep]


def _interpolate_positions_at_times_with_symmetric_tolerance(
    reference_times_s: np.ndarray,
    reference_positions_m: np.ndarray,
    query_times_s: np.ndarray,
    *,
    max_time_delta_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Accept tolerance-equivalent queries at either trajectory endpoint."""

    interpolated, valid = _ORIGINAL_INTERPOLATE_POSITIONS_AT_TIMES(
        reference_times_s,
        reference_positions_m,
        query_times_s,
        max_time_delta_s=max_time_delta_s,
    )

    reference_array = np.asarray(reference_positions_m, dtype=float)
    reference_dimensions = (
        reference_array.shape[1] if reference_array.ndim == 2 else 3
    )
    reference_times, _ = _IMPL._prepare_time_position_series(
        reference_times_s,
        reference_array,
        dimensions=reference_dimensions,
    )
    if reference_times.size == 0:
        return interpolated, valid

    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    endpoint_equivalent = np.isfinite(query) & (
        np.isclose(
            query,
            reference_times[0],
            rtol=0.0,
            atol=_ENDPOINT_ATOL_S,
        )
        | np.isclose(
            query,
            reference_times[-1],
            rtol=0.0,
            atol=_ENDPOINT_ATOL_S,
        )
    )
    return interpolated, valid | endpoint_equivalent


_IMPL._validate_max_time_delta_s = _validate_max_time_delta_s_without_masked
_IMPL.nearest_time_indices = _nearest_time_indices_with_finite_queries
_IMPL._truth_grid_with_estimate_support = _truth_grid_with_symmetric_tolerance
_IMPL.interpolate_positions_at_times = (
    _interpolate_positions_at_times_with_symmetric_tolerance
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
