"""Compatibility wrapper with symmetric metric timestamp tolerance.

The maintained implementation lives in the sibling ``metrics.py`` module. This
package preserves the public import path while ensuring that timestamps within
the existing 1 ns equality tolerance are accepted at either interpolation
endpoint, regardless of whether a maximum time-delta gate is configured. The
same endpoint rule is applied to both truth-grid metrics and paper-table
interpolation at estimate timestamps. Non-finite and masked nearest-time queries
are rejected, masked reference timestamps are ignored, masked trajectory and
error samples are excluded, masked interpolation queries are returned as invalid,
and complex trajectory and error values are rejected before NumPy can discard
their imaginary parts.
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
_ORIGINAL_PREPARE_TIME_POSITION_SAMPLES = _IMPL._prepare_time_position_samples
_ORIGINAL_INTERPOLATE_POSITIONS_AT_TIMES = _IMPL.interpolate_positions_at_times
_ORIGINAL_SUMMARIZE_ERRORS = _IMPL.summarize_errors


def _reject_complex_values(value: object, *, name: str) -> None:
    """Reject complex payloads before float conversion can discard information."""

    masked = np.ma.asarray(value)
    if np.iscomplexobj(masked):
        raise ValueError(f"{name} must contain only real values")
    if masked.dtype != object:
        return
    for item in masked.compressed().reshape(-1):
        if np.iscomplexobj(np.asanyarray(item)):
            raise ValueError(f"{name} must contain only real values")


def _validate_max_time_delta_s_without_masked(value: object) -> float | None:
    """Reject masked scalar gates before NumPy exposes their hidden payload."""

    if value is not None and np.ma.is_masked(value):
        raise ValueError("max_time_delta_s must be a finite, non-negative scalar")
    return _ORIGINAL_VALIDATE_MAX_TIME_DELTA_S(value)


def _prepare_time_position_samples_without_masked(
    times_s: np.ndarray,
    positions_m: np.ndarray,
    *,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Treat masked trajectory values like non-finite samples."""

    _reject_complex_values(times_s, name="times_s")
    _reject_complex_values(positions_m, name="positions_m")
    masked_times = np.ma.asarray(times_s, dtype=float)
    masked_positions = np.ma.asarray(positions_m, dtype=float)
    times = np.asarray(masked_times.filled(np.nan), dtype=float)
    positions = np.asarray(masked_positions.filled(np.nan), dtype=float)
    return _ORIGINAL_PREPARE_TIME_POSITION_SAMPLES(
        times,
        positions,
        dimensions=dimensions,
    )


def _nearest_time_indices_with_finite_queries(
    reference_times_s: np.ndarray,
    query_times_s: np.ndarray,
) -> np.ndarray:
    """Ignore masked references and reject invalid nearest-time queries."""

    _reject_complex_values(reference_times_s, name="reference_times_s")
    _reject_complex_values(query_times_s, name="query_times_s")
    reference_masked = np.ma.asarray(reference_times_s, dtype=float).reshape(-1)
    reference = np.asarray(reference_masked.filled(np.nan), dtype=float)

    query_masked = np.ma.asarray(query_times_s, dtype=float).reshape(-1)
    if bool(np.ma.getmaskarray(query_masked).any()):
        raise ValueError("query_times_s must contain only finite timestamps")
    query = np.asarray(query_masked.filled(np.nan), dtype=float)
    if not np.isfinite(query).all():
        raise ValueError("query_times_s must contain only finite timestamps")
    return _ORIGINAL_NEAREST_TIME_INDICES(reference, query)


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
    """Accept endpoint-equivalent queries while preserving input masks."""

    _reject_complex_values(reference_times_s, name="reference_times_s")
    _reject_complex_values(reference_positions_m, name="reference_positions_m")
    _reject_complex_values(query_times_s, name="query_times_s")
    masked_reference_times = np.ma.asarray(reference_times_s, dtype=float)
    masked_reference_positions = np.ma.asarray(reference_positions_m, dtype=float)
    masked_query = np.ma.asarray(query_times_s, dtype=float).reshape(-1)
    reference_times = np.asarray(masked_reference_times.filled(np.nan), dtype=float)
    reference_positions = np.asarray(
        masked_reference_positions.filled(np.nan),
        dtype=float,
    )
    query = np.asarray(masked_query.filled(np.nan), dtype=float)

    interpolated, valid = _ORIGINAL_INTERPOLATE_POSITIONS_AT_TIMES(
        reference_times,
        reference_positions,
        query,
        max_time_delta_s=max_time_delta_s,
    )

    reference_dimensions = (
        reference_positions.shape[1] if reference_positions.ndim == 2 else 3
    )
    prepared_reference_times, _ = _IMPL._prepare_time_position_series(
        reference_times,
        reference_positions,
        dimensions=reference_dimensions,
    )
    if prepared_reference_times.size == 0:
        return interpolated, valid

    endpoint_equivalent = np.isfinite(query) & (
        np.isclose(
            query,
            prepared_reference_times[0],
            rtol=0.0,
            atol=_ENDPOINT_ATOL_S,
        )
        | np.isclose(
            query,
            prepared_reference_times[-1],
            rtol=0.0,
            atol=_ENDPOINT_ATOL_S,
        )
    )
    return interpolated, valid | endpoint_equivalent


def _summarize_errors_without_masked(
    errors_m: np.ndarray,
) -> dict[str, float | None]:
    """Exclude masked errors and reject complex values before summarization."""

    _reject_complex_values(errors_m, name="errors_m")
    masked_errors = np.ma.asarray(errors_m, dtype=float).reshape(-1)
    errors = np.asarray(masked_errors.filled(np.nan), dtype=float)
    return _ORIGINAL_SUMMARIZE_ERRORS(errors)


_IMPL._validate_max_time_delta_s = _validate_max_time_delta_s_without_masked
_IMPL._prepare_time_position_samples = _prepare_time_position_samples_without_masked
_IMPL.nearest_time_indices = _nearest_time_indices_with_finite_queries
_IMPL._truth_grid_with_estimate_support = _truth_grid_with_symmetric_tolerance
_IMPL.interpolate_positions_at_times = (
    _interpolate_positions_at_times_with_symmetric_tolerance
)
_IMPL.summarize_errors = _summarize_errors_without_masked

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
