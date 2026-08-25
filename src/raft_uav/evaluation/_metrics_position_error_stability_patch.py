"""Keep shared position-error calculations stable for finite inputs."""

from __future__ import annotations

from importlib import import_module

import numpy as np


_metrics = import_module("raft_uav.evaluation.metrics")
_IMPL = getattr(_metrics, "_IMPL", _metrics)
_PATCH_MARKER = "_stable_position_error_norms_patch_applied"

_ORIGINAL_NEAREST_TIME_INDICES = _metrics.nearest_time_indices
_ORIGINAL_TRUTH_GRID_WITH_ESTIMATE_SUPPORT = _IMPL._truth_grid_with_estimate_support
_ORIGINAL_INTERPOLATE_POSITIONS_AT_TIMES = _IMPL.interpolate_positions_at_times


def nearest_time_indices(
    reference_times_s: np.ndarray,
    query_times_s: np.ndarray,
) -> np.ndarray:
    """Return nearest timestamps without exposing benign subtraction overflow."""

    # The inputs are validated as finite by the active metrics wrapper.  A
    # distance to the farther bracketing sample can nevertheless exceed the
    # float64 range while the nearer distance remains representable.  Infinity
    # is the correct comparison sentinel in that case; it must not leak the
    # caller's global ``np.seterr(over="raise")`` policy as an exception.
    with np.errstate(over="ignore", invalid="ignore"):
        return _ORIGINAL_NEAREST_TIME_INDICES(
            reference_times_s,
            query_times_s,
        )


def _truth_grid_with_estimate_support(
    estimate_times: np.ndarray,
    truth_times: np.ndarray,
    truth_positions: np.ndarray,
    *,
    max_time_delta_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply truth-grid support gates without overflowing finite time deltas."""

    with np.errstate(over="ignore", invalid="ignore"):
        return _ORIGINAL_TRUTH_GRID_WITH_ESTIMATE_SUPPORT(
            estimate_times,
            truth_times,
            truth_positions,
            max_time_delta_s=max_time_delta_s,
        )


def interpolate_positions_at_times(
    reference_times_s: np.ndarray,
    reference_positions_m: np.ndarray,
    query_times_s: np.ndarray,
    *,
    max_time_delta_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate while treating unrepresentable finite time gaps as infinite."""

    with np.errstate(over="ignore", invalid="ignore"):
        return _ORIGINAL_INTERPOLATE_POSITIONS_AT_TIMES(
            reference_times_s,
            reference_positions_m,
            query_times_s,
            max_time_delta_s=max_time_delta_s,
        )


def _stable_euclidean_norms(deltas: np.ndarray) -> np.ndarray:
    """Return row-wise Euclidean norms without squaring unscaled magnitudes."""

    matrix = np.asarray(deltas, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("deltas must be a 2D array")
    return np.hypot.reduce(np.abs(matrix), axis=1)


def _single_sample_position_errors_m(
    estimate_times: np.ndarray,
    estimate_positions: np.ndarray,
    truth_times: np.ndarray,
    truth_positions: np.ndarray,
    *,
    max_time_delta_s: float | None,
    dimensions: int,
) -> np.ndarray:
    """Evaluate a singleton estimate without overflowing its Euclidean error."""

    query_times, query_truth_positions = _IMPL._truth_grid_with_estimate_support(
        estimate_times,
        truth_times,
        truth_positions,
        max_time_delta_s=max_time_delta_s,
    )
    if query_times.size == 0:
        return np.array([], dtype=float)

    query_estimate_positions = _IMPL._interpolate_positions(
        estimate_times,
        estimate_positions,
        query_times,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        deltas = (
            query_estimate_positions[:, :dimensions]
            - query_truth_positions[:, :dimensions]
        )
    errors = _stable_euclidean_norms(deltas)
    return errors[np.isfinite(errors)]


def position_errors_m(
    estimate_times_s: np.ndarray,
    estimate_positions_m: np.ndarray,
    truth_times_s: np.ndarray,
    truth_positions_m: np.ndarray,
    max_time_delta_s: float | None = None,
    dimensions: int = 3,
) -> np.ndarray:
    """Compute truth-grid errors without dropping large representable norms."""

    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    max_time_delta_s = _IMPL._validate_max_time_delta_s(max_time_delta_s)

    estimate_times, estimate_positions = _IMPL._prepare_time_position_series(
        estimate_times_s,
        estimate_positions_m,
        dimensions=dimensions,
    )
    truth_times, truth_positions = _IMPL._prepare_time_position_series(
        truth_times_s,
        truth_positions_m,
        dimensions=dimensions,
    )
    if estimate_times.size == 0 or truth_times.size == 0:
        return np.array([], dtype=float)
    if estimate_times.size == 1:
        return _IMPL._single_sample_position_errors_m(
            estimate_times,
            estimate_positions,
            truth_times,
            truth_positions,
            max_time_delta_s=max_time_delta_s,
            dimensions=dimensions,
        )

    query_times, query_truth_positions = _IMPL._truth_grid_with_estimate_support(
        estimate_times,
        truth_times,
        truth_positions,
        max_time_delta_s=max_time_delta_s,
    )
    if query_times.size == 0:
        return np.array([], dtype=float)

    query_estimate_positions = _IMPL._interpolate_positions(
        estimate_times,
        estimate_positions,
        query_times,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        deltas = (
            query_estimate_positions[:, :dimensions]
            - query_truth_positions[:, :dimensions]
        )
    errors = _stable_euclidean_norms(deltas)
    return errors[np.isfinite(errors)]


def position_errors_at_estimates_m(
    estimate_times_s: np.ndarray,
    estimate_positions_m: np.ndarray,
    truth_times_s: np.ndarray,
    truth_positions_m: np.ndarray,
    max_time_delta_s: float | None = None,
    dimensions: int = 3,
) -> np.ndarray:
    """Compute nearest-truth sample errors with overflow-stable norms."""

    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    max_time_delta_s = _IMPL._validate_max_time_delta_s(max_time_delta_s)

    estimate_times, estimate_positions = _IMPL._prepare_time_position_samples(
        estimate_times_s,
        estimate_positions_m,
        dimensions=dimensions,
    )
    truth_times, truth_positions = _IMPL._prepare_time_position_series(
        truth_times_s,
        truth_positions_m,
        dimensions=dimensions,
    )
    if estimate_times.size == 0 or truth_times.size == 0:
        return np.array([], dtype=float)

    truth_indices = _IMPL.nearest_time_indices(truth_times, estimate_times)
    keep = np.ones(estimate_times.size, dtype=bool)
    if max_time_delta_s is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            time_deltas = np.abs(truth_times[truth_indices] - estimate_times)
        keep &= time_deltas <= float(max_time_delta_s)
    if not bool(keep.any()):
        return np.array([], dtype=float)

    with np.errstate(over="ignore", invalid="ignore"):
        deltas = (
            estimate_positions[keep, :dimensions]
            - truth_positions[truth_indices[keep], :dimensions]
        )
    errors = _stable_euclidean_norms(deltas)
    return errors[np.isfinite(errors)]


def position_errors_at_times_m(
    estimate_times_s: np.ndarray,
    estimate_positions_m: np.ndarray,
    truth_times_s: np.ndarray,
    truth_positions_m: np.ndarray,
    max_time_delta_s: float | None = None,
    dimensions: int = 3,
) -> np.ndarray:
    """Compute interpolated-truth sample errors with overflow-stable norms."""

    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    max_time_delta_s = _IMPL._validate_max_time_delta_s(max_time_delta_s)

    estimate_times, estimate_positions = _IMPL._prepare_time_position_samples(
        estimate_times_s,
        estimate_positions_m,
        dimensions=dimensions,
    )
    truth_times, truth_positions = _IMPL._prepare_time_position_series(
        truth_times_s,
        truth_positions_m,
        dimensions=dimensions,
    )
    if estimate_times.size == 0 or truth_times.size == 0:
        return np.array([], dtype=float)

    truth_at_estimate, valid = _IMPL.interpolate_positions_at_times(
        truth_times,
        truth_positions,
        estimate_times,
        max_time_delta_s=max_time_delta_s,
    )
    finite = (
        valid
        & np.isfinite(estimate_positions[:, :dimensions]).all(axis=1)
        & np.isfinite(truth_at_estimate[:, :dimensions]).all(axis=1)
    )
    if not bool(finite.any()):
        return np.array([], dtype=float)

    with np.errstate(over="ignore", invalid="ignore"):
        deltas = (
            estimate_positions[finite, :dimensions]
            - truth_at_estimate[finite, :dimensions]
        )
    errors = _stable_euclidean_norms(deltas)
    return errors[np.isfinite(errors)]


def install() -> None:
    """Install stable timestamp and norm implementations on public modules."""

    if getattr(_metrics, _PATCH_MARKER, False):
        return

    for module in (_IMPL, _metrics):
        module.nearest_time_indices = nearest_time_indices
        module._truth_grid_with_estimate_support = _truth_grid_with_estimate_support
        module.interpolate_positions_at_times = interpolate_positions_at_times
        module._single_sample_position_errors_m = _single_sample_position_errors_m
        module.position_errors_m = position_errors_m
        module.position_errors_at_estimates_m = position_errors_at_estimates_m
        module.position_errors_at_times_m = position_errors_at_times_m

    setattr(_metrics, _PATCH_MARKER, True)
    setattr(_IMPL, _PATCH_MARKER, True)


install()
