"""Tracking metrics for position trajectories."""

from __future__ import annotations

import numpy as np


def nearest_time_indices(
    reference_times_s: np.ndarray, query_times_s: np.ndarray
) -> np.ndarray:
    """Return indices of nearest reference timestamps for each query timestamp.

    The returned indices refer to the input ``reference_times_s`` array.  Callers
    do not have to pre-sort the reference timestamps; this helper sorts a finite
    working copy internally before using ``searchsorted``.
    """

    reference = np.asarray(reference_times_s, dtype=float).reshape(-1)
    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    finite_reference = np.isfinite(reference)
    if not bool(np.any(finite_reference)):
        raise ValueError("reference_times_s must contain at least one finite timestamp")
    original_indices = np.flatnonzero(finite_reference)
    finite_values = reference[finite_reference]
    sort_order = np.argsort(finite_values, kind="mergesort")
    sorted_reference = finite_values[sort_order]
    sorted_original_indices = original_indices[sort_order]
    insertion = np.searchsorted(sorted_reference, query)
    right = np.clip(insertion, 0, sorted_reference.size - 1)
    left = np.clip(insertion - 1, 0, sorted_reference.size - 1)
    use_right = np.abs(sorted_reference[right] - query) < np.abs(sorted_reference[left] - query)
    return sorted_original_indices[np.where(use_right, right, left)]


def position_errors_m(
    estimate_times_s: np.ndarray,
    estimate_positions_m: np.ndarray,
    truth_times_s: np.ndarray,
    truth_positions_m: np.ndarray,
    max_time_delta_s: float | None = None,
    dimensions: int = 3,
) -> np.ndarray:
    """Compute truth-grid trajectory position errors against truth.

    The estimates are linearly interpolated to truth timestamps before the
    Euclidean errors are computed. This makes RMSE/P95 comparable across
    methods that emit posterior records at different measurement/update times.
    ``max_time_delta_s`` rejects truth samples whose bracketing estimate samples
    are farther away than the requested tolerance, preventing long-gap
    interpolation from dominating the metric.
    """

    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    max_time_delta_s = _validate_max_time_delta_s(max_time_delta_s)

    estimate_times, estimate_positions = _prepare_time_position_series(
        estimate_times_s,
        estimate_positions_m,
        dimensions=dimensions,
    )
    truth_times, truth_positions = _prepare_time_position_series(
        truth_times_s,
        truth_positions_m,
        dimensions=dimensions,
    )
    if estimate_times.size == 0 or truth_times.size == 0:
        return np.array([], dtype=float)
    if estimate_times.size == 1:
        return _single_sample_position_errors_m(
            estimate_times,
            estimate_positions,
            truth_times,
            truth_positions,
            max_time_delta_s=max_time_delta_s,
            dimensions=dimensions,
        )

    query_times, query_truth_positions = _truth_grid_with_estimate_support(
        estimate_times,
        truth_times,
        truth_positions,
        max_time_delta_s=max_time_delta_s,
    )
    if query_times.size == 0:
        return np.array([], dtype=float)

    query_estimate_positions = _interpolate_positions(
        estimate_times,
        estimate_positions,
        query_times,
    )
    deltas = (
        query_estimate_positions[:, :dimensions]
        - query_truth_positions[:, :dimensions]
    )
    errors = np.linalg.norm(deltas, axis=1)
    return errors[np.isfinite(errors)]


def position_errors_at_estimates_m(
    estimate_times_s: np.ndarray,
    estimate_positions_m: np.ndarray,
    truth_times_s: np.ndarray,
    truth_positions_m: np.ndarray,
    max_time_delta_s: float | None = None,
    dimensions: int = 3,
) -> np.ndarray:
    """Compute paper-style per-estimate position errors against nearest truth.

    ``position_errors_m`` is intentionally truth-grid/interpolation based so
    RMSE/P95 are comparable across methods with different update rates.  The
    AERPAW paper tables, however, report mean/std/max errors at the sensor or
    fusion sample timestamps.  This helper preserves every finite estimate row,
    including duplicate timestamps, and compares it to the nearest truth sample
    within ``max_time_delta_s``.
    """

    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    max_time_delta_s = _validate_max_time_delta_s(max_time_delta_s)

    estimate_times, estimate_positions = _prepare_time_position_samples(
        estimate_times_s,
        estimate_positions_m,
        dimensions=dimensions,
    )
    truth_times, truth_positions = _prepare_time_position_samples(
        truth_times_s,
        truth_positions_m,
        dimensions=dimensions,
    )
    if estimate_times.size == 0 or truth_times.size == 0:
        return np.array([], dtype=float)

    truth_indices = nearest_time_indices(truth_times, estimate_times)
    time_deltas = np.abs(truth_times[truth_indices] - estimate_times)
    keep = np.ones(estimate_times.size, dtype=bool)
    if max_time_delta_s is not None:
        keep &= time_deltas <= float(max_time_delta_s)
    if not keep.any():
        return np.array([], dtype=float)

    deltas = (
        estimate_positions[keep, :dimensions]
        - truth_positions[truth_indices[keep], :dimensions]
    )
    errors = np.linalg.norm(deltas, axis=1)
    return errors[np.isfinite(errors)]


def sampled_position_errors_m(
    estimate_times_s: np.ndarray,
    estimate_positions_m: np.ndarray,
    truth_times_s: np.ndarray,
    truth_positions_m: np.ndarray,
    max_time_delta_s: float | None = None,
    dimensions: int = 3,
) -> np.ndarray:
    """Compute position errors at estimate sample times against nearest truth."""

    return position_errors_at_estimates_m(
        estimate_times_s,
        estimate_positions_m,
        truth_times_s,
        truth_positions_m,
        max_time_delta_s=max_time_delta_s,
        dimensions=dimensions,
    )


def _validate_max_time_delta_s(value: float | None) -> float | None:
    if value is None:
        return None

    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError("max_time_delta_s must be a finite, non-negative scalar")
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError("max_time_delta_s must be a finite, non-negative scalar")
    try:
        parsed = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "max_time_delta_s must be a finite, non-negative scalar"
        ) from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError("max_time_delta_s must be a finite, non-negative scalar")
    return parsed


def _prepare_time_position_series(
    times_s: np.ndarray,
    positions_m: np.ndarray,
    *,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    times, positions = _prepare_time_position_samples(
        times_s,
        positions_m,
        dimensions=dimensions,
    )
    if times.size == 0:
        return times, positions
    keep_last_duplicate = np.ones(times.size, dtype=bool)
    keep_last_duplicate[:-1] = times[:-1] != times[1:]
    return times[keep_last_duplicate], positions[keep_last_duplicate]


def _prepare_time_position_samples(
    times_s: np.ndarray,
    positions_m: np.ndarray,
    *,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times_s, dtype=float).reshape(-1)
    positions = np.asarray(positions_m, dtype=float)
    if positions.ndim != 2:
        raise ValueError("positions_m must be a 2D array")
    if positions.shape[0] != times.size:
        raise ValueError("positions_m and times_s must have the same row count")
    if positions.shape[1] < dimensions:
        raise ValueError("positions_m has fewer columns than requested dimensions")

    positions = positions[:, :dimensions]
    finite = np.isfinite(times) & np.isfinite(positions).all(axis=1)
    times = times[finite]
    positions = positions[finite]
    if times.size == 0:
        return times, positions

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    positions = positions[order]
    return times, positions


def _single_sample_position_errors_m(
    estimate_times: np.ndarray,
    estimate_positions: np.ndarray,
    truth_times: np.ndarray,
    truth_positions: np.ndarray,
    *,
    max_time_delta_s: float | None,
    dimensions: int,
) -> np.ndarray:
    truth_index = nearest_time_indices(truth_times, estimate_times)[0]
    time_delta = abs(float(truth_times[truth_index] - estimate_times[0]))
    if max_time_delta_s is not None and time_delta > float(max_time_delta_s):
        return np.array([], dtype=float)
    delta = (
        estimate_positions[0, :dimensions]
        - truth_positions[truth_index, :dimensions]
    )
    error = float(np.linalg.norm(delta))
    if np.isfinite(error):
        return np.array([error], dtype=float)
    return np.array([], dtype=float)


def _truth_grid_with_estimate_support(
    estimate_times: np.ndarray,
    truth_times: np.ndarray,
    truth_positions: np.ndarray,
    *,
    max_time_delta_s: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    supported = (truth_times >= estimate_times[0]) & (truth_times <= estimate_times[-1])
    query_times = truth_times[supported]
    query_truth_positions = truth_positions[supported]
    if query_times.size == 0 or max_time_delta_s is None:
        return query_times, query_truth_positions

    max_delta = float(max_time_delta_s)
    right = np.searchsorted(estimate_times, query_times, side="left")
    right = np.clip(right, 0, estimate_times.size - 1)
    left = np.clip(right - 1, 0, estimate_times.size - 1)
    exact = np.isclose(estimate_times[right], query_times, rtol=0.0, atol=1.0e-9)
    left_delta = np.abs(query_times - estimate_times[left])
    right_delta = np.abs(estimate_times[right] - query_times)
    close_to_bracket = (left_delta <= max_delta) & (right_delta <= max_delta)
    keep = exact | close_to_bracket
    return query_times[keep], query_truth_positions[keep]


def _interpolate_positions(
    sample_times: np.ndarray,
    sample_positions: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    columns = [
        np.interp(query_times, sample_times, sample_positions[:, dim])
        for dim in range(sample_positions.shape[1])
    ]
    return np.column_stack(columns)


def summarize_errors(errors_m: np.ndarray) -> dict[str, float | None]:
    """Summarize scalar position errors."""

    errors = np.asarray(errors_m, dtype=float).reshape(-1)
    errors = errors[np.isfinite(errors)]
    if errors.size == 0:
        return {
            "count": 0.0,
            "mean_m": None,
            "std_m": None,
            "rmse_m": None,
            "mae_m": None,
            "p50_m": None,
            "p95_m": None,
            "max_m": None,
        }
    mean_error_m = float(np.mean(errors))
    return {
        "count": float(errors.size),
        "mean_m": mean_error_m,
        "std_m": float(np.std(errors)),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "mae_m": mean_error_m,
        "p50_m": float(np.percentile(errors, 50)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
    }

def interpolate_positions_at_times(
    reference_times_s: np.ndarray,
    reference_positions_m: np.ndarray,
    query_times_s: np.ndarray,
    *,
    max_time_delta_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a position trajectory onto arbitrary query timestamps.

    The returned mask is true only for query times inside the reference time
    span and, when ``max_time_delta_s`` is provided, close enough to the nearest
    reference sample.  This is the paper-table direction: evaluate an estimate
    or sensor measurement at its own timestamp against interpolated truth.
    """

    max_time_delta_s = _validate_max_time_delta_s(max_time_delta_s)
    reference_array = np.asarray(reference_positions_m, dtype=float)
    reference_dimensions = reference_array.shape[1] if reference_array.ndim == 2 else 3
    reference_times, reference_positions = _prepare_time_position_series(
        reference_times_s,
        reference_array,
        dimensions=reference_dimensions,
    )
    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    if reference_times.size == 0:
        return np.full((query.size, reference_positions.shape[1] if reference_positions.ndim == 2 else 3), np.nan), np.zeros(query.size, dtype=bool)
    interpolated = _interpolate_positions(reference_times, reference_positions, query)
    valid = np.isfinite(query) & (query >= reference_times[0]) & (query <= reference_times[-1])
    if max_time_delta_s is not None:
        insertion = np.searchsorted(reference_times, query)
        right = np.clip(insertion, 0, reference_times.size - 1)
        left = np.clip(insertion - 1, 0, reference_times.size - 1)
        nearest_delta = np.minimum(np.abs(reference_times[right] - query), np.abs(reference_times[left] - query))
        valid &= nearest_delta <= float(max_time_delta_s)
    return interpolated, valid


def position_errors_at_times_m(
    estimate_times_s: np.ndarray,
    estimate_positions_m: np.ndarray,
    truth_times_s: np.ndarray,
    truth_positions_m: np.ndarray,
    max_time_delta_s: float | None = None,
    dimensions: int = 3,
) -> np.ndarray:
    """Compute position errors at estimate/measurement timestamps.

    This is distinct from :func:`position_errors_m`, which interpolates the
    estimate trajectory to the truth grid.  The reference paper table compares
    RF/radar/KF outputs to truth interpolated at each output timestamp, so this
    helper preserves the output count and timestamp support of the method under
    evaluation.
    """

    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    max_time_delta_s = _validate_max_time_delta_s(max_time_delta_s)

    # Paper-table metrics are sample metrics: every emitted sensor/fusion row
    # contributes one error.  Do not collapse duplicate estimate timestamps here;
    # radar frames and fused outputs can legitimately contain repeated times, and
    # dropping them changes both the count fingerprint and the mean/std/max rows.
    estimate_times, estimate_positions = _prepare_time_position_samples(
        estimate_times_s,
        estimate_positions_m,
        dimensions=dimensions,
    )
    truth_times, truth_positions = _prepare_time_position_series(
        truth_times_s,
        truth_positions_m,
        dimensions=dimensions,
    )
    if estimate_times.size == 0 or truth_times.size == 0:
        return np.array([], dtype=float)
    truth_at_estimate, valid = interpolate_positions_at_times(
        truth_times,
        truth_positions,
        estimate_times,
        max_time_delta_s=max_time_delta_s,
    )
    finite = valid & np.isfinite(estimate_positions[:, :dimensions]).all(axis=1) & np.isfinite(truth_at_estimate[:, :dimensions]).all(axis=1)
    if not finite.any():
        return np.array([], dtype=float)
    deltas = estimate_positions[finite, :dimensions] - truth_at_estimate[finite, :dimensions]
    errors = np.linalg.norm(deltas, axis=1)
    return errors[np.isfinite(errors)]


def empirical_position_covariance_at_times(
    estimate_times_s: np.ndarray,
    estimate_positions_m: np.ndarray,
    truth_times_s: np.ndarray,
    truth_positions_m: np.ndarray,
    max_time_delta_s: float | None = None,
    dimensions: int = 3,
) -> np.ndarray:
    """Estimate residual covariance against truth interpolated to output times.

    The covariance convention matches NumPy/MATLAB's unbiased sample covariance
    (``ddof=1``).  A ``ValueError`` is raised if fewer than two valid residuals
    are available.
    """

    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    max_time_delta_s = _validate_max_time_delta_s(max_time_delta_s)
    # Match the paper-table sample convention used by
    # ``position_errors_at_times_m``: every finite measurement/output row gets a
    # residual.  Only the truth trajectory is de-duplicated for interpolation.
    estimate_times, estimate_positions = _prepare_time_position_samples(
        estimate_times_s,
        estimate_positions_m,
        dimensions=dimensions,
    )
    truth_times, truth_positions = _prepare_time_position_series(
        truth_times_s,
        truth_positions_m,
        dimensions=dimensions,
    )
    if estimate_times.size == 0 or truth_times.size == 0:
        raise ValueError("cannot estimate covariance from empty estimate/truth series")
    truth_at_estimate, valid = interpolate_positions_at_times(
        truth_times,
        truth_positions,
        estimate_times,
        max_time_delta_s=max_time_delta_s,
    )
    finite = valid & np.isfinite(estimate_positions[:, :dimensions]).all(axis=1) & np.isfinite(truth_at_estimate[:, :dimensions]).all(axis=1)
    residuals = estimate_positions[finite, :dimensions] - truth_at_estimate[finite, :dimensions]
    if residuals.shape[0] < 2:
        raise ValueError("at least two valid residuals are required to estimate covariance")
    covariance = np.cov(residuals, rowvar=False, ddof=1)
    covariance = np.asarray(covariance, dtype=float).reshape(dimensions, dimensions)
    return 0.5 * (covariance + covariance.T)
