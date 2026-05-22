"""Tracking metrics for position trajectories."""

from __future__ import annotations

import numpy as np


def nearest_time_indices(
    reference_times_s: np.ndarray, query_times_s: np.ndarray
) -> np.ndarray:
    """Return indices of nearest reference timestamps for each query timestamp."""

    reference = np.asarray(reference_times_s, dtype=float).reshape(-1)
    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    if reference.size == 0:
        raise ValueError("reference_times_s must not be empty")
    insertion = np.searchsorted(reference, query)
    right = np.clip(insertion, 0, reference.size - 1)
    left = np.clip(insertion - 1, 0, reference.size - 1)
    use_right = np.abs(reference[right] - query) < np.abs(reference[left] - query)
    return np.where(use_right, right, left)


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
