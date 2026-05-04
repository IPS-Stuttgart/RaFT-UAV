"""Tracking metrics for position trajectories."""

from __future__ import annotations

import numpy as np


def nearest_time_indices(reference_times_s: np.ndarray, query_times_s: np.ndarray) -> np.ndarray:
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
) -> np.ndarray:
    """Compute nearest-neighbor 3D position errors against truth."""

    estimate_times = np.asarray(estimate_times_s, dtype=float).reshape(-1)
    estimate_positions = np.asarray(estimate_positions_m, dtype=float)
    truth_times = np.asarray(truth_times_s, dtype=float).reshape(-1)
    truth_positions = np.asarray(truth_positions_m, dtype=float)

    truth_indices = nearest_time_indices(truth_times, estimate_times)
    deltas_t = np.abs(truth_times[truth_indices] - estimate_times)
    deltas_xyz = estimate_positions[:, :3] - truth_positions[truth_indices, :3]
    errors = np.linalg.norm(deltas_xyz, axis=1)
    if max_time_delta_s is None:
        return errors
    return errors[deltas_t <= float(max_time_delta_s)]


def summarize_errors(errors_m: np.ndarray) -> dict[str, float]:
    """Summarize scalar position errors."""

    errors = np.asarray(errors_m, dtype=float).reshape(-1)
    if errors.size == 0:
        return {"count": 0.0, "rmse_m": float("nan"), "mae_m": float("nan"), "p95_m": float("nan")}
    return {
        "count": float(errors.size),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "mae_m": float(np.mean(np.abs(errors))),
        "p95_m": float(np.percentile(errors, 95)),
    }
