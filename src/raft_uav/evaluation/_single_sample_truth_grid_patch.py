"""Keep one-sample trajectory metrics on the truth-grid support interval."""

from __future__ import annotations

from importlib import import_module

import numpy as np


_metrics = import_module("raft_uav.evaluation.metrics")


def _single_sample_position_errors_m(
    estimate_times: np.ndarray,
    estimate_positions: np.ndarray,
    truth_times: np.ndarray,
    truth_positions: np.ndarray,
    *,
    max_time_delta_s: float | None,
    dimensions: int,
) -> np.ndarray:
    """Evaluate a singleton estimate only where its zero-width support meets truth."""

    query_times, query_truth_positions = _metrics._truth_grid_with_estimate_support(
        estimate_times,
        truth_times,
        truth_positions,
        max_time_delta_s=max_time_delta_s,
    )
    if query_times.size == 0:
        return np.array([], dtype=float)

    query_estimate_positions = _metrics._interpolate_positions(
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


def install() -> None:
    """Install the singleton truth-grid support fix."""

    if getattr(_metrics, "_single_sample_truth_grid_patch_applied", False):
        return
    _metrics._single_sample_position_errors_m = _single_sample_position_errors_m
    _metrics._single_sample_truth_grid_patch_applied = True


install()
