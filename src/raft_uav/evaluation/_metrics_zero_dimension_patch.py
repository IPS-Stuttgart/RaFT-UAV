"""Reject interpolation trajectories with zero coordinate dimensions."""

from __future__ import annotations

from functools import wraps
from importlib import import_module

import numpy as np

_metrics = import_module("raft_uav.evaluation.metrics")
_PATCH_MARKER = "_metrics_zero_dimension_patch_applied"


def install() -> None:
    """Install a deterministic zero-coordinate validation boundary."""

    if getattr(_metrics, _PATCH_MARKER, False):
        return

    original = _metrics.interpolate_positions_at_times

    @wraps(original)
    def interpolate_positions_at_times(
        reference_times_s: np.ndarray,
        reference_positions_m: np.ndarray,
        query_times_s: np.ndarray,
        *,
        max_time_delta_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        reference_array = np.ma.asarray(reference_positions_m)
        if reference_array.ndim == 2 and reference_array.shape[1] == 0:
            raise ValueError(
                "reference_positions_m must contain at least one coordinate dimension"
            )
        return original(
            reference_times_s,
            reference_positions_m,
            query_times_s,
            max_time_delta_s=max_time_delta_s,
        )

    _metrics.interpolate_positions_at_times = interpolate_positions_at_times
    implementation = getattr(_metrics, "_IMPL", None)
    if implementation is not None:
        implementation.interpolate_positions_at_times = interpolate_positions_at_times
    setattr(_metrics, _PATCH_MARKER, True)


install()
