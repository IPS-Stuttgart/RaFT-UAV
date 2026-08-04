"""Keep final same-time truth rows during uncertainty calibration."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_raft_uav_prefers_final_duplicate_uncertainty_truth"


def install() -> None:
    """Install final-duplicate truth matching for uncertainty residuals."""

    from raft_uav import uncertainty as uncertainty_module

    implementation_module = getattr(
        uncertainty_module,
        "_legacy",
        uncertainty_module,
    )
    original: Callable[..., Any] = implementation_module._nearest_time_indices
    if getattr(original, _PATCH_MARKER, False):
        uncertainty_module._nearest_time_indices = original
        return

    @wraps(original)
    def nearest_time_indices(reference_times_s, query_times_s):
        reference = np.asarray(reference_times_s, dtype=float).reshape(-1)
        finite_indices = np.flatnonzero(np.isfinite(reference))
        if finite_indices.size == 0:
            return original(reference_times_s, query_times_s)

        finite_values = reference[finite_indices]
        order = np.argsort(finite_values, kind="mergesort")
        sorted_indices = finite_indices[order]
        sorted_values = finite_values[order]

        keep_last = np.ones(sorted_values.size, dtype=bool)
        keep_last[:-1] = sorted_values[:-1] != sorted_values[1:]
        authoritative_indices = sorted_indices[keep_last]

        matched_subset_indices = original(
            reference[authoritative_indices],
            query_times_s,
        )
        return authoritative_indices[matched_subset_indices]

    setattr(nearest_time_indices, _PATCH_MARKER, True)
    implementation_module._nearest_time_indices = nearest_time_indices
    uncertainty_module._nearest_time_indices = nearest_time_indices
