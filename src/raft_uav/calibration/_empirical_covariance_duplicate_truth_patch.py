"""Use final duplicate truth samples in empirical covariance alignment."""

from __future__ import annotations

import numpy as np

from raft_uav.calibration import empirical_covariance as _IMPL

_ORIGINAL_NEAREST_TIME_INDICES = _IMPL._nearest_time_indices
_PATCH_MARKER = "_raft_uav_prefers_final_duplicate_covariance_truth"


def _nearest_time_indices(
    reference_times_s: np.ndarray,
    query_times_s: np.ndarray,
) -> np.ndarray:
    """Resolve selected duplicate timestamps to their final original row."""

    reference = np.asarray(reference_times_s, dtype=float).reshape(-1)
    selected = _ORIGINAL_NEAREST_TIME_INDICES(reference, query_times_s)
    final_index_by_time: dict[float, int] = {}
    for index in np.flatnonzero(np.isfinite(reference)):
        final_index_by_time[float(reference[index])] = int(index)
    return np.asarray(
        [final_index_by_time[float(reference[index])] for index in selected],
        dtype=int,
    )


setattr(_nearest_time_indices, _PATCH_MARKER, True)
if not getattr(_IMPL._nearest_time_indices, _PATCH_MARKER, False):
    _IMPL._nearest_time_indices = _nearest_time_indices
