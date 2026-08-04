"""Use final duplicate truth samples in empirical covariance alignment."""

from __future__ import annotations

import numpy as np

from raft_uav.calibration import empirical_covariance as _PUBLIC_MODULE

_IMPLEMENTATION_MODULE = getattr(_PUBLIC_MODULE, "_IMPL", _PUBLIC_MODULE)
_ORIGINAL_NEAREST_TIME_INDICES = _IMPLEMENTATION_MODULE._nearest_time_indices
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


if getattr(_ORIGINAL_NEAREST_TIME_INDICES, _PATCH_MARKER, False):
    _PUBLIC_MODULE._nearest_time_indices = _ORIGINAL_NEAREST_TIME_INDICES
else:
    setattr(_nearest_time_indices, _PATCH_MARKER, True)
    _IMPLEMENTATION_MODULE._nearest_time_indices = _nearest_time_indices
    _PUBLIC_MODULE._nearest_time_indices = _nearest_time_indices
