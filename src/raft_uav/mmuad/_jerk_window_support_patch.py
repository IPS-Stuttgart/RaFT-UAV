"""Keep jerk-window diagnostics aligned when invalid timestamp windows are skipped."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_raft_uav_preserves_jerk_window_support"


def install() -> None:
    """Install support-aware row attribution for Track 5 jerk windows."""

    from raft_uav.mmuad import track5_jerk_limit

    original: Callable[..., np.ndarray] = track5_jerk_limit._row_jerk_proxy
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def aligned(times: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        count = len(times)
        row_jerk = np.full(count, np.nan, dtype=float)
        d3 = track5_jerk_limit._third_derivative_matrix(times)
        if d3.size == 0:
            return row_jerk

        jerk_windows = d3 @ np.asarray(xyz, dtype=float)
        norms = np.linalg.norm(jerk_windows, axis=1)
        for coefficients, norm in zip(d3, norms, strict=True):
            for row_index in np.flatnonzero(coefficients):
                current = row_jerk[row_index]
                if np.isnan(current) or norm > current:
                    row_jerk[row_index] = float(norm)
        return row_jerk

    setattr(aligned, _PATCH_MARKER, True)
    track5_jerk_limit._row_jerk_proxy = aligned
    implementation: Any = getattr(track5_jerk_limit, "_IMPL", None)
    if implementation is not None and hasattr(implementation, "_row_jerk_proxy"):
        implementation._row_jerk_proxy = aligned
