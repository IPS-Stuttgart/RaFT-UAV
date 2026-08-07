"""Reject malformed candidate-pull temporal tolerances."""

from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float

_INSTALLED = False
_ERROR = "time_tolerance_s must be a finite non-negative real scalar"


def _time_tolerance(value: object) -> float:
    """Return a finite non-negative temporal tolerance."""

    parsed = optional_float(value)
    if parsed is None or parsed < 0.0:
        raise ValueError(_ERROR)
    return parsed


def install() -> None:
    """Install strict candidate-pull tolerance validation exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from raft_uav.mmuad import candidate_pull

    original_candidate_centers = candidate_pull.candidate_centers_for_results
    original_align_centers = candidate_pull.align_candidate_centers

    def candidate_centers_for_results(
        candidates: pd.DataFrame,
        results: pd.DataFrame,
        current_xyz: np.ndarray,
        *,
        top_k: int = 5,
        time_tolerance_s: float = 0.5,
    ) -> pd.DataFrame:
        """Build row-wise centers with a valid temporal matching tolerance."""

        return original_candidate_centers(
            candidates,
            results,
            current_xyz,
            top_k=top_k,
            time_tolerance_s=_time_tolerance(time_tolerance_s),
        )

    def align_candidate_centers(
        results: pd.DataFrame,
        centers: pd.DataFrame,
        *,
        time_tolerance_s: float,
    ) -> pd.DataFrame:
        """Align centers with a valid temporal matching tolerance."""

        return original_align_centers(
            results,
            centers,
            time_tolerance_s=_time_tolerance(time_tolerance_s),
        )

    candidate_pull.candidate_centers_for_results = candidate_centers_for_results
    candidate_pull.align_candidate_centers = align_candidate_centers
    implementation = getattr(candidate_pull, "_IMPL", None)
    if implementation is not None:
        implementation.candidate_centers_for_results = candidate_centers_for_results
        implementation.align_candidate_centers = align_candidate_centers
    _INSTALLED = True
