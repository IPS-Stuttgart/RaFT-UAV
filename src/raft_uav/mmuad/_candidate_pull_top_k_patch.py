"""Reject lossy candidate-pull top-k controls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int

_INSTALLED = False


def _positive_top_k(value: object) -> int:
    """Return a positive integer without truncating malformed controls."""

    parsed = optional_int(value)
    if parsed is None or parsed <= 0:
        raise ValueError("top_k must be a positive integer")
    return parsed


def install() -> None:
    """Install strict candidate-pull top-k validation exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from raft_uav.mmuad import candidate_pull

    original_topk_candidate_centers = candidate_pull.topk_candidate_centers
    original_candidate_centers_for_results = candidate_pull.candidate_centers_for_results

    def topk_candidate_centers(
        candidates: pd.DataFrame,
        *,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Return candidate centers after validating the requested count."""

        return original_topk_candidate_centers(
            candidates,
            top_k=_positive_top_k(top_k),
        )

    def candidate_centers_for_results(
        candidates: pd.DataFrame,
        results: pd.DataFrame,
        current_xyz: np.ndarray,
        *,
        top_k: int = 5,
        time_tolerance_s: float = 0.5,
    ) -> pd.DataFrame:
        """Return row-wise centers after validating the requested count."""

        return original_candidate_centers_for_results(
            candidates,
            results,
            current_xyz,
            top_k=_positive_top_k(top_k),
            time_tolerance_s=time_tolerance_s,
        )

    candidate_pull.topk_candidate_centers = topk_candidate_centers
    candidate_pull.candidate_centers_for_results = candidate_centers_for_results
    implementation = getattr(candidate_pull, "_IMPL", None)
    if implementation is not None:
        implementation.topk_candidate_centers = topk_candidate_centers
        implementation.candidate_centers_for_results = candidate_centers_for_results
    _INSTALLED = True
