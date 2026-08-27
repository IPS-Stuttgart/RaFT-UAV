"""Reject malformed candidate-pull temporal tolerances and stabilize matching."""

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


def _timestamp_gap_s(left: object, right: object) -> float:
    """Return an absolute finite-timestamp gap without NumPy overflow signaling."""

    return abs(float(left) - float(right))


def _nearest_candidate_frame(
    candidates: pd.DataFrame,
    *,
    sequence: object,
    target_time_s: object,
    tolerance_s: float,
) -> pd.DataFrame:
    """Return hypotheses from the nearest timestamp without vector overflow."""

    sequence_rows = candidates.loc[
        candidates["Sequence"].astype(str) == str(sequence)
    ].copy()
    if sequence_rows.empty:
        return sequence_rows

    candidate_times = pd.to_numeric(sequence_rows["Timestamp"], errors="coerce")
    target_time = float(target_time_s)
    finite = np.isfinite(candidate_times.to_numpy(dtype=float))
    if not np.isfinite(target_time) or not bool(finite.any()):
        return sequence_rows.iloc[0:0].copy()

    unique_times = np.unique(candidate_times.loc[finite].to_numpy(dtype=float))
    deltas = np.fromiter(
        (
            _timestamp_gap_s(candidate_time, target_time)
            for candidate_time in unique_times
        ),
        dtype=float,
        count=len(unique_times),
    )
    eligible = deltas <= float(tolerance_s)
    if not bool(eligible.any()):
        return sequence_rows.iloc[0:0].copy()
    eligible_times = unique_times[eligible]
    eligible_deltas = deltas[eligible]
    nearest_time = float(eligible_times[int(np.argmin(eligible_deltas))])
    return sequence_rows.loc[candidate_times.eq(nearest_time)].copy()


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

    candidate_pull._nearest_candidate_frame = _nearest_candidate_frame
    candidate_pull.candidate_centers_for_results = candidate_centers_for_results
    candidate_pull.align_candidate_centers = align_candidate_centers
    implementation = getattr(candidate_pull, "_IMPL", None)
    if implementation is not None:
        implementation.candidate_centers_for_results = candidate_centers_for_results
        implementation.align_candidate_centers = align_candidate_centers
    _INSTALLED = True
