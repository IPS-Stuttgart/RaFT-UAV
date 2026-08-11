"""Use matched truth identity when aggregating trajectory final displacement error."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pandas as pd


_evaluate = import_module("raft_uav.mmuad.evaluate")


def _append_anonymous_final_errors(
    frame: pd.DataFrame,
    column: str,
    final_errors: list[float],
) -> None:
    """Preserve the legacy handoff heuristic when truth identity is unavailable."""

    if frame.empty:
        return
    if _evaluate._has_overlapping_track_intervals(frame):
        trajectories = (
            trajectory
            for _, trajectory in frame.groupby("track_id", sort=False, dropna=False)
        )
    else:
        trajectories = (frame,)
    for trajectory in trajectories:
        endpoint = _evaluate._IMPL._final_error(trajectory, column)
        if endpoint is not None:
            final_errors.append(float(endpoint))


def _mean_final_error(frame: pd.DataFrame, column: str) -> float | None:
    """Average one endpoint per identified truth trajectory.

    Predicted track IDs are implementation artifacts and may change across a
    physical trajectory. Conversely, distinct truth tracks may have disjoint
    lifetimes. Prefer ``truth_track_id`` whenever matching supplied it, and use
    the existing overlap-based handoff heuristic only for anonymous truth rows.
    """

    if frame.empty:
        return _evaluate._IMPL._final_error(frame, column)

    sequence_groups = (
        [group for _, group in frame.groupby("sequence_id", sort=False, dropna=False)]
        if "sequence_id" in frame.columns
        else [frame]
    )
    final_errors: list[float] = []
    for sequence in sequence_groups:
        anonymous = sequence
        if "truth_track_id" in sequence.columns:
            truth_track_ids = sequence["truth_track_id"].map(
                _evaluate._valid_track_id_text
            )
            identified = truth_track_ids.notna()
            if bool(identified.any()):
                identified_rows = sequence.loc[identified]
                identified_ids = truth_track_ids.loc[identified]
                for _, trajectory in identified_rows.groupby(
                    identified_ids,
                    sort=False,
                    dropna=False,
                ):
                    endpoint = _evaluate._IMPL._final_error(trajectory, column)
                    if endpoint is not None:
                        final_errors.append(float(endpoint))
                anonymous = sequence.loc[~identified]

        _append_anonymous_final_errors(anonymous, column, final_errors)

    return float(np.mean(final_errors)) if final_errors else None


def install() -> None:
    """Install truth-identity-aware FDE on public and legacy evaluator paths."""

    if getattr(_evaluate, "_fde_truth_identity_patch_applied", False):
        return

    _evaluate._mean_final_error = _mean_final_error
    implementation = getattr(_evaluate, "_IMPL", None)
    if implementation is not None:
        implementation._mean_final_error = _mean_final_error
    _evaluate._fde_truth_identity_patch_applied = True


install()
