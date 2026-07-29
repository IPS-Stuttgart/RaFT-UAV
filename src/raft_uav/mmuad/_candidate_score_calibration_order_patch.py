"""Preserve candidate row alignment while attaching calibration truth targets."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

_INSTALLED = False


def _preserve_candidate_order(
    original: Callable[..., pd.DataFrame],
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    good_threshold_m: float,
    max_truth_time_delta_s: float,
) -> pd.DataFrame:
    """Call the legacy matcher without allowing sequence grouping to reorder rows."""

    candidate_rows = pd.DataFrame(candidates).copy()
    marker = "__raft_uav_candidate_score_calibration_input_order__"
    while marker in candidate_rows.columns:
        marker += "_"
    candidate_rows[marker] = np.arange(len(candidate_rows), dtype=np.int64)

    labelled = original(
        candidate_rows,
        truth,
        good_threshold_m=good_threshold_m,
        max_truth_time_delta_s=max_truth_time_delta_s,
    )
    if marker not in labelled.columns:
        raise RuntimeError("candidate score calibration discarded its row-order marker")

    marker_values = pd.to_numeric(labelled[marker], errors="coerce").to_numpy(dtype=float)
    expected = np.arange(len(candidate_rows), dtype=float)
    if len(labelled) != len(candidate_rows) or not np.array_equal(
        np.sort(marker_values),
        expected,
    ):
        raise RuntimeError("candidate score calibration changed the candidate row set")

    return (
        labelled.sort_values(marker, kind="stable")
        .drop(columns=[marker])
        .reset_index(drop=True)
    )


def install() -> None:
    """Install candidate truth-target row-order preservation exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return

    from raft_uav.mmuad import candidate_score_calibration

    original = candidate_score_calibration._attach_truth_targets

    def _attach_truth_targets(
        candidates: pd.DataFrame,
        truth: pd.DataFrame,
        *,
        good_threshold_m: float,
        max_truth_time_delta_s: float,
    ) -> pd.DataFrame:
        return _preserve_candidate_order(
            original,
            candidates,
            truth,
            good_threshold_m=good_threshold_m,
            max_truth_time_delta_s=max_truth_time_delta_s,
        )

    candidate_score_calibration._attach_truth_targets = _attach_truth_targets
    _INSTALLED = True
