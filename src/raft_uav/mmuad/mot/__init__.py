"""Compatibility fixes for pooled MOT identity and timestamp bookkeeping.

The maintained implementation lives in the sibling ``mot.py`` module. This
package preserves the public import path while ensuring that pooled MOT metrics
scope object identities by sequence and count tolerance-matched frames once.
"""

from __future__ import annotations

from collections.abc import Iterator
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "mot.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._mot_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load MMUAD MOT implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_COMPUTE_MULTI_OBJECT_METRICS = _IMPL.compute_multi_object_metrics
_ORIGINAL_GREEDY_TRUTH_MATCHES = _IMPL._greedy_truth_matches


def _scope_truth_track_ids(truth: pd.DataFrame | None) -> pd.DataFrame | None:
    """Return truth rows whose present object IDs are namespaced by sequence."""

    if truth is None or truth.empty:
        return truth
    if "sequence_id" not in truth.columns or "track_id" not in truth.columns:
        return truth

    scoped = truth.copy()
    present = _IMPL._track_id_present_mask(scoped["track_id"])
    if not present.any():
        return scoped

    scoped["track_id"] = scoped["track_id"].astype(object)
    scoped_ids = pd.Series(
        [
            (str(sequence_id), track_id)
            for sequence_id, track_id in scoped.loc[
                present,
                ["sequence_id", "track_id"],
            ].itertuples(index=False, name=None)
        ],
        index=scoped.index[present],
        dtype=object,
    )
    scoped.loc[present, "track_id"] = scoped_ids
    return scoped


def _metric_frame_pairs(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield tolerance-clustered frame pairs exactly once."""

    estimates = estimates.copy()
    truth = truth.copy()
    estimates["time_s"] = pd.to_numeric(estimates["time_s"], errors="coerce")
    truth["time_s"] = pd.to_numeric(truth["time_s"], errors="coerce")

    if "sequence_id" in estimates.columns and "sequence_id" in truth.columns:
        estimates["_metric_sequence_id"] = estimates["sequence_id"].astype(str)
        truth["_metric_sequence_id"] = truth["sequence_id"].astype(str)
        sequence_ids = sorted(
            set(estimates["_metric_sequence_id"]) | set(truth["_metric_sequence_id"])
        )
        for sequence_id in sequence_ids:
            pred = estimates.loc[
                estimates["_metric_sequence_id"] == sequence_id
            ].copy()
            gt = truth.loc[truth["_metric_sequence_id"] == sequence_id].copy()
            yield from _metric_time_cluster_pairs(pred, gt)
        return

    yield from _metric_time_cluster_pairs(estimates, truth)


def _metric_time_cluster_pairs(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Group adjacent timestamps within the absolute matching tolerance."""

    estimate_times = pd.to_numeric(estimates["time_s"], errors="coerce").to_numpy(float)
    truth_times = pd.to_numeric(truth["time_s"], errors="coerce").to_numpy(float)
    finite_times = np.concatenate(
        (
            estimate_times[np.isfinite(estimate_times)],
            truth_times[np.isfinite(truth_times)],
        )
    )
    if finite_times.size == 0:
        return

    ordered_times = np.unique(np.sort(finite_times))
    cluster_start = float(ordered_times[0])
    previous_time = cluster_start
    for current_value in ordered_times[1:]:
        current_time = float(current_value)
        if current_time - previous_time > _IMPL._MOT_TIME_MATCH_ATOL_S:
            yield (
                _metric_rows_in_time_cluster(
                    estimates,
                    estimate_times,
                    cluster_start,
                    previous_time,
                ),
                _metric_rows_in_time_cluster(
                    truth,
                    truth_times,
                    cluster_start,
                    previous_time,
                ),
            )
            cluster_start = current_time
        previous_time = current_time

    yield (
        _metric_rows_in_time_cluster(
            estimates,
            estimate_times,
            cluster_start,
            previous_time,
        ),
        _metric_rows_in_time_cluster(
            truth,
            truth_times,
            cluster_start,
            previous_time,
        ),
    )


def _metric_rows_in_time_cluster(
    frame: pd.DataFrame,
    times: np.ndarray,
    start: float,
    end: float,
) -> pd.DataFrame:
    """Return rows belonging to one finite timestamp cluster."""

    mask = np.isfinite(times) & (times >= start) & (times <= end)
    return frame.loc[mask].copy()


def _validated_match_distance_m(value: Any) -> float:
    """Return a finite nonnegative MOT matching radius."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("match_distance_m must be finite and nonnegative")
    try:
        array = np.asarray(value)
        if array.ndim != 0:
            raise TypeError
        distance_m = float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("match_distance_m must be finite and nonnegative") from exc
    if not np.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("match_distance_m must be finite and nonnegative")
    return distance_m


def _greedy_truth_matches(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    max_distance_m: float,
) -> list[tuple[int, int, float]]:
    """Match one frame after validating the distance threshold."""

    return _ORIGINAL_GREEDY_TRUTH_MATCHES(
        pred,
        gt,
        max_distance_m=_validated_match_distance_m(max_distance_m),
    )


def compute_multi_object_metrics(
    estimates: pd.DataFrame,
    truth: pd.DataFrame | None,
    *,
    match_distance_m: float = 25.0,
) -> dict[str, Any]:
    """Compute MOT metrics with sequence-scoped identities and unique frames."""

    return _ORIGINAL_COMPUTE_MULTI_OBJECT_METRICS(
        estimates,
        _scope_truth_track_ids(truth),
        match_distance_m=_validated_match_distance_m(match_distance_m),
    )


_IMPL._metric_frame_pairs = _metric_frame_pairs
_IMPL._greedy_truth_matches = _greedy_truth_matches
_IMPL.compute_multi_object_metrics = compute_multi_object_metrics

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_scope_truth_track_ids"] = _scope_truth_track_ids
globals()["_metric_frame_pairs"] = _metric_frame_pairs
globals()["_metric_time_cluster_pairs"] = _metric_time_cluster_pairs
globals()["_metric_rows_in_time_cluster"] = _metric_rows_in_time_cluster
globals()["_validated_match_distance_m"] = _validated_match_distance_m
globals()["_greedy_truth_matches"] = _greedy_truth_matches
globals()["compute_multi_object_metrics"] = compute_multi_object_metrics

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
