"""Compatibility fixes for pooled MOT identity and timestamp bookkeeping.

The maintained implementation lives in the sibling ``mot.py`` module. This
package preserves the public import path while ensuring that pooled MOT metrics
scope object identities by sequence, count tolerance-matched frames once,
validate matching thresholds, resolve exact association ties deterministically,
enforce timestamp tolerance on every matched row pair, and use globally optimal
frame matching for both tracking and evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

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
_ORIGINAL_TRACK_COUNT = _IMPL._track_count


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


def _track_count(estimates: pd.DataFrame) -> int:
    """Count prediction identities without merging equal IDs across sequences."""

    if (
        estimates.empty
        or "sequence_id" not in estimates.columns
        or "output_track_id" not in estimates.columns
    ):
        return _ORIGINAL_TRACK_COUNT(estimates)

    identities = estimates.loc[:, ["sequence_id", "output_track_id"]].astype(str)
    return int(len(identities.drop_duplicates()))


def _nearest_track_id(
    z: np.ndarray,
    active: dict[int, Any],
    unmatched_tracks: set[int],
    config: Any,
) -> int | None:
    """Return the nearest eligible track with a stable lowest-ID tie break."""

    if not np.isfinite(z).all():
        return None
    best_track: int | None = None
    best_distance = float("inf")
    for track_id in sorted(unmatched_tracks):
        distance = float(np.linalg.norm(active[track_id].state[:3] - z))
        if distance < best_distance:
            best_distance = distance
            best_track = track_id
    if best_distance <= config.max_association_distance_m:
        return best_track
    return None


def _cardinality_first_assignment(
    distances: np.ndarray,
    eligible: np.ndarray,
    *,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign rows to eligible real columns before minimizing total distance."""

    row_count, column_count = distances.shape
    if row_count == 0 or column_count == 0 or not eligible.any():
        return np.array([], dtype=int), np.array([], dtype=int)

    max_matches = min(row_count, column_count)
    distance_weight = 0.5 / float(max_matches + 1)
    normalized_distances = (
        distances / max_distance if max_distance > 0.0 else np.zeros_like(distances)
    )
    costs = np.full(
        (row_count, column_count + row_count),
        1.0,
        dtype=float,
    )
    costs[:, :column_count] = np.where(
        eligible,
        distance_weight * np.minimum(normalized_distances, 1.0),
        2.0,
    )

    row_indices, assignment_columns = linear_sum_assignment(costs)
    real = (assignment_columns < column_count) & eligible[
        row_indices,
        np.minimum(assignment_columns, column_count - 1),
    ]
    return row_indices[real], assignment_columns[real]


def _optimal_track_matches(
    detections: pd.DataFrame,
    active: dict[int, Any],
    config: Any,
) -> dict[int, int]:
    """Return a maximum-cardinality track assignment for one detection frame."""

    if detections.empty or not active:
        return {}

    track_ids = sorted(active)
    detection_xyz = detections[["x_m", "y_m", "z_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)
    track_xyz = np.stack(
        [np.asarray(active[track_id].state[:3], float) for track_id in track_ids]
    )
    distances = np.linalg.norm(
        detection_xyz[:, np.newaxis, :] - track_xyz[np.newaxis, :, :],
        axis=2,
    )
    finite = np.isfinite(detection_xyz).all(axis=1)[:, np.newaxis] & np.isfinite(
        track_xyz
    ).all(axis=1)[np.newaxis, :]
    eligible = finite & np.isfinite(distances) & (
        distances <= config.max_association_distance_m
    )
    detection_indices, track_columns = _cardinality_first_assignment(
        distances,
        eligible,
        max_distance=config.max_association_distance_m,
    )
    return {
        int(detection_index): track_ids[int(track_column)]
        for detection_index, track_column in zip(detection_indices, track_columns)
    }


def _run_multi_sequence(
    candidates: pd.DataFrame,
    sequence_truth: pd.DataFrame | None,
    *,
    config: Any,
) -> pd.DataFrame:
    """Run one sequence with globally optimal gated association per frame."""

    candidates = candidates.loc[_IMPL._finite_position_mask(candidates)].copy()
    if candidates.empty:
        return pd.DataFrame()
    next_track_id = 1
    active: dict[int, Any] = {}
    last_update: dict[int, float] = {}
    records: list[dict[str, Any]] = []
    for time_s, group in candidates.sort_values("time_s").groupby("time_s", sort=True):
        time_s = float(time_s)
        for track_id in list(active):
            if time_s - last_update[track_id] > config.max_track_age_s:
                active.pop(track_id, None)
                last_update.pop(track_id, None)
        for filt in active.values():
            filt.predict(time_s)

        detections = group.copy()
        detections["_mot_confidence"] = _IMPL._mot_confidence_values(detections)
        detections = detections.sort_values(
            "_mot_confidence",
            ascending=False,
            kind="mergesort",
        ).reset_index(drop=True)
        matched_tracks = _optimal_track_matches(detections, active, config)

        for detection_index, detection in detections.iterrows():
            z = detection[["x_m", "y_m", "z_m"]].to_numpy(float)
            output_track_id = matched_tracks.get(int(detection_index))
            action = "matched_update"
            if output_track_id is None:
                confidence = float(detection["_mot_confidence"])
                if confidence < config.min_new_track_confidence:
                    continue
                output_track_id = next_track_id
                next_track_id += 1
                active[output_track_id] = _IMPL._ConstantVelocityFilter(
                    acceleration_std_mps2=config.acceleration_std_mps2,
                    initial_time_s=time_s,
                    initial_position=z,
                )
                action = "new_track"
            filt = active[output_track_id]
            std_xy = _IMPL._positive_float(
                detection.get("std_xy_m", 10.0),
                default=10.0,
            )
            std_z = _IMPL._positive_float(
                detection.get("std_z_m", std_xy),
                default=std_xy,
            )
            covariance = (
                np.diag([std_xy**2, std_xy**2, std_z**2]) * config.covariance_scale
            )
            filt.update(z, covariance)
            last_update[output_track_id] = time_s
            state = filt.state.copy()
            records.append(
                {
                    "time_s": time_s,
                    "source": detection.get("source"),
                    "track_id": detection.get("track_id"),
                    "class_name": detection.get("class_name"),
                    "output_track_id": f"mot_{output_track_id}",
                    "update_action": action,
                    "selected_path_update": True,
                    "state_x_m": state[0],
                    "state_y_m": state[1],
                    "state_z_m": state[2],
                    "v_x_mps": state[3],
                    "v_y_mps": state[4],
                    "v_z_mps": state[5],
                }
            )
    estimates = pd.DataFrame.from_records(records)
    if estimates.empty:
        return estimates
    if (
        sequence_truth is not None
        and not sequence_truth.empty
        and not _IMPL._truth_has_track_ids(sequence_truth)
    ):
        estimates = _IMPL.add_truth_errors(estimates, sequence_truth)
    return estimates


def _greedy_truth_matches(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    max_distance_m: float,
) -> list[tuple[int, int, float]]:
    """Return cardinality-first optimal matches under the maintained gates."""

    max_distance_m = _validated_match_distance_m(max_distance_m)
    if pred.empty or gt.empty:
        return []

    pred_xyz = pred[["state_x_m", "state_y_m", "state_z_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)
    gt_xyz = gt[["x_m", "y_m", "z_m"]].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(float)
    distances = np.linalg.norm(
        pred_xyz[:, np.newaxis, :] - gt_xyz[np.newaxis, :, :],
        axis=2,
    )
    finite = np.isfinite(pred_xyz).all(axis=1)[:, np.newaxis] & np.isfinite(
        gt_xyz
    ).all(axis=1)[np.newaxis, :]
    eligible = finite & np.isfinite(distances) & (distances <= max_distance_m)
    if "time_s" in pred.columns and "time_s" in gt.columns:
        pred_times = pd.to_numeric(pred["time_s"], errors="coerce").to_numpy(float)
        gt_times = pd.to_numeric(gt["time_s"], errors="coerce").to_numpy(float)
        time_deltas = np.abs(pred_times[:, np.newaxis] - gt_times[np.newaxis, :])
        eligible &= np.isfinite(time_deltas) & (
            time_deltas <= _IMPL._MOT_TIME_MATCH_ATOL_S
        )
    pred_indices, gt_indices = _cardinality_first_assignment(
        distances,
        eligible,
        max_distance=max_distance_m,
    )
    matches = [
        (int(pred_index), int(gt_index), float(distances[pred_index, gt_index]))
        for pred_index, gt_index in zip(pred_indices, gt_indices)
    ]
    return sorted(matches, key=lambda item: (item[2], item[0], item[1]))


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
_IMPL._track_count = _track_count
_IMPL._nearest_track_id = _nearest_track_id
_IMPL._cardinality_first_assignment = _cardinality_first_assignment
_IMPL._optimal_track_matches = _optimal_track_matches
_IMPL._run_multi_sequence = _run_multi_sequence
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
globals()["_track_count"] = _track_count
globals()["_nearest_track_id"] = _nearest_track_id
globals()["_cardinality_first_assignment"] = _cardinality_first_assignment
globals()["_optimal_track_matches"] = _optimal_track_matches
globals()["_run_multi_sequence"] = _run_multi_sequence
globals()["_greedy_truth_matches"] = _greedy_truth_matches
globals()["compute_multi_object_metrics"] = compute_multi_object_metrics

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
