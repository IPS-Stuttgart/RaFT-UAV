"""Small multi-object tracking baseline for MMUAD-style exported detections.

This is intentionally modest: it is a tracking-by-detection backend for smoke
experiments and ablations, not a replacement for the official UG2+ evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import CandidateFrame, TruthFrame
from raft_uav.mmuad.tracker import (
    TrackerOutput,
    _ConstantVelocityFilter,
    add_truth_errors,
    compute_metrics,
)


@dataclass(frozen=True)
class MultiObjectTrackerConfig:
    """Configuration for the simple greedy multi-object tracker."""

    acceleration_std_mps2: float = 8.0
    max_association_distance_m: float = 15.0
    max_track_age_s: float = 1.5
    min_new_track_confidence: float = 0.05
    covariance_scale: float = 1.0


def run_mmuad_multi_object_tracker(
    candidates: CandidateFrame,
    truth: TruthFrame | None = None,
    *,
    config: MultiObjectTrackerConfig | None = None,
) -> TrackerOutput:
    """Run a basic greedy multi-object tracker over normalized detections."""

    config = config or MultiObjectTrackerConfig()
    candidates.validate()
    rows = candidates.rows.copy()
    if rows.empty:
        return TrackerOutput(pd.DataFrame(), {"count": 0}, pd.DataFrame())
    truth_rows = truth.rows if truth is not None else None
    estimate_frames: list[pd.DataFrame] = []
    metrics_by_sequence: dict[str, Any] = {}
    for sequence_id, sequence_candidates in rows.groupby("sequence_id", sort=True):
        sequence_truth = None
        if truth_rows is not None:
            sequence_truth = truth_rows.loc[truth_rows["sequence_id"] == sequence_id]
        estimates = _run_multi_sequence(sequence_candidates, sequence_truth, config=config)
        estimates["sequence_id"] = sequence_id
        estimate_frames.append(estimates)
        metrics_by_sequence[str(sequence_id)] = compute_multi_object_metrics(estimates, sequence_truth)
    estimates_all = pd.concat(estimate_frames, ignore_index=True) if estimate_frames else pd.DataFrame()
    metrics = {
        "sequences": metrics_by_sequence,
        "pooled": compute_multi_object_metrics(estimates_all, truth_rows),
    }
    selected = _selected_frame_from_estimates(estimates_all)
    return TrackerOutput(estimates_all, metrics, selected)


def _run_multi_sequence(
    candidates: pd.DataFrame,
    sequence_truth: pd.DataFrame | None,
    *,
    config: MultiObjectTrackerConfig,
) -> pd.DataFrame:
    next_track_id = 1
    active: dict[int, _ConstantVelocityFilter] = {}
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
        unmatched_tracks = set(active)
        detections = group.sort_values("confidence", ascending=False).copy()
        for _, detection in detections.iterrows():
            z = detection[["x_m", "y_m", "z_m"]].to_numpy(float)
            output_track_id = _nearest_track_id(z, active, unmatched_tracks, config)
            action = "matched_update"
            if output_track_id is None:
                confidence = float(detection.get("confidence", 1.0))
                if confidence < config.min_new_track_confidence:
                    continue
                output_track_id = next_track_id
                next_track_id += 1
                active[output_track_id] = _ConstantVelocityFilter(
                    acceleration_std_mps2=config.acceleration_std_mps2,
                    initial_time_s=time_s,
                    initial_position=z,
                )
                action = "new_track"
            filt = active[output_track_id]
            std_xy = float(detection.get("std_xy_m", 10.0))
            std_z = float(detection.get("std_z_m", std_xy))
            covariance = np.diag([std_xy**2, std_xy**2, std_z**2]) * config.covariance_scale
            filt.update(z, covariance)
            last_update[output_track_id] = time_s
            unmatched_tracks.discard(output_track_id)
            state = filt.state.copy()
            records.append(
                {
                    "time_s": time_s,
                    "source": detection.get("source"),
                    "track_id": detection.get("track_id"),
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
    if sequence_truth is not None and not sequence_truth.empty and "track_id" not in sequence_truth.columns:
        estimates = add_truth_errors(estimates, sequence_truth)
    return estimates


def _nearest_track_id(
    z: np.ndarray,
    active: dict[int, _ConstantVelocityFilter],
    unmatched_tracks: set[int],
    config: MultiObjectTrackerConfig,
) -> int | None:
    best_track: int | None = None
    best_distance = float("inf")
    for track_id in unmatched_tracks:
        distance = float(np.linalg.norm(active[track_id].state[:3] - z))
        if distance < best_distance:
            best_distance = distance
            best_track = track_id
    if best_distance <= config.max_association_distance_m:
        return best_track
    return None


def compute_multi_object_metrics(
    estimates: pd.DataFrame,
    truth: pd.DataFrame | None,
    *,
    match_distance_m: float = 25.0,
) -> dict[str, Any]:
    """Compute simple MOT-style metrics for exported multi-object truth.

    If truth lacks object IDs, this falls back to the single-trajectory metrics
    used by the MMUAD smoke tracker.
    """

    if estimates.empty:
        return {"count": 0}
    if truth is None or truth.empty:
        return {"count": int(len(estimates)), "track_count": int(estimates["output_track_id"].nunique())}
    if "track_id" not in truth.columns:
        return compute_metrics(add_truth_errors(estimates.copy(), truth), truth)
    matched_distances: list[float] = []
    false_positive = 0
    false_negative = 0
    id_switches = 0
    gt_last_match: dict[str, str] = {}
    for time_s in sorted(set(estimates["time_s"]).union(set(truth["time_s"]))):
        pred = estimates.loc[np.isclose(estimates["time_s"], time_s)].copy()
        gt = truth.loc[np.isclose(truth["time_s"], time_s)].copy()
        matches = _greedy_truth_matches(pred, gt, max_distance_m=match_distance_m)
        matched_pred = {pred_idx for pred_idx, _, _ in matches}
        matched_gt = {gt_idx for _, gt_idx, _ in matches}
        false_positive += int(len(pred) - len(matched_pred))
        false_negative += int(len(gt) - len(matched_gt))
        for pred_idx, gt_idx, distance in matches:
            pred_track = str(pred.iloc[pred_idx]["output_track_id"])
            gt_track = str(gt.iloc[gt_idx]["track_id"])
            if gt_track in gt_last_match and gt_last_match[gt_track] != pred_track:
                id_switches += 1
            gt_last_match[gt_track] = pred_track
            matched_distances.append(float(distance))
    gt_count = int(len(truth))
    match_count = len(matched_distances)
    mota = 1.0 - (false_positive + false_negative + id_switches) / max(1, gt_count)
    return {
        "count": int(len(estimates)),
        "gt_count": gt_count,
        "track_count": int(estimates["output_track_id"].nunique()),
        "matches": int(match_count),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "id_switches": int(id_switches),
        "mota_like": float(mota),
        "motp_3d_m": float(np.mean(matched_distances)) if matched_distances else float("nan"),
        "recall": float(match_count / max(1, gt_count)),
        "precision": float(match_count / max(1, len(estimates))),
    }


def _greedy_truth_matches(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    max_distance_m: float,
) -> list[tuple[int, int, float]]:
    if pred.empty or gt.empty:
        return []
    pred_xyz = pred[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    gt_xyz = gt[["x_m", "y_m", "z_m"]].to_numpy(float)
    pairs: list[tuple[float, int, int]] = []
    for pred_idx, p in enumerate(pred_xyz):
        for gt_idx, g in enumerate(gt_xyz):
            distance = float(np.linalg.norm(p - g))
            if distance <= max_distance_m:
                pairs.append((distance, pred_idx, gt_idx))
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for distance, pred_idx, gt_idx in sorted(pairs):
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, distance))
    return matches


def _selected_frame_from_estimates(estimates: pd.DataFrame) -> pd.DataFrame:
    if estimates.empty:
        return pd.DataFrame()
    columns = ["sequence_id", "time_s", "source", "track_id", "output_track_id"]
    return estimates.loc[estimates["selected_path_update"].astype(bool), columns].copy()
