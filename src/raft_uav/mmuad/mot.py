"""Small multi-object tracking baseline for MMUAD-style exported detections.

This is intentionally modest: it is a tracking-by-detection backend for smoke
experiments and ablations, not a replacement for the official UG2+ evaluator.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import CandidateFrame, TruthFrame
from raft_uav.mmuad.tracker import (
    TrackerOutput,
    _ConstantVelocityFilter,
    _candidate_rows_with_optional_defaults,
    _finite_position_mask,
    _positive_float,
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

    def __post_init__(self) -> None:
        for name in (
            "acceleration_std_mps2",
            "max_association_distance_m",
            "max_track_age_s",
            "min_new_track_confidence",
        ):
            _require_nonnegative_config(name, getattr(self, name))
        _require_positive_config("covariance_scale", self.covariance_scale)


def _finite_config_float(name: str, value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_nonnegative_config(name: str, value: object) -> float:
    number = _finite_config_float(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _require_positive_config(name: str, value: object) -> float:
    number = _finite_config_float(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def run_mmuad_multi_object_tracker(
    candidates: CandidateFrame,
    truth: TruthFrame | None = None,
    *,
    config: MultiObjectTrackerConfig | None = None,
) -> TrackerOutput:
    """Run a basic greedy multi-object tracker over normalized detections."""

    config = config or MultiObjectTrackerConfig()
    candidates.validate()
    rows = _candidate_rows_with_optional_defaults(candidates.rows)
    truth_rows = truth.rows if truth is not None else None
    if rows.empty:
        empty = pd.DataFrame()
        metrics = {
            "sequences": _truth_only_sequence_metrics(truth_rows),
            "pooled": compute_multi_object_metrics(empty, truth_rows),
        }
        return TrackerOutput(empty, metrics, empty)
    estimate_frames: list[pd.DataFrame] = []
    metrics_by_sequence: dict[str, Any] = {}
    for sequence_id, sequence_candidates in rows.groupby("sequence_id", sort=True):
        sequence_truth = None
        if truth_rows is not None:
            sequence_truth = truth_rows.loc[truth_rows["sequence_id"] == sequence_id]
        estimates = _run_multi_sequence(sequence_candidates, sequence_truth, config=config)
        estimates["sequence_id"] = sequence_id
        estimate_frames.append(estimates)
        metrics_by_sequence[str(sequence_id)] = compute_multi_object_metrics(
            estimates,
            sequence_truth,
        )
    for sequence_id, metrics in _truth_only_sequence_metrics(truth_rows).items():
        metrics_by_sequence.setdefault(sequence_id, metrics)
    estimates_all = (
        pd.concat(estimate_frames, ignore_index=True)
        if estimate_frames
        else pd.DataFrame()
    )
    metrics = {
        "sequences": metrics_by_sequence,
        "pooled": compute_multi_object_metrics(estimates_all, truth_rows),
    }
    selected = _selected_frame_from_estimates(estimates_all)
    return TrackerOutput(estimates_all, metrics, selected)


def _truth_only_sequence_metrics(truth_rows: pd.DataFrame | None) -> dict[str, Any]:
    metrics_by_sequence: dict[str, Any] = {}
    if truth_rows is None or truth_rows.empty or "sequence_id" not in truth_rows.columns:
        return metrics_by_sequence
    empty = pd.DataFrame()
    for sequence_id, sequence_truth in truth_rows.groupby("sequence_id", sort=True):
        metrics_by_sequence[str(sequence_id)] = compute_multi_object_metrics(empty, sequence_truth)
    return metrics_by_sequence


def _run_multi_sequence(
    candidates: pd.DataFrame,
    sequence_truth: pd.DataFrame | None,
    *,
    config: MultiObjectTrackerConfig,
) -> pd.DataFrame:
    candidates = candidates.loc[_finite_position_mask(candidates)].copy()
    if candidates.empty:
        return pd.DataFrame()
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
        detections = group.copy()
        detections["_mot_confidence"] = _mot_confidence_values(detections)
        detections = detections.sort_values("_mot_confidence", ascending=False)
        for _, detection in detections.iterrows():
            z = detection[["x_m", "y_m", "z_m"]].to_numpy(float)
            output_track_id = _nearest_track_id(z, active, unmatched_tracks, config)
            action = "matched_update"
            if output_track_id is None:
                confidence = float(detection["_mot_confidence"])
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
            std_xy = _positive_float(detection.get("std_xy_m", 10.0), default=10.0)
            std_z = _positive_float(detection.get("std_z_m", std_xy), default=std_xy)
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
        and not _truth_has_track_ids(sequence_truth)
    ):
        estimates = add_truth_errors(estimates, sequence_truth)
    return estimates


def _mot_confidence_values(frame: pd.DataFrame) -> np.ndarray:
    if "confidence" not in frame.columns:
        return np.ones(len(frame), dtype=float)
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").to_numpy(float)
    return np.where(np.isfinite(confidence), confidence, 0.0)


def _nearest_track_id(
    z: np.ndarray,
    active: dict[int, _ConstantVelocityFilter],
    unmatched_tracks: set[int],
    config: MultiObjectTrackerConfig,
) -> int | None:
    if not np.isfinite(z).all():
        return None
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

    if truth is None or truth.empty:
        estimates = _finite_mot_estimates(estimates)
        return {"count": int(len(estimates)), "track_count": _track_count(estimates)}
    if not _truth_has_track_ids(truth):
        if estimates.empty:
            return compute_metrics(estimates.copy(), truth)
        return compute_metrics(add_truth_errors(estimates.copy(), truth), truth)
    estimates = _finite_mot_estimates(estimates)
    truth = _finite_mot_truth(truth)
    if estimates.empty:
        return _empty_mot_prediction_metrics(truth)
    matched_distances: list[float] = []
    false_positive = 0
    false_negative = 0
    id_switches = 0
    gt_last_match: dict[str, str] = {}
    for pred, gt in _metric_frame_pairs(estimates, truth):
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
        "track_count": _track_count(estimates),
        "matches": int(match_count),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "id_switches": int(id_switches),
        "mota_like": float(mota),
        "motp_3d_m": float(np.mean(matched_distances)) if matched_distances else None,
        "recall": float(match_count / max(1, gt_count)),
        "precision": float(match_count / max(1, len(estimates))),
    }


def _empty_mot_prediction_metrics(truth: pd.DataFrame) -> dict[str, Any]:
    gt_count = int(len(truth))
    return {
        "count": 0,
        "gt_count": gt_count,
        "track_count": 0,
        "matches": 0,
        "false_positive": 0,
        "false_negative": gt_count,
        "id_switches": 0,
        "mota_like": float(1.0 - gt_count / max(1, gt_count)),
        "motp_3d_m": None,
        "recall": 0.0,
        "precision": 0.0,
    }


def _finite_mot_estimates(estimates: pd.DataFrame) -> pd.DataFrame:
    if estimates.empty:
        return estimates.copy()
    required = ["time_s", "state_x_m", "state_y_m", "state_z_m"]
    if any(column not in estimates.columns for column in required):
        return estimates.iloc[0:0].copy()
    numeric = estimates.loc[:, required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    if "output_track_id" in estimates.columns:
        finite &= estimates["output_track_id"].notna().to_numpy(dtype=bool)
    return estimates.loc[finite].copy()


def _track_count(estimates: pd.DataFrame) -> int:
    if estimates.empty or "output_track_id" not in estimates.columns:
        return 0
    return int(estimates["output_track_id"].nunique())


def _finite_mot_truth(truth: pd.DataFrame) -> pd.DataFrame:
    if truth.empty:
        return truth.copy()
    required = ["time_s", "x_m", "y_m", "z_m"]
    if any(column not in truth.columns for column in required):
        return truth.iloc[0:0].copy()
    numeric = truth.loc[:, required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    if "track_id" in truth.columns:
        finite &= truth["track_id"].notna().to_numpy(dtype=bool)
    return truth.loc[finite].copy()


def _truth_has_track_ids(truth: pd.DataFrame) -> bool:
    if "track_id" not in truth.columns:
        return False
    values = truth["track_id"]
    if values.empty:
        return False
    present = values.notna()
    if not present.any():
        return False
    text = values.loc[present].astype(str).str.strip().str.lower()
    missing_like = text.eq("") | text.isin({"nan", "none", "<na>"})
    return bool((~missing_like).any())


def _metric_frame_pairs(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield same-frame estimate/truth pairs without crossing sequence boundaries."""

    estimates = estimates.copy()
    truth = truth.copy()
    estimates["time_s"] = pd.to_numeric(estimates["time_s"], errors="coerce")
    truth["time_s"] = pd.to_numeric(truth["time_s"], errors="coerce")

    if "sequence_id" in estimates.columns and "sequence_id" in truth.columns:
        estimates["_metric_sequence_id"] = estimates["sequence_id"].astype(str)
        truth["_metric_sequence_id"] = truth["sequence_id"].astype(str)
        estimate_keys = {
            (str(sequence_id), float(time_s))
            for sequence_id, time_s in estimates[
                ["_metric_sequence_id", "time_s"]
            ].itertuples(index=False, name=None)
        }
        truth_keys = {
            (str(sequence_id), float(time_s))
            for sequence_id, time_s in truth[
                ["_metric_sequence_id", "time_s"]
            ].itertuples(index=False, name=None)
        }
        for sequence_id, time_s in sorted(estimate_keys | truth_keys):
            pred = estimates.loc[
                (estimates["_metric_sequence_id"] == sequence_id)
                & np.isclose(estimates["time_s"], time_s)
            ].copy()
            gt = truth.loc[
                (truth["_metric_sequence_id"] == sequence_id)
                & np.isclose(truth["time_s"], time_s)
            ].copy()
            yield pred, gt
        return

    for time_s in sorted(set(estimates["time_s"]).union(set(truth["time_s"]))):
        pred = estimates.loc[np.isclose(estimates["time_s"], time_s)].copy()
        gt = truth.loc[np.isclose(truth["time_s"], time_s)].copy()
        yield pred, gt


def _greedy_truth_matches(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    max_distance_m: float,
) -> list[tuple[int, int, float]]:
    if pred.empty or gt.empty:
        return []
    pred_xyz = pred[["state_x_m", "state_y_m", "state_z_m"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    gt_xyz = gt[["x_m", "y_m", "z_m"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pairs: list[tuple[float, int, int]] = []
    for pred_idx, p in enumerate(pred_xyz):
        if not np.isfinite(p).all():
            continue
        for gt_idx, g in enumerate(gt_xyz):
            if not np.isfinite(g).all():
                continue
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
