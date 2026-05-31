"""Top-k Fortem tracklet graph association with weak-z replay.

This module is intended for raw-stream SOTA experiments, not for the frozen
Table-II artifact/proxy branch.  It turns the lessons from the artifact audit
into a truth-free method row:

* build continuous Fortem tracklets from normalized radar candidates;
* keep top-k globally consistent tracklet paths using radar metadata and motion
  plausibility only;
* replay each path through the existing asynchronous CV Kalman tracker with a
  weak radar-altitude covariance;
* optionally soft-weight RF updates by consistency with the selected radar path;
* choose the final path by tracklet objective plus innovation consistency, never
  by truth error.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement, run_async_cv_baseline
from raft_uav.baselines.smoothing import smooth_tracking_records


@dataclass(frozen=True)
class TopKWeakZTrackletConfig:
    """Configuration for top-k tracklet graph + weak-z replay."""

    top_k_paths: int = 8
    beam_width: int = 64
    max_tracklets: int = 256
    min_tracklet_length: int = 3
    max_intra_tracklet_gap_s: float = 1.5
    max_transition_gap_s: float = 30.0
    max_transition_speed_mps: float = 80.0
    max_transition_altitude_jump_m: float = 160.0
    range_gate_m: float | None = 900.0
    range_slack_m: float = 150.0
    track_switch_cost: float = 8.0
    gap_cost_per_s: float = 0.03
    speed_cost_weight: float = 0.35
    altitude_jump_cost_weight: float = 0.02
    tracklet_length_reward: float = 0.08
    catprob_reward_weight: float = 3.0
    confidence_reward_weight: float = 1.0
    range_penalty_weight: float = 2.0
    weakz_radar_xy_std_m: float = 360.0
    weakz_radar_z_std_m: float = 20000.0
    acceleration_std_mps2: float = 14.0
    smoother: str = "fixed-lag"
    smoother_lag_s: float = 15.0
    smoother_acceleration_std_mps2: float = 28.0
    rf_soft_weight: bool = True
    rf_radar_consistency_std_m: float = 160.0
    rf_min_reliability: float = 0.05
    rf_max_covariance_scale: float = 50.0
    rf_outside_radar_scale: float = 6.0
    rf_reject_distance_m: float | None = None
    replay_nis_weight: float = 0.05
    replay_rejection_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.top_k_paths < 1:
            raise ValueError("top_k_paths must be positive")
        if self.beam_width < self.top_k_paths:
            raise ValueError("beam_width must be >= top_k_paths")
        if self.max_tracklets < 1:
            raise ValueError("max_tracklets must be positive")
        if self.min_tracklet_length < 1:
            raise ValueError("min_tracklet_length must be positive")
        positive = (
            "max_intra_tracklet_gap_s",
            "max_transition_gap_s",
            "max_transition_speed_mps",
            "max_transition_altitude_jump_m",
            "range_slack_m",
            "weakz_radar_xy_std_m",
            "weakz_radar_z_std_m",
            "acceleration_std_mps2",
            "smoother_lag_s",
            "smoother_acceleration_std_mps2",
            "rf_radar_consistency_std_m",
            "rf_min_reliability",
            "rf_max_covariance_scale",
            "rf_outside_radar_scale",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        nonnegative = (
            "track_switch_cost",
            "gap_cost_per_s",
            "speed_cost_weight",
            "altitude_jump_cost_weight",
            "tracklet_length_reward",
            "catprob_reward_weight",
            "confidence_reward_weight",
            "range_penalty_weight",
            "replay_nis_weight",
            "replay_rejection_penalty",
        )
        for name in nonnegative:
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.range_gate_m is not None and float(self.range_gate_m) <= 0.0:
            raise ValueError("range_gate_m must be positive or None")
        if self.rf_reject_distance_m is not None and float(self.rf_reject_distance_m) <= 0.0:
            raise ValueError("rf_reject_distance_m must be positive or None")


@dataclass(frozen=True)
class TrackletSegment:
    """Continuous same-track-id radar segment."""

    segment_id: int
    track_id: int | None
    row_indices: tuple[int, ...]
    start_time_s: float
    end_time_s: float
    start_position_m: np.ndarray
    end_position_m: np.ndarray
    mean_catprob: float
    mean_confidence: float
    mean_range_m: float
    unary_cost: float

    @property
    def row_count(self) -> int:
        return len(self.row_indices)


@dataclass(frozen=True)
class TrackletPath:
    """A globally consistent sequence of radar tracklets."""

    path_id: int
    segment_ids: tuple[int, ...]
    cost: float


@dataclass(frozen=True)
class TopKWeakZResult:
    """Result from a top-k weak-z replay run."""

    records: list[dict[str, object]]
    filtered_records: list[dict[str, object]]
    selected_radar: pd.DataFrame
    attempted_radar: pd.DataFrame
    path_diagnostics: pd.DataFrame
    tracklet_diagnostics: pd.DataFrame
    selected_path_summary: dict[str, object]


def run_topk_tracklet_graph_weakz_smoother(
    *,
    rf_measurements: Iterable[TrackingMeasurement],
    radar: pd.DataFrame,
    config: TopKWeakZTrackletConfig | None = None,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    max_residual_norms_by_source: Mapping[str, float | None] | None = None,
) -> TopKWeakZResult:
    """Run top-k tracklet graph association and return the best weak-z replay.

    The selected path is chosen by truth-free path cost plus replay innovation
    consistency.  Truth should only be used by callers for post-run metrics.
    """

    cfg = config or TopKWeakZTrackletConfig()
    rf = list(rf_measurements)
    tracklets = build_fortem_tracklets(radar, cfg)
    tracklet_diagnostics = tracklets_to_frame(tracklets)
    paths = top_k_tracklet_paths(tracklets, cfg)
    if not paths:
        records = _run_replay(
            measurements=rf,
            cfg=cfg,
            gate_probabilities_by_source=gate_probabilities_by_source,
            safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
            robust_update_by_source=robust_update_by_source,
            inflation_alpha_by_source=inflation_alpha_by_source,
            max_residual_norms_by_source=max_residual_norms_by_source,
        )
        smoothed = _smooth_records(records, cfg)
        return TopKWeakZResult(
            records=smoothed,
            filtered_records=records,
            selected_radar=radar.iloc[0:0].copy(),
            attempted_radar=radar.iloc[0:0].copy(),
            path_diagnostics=pd.DataFrame(),
            tracklet_diagnostics=tracklet_diagnostics,
            selected_path_summary={"path_id": None, "reason": "no_tracklet_path"},
        )

    path_rows: list[dict[str, object]] = []
    path_results: list[
        tuple[float, TrackletPath, list[dict[str, object]], list[dict[str, object]], pd.DataFrame]
    ] = []
    segment_by_id = {segment.segment_id: segment for segment in tracklets}
    for path in paths:
        selected = selected_radar_for_tracklet_path(radar, path, segment_by_id)
        radar_measurements = radar_tracklet_measurements(selected, cfg)
        weighted_rf = rf_measurements_with_radar_reliability(rf, selected, cfg)
        filtered_records = _run_replay(
            measurements=[*weighted_rf, *radar_measurements],
            cfg=cfg,
            gate_probabilities_by_source=gate_probabilities_by_source,
            safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
            robust_update_by_source=robust_update_by_source,
            inflation_alpha_by_source=inflation_alpha_by_source,
            max_residual_norms_by_source=max_residual_norms_by_source,
        )
        smoothed_records = _smooth_records(filtered_records, cfg)
        replay_score = replay_innovation_score(filtered_records, cfg)
        final_score = float(path.cost) + replay_score
        path_rows.append(
            {
                "path_id": int(path.path_id),
                "segment_ids": ",".join(str(value) for value in path.segment_ids),
                "segment_count": int(len(path.segment_ids)),
                "radar_rows": int(len(selected)),
                "path_cost": float(path.cost),
                "replay_score": float(replay_score),
                "final_score": float(final_score),
                "posterior_records": int(len(smoothed_records)),
                "filtered_records": int(len(filtered_records)),
                "rf_measurements_after_reliability": int(len(weighted_rf)),
                "radar_measurements": int(len(radar_measurements)),
                "track_switches": int(_path_track_switches(path, segment_by_id)),
            }
        )
        path_results.append((final_score, path, smoothed_records, filtered_records, selected))

    path_results.sort(key=lambda item: (item[0], item[1].cost, item[1].path_id))
    best_score, best_path, best_records, best_filtered_records, best_selected = path_results[0]
    path_diagnostics = pd.DataFrame.from_records(path_rows).sort_values(
        ["final_score", "path_cost", "path_id"],
        na_position="last",
    ).reset_index(drop=True)
    best_selected = best_selected.copy()
    best_selected["topk_weakz_selected_path_id"] = int(best_path.path_id)
    attempted = pd.concat(
        [selected_radar_for_tracklet_path(radar, path, segment_by_id) for path in paths],
        ignore_index=True,
    ) if paths else radar.iloc[0:0].copy()
    selected_summary = {
        "path_id": int(best_path.path_id),
        "segment_ids": list(best_path.segment_ids),
        "path_cost": float(best_path.cost),
        "final_score": float(best_score),
        "radar_rows": int(len(best_selected)),
        "track_switches": int(_path_track_switches(best_path, segment_by_id)),
        "weakz_radar_xy_std_m": float(cfg.weakz_radar_xy_std_m),
        "weakz_radar_z_std_m": float(cfg.weakz_radar_z_std_m),
        "smoother": cfg.smoother,
        "smoother_lag_s": float(cfg.smoother_lag_s),
        "smoother_acceleration_std_mps2": float(cfg.smoother_acceleration_std_mps2),
    }
    return TopKWeakZResult(
        records=best_records,
        filtered_records=best_filtered_records,
        selected_radar=best_selected,
        attempted_radar=attempted,
        path_diagnostics=path_diagnostics,
        tracklet_diagnostics=tracklet_diagnostics,
        selected_path_summary=selected_summary,
    )


def build_fortem_tracklets(
    radar: pd.DataFrame,
    config: TopKWeakZTrackletConfig | None = None,
) -> list[TrackletSegment]:
    """Split normalized radar candidates into continuous same-track-id segments."""

    cfg = config or TopKWeakZTrackletConfig()
    if radar.empty:
        return []
    required = {"time_s", "east_m", "north_m", "up_m"}
    missing = required.difference(radar.columns)
    if missing:
        raise ValueError(f"radar frame missing required columns: {sorted(missing)}")

    frame = radar.copy().reset_index(drop=False).rename(columns={"index": "radar_row_index"})
    for column in ("time_s", "east_m", "north_m", "up_m", "range_m", "track_id"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(
        frame[["time_s", "east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    ).all(axis=1)
    frame = frame.loc[finite].copy()
    if cfg.range_gate_m is not None and "range_m" in frame.columns:
        frame = frame.loc[
            (~np.isfinite(frame["range_m"]))
            | (frame["range_m"] <= float(cfg.range_gate_m) + float(cfg.range_slack_m))
        ].copy()
    if frame.empty:
        return []

    group_column = "track_id" if "track_id" in frame.columns else None
    if group_column is None:
        frame["_synthetic_track_id"] = np.arange(len(frame), dtype=int)
        group_column = "_synthetic_track_id"

    segments: list[TrackletSegment] = []
    segment_id = 0
    grouped = frame.sort_values([group_column, "time_s"]).groupby(group_column, dropna=False)
    for track_id_value, group in grouped:
        rows = group.sort_values("time_s")
        current: list[pd.Series] = []
        last_time: float | None = None
        for _, row in rows.iterrows():
            time_s = float(row["time_s"])
            if last_time is not None and time_s - last_time > float(cfg.max_intra_tracklet_gap_s):
                segment = _make_tracklet_segment(current, segment_id, cfg)
                if segment is not None:
                    segments.append(segment)
                    segment_id += 1
                current = []
            current.append(row)
            last_time = time_s
        segment = _make_tracklet_segment(current, segment_id, cfg)
        if segment is not None:
            segments.append(segment)
            segment_id += 1

    segments.sort(
        key=lambda segment: (segment.start_time_s, segment.end_time_s, segment.unary_cost)
    )
    if len(segments) > cfg.max_tracklets:
        # Keep high-quality long segments but preserve chronological order afterwards.
        quality_sorted = sorted(
            segments,
            key=lambda segment: (segment.unary_cost, -segment.row_count, segment.start_time_s),
        )[: cfg.max_tracklets]
        keep = {segment.segment_id for segment in quality_sorted}
        segments = [segment for segment in segments if segment.segment_id in keep]
    return segments


def top_k_tracklet_paths(
    tracklets: Sequence[TrackletSegment],
    config: TopKWeakZTrackletConfig | None = None,
) -> list[TrackletPath]:
    """Return top-k globally consistent tracklet paths by dynamic beam search."""

    cfg = config or TopKWeakZTrackletConfig()
    if not tracklets:
        return []
    ordered = sorted(tracklets, key=lambda segment: (segment.start_time_s, segment.unary_cost))
    partials: list[tuple[float, tuple[int, ...]]] = []
    by_id = {segment.segment_id: segment for segment in ordered}
    for segment in ordered:
        candidates: list[tuple[float, tuple[int, ...]]] = [
            (segment.unary_cost, (segment.segment_id,))
        ]
        for cost, ids in partials:
            previous = by_id[ids[-1]]
            edge_cost = _tracklet_transition_cost(previous, segment, cfg)
            if edge_cost is None:
                continue
            candidates.append((cost + edge_cost + segment.unary_cost, (*ids, segment.segment_id)))
        partials.extend(candidates)
        partials = _dedupe_and_prune_paths(partials, cfg.beam_width)
    partials = _dedupe_and_prune_paths(partials, max(cfg.top_k_paths, cfg.beam_width))
    paths = [
        TrackletPath(path_id=index + 1, segment_ids=ids, cost=float(cost))
        for index, (cost, ids) in enumerate(partials[: cfg.top_k_paths])
    ]
    return paths


def selected_radar_for_tracklet_path(
    radar: pd.DataFrame,
    path: TrackletPath,
    segment_by_id: Mapping[int, TrackletSegment],
) -> pd.DataFrame:
    """Return radar rows selected by a path."""

    indices: list[int] = []
    for segment_id in path.segment_ids:
        indices.extend(segment_by_id[int(segment_id)].row_indices)
    if not indices:
        return radar.iloc[0:0].copy()
    selected = radar.iloc[indices].copy()
    segment_lookup: dict[int, int] = {}
    path_order_lookup: dict[int, int] = {}
    for path_order, segment_id in enumerate(path.segment_ids):
        for index in segment_by_id[int(segment_id)].row_indices:
            segment_lookup[int(index)] = int(segment_id)
            path_order_lookup[int(index)] = int(path_order)
    selected["topk_weakz_segment_id"] = [segment_lookup.get(int(i), -1) for i in selected.index]
    selected["topk_weakz_path_order"] = [path_order_lookup.get(int(i), -1) for i in selected.index]
    selected["topk_weakz_path_cost"] = float(path.cost)
    return selected.sort_values(["time_s", "topk_weakz_path_order"]).reset_index(drop=True)


def radar_tracklet_measurements(
    selected_radar: pd.DataFrame,
    config: TopKWeakZTrackletConfig | None = None,
) -> list[TrackingMeasurement]:
    """Convert selected radar rows into weak-z 3D measurements."""

    cfg = config or TopKWeakZTrackletConfig()
    if selected_radar.empty:
        return []
    covariance = np.diag(
        [
            float(cfg.weakz_radar_xy_std_m) ** 2,
            float(cfg.weakz_radar_xy_std_m) ** 2,
            float(cfg.weakz_radar_z_std_m) ** 2,
        ]
    )
    measurements: list[TrackingMeasurement] = []
    for _, row in selected_radar.iterrows():
        vector = np.array([float(row["east_m"]), float(row["north_m"]), float(row["up_m"])])
        if not np.isfinite(vector).all() or not np.isfinite(float(row["time_s"])):
            continue
        measurements.append(
            TrackingMeasurement(
                time_s=float(row["time_s"]),
                vector=vector,
                covariance=covariance,
                source="radar",
            )
        )
    return measurements


def rf_measurements_with_radar_reliability(
    rf_measurements: Sequence[TrackingMeasurement],
    selected_radar: pd.DataFrame,
    config: TopKWeakZTrackletConfig | None = None,
) -> list[TrackingMeasurement]:
    """Soft-weight RF covariance using the selected radar path as a truth-free cue."""

    cfg = config or TopKWeakZTrackletConfig()
    rf = list(rf_measurements)
    if not cfg.rf_soft_weight or not rf or selected_radar.empty:
        return rf
    radar_path = _radar_path_frame(selected_radar)
    if len(radar_path) < 2:
        return [
            _scaled_measurement(measurement, cfg.rf_outside_radar_scale)
            for measurement in rf
        ]

    times = radar_path["time_s"].to_numpy(dtype=float)
    east = radar_path["east_m"].to_numpy(dtype=float)
    north = radar_path["north_m"].to_numpy(dtype=float)
    out: list[TrackingMeasurement] = []
    for measurement in rf:
        t = float(measurement.time_s)
        if t < times[0] or t > times[-1]:
            out.append(_scaled_measurement(measurement, cfg.rf_outside_radar_scale))
            continue
        rf_position = measurement.vector[:2]
        interp = np.array([np.interp(t, times, east), np.interp(t, times, north)], dtype=float)
        distance = float(np.linalg.norm(rf_position - interp))
        if cfg.rf_reject_distance_m is not None and distance > float(cfg.rf_reject_distance_m):
            continue
        reliability = math.exp(
            -0.5 * (distance / float(cfg.rf_radar_consistency_std_m)) ** 2
        )
        reliability = max(float(cfg.rf_min_reliability), reliability)
        covariance_scale = min(float(cfg.rf_max_covariance_scale), 1.0 / reliability)
        out.append(_scaled_measurement(measurement, covariance_scale))
    return out


def replay_innovation_score(
    records: Sequence[Mapping[str, object]],
    config: TopKWeakZTrackletConfig | None = None,
) -> float:
    """Truth-free replay score from update diagnostics."""

    cfg = config or TopKWeakZTrackletConfig()
    if not records:
        return float("inf")
    nis_values: list[float] = []
    rejected = 0
    for record in records:
        nis = record.get("nis")
        try:
            value = float(nis)
        except (TypeError, ValueError):
            value = np.nan
        if np.isfinite(value):
            nis_values.append(min(value, 100.0))
        accepted = record.get("accepted", True)
        if accepted is False:
            rejected += 1
    mean_nis = float(np.mean(nis_values)) if nis_values else 0.0
    return float(cfg.replay_nis_weight) * mean_nis + float(cfg.replay_rejection_penalty) * rejected


def tracklets_to_frame(tracklets: Sequence[TrackletSegment]) -> pd.DataFrame:
    """Return CSV-friendly tracklet diagnostics."""

    records = []
    for segment in tracklets:
        records.append(
            {
                "segment_id": int(segment.segment_id),
                "track_id": segment.track_id,
                "row_count": int(segment.row_count),
                "start_time_s": float(segment.start_time_s),
                "end_time_s": float(segment.end_time_s),
                "duration_s": float(segment.end_time_s - segment.start_time_s),
                "mean_catprob": float(segment.mean_catprob),
                "mean_confidence": float(segment.mean_confidence),
                "mean_range_m": float(segment.mean_range_m),
                "unary_cost": float(segment.unary_cost),
            }
        )
    return pd.DataFrame.from_records(records)


def _make_tracklet_segment(
    rows: list[pd.Series],
    segment_id: int,
    cfg: TopKWeakZTrackletConfig,
) -> TrackletSegment | None:
    if len(rows) < cfg.min_tracklet_length:
        return None
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("time_s")
    positions = frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    times = frame["time_s"].to_numpy(dtype=float)
    if not np.isfinite(positions).all() or not np.isfinite(times).all():
        return None
    track_id = _optional_int(frame["track_id"].iloc[0]) if "track_id" in frame.columns else None
    catprob = _finite_mean(frame.get("cat_prob_uav"), default=0.0)
    confidence = _finite_mean(frame.get("confidence"), default=0.0)
    mean_range = _finite_mean(frame.get("range_m"), default=np.nan)
    range_cost = 0.0
    if cfg.range_gate_m is not None and np.isfinite(mean_range):
        excess = max(0.0, mean_range - float(cfg.range_gate_m))
        range_cost = float(cfg.range_penalty_weight) * excess / max(float(cfg.range_slack_m), 1.0)
    unary = (
        range_cost
        - float(cfg.tracklet_length_reward) * math.log1p(len(frame))
        - float(cfg.catprob_reward_weight) * catprob
        - float(cfg.confidence_reward_weight) * confidence
    )
    row_indices = tuple(int(value) for value in frame["radar_row_index"].tolist())
    return TrackletSegment(
        segment_id=int(segment_id),
        track_id=track_id,
        row_indices=row_indices,
        start_time_s=float(times[0]),
        end_time_s=float(times[-1]),
        start_position_m=positions[0].copy(),
        end_position_m=positions[-1].copy(),
        mean_catprob=float(catprob),
        mean_confidence=float(confidence),
        mean_range_m=float(mean_range),
        unary_cost=float(unary),
    )


def _tracklet_transition_cost(
    previous: TrackletSegment,
    current: TrackletSegment,
    cfg: TopKWeakZTrackletConfig,
) -> float | None:
    gap_s = float(current.start_time_s - previous.end_time_s)
    if gap_s <= 0.0 or gap_s > float(cfg.max_transition_gap_s):
        return None
    delta = current.start_position_m - previous.end_position_m
    distance = float(np.linalg.norm(delta[:2]))
    speed = distance / max(gap_s, 1.0e-6)
    altitude_jump = abs(float(delta[2]))
    if speed > float(cfg.max_transition_speed_mps) or altitude_jump > float(
        cfg.max_transition_altitude_jump_m
    ):
        return None
    switch_cost = 0.0
    if previous.track_id is None or current.track_id is None:
        switch_cost = 0.5 * float(cfg.track_switch_cost)
    elif previous.track_id != current.track_id:
        switch_cost = float(cfg.track_switch_cost)
    speed_cost = float(cfg.speed_cost_weight) * (speed / float(cfg.max_transition_speed_mps)) ** 2
    altitude_cost = float(cfg.altitude_jump_cost_weight) * altitude_jump
    return float(cfg.gap_cost_per_s) * gap_s + switch_cost + speed_cost + altitude_cost


def _dedupe_and_prune_paths(
    paths: list[tuple[float, tuple[int, ...]]],
    limit: int,
) -> list[tuple[float, tuple[int, ...]]]:
    best_by_ids: dict[tuple[int, ...], float] = {}
    for cost, ids in paths:
        old = best_by_ids.get(ids)
        if old is None or cost < old:
            best_by_ids[ids] = float(cost)
    return sorted(
        [(cost, ids) for ids, cost in best_by_ids.items()],
        key=lambda item: (item[0], len(item[1]), item[1]),
    )[: int(limit)]


def _path_track_switches(path: TrackletPath, segment_by_id: Mapping[int, TrackletSegment]) -> int:
    switches = 0
    previous: int | None = None
    for segment_id in path.segment_ids:
        track_id = segment_by_id[int(segment_id)].track_id
        if previous is not None and track_id is not None and track_id != previous:
            switches += 1
        if track_id is not None:
            previous = track_id
    return switches


def _run_replay(
    *,
    measurements: Iterable[TrackingMeasurement],
    cfg: TopKWeakZTrackletConfig,
    gate_probabilities_by_source: Mapping[str, float | None] | None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None,
    robust_update_by_source: Mapping[str, str | None] | None,
    inflation_alpha_by_source: Mapping[str, float] | None,
    max_residual_norms_by_source: Mapping[str, float | None] | None,
) -> list[dict[str, object]]:
    measurements_list = sorted(list(measurements), key=lambda measurement: measurement.time_s)
    if not measurements_list:
        return []
    return run_async_cv_baseline(
        measurements_list,
        acceleration_std_mps2=float(cfg.acceleration_std_mps2),
        gate_probabilities_by_source=gate_probabilities_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
    )


def _smooth_records(
    records: list[dict[str, object]],
    cfg: TopKWeakZTrackletConfig,
) -> list[dict[str, object]]:
    if not records:
        return []
    return smooth_tracking_records(
        records,
        method=cfg.smoother,
        lag_s=float(cfg.smoother_lag_s),
        acceleration_std_mps2=float(cfg.smoother_acceleration_std_mps2),
    )


def _radar_path_frame(selected_radar: pd.DataFrame) -> pd.DataFrame:
    return (
        selected_radar[["time_s", "east_m", "north_m"]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby("time_s", as_index=False)
        .mean(numeric_only=True)
        .sort_values("time_s")
        .reset_index(drop=True)
    )


def _scaled_measurement(
    measurement: TrackingMeasurement,
    covariance_scale: float,
) -> TrackingMeasurement:
    return TrackingMeasurement(
        time_s=measurement.time_s,
        vector=measurement.vector.copy(),
        covariance=measurement.covariance.copy() * float(covariance_scale),
        source=measurement.source,
    )


def _finite_mean(values: object, *, default: float) -> float:
    if values is None:
        return float(default)
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float(default)
    return float(np.mean(array))


def _optional_int(value: object) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number)


def records_to_frame(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Return a CSV-friendly posterior frame."""

    rows: list[dict[str, object]] = []
    for record in records:
        state = np.asarray(record.get("state"), dtype=float).reshape(6)
        rows.append(
            {
                "time_s": float(record.get("time_s", np.nan)),
                "source": record.get("source"),
                "accepted": record.get("accepted"),
                "update_action": record.get("update_action"),
                "east_m": float(state[0]),
                "north_m": float(state[1]),
                "up_m": float(state[2]),
                "v_east_mps": float(state[3]),
                "v_north_mps": float(state[4]),
                "v_up_mps": float(state[5]),
                "nis": record.get("nis"),
                "residual_norm_m": record.get("residual_norm_m"),
            }
        )
    return pd.DataFrame.from_records(rows)
