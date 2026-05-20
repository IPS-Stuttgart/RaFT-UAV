"""Oracle candidate-retention diagnostics for radar-tracklet pruning."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.radar_association import _catprob_candidate_pool, _events
from raft_uav.baselines.tracklet_viterbi import (
    TrackletViterbiAssociationConfig,
    _build_rf_anchor_states,
    _first_rf_bootstrap_index,
    _nodes_for_radar_frame,
)
from raft_uav.evaluation.radar_oracle_diagnostics import interpolate_truth_positions

_FRAME_COLUMNS = (
    "event_index",
    "event_key",
    "time_s",
    "truth_time_s",
    "truth_time_delta_s",
    "frame_candidate_count",
    "eligible_candidate_count",
    "retained_candidate_count",
    "oracle_available",
    "oracle_retained",
    "oracle_drop_reason",
    "oracle_miss_streak_before",
    "oracle_track_id",
    "oracle_track_index",
    "oracle_range_m",
    "oracle_cat_prob_uav",
    "oracle_truth_error_m",
    "oracle_truth_error_2d_m",
    "oracle_passed_catprob_threshold",
    "oracle_rank_by_catprob",
    "oracle_rank_by_unary_score",
)
_BUCKET_COLUMNS = (
    "bucket_type",
    "bucket",
    "frame_count",
    "retained_count",
    "retention_rate",
    "catprob_threshold_drop_count",
    "top_k_drop_count",
)


@dataclass(frozen=True)
class OracleCandidateCoverageResult:
    """Frame-level and aggregated oracle-retention diagnostic tables."""

    frame_coverage: pd.DataFrame
    bucket_summary: pd.DataFrame
    summary: dict[str, Any]


def build_oracle_candidate_coverage(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    rf_measurements: Iterable[TrackingMeasurement] = (),
    candidate_catprob_threshold: float | None = 0.5,
    config: TrackletViterbiAssociationConfig | None = None,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    truth_time_gate_s: float | None = 1.0,
    gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    safety_gate_probabilities_by_source: Mapping[str, float | None] | None = None,
    safety_gate_thresholds_by_source: Mapping[str, float | None] | None = None,
    robust_update_by_source: Mapping[str, str | None] | None = None,
    inflation_alpha_by_source: Mapping[str, float] | None = None,
    max_residual_norms_by_source: Mapping[str, float | None] | None = None,
) -> OracleCandidateCoverageResult:
    """Measure whether the truth-nearest radar row survives Viterbi candidate pruning.

    The diagnostic mirrors the base tracklet-Viterbi candidate construction: RF-only
    anchor states are built causally, each radar frame is scored by
    :func:`_nodes_for_radar_frame`, and only the top ``max_candidates_per_frame``
    non-miss nodes are counted as retained.  Ground truth is used only after that
    pruning step to label the closest radar row and is therefore suitable for
    offline error analysis, not for online tracking.
    """

    cfg = config or TrackletViterbiAssociationConfig()
    covariance = np.diag(
        [float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]
    )
    rf_measurement_list = list(rf_measurements)
    events = _events(rf_measurement_list, radar)
    if not events:
        return _empty_result(
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=cfg,
            truth_time_gate_s=truth_time_gate_s,
        )

    bootstrap_index = _first_rf_bootstrap_index(events)
    if bootstrap_index is None:
        return _empty_result(
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=cfg,
            truth_time_gate_s=truth_time_gate_s,
        )
    events = events[bootstrap_index:]
    anchors = _build_rf_anchor_states(
        events=events,
        acceleration_std_mps2=acceleration_std_mps2,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
    )

    rows: list[dict[str, Any]] = []
    miss_streak = 0
    for event_index, event in enumerate(events):
        if event.get("kind") != "radar":
            continue
        candidates = event["candidates"]
        assert isinstance(candidates, pd.DataFrame)
        frame_row, retained = _oracle_coverage_row(
            event_index=event_index,
            event=event,
            candidates=candidates,
            truth=truth,
            anchor=anchors.get(event_index),
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=cfg,
            truth_time_gate_s=truth_time_gate_s,
            previous_miss_streak=miss_streak,
        )
        rows.append(frame_row)
        if frame_row.get("oracle_available"):
            miss_streak = 0 if retained else miss_streak + 1

    frame = pd.DataFrame.from_records(rows, columns=_FRAME_COLUMNS)
    summary = _coverage_summary(
        frame,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=cfg,
        truth_time_gate_s=truth_time_gate_s,
    )
    return OracleCandidateCoverageResult(
        frame_coverage=frame,
        bucket_summary=_bucket_summary(frame),
        summary=summary,
    )


def _oracle_coverage_row(
    *,
    event_index: int,
    event: Mapping[str, object],
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    anchor: object | None,
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
    truth_time_gate_s: float | None,
    previous_miss_streak: int,
) -> tuple[dict[str, Any], bool]:
    time_s = float(event["time_s"])
    base: dict[str, Any] = {
        "event_index": int(event_index),
        "event_key": _event_key(candidates, time_s),
        "time_s": time_s,
        "truth_time_s": np.nan,
        "truth_time_delta_s": np.nan,
        "frame_candidate_count": int(len(candidates)),
        "eligible_candidate_count": 0,
        "retained_candidate_count": 0,
        "oracle_available": False,
        "oracle_retained": False,
        "oracle_drop_reason": "no_truth_support",
        "oracle_miss_streak_before": int(previous_miss_streak),
        "oracle_track_id": np.nan,
        "oracle_track_index": np.nan,
        "oracle_range_m": np.nan,
        "oracle_cat_prob_uav": np.nan,
        "oracle_truth_error_m": np.nan,
        "oracle_truth_error_2d_m": np.nan,
        "oracle_passed_catprob_threshold": False,
        "oracle_rank_by_catprob": np.nan,
        "oracle_rank_by_unary_score": np.nan,
    }
    truth_position, truth_time_s, truth_time_delta_s = _interpolated_truth_position(
        truth,
        time_s,
        max_time_delta_s=truth_time_gate_s,
    )
    if truth_position is None:
        return base, False
    base["truth_time_s"] = truth_time_s
    base["truth_time_delta_s"] = truth_time_delta_s

    positions = _candidate_positions(candidates)
    finite = np.isfinite(positions).all(axis=1)
    if not finite.any():
        base["oracle_drop_reason"] = "no_finite_candidates"
        return base, False

    errors = np.full(len(candidates), np.inf, dtype=float)
    errors_2d = np.full(len(candidates), np.inf, dtype=float)
    residuals = positions[finite] - truth_position
    errors[finite] = np.linalg.norm(residuals, axis=1)
    errors_2d[finite] = np.linalg.norm(residuals[:, :2], axis=1)
    oracle_iloc = int(np.argmin(errors))
    oracle = candidates.iloc[oracle_iloc]
    oracle_key = _candidate_key(oracle)

    eligible = _catprob_candidate_pool(candidates, candidate_catprob_threshold)
    retained_nodes = _nodes_for_radar_frame(
        event_index=event_index,
        candidates=candidates,
        anchor=anchor,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=config,
    )
    retained_rows = [node.row for node in retained_nodes if node.row is not None]
    retained_keys = {_candidate_key(row) for row in retained_rows}

    all_scored_nodes = _nodes_for_radar_frame(
        event_index=event_index,
        candidates=candidates,
        anchor=anchor,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=replace(config, max_candidates_per_frame=max(int(len(eligible)), 1)),
    )
    score_rank_by_key = {
        _candidate_key(node.row): rank
        for rank, node in enumerate(all_scored_nodes, start=1)
        if node.row is not None
    }
    eligible_keys = {_candidate_key(row) for _, row in eligible.iterrows()}
    oracle_passed_threshold = oracle_key in eligible_keys
    oracle_retained = oracle_key in retained_keys
    if oracle_retained:
        drop_reason = "retained"
    elif not oracle_passed_threshold:
        drop_reason = "catprob_threshold"
    else:
        drop_reason = "top_k"

    base.update(
        {
            "eligible_candidate_count": int(len(eligible)),
            "retained_candidate_count": int(len(retained_rows)),
            "oracle_available": True,
            "oracle_retained": bool(oracle_retained),
            "oracle_drop_reason": drop_reason,
            "oracle_track_id": _optional_int(oracle.get("track_id")),
            "oracle_track_index": _optional_int(oracle.get("track_index")),
            "oracle_range_m": _candidate_range_m(oracle),
            "oracle_cat_prob_uav": _optional_float(oracle.get("cat_prob_uav")),
            "oracle_truth_error_m": float(errors[oracle_iloc]),
            "oracle_truth_error_2d_m": float(errors_2d[oracle_iloc]),
            "oracle_passed_catprob_threshold": bool(oracle_passed_threshold),
            "oracle_rank_by_catprob": _rank_by_catprob(candidates, oracle_iloc),
            "oracle_rank_by_unary_score": score_rank_by_key.get(oracle_key, np.nan),
        }
    )
    return base, bool(oracle_retained)


def _interpolated_truth_position(
    truth: pd.DataFrame,
    time_s: float,
    *,
    max_time_delta_s: float | None,
) -> tuple[np.ndarray | None, float, float]:
    positions, valid = interpolate_truth_positions(
        truth,
        [float(time_s)],
        max_time_delta_s=max_time_delta_s,
    )
    if not bool(valid[0]):
        return None, float("nan"), float("nan")
    return positions[0], float(time_s), _nearest_truth_time_delta_s(truth, float(time_s))


def _nearest_truth_time_delta_s(truth: pd.DataFrame, time_s: float) -> float:
    if truth.empty or "time_s" not in truth.columns:
        return float("nan")
    truth_times = pd.to_numeric(truth["time_s"], errors="coerce").dropna().to_numpy(dtype=float)
    truth_times = np.sort(truth_times[np.isfinite(truth_times)])
    if truth_times.size == 0:
        return float("nan")
    insertion = int(np.searchsorted(truth_times, float(time_s)))
    right = int(np.clip(insertion, 0, truth_times.size - 1))
    left = int(np.clip(insertion - 1, 0, truth_times.size - 1))
    return float(min(abs(truth_times[left] - time_s), abs(truth_times[right] - time_s)))


def _candidate_positions(candidates: pd.DataFrame) -> np.ndarray:
    required = ["east_m", "north_m", "up_m"]
    missing = [column for column in required if column not in candidates.columns]
    if missing:
        raise KeyError(f"radar candidates are missing required columns: {missing}")
    return candidates[required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _candidate_key(row: pd.Series) -> tuple[tuple[str, object], ...]:
    key_columns = (
        "frame_index",
        "track_index",
        "track_id",
        "time_s",
        "east_m",
        "north_m",
        "up_m",
    )
    columns = [column for column in key_columns if column in row.index]
    return tuple((column, _stable_value(row[column])) for column in columns)


def _stable_value(value: object) -> object:
    number = _optional_float(value)
    if number is None:
        return str(value)
    rounded = round(float(number), 9)
    return int(rounded) if float(rounded).is_integer() else rounded


def _candidate_range_m(row: pd.Series) -> float:
    range_m = _optional_float(row.get("range_m"))
    if range_m is not None:
        return range_m
    position = np.array(
        [
            _optional_float(row.get("east_m")),
            _optional_float(row.get("north_m")),
            _optional_float(row.get("up_m")),
        ],
        dtype=float,
    )
    return float(np.linalg.norm(position)) if np.isfinite(position).all() else float("nan")


def _event_key(candidates: pd.DataFrame, time_s: float) -> str:
    if "frame_index" in candidates.columns and not candidates.empty:
        values = pd.to_numeric(candidates["frame_index"], errors="coerce").dropna()
        if not values.empty:
            return f"frame_index:{int(values.iloc[0])}"
    return f"time_s:{float(time_s):.9f}"


def _rank_by_catprob(candidates: pd.DataFrame, oracle_iloc: int) -> float:
    if "cat_prob_uav" not in candidates.columns:
        return float("nan")
    values = pd.to_numeric(candidates["cat_prob_uav"], errors="coerce").fillna(-np.inf)
    order = np.argsort(-values.to_numpy(dtype=float), kind="mergesort")
    matches = np.flatnonzero(order == int(oracle_iloc))
    return float(matches[0] + 1) if matches.size else float("nan")


def _coverage_summary(
    frame: pd.DataFrame,
    *,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
    truth_time_gate_s: float | None,
) -> dict[str, Any]:
    available = frame.loc[frame["oracle_available"].fillna(False).astype(bool)]
    retained = int(available["oracle_retained"].fillna(False).astype(bool).sum())
    total = int(len(available))
    return {
        "candidate_catprob_threshold": candidate_catprob_threshold,
        "max_candidates_per_frame": int(config.max_candidates_per_frame),
        "truth_time_gate_s": truth_time_gate_s,
        "radar_frames": int(len(frame)),
        "oracle_available_frames": total,
        "oracle_unavailable_frames": int(len(frame) - total),
        "oracle_retained_frames": retained,
        "oracle_retention_rate": _safe_rate(retained, total),
        "catprob_threshold_drop_frames": int(
            (available["oracle_drop_reason"] == "catprob_threshold").sum()
        ),
        "top_k_drop_frames": int((available["oracle_drop_reason"] == "top_k").sum()),
        "mean_retained_candidate_count": _safe_mean(available, "retained_candidate_count"),
        "mean_frame_candidate_count": _safe_mean(available, "frame_candidate_count"),
        "mean_oracle_truth_error_m": _safe_mean(available, "oracle_truth_error_m"),
    }


def _bucket_summary(frame: pd.DataFrame) -> pd.DataFrame:
    available = frame.loc[frame["oracle_available"].fillna(False).astype(bool)].copy()
    if available.empty:
        return pd.DataFrame(columns=_BUCKET_COLUMNS)
    rows: list[dict[str, Any]] = []
    _append_bucket_summary(
        rows,
        available,
        bucket_type="range_m",
        bucket=_numeric_bucket(
            available["oracle_range_m"],
            bins=[0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, np.inf],
            labels=["0-250", "250-500", "500-750", "750-1000", "1000-1500", "1500+"],
        ),
    )
    _append_bucket_summary(
        rows,
        available,
        bucket_type="frame_candidate_count",
        bucket=_numeric_bucket(
            available["frame_candidate_count"],
            bins=[0.0, 1.0, 4.0, 8.0, 16.0, 32.0, np.inf],
            labels=["1", "2-4", "5-8", "9-16", "17-32", "33+"],
        ),
    )
    _append_bucket_summary(
        rows,
        available,
        bucket_type="oracle_cat_prob_uav",
        bucket=_numeric_bucket(
            available["oracle_cat_prob_uav"],
            bins=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0],
            labels=["0-0.25", "0.25-0.5", "0.5-0.75", "0.75-0.9", "0.9-1.0"],
        ),
    )
    _append_bucket_summary(
        rows,
        available,
        bucket_type="oracle_miss_streak_before",
        bucket=available["oracle_miss_streak_before"].map(_miss_streak_bucket),
    )
    return pd.DataFrame.from_records(rows, columns=_BUCKET_COLUMNS)


def _numeric_bucket(values: pd.Series, *, bins: list[float], labels: list[str]) -> pd.Series:
    bucket = pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype(object)
    return bucket.where(pd.notna(bucket), "unknown").astype(str)


def _append_bucket_summary(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    bucket_type: str,
    bucket: pd.Series,
) -> None:
    tmp = frame.copy()
    tmp["_bucket"] = bucket.to_numpy()
    for label, group in tmp.groupby("_bucket", sort=True, dropna=False):
        count = int(len(group))
        retained = int(group["oracle_retained"].fillna(False).astype(bool).sum())
        rows.append(
            {
                "bucket_type": bucket_type,
                "bucket": str(label),
                "frame_count": count,
                "retained_count": retained,
                "retention_rate": _safe_rate(retained, count),
                "catprob_threshold_drop_count": int(
                    (group["oracle_drop_reason"] == "catprob_threshold").sum()
                ),
                "top_k_drop_count": int((group["oracle_drop_reason"] == "top_k").sum()),
            }
        )


def _miss_streak_bucket(value: object) -> str:
    streak = _optional_int(value)
    if streak is None or streak <= 0:
        return "0"
    if streak == 1:
        return "1"
    if streak == 2:
        return "2"
    if streak <= 4:
        return "3-4"
    return "5+"


def _empty_result(
    *,
    candidate_catprob_threshold: float | None,
    config: TrackletViterbiAssociationConfig,
    truth_time_gate_s: float | None,
) -> OracleCandidateCoverageResult:
    frame = pd.DataFrame(columns=_FRAME_COLUMNS)
    return OracleCandidateCoverageResult(
        frame_coverage=frame,
        bucket_summary=pd.DataFrame(columns=_BUCKET_COLUMNS),
        summary=_coverage_summary(
            frame,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=config,
            truth_time_gate_s=truth_time_gate_s,
        ),
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return None if number is None else int(number)
