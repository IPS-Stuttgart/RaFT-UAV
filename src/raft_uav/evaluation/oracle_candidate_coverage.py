"""Oracle coverage diagnostics for tracklet-Viterbi radar candidate pruning.

The functions in this module use ground truth only for offline diagnostics. They
answer a specific question: before Viterbi decoding, does the candidate set kept
by the configured tracklet node builder still contain the truth-nearest radar
row for each radar frame?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines import tracklet_viterbi as _base_tracklet
from raft_uav.baselines import tracklet_viterbi_retention as _retention_tracklet
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.radar_association import _catprob_candidate_pool, _events
from raft_uav.evaluation.metrics import nearest_time_indices
from raft_uav.io.aerpaw import (
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    rf_measurements_to_enu,
    select_flight,
)

_ROW_ID_COLUMN = "__oracle_coverage_row_id"
_RANGE_BINS_M = (0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, np.inf)
_CATPROB_BINS = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def build_oracle_candidate_coverage_diagnostics(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    rf_measurements: list[TrackingMeasurement] | None = None,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    candidate_catprob_threshold: float | None = 0.5,
    truth_time_gate_s: float = 1.0,
    truth_gate_m: float | None = None,
    config: _base_tracklet.TrackletViterbiAssociationConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return per-frame oracle retention diagnostics plus aggregate coverage.

    The diagnostic does not change association results. It rebuilds the same
    radar-frame event stream used by tracklet Viterbi, finds the ground-truth
    nearest radar row in each frame, and then checks whether that row survives:

    * the cat-probability candidate pool used by the base node builder,
    * the base top-K Viterbi node truncation, and
    * the retention-aware top-K plus per-track node builder.

    ``truth_gate_m`` is optional. When provided, summary coverage denominators
    include only frames whose nearest radar row lies within that distance gate.
    """

    cfg = config or _base_tracklet.TrackletViterbiAssociationConfig()
    if radar.empty:
        report = _empty_report()
        return report, summarize_oracle_candidate_coverage(report)
    if truth.empty:
        report = _empty_report()
        return report, summarize_oracle_candidate_coverage(report)
    if _ROW_ID_COLUMN in radar.columns:
        raise ValueError(f"radar already contains reserved diagnostic column {_ROW_ID_COLUMN!r}")

    radar_with_ids = radar.copy().reset_index(drop=True)
    radar_with_ids[_ROW_ID_COLUMN] = np.arange(len(radar_with_ids), dtype=int)
    covariance = np.diag(
        [float(radar_xy_std_m) ** 2, float(radar_xy_std_m) ** 2, float(radar_z_std_m) ** 2]
    )
    events = _events(list(rf_measurements or []), radar_with_ids)
    bootstrap_index = _base_tracklet._first_rf_bootstrap_index(events)
    if bootstrap_index is None:
        report = _empty_report()
        return report, summarize_oracle_candidate_coverage(report)
    events = events[bootstrap_index:]
    anchors = _base_tracklet._build_rf_anchor_states(
        events=events,
        acceleration_std_mps2=acceleration_std_mps2,
        gate_probabilities_by_source=None,
        gate_thresholds_by_source=None,
        safety_gate_probabilities_by_source=None,
        safety_gate_thresholds_by_source=None,
        robust_update_by_source=None,
        inflation_alpha_by_source=None,
        max_residual_norms_by_source=None,
    )
    track_support_by_event = _retention_tracklet._track_support_by_event_prefix(events)

    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        if event.get("kind") != "radar":
            continue
        candidates = event.get("candidates")
        if not isinstance(candidates, pd.DataFrame):
            continue
        anchor = anchors.get(event_index)
        oracle = _oracle_candidate_for_frame(
            candidates,
            truth=truth,
            truth_time_gate_s=truth_time_gate_s,
            truth_gate_m=truth_gate_m,
        )
        base_nodes = _base_tracklet._nodes_for_radar_frame(
            event_index=event_index,
            candidates=candidates,
            anchor=anchor,
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=cfg,
        )
        retention_nodes = _retention_tracklet._nodes_for_radar_frame_with_track_retention(
            event_index=event_index,
            candidates=candidates,
            anchor=anchor,
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=cfg,
            track_support_by_id=track_support_by_event.get(event_index, {}),
        )
        base_ids = _node_row_ids(base_nodes)
        retention_ids = _node_row_ids(retention_nodes)
        oracle_row_id = oracle.get("oracle_row_id")
        base_rank, base_score, base_pool_size, in_catprob_pool = _base_unary_rank(
            candidates,
            oracle_row_id=oracle_row_id,
            anchor=anchor,
            covariance=covariance,
            candidate_catprob_threshold=candidate_catprob_threshold,
            config=cfg,
        )
        rows.append(
            {
                **_frame_identity(event_index, candidates),
                "anchor_available": anchor is not None,
                "candidate_count": int(len(candidates)),
                "candidate_catprob_threshold": _optional_float(candidate_catprob_threshold),
                "max_candidates_per_frame": int(cfg.max_candidates_per_frame),
                **oracle,
                "oracle_in_catprob_pool": bool(in_catprob_pool),
                "base_pool_candidate_count": int(base_pool_size),
                "base_unary_rank": base_rank,
                "base_unary_score": base_score,
                "base_retained_candidate_count": _nonmiss_node_count(base_nodes),
                "base_retained": bool(oracle_row_id is not None and oracle_row_id in base_ids),
                "retention_retained_candidate_count": _nonmiss_node_count(retention_nodes),
                "retention_retained": bool(
                    oracle_row_id is not None and oracle_row_id in retention_ids
                ),
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        report = _empty_report()
    return report, summarize_oracle_candidate_coverage(report)


def summarize_oracle_candidate_coverage(report: pd.DataFrame) -> dict[str, Any]:
    """Summarize oracle-candidate survival rates from a per-frame report."""

    frame = report.copy()
    if frame.empty:
        return {
            "radar_frame_count": 0,
            "eligible_frame_count": 0,
            "catprob_pool": _coverage_stats(frame, "oracle_in_catprob_pool"),
            "base": _coverage_stats(frame, "base_retained"),
            "retention": _coverage_stats(frame, "retention_retained"),
            "by_candidate_count": [],
            "by_range_m": [],
            "by_catprob": [],
        }

    eligible = _eligible_mask(frame)
    summary = {
        "radar_frame_count": int(len(frame)),
        "eligible_frame_count": int(np.count_nonzero(eligible)),
        "catprob_pool": _coverage_stats(frame.loc[eligible], "oracle_in_catprob_pool"),
        "base": _coverage_stats(frame.loc[eligible], "base_retained"),
        "retention": _coverage_stats(frame.loc[eligible], "retention_retained"),
        "by_candidate_count": _grouped_coverage(frame, eligible, "candidate_count"),
    }
    if "oracle_range_m" in frame.columns:
        range_bucket = pd.cut(
            pd.to_numeric(frame["oracle_range_m"], errors="coerce"),
            bins=_RANGE_BINS_M,
            right=False,
            include_lowest=True,
        )
        frame = frame.assign(_range_bucket=range_bucket.astype(str))
        summary["by_range_m"] = _grouped_coverage(frame, eligible, "_range_bucket")
    else:
        summary["by_range_m"] = []
    if "oracle_cat_prob_uav" in frame.columns:
        catprob_bucket = pd.cut(
            pd.to_numeric(frame["oracle_cat_prob_uav"], errors="coerce"),
            bins=_CATPROB_BINS,
            right=False,
            include_lowest=True,
        )
        frame = frame.assign(_catprob_bucket=catprob_bucket.astype(str))
        summary["by_catprob"] = _grouped_coverage(frame, eligible, "_catprob_bucket")
    else:
        summary["by_catprob"] = []
    return summary


def _oracle_candidate_for_frame(
    candidates: pd.DataFrame,
    *,
    truth: pd.DataFrame,
    truth_time_gate_s: float,
    truth_gate_m: float | None,
) -> dict[str, Any]:
    time_s = float(candidates["time_s"].median()) if "time_s" in candidates else float("nan")
    truth_position, truth_delta_s = _nearest_truth_position(
        truth,
        time_s=time_s,
        max_delta_s=float(truth_time_gate_s),
    )
    base = {
        "truth_available": truth_position is not None,
        "truth_time_delta_s": truth_delta_s,
        "oracle_candidate_found": False,
        "oracle_within_truth_gate": False,
        "oracle_row_id": None,
        "oracle_rank_by_truth_error": None,
        "oracle_truth_error_2d_m": None,
        "oracle_truth_error_3d_m": None,
        "oracle_track_id": None,
        "oracle_track_index": None,
        "oracle_cat_prob_uav": None,
        "oracle_range_m": None,
    }
    if truth_position is None or candidates.empty:
        return base

    positions = candidates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    finite = np.isfinite(positions).all(axis=1)
    if not finite.any():
        return base
    errors_3d = np.full(len(candidates), np.inf, dtype=float)
    deltas = positions[finite] - truth_position.reshape(1, 3)
    errors_3d[finite] = np.linalg.norm(deltas, axis=1)
    best_iloc = int(np.argmin(errors_3d))
    best_row = candidates.iloc[best_iloc]
    best_position = positions[best_iloc]
    error_3d_m = float(errors_3d[best_iloc])
    error_2d_m = float(np.linalg.norm(best_position[:2] - truth_position[:2]))
    rank_by_error = int(np.where(np.argsort(errors_3d, kind="mergesort") == best_iloc)[0][0] + 1)
    within_gate = truth_gate_m is None or error_3d_m <= float(truth_gate_m)
    return {
        **base,
        "oracle_candidate_found": True,
        "oracle_within_truth_gate": bool(within_gate),
        "oracle_row_id": _row_id(best_row),
        "oracle_rank_by_truth_error": rank_by_error,
        "oracle_truth_error_2d_m": error_2d_m,
        "oracle_truth_error_3d_m": error_3d_m,
        "oracle_track_id": _optional_int(best_row.get("track_id")),
        "oracle_track_index": _optional_int(best_row.get("track_index")),
        "oracle_cat_prob_uav": _optional_float(best_row.get("cat_prob_uav")),
        "oracle_range_m": _optional_float(best_row.get("range_m")),
    }


def _nearest_truth_position(
    truth: pd.DataFrame,
    *,
    time_s: float,
    max_delta_s: float,
) -> tuple[np.ndarray | None, float | None]:
    if truth.empty or not np.isfinite(time_s):
        return None, None
    truth_times = truth["time_s"].to_numpy(dtype=float)
    if truth_times.size == 0:
        return None, None
    index = int(nearest_time_indices(truth_times, np.array([time_s], dtype=float))[0])
    delta_s = float(abs(truth_times[index] - time_s))
    if delta_s > float(max_delta_s):
        return None, delta_s
    position = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)[index]
    if not np.isfinite(position).all():
        return None, delta_s
    return position, delta_s


def _base_unary_rank(
    candidates: pd.DataFrame,
    *,
    oracle_row_id: int | None,
    anchor: _base_tracklet._AnchorState | None,
    covariance: np.ndarray,
    candidate_catprob_threshold: float | None,
    config: _base_tracklet.TrackletViterbiAssociationConfig,
) -> tuple[int | None, float | None, int, bool]:
    if oracle_row_id is None:
        return None, None, 0, False
    pool = _catprob_candidate_pool(candidates, candidate_catprob_threshold)
    scored: list[tuple[float, int, int]] = []
    for order, (_, row) in enumerate(pool.iterrows()):
        position = _base_tracklet._row_position(row)
        row_id = _row_id(row)
        if position is None or row_id is None:
            continue
        anchor_nis, catprob_cost, range_cost = _base_tracklet._candidate_cost_terms(
            row=row,
            position=position,
            anchor=anchor,
            covariance=covariance,
            config=config,
        )
        score = float(config.anchor_nis_weight) * anchor_nis + catprob_cost + range_cost
        scored.append((float(score), int(order), int(row_id)))
    scored.sort(key=lambda item: (item[0], item[1]))
    for rank, (score, _, row_id) in enumerate(scored, start=1):
        if int(row_id) == int(oracle_row_id):
            return rank, float(score), len(scored), True
    return None, None, len(scored), False


def _frame_identity(event_index: int, candidates: pd.DataFrame) -> dict[str, Any]:
    frame_index = None
    if "frame_index" in candidates.columns:
        frame_values = pd.to_numeric(candidates["frame_index"], errors="coerce").dropna()
        if not frame_values.empty:
            frame_index = int(frame_values.iloc[0])
    return {
        "event_index": int(event_index),
        "frame_index": frame_index,
        "time_s": float(candidates["time_s"].median()) if "time_s" in candidates else None,
    }


def _node_row_ids(nodes: list[_base_tracklet._ViterbiNode]) -> set[int]:
    row_ids: set[int] = set()
    for node in nodes:
        if node.is_miss or node.row is None:
            continue
        row_id = _row_id(node.row)
        if row_id is not None:
            row_ids.add(row_id)
    return row_ids


def _nonmiss_node_count(nodes: list[_base_tracklet._ViterbiNode]) -> int:
    return int(sum(1 for node in nodes if not node.is_miss))


def _eligible_mask(frame: pd.DataFrame) -> np.ndarray:
    found = frame.get("oracle_candidate_found", pd.Series(False, index=frame.index)).astype(bool)
    within = frame.get("oracle_within_truth_gate", pd.Series(False, index=frame.index)).astype(bool)
    available = frame.get("truth_available", pd.Series(False, index=frame.index)).astype(bool)
    return (found & within & available).to_numpy(dtype=bool)


def _coverage_stats(frame: pd.DataFrame, column: str) -> dict[str, float | int]:
    if frame.empty or column not in frame.columns:
        return {"frames": 0, "covered": 0, "coverage_rate": float("nan")}
    values = frame[column].fillna(False).astype(bool).to_numpy(dtype=bool)
    covered = int(np.count_nonzero(values))
    return {
        "frames": int(values.size),
        "covered": covered,
        "coverage_rate": float(covered / values.size) if values.size else float("nan"),
    }


def _grouped_coverage(
    frame: pd.DataFrame,
    eligible: np.ndarray,
    group_column: str,
) -> list[dict[str, Any]]:
    if frame.empty or group_column not in frame.columns:
        return []
    eligible_frame = frame.loc[eligible].copy()
    if eligible_frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for bucket, group in eligible_frame.groupby(group_column, dropna=False, sort=True):
        catprob = _coverage_stats(group, "oracle_in_catprob_pool")
        base = _coverage_stats(group, "base_retained")
        retention = _coverage_stats(group, "retention_retained")
        rows.append(
            {
                "bucket": str(bucket),
                "frames": int(len(group)),
                "catprob_pool_coverage_rate": catprob["coverage_rate"],
                "base_coverage_rate": base["coverage_rate"],
                "retention_coverage_rate": retention["coverage_rate"],
            }
        )
    return rows


def _empty_report() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_index",
            "frame_index",
            "time_s",
            "anchor_available",
            "candidate_count",
            "candidate_catprob_threshold",
            "max_candidates_per_frame",
            "truth_available",
            "truth_time_delta_s",
            "oracle_candidate_found",
            "oracle_within_truth_gate",
            "oracle_row_id",
            "oracle_rank_by_truth_error",
            "oracle_truth_error_2d_m",
            "oracle_truth_error_3d_m",
            "oracle_track_id",
            "oracle_track_index",
            "oracle_cat_prob_uav",
            "oracle_range_m",
            "oracle_in_catprob_pool",
            "base_pool_candidate_count",
            "base_unary_rank",
            "base_unary_score",
            "base_retained_candidate_count",
            "base_retained",
            "retention_retained_candidate_count",
            "retention_retained",
        ]
    )


def _row_id(row: pd.Series) -> int | None:
    if _ROW_ID_COLUMN not in row.index:
        return None
    value = row.get(_ROW_ID_COLUMN)
    if value is None or pd.isna(value):
        return None
    return int(value)


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


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns or truth.empty:
        return frame
    t_min = float(truth["time_s"].min())
    t_max = float(truth["time_s"].max())
    return frame.loc[frame["time_s"].between(t_min, t_max)].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """Run oracle candidate coverage diagnostics for one flight."""

    parser = argparse.ArgumentParser(prog="raft-uav-diagnose-oracle-coverage")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/oracle_candidate_coverage"),
    )
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--radar-xy-std-m", type=float, default=25.0)
    parser.add_argument("--radar-z-std-m", type=float, default=35.0)
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.5)
    parser.add_argument("--tracklet-max-candidates-per-frame", type=int, default=8)
    parser.add_argument("--truth-time-gate-s", type=float, default=1.0)
    parser.add_argument(
        "--truth-gate-m",
        type=float,
        default=0.0,
        help="distance gate for summary denominators; <=0 includes all truth-nearest rows",
    )
    args = parser.parse_args(argv)
    if args.tracklet_max_candidates_per_frame < 1:
        raise ValueError("--tracklet-max-candidates-per-frame must be positive")
    if args.truth_time_gate_s <= 0.0:
        raise ValueError("--truth-time-gate-s must be positive")

    flight = select_flight(args.dataset_root, args.flight)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    if flight.radar_json is None:
        raise FileNotFoundError(f"{flight.name} has no radar JSON file")

    truth_raw = read_truth(flight.truth_txt)
    truth, projector, truth_origin_time = normalize_truth(truth_raw)
    rf_measurements: list[TrackingMeasurement] = []
    if flight.rf_csv is not None:
        rf = _inside_truth_window(
            normalize_rf(read_rf_csv(flight.rf_csv), projector, truth_origin_time), truth
        )
        rf_measurements = rf_measurements_to_enu(rf)
    radar = _inside_truth_window(
        normalize_radar(read_radar_tracks_json(flight.radar_json), projector, truth_origin_time),
        truth,
    )

    config = replace(
        _base_tracklet.TrackletViterbiAssociationConfig(),
        max_candidates_per_frame=int(args.tracklet_max_candidates_per_frame),
    )
    report, summary = build_oracle_candidate_coverage_diagnostics(
        radar=radar,
        truth=truth,
        rf_measurements=rf_measurements,
        acceleration_std_mps2=args.acceleration_std,
        radar_xy_std_m=args.radar_xy_std_m,
        radar_z_std_m=args.radar_z_std_m,
        candidate_catprob_threshold=args.radar_catprob_threshold,
        truth_time_gate_s=args.truth_time_gate_s,
        truth_gate_m=None if args.truth_gate_m <= 0.0 else args.truth_gate_m,
        config=config,
    )

    flight_output = args.output_dir / flight.name
    flight_output.mkdir(parents=True, exist_ok=True)
    csv_path = flight_output / "oracle_candidate_coverage.csv"
    summary_path = flight_output / "oracle_candidate_coverage_summary.json"
    report.to_csv(csv_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"flight={flight.name}")
    print(f"radar_frames={summary['radar_frame_count']}")
    print(f"eligible_frames={summary['eligible_frame_count']}")
    print(f"base_coverage_rate={summary['base']['coverage_rate']:.6g}")
    print(f"retention_coverage_rate={summary['retention']['coverage_rate']:.6g}")
    print(f"oracle_candidate_coverage_csv={csv_path}")
    print(f"oracle_candidate_coverage_summary_json={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
