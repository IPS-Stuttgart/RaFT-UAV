"""One-to-one temporal assignment for MMUAD candidate consensus.

The standard temporal-consensus path scores every candidate against its nearest
candidate in the adjacent frame. In a dense branch-preserving pool, several
duplicated hypotheses can therefore claim support from the same neighbor. This
module adds an optional global one-to-one assignment for each adjacent frame
pair, preventing one observation from supporting many competing candidates.

The assignment is truth-free and can use a train-selected temporal-consensus
configuration produced by ``candidate_temporal_consensus_train_cv``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from raft_uav.mmuad.candidate_temporal_consensus import (
    TemporalConsensusConfig,
    _add_consensus_score,
    _base_score,
    _branch_series,
    _candidate_rows,
    _initialize_temporal_columns,
    _validate_config,
    _write_bidirectional_metrics,
    _write_neighbor_match,
    add_temporal_candidate_consensus,
    temporal_consensus_summary,
    write_temporal_consensus_outputs,
)
from raft_uav.mmuad.candidate_temporal_consensus_train_cv import (
    load_train_selected_temporal_consensus_config,
)
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns

ASSIGNMENT_MODES = ("nearest", "one-to-one")


def add_assignment_temporal_candidate_consensus(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    config: TemporalConsensusConfig | None = None,
    assignment_mode: str = "one-to-one",
) -> CandidateFrame:
    """Attach temporal consensus with nearest or one-to-one neighbor assignment."""

    cfg = config or TemporalConsensusConfig()
    _validate_config(cfg)
    mode = _validate_assignment_mode(assignment_mode)
    if mode == "nearest":
        nearest = add_temporal_candidate_consensus(candidates, config=cfg)
        rows = nearest.rows.copy()
        rows["candidate_temporal_assignment_mode"] = mode
        for direction in ("backward", "forward"):
            distance = pd.to_numeric(
                rows.get(f"candidate_reservoir_temporal_{direction}_distance_m"),
                errors="coerce",
            )
            rows[f"candidate_reservoir_temporal_{direction}_assignment_matched"] = (
                np.isfinite(distance.to_numpy(float)).astype(float)
            )
        return CandidateFrame(normalize_candidate_columns(rows))

    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(rows)

    out = rows.copy().reset_index(drop=True)
    out["_temporal_row_id"] = np.arange(len(out), dtype=int)
    out["candidate_branch"] = _branch_series(out)
    out["candidate_reservoir_temporal_base_score"] = _base_score(out, cfg)
    _initialize_temporal_columns(out)
    for direction in ("backward", "forward"):
        out[f"candidate_reservoir_temporal_{direction}_assignment_matched"] = 0.0
        out[f"candidate_temporal_{direction}_track_id"] = ""

    for _, sequence in out.groupby("sequence_id", sort=False):
        _annotate_sequence_one_to_one(out, sequence, cfg)

    _add_consensus_score(out, cfg)
    out["candidate_temporal_assignment_mode"] = mode
    out = out.drop(columns=["_temporal_row_id"], errors="ignore")
    return CandidateFrame(normalize_candidate_columns(out))


def assignment_temporal_consensus_summary(
    candidates: CandidateFrame | pd.DataFrame,
) -> dict[str, Any]:
    """Return temporal-consensus diagnostics including assignment coverage."""

    rows = candidates.rows if isinstance(candidates, CandidateFrame) else pd.DataFrame(candidates)
    summary = temporal_consensus_summary(rows)
    modes = (
        rows.get("candidate_temporal_assignment_mode", pd.Series(dtype=str))
        .fillna("")
        .astype(str)
        .value_counts()
    )
    summary["assignment_mode_counts"] = {
        str(key): int(value) for key, value in modes.sort_index().items()
    }
    for direction in ("backward", "forward"):
        matched = pd.to_numeric(
            rows.get(
                f"candidate_reservoir_temporal_{direction}_assignment_matched",
                pd.Series(0.0, index=rows.index),
            ),
            errors="coerce",
        ).fillna(0.0)
        summary[f"{direction}_assignment_matched_count"] = int((matched > 0.0).sum())
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-temporal-consensus-assigned",
        description="add one-to-one temporal assignment consensus to MMUAD candidates",
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--assignment-mode", choices=ASSIGNMENT_MODES, default="one-to-one")
    parser.add_argument("--max-time-gap-s", type=float, default=2.0)
    parser.add_argument("--max-speed-mps", type=float, default=60.0)
    parser.add_argument("--distance-scale-m", type=float, default=5.0)
    parser.add_argument("--acceleration-scale-mps2", type=float, default=20.0)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--fallback-score-column", default="confidence")
    parser.add_argument("--base-score-weight", type=float, default=0.25)
    parser.add_argument("--backward-support-weight", type=float, default=1.0)
    parser.add_argument("--forward-support-weight", type=float, default=1.0)
    parser.add_argument("--bidirectional-bonus", type=float, default=0.75)
    parser.add_argument("--interpolation-weight", type=float, default=0.75)
    parser.add_argument("--acceleration-weight", type=float, default=0.5)
    parser.add_argument("--source-diversity-bonus", type=float, default=0.25)
    parser.add_argument("--branch-diversity-bonus", type=float, default=0.25)
    parser.add_argument("--replace-confidence", action="store_true")
    args = parser.parse_args(argv)

    config = (
        _load_selected_config(args.config_json)
        if args.config_json is not None
        else TemporalConsensusConfig(
            max_time_gap_s=args.max_time_gap_s,
            max_speed_mps=args.max_speed_mps,
            distance_scale_m=args.distance_scale_m,
            acceleration_scale_mps2=args.acceleration_scale_mps2,
            score_column=args.score_column,
            fallback_score_column=args.fallback_score_column,
            base_score_weight=args.base_score_weight,
            backward_support_weight=args.backward_support_weight,
            forward_support_weight=args.forward_support_weight,
            bidirectional_bonus=args.bidirectional_bonus,
            interpolation_weight=args.interpolation_weight,
            acceleration_weight=args.acceleration_weight,
            source_diversity_bonus=args.source_diversity_bonus,
            branch_diversity_bonus=args.branch_diversity_bonus,
        )
    )
    augmented = add_assignment_temporal_candidate_consensus(
        load_candidate_file(args.candidate_csv),
        config=config,
        assignment_mode=args.assignment_mode,
    )
    if args.replace_confidence and not augmented.rows.empty:
        rows = augmented.rows.copy()
        rows["raw_confidence"] = pd.to_numeric(rows.get("confidence"), errors="coerce")
        rows["confidence"] = pd.to_numeric(
            rows["candidate_temporal_consensus_score"],
            errors="coerce",
        )
        augmented = CandidateFrame(normalize_candidate_columns(rows))

    write_temporal_consensus_outputs(
        augmented,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        provenance={
            "candidate_csv": str(args.candidate_csv),
            "config_json": str(args.config_json) if args.config_json is not None else None,
            "config": asdict(config),
            "assignment_mode": args.assignment_mode,
            "replace_confidence": bool(args.replace_confidence),
            "assignment_summary": assignment_temporal_consensus_summary(augmented),
        },
    )
    print("mmuad_temporal_assignment_consensus=ok")
    print(f"assignment_mode={args.assignment_mode}")
    print(f"output_csv={args.output_csv}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    return 0


def _load_selected_config(path: Path) -> TemporalConsensusConfig:
    payload = load_train_selected_temporal_consensus_config(path)
    values = payload.get("temporal_consensus_config")
    if not isinstance(values, dict):
        raise ValueError("selected temporal-consensus config is missing config values")
    return TemporalConsensusConfig(**values)


def _validate_assignment_mode(value: str) -> str:
    mode = str(value)
    if mode not in ASSIGNMENT_MODES:
        raise ValueError(
            f"unsupported temporal assignment mode {mode!r}; "
            f"expected one of {ASSIGNMENT_MODES}"
        )
    return mode


def _annotate_sequence_one_to_one(
    out: pd.DataFrame,
    sequence: pd.DataFrame,
    config: TemporalConsensusConfig,
) -> None:
    times = np.sort(sequence["time_s"].dropna().unique().astype(float))
    frames = {
        float(time_s): sequence.loc[sequence["time_s"] == float(time_s)].copy()
        for time_s in times
    }
    for position, time_s in enumerate(times):
        current = frames[float(time_s)]
        previous = None
        next_frame = None
        previous_dt = np.nan
        next_dt = np.nan
        if position > 0:
            candidate_time = float(times[position - 1])
            dt = float(time_s - candidate_time)
            if 0.0 < dt <= config.max_time_gap_s:
                previous = frames[candidate_time]
                previous_dt = dt
        if position + 1 < len(times):
            candidate_time = float(times[position + 1])
            dt = float(candidate_time - time_s)
            if 0.0 < dt <= config.max_time_gap_s:
                next_frame = frames[candidate_time]
                next_dt = dt

        previous_match = _one_to_one_neighbor_match(
            current,
            previous,
            previous_dt,
            config,
        )
        next_match = _one_to_one_neighbor_match(
            current,
            next_frame,
            next_dt,
            config,
        )
        _write_assignment_match(
            out,
            current,
            previous_match,
            direction="backward",
        )
        _write_assignment_match(
            out,
            current,
            next_match,
            direction="forward",
        )
        _write_bidirectional_metrics(
            out,
            current,
            previous,
            next_frame,
            previous_match,
            next_match,
            previous_dt,
            next_dt,
        )


def _one_to_one_neighbor_match(
    current: pd.DataFrame,
    neighbor: pd.DataFrame | None,
    dt_s: float,
    config: TemporalConsensusConfig,
) -> dict[str, Any]:
    count = len(current)
    empty = _empty_match(count)
    if neighbor is None or neighbor.empty or not np.isfinite(dt_s) or dt_s <= 0.0:
        return empty

    current_xyz = current[["x_m", "y_m", "z_m"]].to_numpy(float)
    neighbor_xyz = neighbor[["x_m", "y_m", "z_m"]].to_numpy(float)
    distance = np.linalg.norm(current_xyz[:, None, :] - neighbor_xyz[None, :, :], axis=2)
    speed = distance / float(dt_s)
    eligible = np.isfinite(speed) & (speed <= config.max_speed_mps)

    neighbor_count = len(neighbor)
    gate_distance = float(config.max_speed_mps * dt_s)
    unmatched_cost = gate_distance + max(1.0e-6, gate_distance * 1.0e-6)
    ineligible_cost = unmatched_cost + max(unmatched_cost, 1.0)
    cost = np.full(
        (count, neighbor_count + count),
        ineligible_cost,
        dtype=float,
    )
    cost[:, :neighbor_count] = np.where(eligible, distance, ineligible_cost)
    for row_position in range(count):
        cost[row_position, neighbor_count + row_position] = unmatched_cost

    row_positions, column_positions = linear_sum_assignment(cost)
    nearest_position = np.full(count, -1, dtype=int)
    nearest_distance = np.full(count, np.nan, dtype=float)
    for row_position, column_position in zip(
        row_positions,
        column_positions,
        strict=True,
    ):
        if column_position >= neighbor_count:
            continue
        if not eligible[row_position, column_position]:
            continue
        nearest_position[row_position] = int(column_position)
        nearest_distance[row_position] = float(distance[row_position, column_position])

    matched = nearest_position >= 0
    nearest_speed = np.where(matched, nearest_distance / float(dt_s), np.nan)
    current_source = current["source"].fillna("").astype(str).to_numpy()
    neighbor_source = neighbor["source"].fillna("").astype(str).to_numpy()
    current_branch = current["candidate_branch"].fillna("").astype(str).to_numpy()
    neighbor_branch = neighbor["candidate_branch"].fillna("").astype(str).to_numpy()
    other_source = np.zeros(count, dtype=float)
    other_branch = np.zeros(count, dtype=float)
    for row_position in np.flatnonzero(matched):
        neighbor_position = int(nearest_position[row_position])
        other_source[row_position] = float(
            current_source[row_position] != neighbor_source[neighbor_position]
        )
        other_branch[row_position] = float(
            current_branch[row_position] != neighbor_branch[neighbor_position]
        )
    return {
        "neighbor_position": nearest_position,
        "distance_m": nearest_distance,
        "speed_mps": nearest_speed,
        "support_count": matched.astype(float),
        "other_source_count": other_source,
        "other_branch_count": other_branch,
        "neighbor_rows": neighbor,
    }


def _empty_match(count: int) -> dict[str, Any]:
    return {
        "neighbor_position": np.full(count, -1, dtype=int),
        "distance_m": np.full(count, np.nan, dtype=float),
        "speed_mps": np.full(count, np.nan, dtype=float),
        "support_count": np.zeros(count, dtype=float),
        "other_source_count": np.zeros(count, dtype=float),
        "other_branch_count": np.zeros(count, dtype=float),
    }


def _write_assignment_match(
    out: pd.DataFrame,
    current: pd.DataFrame,
    match: dict[str, Any],
    *,
    direction: str,
) -> None:
    _write_neighbor_match(out, current, match, direction=direction)
    positions = np.asarray(match["neighbor_position"], dtype=int)
    matched = positions >= 0
    out.loc[
        current.index,
        f"candidate_reservoir_temporal_{direction}_assignment_matched",
    ] = matched.astype(float)

    neighbor_rows = match.get("neighbor_rows")
    if neighbor_rows is None or neighbor_rows.empty:
        return
    track_ids: list[str] = []
    for position in positions:
        if position < 0:
            track_ids.append("")
        else:
            track_ids.append(str(neighbor_rows.iloc[int(position)].get("track_id", "")))
    out.loc[current.index, f"candidate_temporal_{direction}_track_id"] = track_ids


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
