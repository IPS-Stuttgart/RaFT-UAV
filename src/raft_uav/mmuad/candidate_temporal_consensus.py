"""Truth-free temporal-consensus features for MMUAD candidate assignment.

The branch-preserving MMUAD pool often contains a good candidate that is buried
by a per-frame ranker.  This module adds bidirectional temporal support features
without committing to a hard track: every candidate is compared with candidates
in the nearest previous and next frames, subject to a speed gate.  The resulting
score can be used by candidate reservoirs, learned uncertainty, or mixture-MAP.

No truth is read or required at inference time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


@dataclass(frozen=True)
class TemporalConsensusConfig:
    """Configuration for bidirectional candidate continuity scoring."""

    max_time_gap_s: float = 2.0
    max_speed_mps: float = 60.0
    distance_scale_m: float = 5.0
    acceleration_scale_mps2: float = 20.0
    score_column: str = "ranker_score"
    fallback_score_column: str = "confidence"
    base_score_weight: float = 0.25
    backward_support_weight: float = 1.0
    forward_support_weight: float = 1.0
    bidirectional_bonus: float = 0.75
    interpolation_weight: float = 0.75
    acceleration_weight: float = 0.5
    source_diversity_bonus: float = 0.25
    branch_diversity_bonus: float = 0.25


def add_temporal_candidate_consensus(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    config: TemporalConsensusConfig | None = None,
) -> CandidateFrame:
    """Attach bidirectional temporal-support features and a consensus score."""

    cfg = config or TemporalConsensusConfig()
    _validate_config(cfg)
    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(rows)

    out = rows.copy().reset_index(drop=True)
    out["_temporal_row_id"] = np.arange(len(out), dtype=int)
    out["candidate_branch"] = _branch_series(out)
    out["candidate_reservoir_temporal_base_score"] = _base_score(out, cfg)
    _initialize_temporal_columns(out)

    for _, sequence in out.groupby("sequence_id", sort=False):
        _annotate_sequence(out, sequence, cfg)

    _add_consensus_score(out, cfg)
    out = out.drop(columns=["_temporal_row_id"], errors="ignore")
    return CandidateFrame(normalize_candidate_columns(out))


def temporal_consensus_summary(rows: CandidateFrame | pd.DataFrame) -> dict[str, Any]:
    """Return compact JSON-serializable diagnostics for augmented candidates."""

    frame = rows.rows.copy() if isinstance(rows, CandidateFrame) else pd.DataFrame(rows).copy()
    if frame.empty:
        return {"row_count": 0, "sequence_count": 0}
    score = pd.to_numeric(frame.get("candidate_temporal_consensus_score"), errors="coerce")
    backward = pd.to_numeric(
        frame.get("candidate_reservoir_temporal_backward_distance_m"), errors="coerce"
    )
    forward = pd.to_numeric(
        frame.get("candidate_reservoir_temporal_forward_distance_m"), errors="coerce"
    )
    bidirectional = pd.to_numeric(
        frame.get("candidate_reservoir_temporal_bidirectional"), errors="coerce"
    ).fillna(0.0)
    return {
        "row_count": int(len(frame)),
        "sequence_count": int(frame["sequence_id"].astype(str).nunique()),
        "frame_count": int(frame[["sequence_id", "time_s"]].drop_duplicates().shape[0]),
        "backward_supported_count": int(np.isfinite(backward.to_numpy(float)).sum()),
        "forward_supported_count": int(np.isfinite(forward.to_numpy(float)).sum()),
        "bidirectional_supported_count": int((bidirectional > 0.0).sum()),
        "score_mean": _safe_mean(score),
        "score_p95": _safe_quantile(score, 0.95),
        "score_max": _safe_max(score),
    }


def write_temporal_consensus_outputs(
    candidates: CandidateFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Write augmented candidates and optional configuration/summary JSON."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.rows.to_csv(output_csv, index=False)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(provenance or {})
        payload["summary"] = temporal_consensus_summary(candidates)
        payload["output_csv"] = str(output_csv)
        summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-temporal-consensus",
        description="add truth-free bidirectional temporal support to MMUAD candidates",
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
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

    config = TemporalConsensusConfig(
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
    augmented = add_temporal_candidate_consensus(
        load_candidate_file(args.candidate_csv),
        config=config,
    )
    if args.replace_confidence and not augmented.rows.empty:
        rows = augmented.rows.copy()
        rows["raw_confidence"] = pd.to_numeric(rows.get("confidence"), errors="coerce")
        rows["confidence"] = pd.to_numeric(
            rows["candidate_temporal_consensus_score"], errors="coerce"
        )
        augmented = CandidateFrame(normalize_candidate_columns(rows))
    write_temporal_consensus_outputs(
        augmented,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        provenance={
            "candidate_csv": str(args.candidate_csv),
            "config": asdict(config),
            "replace_confidence": bool(args.replace_confidence),
        },
    )
    print("mmuad_temporal_candidate_consensus=ok")
    print(f"output_csv={args.output_csv}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    return 0


def _validate_config(config: TemporalConsensusConfig) -> None:
    if config.max_time_gap_s <= 0.0:
        raise ValueError("max_time_gap_s must be positive")
    if config.max_speed_mps <= 0.0:
        raise ValueError("max_speed_mps must be positive")
    if config.distance_scale_m <= 0.0:
        raise ValueError("distance_scale_m must be positive")
    if config.acceleration_scale_mps2 <= 0.0:
        raise ValueError("acceleration_scale_mps2 must be positive")


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    rows = candidates.rows.copy() if isinstance(candidates, CandidateFrame) else pd.DataFrame(candidates)
    rows = normalize_candidate_columns(rows)
    if rows.empty:
        return rows
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["time_s"] = pd.to_numeric(rows["time_s"], errors="coerce")
    return rows.loc[np.isfinite(rows["time_s"].to_numpy(float))].copy()


def _branch_series(rows: pd.DataFrame) -> pd.Series:
    if "candidate_branch" in rows.columns:
        branch = rows["candidate_branch"].fillna("").astype(str).str.strip()
    else:
        branch = pd.Series("", index=rows.index, dtype=str)
    source = rows.get("source", pd.Series("candidate", index=rows.index)).fillna("candidate")
    source = source.astype(str).str.strip()
    return branch.where(branch.str.len() > 0, source).replace("", "candidate")


def _base_score(rows: pd.DataFrame, config: TemporalConsensusConfig) -> pd.Series:
    primary = pd.to_numeric(rows.get(config.score_column), errors="coerce")
    fallback = pd.to_numeric(rows.get(config.fallback_score_column), errors="coerce")
    if not isinstance(primary, pd.Series):
        primary = pd.Series(np.nan, index=rows.index, dtype=float)
    if not isinstance(fallback, pd.Series):
        fallback = pd.Series(1.0, index=rows.index, dtype=float)
    score = primary.fillna(fallback).fillna(0.0).astype(float)
    normalized = pd.Series(0.0, index=rows.index, dtype=float)
    for _, group in rows.assign(_score=score).groupby(["sequence_id", "time_s"], sort=False):
        values = group["_score"].to_numpy(float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        minimum = float(np.min(values[finite]))
        maximum = float(np.max(values[finite]))
        if maximum - minimum <= 1.0e-12:
            normalized.loc[group.index] = 1.0
        else:
            normalized.loc[group.index] = (values - minimum) / (maximum - minimum)
    return normalized


def _initialize_temporal_columns(rows: pd.DataFrame) -> None:
    numeric_defaults = {
        "candidate_reservoir_temporal_backward_distance_m": np.nan,
        "candidate_reservoir_temporal_forward_distance_m": np.nan,
        "candidate_reservoir_temporal_backward_speed_mps": np.nan,
        "candidate_reservoir_temporal_forward_speed_mps": np.nan,
        "candidate_reservoir_temporal_backward_support_count": 0.0,
        "candidate_reservoir_temporal_forward_support_count": 0.0,
        "candidate_reservoir_temporal_other_source_support_count": 0.0,
        "candidate_reservoir_temporal_other_branch_support_count": 0.0,
        "candidate_reservoir_temporal_bidirectional": 0.0,
        "candidate_reservoir_temporal_interpolation_residual_m": np.nan,
        "candidate_reservoir_temporal_acceleration_mps2": np.nan,
    }
    for column, default in numeric_defaults.items():
        rows[column] = default
    for direction in ("backward", "forward"):
        rows[f"candidate_temporal_{direction}_source"] = ""
        rows[f"candidate_temporal_{direction}_branch"] = ""


def _annotate_sequence(
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

        previous_match = _neighbor_match(current, previous, previous_dt, config)
        next_match = _neighbor_match(current, next_frame, next_dt, config)
        _write_neighbor_match(out, current, previous_match, direction="backward")
        _write_neighbor_match(out, current, next_match, direction="forward")
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


def _neighbor_match(
    current: pd.DataFrame,
    neighbor: pd.DataFrame | None,
    dt_s: float,
    config: TemporalConsensusConfig,
) -> dict[str, Any]:
    count = len(current)
    empty = {
        "neighbor_position": np.full(count, -1, dtype=int),
        "distance_m": np.full(count, np.nan, dtype=float),
        "speed_mps": np.full(count, np.nan, dtype=float),
        "support_count": np.zeros(count, dtype=float),
        "other_source_count": np.zeros(count, dtype=float),
        "other_branch_count": np.zeros(count, dtype=float),
    }
    if neighbor is None or neighbor.empty or not np.isfinite(dt_s) or dt_s <= 0.0:
        return empty

    current_xyz = current[["x_m", "y_m", "z_m"]].to_numpy(float)
    neighbor_xyz = neighbor[["x_m", "y_m", "z_m"]].to_numpy(float)
    distance = np.linalg.norm(current_xyz[:, None, :] - neighbor_xyz[None, :, :], axis=2)
    speed = distance / float(dt_s)
    eligible = np.isfinite(speed) & (speed <= config.max_speed_mps)
    masked = np.where(eligible, distance, np.inf)
    nearest_position = np.argmin(masked, axis=1)
    nearest_distance = masked[np.arange(count), nearest_position]
    has_neighbor = np.isfinite(nearest_distance)
    nearest_position = np.where(has_neighbor, nearest_position, -1)
    nearest_speed = np.where(has_neighbor, nearest_distance / float(dt_s), np.nan)

    current_source = current["source"].fillna("").astype(str).to_numpy()
    neighbor_source = neighbor["source"].fillna("").astype(str).to_numpy()
    current_branch = current["candidate_branch"].fillna("").astype(str).to_numpy()
    neighbor_branch = neighbor["candidate_branch"].fillna("").astype(str).to_numpy()
    other_source = eligible & (current_source[:, None] != neighbor_source[None, :])
    other_branch = eligible & (current_branch[:, None] != neighbor_branch[None, :])
    return {
        "neighbor_position": nearest_position,
        "distance_m": np.where(has_neighbor, nearest_distance, np.nan),
        "speed_mps": nearest_speed,
        "support_count": eligible.sum(axis=1).astype(float),
        "other_source_count": other_source.sum(axis=1).astype(float),
        "other_branch_count": other_branch.sum(axis=1).astype(float),
        "neighbor_rows": neighbor,
    }


def _write_neighbor_match(
    out: pd.DataFrame,
    current: pd.DataFrame,
    match: dict[str, Any],
    *,
    direction: str,
) -> None:
    indices = current.index
    out.loc[indices, f"candidate_reservoir_temporal_{direction}_distance_m"] = match[
        "distance_m"
    ]
    out.loc[indices, f"candidate_reservoir_temporal_{direction}_speed_mps"] = match[
        "speed_mps"
    ]
    out.loc[indices, f"candidate_reservoir_temporal_{direction}_support_count"] = match[
        "support_count"
    ]
    other_source_column = "candidate_reservoir_temporal_other_source_support_count"
    other_branch_column = "candidate_reservoir_temporal_other_branch_support_count"
    out.loc[indices, other_source_column] += match["other_source_count"]
    out.loc[indices, other_branch_column] += match["other_branch_count"]

    neighbor_rows = match.get("neighbor_rows")
    if neighbor_rows is None or neighbor_rows.empty:
        return
    positions = match["neighbor_position"]
    source_values: list[str] = []
    branch_values: list[str] = []
    for neighbor_position in positions:
        if int(neighbor_position) < 0:
            source_values.append("")
            branch_values.append("")
            continue
        row = neighbor_rows.iloc[int(neighbor_position)]
        source_values.append(str(row.get("source", "")))
        branch_values.append(str(row.get("candidate_branch", "")))
    out.loc[indices, f"candidate_temporal_{direction}_source"] = source_values
    out.loc[indices, f"candidate_temporal_{direction}_branch"] = branch_values


def _write_bidirectional_metrics(
    out: pd.DataFrame,
    current: pd.DataFrame,
    previous: pd.DataFrame | None,
    next_frame: pd.DataFrame | None,
    previous_match: dict[str, Any],
    next_match: dict[str, Any],
    previous_dt: float,
    next_dt: float,
) -> None:
    if previous is None or next_frame is None:
        return
    previous_position = previous_match["neighbor_position"]
    next_position = next_match["neighbor_position"]
    supported = (previous_position >= 0) & (next_position >= 0)
    if not supported.any():
        return

    current_xyz = current[["x_m", "y_m", "z_m"]].to_numpy(float)
    interpolation = np.full(len(current), np.nan, dtype=float)
    acceleration = np.full(len(current), np.nan, dtype=float)
    for current_position in np.flatnonzero(supported):
        previous_xyz = previous.iloc[int(previous_position[current_position])][
            ["x_m", "y_m", "z_m"]
        ].to_numpy(float)
        next_xyz = next_frame.iloc[int(next_position[current_position])][
            ["x_m", "y_m", "z_m"]
        ].to_numpy(float)
        fraction = float(previous_dt / (previous_dt + next_dt))
        interpolated_xyz = previous_xyz + fraction * (next_xyz - previous_xyz)
        interpolation[current_position] = float(
            np.linalg.norm(current_xyz[current_position] - interpolated_xyz)
        )
        backward_velocity = (current_xyz[current_position] - previous_xyz) / previous_dt
        forward_velocity = (next_xyz - current_xyz[current_position]) / next_dt
        mean_dt = max(0.5 * (previous_dt + next_dt), 1.0e-9)
        acceleration[current_position] = float(
            np.linalg.norm(forward_velocity - backward_velocity) / mean_dt
        )

    indices = current.index
    out.loc[indices, "candidate_reservoir_temporal_bidirectional"] = supported.astype(float)
    out.loc[indices, "candidate_reservoir_temporal_interpolation_residual_m"] = interpolation
    out.loc[indices, "candidate_reservoir_temporal_acceleration_mps2"] = acceleration


def _add_consensus_score(out: pd.DataFrame, config: TemporalConsensusConfig) -> None:
    backward_distance = pd.to_numeric(
        out["candidate_reservoir_temporal_backward_distance_m"], errors="coerce"
    )
    forward_distance = pd.to_numeric(
        out["candidate_reservoir_temporal_forward_distance_m"], errors="coerce"
    )
    interpolation = pd.to_numeric(
        out["candidate_reservoir_temporal_interpolation_residual_m"], errors="coerce"
    )
    acceleration = pd.to_numeric(
        out["candidate_reservoir_temporal_acceleration_mps2"], errors="coerce"
    )
    backward_support = np.exp(-backward_distance.fillna(np.inf) / config.distance_scale_m)
    forward_support = np.exp(-forward_distance.fillna(np.inf) / config.distance_scale_m)
    interpolation_support = np.exp(-interpolation.fillna(np.inf) / config.distance_scale_m)
    acceleration_support = np.exp(
        -acceleration.fillna(np.inf) / config.acceleration_scale_mps2
    )
    source_diversity = (
        pd.to_numeric(
            out["candidate_reservoir_temporal_other_source_support_count"], errors="coerce"
        ).fillna(0.0)
        > 0.0
    ).astype(float)
    branch_diversity = (
        pd.to_numeric(
            out["candidate_reservoir_temporal_other_branch_support_count"], errors="coerce"
        ).fillna(0.0)
        > 0.0
    ).astype(float)
    bidirectional = pd.to_numeric(
        out["candidate_reservoir_temporal_bidirectional"], errors="coerce"
    ).fillna(0.0)
    base = pd.to_numeric(
        out["candidate_reservoir_temporal_base_score"], errors="coerce"
    ).fillna(0.0)
    score = (
        config.base_score_weight * base
        + config.backward_support_weight * backward_support
        + config.forward_support_weight * forward_support
        + config.bidirectional_bonus * bidirectional
        + config.interpolation_weight * interpolation_support
        + config.acceleration_weight * acceleration_support
        + config.source_diversity_bonus * source_diversity
        + config.branch_diversity_bonus * branch_diversity
    )
    out["candidate_temporal_consensus_score"] = score.astype(float)
    out["candidate_reservoir_temporal_score"] = score.astype(float)
    out["candidate_reservoir_temporal_supported"] = (
        np.isfinite(backward_distance.to_numpy(float))
        | np.isfinite(forward_distance.to_numpy(float))
    ).astype(float)


def _safe_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else float("nan")


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.quantile(float(quantile))) if len(finite) else float("nan")


def _safe_max(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.max()) if len(finite) else float("nan")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
