"""Temporal-support quotas for branch-preserving MMUAD candidate reservoirs.

The branch reservoir protects source and branch provenance, but a good candidate
can still be ranked below several isolated distractors inside the same
provenance cell. This module adds a light-weight, truth-free recall guard:
candidates that have motion-compatible neighbours in adjacent frames receive
temporal-support diagnostics, and a configurable number of the strongest
supported candidates are added to the reservoir before the final frame cap.

This is deliberately a pre-inference quota rather than another trajectory
smoother. It preserves coherent low-score hypotheses so the maintained
pair-state and learned-sigma Huber mixture-MAP stages can evaluate them.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    _apply_frame_cap,
    build_candidate_reservoir,
    build_oracle_recall_tables,
    build_reservoir_summary,
    load_candidate_inputs,
)
from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns


@dataclass(frozen=True)
class TemporalSupportConfig:
    """Configuration for the temporal-support reservoir quota."""

    temporal_top_n: int = 2
    max_frame_gap_s: float = 1.0
    max_speed_mps: float = 60.0
    distance_scale_m: float = 5.0
    min_support_sides: int = 1
    require_same_source: bool = False
    require_same_branch: bool = False
    protect_temporal_quota: bool = True
    reason_prefix: str = "temporal_support"


def attach_temporal_support_features(
    candidates: pd.DataFrame,
    *,
    config: TemporalSupportConfig | None = None,
) -> pd.DataFrame:
    """Attach adjacent-frame motion-support features to candidate rows.

    Each candidate is compared with candidates at the nearest earlier and later
    timestamp in the same sequence. A side contributes support when at least one
    compatible candidate implies a speed no greater than ``max_speed_mps``.
    Optional source/branch constraints can make the support test provenance
    specific.
    """

    config = config or TemporalSupportConfig()
    _validate_temporal_config(config)
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        return _empty_temporal_columns(rows)

    rows = rows.reset_index(drop=True)
    rows = _validate_candidate_geometry(rows)
    rows["_temporal_input_row"] = np.arange(len(rows), dtype=int)
    if "source" not in rows.columns:
        rows["source"] = "unknown"
    rows["source"] = rows["source"].fillna("unknown").astype(str)
    if "candidate_branch" not in rows.columns:
        rows["candidate_branch"] = rows["source"]
    rows["candidate_branch"] = rows["candidate_branch"].fillna("candidate").astype(str)

    feature_rows: list[pd.DataFrame] = []
    for _, sequence_rows in rows.groupby("sequence_id", sort=False, dropna=False):
        feature_rows.append(_attach_sequence_temporal_support(sequence_rows, config=config))
    out = pd.concat(feature_rows, ignore_index=True) if feature_rows else rows.iloc[0:0].copy()
    return out.sort_values("_temporal_input_row").reset_index(drop=True)


def build_temporal_support_reservoir(
    candidates: pd.DataFrame,
    *,
    reservoir_config: ReservoirConfig | None = None,
    temporal_config: TemporalSupportConfig | None = None,
) -> pd.DataFrame:
    """Build a branch reservoir augmented with a temporal-support quota."""

    reservoir_config = reservoir_config or ReservoirConfig()
    temporal_config = temporal_config or TemporalSupportConfig()
    _validate_temporal_config(temporal_config)

    annotated = attach_temporal_support_features(candidates, config=temporal_config)
    if annotated.empty:
        return build_candidate_reservoir(annotated, config=reservoir_config)

    base = build_candidate_reservoir(annotated, config=reservoir_config)
    temporal = _select_temporal_quota(
        annotated,
        temporal_top_n=temporal_config.temporal_top_n,
        min_support_sides=temporal_config.min_support_sides,
        reason_prefix=temporal_config.reason_prefix,
        score_column=reservoir_config.score_column,
        fallback_score_column=reservoir_config.fallback_score_column,
    )
    combined = _combine_base_and_temporal(base, temporal)

    preserve_prefixes = tuple(reservoir_config.preserve_reason_prefixes)
    if temporal_config.protect_temporal_quota:
        preserve_prefixes = (*preserve_prefixes, temporal_config.reason_prefix)
    combined = _apply_frame_cap(
        combined,
        max_candidates_per_frame=reservoir_config.max_candidates_per_frame,
        cap_reason_bonus=reservoir_config.cap_reason_bonus,
        preserve_reason_prefixes=preserve_prefixes,
    )
    combined = combined.sort_values(
        ["sequence_id", "time_s", "candidate_reservoir_rank", "source"],
    ).reset_index(drop=True)
    return combined.drop(columns=["_temporal_input_row"], errors="ignore")


def build_temporal_support_summary(
    candidates: pd.DataFrame,
    reservoir: pd.DataFrame,
    *,
    reservoir_config: ReservoirConfig,
    temporal_config: TemporalSupportConfig,
) -> dict[str, Any]:
    """Build reservoir and temporal-support diagnostics."""

    summary = build_reservoir_summary(candidates, reservoir)
    support_sides = pd.to_numeric(
        reservoir.get("candidate_temporal_support_sides", pd.Series(dtype=float)),
        errors="coerce",
    )
    support_score = pd.to_numeric(
        reservoir.get("candidate_temporal_support_score", pd.Series(dtype=float)),
        errors="coerce",
    )
    temporal_reason = reservoir.get("candidate_reservoir_reason", pd.Series(dtype=str))
    temporal_selected = pd.Series(temporal_reason).fillna("").astype(str).str.contains(
        temporal_config.reason_prefix,
        regex=False,
    )
    summary.update(
        {
            "reservoir_config": _json_ready(asdict(reservoir_config)),
            "temporal_support_config": _json_ready(asdict(temporal_config)),
            "temporal_quota_candidate_rows": int(temporal_selected.sum()),
            "temporal_supported_rows": int((support_sides > 0).sum()),
            "temporal_two_sided_rows": int((support_sides >= 2).sum()),
            "temporal_support_sides_mean": _finite_mean(support_sides),
            "temporal_support_score_mean": _finite_mean(support_score),
            "temporal_support_score_p95": _finite_quantile(support_score, 0.95),
        }
    )
    return summary


def write_temporal_support_outputs(
    reservoir: pd.DataFrame,
    *,
    output_csv: Path,
    summary_json: Path | None,
    input_candidates: pd.DataFrame,
    reservoir_config: ReservoirConfig,
    temporal_config: TemporalSupportConfig,
) -> None:
    """Write the augmented reservoir and optional summary."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    reservoir.to_csv(output_csv, index=False)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = build_temporal_support_summary(
            input_candidates,
            reservoir,
            reservoir_config=reservoir_config,
            temporal_config=temporal_config,
        )
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mmuad-temporal-support-reservoir",
        description="preserve motion-supported candidates in an MMUAD branch reservoir",
    )
    parser.add_argument(
        "--candidate-csv",
        "--candidate",
        action="append",
        dest="candidate_specs",
        default=[],
        help="candidate CSV as BRANCH=path; may be repeated",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--oracle-frame-csv", type=Path)
    parser.add_argument("--oracle-summary-csv", type=Path)
    parser.add_argument("--oracle-by-sequence-csv", type=Path)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--fallback-score-column", default="confidence")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--cap-reason-bonus", type=float, default=0.0)
    parser.add_argument("--temporal-top-n", type=int, default=2)
    parser.add_argument("--max-frame-gap-s", type=float, default=1.0)
    parser.add_argument("--max-speed-mps", type=float, default=60.0)
    parser.add_argument("--distance-scale-m", type=float, default=5.0)
    parser.add_argument("--min-support-sides", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--require-same-source", action="store_true")
    parser.add_argument("--require-same-branch", action="store_true")
    parser.add_argument("--do-not-protect-temporal-quota", action="store_true")
    parser.add_argument("--top-k", type=int, action="append", default=None)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    if not args.candidate_specs:
        raise ValueError("at least one --candidate-csv BRANCH=PATH entry is required")
    reservoir_config = ReservoirConfig(
        global_top_n=args.global_top_n,
        per_source_top_n=args.per_source_top_n,
        per_branch_top_n=args.per_branch_top_n,
        max_candidates_per_frame=args.max_candidates_per_frame,
        score_column=args.score_column,
        fallback_score_column=args.fallback_score_column,
        score_floor_quantile=args.score_floor_quantile,
        cap_reason_bonus=args.cap_reason_bonus,
    )
    temporal_config = TemporalSupportConfig(
        temporal_top_n=args.temporal_top_n,
        max_frame_gap_s=args.max_frame_gap_s,
        max_speed_mps=args.max_speed_mps,
        distance_scale_m=args.distance_scale_m,
        min_support_sides=args.min_support_sides,
        require_same_source=args.require_same_source,
        require_same_branch=args.require_same_branch,
        protect_temporal_quota=not args.do_not_protect_temporal_quota,
    )
    candidates = load_candidate_inputs(args.candidate_specs)
    reservoir = build_temporal_support_reservoir(
        candidates,
        reservoir_config=reservoir_config,
        temporal_config=temporal_config,
    )
    write_temporal_support_outputs(
        reservoir,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        input_candidates=candidates,
        reservoir_config=reservoir_config,
        temporal_config=temporal_config,
    )
    print("mmuad_temporal_support_reservoir=ok")
    print(f"candidate_rows={len(candidates)}")
    print(f"reservoir_rows={len(reservoir)}")
    print(f"output_csv={args.output_csv}")

    if args.truth_csv is not None:
        truth = normalize_truth_columns(pd.read_csv(args.truth_csv))
        top_k_values = tuple(args.top_k) if args.top_k else (1, 3, 5, 10, 20)
        frame_rows, pooled, by_sequence = build_oracle_recall_tables(
            reservoir,
            truth,
            top_k_values=top_k_values,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
        )
        _write_optional_csv(frame_rows, args.oracle_frame_csv)
        _write_optional_csv(pooled, args.oracle_summary_csv)
        _write_optional_csv(by_sequence, args.oracle_by_sequence_csv)
        print(f"oracle_frames={len(frame_rows)}")
        if not pooled.empty:
            print(f"oracle_all_mse={pooled.loc[0, 'oracle_all_3d_m_mse']}")
    return 0


def _attach_sequence_temporal_support(
    sequence_rows: pd.DataFrame,
    *,
    config: TemporalSupportConfig,
) -> pd.DataFrame:
    rows = sequence_rows.copy()
    times = np.sort(pd.to_numeric(rows["time_s"], errors="coerce").dropna().unique())
    by_time = {
        float(time_s): group.copy()
        for time_s, group in rows.groupby("time_s", sort=False, dropna=False)
        if np.isfinite(float(time_s))
    }
    records: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        time_s = float(row["time_s"])
        previous_time, next_time = _adjacent_times(times, time_s)
        previous = _best_temporal_match(
            row,
            by_time.get(previous_time),
            time_delta_s=None if previous_time is None else time_s - previous_time,
            config=config,
        )
        following = _best_temporal_match(
            row,
            by_time.get(next_time),
            time_delta_s=None if next_time is None else next_time - time_s,
            config=config,
        )
        support_sides = int(previous["supported"]) + int(following["supported"])
        support_score = float(previous["quality"]) + float(following["quality"])
        record = row.to_dict()
        record.update(
            {
                "candidate_temporal_prev_distance_m": previous["distance_m"],
                "candidate_temporal_prev_speed_mps": previous["speed_mps"],
                "candidate_temporal_prev_dt_s": previous["dt_s"],
                "candidate_temporal_next_distance_m": following["distance_m"],
                "candidate_temporal_next_speed_mps": following["speed_mps"],
                "candidate_temporal_next_dt_s": following["dt_s"],
                "candidate_temporal_support_sides": support_sides,
                "candidate_temporal_support_score": support_score,
                "candidate_temporal_two_sided": support_sides == 2,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _adjacent_times(times: np.ndarray, time_s: float) -> tuple[float | None, float | None]:
    position = int(np.searchsorted(times, time_s, side="left"))
    previous = float(times[position - 1]) if position > 0 else None
    next_index = int(np.searchsorted(times, time_s, side="right"))
    following = float(times[next_index]) if next_index < len(times) else None
    return previous, following


def _best_temporal_match(
    row: pd.Series,
    candidates: pd.DataFrame | None,
    *,
    time_delta_s: float | None,
    config: TemporalSupportConfig,
) -> dict[str, Any]:
    empty = {
        "supported": False,
        "distance_m": float("nan"),
        "speed_mps": float("nan"),
        "dt_s": float("nan") if time_delta_s is None else float(time_delta_s),
        "quality": 0.0,
    }
    if candidates is None or candidates.empty or time_delta_s is None:
        return empty
    dt = float(time_delta_s)
    if not np.isfinite(dt) or dt <= 0.0 or dt > config.max_frame_gap_s:
        return empty
    compatible = candidates.copy()
    if config.require_same_source:
        compatible = compatible.loc[compatible["source"].astype(str) == str(row["source"])]
    if config.require_same_branch:
        compatible = compatible.loc[
            compatible["candidate_branch"].astype(str) == str(row["candidate_branch"])
        ]
    if compatible.empty:
        return empty
    origin = row[["x_m", "y_m", "z_m"]].to_numpy(float)
    target = compatible[["x_m", "y_m", "z_m"]].to_numpy(float)
    distances = np.linalg.norm(target - origin, axis=1)
    finite = np.isfinite(distances)
    if not finite.any():
        return empty
    distance = float(np.min(distances[finite]))
    speed = distance / dt
    supported = bool(speed <= config.max_speed_mps)
    quality = float(np.exp(-distance / config.distance_scale_m)) if supported else 0.0
    return {
        "supported": supported,
        "distance_m": distance,
        "speed_mps": speed,
        "dt_s": dt,
        "quality": quality,
    }


def _select_temporal_quota(
    rows: pd.DataFrame,
    *,
    temporal_top_n: int,
    min_support_sides: int,
    reason_prefix: str,
    score_column: str,
    fallback_score_column: str,
) -> pd.DataFrame:
    if temporal_top_n <= 0 or rows.empty:
        return rows.iloc[0:0].copy()
    work = rows.copy()
    work["_temporal_ranker_score"] = _rowwise_score(
        work,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
    )
    work["_temporal_stable_key"] = work.apply(_stable_candidate_key, axis=1)
    selected: list[pd.DataFrame] = []
    for _, frame in work.groupby(["sequence_id", "time_s"], sort=False, dropna=False):
        supported = frame.loc[
            pd.to_numeric(frame["candidate_temporal_support_sides"], errors="coerce")
            >= int(min_support_sides)
        ].copy()
        if supported.empty:
            continue
        supported = supported.sort_values(
            [
                "candidate_temporal_support_sides",
                "candidate_temporal_support_score",
                "_temporal_ranker_score",
                "_temporal_stable_key",
            ],
            ascending=[False, False, False, True],
        ).head(int(temporal_top_n))
        supported["candidate_reservoir_score"] = supported["_temporal_ranker_score"]
        supported["candidate_reservoir_reason"] = [
            f"{reason_prefix}:{int(value)}side"
            for value in supported["candidate_temporal_support_sides"]
        ]
        supported["candidate_reservoir_reasons"] = supported["candidate_reservoir_reason"]
        selected.append(supported)
    if not selected:
        return work.iloc[0:0].drop(
            columns=["_temporal_ranker_score", "_temporal_stable_key"],
            errors="ignore",
        )
    return pd.concat(selected, ignore_index=True).drop(
        columns=["_temporal_ranker_score", "_temporal_stable_key"],
        errors="ignore",
    )


def _combine_base_and_temporal(base: pd.DataFrame, temporal: pd.DataFrame) -> pd.DataFrame:
    if temporal.empty:
        return base.copy()
    if base.empty:
        combined = temporal.copy()
    else:
        combined = pd.concat([base, temporal], ignore_index=True, sort=False)
    if "_temporal_input_row" not in combined.columns:
        raise ValueError("temporal reservoir rows require _temporal_input_row provenance")

    records: list[pd.Series] = []
    for _, group in combined.groupby("_temporal_input_row", sort=False, dropna=False):
        row = group.iloc[0].copy()
        reasons: set[str] = set()
        for value in group.get("candidate_reservoir_reason", pd.Series(dtype=str)):
            for token in str(value).replace(",", ";").split(";"):
                token = token.strip()
                if token and token.lower() != "nan":
                    reasons.add(token)
        row["candidate_reservoir_reason"] = ";".join(sorted(reasons))
        row["candidate_reservoir_reasons"] = row["candidate_reservoir_reason"]
        records.append(row)
    out = pd.DataFrame(records).reset_index(drop=True)
    return out.drop(
        columns=[
            "candidate_reservoir_reason_count",
            "candidate_reservoir_cap_score",
            "candidate_reservoir_protected",
            "candidate_reservoir_rank",
        ],
        errors="ignore",
    )


def _rowwise_score(
    rows: pd.DataFrame,
    *,
    score_column: str,
    fallback_score_column: str,
) -> pd.Series:
    primary = _numeric_series(rows, score_column, default=np.nan)
    fallback = _numeric_series(rows, fallback_score_column, default=0.0)
    primary = primary.where(np.isfinite(primary), np.nan)
    fallback = fallback.where(np.isfinite(fallback), 0.0)
    return primary.fillna(fallback).fillna(0.0).astype(float)


def _numeric_series(rows: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(default, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _stable_candidate_key(row: pd.Series) -> str:
    values = (
        row.get("source", ""),
        row.get("candidate_branch", ""),
        row.get("track_id", ""),
        row.get("x_m", ""),
        row.get("y_m", ""),
        row.get("z_m", ""),
    )
    return "|".join(str(value) for value in values)


def _validate_candidate_geometry(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    numeric_columns = ("time_s", "x_m", "y_m", "z_m")
    numeric = out.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(float)).all(axis=1)
    if not bool(valid.all()):
        bad_rows = np.flatnonzero(~valid).tolist()
        raise ValueError(
            "temporal-support candidates require finite time/position rows; "
            f"invalid row indices: {bad_rows[:10]}"
        )
    out.loc[:, numeric_columns] = numeric
    return out


def _validate_temporal_config(config: TemporalSupportConfig) -> None:
    numeric = {
        "max_frame_gap_s": config.max_frame_gap_s,
        "max_speed_mps": config.max_speed_mps,
        "distance_scale_m": config.distance_scale_m,
    }
    for name, value in numeric.items():
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if int(config.temporal_top_n) < 0:
        raise ValueError("temporal_top_n must be non-negative")
    if int(config.min_support_sides) not in (0, 1, 2):
        raise ValueError("min_support_sides must be 0, 1, or 2")
    if not str(config.reason_prefix).strip():
        raise ValueError("reason_prefix must be non-empty")


def _empty_temporal_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    for column in (
        "candidate_temporal_prev_distance_m",
        "candidate_temporal_prev_speed_mps",
        "candidate_temporal_prev_dt_s",
        "candidate_temporal_next_distance_m",
        "candidate_temporal_next_speed_mps",
        "candidate_temporal_next_dt_s",
        "candidate_temporal_support_score",
    ):
        out[column] = pd.Series(dtype=float)
    out["candidate_temporal_support_sides"] = pd.Series(dtype=int)
    out["candidate_temporal_two_sided"] = pd.Series(dtype=bool)
    return out


def _write_optional_csv(rows: pd.DataFrame, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def _finite_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else float("nan")


def _finite_quantile(values: pd.Series, quantile: float) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.quantile(float(quantile))) if len(finite) else float("nan")


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
