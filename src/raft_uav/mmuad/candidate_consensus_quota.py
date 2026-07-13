"""Preserve cross-sensor-supported MMUAD candidates before trajectory inference.

Branch, source, and source-by-branch quotas preserve provenance, but they remain
unary-score driven inside each cell. A lower-scoring candidate can still be the
only hypothesis geometrically supported by another sensor. This module adds a
truth-free per-frame quota for candidates with independent cross-source support
before the final reservoir cap is applied.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_branch_consensus import (
    DEFAULT_ORIGIN_COLUMN_ALIASES,
    attach_candidate_branch_consensus,
)
from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    _apply_frame_cap,
    build_candidate_reservoir,
    build_oracle_recall_tables,
    build_reservoir_summary,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns

_CONSENSUS_REASON = "consensus:cross_source"
_CAP_DIAGNOSTIC_COLUMNS = (
    "candidate_reservoir_rank",
    "candidate_reservoir_reason_count",
    "candidate_reservoir_cap_score",
    "candidate_reservoir_protected",
)


def build_consensus_quota_reservoir(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    reservoir_config: ReservoirConfig | None = None,
    consensus_top_n: int = 2,
    min_neighbor_count: int = 1,
    min_unique_source_count: int = 1,
    max_nearest_distance_m: float = 5.0,
    max_per_origin: int = 1,
    max_per_source: int = 0,
    selection_score_column: str = "branch_consensus_score",
    time_window_s: float = 0.05,
    time_scale_s: float | None = None,
    distance_gate_m: float = 5.0,
    distance_scale_m: float = 5.0,
    base_score_column: str = "ranker_score",
    base_score_weight: float = 1.0,
    consensus_weight: float = 1.0,
    pair_advantage_weight: float = 0.25,
    branch_column: str | None = None,
    origin_column: str | None = None,
    exclude_same_origin_support: bool = True,
) -> CandidateFrame:
    """Build a reservoir with a protected quota for cross-source consensus rows.

    Consensus is computed on the complete input pool before any pruning. The base
    branch/source reservoir is then built without a final cap, consensus-supported
    rows are added, and the hard frame cap is applied while retaining the selected
    quota rows. ``max_per_origin`` prevents raw/calibrated copies of one physical
    observation from consuming the complete quota.
    """

    config = reservoir_config or ReservoirConfig()
    _validate_controls(
        consensus_top_n=consensus_top_n,
        min_neighbor_count=min_neighbor_count,
        min_unique_source_count=min_unique_source_count,
        max_nearest_distance_m=max_nearest_distance_m,
        max_per_origin=max_per_origin,
        max_per_source=max_per_source,
        time_window_s=time_window_s,
        time_scale_s=time_scale_s,
        distance_gate_m=distance_gate_m,
        distance_scale_m=distance_scale_m,
    )
    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(normalize_candidate_columns(rows))

    rows = rows.copy().reset_index(drop=True)
    rows["_consensus_quota_row_id"] = np.arange(len(rows), dtype=int)
    augmented = attach_candidate_branch_consensus(
        CandidateFrame(normalize_candidate_columns(rows)),
        time_window_s=time_window_s,
        time_scale_s=time_scale_s,
        distance_gate_m=distance_gate_m,
        distance_scale_m=distance_scale_m,
        base_score_column=base_score_column,
        base_score_weight=base_score_weight,
        consensus_weight=consensus_weight,
        pair_advantage_weight=pair_advantage_weight,
        branch_column=branch_column,
        origin_column=origin_column,
        exclude_same_origin_support=exclude_same_origin_support,
    ).rows.copy()
    if selection_score_column not in augmented.columns:
        raise ValueError(
            f"selection_score_column {selection_score_column!r} is not present after consensus"
        )
    augmented["_consensus_quota_base_score"] = _resolve_score(
        augmented,
        primary=config.score_column,
        fallback=config.fallback_score_column,
    )
    augmented["candidate_consensus_supported"] = _supported_mask(
        augmented,
        min_neighbor_count=min_neighbor_count,
        min_unique_source_count=min_unique_source_count,
        max_nearest_distance_m=max_nearest_distance_m,
    )

    if int(consensus_top_n) == 0:
        base = build_candidate_reservoir(augmented, config=config)
        out = base.copy()
        out["candidate_consensus_quota_selected"] = False
        out["candidate_consensus_quota_rank"] = np.nan
        out["candidate_consensus_quota_top_n"] = 0
        return CandidateFrame(_drop_private_columns(normalize_candidate_columns(out)))

    uncapped_config = replace(config, max_candidates_per_frame=0)
    base = build_candidate_reservoir(augmented, config=uncapped_config)
    selected_by_id = _records_by_row_id(base)
    resolved_origin = _resolve_origin_column(augmented, origin_column)

    for _, frame in augmented.groupby(["sequence_id", "time_s"], sort=False, dropna=False):
        selected = _select_consensus_rows(
            frame,
            count=int(consensus_top_n),
            selection_score_column=selection_score_column,
            origin_column=resolved_origin,
            max_per_origin=int(max_per_origin),
            max_per_source=int(max_per_source),
        )
        for _, candidate in selected.iterrows():
            row_id = int(candidate["_consensus_quota_row_id"])
            diagnostics = {
                "candidate_consensus_quota_selected": True,
                "candidate_consensus_quota_rank": int(candidate["_consensus_selection_rank"]),
                "candidate_consensus_quota_top_n": int(consensus_top_n),
                "candidate_consensus_quota_selection_score": float(
                    candidate["_consensus_selection_score"]
                ),
            }
            existing = selected_by_id.get(row_id)
            if existing is None:
                existing = candidate.to_dict()
                existing["candidate_reservoir_score"] = float(
                    candidate["_consensus_quota_base_score"]
                )
                existing["candidate_reservoir_reason"] = _CONSENSUS_REASON
                existing["candidate_reservoir_reasons"] = _CONSENSUS_REASON
                existing.update(diagnostics)
                selected_by_id[row_id] = existing
            else:
                reason = _merge_reason_tokens(
                    existing.get("candidate_reservoir_reason", ""),
                    _CONSENSUS_REASON,
                )
                existing["candidate_reservoir_reason"] = reason
                existing["candidate_reservoir_reasons"] = reason
                existing.update(diagnostics)

    union = pd.DataFrame.from_records(list(selected_by_id.values()))
    if union.empty:
        return CandidateFrame(normalize_candidate_columns(union))
    if "candidate_consensus_quota_selected" not in union.columns:
        union["candidate_consensus_quota_selected"] = False
    union["candidate_consensus_quota_selected"] = (
        union["candidate_consensus_quota_selected"].fillna(False).astype(bool)
    )
    union["candidate_consensus_quota_top_n"] = int(consensus_top_n)

    preserve_prefixes = tuple(config.preserve_reason_prefixes)
    if "consensus:" not in preserve_prefixes:
        preserve_prefixes = (*preserve_prefixes, "consensus:")
    capped = _cap_with_mandatory_consensus(
        union,
        max_candidates_per_frame=int(config.max_candidates_per_frame),
        cap_reason_bonus=float(config.cap_reason_bonus),
        preserve_reason_prefixes=preserve_prefixes,
    )
    capped = capped.sort_values(
        ["sequence_id", "time_s", "candidate_reservoir_rank", "source"],
    ).reset_index(drop=True)
    return CandidateFrame(_drop_private_columns(normalize_candidate_columns(capped)))


def consensus_quota_summary(
    candidates: CandidateFrame | pd.DataFrame,
    reservoir: CandidateFrame | pd.DataFrame,
) -> dict[str, Any]:
    """Return base reservoir diagnostics plus cross-source quota statistics."""

    input_rows = _candidate_rows(candidates)
    output_rows = _candidate_rows(reservoir)
    summary = build_reservoir_summary(input_rows, output_rows)
    selected = _bool_column(output_rows, "candidate_consensus_quota_selected")
    supported = _bool_column(output_rows, "candidate_consensus_supported")
    nearest = _finite_numeric(
        output_rows,
        "branch_consensus_nearest_cross_source_distance_m",
    )
    unique_sources = _finite_numeric(output_rows, "branch_consensus_unique_source_count")
    summary.update(
        {
            "consensus_supported_output_rows": int(supported.sum()),
            "consensus_quota_selected_rows": int(selected.sum()),
            "consensus_quota_selected_frame_count": int(
                output_rows.loc[selected, ["sequence_id", "time_s"]].drop_duplicates().shape[0]
            )
            if len(output_rows)
            else 0,
            "consensus_nearest_distance_mean_m": _safe_mean(nearest),
            "consensus_nearest_distance_p95_m": _safe_quantile(nearest, 0.95),
            "consensus_unique_source_count_mean": _safe_mean(unique_sources),
        }
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-consensus-quota",
        description="preserve cross-sensor-supported candidates in an MMUAD reservoir",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-reservoir-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--score-column", default="candidate_risk_adjusted_score")
    parser.add_argument("--fallback-score-column", default="ranker_score")
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--cap-reason-bonus", type=float, default=0.0)
    parser.add_argument("--consensus-top-n", type=int, default=2)
    parser.add_argument("--min-neighbor-count", type=int, default=1)
    parser.add_argument("--min-unique-source-count", type=int, default=1)
    parser.add_argument("--max-nearest-distance-m", type=float, default=5.0)
    parser.add_argument("--max-per-origin", type=int, default=1)
    parser.add_argument("--max-per-source", type=int, default=0)
    parser.add_argument("--selection-score-column", default="branch_consensus_score")
    parser.add_argument("--consensus-time-window-s", type=float, default=0.05)
    parser.add_argument("--consensus-time-scale-s", type=float)
    parser.add_argument("--consensus-distance-gate-m", type=float, default=5.0)
    parser.add_argument("--consensus-distance-scale-m", type=float, default=5.0)
    parser.add_argument("--consensus-base-score-column", default="ranker_score")
    parser.add_argument("--consensus-base-score-weight", type=float, default=1.0)
    parser.add_argument("--consensus-weight", type=float, default=1.0)
    parser.add_argument("--consensus-pair-advantage-weight", type=float, default=0.25)
    parser.add_argument("--branch-column")
    parser.add_argument("--origin-column")
    parser.add_argument("--allow-same-origin-support", action="store_true")
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--oracle-frame-csv", type=Path)
    parser.add_argument("--oracle-summary-csv", type=Path)
    parser.add_argument("--oracle-by-sequence-csv", type=Path)
    parser.add_argument("--top-k", action="append", type=int, default=None)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    candidates = load_candidate_file(args.candidates_csv)
    config = ReservoirConfig(
        global_top_n=args.global_top_n,
        per_source_top_n=args.per_source_top_n,
        per_branch_top_n=args.per_branch_top_n,
        max_candidates_per_frame=args.max_candidates_per_frame,
        score_column=args.score_column,
        fallback_score_column=args.fallback_score_column,
        score_floor_quantile=args.score_floor_quantile,
        cap_reason_bonus=args.cap_reason_bonus,
    )
    reservoir = build_consensus_quota_reservoir(
        candidates,
        reservoir_config=config,
        consensus_top_n=args.consensus_top_n,
        min_neighbor_count=args.min_neighbor_count,
        min_unique_source_count=args.min_unique_source_count,
        max_nearest_distance_m=args.max_nearest_distance_m,
        max_per_origin=args.max_per_origin,
        max_per_source=args.max_per_source,
        selection_score_column=args.selection_score_column,
        time_window_s=args.consensus_time_window_s,
        time_scale_s=args.consensus_time_scale_s,
        distance_gate_m=args.consensus_distance_gate_m,
        distance_scale_m=args.consensus_distance_scale_m,
        base_score_column=args.consensus_base_score_column,
        base_score_weight=args.consensus_base_score_weight,
        consensus_weight=args.consensus_weight,
        pair_advantage_weight=args.consensus_pair_advantage_weight,
        branch_column=args.branch_column,
        origin_column=args.origin_column,
        exclude_same_origin_support=not args.allow_same_origin_support,
    )
    args.output_reservoir_csv.parent.mkdir(parents=True, exist_ok=True)
    reservoir.rows.to_csv(args.output_reservoir_csv, index=False)

    summary = consensus_quota_summary(candidates, reservoir)
    summary.update(
        {
            "consensus_top_n": int(args.consensus_top_n),
            "min_neighbor_count": int(args.min_neighbor_count),
            "min_unique_source_count": int(args.min_unique_source_count),
            "max_nearest_distance_m": float(args.max_nearest_distance_m),
            "max_per_origin": int(args.max_per_origin),
            "max_per_source": int(args.max_per_source),
            "selection_score_column": str(args.selection_score_column),
            "consensus_time_window_s": float(args.consensus_time_window_s),
            "consensus_distance_gate_m": float(args.consensus_distance_gate_m),
            "consensus_distance_scale_m": float(args.consensus_distance_scale_m),
        }
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.truth_csv is not None:
        truth = load_evaluation_truth_file(args.truth_csv).rows
        top_k_values = tuple(args.top_k) if args.top_k is not None else (1, 3, 5, 10, 20)
        frame_rows, pooled, by_sequence = build_oracle_recall_tables(
            reservoir.rows,
            truth,
            top_k_values=top_k_values,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
        )
        _write_optional_csv(frame_rows, args.oracle_frame_csv)
        _write_optional_csv(pooled, args.oracle_summary_csv)
        _write_optional_csv(by_sequence, args.oracle_by_sequence_csv)

    print("mmuad_candidate_consensus_quota=ok")
    print(f"candidate_rows={len(_candidate_rows(candidates))}")
    print(f"reservoir_rows={len(reservoir.rows)}")
    print(f"output_reservoir_csv={args.output_reservoir_csv}")
    return 0


def _select_consensus_rows(
    frame: pd.DataFrame,
    *,
    count: int,
    selection_score_column: str,
    origin_column: str | None,
    max_per_origin: int,
    max_per_source: int,
) -> pd.DataFrame:
    supported = frame.loc[frame["candidate_consensus_supported"].fillna(False)].copy()
    if count <= 0 or supported.empty:
        return supported.iloc[0:0].copy()
    supported["_consensus_selection_score"] = pd.to_numeric(
        supported[selection_score_column],
        errors="coerce",
    ).fillna(float("-inf"))
    supported["_consensus_nearest_distance"] = pd.to_numeric(
        supported["branch_consensus_nearest_cross_source_distance_m"],
        errors="coerce",
    ).fillna(float("inf"))
    supported["_consensus_unique_sources"] = pd.to_numeric(
        supported["branch_consensus_unique_source_count"],
        errors="coerce",
    ).fillna(0.0)
    supported["_consensus_neighbors"] = pd.to_numeric(
        supported["branch_consensus_neighbor_count"],
        errors="coerce",
    ).fillna(0.0)
    supported["_consensus_origin_key"] = _origin_keys(supported, origin_column)
    supported = supported.sort_values(
        [
            "_consensus_selection_score",
            "_consensus_unique_sources",
            "_consensus_neighbors",
            "_consensus_nearest_distance",
            "_consensus_quota_base_score",
            "_consensus_quota_row_id",
        ],
        ascending=[False, False, False, True, False, True],
    )
    origin_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    selected_indices: list[int] = []
    for index, row in supported.iterrows():
        origin = str(row["_consensus_origin_key"])
        source = str(row.get("source", "unknown"))
        if origin_counts[origin] >= int(max_per_origin):
            continue
        if int(max_per_source) > 0 and source_counts[source] >= int(max_per_source):
            continue
        selected_indices.append(int(index))
        origin_counts[origin] += 1
        source_counts[source] += 1
        if len(selected_indices) >= int(count):
            break
    selected = supported.loc[selected_indices].copy()
    selected["_consensus_selection_rank"] = np.arange(1, len(selected) + 1, dtype=int)
    return selected


def _cap_with_mandatory_consensus(
    rows: pd.DataFrame,
    *,
    max_candidates_per_frame: int,
    cap_reason_bonus: float,
    preserve_reason_prefixes: Sequence[str],
) -> pd.DataFrame:
    clean = _drop_cap_diagnostics(rows)
    if max_candidates_per_frame <= 0:
        return _apply_frame_cap(
            clean,
            max_candidates_per_frame=max_candidates_per_frame,
            cap_reason_bonus=cap_reason_bonus,
            preserve_reason_prefixes=preserve_reason_prefixes,
        )

    parts: list[pd.DataFrame] = []
    for _, frame in clean.groupby(["sequence_id", "time_s"], sort=False, dropna=False):
        mandatory = frame.loc[
            frame["candidate_consensus_quota_selected"].fillna(False).astype(bool)
        ].sort_values(
            ["candidate_consensus_quota_rank", "candidate_reservoir_score"],
            ascending=[True, False],
        )
        mandatory = mandatory.head(int(max_candidates_per_frame)).copy()
        remaining_budget = int(max_candidates_per_frame) - len(mandatory)
        remainder = frame.drop(index=mandatory.index)
        if remaining_budget > 0 and not remainder.empty:
            remainder = _apply_frame_cap(
                _drop_cap_diagnostics(remainder),
                max_candidates_per_frame=remaining_budget,
                cap_reason_bonus=cap_reason_bonus,
                preserve_reason_prefixes=preserve_reason_prefixes,
            )
            combined = pd.concat([mandatory, remainder], ignore_index=True)
        else:
            combined = mandatory
        combined = _apply_frame_cap(
            _drop_cap_diagnostics(combined),
            max_candidates_per_frame=0,
            cap_reason_bonus=cap_reason_bonus,
            preserve_reason_prefixes=preserve_reason_prefixes,
        )
        parts.append(combined)
    return pd.concat(parts, ignore_index=True) if parts else clean.iloc[0:0].copy()


def _supported_mask(
    rows: pd.DataFrame,
    *,
    min_neighbor_count: int,
    min_unique_source_count: int,
    max_nearest_distance_m: float,
) -> pd.Series:
    neighbors = pd.to_numeric(rows["branch_consensus_neighbor_count"], errors="coerce").fillna(0)
    sources = pd.to_numeric(
        rows["branch_consensus_unique_source_count"],
        errors="coerce",
    ).fillna(0)
    nearest = pd.to_numeric(
        rows["branch_consensus_nearest_cross_source_distance_m"],
        errors="coerce",
    )
    return (
        (neighbors >= int(min_neighbor_count))
        & (sources >= int(min_unique_source_count))
        & np.isfinite(nearest.to_numpy(float))
        & (nearest <= float(max_nearest_distance_m))
    )


def _records_by_row_id(rows: pd.DataFrame) -> dict[int, dict[str, Any]]:
    return {
        int(row_id): record
        for row_id, record in zip(
            pd.to_numeric(rows["_consensus_quota_row_id"], errors="raise").astype(int),
            rows.to_dict(orient="records"),
            strict=True,
        )
    }


def _resolve_origin_column(rows: pd.DataFrame, requested: str | None) -> str | None:
    if requested is not None:
        if requested not in rows.columns:
            raise ValueError(f"origin_column {requested!r} is not present")
        return requested
    return next((column for column in DEFAULT_ORIGIN_COLUMN_ALIASES if column in rows.columns), None)


def _origin_keys(rows: pd.DataFrame, origin_column: str | None) -> pd.Series:
    if origin_column is None:
        return rows["_consensus_quota_row_id"].map(lambda value: f"row:{int(value)}")
    values = rows[origin_column]
    keys: list[str] = []
    for row_id, value in zip(rows["_consensus_quota_row_id"], values, strict=True):
        if pd.isna(value) or not str(value).strip():
            keys.append(f"row:{int(row_id)}")
        else:
            keys.append(f"origin:{str(value).strip()}")
    return pd.Series(keys, index=rows.index, dtype=str)


def _resolve_score(rows: pd.DataFrame, *, primary: str, fallback: str) -> pd.Series:
    primary_values = _finite_or_nan(rows.get(primary, pd.Series(np.nan, index=rows.index)))
    fallback_values = _finite_or_nan(rows.get(fallback, pd.Series(0.0, index=rows.index)))
    return primary_values.fillna(fallback_values).fillna(0.0).astype(float)


def _finite_or_nan(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.where(np.isfinite(numeric.to_numpy(float)), np.nan)


def _merge_reason_tokens(existing: Any, extra: str) -> str:
    tokens = {
        token.strip()
        for token in str(existing).replace(",", ";").split(";")
        if token.strip()
    }
    tokens.add(str(extra))
    return ";".join(sorted(tokens))


def _drop_cap_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.drop(columns=list(_CAP_DIAGNOSTIC_COLUMNS), errors="ignore").copy()


def _drop_private_columns(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.drop(
        columns=[column for column in rows.columns if column.startswith("_consensus_")],
        errors="ignore",
    )


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    if isinstance(candidates, CandidateFrame):
        return normalize_candidate_columns(candidates.rows.copy())
    return normalize_candidate_columns(pd.DataFrame(candidates).copy())


def _bool_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(False, index=rows.index, dtype=bool)
    return rows[column].fillna(False).astype(bool)


def _finite_numeric(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(rows[column], errors="coerce")
    return values.loc[np.isfinite(values.to_numpy(float))]


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    return float(values.quantile(float(quantile))) if len(values) else float("nan")


def _write_optional_csv(rows: pd.DataFrame, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def _validate_controls(
    *,
    consensus_top_n: int,
    min_neighbor_count: int,
    min_unique_source_count: int,
    max_nearest_distance_m: float,
    max_per_origin: int,
    max_per_source: int,
    time_window_s: float,
    time_scale_s: float | None,
    distance_gate_m: float,
    distance_scale_m: float,
) -> None:
    integer_nonnegative = {
        "consensus_top_n": consensus_top_n,
        "min_neighbor_count": min_neighbor_count,
        "min_unique_source_count": min_unique_source_count,
        "max_per_source": max_per_source,
    }
    for name, value in integer_nonnegative.items():
        if int(value) != value or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if int(max_per_origin) != max_per_origin or int(max_per_origin) <= 0:
        raise ValueError("max_per_origin must be a positive integer")
    finite_positive = {
        "max_nearest_distance_m": max_nearest_distance_m,
        "distance_gate_m": distance_gate_m,
        "distance_scale_m": distance_scale_m,
    }
    for name, value in finite_positive.items():
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not np.isfinite(float(time_window_s)) or float(time_window_s) < 0.0:
        raise ValueError("time_window_s must be non-negative and finite")
    if time_scale_s is not None and (
        not np.isfinite(float(time_scale_s)) or float(time_scale_s) <= 0.0
    ):
        raise ValueError("time_scale_s must be positive and finite when provided")


if __name__ == "__main__":
    raise SystemExit(main())
