"""Build source-by-branch quota reservoirs for MMUAD candidate inference.

The ordinary branch-preserving reservoir keeps top candidates per source and per
branch independently. That can still drop an entire source/branch intersection:
for example, the best translated Livox candidate may be neither the best Livox
candidate nor the best translated candidate. This module adds an optional quota
for every ``(source, candidate_branch)`` cell before the final per-frame cap.

When a cell quota keeps multiple rows, an optional spatial-diversity term avoids
spending the quota on near-duplicate hypotheses. The quota remains inference-safe:
it uses only candidate scores, geometry, and metadata. Optional truth is used only
for oracle-recall diagnostics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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

_DEFAULT_SOURCE_BRANCH_REASON_PREFIX = "source_branch:"


def build_source_branch_reservoir(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    reservoir_config: ReservoirConfig | None = None,
    per_source_branch_top_n: int = 1,
    source_branch_diversity_weight: float = 0.0,
    source_branch_diversity_scale_m: float = 10.0,
    source_branch_distance_cap_m: float = 50.0,
) -> CandidateFrame:
    """Return a branch-preserving reservoir with source/branch intersection quotas.

    The base reservoir is first built without its final cap. Top candidates from
    each ``(source, candidate_branch)`` cell are then added, selection reasons are
    merged, and the configured cap is applied once to the complete union.

    For quotas larger than one, ``source_branch_diversity_weight`` can trade a
    normalized within-cell score against distance from candidates already retained
    in the same cell. A weight of zero exactly recovers score-only selection.
    """

    config = reservoir_config or ReservoirConfig()
    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(normalize_candidate_columns(rows))
    _validate_selection_config(
        per_source_branch_top_n=per_source_branch_top_n,
        diversity_weight=source_branch_diversity_weight,
        diversity_scale_m=source_branch_diversity_scale_m,
        distance_cap_m=source_branch_distance_cap_m,
    )

    rows = _normalize_source_branch_columns(rows)
    rows["_source_branch_reservoir_row_id"] = np.arange(len(rows), dtype=int)
    rows["_source_branch_base_score"] = _resolve_score(
        rows,
        primary=config.score_column,
        fallback=config.fallback_score_column,
    )

    uncapped_config = replace(config, max_candidates_per_frame=0)
    base = build_candidate_reservoir(rows, config=uncapped_config)
    selected_by_id = {
        int(row_id): record
        for row_id, record in zip(
            pd.to_numeric(base["_source_branch_reservoir_row_id"], errors="raise").astype(int),
            base.to_dict(orient="records"),
            strict=True,
        )
    }

    if int(per_source_branch_top_n) > 0:
        group_columns = ["sequence_id", "time_s", "source", "candidate_branch"]
        for group_key, group in rows.groupby(group_columns, sort=False, dropna=False):
            selected = _select_source_branch_rows(
                group,
                count=int(per_source_branch_top_n),
                diversity_weight=float(source_branch_diversity_weight),
                diversity_scale_m=float(source_branch_diversity_scale_m),
                distance_cap_m=float(source_branch_distance_cap_m),
            )
            source = str(group_key[2])
            branch = str(group_key[3])
            reason = f"{_DEFAULT_SOURCE_BRANCH_REASON_PREFIX}{source}|{branch}"
            for _, candidate in selected.iterrows():
                row_id = int(candidate["_source_branch_reservoir_row_id"])
                existing = selected_by_id.get(row_id)
                diagnostics = {
                    "candidate_source_branch_selection_rank": int(
                        candidate["_source_branch_selection_rank"]
                    ),
                    "candidate_source_branch_min_distance_m": float(
                        candidate["_source_branch_min_distance_m"]
                    ),
                    "candidate_source_branch_diversity_term": float(
                        candidate["_source_branch_diversity_term"]
                    ),
                    "candidate_source_branch_selection_utility": float(
                        candidate["_source_branch_selection_utility"]
                    ),
                }
                if existing is None:
                    existing = candidate.to_dict()
                    existing["candidate_reservoir_score"] = float(
                        candidate["_source_branch_base_score"]
                    )
                    existing["candidate_reservoir_reason"] = reason
                    existing["candidate_reservoir_reasons"] = reason
                    existing.update(diagnostics)
                    selected_by_id[row_id] = existing
                else:
                    merged_reason = _merge_reason_tokens(
                        existing.get("candidate_reservoir_reason", ""),
                        reason,
                    )
                    existing["candidate_reservoir_reason"] = merged_reason
                    existing["candidate_reservoir_reasons"] = merged_reason
                    existing.update(diagnostics)

    union = pd.DataFrame.from_records(list(selected_by_id.values()))
    if union.empty:
        return CandidateFrame(normalize_candidate_columns(union))
    preserve_prefixes = tuple(config.preserve_reason_prefixes)
    if _DEFAULT_SOURCE_BRANCH_REASON_PREFIX not in preserve_prefixes:
        preserve_prefixes = (*preserve_prefixes, _DEFAULT_SOURCE_BRANCH_REASON_PREFIX)
    capped = _apply_frame_cap(
        union,
        max_candidates_per_frame=config.max_candidates_per_frame,
        cap_reason_bonus=float(config.cap_reason_bonus),
        preserve_reason_prefixes=preserve_prefixes,
    )
    quota_selected = (
        capped["candidate_reservoir_reason"]
        .fillna("")
        .astype(str)
        .str.contains(_DEFAULT_SOURCE_BRANCH_REASON_PREFIX, regex=False)
    )
    capped["candidate_source_branch_quota_top_n"] = int(per_source_branch_top_n)
    capped["candidate_source_branch_quota_selected"] = quota_selected
    capped["candidate_source_branch_diversity_weight"] = float(
        source_branch_diversity_weight
    )
    capped["candidate_source_branch_diversity_selected"] = quota_selected & (
        float(source_branch_diversity_weight) > 0.0
    )
    capped = capped.sort_values(
        ["sequence_id", "time_s", "candidate_reservoir_rank", "source", "candidate_branch"],
    ).reset_index(drop=True)
    capped = capped.drop(
        columns=[
            "_source_branch_reservoir_row_id",
            "_source_branch_base_score",
            "_source_branch_selection_rank",
            "_source_branch_min_distance_m",
            "_source_branch_diversity_term",
            "_source_branch_selection_utility",
        ],
        errors="ignore",
    )
    return CandidateFrame(normalize_candidate_columns(capped))


def source_branch_reservoir_summary(
    candidates: CandidateFrame | pd.DataFrame,
    reservoir: CandidateFrame | pd.DataFrame,
) -> dict[str, Any]:
    """Return reservoir diagnostics including retained source/branch cells."""

    input_rows = _candidate_rows(candidates)
    reservoir_rows = _candidate_rows(reservoir)
    summary = build_reservoir_summary(input_rows, reservoir_rows)
    input_cells = _frame_cell_count(input_rows)
    retained_cells = _frame_cell_count(reservoir_rows)
    quota_selected = reservoir_rows.get(
        "candidate_source_branch_quota_selected",
        pd.Series(False, index=reservoir_rows.index, dtype=bool),
    )
    diversity_selected = reservoir_rows.get(
        "candidate_source_branch_diversity_selected",
        pd.Series(False, index=reservoir_rows.index, dtype=bool),
    )
    selected_distances = pd.to_numeric(
        reservoir_rows.get(
            "candidate_source_branch_min_distance_m",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    ).dropna()
    summary.update(
        {
            "input_source_branch_cells": int(input_cells),
            "retained_source_branch_cells": int(retained_cells),
            "source_branch_cell_recall": (
                float(retained_cells / input_cells) if input_cells > 0 else 1.0
            ),
            "source_branch_quota_selected_rows": int(
                pd.Series(quota_selected).fillna(False).astype(bool).sum()
            ),
            "source_branch_diversity_selected_rows": int(
                pd.Series(diversity_selected).fillna(False).astype(bool).sum()
            ),
            "source_branch_selected_min_distance_mean_m": _safe_mean(selected_distances),
            "source_branch_selected_min_distance_p50_m": _safe_quantile(
                selected_distances,
                0.50,
            ),
            "source_branch_selected_min_distance_p95_m": _safe_quantile(
                selected_distances,
                0.95,
            ),
        }
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-source-branch-reservoir",
        description="build MMUAD candidate reservoirs with source-by-branch quotas",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-reservoir-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--score-column", default="candidate_risk_adjusted_score")
    parser.add_argument("--fallback-score-column", default="ranker_score")
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--per-source-branch-top-n", type=int, default=1)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--cap-reason-bonus", type=float, default=0.0)
    parser.add_argument(
        "--source-branch-diversity-weight",
        type=float,
        default=0.0,
        help="within-cell score/diversity trade-off; zero preserves score-only quotas",
    )
    parser.add_argument("--source-branch-diversity-scale-m", type=float, default=10.0)
    parser.add_argument("--source-branch-distance-cap-m", type=float, default=50.0)
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
    reservoir = build_source_branch_reservoir(
        candidates,
        reservoir_config=config,
        per_source_branch_top_n=args.per_source_branch_top_n,
        source_branch_diversity_weight=args.source_branch_diversity_weight,
        source_branch_diversity_scale_m=args.source_branch_diversity_scale_m,
        source_branch_distance_cap_m=args.source_branch_distance_cap_m,
    )
    args.output_reservoir_csv.parent.mkdir(parents=True, exist_ok=True)
    reservoir.rows.to_csv(args.output_reservoir_csv, index=False)

    summary = source_branch_reservoir_summary(candidates, reservoir)
    summary.update(
        {
            "score_column": args.score_column,
            "fallback_score_column": args.fallback_score_column,
            "global_top_n": int(args.global_top_n),
            "per_source_top_n": int(args.per_source_top_n),
            "per_branch_top_n": int(args.per_branch_top_n),
            "per_source_branch_top_n": int(args.per_source_branch_top_n),
            "max_candidates_per_frame": int(args.max_candidates_per_frame),
            "source_branch_diversity_weight": float(
                args.source_branch_diversity_weight
            ),
            "source_branch_diversity_scale_m": float(
                args.source_branch_diversity_scale_m
            ),
            "source_branch_distance_cap_m": float(args.source_branch_distance_cap_m),
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

    print("mmuad_source_branch_reservoir=ok")
    print(f"candidate_rows={len(_candidate_rows(candidates))}")
    print(f"reservoir_rows={len(reservoir.rows)}")
    print(f"output_reservoir_csv={args.output_reservoir_csv}")
    return 0


def _select_source_branch_rows(
    group: pd.DataFrame,
    *,
    count: int,
    diversity_weight: float,
    diversity_scale_m: float,
    distance_cap_m: float,
) -> pd.DataFrame:
    """Greedily retain high-score, spatially distinct candidates within one cell."""

    if count <= 0 or group.empty:
        return group.iloc[0:0].copy()
    work = group.copy()
    work["_source_branch_score_norm"] = _normalize_score(
        work["_source_branch_base_score"]
    )
    remaining_ids = set(work["_source_branch_reservoir_row_id"].astype(int).tolist())
    selected_ids: list[int] = []
    records: list[dict[str, float | int]] = []
    budget = min(int(count), len(work))

    while len(selected_ids) < budget and remaining_ids:
        remaining = work.loc[
            work["_source_branch_reservoir_row_id"].astype(int).isin(remaining_ids)
        ].copy()
        if not selected_ids:
            remaining["_source_branch_min_distance_m"] = np.nan
            remaining["_source_branch_diversity_term"] = 0.0
        else:
            selected_xyz = work.loc[
                work["_source_branch_reservoir_row_id"].astype(int).isin(selected_ids),
                ["x_m", "y_m", "z_m"],
            ].to_numpy(float)
            candidate_xyz = remaining[["x_m", "y_m", "z_m"]].to_numpy(float)
            distances = np.linalg.norm(
                candidate_xyz[:, None, :] - selected_xyz[None, :, :],
                axis=2,
            )
            min_distance = np.min(distances, axis=1)
            bounded_distance = np.minimum(min_distance, float(distance_cap_m))
            remaining["_source_branch_min_distance_m"] = min_distance
            remaining["_source_branch_diversity_term"] = 1.0 - np.exp(
                -bounded_distance / float(diversity_scale_m)
            )
        remaining["_source_branch_selection_utility"] = (
            remaining["_source_branch_score_norm"]
            + float(diversity_weight) * remaining["_source_branch_diversity_term"]
        )
        chosen = remaining.sort_values(
            [
                "_source_branch_selection_utility",
                "_source_branch_base_score",
                "_source_branch_reservoir_row_id",
            ],
            ascending=[False, False, True],
        ).iloc[0]
        row_id = int(chosen["_source_branch_reservoir_row_id"])
        selected_ids.append(row_id)
        remaining_ids.remove(row_id)
        records.append(
            {
                "_source_branch_reservoir_row_id": row_id,
                "_source_branch_selection_rank": len(selected_ids),
                "_source_branch_min_distance_m": float(
                    chosen["_source_branch_min_distance_m"]
                ),
                "_source_branch_diversity_term": float(
                    chosen["_source_branch_diversity_term"]
                ),
                "_source_branch_selection_utility": float(
                    chosen["_source_branch_selection_utility"]
                ),
            }
        )

    diagnostics = pd.DataFrame.from_records(records)
    selected = work.loc[
        work["_source_branch_reservoir_row_id"].astype(int).isin(selected_ids)
    ].merge(diagnostics, on="_source_branch_reservoir_row_id", how="inner")
    return selected.sort_values("_source_branch_selection_rank").reset_index(drop=True)


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    if isinstance(candidates, CandidateFrame):
        return normalize_candidate_columns(candidates.rows.copy())
    return normalize_candidate_columns(pd.DataFrame(candidates).copy())


def _normalize_source_branch_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "source" not in out.columns:
        out["source"] = "unknown"
    out["source"] = _text_column(out["source"], default="unknown")
    if "candidate_branch" not in out.columns:
        out["candidate_branch"] = out["source"]
    out["candidate_branch"] = _text_column(out["candidate_branch"], default="candidate")
    return out


def _text_column(values: pd.Series, *, default: str) -> pd.Series:
    text = values.where(values.notna(), default).astype(str).str.strip()
    missing = text.eq("") | text.str.lower().isin({"nan", "none", "<na>"})
    return text.where(~missing, default)


def _resolve_score(rows: pd.DataFrame, *, primary: str, fallback: str) -> pd.Series:
    primary_values = _numeric_column(rows, primary, default=np.nan)
    fallback_values = _numeric_column(rows, fallback, default=0.0)
    resolved = primary_values.where(np.isfinite(primary_values), fallback_values)
    return resolved.fillna(0.0).astype(float)


def _numeric_column(rows: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(default, index=rows.index, dtype=float)
    values = pd.to_numeric(rows[column], errors="coerce")
    return values.where(np.isfinite(values), np.nan)


def _normalize_score(values: pd.Series) -> pd.Series:
    score = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    minimum = float(score.min())
    maximum = float(score.max())
    if maximum <= minimum:
        return pd.Series(0.5, index=score.index, dtype=float)
    return (score - minimum) / (maximum - minimum)


def _merge_reason_tokens(existing: Any, new_reason: str) -> str:
    tokens = {
        token.strip()
        for token in str(existing).replace(",", ";").split(";")
        if token.strip()
    }
    tokens.add(str(new_reason))
    return ";".join(sorted(tokens))


def _frame_cell_count(rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    normalized = _normalize_source_branch_columns(rows)
    columns = ["sequence_id", "time_s", "source", "candidate_branch"]
    return int(len(normalized[columns].drop_duplicates()))


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else 0.0


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    return float(values.quantile(quantile)) if not values.empty else 0.0


def _validate_selection_config(
    *,
    per_source_branch_top_n: int,
    diversity_weight: float,
    diversity_scale_m: float,
    distance_cap_m: float,
) -> None:
    if int(per_source_branch_top_n) < 0:
        raise ValueError("per_source_branch_top_n must be non-negative")
    if float(diversity_weight) < 0.0:
        raise ValueError("source_branch_diversity_weight must be non-negative")
    if float(diversity_scale_m) <= 0.0:
        raise ValueError("source_branch_diversity_scale_m must be positive")
    if float(distance_cap_m) <= 0.0:
        raise ValueError("source_branch_distance_cap_m must be positive")


def _write_optional_csv(rows: pd.DataFrame, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
