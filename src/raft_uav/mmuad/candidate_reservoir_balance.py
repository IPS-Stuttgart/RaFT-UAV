"""Apply a diversity-preserving hard cap to MMUAD candidate reservoirs.

A branch-preserving reservoir can still lose its low-scored raw, dynamic, or
calibrated candidates when a later global top-N cap is applied.  This module
keeps configurable per-branch and per-source representatives before filling
remaining slots by score.  The output is intended for learned-sigma / robust
mixture-MAP experiments that need bounded frame sizes without silently
collapsing candidate diversity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import normalize_candidate_columns

_REQUIRED_COLUMNS = ("sequence_id", "time_s", "x_m", "y_m", "z_m")


@dataclass(frozen=True)
class ReservoirBalanceConfig:
    """Configuration for a diversity-preserving candidate reservoir cap."""

    max_candidates_per_frame: int = 40
    min_per_branch: int = 1
    min_per_source: int = 1
    score_column: str = "candidate_reservoir_score"
    fallback_score_column: str = "ranker_score"
    second_fallback_score_column: str = "confidence"


def balance_candidate_reservoir(
    candidates: pd.DataFrame,
    *,
    config: ReservoirBalanceConfig | None = None,
) -> pd.DataFrame:
    """Return a bounded reservoir while preserving branch/source diversity.

    Branch representatives are protected first, followed by source
    representatives.  Any remaining capacity is filled by global score.  If
    there are more protected representatives than available slots, the
    highest-scored representatives are kept deterministically.
    """

    config = ReservoirBalanceConfig() if config is None else config
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        return rows.assign(
            candidate_reservoir_balance_score=pd.Series(dtype=float),
            candidate_reservoir_balance_reason=pd.Series(dtype=str),
            candidate_reservoir_balance_protected=pd.Series(dtype=bool),
            candidate_reservoir_balanced_rank=pd.Series(dtype=float),
        )
    _validate_required_columns(rows)
    rows = rows.reset_index(drop=True)
    if "source" not in rows.columns:
        rows["source"] = "unknown"
    if "candidate_branch" not in rows.columns:
        rows["candidate_branch"] = rows["source"].fillna("candidate").astype(str)
    rows["source"] = rows["source"].fillna("unknown").astype(str)
    rows["candidate_branch"] = rows["candidate_branch"].fillna("candidate").astype(str)
    rows["_candidate_balance_input_row"] = np.arange(len(rows), dtype=int)
    rows["candidate_reservoir_balance_score"] = _balance_score(rows, config=config)

    parts: list[pd.DataFrame] = []
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=True, dropna=False):
        parts.append(_balance_frame(frame.copy(), config=config))
    balanced = pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0].copy()
    balanced = balanced.sort_values(
        ["sequence_id", "time_s", "candidate_reservoir_balanced_rank", "source"],
        kind="mergesort",
    ).reset_index(drop=True)
    return balanced.drop(columns=["_candidate_balance_input_row"], errors="ignore")


def build_balance_summary(
    candidates: pd.DataFrame,
    balanced: pd.DataFrame,
) -> dict[str, Any]:
    """Build a JSON-serializable diversity/cap summary."""

    before = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    after = normalize_candidate_columns(pd.DataFrame(balanced).copy())
    before_counts = _frame_counts(before)
    after_counts = _frame_counts(after)
    return {
        "input_candidate_rows": int(len(before)),
        "balanced_candidate_rows": int(len(after)),
        "frame_count": int(len(before_counts)),
        "input_candidates_per_frame_mean": _safe_mean(before_counts),
        "balanced_candidates_per_frame_mean": _safe_mean(after_counts),
        "balanced_candidates_per_frame_p95": _safe_quantile(after_counts, 0.95),
        "balanced_candidates_per_frame_max": _safe_max(after_counts),
        "frames_with_branch_coverage_loss": _coverage_loss_frames(
            before,
            after,
            column="candidate_branch",
        ),
        "frames_with_source_coverage_loss": _coverage_loss_frames(
            before,
            after,
            column="source",
        ),
        "input_branch_counts": _value_counts(before, "candidate_branch"),
        "balanced_branch_counts": _value_counts(after, "candidate_branch"),
        "input_source_counts": _value_counts(before, "source"),
        "balanced_source_counts": _value_counts(after, "source"),
        "balance_reason_counts": _reason_counts(after),
    }


def write_balance_outputs(
    balanced: pd.DataFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    input_candidates: pd.DataFrame | None = None,
) -> None:
    """Write balanced reservoir rows and optional summary JSON."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(output_csv, index=False)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = build_balance_summary(
            balanced if input_candidates is None else input_candidates,
            balanced,
        )
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-balance-candidate-reservoir",
        description="apply a diversity-preserving hard cap to an MMUAD candidate reservoir",
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--min-per-branch", type=int, default=1)
    parser.add_argument("--min-per-source", type=int, default=1)
    parser.add_argument("--score-column", default="candidate_reservoir_score")
    parser.add_argument("--fallback-score-column", default="ranker_score")
    parser.add_argument("--second-fallback-score-column", default="confidence")
    args = parser.parse_args(argv)

    candidates = pd.read_csv(args.input_csv)
    balanced = balance_candidate_reservoir(
        candidates,
        config=ReservoirBalanceConfig(
            max_candidates_per_frame=args.max_candidates_per_frame,
            min_per_branch=args.min_per_branch,
            min_per_source=args.min_per_source,
            score_column=args.score_column,
            fallback_score_column=args.fallback_score_column,
            second_fallback_score_column=args.second_fallback_score_column,
        ),
    )
    write_balance_outputs(
        balanced,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        input_candidates=candidates,
    )
    print("mmuad_candidate_reservoir_balance=ok")
    print(f"input_rows={len(candidates)}")
    print(f"balanced_rows={len(balanced)}")
    print(f"output_csv={args.output_csv}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    return 0


def _balance_frame(frame: pd.DataFrame, *, config: ReservoirBalanceConfig) -> pd.DataFrame:
    frame = frame.sort_values(
        [
            "candidate_reservoir_balance_score",
            "candidate_branch",
            "source",
            "_candidate_balance_input_row",
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    reasons: dict[int, set[str]] = {}
    protected_order: list[int] = []
    _append_quota_representatives(
        frame,
        column="candidate_branch",
        count=max(0, int(config.min_per_branch)),
        reason_prefix="branch",
        protected_order=protected_order,
        reasons=reasons,
    )
    _append_quota_representatives(
        frame,
        column="source",
        count=max(0, int(config.min_per_source)),
        reason_prefix="source",
        protected_order=protected_order,
        reasons=reasons,
    )

    cap = int(config.max_candidates_per_frame)
    if cap <= 0:
        selected_ids = list(frame["_candidate_balance_input_row"].astype(int))
    else:
        selected_ids = protected_order[:cap]
        if len(selected_ids) < cap:
            for row_id in frame["_candidate_balance_input_row"].astype(int):
                if row_id in selected_ids:
                    continue
                selected_ids.append(int(row_id))
                reasons.setdefault(int(row_id), set()).add("global_fill")
                if len(selected_ids) >= cap:
                    break
    if not selected_ids:
        return frame.iloc[0:0].copy()

    selected = frame.loc[frame["_candidate_balance_input_row"].isin(selected_ids)].copy()
    order = {row_id: rank for rank, row_id in enumerate(selected_ids, start=1)}
    selected["candidate_reservoir_balanced_rank"] = selected[
        "_candidate_balance_input_row"
    ].map(order).astype(float)
    selected["candidate_reservoir_balance_reason"] = selected[
        "_candidate_balance_input_row"
    ].map(lambda row_id: ";".join(sorted(reasons.get(int(row_id), {"global_fill"}))))
    selected["candidate_reservoir_balance_protected"] = selected[
        "candidate_reservoir_balance_reason"
    ].str.contains(r"(^|;)(branch|source):", regex=True)
    return selected.sort_values("candidate_reservoir_balanced_rank", kind="mergesort")


def _append_quota_representatives(
    frame: pd.DataFrame,
    *,
    column: str,
    count: int,
    reason_prefix: str,
    protected_order: list[int],
    reasons: dict[int, set[str]],
) -> None:
    if count <= 0 or column not in frame.columns:
        return
    ranked_groups = {
        str(value): group.sort_values(
            ["candidate_reservoir_balance_score", "_candidate_balance_input_row"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        for value, group in frame.groupby(column, sort=True, dropna=False)
    }
    for depth in range(count):
        representatives: list[tuple[float, str, int]] = []
        for value, group in ranked_groups.items():
            if depth >= len(group):
                continue
            row = group.iloc[depth]
            representatives.append(
                (
                    float(row["candidate_reservoir_balance_score"]),
                    value,
                    int(row["_candidate_balance_input_row"]),
                )
            )
        representatives.sort(key=lambda item: (-item[0], item[1], item[2]))
        for _, value, row_id in representatives:
            reasons.setdefault(row_id, set()).add(f"{reason_prefix}:{value}")
            if row_id not in protected_order:
                protected_order.append(row_id)


def _balance_score(rows: pd.DataFrame, *, config: ReservoirBalanceConfig) -> pd.Series:
    score = _numeric_column(rows, config.score_column, default=np.nan)
    fallback = _numeric_column(rows, config.fallback_score_column, default=np.nan)
    second_fallback = _numeric_column(
        rows,
        config.second_fallback_score_column,
        default=0.0,
    )
    return score.fillna(fallback).fillna(second_fallback).fillna(0.0).astype(float)


def _numeric_column(rows: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce")
    return pd.Series(default, index=rows.index, dtype=float)


def _validate_required_columns(rows: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"candidate reservoir missing required columns: {missing}")


def _frame_counts(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.groupby(["sequence_id", "time_s"], dropna=False).size().astype(float)


def _coverage_loss_frames(before: pd.DataFrame, after: pd.DataFrame, *, column: str) -> int:
    if before.empty or column not in before.columns:
        return 0
    after_groups = {
        key: set(group[column].fillna("<missing>").astype(str))
        for key, group in after.groupby(["sequence_id", "time_s"], dropna=False)
    }
    losses = 0
    for key, group in before.groupby(["sequence_id", "time_s"], dropna=False):
        before_values = set(group[column].fillna("<missing>").astype(str))
        if not before_values.issubset(after_groups.get(key, set())):
            losses += 1
    return int(losses)


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if rows.empty or column not in rows.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in rows[column].value_counts(dropna=False).items()
    }


def _reason_counts(rows: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if rows.empty or "candidate_reservoir_balance_reason" not in rows.columns:
        return counts
    for value in rows["candidate_reservoir_balance_reason"].fillna("").astype(str):
        for reason in value.split(";"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return counts


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else 0.0


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    return float(values.quantile(quantile)) if not values.empty else 0.0


def _safe_max(values: pd.Series) -> int:
    return int(values.max()) if not values.empty else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
