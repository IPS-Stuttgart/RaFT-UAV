"""Uncertainty-aware final caps for MMUAD candidate reservoirs.

Branch/source quotas keep provenance diversity, while learned candidate uncertainty
provides a complementary signal: a candidate can have a modest ranker score but a
low predicted position error. This module reserves a small low-uncertainty quota
and optionally uses uncertainty when filling the remaining per-frame budget.

The operation is truth-free at inference time. It expects uncertainty values such
as ``predicted_sigma_m_hgb`` that were learned from the training split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import build_oracle_recall_tables
from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns

_DEFAULT_UNCERTAINTY_COLUMNS = (
    "predicted_sigma_m_hgb",
    "predicted_sigma_m",
    "candidate_sigma_m",
    "sigma_m",
)


def uncertainty_aware_cap_reservoir(
    reservoir: pd.DataFrame,
    *,
    max_candidates_per_frame: int = 40,
    min_per_source: int = 1,
    min_per_branch: int = 1,
    min_low_uncertainty: int = 3,
    uncertainty_columns: Sequence[str] = _DEFAULT_UNCERTAINTY_COLUMNS,
    uncertainty_weight: float = 0.25,
    score_column: str = "candidate_reservoir_score",
    fallback_score_column: str = "confidence",
    branch_column: str = "candidate_branch",
) -> pd.DataFrame:
    """Return a capped reservoir preserving diversity and low-uncertainty rows.

    Selection proceeds per timestamp:

    1. protect the best score rows per source and per candidate branch;
    2. protect the globally lowest predicted-uncertainty rows;
    3. fill remaining slots with a normalized score minus an uncertainty penalty.

    Missing uncertainty values are treated as least reliable for the optional
    utility term, but they remain eligible through score/source/branch selection.
    Defaults are opt-in through this dedicated utility and do not change the
    existing candidate-reservoir behavior.
    """

    rows = normalize_candidate_columns(pd.DataFrame(reservoir).copy())
    if rows.empty:
        return rows.assign(
            candidate_uncertainty_cap_rank=pd.Series(dtype=float),
            candidate_uncertainty_cap_reason=pd.Series(dtype=str),
            candidate_uncertainty_value_m=pd.Series(dtype=float),
            candidate_uncertainty_score_norm=pd.Series(dtype=float),
            candidate_uncertainty_norm=pd.Series(dtype=float),
            candidate_uncertainty_selection_utility=pd.Series(dtype=float),
            candidate_uncertainty_column=pd.Series(dtype=str),
        )

    rows = rows.copy().reset_index(drop=True)
    _ensure_columns(rows, branch_column=branch_column)
    rows["_uncertainty_row_id"] = np.arange(len(rows), dtype=int)
    rows["_uncertainty_score"] = _score(rows, score_column, fallback_score_column)
    uncertainty, used_column = _uncertainty(rows, uncertainty_columns)
    rows["_uncertainty_value"] = uncertainty
    rows["candidate_uncertainty_column"] = used_column

    parts: list[pd.DataFrame] = []
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=False, dropna=False):
        parts.append(
            _cap_frame(
                frame.copy(),
                max_candidates_per_frame=int(max_candidates_per_frame),
                min_per_source=int(min_per_source),
                min_per_branch=int(min_per_branch),
                min_low_uncertainty=int(min_low_uncertainty),
                uncertainty_weight=float(uncertainty_weight),
                branch_column=branch_column,
            )
        )

    out = pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0].copy()
    out = out.drop(
        columns=(
            "_uncertainty_row_id",
            "_uncertainty_score",
            "_uncertainty_value",
            "_uncertainty_score_norm",
            "_uncertainty_norm",
            "_uncertainty_utility",
        ),
        errors="ignore",
    )
    return out.sort_values(
        ["sequence_id", "time_s", "candidate_uncertainty_cap_rank"],
    ).reset_index(drop=True)


def uncertainty_cap_summary(input_rows: pd.DataFrame, output_rows: pd.DataFrame) -> dict[str, Any]:
    """Build a compact summary of an uncertainty-aware cap."""

    output_uncertainty = pd.to_numeric(
        output_rows.get("candidate_uncertainty_value_m", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    return {
        "input_rows": int(len(input_rows)),
        "output_rows": int(len(output_rows)),
        "input_frame_count": int(_frame_counts(input_rows).size),
        "output_frame_count": int(_frame_counts(output_rows).size),
        "input_candidates_per_frame_mean": _mean(_frame_counts(input_rows)),
        "output_candidates_per_frame_mean": _mean(_frame_counts(output_rows)),
        "output_candidates_per_frame_p95": _quantile(_frame_counts(output_rows), 0.95),
        "output_candidates_per_frame_max": _max(_frame_counts(output_rows)),
        "selected_uncertainty_mean_m": _mean(output_uncertainty),
        "selected_uncertainty_p50_m": _quantile(output_uncertainty, 0.50),
        "selected_uncertainty_p95_m": _quantile(output_uncertainty, 0.95),
        "source_counts": _value_counts(output_rows, "source"),
        "branch_counts": _value_counts(output_rows, "candidate_branch"),
        "uncertainty_cap_reason_counts": _reason_counts(output_rows),
        "uncertainty_column_counts": _value_counts(output_rows, "candidate_uncertainty_column"),
    }


def write_uncertainty_cap_outputs(
    capped: pd.DataFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    input_rows: pd.DataFrame | None = None,
) -> None:
    """Write capped candidates and an optional JSON summary."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    capped.to_csv(output_csv, index=False)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(
                uncertainty_cap_summary(
                    input_rows if input_rows is not None else capped,
                    capped,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )


def _cap_frame(
    frame: pd.DataFrame,
    *,
    max_candidates_per_frame: int,
    min_per_source: int,
    min_per_branch: int,
    min_low_uncertainty: int,
    uncertainty_weight: float,
    branch_column: str,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["_uncertainty_score_norm"] = _normalize_high_good(frame["_uncertainty_score"])
    frame["_uncertainty_norm"] = _normalize_low_good(frame["_uncertainty_value"])
    frame["_uncertainty_utility"] = (
        frame["_uncertainty_score_norm"]
        - max(float(uncertainty_weight), 0.0) * frame["_uncertainty_norm"]
    )

    protected: set[int] = set()
    reasons: dict[int, set[str]] = {}
    _protect_group_topn(
        frame,
        group_column="source",
        count=min_per_source,
        reason_prefix="source",
        protected=protected,
        reasons=reasons,
    )
    _protect_group_topn(
        frame,
        group_column=branch_column,
        count=min_per_branch,
        reason_prefix="branch",
        protected=protected,
        reasons=reasons,
    )
    low_uncertainty_ids = _lowest_uncertainty_ids(frame, min_low_uncertainty)
    for row_id in low_uncertainty_ids:
        protected.add(row_id)
        reasons.setdefault(row_id, set()).add("low_uncertainty")

    budget = len(frame) if max_candidates_per_frame <= 0 else min(
        int(max_candidates_per_frame),
        len(frame),
    )
    if budget <= 0:
        return frame.iloc[0:0].copy()

    protected_frame = frame.loc[frame["_uncertainty_row_id"].isin(protected)].copy()
    if len(protected_frame) >= budget:
        protected_frame["_is_low_uncertainty"] = protected_frame["_uncertainty_row_id"].isin(
            low_uncertainty_ids,
        )
        selected = protected_frame.sort_values(
            ["_is_low_uncertainty", "_uncertainty_utility", "_uncertainty_score"],
            ascending=[False, False, False],
        ).head(budget)
        selected = selected.drop(columns="_is_low_uncertainty")
        for row_id in selected["_uncertainty_row_id"].astype(int):
            reasons.setdefault(row_id, set()).add("protected_cap")
    else:
        selected_ids = set(protected_frame["_uncertainty_row_id"].astype(int))
        fill = frame.loc[~frame["_uncertainty_row_id"].isin(selected_ids)].sort_values(
            ["_uncertainty_utility", "_uncertainty_score"],
            ascending=[False, False],
        ).head(max(budget - len(selected_ids), 0))
        for row_id in fill["_uncertainty_row_id"].astype(int):
            reasons.setdefault(row_id, set()).add("uncertainty_score_fill")
        selected = pd.concat([protected_frame, fill], ignore_index=False)

    selected = selected.sort_values(
        ["_uncertainty_utility", "_uncertainty_score"],
        ascending=[False, False],
    ).copy()
    selected["candidate_uncertainty_cap_rank"] = np.arange(1, len(selected) + 1, dtype=float)
    selected["candidate_uncertainty_cap_reason"] = [
        ";".join(sorted(reasons.get(int(row_id), {"uncertainty_score_fill"})))
        for row_id in selected["_uncertainty_row_id"].astype(int)
    ]
    selected["candidate_uncertainty_value_m"] = selected["_uncertainty_value"]
    selected["candidate_uncertainty_score_norm"] = selected["_uncertainty_score_norm"]
    selected["candidate_uncertainty_norm"] = selected["_uncertainty_norm"]
    selected["candidate_uncertainty_selection_utility"] = selected["_uncertainty_utility"]
    return selected


def _ensure_columns(rows: pd.DataFrame, *, branch_column: str) -> None:
    if "source" not in rows.columns:
        rows["source"] = "candidate"
    rows["source"] = rows["source"].fillna("candidate").astype(str)
    if branch_column not in rows.columns:
        rows[branch_column] = rows.get("candidate_branch", rows["source"])
    rows[branch_column] = rows[branch_column].fillna("candidate").astype(str)
    if "candidate_branch" not in rows.columns:
        rows["candidate_branch"] = rows[branch_column]


def _protect_group_topn(
    frame: pd.DataFrame,
    *,
    group_column: str,
    count: int,
    reason_prefix: str,
    protected: set[int],
    reasons: dict[int, set[str]],
) -> None:
    if count <= 0 or group_column not in frame.columns:
        return
    for value, group in frame.groupby(group_column, sort=False, dropna=False):
        selected = group.sort_values(
            ["_uncertainty_score", "_uncertainty_row_id"],
            ascending=[False, True],
        ).head(int(count))
        for row_id in selected["_uncertainty_row_id"].astype(int):
            protected.add(row_id)
            reasons.setdefault(row_id, set()).add(f"{reason_prefix}:{value}")


def _lowest_uncertainty_ids(frame: pd.DataFrame, count: int) -> set[int]:
    if count <= 0 or frame.empty:
        return set()
    finite = frame.loc[np.isfinite(frame["_uncertainty_value"].to_numpy(float))].copy()
    if finite.empty:
        return set()
    selected = finite.sort_values(
        ["_uncertainty_value", "_uncertainty_score", "_uncertainty_row_id"],
        ascending=[True, False, True],
    ).head(int(count))
    return set(selected