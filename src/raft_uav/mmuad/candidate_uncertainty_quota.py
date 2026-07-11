"""Preserve low-uncertainty MMUAD candidates inside source/branch reservoirs.

The ordinary source/branch quota is score driven.  That can discard a candidate
that the learned uncertainty model considers reliable when its ranker score is
only moderate.  This module augments the existing reservoir with an optional
per-cell quota for the lowest predicted-sigma rows before the final frame cap.

The selection is inference safe: it uses candidate metadata and predicted
uncertainty only.  Ground-truth columns are never consulted.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import ReservoirConfig, _apply_frame_cap
from raft_uav.mmuad.candidate_source_branch_reservoir import build_source_branch_reservoir
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns

_DEFAULT_SIGMA_COLUMNS = (
    "predicted_sigma_m",
    "predicted_sigma_m_hgb",
    "candidate_sigma_m",
    "sigma_m",
)
_REASON_PREFIX = "source_branch_uncertainty:"


def build_uncertainty_quota_reservoir(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    reservoir_config: ReservoirConfig | None = None,
    per_source_branch_top_n: int = 1,
    uncertainty_top_n: int = 1,
    sigma_columns: Sequence[str] = _DEFAULT_SIGMA_COLUMNS,
    source_branch_diversity_weight: float = 0.0,
    source_branch_diversity_scale_m: float = 10.0,
    source_branch_distance_cap_m: float = 50.0,
) -> CandidateFrame:
    """Return a source/branch reservoir augmented with low-sigma candidates."""

    if uncertainty_top_n < 0:
        raise ValueError("uncertainty_top_n must be non-negative")
    config = reservoir_config or ReservoirConfig()
    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(normalize_candidate_columns(rows))
    rows = _normalize_columns(rows)
    rows["_uncertainty_quota_row_id"] = np.arange(len(rows), dtype=int)
    rows["candidate_uncertainty_quota_sigma_m"] = _first_finite_sigma(rows, sigma_columns)

    uncapped_config = replace(config, max_candidates_per_frame=None)
    base = build_source_branch_reservoir(
        rows,
        reservoir_config=uncapped_config,
        per_source_branch_top_n=per_source_branch_top_n,
        source_branch_diversity_weight=source_branch_diversity_weight,
        source_branch_diversity_scale_m=source_branch_diversity_scale_m,
        source_branch_distance_cap_m=source_branch_distance_cap_m,
    ).rows.copy()

    selected = {
        int(row["_uncertainty_quota_row_id"]): row.to_dict()
        for _, row in base.iterrows()
    }
    if uncertainty_top_n > 0:
        group_cols = ["sequence_id", "time_s", "source", "candidate_branch"]
        for key, group in rows.groupby(group_cols, sort=False, dropna=False):
            finite = group.loc[group["candidate_uncertainty_quota_sigma_m"].notna()].copy()
            if finite.empty:
                continue
            kept = finite.sort_values(
                ["candidate_uncertainty_quota_sigma_m", "_uncertainty_quota_row_id"],
                ascending=[True, True],
            ).head(int(uncertainty_top_n))
            reason = f"{_REASON_PREFIX}{key[2]}|{key[3]}"
            for _, candidate in kept.iterrows():
                row_id = int(candidate["_uncertainty_quota_row_id"])
                existing = selected.get(row_id, candidate.to_dict())
                existing["candidate_reservoir_reason"] = _merge_reason(
                    existing.get("candidate_reservoir_reason", ""), reason
                )
                existing["candidate_reservoir_reasons"] = existing[
                    "candidate_reservoir_reason"
                ]
                existing["candidate_uncertainty_quota_selected"] = True
                selected[row_id] = existing

    union = pd.DataFrame.from_records(list(selected.values()))
    union = union.drop(
        columns=["candidate_reservoir_reason_count", "candidate_reservoir_cap_score"],
        errors="ignore",
    )
    capped = _apply_frame_cap(
        union,
        max_candidates_per_frame=config.max_candidates_per_frame,
        cap_reason_bonus=float(config.cap_reason_bonus),
        preserve_reason_prefixes=(_REASON_PREFIX,),
    )
    capped["candidate_uncertainty_quota_top_n"] = int(uncertainty_top_n)
    capped["candidate_uncertainty_quota_selected"] = capped.get(
        "candidate_uncertainty_quota_selected", False
    )
    capped = capped.drop(columns=["_uncertainty_quota_row_id"], errors="ignore")
    return CandidateFrame(normalize_candidate_columns(capped))


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    if isinstance(candidates, CandidateFrame):
        return candidates.rows.copy()
    return pd.DataFrame(candidates).copy()


def _normalize_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    if "source" not in out.columns:
        out["source"] = "unknown"
    if "candidate_branch" not in out.columns:
        out["candidate_branch"] = "default"
    return out


def _first_finite_sigma(rows: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    sigma = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in columns:
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        values = values.where(np.isfinite(values) & (values > 0.0), np.nan)
        sigma = sigma.where(sigma.notna(), values)
    return sigma


def _merge_reason(existing: object, new_reason: str) -> str:
    tokens = {token.strip() for token in str(existing).split(",") if token.strip()}
    tokens.add(new_reason)
    return ",".join(sorted(tokens))
