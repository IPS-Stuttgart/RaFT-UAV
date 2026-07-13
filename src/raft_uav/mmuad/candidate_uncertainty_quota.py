"""Preserve low-uncertainty MMUAD candidates inside source/branch reservoirs.

The ordinary source/branch quota is score driven. That can discard a candidate
that the learned uncertainty model considers reliable when its ranker score is
only moderate. This module augments the existing reservoir with an optional
per-cell quota for the lowest predicted-sigma rows before the final frame cap.

An optional novelty radius makes the quota additive: low-sigma rows that are
already represented by the score-driven reservoir, or by an earlier quota pick,
do not consume the limited uncertainty budget.

The selection is inference safe: it uses candidate metadata and predicted
uncertainty only. Ground-truth columns are never consulted.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import ReservoirConfig, _cap_per_frame
from raft_uav.mmuad.candidate_source_branch_reservoir import (
    _DEFAULT_SOURCE_BRANCH_REASON_PREFIX,
    build_source_branch_reservoir,
)
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns

_DEFAULT_SIGMA_COLUMNS = (
    "predicted_sigma_m",
    "predicted_sigma_m_hgb",
    "candidate_sigma_m",
    "sigma_m",
)
_REASON_PREFIX = "source_branch_uncertainty:"
_GROUP_COLUMNS = ("sequence_id", "time_s", "source", "candidate_branch")
_XYZ_COLUMNS = ("x_m", "y_m", "z_m")


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
    uncertainty_novelty_radius_m: float = 0.0,
) -> CandidateFrame:
    """Return a source/branch reservoir augmented with low-sigma candidates.

    With ``uncertainty_novelty_radius_m=0`` the historical per-cell lowest-sigma
    quota is used exactly. A positive radius instead adds up to
    ``uncertainty_top_n`` candidates per cell that are spatially distinct from
    the score/source-branch reservoir and from earlier uncertainty-quota picks.
    This prevents already represented hypotheses from consuming the quota.
    """

    if uncertainty_top_n < 0:
        raise ValueError("uncertainty_top_n must be non-negative")
    novelty_radius = _nonnegative_finite(
        uncertainty_novelty_radius_m,
        name="uncertainty_novelty_radius_m",
    )
    config = reservoir_config or ReservoirConfig()
    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(normalize_candidate_columns(rows))
    rows = _normalize_columns(rows)
    rows["_uncertainty_quota_row_id"] = np.arange(len(rows), dtype=int)
    rows["candidate_uncertainty_quota_sigma_m"] = _first_finite_sigma(
        rows,
        sigma_columns,
    )

    # The source/branch builder uses zero, not None, to mean uncapped. Keeping
    # that convention also avoids int(None) when it applies its final cap.
    uncapped_config = replace(config, max_candidates_per_frame=0)
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
        if novelty_radius > 0.0:
            _add_novel_uncertainty_candidates(
                rows,
                base=base,
                selected=selected,
                uncertainty_top_n=int(uncertainty_top_n),
                novelty_radius_m=novelty_radius,
            )
        else:
            _add_lowest_sigma_candidates(
                rows,
                selected=selected,
                uncertainty_top_n=int(uncertainty_top_n),
            )

    union = pd.DataFrame.from_records(list(selected.values()))
    union = union.drop(
        columns=["candidate_reservoir_reason_count", "candidate_reservoir_cap_score"],
        errors="ignore",
    )
    preserve_prefixes = tuple(config.preserve_reason_prefixes)
    for prefix in (_DEFAULT_SOURCE_BRANCH_REASON_PREFIX, _REASON_PREFIX):
        if prefix not in preserve_prefixes:
            preserve_prefixes = (*preserve_prefixes, prefix)
    capped = _cap_per_frame(
        union,
        max_candidates_per_frame=int(config.max_candidates_per_frame),
        cap_reason_bonus=float(config.cap_reason_bonus),
        preserve_reason_prefixes=preserve_prefixes,
    )
    capped["candidate_uncertainty_quota_top_n"] = int(uncertainty_top_n)
    capped["candidate_uncertainty_quota_novelty_radius_m"] = novelty_radius
    capped["candidate_uncertainty_quota_selected"] = capped.get(
        "candidate_uncertainty_quota_selected",
        False,
    )
    if "candidate_uncertainty_quota_novelty_distance_m" not in capped.columns:
        capped["candidate_uncertainty_quota_novelty_distance_m"] = np.nan
    capped = capped.drop(columns=["_uncertainty_quota_row_id"], errors="ignore")
    return CandidateFrame(normalize_candidate_columns(capped))


def _add_lowest_sigma_candidates(
    rows: pd.DataFrame,
    *,
    selected: dict[int, dict[str, object]],
    uncertainty_top_n: int,
) -> None:
    for key, group in rows.groupby(list(_GROUP_COLUMNS), sort=False, dropna=False):
        finite = group.loc[group["candidate_uncertainty_quota_sigma_m"].notna()].copy()
        if finite.empty:
            continue
        kept = finite.sort_values(
            ["candidate_uncertainty_quota_sigma_m", "_uncertainty_quota_row_id"],
            ascending=[True, True],
        ).head(uncertainty_top_n)
        reason = _uncertainty_reason(key)
        for _, candidate in kept.iterrows():
            _mark_uncertainty_selected(
                candidate,
                selected=selected,
                reason=reason,
                novelty_distance_m=np.nan,
            )


def _add_novel_uncertainty_candidates(
    rows: pd.DataFrame,
    *,
    base: pd.DataFrame,
    selected: dict[int, dict[str, object]],
    uncertainty_top_n: int,
    novelty_radius_m: float,
) -> None:
    base_xyz = _group_xyz(base)
    for key, group in rows.groupby(list(_GROUP_COLUMNS), sort=False, dropna=False):
        finite = group.loc[group["candidate_uncertainty_quota_sigma_m"].notna()].copy()
        if finite.empty:
            continue
        ordered = finite.sort_values(
            ["candidate_uncertainty_quota_sigma_m", "_uncertainty_quota_row_id"],
            ascending=[True, True],
        )
        references = [xyz.copy() for xyz in base_xyz.get(key, [])]
        reason = _uncertainty_reason(key)
        added = 0
        for _, candidate in ordered.iterrows():
            row_id = int(candidate["_uncertainty_quota_row_id"])
            if row_id in selected:
                continue
            xyz = candidate[list(_XYZ_COLUMNS)].to_numpy(dtype=float)
            if not np.isfinite(xyz).all():
                continue
            distance = _minimum_distance(xyz, references)
            if distance is not None and distance < novelty_radius_m:
                continue
            _mark_uncertainty_selected(
                candidate,
                selected=selected,
                reason=reason,
                novelty_distance_m=np.nan if distance is None else distance,
            )
            references.append(xyz)
            added += 1
            if added >= uncertainty_top_n:
                break


def _mark_uncertainty_selected(
    candidate: pd.Series,
    *,
    selected: dict[int, dict[str, object]],
    reason: str,
    novelty_distance_m: float,
) -> None:
    row_id = int(candidate["_uncertainty_quota_row_id"])
    existing = selected.get(row_id, candidate.to_dict())
    existing["candidate_reservoir_reason"] = _merge_reason(
        existing.get("candidate_reservoir_reason", ""),
        reason,
    )
    existing["candidate_reservoir_reasons"] = existing["candidate_reservoir_reason"]
    existing["candidate_uncertainty_quota_selected"] = True
    existing["candidate_uncertainty_quota_novelty_distance_m"] = novelty_distance_m
    selected[row_id] = existing


def _group_xyz(rows: pd.DataFrame) -> dict[tuple[object, ...], list[np.ndarray]]:
    if rows.empty:
        return {}
    result: dict[tuple[object, ...], list[np.ndarray]] = {}
    for key, group in rows.groupby(list(_GROUP_COLUMNS), sort=False, dropna=False):
        values = group[list(_XYZ_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        result[key] = [xyz for xyz in values if np.isfinite(xyz).all()]
    return result


def _minimum_distance(
    xyz: np.ndarray,
    references: list[np.ndarray],
) -> float | None:
    if not references:
        return None
    return min(float(np.linalg.norm(xyz - reference)) for reference in references)


def _uncertainty_reason(key: tuple[object, ...]) -> str:
    return f"{_REASON_PREFIX}{key[2]}|{key[3]}"


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


def _nonnegative_finite(value: float, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _merge_reason(existing: object, new_reason: str) -> str:
    tokens = {
        token.strip()
        for token in str(existing).replace(";", ",").split(",")
        if token.strip()
    }
    tokens.add(new_reason)
    return ";".join(sorted(tokens))
