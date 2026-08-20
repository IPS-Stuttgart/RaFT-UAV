"""Compatibility fixes for canonical branch-consensus identities and flight scope.

The maintained implementation lives in the sibling
``candidate_branch_consensus.py`` module. This package preserves the public
import path while comparing source and branch labels with whitespace-insensitive,
Unicode-aware identities and preventing consensus evidence from crossing
independent physical flights. Caller-visible labels remain unchanged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_branch_consensus.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_branch_consensus_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate branch consensus from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

CandidateFrame = _IMPL.CandidateFrame
DEFAULT_SCORE_OUTPUT_COLUMN = _IMPL.DEFAULT_SCORE_OUTPUT_COLUMN

_SOURCE_IDENTITY_COLUMN = "_branch_consensus_source_identity"
_BRANCH_IDENTITY_COLUMN = "_branch_consensus_branch_identity"
_MISSING_IDENTITY_TEXT = frozenset({"", "nan", "none", "<na>", "nat"})


def _canonical_identity_labels(values: Any, *, default: str) -> np.ndarray:
    """Return canonical categorical identities without rewriting visible labels."""

    text = pd.Series(values, copy=False).astype("string").str.strip().str.casefold()
    missing = text.isna() | text.isin(_MISSING_IDENTITY_TEXT)
    return text.where(~missing, str(default).strip().casefold()).to_numpy(object)


def _physical_scope_columns(rows: pd.DataFrame) -> tuple[str, ...]:
    """Return every available physical-flight scope dimension."""

    columns = ["sequence_id"]
    if "flight_id" in rows.columns:
        columns.append("flight_id")
    return tuple(columns)


def _pair_consensus_advantage_scoped(
    rows: pd.DataFrame,
    *,
    origin_column: str | None,
    distance_column: str,
    missing_support_margin_m: float,
) -> np.ndarray:
    """Compare sibling branches only inside the same physical flight."""

    advantage = np.full(len(rows), np.nan, dtype=float)
    if origin_column is None or origin_column not in rows.columns:
        return advantage
    group_keys = [*_physical_scope_columns(rows), "source", origin_column]
    valid_origin = (
        rows[origin_column].notna()
        & rows[origin_column].astype(str).str.strip().ne("")
    )
    for _, group in rows.loc[valid_origin].groupby(
        group_keys,
        sort=False,
        dropna=False,
    ):
        if len(group) < 2:
            continue
        distances = pd.to_numeric(group[distance_column], errors="coerce")
        for row_index in group.index:
            current = distances.loc[row_index]
            siblings = distances.drop(index=row_index)
            finite_siblings = siblings[np.isfinite(siblings.to_numpy(float))]
            current_finite = bool(np.isfinite(current))
            if current_finite and finite_siblings.empty:
                advantage[int(row_index)] = float(missing_support_margin_m)
            elif not current_finite and not finite_siblings.empty:
                advantage[int(row_index)] = -float(missing_support_margin_m)
            elif current_finite and not finite_siblings.empty:
                advantage[int(row_index)] = float(finite_siblings.min() - float(current))
    return advantage


def attach_candidate_branch_consensus(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    time_window_s: float = 0.05,
    time_scale_s: float | None = None,
    distance_gate_m: float = 5.0,
    distance_scale_m: float = 5.0,
    base_score_column: str = "ranker_score",
    score_output_column: str = DEFAULT_SCORE_OUTPUT_COLUMN,
    base_score_weight: float = 1.0,
    consensus_weight: float = 1.0,
    pair_advantage_weight: float = 0.25,
    branch_column: str | None = None,
    origin_column: str | None = None,
    exclude_same_origin_support: bool = True,
    replace_confidence: bool = False,
) -> CandidateFrame:
    """Attach consensus features using canonical identities and physical scope."""

    rows = _IMPL._candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(rows)
    if float(time_window_s) < 0.0:
        raise ValueError("time_window_s must be non-negative")
    if float(distance_gate_m) <= 0.0:
        raise ValueError("distance_gate_m must be positive")
    if float(distance_scale_m) <= 0.0:
        raise ValueError("distance_scale_m must be positive")
    resolved_time_scale_s = _IMPL._resolved_time_scale(time_window_s, time_scale_s)

    out = rows.copy().reset_index(drop=True)
    scope_columns = _physical_scope_columns(out)
    resolved_branch = _IMPL._resolve_column(
        out,
        branch_column,
        _IMPL.DEFAULT_BRANCH_COLUMN_ALIASES,
    )
    resolved_origin = _IMPL._resolve_column(
        out,
        origin_column,
        _IMPL.DEFAULT_ORIGIN_COLUMN_ALIASES,
    )
    out["candidate_branch"] = _IMPL._branch_values(out, resolved_branch)
    out[_SOURCE_IDENTITY_COLUMN] = _canonical_identity_labels(
        out["source"],
        default="candidate",
    )
    out[_BRANCH_IDENTITY_COLUMN] = _canonical_identity_labels(
        out["candidate_branch"],
        default="unbranched",
    )
    out["branch_consensus_base_score"] = _IMPL._base_score(out, base_score_column)
    out["branch_consensus_base_score_normalized"] = _IMPL._group_minmax_score(
        out,
        "branch_consensus_base_score",
        group_columns=(
            *scope_columns,
            _SOURCE_IDENTITY_COLUMN,
            _BRANCH_IDENTITY_COLUMN,
        ),
    )

    nearest_distance = np.full(len(out), np.nan, dtype=float)
    nearest_time_delta = np.full(len(out), np.nan, dtype=float)
    nearest_source = np.full(len(out), "", dtype=object)
    nearest_branch = np.full(len(out), "", dtype=object)
    neighbor_count = np.zeros(len(out), dtype=int)
    unique_source_count = np.zeros(len(out), dtype=int)
    unique_branch_count = np.zeros(len(out), dtype=int)

    scope_group_key: str | list[str]
    if len(scope_columns) == 1:
        scope_group_key = scope_columns[0]
    else:
        scope_group_key = list(scope_columns)
    for _, scope_rows in out.groupby(
        scope_group_key,
        sort=False,
        dropna=False,
    ):
        ordered = scope_rows.sort_values("time_s")
        ordered_indices = ordered.index.to_numpy(int)
        times = ordered["time_s"].to_numpy(float)
        xyz = ordered[["x_m", "y_m", "z_m"]].to_numpy(float)
        visible_sources = ordered["source"].fillna("").astype(str).to_numpy(object)
        visible_branches = (
            ordered["candidate_branch"].fillna("").astype(str).to_numpy(object)
        )
        source_identities = ordered[_SOURCE_IDENTITY_COLUMN].to_numpy(object)
        branch_identities = ordered[_BRANCH_IDENTITY_COLUMN].to_numpy(object)
        origins = _IMPL._origin_values(ordered, resolved_origin)
        for local_index, global_index in enumerate(ordered_indices):
            lower = int(
                np.searchsorted(
                    times,
                    times[local_index] - float(time_window_s),
                    side="left",
                )
            )
            upper = int(
                np.searchsorted(
                    times,
                    times[local_index] + float(time_window_s),
                    side="right",
                )
            )
            candidate_indices = np.arange(lower, upper, dtype=int)
            candidate_indices = candidate_indices[candidate_indices != local_index]
            if candidate_indices.size == 0:
                continue
            different_source = (
                source_identities[candidate_indices] != source_identities[local_index]
            )
            candidate_indices = candidate_indices[different_source]
            if candidate_indices.size == 0:
                continue
            if exclude_same_origin_support and origins is not None:
                current_origin = origins[local_index]
                if current_origin is not None:
                    different_origin = np.array(
                        [
                            origin is None or origin != current_origin
                            for origin in origins[candidate_indices]
                        ],
                        dtype=bool,
                    )
                    candidate_indices = candidate_indices[different_origin]
            if candidate_indices.size == 0:
                continue
            distances = np.linalg.norm(
                xyz[candidate_indices] - xyz[local_index],
                axis=1,
            )
            finite = np.isfinite(distances)
            if not finite.any():
                continue
            candidate_indices = candidate_indices[finite]
            distances = distances[finite]
            best_local = int(np.argmin(distances))
            best_index = int(candidate_indices[best_local])
            nearest_distance[global_index] = float(distances[best_local])
            nearest_time_delta[global_index] = float(
                abs(times[best_index] - times[local_index])
            )
            nearest_source[global_index] = str(visible_sources[best_index])
            nearest_branch[global_index] = str(visible_branches[best_index])
            supported = distances <= float(distance_gate_m)
            neighbor_count[global_index] = int(supported.sum())
            if supported.any():
                support_indices = candidate_indices[supported]
                unique_source_count[global_index] = len(
                    set(source_identities[support_indices])
                )
                unique_branch_count[global_index] = len(
                    set(branch_identities[support_indices])
                )

    out["branch_consensus_nearest_cross_source_distance_m"] = nearest_distance
    out["branch_consensus_nearest_cross_source_time_delta_s"] = nearest_time_delta
    out["branch_consensus_nearest_cross_source"] = nearest_source
    out["branch_consensus_nearest_cross_source_branch"] = nearest_branch
    out["branch_consensus_neighbor_count"] = neighbor_count
    out["branch_consensus_unique_source_count"] = unique_source_count
    out["branch_consensus_unique_branch_count"] = unique_branch_count

    finite_distance = np.isfinite(nearest_distance)
    distance_score = np.zeros(len(out), dtype=float)
    distance_score[finite_distance] = np.exp(
        -nearest_distance[finite_distance] / float(distance_scale_m)
    )
    time_score = np.zeros(len(out), dtype=float)
    finite_time = np.isfinite(nearest_time_delta)
    time_score[finite_time] = np.exp(
        -nearest_time_delta[finite_time] / resolved_time_scale_s
    )
    joint_score = distance_score * time_score
    support_score = 1.0 - np.exp(-unique_source_count.astype(float))
    out["branch_consensus_distance_score"] = distance_score
    out["branch_consensus_time_score"] = time_score
    out["branch_consensus_support_score"] = support_score
    out["branch_consensus_score"] = 0.7 * joint_score + 0.3 * support_score

    pair_rows = out.copy()
    pair_rows["source"] = pair_rows[_SOURCE_IDENTITY_COLUMN]
    pair_advantage = _pair_consensus_advantage_scoped(
        pair_rows,
        origin_column=resolved_origin,
        distance_column="branch_consensus_nearest_cross_source_distance_m",
        missing_support_margin_m=float(distance_gate_m),
    )
    out["branch_consensus_pair_advantage_m"] = pair_advantage
    pair_preference = np.zeros(len(out), dtype=float)
    finite_advantage = np.isfinite(pair_advantage)
    pair_preference[finite_advantage] = np.tanh(
        pair_advantage[finite_advantage] / float(distance_scale_m)
    )
    out["branch_consensus_pair_preference"] = pair_preference

    rank_score = (
        float(base_score_weight)
        * out["branch_consensus_base_score_normalized"].to_numpy(float)
        + float(consensus_weight) * out["branch_consensus_score"].to_numpy(float)
        + float(pair_advantage_weight) * pair_preference
    )
    out[score_output_column] = rank_score
    out["branch_consensus_rank_percentile"] = _IMPL._group_minmax_score(
        out,
        score_output_column,
        group_columns=scope_columns,
    )
    if replace_confidence:
        out["confidence"] = rank_score
    out = out.drop(
        columns=[_SOURCE_IDENTITY_COLUMN, _BRANCH_IDENTITY_COLUMN],
        errors="ignore",
    )
    return CandidateFrame(_IMPL.normalize_candidate_columns(out))


_IMPL.attach_candidate_branch_consensus = attach_candidate_branch_consensus

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_canonical_identity_labels"] = _canonical_identity_labels
globals()["_physical_scope_columns"] = _physical_scope_columns
globals()["_pair_consensus_advantage_scoped"] = _pair_consensus_advantage_scoped
globals()["attach_candidate_branch_consensus"] = attach_candidate_branch_consensus

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
