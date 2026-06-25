"""Cross-sensor consensus features for branch-preserving MMUAD candidates.

Raw, dynamic, calibrated, and merged candidate branches can carry incomparable
scores. A global top-K can therefore discard a geometrically supported
hypothesis before the trajectory optimizer sees it. This module adds a
truth-free consensus score based on nearby candidates from different sensors
and, when available, compares raw/calibrated siblings from the same origin row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


DEFAULT_SCORE_OUTPUT_COLUMN = "branch_consensus_rank_score"
DEFAULT_ORIGIN_COLUMN_ALIASES = (
    "mmuad_calibration_origin_row",
    "candidate_origin_row",
    "origin_row",
)
DEFAULT_BRANCH_COLUMN_ALIASES = (
    "candidate_branch",
    "mmuad_source_calibration_branch",
    "branch",
    "candidate_stream",
)


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
    """Attach non-oracle cross-sensor consensus and branch-pair features.

    Candidate support is computed only from different sensor sources. This
    prevents raw and calibrated copies of one observation from voting for each
    other. If a calibration-origin column is present, the output also records
    which sibling branch has stronger independent sensor support.
    """

    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(rows)
    if float(time_window_s) < 0.0:
        raise ValueError("time_window_s must be non-negative")
    if float(distance_gate_m) <= 0.0:
        raise ValueError("distance_gate_m must be positive")
    if float(distance_scale_m) <= 0.0:
        raise ValueError("distance_scale_m must be positive")
    resolved_time_scale_s = _resolved_time_scale(time_window_s, time_scale_s)

    out = rows.copy().reset_index(drop=True)
    resolved_branch = _resolve_column(
        out,
        branch_column,
        DEFAULT_BRANCH_COLUMN_ALIASES,
    )
    resolved_origin = _resolve_column(
        out,
        origin_column,
        DEFAULT_ORIGIN_COLUMN_ALIASES,
    )
    out["candidate_branch"] = _branch_values(out, resolved_branch)
    out["branch_consensus_base_score"] = _base_score(out, base_score_column)
    out["branch_consensus_base_score_normalized"] = _group_minmax_score(
        out,
        "branch_consensus_base_score",
        group_columns=("sequence_id", "source", "candidate_branch"),
    )

    nearest_distance = np.full(len(out), np.nan, dtype=float)
    nearest_time_delta = np.full(len(out), np.nan, dtype=float)
    nearest_source = np.full(len(out), "", dtype=object)
    nearest_branch = np.full(len(out), "", dtype=object)
    neighbor_count = np.zeros(len(out), dtype=int)
    unique_source_count = np.zeros(len(out), dtype=int)
    unique_branch_count = np.zeros(len(out), dtype=int)

    for _, sequence_rows in out.groupby("sequence_id", sort=False):
        ordered = sequence_rows.sort_values("time_s")
        ordered_indices = ordered.index.to_numpy(int)
        times = ordered["time_s"].to_numpy(float)
        xyz = ordered[["x_m", "y_m", "z_m"]].to_numpy(float)
        sources = ordered["source"].fillna("").astype(str).to_numpy(object)
        branches = (
            ordered["candidate_branch"].fillna("").astype(str).to_numpy(object)
        )
        origins = _origin_values(ordered, resolved_origin)
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
            different_source = sources[candidate_indices] != sources[local_index]
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
            nearest_source[global_index] = str(sources[best_index])
            nearest_branch[global_index] = str(branches[best_index])
            supported = distances <= float(distance_gate_m)
            neighbor_count[global_index] = int(supported.sum())
            if supported.any():
                support_indices = candidate_indices[supported]
                unique_source_count[global_index] = int(
                    len(set(sources[support_indices]))
                )
                unique_branch_count[global_index] = int(
                    len(set(branches[support_indices]))
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

    pair_advantage = _pair_consensus_advantage(
        out,
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
    out["branch_consensus_rank_percentile"] = _group_minmax_score(
        out,
        score_output_column,
        group_columns=("sequence_id",),
    )
    if replace_confidence:
        out["confidence"] = rank_score
    return CandidateFrame(normalize_candidate_columns(out))


def branch_consensus_summary(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    score_output_column: str = DEFAULT_SCORE_OUTPUT_COLUMN,
) -> dict[str, Any]:
    """Return compact diagnostics for a consensus-augmented candidate table."""

    rows = _candidate_rows(candidates)
    nearest = _numeric_column(
        rows,
        "branch_consensus_nearest_cross_source_distance_m",
    )
    finite_nearest = nearest[np.isfinite(nearest.to_numpy(float))]
    pair_advantage = _numeric_column(rows, "branch_consensus_pair_advantage_m")
    finite_pair = pair_advantage[np.isfinite(pair_advantage.to_numpy(float))]
    score = _numeric_column(rows, score_output_column)
    finite_score = score[np.isfinite(score.to_numpy(float))]
    return {
        "row_count": int(len(rows)),
        "sequence_count": (
            int(rows["sequence_id"].astype(str).nunique()) if not rows.empty else 0
        ),
        "candidate_branch_counts": _value_counts(rows, "candidate_branch"),
        "source_counts": _value_counts(rows, "source"),
        "cross_source_match_count": int(len(finite_nearest)),
        "cross_source_match_fraction": (
            float(len(finite_nearest) / len(rows)) if len(rows) else 0.0
        ),
        "nearest_cross_source_distance_mean_m": _safe_mean(finite_nearest),
        "nearest_cross_source_distance_p95_m": _safe_quantile(
            finite_nearest,
            0.95,
        ),
        "paired_hypothesis_count": int(len(finite_pair)),
        "paired_hypothesis_positive_advantage_count": int(
            (finite_pair > 0.0).sum()
        ),
        "score_output_column": str(score_output_column),
        "rank_score_mean": _safe_mean(finite_score),
        "rank_score_p95": _safe_quantile(finite_score, 0.95),
    }


def write_candidate_branch_consensus(
    candidates: CandidateFrame,
    *,
    output_csv: Path,
    provenance_json: Path | None = None,
    provenance: dict[str, Any] | None = None,
    score_output_column: str = DEFAULT_SCORE_OUTPUT_COLUMN,
) -> None:
    """Write consensus candidates and optional provenance/summary JSON."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.rows.to_csv(output_csv, index=False)
    if provenance_json is None:
        return
    provenance_json.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(provenance or {})
    payload.update(
        branch_consensus_summary(
            candidates,
            score_output_column=score_output_column,
        )
    )
    payload["output_csv"] = str(output_csv)
    provenance_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-branch-consensus",
        description=(
            "attach cross-sensor branch-consensus ranking features to MMUAD "
            "candidates"
        ),
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument("--time-window-s", type=float, default=0.05)
    parser.add_argument("--time-scale-s", type=float)
    parser.add_argument("--distance-gate-m", type=float, default=5.0)
    parser.add_argument("--distance-scale-m", type=float, default=5.0)
    parser.add_argument("--base-score-column", default="ranker_score")
    parser.add_argument("--score-output-column", default=DEFAULT_SCORE_OUTPUT_COLUMN)
    parser.add_argument("--base-score-weight", type=float, default=1.0)
    parser.add_argument("--consensus-weight", type=float, default=1.0)
    parser.add_argument("--pair-advantage-weight", type=float, default=0.25)
    parser.add_argument("--branch-column")
    parser.add_argument("--origin-column")
    parser.add_argument("--allow-same-origin-support", action="store_true")
    parser.add_argument("--replace-confidence", action="store_true")
    args = parser.parse_args(argv)

    candidates = load_candidate_file(args.candidate_csv)
    augmented = attach_candidate_branch_consensus(
        candidates,
        time_window_s=args.time_window_s,
        time_scale_s=args.time_scale_s,
        distance_gate_m=args.distance_gate_m,
        distance_scale_m=args.distance_scale_m,
        base_score_column=args.base_score_column,
        score_output_column=args.score_output_column,
        base_score_weight=args.base_score_weight,
        consensus_weight=args.consensus_weight,
        pair_advantage_weight=args.pair_advantage_weight,
        branch_column=args.branch_column,
        origin_column=args.origin_column,
        exclude_same_origin_support=not args.allow_same_origin_support,
        replace_confidence=args.replace_confidence,
    )
    write_candidate_branch_consensus(
        augmented,
        output_csv=args.output_csv,
        provenance_json=args.provenance_json,
        score_output_column=args.score_output_column,
        provenance={
            "candidate_csv": str(args.candidate_csv),
            "time_window_s": float(args.time_window_s),
            "time_scale_s": args.time_scale_s,
            "distance_gate_m": float(args.distance_gate_m),
            "distance_scale_m": float(args.distance_scale_m),
            "base_score_column": str(args.base_score_column),
            "base_score_weight": float(args.base_score_weight),
            "consensus_weight": float(args.consensus_weight),
            "pair_advantage_weight": float(args.pair_advantage_weight),
            "branch_column": args.branch_column,
            "origin_column": args.origin_column,
            "exclude_same_origin_support": not args.allow_same_origin_support,
            "replace_confidence": bool(args.replace_confidence),
        },
    )
    print("mmuad_candidate_branch_consensus=ok")
    print(f"output_csv={args.output_csv}")
    if args.provenance_json is not None:
        print(f"provenance_json={args.provenance_json}")
    return 0


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    rows = (
        candidates.rows.copy()
        if isinstance(candidates, CandidateFrame)
        else pd.DataFrame(candidates)
    )
    return normalize_candidate_columns(rows)


def _resolve_column(
    rows: pd.DataFrame,
    explicit: str | None,
    aliases: Iterable[str],
) -> str | None:
    if explicit is not None:
        if explicit not in rows.columns:
            raise ValueError(f"candidate column {explicit!r} is not present")
        return explicit
    lower_to_original = {
        str(column).lower(): str(column)
        for column in rows.columns
    }
    for alias in aliases:
        lowered = str(alias).lower()
        if lowered in lower_to_original:
            return lower_to_original[lowered]
    return None


def _branch_values(rows: pd.DataFrame, branch_column: str | None) -> pd.Series:
    raw = rows["source"] if branch_column is None else rows[branch_column]
    text = raw.where(raw.notna(), "unbranched").astype(str).str.strip()
    missing = text.eq("") | text.str.lower().isin({"nan", "none", "<na>"})
    return text.where(~missing, "unbranched")


def _origin_values(
    rows: pd.DataFrame,
    origin_column: str | None,
) -> np.ndarray | None:
    if origin_column is None:
        return None
    values: list[object | None] = []
    for value in rows[origin_column]:
        if value is None or pd.isna(value) or str(value).strip() == "":
            values.append(None)
        else:
            values.append(str(value))
    return np.asarray(values, dtype=object)


def _base_score(rows: pd.DataFrame, requested_column: str) -> pd.Series:
    for column in (requested_column, "ranker_score", "confidence", "score"):
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(float))]
        if not finite.empty:
            return values.fillna(float(finite.min()))
    return pd.Series(np.ones(len(rows), dtype=float), index=rows.index)


def _group_minmax_score(
    rows: pd.DataFrame,
    column: str,
    *,
    group_columns: tuple[str, ...],
) -> pd.Series:
    values = pd.to_numeric(rows[column], errors="coerce")
    result = pd.Series(np.zeros(len(rows), dtype=float), index=rows.index)
    working = rows.assign(_branch_consensus_score=values)
    for _, group in working.groupby(list(group_columns), sort=False, dropna=False):
        group_values = pd.to_numeric(
            group["_branch_consensus_score"],
            errors="coerce",
        )
        finite = group_values[np.isfinite(group_values.to_numpy(float))]
        if finite.empty:
            result.loc[group.index] = 0.0
            continue
        lower = float(finite.min())
        upper = float(finite.max())
        if upper - lower <= 1.0e-12:
            result.loc[group.index] = 0.5
        else:
            result.loc[group.index] = (group_values - lower) / (upper - lower)
    return result.fillna(0.0)


def _pair_consensus_advantage(
    rows: pd.DataFrame,
    *,
    origin_column: str | None,
    distance_column: str,
    missing_support_margin_m: float,
) -> np.ndarray:
    advantage = np.full(len(rows), np.nan, dtype=float)
    if origin_column is None or origin_column not in rows.columns:
        return advantage
    group_keys = ["sequence_id", "source", origin_column]
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
            finite_siblings = siblings[
                np.isfinite(siblings.to_numpy(float))
            ]
            current_finite = bool(np.isfinite(current))
            if current_finite and finite_siblings.empty:
                advantage[int(row_index)] = float(missing_support_margin_m)
            elif not current_finite and not finite_siblings.empty:
                advantage[int(row_index)] = -float(missing_support_margin_m)
            elif current_finite and not finite_siblings.empty:
                advantage[int(row_index)] = float(
                    finite_siblings.min() - float(current)
                )
    return advantage


def _resolved_time_scale(
    time_window_s: float,
    time_scale_s: float | None,
) -> float:
    if time_scale_s is not None:
        if float(time_scale_s) <= 0.0:
            raise ValueError("time_scale_s must be positive")
        return float(time_scale_s)
    return max(float(time_window_s) / 2.0, 1.0e-6)


def _numeric_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns or rows.empty:
        return {}
    counts = rows[column].fillna("").astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _safe_mean(values: pd.Series) -> float | None:
    return None if values.empty else float(values.mean())


def _safe_quantile(values: pd.Series, quantile: float) -> float | None:
    return None if values.empty else float(values.quantile(float(quantile)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
