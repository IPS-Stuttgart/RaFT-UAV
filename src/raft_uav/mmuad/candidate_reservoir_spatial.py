"""Spatially diverse final caps for MMUAD candidate reservoirs.

Branch/source quotas preserve candidate-generation provenance, but a score-only
cap can still keep several nearly identical candidates while discarding a lower
scored, geometrically distinct hypothesis. This module fills the remaining
per-frame budget with a maximum-marginal-relevance style score that combines
normalized candidate score and distance from already selected candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import build_oracle_recall_tables
from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns


def spatial_diversity_cap_reservoir(
    reservoir: pd.DataFrame,
    *,
    max_candidates_per_frame: int = 40,
    min_per_source: int = 1,
    min_per_branch: int = 1,
    score_column: str = "candidate_reservoir_score",
    fallback_score_column: str = "confidence",
    branch_column: str = "candidate_branch",
    spatial_diversity_weight: float = 1.0,
    spatial_diversity_scale_m: float = 10.0,
    spatial_distance_cap_m: float = 50.0,
) -> pd.DataFrame:
    """Cap reservoir rows while preserving provenance and spatial alternatives.

    The algorithm first protects the best rows per source and branch. Remaining
    slots are filled greedily using

    ``score_normalized + weight * (1 - exp(-distance / scale))``.

    Distance is the minimum 3D distance to any already selected candidate and
    can be capped to prevent a single extreme candidate from dominating.
    """

    rows = normalize_candidate_columns(pd.DataFrame(reservoir).copy())
    if rows.empty:
        return rows.assign(
            candidate_spatial_cap_rank=pd.Series(dtype=float),
            candidate_spatial_cap_reason=pd.Series(dtype=str),
            candidate_spatial_min_distance_m=pd.Series(dtype=float),
            candidate_spatial_diversity_term=pd.Series(dtype=float),
            candidate_spatial_selection_utility=pd.Series(dtype=float),
        )
    rows = rows.copy().reset_index(drop=True)
    _ensure_columns(rows, branch_column=branch_column)
    rows["_spatial_row_id"] = np.arange(len(rows), dtype=int)
    rows["_spatial_score"] = _score(rows, score_column, fallback_score_column)

    parts: list[pd.DataFrame] = []
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=False, dropna=False):
        parts.append(
            _cap_frame(
                frame.copy(),
                max_candidates_per_frame=int(max_candidates_per_frame),
                min_per_source=int(min_per_source),
                min_per_branch=int(min_per_branch),
                branch_column=branch_column,
                spatial_diversity_weight=float(spatial_diversity_weight),
                spatial_diversity_scale_m=float(spatial_diversity_scale_m),
                spatial_distance_cap_m=float(spatial_distance_cap_m),
            )
        )
    out = pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0].copy()
    out = out.drop(columns=["_spatial_row_id", "_spatial_score", "_spatial_score_norm"], errors="ignore")
    return out.sort_values(["sequence_id", "time_s", "candidate_spatial_cap_rank"]).reset_index(
        drop=True,
    )


def spatial_diversity_summary(input_rows: pd.DataFrame, output_rows: pd.DataFrame) -> dict[str, Any]:
    """Build a compact JSON-serializable summary."""

    output_distances = pd.to_numeric(
        output_rows.get("candidate_spatial_min_distance_m", pd.Series(dtype=float)),
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
        "selected_min_distance_mean_m": _mean(output_distances),
        "selected_min_distance_p50_m": _quantile(output_distances, 0.50),
        "selected_min_distance_p95_m": _quantile(output_distances, 0.95),
        "source_counts": _value_counts(output_rows, "source"),
        "branch_counts": _value_counts(output_rows, "candidate_branch"),
        "spatial_cap_reason_counts": _reason_counts(output_rows),
    }


def write_spatial_diversity_outputs(
    capped: pd.DataFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    input_rows: pd.DataFrame | None = None,
) -> None:
    """Write capped candidates and optional summary JSON."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    capped.to_csv(output_csv, index=False)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(
                spatial_diversity_summary(input_rows if input_rows is not None else capped, capped),
                indent=2,
            ),
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-spatial-diversity-reservoir",
        description="apply a spatially diverse final cap to an MMUAD candidate reservoir",
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--oracle-frame-csv", type=Path)
    parser.add_argument("--oracle-summary-csv", type=Path)
    parser.add_argument("--oracle-by-sequence-csv", type=Path)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--min-per-source", type=int, default=1)
    parser.add_argument("--min-per-branch", type=int, default=1)
    parser.add_argument("--score-column", default="candidate_reservoir_score")
    parser.add_argument("--fallback-score-column", default="confidence")
    parser.add_argument("--branch-column", default="candidate_branch")
    parser.add_argument("--spatial-diversity-weight", type=float, default=1.0)
    parser.add_argument("--spatial-diversity-scale-m", type=float, default=10.0)
    parser.add_argument("--spatial-distance-cap-m", type=float, default=50.0)
    parser.add_argument("--top-k", type=int, action="append", default=[1, 3, 5, 10, 20])
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    rows = pd.read_csv(args.input_csv)
    capped = spatial_diversity_cap_reservoir(
        rows,
        max_candidates_per_frame=args.max_candidates_per_frame,
        min_per_source=args.min_per_source,
        min_per_branch=args.min_per_branch,
        score_column=args.score_column,
        fallback_score_column=args.fallback_score_column,
        branch_column=args.branch_column,
        spatial_diversity_weight=args.spatial_diversity_weight,
        spatial_diversity_scale_m=args.spatial_diversity_scale_m,
        spatial_distance_cap_m=args.spatial_distance_cap_m,
    )
    write_spatial_diversity_outputs(
        capped,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        input_rows=rows,
    )
    print("mmuad_spatial_diversity_reservoir=ok")
    print(f"input_rows={len(rows)}")
    print(f"output_rows={len(capped)}")
    print(f"output_csv={args.output_csv}")

    if args.truth_csv is not None:
        truth = normalize_truth_columns(pd.read_csv(args.truth_csv))
        frame_rows, pooled, by_sequence = build_oracle_recall_tables(
            capped,
            truth,
            top_k_values=tuple(args.top_k),
            max_truth_time_delta_s=args.max_truth_time_delta_s,
        )
        _write_optional_csv(frame_rows, args.oracle_frame_csv)
        _write_optional_csv(pooled, args.oracle_summary_csv)
        _write_optional_csv(by_sequence, args.oracle_by_sequence_csv)
        if not pooled.empty:
            print(f"oracle_all_mse={pooled.loc[0, 'oracle_all_3d_m_mse']}")
    return 0


def _cap_frame(
    frame: pd.DataFrame,
    *,
    max_candidates_per_frame: int,
    min_per_source: int,
    min_per_branch: int,
    branch_column: str,
    spatial_diversity_weight: float,
    spatial_diversity_scale_m: float,
    spatial_distance_cap_m: float,
) -> pd.DataFrame:
    frame = frame.copy()
    frame["_spatial_score_norm"] = _normalize_score(frame["_spatial_score"])
    reasons: dict[int, set[str]] = {}
    protected: set[int] = set()
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

    if max_candidates_per_frame <= 0:
        selected_order = frame.sort_values("_spatial_score", ascending=False)[
            "_spatial_row_id"
        ].astype(int).tolist()
        for row_id in selected_order:
            reasons.setdefault(row_id, set()).add("unbounded")
    else:
        budget = min(int(max_candidates_per_frame), len(frame))
        protected_frame = frame.loc[frame["_spatial_row_id"].isin(protected)]
        if len(protected_frame) >= budget:
            selected_order = _greedy_select(
                protected_frame,
                count=budget,
                initial_ids=[],
                spatial_diversity_weight=spatial_diversity_weight,
                spatial_diversity_scale_m=spatial_diversity_scale_m,
                spatial_distance_cap_m=spatial_distance_cap_m,
            )
            for row_id in selected_order:
                reasons.setdefault(row_id, set()).add("protected_cap")
        else:
            selected_order = protected_frame.sort_values("_spatial_score", ascending=False)[
                "_spatial_row_id"
            ].astype(int).tolist()
            if not selected_order and budget > 0:
                seed = int(frame.sort_values("_spatial_score", ascending=False).iloc[0]["_spatial_row_id"])
                selected_order.append(seed)
                reasons.setdefault(seed, set()).add("score_seed")
            fill_count = max(budget - len(selected_order), 0)
            fill = _greedy_select(
                frame.loc[~frame["_spatial_row_id"].isin(selected_order)],
                count=fill_count,
                initial_ids=selected_order,
                reference_frame=frame,
                spatial_diversity_weight=spatial_diversity_weight,
                spatial_diversity_scale_m=spatial_diversity_scale_m,
                spatial_distance_cap_m=spatial_distance_cap_m,
            )
            for row_id in fill:
                reason = "spatial_fill" if spatial_diversity_weight > 0 else "score_fill"
                reasons.setdefault(row_id, set()).add(reason)
            selected_order.extend(fill)

    diagnostics = _selection_diagnostics(
        frame,
        selected_order,
        spatial_diversity_weight=spatial_diversity_weight,
        spatial_diversity_scale_m=spatial_diversity_scale_m,
        spatial_distance_cap_m=spatial_distance_cap_m,
    )
    selected = frame.set_index("_spatial_row_id").loc[selected_order].reset_index()
    selected["candidate_spatial_cap_rank"] = np.arange(1, len(selected) + 1, dtype=float)
    selected["candidate_spatial_cap_reason"] = [
        ";".join(sorted(reasons.get(int(row_id), {"score_fill"})))
        for row_id in selected["_spatial_row_id"].astype(int)
    ]
    selected["candidate_spatial_min_distance_m"] = [
        diagnostics[int(row_id)]["min_distance_m"]
        for row_id in selected["_spatial_row_id"].astype(int)
    ]
    selected["candidate_spatial_diversity_term"] = [
        diagnostics[int(row_id)]["diversity_term"]
        for row_id in selected["_spatial_row_id"].astype(int)
    ]
    selected["candidate_spatial_selection_utility"] = [
        diagnostics[int(row_id)]["utility"]
        for row_id in selected["_spatial_row_id"].astype(int)
    ]
    return selected


def _greedy_select(
    candidates: pd.DataFrame,
    *,
    count: int,
    initial_ids: list[int],
    spatial_diversity_weight: float,
    spatial_diversity_scale_m: float,
    spatial_distance_cap_m: float,
    reference_frame: pd.DataFrame | None = None,
) -> list[int]:
    if count <= 0 or candidates.empty:
        return []
    reference = candidates if reference_frame is None else reference_frame
    remaining = candidates.copy()
    selected_ids = list(initial_ids)
    chosen: list[int] = []
    while len(chosen) < count and not remaining.empty:
        if not selected_ids:
            best = remaining.sort_values(
                ["_spatial_score", "_spatial_row_id"],
                ascending=[False, True],
            ).iloc[0]
        else:
            selected_xyz = reference.loc[
                reference["_spatial_row_id"].isin(selected_ids), ["x_m", "y_m", "z_m"]
            ].to_numpy(float)
            candidate_xyz = remaining[["x_m", "y_m", "z_m"]].to_numpy(float)
            min_distances = _minimum_distances(candidate_xyz, selected_xyz)
            diversity = _diversity_term(
                min_distances,
                scale_m=spatial_diversity_scale_m,
                cap_m=spatial_distance_cap_m,
            )
            utility = (
                remaining["_spatial_score_norm"].to_numpy(float)
                + spatial_diversity_weight * diversity
            )
            ranked = remaining.assign(_spatial_utility=utility).sort_values(
                ["_spatial_utility", "_spatial_score", "_spatial_row_id"],
                ascending=[False, False, True],
            )
            best = ranked.iloc[0]
        row_id = int(best["_spatial_row_id"])
        chosen.append(row_id)
        selected_ids.append(row_id)
        remaining = remaining.loc[remaining["_spatial_row_id"] != row_id]
    return chosen


def _selection_diagnostics(
    frame: pd.DataFrame,
    selected_order: list[int],
    *,
    spatial_diversity_weight: float,
    spatial_diversity_scale_m: float,
    spatial_distance_cap_m: float,
) -> dict[int, dict[str, float]]:
    diagnostics: dict[int, dict[str, float]] = {}
    selected_xyz: list[np.ndarray] = []
    indexed = frame.set_index("_spatial_row_id")
    for row_id in selected_order:
        row = indexed.loc[int(row_id)]
        xyz = row[["x_m", "y_m", "z_m"]].to_numpy(float)
        if selected_xyz:
            previous = np.vstack(selected_xyz)
            min_distance = float(np.min(np.linalg.norm(previous - xyz, axis=1)))
            diversity = float(
                _diversity_term(
                    np.asarray([min_distance]),
                    scale_m=spatial_diversity_scale_m,
                    cap_m=spatial_distance_cap_m,
                )[0]
            )
        else:
            min_distance = float("nan")
            diversity = 0.0
        score_norm = float(row["_spatial_score_norm"])
        diagnostics[int(row_id)] = {
            "min_distance_m": min_distance,
            "diversity_term": diversity,
            "utility": score_norm + spatial_diversity_weight * diversity,
        }
        selected_xyz.append(xyz)
    return diagnostics


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
        selected = group.sort_values("_spatial_score", ascending=False).head(int(count))
        for row_id in selected["_spatial_row_id"].astype(int):
            protected.add(int(row_id))
            reasons.setdefault(int(row_id), set()).add(f"{reason_prefix}:{value}")


def _ensure_columns(rows: pd.DataFrame, *, branch_column: str) -> None:
    if "source" not in rows.columns:
        rows["source"] = "candidate"
    rows["source"] = rows["source"].fillna("candidate").astype(str)
    if branch_column not in rows.columns:
        if "candidate_branch" in rows.columns:
            rows[branch_column] = rows["candidate_branch"]
        else:
            rows[branch_column] = rows["source"]
    rows[branch_column] = rows[branch_column].fillna("candidate").astype(str)


def _score(rows: pd.DataFrame, score_column: str, fallback_score_column: str) -> pd.Series:
    primary = _numeric(rows, score_column, default=np.nan)
    fallback = _numeric(rows, fallback_score_column, default=1.0)
    return primary.fillna(fallback).fillna(0.0).astype(float)


def _normalize_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        return pd.Series(0.0, index=numeric.index, dtype=float)
    return (numeric - minimum) / (maximum - minimum)


def _minimum_distances(candidates_xyz: np.ndarray, selected_xyz: np.ndarray) -> np.ndarray:
    if selected_xyz.size == 0:
        return np.full(len(candidates_xyz), np.inf, dtype=float)
    differences = candidates_xyz[:, None, :] - selected_xyz[None, :, :]
    return np.min(np.linalg.norm(differences, axis=2), axis=1)


def _diversity_term(distances_m: np.ndarray, *, scale_m: float, cap_m: float) -> np.ndarray:
    safe_scale = max(float(scale_m), 1e-6)
    distances = np.asarray(distances_m, dtype=float)
    if cap_m > 0:
        distances = np.minimum(distances, float(cap_m))
    distances = np.maximum(distances, 0.0)
    return 1.0 - np.exp(-distances / safe_scale)


def _numeric(rows: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce")
    return pd.Series(default, index=rows.index, dtype=float)


def _frame_counts(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=int)
    return rows.groupby(["sequence_id", "time_s"], dropna=False).size()


def _mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else 0.0


def _quantile(values: pd.Series, quantile: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(quantile)) if not numeric.empty else 0.0


def _max(values: pd.Series) -> int:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return int(numeric.max()) if not numeric.empty else 0


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns:
        return {}
    return {str(key): int(value) for key, value in rows[column].value_counts(dropna=False).items()}


def _reason_counts(rows: pd.DataFrame) -> dict[str, int]:
    column = "candidate_spatial_cap_reason"
    if column not in rows.columns:
        return {}
    counts: dict[str, int] = {}
    for value in rows[column].dropna().astype(str):
        for reason in value.replace(",", ";").split(";"):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return counts


def _write_optional_csv(rows: pd.DataFrame, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
