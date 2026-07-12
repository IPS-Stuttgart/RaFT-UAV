"""Attach candidate-branch context features for MMUAD ranker experiments.

The branch-preserving MMUAD pipeline keeps raw, dynamic, translated, and merged
candidate streams alive until trajectory optimization.  This module converts the
branch identity on candidate rows into numeric ``image_*`` feature columns that
are already consumed by the existing cluster ranker.  The helper is deliberately
non-oracle: it uses only candidate metadata and optional branch labels supplied
by the pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns

DEFAULT_BRANCH_INTERACTION_COLUMNS = (
    "confidence",
    "cluster_point_count",
    "cluster_extent_xy_m",
    "cluster_extent_3d_m",
    "cluster_density_points_per_m3",
    "cluster_range_3d_m",
    "cluster_height_m",
    "std_xy_m",
    "std_z_m",
)
BRANCH_ALIASES = (
    "candidate_branch",
    "branch",
    "candidate_stream",
    "stream",
    "source_branch",
    "extraction_mode",
)


def attach_candidate_branch_context(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    branch_column: str | None = None,
    interaction_columns: Iterable[str] = DEFAULT_BRANCH_INTERACTION_COLUMNS,
    max_branch_values: int = 32,
    fallback_branch: str = "unbranched",
) -> CandidateFrame:
    """Return candidates with ranker-consumable branch one-hot/interactions.

    The emitted columns use an ``image_`` prefix because the cluster-ranker
    already treats numeric ``image_*`` columns as auxiliary sequence/candidate
    context.  This keeps the ranker model schema unchanged while letting
    branch-preserving experiments train source/branch-aware reliability models.
    """

    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(rows)
    branch_name = _resolve_branch_column(rows, branch_column=branch_column)
    out = rows.copy()
    branches = _normalized_branch_values(out, branch_name=branch_name, fallback_branch=fallback_branch)
    out["candidate_branch"] = branches
    branch_values = _selected_branch_values(branches, max_branch_values=max_branch_values)
    resolved_interaction_columns = tuple(interaction_columns)
    out["image_candidate_branch_available"] = 1.0
    out["image_candidate_branch_count"] = float(len(branch_values))
    for branch in branch_values:
        feature = f"image_candidate_branch_{_safe_feature_name(branch)}"
        indicator = (branches == branch).astype(float)
        out[feature] = indicator
        for column in resolved_interaction_columns:
            if column not in out.columns:
                continue
            values = pd.to_numeric(out[column], errors="coerce")
            out[f"{feature}_x_{_safe_feature_name(column)}"] = indicator * values
    return CandidateFrame(normalize_candidate_columns(out))


def write_candidate_branch_context(
    candidates: CandidateFrame,
    *,
    output_csv: Path,
    provenance_json: Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Write augmented candidates and optional provenance."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.rows.to_csv(output_csv, index=False)
    if provenance_json is not None:
        provenance_json.parent.mkdir(parents=True, exist_ok=True)
        context_columns = [
            str(column)
            for column in candidates.rows.columns
            if str(column).startswith("image_candidate_branch_")
        ]
        payload = dict(provenance or {})
        payload.update(
            {
                "output_csv": str(output_csv),
                "row_count": int(len(candidates.rows)),
                "candidate_branch_values": sorted(
                    str(value)
                    for value in candidates.rows.get("candidate_branch", pd.Series(dtype=str))
                    .dropna()
                    .astype(str)
                    .unique()
                ),
                "branch_context_columns": context_columns,
            }
        )
        provenance_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-branch-context",
        description="attach candidate-branch context features to MMUAD candidate rows",
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument("--branch-column", help="candidate column containing branch labels")
    parser.add_argument(
        "--interaction-column",
        action="append",
        default=[],
        help="candidate numeric column to multiply by branch indicators; may be repeated",
    )
    parser.add_argument("--max-branch-values", type=int, default=32)
    parser.add_argument("--fallback-branch", default="unbranched")
    args = parser.parse_args(argv)

    candidates = load_candidate_file(args.candidate_csv)
    interaction_columns = tuple(args.interaction_column) or DEFAULT_BRANCH_INTERACTION_COLUMNS
    augmented = attach_candidate_branch_context(
        candidates,
        branch_column=args.branch_column,
        interaction_columns=interaction_columns,
        max_branch_values=int(args.max_branch_values),
        fallback_branch=str(args.fallback_branch),
    )
    write_candidate_branch_context(
        augmented,
        output_csv=args.output_csv,
        provenance_json=args.provenance_json,
        provenance={
            "candidate_csv": str(args.candidate_csv),
            "branch_column": args.branch_column,
            "requested_interaction_columns": list(interaction_columns),
            "max_branch_values": int(args.max_branch_values),
            "fallback_branch": str(args.fallback_branch),
        },
    )
    print("mmuad_candidate_branch_context=ok")
    print(f"output_csv={args.output_csv}")
    if args.provenance_json is not None:
        print(f"provenance_json={args.provenance_json}")
    return 0


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    rows = candidates.rows.copy() if isinstance(candidates, CandidateFrame) else pd.DataFrame(candidates).copy()
    rows = normalize_candidate_columns(rows)
    if not rows.empty:
        rows["sequence_id"] = rows["sequence_id"].astype(str)
    return rows


def _resolve_branch_column(rows: pd.DataFrame, *, branch_column: str | None) -> str | None:
    if branch_column:
        if branch_column not in rows.columns:
            raise ValueError(f"branch column {branch_column!r} not present in candidate rows")
        return branch_column
    lower_to_original = {str(column).lower(): str(column) for column in rows.columns}
    for alias in BRANCH_ALIASES:
        if alias.lower() in lower_to_original:
            return lower_to_original[alias.lower()]
    return None


def _normalized_branch_values(
    rows: pd.DataFrame,
    *,
    branch_name: str | None,
    fallback_branch: str,
) -> pd.Series:
    if branch_name is None:
        if "source" in rows.columns:
            raw = rows["source"]
        else:
            raw = pd.Series([fallback_branch] * len(rows), index=rows.index)
    else:
        raw = rows[branch_name]
    text = raw.where(raw.notna(), fallback_branch).astype(str).str.strip()
    missing = text.eq("") | text.str.lower().isin({"nan", "none", "<na>"})
    return text.where(~missing, str(fallback_branch)).map(_safe_feature_name)


def _selected_branch_values(branches: pd.Series, *, max_branch_values: int) -> list[str]:
    counts = branches.value_counts(dropna=False)
    selected = [str(value) for value in counts.head(max(int(max_branch_values), 1)).index]
    return sorted(selected)


def _safe_feature_name(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    text = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)
    return text or "unknown"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
