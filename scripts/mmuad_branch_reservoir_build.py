#!/usr/bin/env python
"""Build a truth-free branch-preserving MMUAD candidate reservoir.

This runner is the inference-side companion to
``mmuad_branch_reservoir_oracle_recall.py``.  It keeps raw, dynamic,
source-calibrated, merged, or other candidate streams as explicit branches and
writes a bounded per-frame reservoir without looking at validation/test truth.
The resulting CSV can be fed into downstream ranker, tracker, or mixture-MAP
experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.io import load_candidate_file  # noqa: E402
from raft_uav.mmuad.schema import normalize_candidate_columns  # noqa: E402

RESERVOIR_CSV = "mmuad_branch_reservoir_candidates.csv"
FRAME_SUMMARY_CSV = "mmuad_branch_reservoir_frame_summary.csv"
BRANCH_SUMMARY_CSV = "mmuad_branch_reservoir_branch_summary.csv"
PROVENANCE_JSON = "mmuad_branch_reservoir_build_provenance.json"


@dataclass(frozen=True)
class CandidateInput:
    branch: str
    path: Path


def parse_candidate_input(value: str) -> CandidateInput:
    """Parse ``BRANCH=path`` or infer a branch from a plain path stem."""

    if "=" in value:
        branch, path_text = value.split("=", 1)
        branch = _safe_label(branch) or _safe_label(Path(path_text).stem)
        return CandidateInput(branch=branch, path=Path(path_text))
    path = Path(value)
    return CandidateInput(branch=_safe_label(path.stem) or "candidate", path=path)


def load_branch_candidate_inputs(inputs: Iterable[CandidateInput]) -> pd.DataFrame:
    """Load repeated candidate CSVs and attach stable branch labels."""

    frames: list[pd.DataFrame] = []
    for item in inputs:
        frame = load_candidate_file(item.path, source=item.branch)
        rows = frame.rows.copy()
        if "candidate_branch" in rows:
            rows["candidate_branch"] = rows["candidate_branch"].fillna(item.branch)
        else:
            rows["candidate_branch"] = item.branch
        rows["candidate_input_path"] = str(item.path)
        frames.append(rows)
    if not frames:
        return normalize_candidate_columns(pd.DataFrame())
    return pd.concat(frames, ignore_index=True, sort=False)


def build_truth_free_branch_reservoir(
    candidates: pd.DataFrame,
    *,
    per_source_top_n: int = 3,
    per_branch_top_n: int = 3,
    global_top_n: int = 20,
    score_column: str = "ranker_score",
    score_floor_quantile: float | None = None,
) -> pd.DataFrame:
    """Return a bounded per-frame reservoir without using truth labels."""

    rows = _finite_candidate_rows(candidates, score_column=score_column)
    if rows.empty:
        return rows
    selected_frames: list[pd.DataFrame] = []
    group_keys = ["sequence_id", "time_s"]
    for _, frame in rows.groupby(group_keys, sort=True, dropna=False):
        selected_reasons: dict[int, set[str]] = {}
        if per_source_top_n > 0:
            for source, group in frame.groupby("source", sort=True, dropna=False):
                _add_reason(selected_reasons, _top_ids(group, per_source_top_n), f"source:{source}")
        if per_branch_top_n > 0:
            for branch, group in frame.groupby("candidate_branch", sort=True, dropna=False):
                _add_reason(selected_reasons, _top_ids(group, per_branch_top_n), f"branch:{branch}")
        if global_top_n > 0:
            _add_reason(selected_reasons, _top_ids(frame, global_top_n), "global")
        if score_floor_quantile is not None:
            q = float(score_floor_quantile)
            if not 0.0 <= q <= 1.0:
                raise ValueError("score_floor_quantile must be in [0, 1]")
            threshold = float(np.nanquantile(frame["_reservoir_score"].to_numpy(float), q))
            score_floor_ids = frame.loc[
                frame["_reservoir_score"] >= threshold, "_candidate_row_id"
            ].astype(int)
            _add_reason(selected_reasons, set(score_floor_ids), "score_floor")
        if not selected_reasons:
            continue
        selected_ids = set(selected_reasons)
        selected = frame.loc[frame["_candidate_row_id"].astype(int).isin(selected_ids)].copy()
        selected = selected.sort_values(["_reservoir_score", "_candidate_row_id"], ascending=[False, True])
        selected["reservoir_rank_in_frame"] = np.arange(1, len(selected) + 1, dtype=int)
        selected["reservoir_input_count_frame"] = int(len(frame))
        selected["reservoir_count_frame"] = int(len(selected))
        selected["reservoir_reason"] = [
            ";".join(sorted(selected_reasons[int(row_id)]))
            for row_id in selected["_candidate_row_id"].astype(int)
        ]
        selected_frames.append(selected)
    if not selected_frames:
        return rows.iloc[0:0].copy()
    return pd.concat(selected_frames, ignore_index=True, sort=False)


def build_frame_summary(reservoir: pd.DataFrame) -> pd.DataFrame:
    """Summarize reservoir size and score spread per timestamp."""

    columns = [
        "sequence_id",
        "time_s",
        "input_candidate_count",
        "reservoir_candidate_count",
        "branch_count",
        "source_count",
        "max_reservoir_score",
        "min_reservoir_score",
    ]
    if reservoir.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, object]] = []
    for (sequence_id, time_s), group in reservoir.groupby(["sequence_id", "time_s"], sort=True):
        scores = pd.to_numeric(group["_reservoir_score"], errors="coerce")
        records.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "input_candidate_count": int(group["reservoir_input_count_frame"].max()),
                "reservoir_candidate_count": int(len(group)),
                "branch_count": int(group["candidate_branch"].nunique(dropna=False)),
                "source_count": int(group["source"].nunique(dropna=False)),
                "max_reservoir_score": float(scores.max()) if not scores.empty else np.nan,
                "min_reservoir_score": float(scores.min()) if not scores.empty else np.nan,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def build_branch_summary(reservoir: pd.DataFrame) -> pd.DataFrame:
    """Summarize retained candidates by branch/source."""

    columns = [
        "candidate_branch",
        "source",
        "candidate_count",
        "frame_count",
        "mean_rank_in_frame",
        "mean_score",
        "p95_score",
    ]
    if reservoir.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, object]] = []
    for (branch, source), group in reservoir.groupby(["candidate_branch", "source"], sort=True):
        scores = pd.to_numeric(group["_reservoir_score"], errors="coerce")
        records.append(
            {
                "candidate_branch": str(branch),
                "source": str(source),
                "candidate_count": int(len(group)),
                "frame_count": int(group[["sequence_id", "time_s"]].drop_duplicates().shape[0]),
                "mean_rank_in_frame": float(
                    pd.to_numeric(group["reservoir_rank_in_frame"], errors="coerce").mean()
                ),
                "mean_score": float(scores.mean()) if not scores.empty else np.nan,
                "p95_score": float(np.nanpercentile(scores.to_numpy(float), 95))
                if not scores.empty
                else np.nan,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        metavar="BRANCH=PATH",
        help="candidate CSV to include; may be repeated; plain PATH uses file stem as branch",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--score-floor-quantile", type=float)
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv BRANCH=PATH")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = tuple(parse_candidate_input(value) for value in args.candidate_csv)
    candidates = load_branch_candidate_inputs(inputs)
    reservoir = build_truth_free_branch_reservoir(
        candidates,
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        global_top_n=int(args.global_top_n),
        score_column=str(args.score_column),
        score_floor_quantile=args.score_floor_quantile,
    )
    frame_summary = build_frame_summary(reservoir)
    branch_summary = build_branch_summary(reservoir)

    reservoir_path = args.output_dir / RESERVOIR_CSV
    frame_summary_path = args.output_dir / FRAME_SUMMARY_CSV
    branch_summary_path = args.output_dir / BRANCH_SUMMARY_CSV
    provenance_path = args.output_dir / PROVENANCE_JSON
    reservoir.to_csv(reservoir_path, index=False)
    frame_summary.to_csv(frame_summary_path, index=False)
    branch_summary.to_csv(branch_summary_path, index=False)
    provenance = {
        "candidate_inputs": [{"branch": item.branch, "path": str(item.path)} for item in inputs],
        "per_source_top_n": int(args.per_source_top_n),
        "per_branch_top_n": int(args.per_branch_top_n),
        "global_top_n": int(args.global_top_n),
        "score_column": str(args.score_column),
        "score_floor_quantile": args.score_floor_quantile,
        "input_candidate_rows": int(len(candidates)),
        "reservoir_candidate_rows": int(len(reservoir)),
        "reservoir_csv": str(reservoir_path),
        "frame_summary_csv": str(frame_summary_path),
        "branch_summary_csv": str(branch_summary_path),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print("mmuad_branch_reservoir_build=ok")
    print(f"reservoir_csv={reservoir_path}")
    print(f"frame_summary_csv={frame_summary_path}")
    print(f"branch_summary_csv={branch_summary_path}")
    print(f"provenance_json={provenance_path}")
    return 0


def _finite_candidate_rows(candidates: pd.DataFrame, *, score_column: str) -> pd.DataFrame:
    rows = normalize_candidate_columns(pd.DataFrame(candidates)).copy()
    if rows.empty:
        return pd.DataFrame(columns=_candidate_columns())
    if "track_id" not in rows.columns:
        rows["track_id"] = np.nan
    if "candidate_branch" not in rows.columns:
        rows["candidate_branch"] = rows.get("source", "candidate")
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["candidate_branch"] = rows["candidate_branch"].map(lambda value: _safe_label(value) or "candidate")
    for column in ("time_s", "x_m", "y_m", "z_m", "confidence"):
        if column not in rows.columns:
            rows[column] = np.nan
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["_reservoir_score"] = _candidate_score(rows, score_column=score_column)
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    rows = rows.loc[finite].copy().reset_index(drop=True)
    rows["_candidate_row_id"] = np.arange(len(rows), dtype=int)
    return rows


def _candidate_score(rows: pd.DataFrame, *, score_column: str) -> pd.Series:
    for column in (score_column, "ranker_score", "confidence", "score"):
        if column in rows.columns:
            score = pd.to_numeric(rows[column], errors="coerce")
            finite = score[np.isfinite(score.to_numpy(float))]
            if not finite.empty:
                return score.fillna(float(finite.min()))
    return pd.Series(np.ones(len(rows), dtype=float), index=rows.index)


def _top_ids(rows: pd.DataFrame, n: int) -> set[int]:
    if int(n) <= 0 or rows.empty:
        return set()
    ordered = rows.sort_values(["_reservoir_score", "_candidate_row_id"], ascending=[False, True])
    return set(ordered.head(int(n))["_candidate_row_id"].astype(int))


def _add_reason(target: dict[int, set[str]], ids: Iterable[int], reason: str) -> None:
    for row_id in ids:
        target.setdefault(int(row_id), set()).add(str(reason))


def _candidate_columns() -> list[str]:
    return [
        "sequence_id",
        "time_s",
        "source",
        "track_id",
        "x_m",
        "y_m",
        "z_m",
        "confidence",
        "candidate_branch",
        "_reservoir_score",
    ]


def _safe_label(value: object) -> str:
    return (
        "" if value is None else str(value)
    ).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
