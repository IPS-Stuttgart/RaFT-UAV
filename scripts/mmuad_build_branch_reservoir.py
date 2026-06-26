#!/usr/bin/env python
"""Build truth-free branch-preserving MMUAD candidate reservoirs.

The oracle-recall diagnostic identifies whether branch-preserving candidate pools
recover good candidates, but those diagnostic rows are truth-aware.  This script
materializes the same style of bounded reservoir without reading truth so it can
be fed into normal tracker or mixture-MAP experiments.
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

RESERVOIR_CSV = "mmuad_truth_free_branch_reservoir_candidates.csv"
FRAME_SUMMARY_CSV = "mmuad_truth_free_branch_reservoir_frame_summary.csv"
BRANCH_SUMMARY_CSV = "mmuad_truth_free_branch_reservoir_branch_summary.csv"
PROVENANCE_JSON = "mmuad_truth_free_branch_reservoir_provenance.json"


@dataclass(frozen=True)
class CandidateInput:
    branch: str
    path: Path


def parse_candidate_input(value: str) -> CandidateInput:
    """Parse ``BRANCH=path`` or use the file stem as branch label."""

    if "=" in value:
        branch, path_text = value.split("=", 1)
        branch = _safe_label(branch) or _safe_label(Path(path_text).stem)
        return CandidateInput(branch=branch, path=Path(path_text))
    path = Path(value)
    return CandidateInput(branch=_safe_label(path.stem) or "candidate", path=path)


def load_branch_candidate_inputs(inputs: Iterable[CandidateInput]) -> pd.DataFrame:
    """Load normalized candidate files and attach explicit branch labels."""

    frames: list[pd.DataFrame] = []
    for item in inputs:
        frame = load_candidate_file(item.path, source=item.branch)
        rows = frame.rows.copy()
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
    max_candidates_per_frame: int | None = None,
) -> pd.DataFrame:
    """Keep top candidates per source, per branch, and globally at each timestamp."""

    rows = _finite_candidate_rows(candidates, score_column=score_column)
    if rows.empty:
        return rows

    reservoirs: list[pd.DataFrame] = []
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=True, dropna=False):
        reservoir = _select_frame_reservoir(
            frame,
            per_source_top_n=per_source_top_n,
            per_branch_top_n=per_branch_top_n,
            global_top_n=global_top_n,
            score_floor_quantile=score_floor_quantile,
            max_candidates_per_frame=max_candidates_per_frame,
        )
        if not reservoir.empty:
            reservoirs.append(reservoir)
    if not reservoirs:
        return rows.iloc[0:0].drop(columns=["_reservoir_score", "_candidate_row_id"])
    result = pd.concat(reservoirs, ignore_index=True, sort=False)
    return result.drop(columns=["_reservoir_score", "_candidate_row_id"])


def build_frame_summary(candidates: pd.DataFrame, reservoir: pd.DataFrame) -> pd.DataFrame:
    """Summarize reservoir retention per sequence timestamp."""

    all_rows = _finite_candidate_rows(candidates, score_column="ranker_score")
    selected = normalize_candidate_columns(pd.DataFrame(reservoir)).copy()
    columns = [
        "sequence_id",
        "time_s",
        "candidate_count",
        "reservoir_count",
        "retained_fraction",
        "branch_count",
        "source_count",
        "reservoir_branch_count",
        "reservoir_source_count",
    ]
    if all_rows.empty:
        return pd.DataFrame(columns=columns)
    selected_keys = selected.groupby(["sequence_id", "time_s"], sort=False) if not selected.empty else {}
    records: list[dict[str, object]] = []
    for key, frame in all_rows.groupby(["sequence_id", "time_s"], sort=True, dropna=False):
        sequence_id, time_s = key
        selected_frame = _group_get(selected_keys, key)
        record = {
            "sequence_id": sequence_id,
            "time_s": float(time_s),
            "candidate_count": int(len(frame)),
            "reservoir_count": int(len(selected_frame)),
            "retained_fraction": float(len(selected_frame) / len(frame)) if len(frame) else 0.0,
            "branch_count": int(frame["candidate_branch"].nunique(dropna=False)),
            "source_count": int(frame["source"].nunique(dropna=False)),
            "reservoir_branch_count": int(selected_frame["candidate_branch"].nunique(dropna=False))
            if not selected_frame.empty
            else 0,
            "reservoir_source_count": int(selected_frame["source"].nunique(dropna=False))
            if not selected_frame.empty
            else 0,
        }
        records.append(record)
    return pd.DataFrame.from_records(records, columns=columns)


def build_branch_summary(
    candidates: pd.DataFrame,
    reservoir: pd.DataFrame,
    *,
    score_column: str = "ranker_score",
) -> pd.DataFrame:
    """Summarize candidate retention by input branch and source."""

    all_rows = _finite_candidate_rows(candidates, score_column=score_column)
    selected = normalize_candidate_columns(pd.DataFrame(reservoir)).copy()
    columns = [
        "candidate_branch",
        "source",
        "candidate_count",
        "reservoir_count",
        "retained_fraction",
        "frame_count",
        "reservoir_frame_count",
    ]
    if all_rows.empty:
        return pd.DataFrame(columns=columns)
    selected_groups = selected.groupby(["candidate_branch", "source"], sort=False) if not selected.empty else {}
    records: list[dict[str, object]] = []
    for key, group in all_rows.groupby(["candidate_branch", "source"], sort=True, dropna=False):
        branch, source = key
        selected_group = _group_get(selected_groups, key)
        records.append(
            {
                "candidate_branch": branch,
                "source": source,
                "candidate_count": int(len(group)),
                "reservoir_count": int(len(selected_group)),
                "retained_fraction": float(len(selected_group) / len(group)) if len(group) else 0.0,
                "frame_count": int(group[["sequence_id", "time_s"]].drop_duplicates().shape[0]),
                "reservoir_frame_count": int(
                    selected_group[["sequence_id", "time_s"]].drop_duplicates().shape[0]
                )
                if not selected_group.empty
                else 0,
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
    parser.add_argument("--max-candidates-per-frame", type=int)
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
        max_candidates_per_frame=args.max_candidates_per_frame,
    )
    frame_summary = build_frame_summary(candidates, reservoir)
    branch_summary = build_branch_summary(candidates, reservoir)

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
        "max_candidates_per_frame": args.max_candidates_per_frame,
        "input_candidate_rows": int(len(candidates)),
        "reservoir_candidate_rows": int(len(reservoir)),
        "reservoir_csv": str(reservoir_path),
        "frame_summary_csv": str(frame_summary_path),
        "branch_summary_csv": str(branch_summary_path),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print("mmuad_truth_free_branch_reservoir=ok")
    print(f"reservoir_csv={reservoir_path}")
    print(f"frame_summary_csv={frame_summary_path}")
    print(f"branch_summary_csv={branch_summary_path}")
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


def _select_frame_reservoir(
    frame: pd.DataFrame,
    *,
    per_source_top_n: int,
    per_branch_top_n: int,
    global_top_n: int,
    score_floor_quantile: float | None,
    max_candidates_per_frame: int | None,
) -> pd.DataFrame:
    selected_reasons: dict[int, set[str]] = {}
    if per_source_top_n > 0:
        for source, group in frame.groupby("source", sort=True):
            _mark_selected(selected_reasons, _top_ids(group, per_source_top_n), f"source:{source}")
    if per_branch_top_n > 0:
        for branch, group in frame.groupby("candidate_branch", sort=True):
            _mark_selected(selected_reasons, _top_ids(group, per_branch_top_n), f"branch:{branch}")
    if global_top_n > 0:
        _mark_selected(selected_reasons, _top_ids(frame, global_top_n), "global")
    if score_floor_quantile is not None:
        q = float(score_floor_quantile)
        if not 0.0 <= q <= 1.0:
            raise ValueError("score_floor_quantile must be in [0, 1]")
        threshold = float(np.nanquantile(frame["_reservoir_score"].to_numpy(float), q))
        floor_ids = frame.loc[frame["_reservoir_score"] >= threshold, "_candidate_row_id"].astype(int)
        _mark_selected(selected_reasons, set(floor_ids), "score_floor")
    if not selected_reasons:
        return frame.iloc[0:0].copy()

    selected = frame.loc[frame["_candidate_row_id"].astype(int).isin(selected_reasons)].copy()
    selected["reservoir_selected_by"] = [
        ";".join(sorted(selected_reasons[int(row_id)]))
        for row_id in selected["_candidate_row_id"].astype(int)
    ]
    selected = selected.sort_values(["_reservoir_score", "time_s"], ascending=[False, True])
    if max_candidates_per_frame is not None and int(max_candidates_per_frame) > 0:
        selected = selected.head(int(max_candidates_per_frame))
    return selected.reset_index(drop=True)


def _mark_selected(target: dict[int, set[str]], ids: Iterable[int], reason: str) -> None:
    for row_id in ids:
        target.setdefault(int(row_id), set()).add(str(reason))


def _candidate_score(rows: pd.DataFrame, *, score_column: str) -> pd.Series:
    for column in (score_column, "ranker_score", "confidence", "score"):
        if column in rows.columns:
            score = pd.to_numeric(rows[column], errors="coerce")
            finite = score[np.isfinite(score.to_numpy(float))]
            if not finite.empty:
                return score.fillna(float(finite.min()))
    return pd.Series(np.ones(len(rows), dtype=float), index=rows.index)


def _top_ids(rows: pd.DataFrame, n: int) -> set[int]:
    top = rows.sort_values(["_reservoir_score", "time_s"], ascending=[False, True]).head(int(n))
    return set(top["_candidate_row_id"].astype(int))


def _group_get(grouped, key) -> pd.DataFrame:
    if not hasattr(grouped, "get_group"):
        return pd.DataFrame()
    try:
        return grouped.get_group(key)
    except KeyError:
        return pd.DataFrame()


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
