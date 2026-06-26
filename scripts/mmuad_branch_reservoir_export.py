#!/usr/bin/env python
"""Truth-free branch-preserving MMUAD candidate reservoir export.

The oracle-recall diagnostic identifies which raw, dynamic, translated, or merged
candidate branches contain useful Track 5 pose candidates.  This companion tool
turns that idea into an inference-safe preprocessing step: it builds a bounded
candidate reservoir from explicit branch inputs without reading truth labels.

The exported ``mmuad_branch_reservoir_candidates.csv`` can be passed to later
tracker or mixture-MAP experiments, including hidden-test runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Iterable

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
PROVENANCE_JSON = "mmuad_branch_reservoir_export_provenance.json"


@dataclass(frozen=True)
class CandidateInput:
    branch: str
    path: Path


@dataclass(frozen=True)
class ReservoirExportConfig:
    per_source_top_n: int = 3
    per_branch_top_n: int = 3
    global_top_n: int = 20
    score_column: str = "ranker_score"
    score_floor_quantile: float | None = None
    candidate_time_bin_s: float = 0.0


def parse_candidate_input(value: str) -> CandidateInput:
    """Parse ``BRANCH=path`` or a plain path candidate argument."""

    if "=" in value:
        branch, path_text = value.split("=", 1)
        branch = _safe_label(branch) or _safe_label(Path(path_text).stem)
        return CandidateInput(branch=branch, path=Path(path_text))
    path = Path(value)
    return CandidateInput(branch=_safe_label(path.stem) or "candidate", path=path)


def load_branch_candidate_inputs(inputs: Iterable[CandidateInput]) -> pd.DataFrame:
    """Load candidate CSVs and attach stable branch labels."""

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


def build_branch_reservoir_export_tables(
    candidates: pd.DataFrame,
    *,
    config: ReservoirExportConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return retained candidates, per-frame summary, and per-branch summary."""

    config = config or ReservoirExportConfig()
    rows = _finite_candidate_rows(candidates, score_column=config.score_column)
    if rows.empty:
        empty_candidates = pd.DataFrame(columns=_candidate_columns())
        return empty_candidates, _frame_summary(empty_candidates), _branch_summary(rows, empty_candidates)

    rows = rows.copy()
    rows["reservoir_time_key_s"] = _time_key(rows["time_s"], config.candidate_time_bin_s)
    retained_frames: list[pd.DataFrame] = []
    frame_summaries: list[dict[str, Any]] = []
    group_columns = ["sequence_id", "reservoir_time_key_s"]
    for (sequence_id, time_key_s), group in rows.groupby(group_columns, sort=True, dropna=False):
        retained = _select_reservoir(
            group,
            per_source_top_n=config.per_source_top_n,
            per_branch_top_n=config.per_branch_top_n,
            global_top_n=config.global_top_n,
            score_floor_quantile=config.score_floor_quantile,
        )
        retained_frames.append(retained)
        frame_summaries.append(
            {
                "sequence_id": str(sequence_id),
                "reservoir_time_key_s": float(time_key_s),
                "time_s_min": float(pd.to_numeric(group["time_s"], errors="coerce").min()),
                "time_s_max": float(pd.to_numeric(group["time_s"], errors="coerce").max()),
                "candidate_count": int(len(group)),
                "retained_count": int(len(retained)),
                "source_count": int(group["source"].nunique(dropna=True)),
                "branch_count": int(group["candidate_branch"].nunique(dropna=True)),
                "retained_source_count": int(retained["source"].nunique(dropna=True))
                if not retained.empty
                else 0,
                "retained_branch_count": int(retained["candidate_branch"].nunique(dropna=True))
                if not retained.empty
                else 0,
            }
        )
    reservoir = (
        pd.concat(retained_frames, ignore_index=True, sort=False)
        if retained_frames
        else rows.iloc[0:0].copy()
    )
    frame_summary = pd.DataFrame.from_records(frame_summaries)
    branch_summary = _branch_summary(rows, reservoir)
    return reservoir, frame_summary, branch_summary


def write_branch_reservoir_export_artifacts(
    *,
    candidates: pd.DataFrame,
    output_dir: Path,
    config: ReservoirExportConfig,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write truth-free reservoir candidates and summaries."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reservoir, frame_summary, branch_summary = build_branch_reservoir_export_tables(
        candidates,
        config=config,
    )
    paths = {
        "reservoir_candidates_csv": output_dir / RESERVOIR_CSV,
        "frame_summary_csv": output_dir / FRAME_SUMMARY_CSV,
        "branch_summary_csv": output_dir / BRANCH_SUMMARY_CSV,
        "provenance_json": output_dir / PROVENANCE_JSON,
    }
    reservoir.to_csv(paths["reservoir_candidates_csv"], index=False)
    frame_summary.to_csv(paths["frame_summary_csv"], index=False)
    branch_summary.to_csv(paths["branch_summary_csv"], index=False)
    metadata = {
        "truth_free": True,
        "config": {
            "per_source_top_n": int(config.per_source_top_n),
            "per_branch_top_n": int(config.per_branch_top_n),
            "global_top_n": int(config.global_top_n),
            "score_column": str(config.score_column),
            "score_floor_quantile": config.score_floor_quantile,
            "candidate_time_bin_s": float(config.candidate_time_bin_s),
        },
        "input_rows": int(len(candidates)),
        "retained_rows": int(len(reservoir)),
        "frame_count": int(len(frame_summary)),
        "provenance": provenance or {},
        "reservoir_candidates_csv": str(paths["reservoir_candidates_csv"]),
        "frame_summary_csv": str(paths["frame_summary_csv"]),
        "branch_summary_csv": str(paths["branch_summary_csv"]),
    }
    paths["provenance_json"].write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    return paths


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
    parser.add_argument(
        "--candidate-time-bin-s",
        type=float,
        default=0.0,
        help="optional bin size for grouping near-synchronous candidate timestamps; 0 keeps exact times",
    )
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv BRANCH=PATH")
    inputs = tuple(parse_candidate_input(value) for value in args.candidate_csv)
    candidates = load_branch_candidate_inputs(inputs)
    config = ReservoirExportConfig(
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        global_top_n=int(args.global_top_n),
        score_column=str(args.score_column),
        score_floor_quantile=args.score_floor_quantile,
        candidate_time_bin_s=float(args.candidate_time_bin_s),
    )
    paths = write_branch_reservoir_export_artifacts(
        candidates=candidates,
        output_dir=args.output_dir,
        config=config,
        provenance={"candidate_inputs": [{"branch": item.branch, "path": str(item.path)} for item in inputs]},
    )
    print("mmuad_branch_reservoir_export=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
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


def _time_key(time_s: pd.Series, candidate_time_bin_s: float) -> pd.Series:
    time_values = pd.to_numeric(time_s, errors="coerce").astype(float)
    if candidate_time_bin_s <= 0:
        return time_values
    bin_s = float(candidate_time_bin_s)
    return np.round(time_values / bin_s) * bin_s


def _select_reservoir(
    group: pd.DataFrame,
    *,
    per_source_top_n: int,
    per_branch_top_n: int,
    global_top_n: int,
    score_floor_quantile: float | None,
) -> pd.DataFrame:
    if group.empty:
        return group.copy()
    ranked = group.copy()
    ranked["reservoir_rank_global"] = _rank_within(ranked)
    ranked["reservoir_rank_source"] = _group_rank_within(ranked, "source")
    ranked["reservoir_rank_branch"] = _group_rank_within(ranked, "candidate_branch")
    reasons: dict[int, set[str]] = {int(row_id): set() for row_id in ranked["_candidate_row_id"]}
    if global_top_n > 0:
        _mark_reasons(reasons, _top_ids(ranked, global_top_n), "global_top")
    if per_source_top_n > 0:
        for _, source_rows in ranked.groupby("source", sort=True):
            _mark_reasons(reasons, _top_ids(source_rows, per_source_top_n), "source_top")
    if per_branch_top_n > 0:
        for _, branch_rows in ranked.groupby("candidate_branch", sort=True):
            _mark_reasons(reasons, _top_ids(branch_rows, per_branch_top_n), "branch_top")
    if score_floor_quantile is not None:
        q = float(score_floor_quantile)
        if not 0.0 <= q <= 1.0:
            raise ValueError("score_floor_quantile must be in [0, 1]")
        threshold = float(np.nanquantile(ranked["_reservoir_score"].to_numpy(float), q))
        score_floor_ids = ranked.loc[ranked["_reservoir_score"] >= threshold, "_candidate_row_id"]
        _mark_reasons(reasons, score_floor_ids.astype(int), "score_floor")
    selected_ids = {row_id for row_id, row_reasons in reasons.items() if row_reasons}
    retained = ranked.loc[ranked["_candidate_row_id"].astype(int).isin(selected_ids)].copy()
    if retained.empty:
        return retained
    retained["reservoir_reason"] = [
        ";".join(sorted(reasons[int(row_id)])) for row_id in retained["_candidate_row_id"]
    ]
    retained["reservoir_input_candidate_count"] = int(len(ranked))
    retained["reservoir_retained_count"] = int(len(retained))
    return retained.sort_values(["_reservoir_score", "time_s"], ascending=[False, True]).reset_index(drop=True)


def _rank_within(rows: pd.DataFrame) -> pd.Series:
    ordered = rows.sort_values(["_reservoir_score", "time_s", "_candidate_row_id"], ascending=[False, True, True])
    ranks = pd.Series(index=ordered.index, data=np.arange(1, len(ordered) + 1, dtype=int))
    return ranks.reindex(rows.index)


def _group_rank_within(rows: pd.DataFrame, column: str) -> pd.Series:
    ranks = pd.Series(index=rows.index, dtype=int)
    for _, group in rows.groupby(column, sort=True, dropna=False):
        ranks.loc[group.index] = _rank_within(group)
    return ranks.astype(int)


def _top_ids(rows: pd.DataFrame, n: int) -> set[int]:
    if n <= 0 or rows.empty:
        return set()
    top = rows.sort_values(
        ["_reservoir_score", "time_s", "_candidate_row_id"],
        ascending=[False, True, True],
    ).head(int(n))
    return set(top["_candidate_row_id"].astype(int))


def _mark_reasons(reasons: dict[int, set[str]], row_ids: Iterable[int], reason: str) -> None:
    for row_id in row_ids:
        reasons.setdefault(int(row_id), set()).add(reason)


def _frame_summary(reservoir: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sequence_id",
        "reservoir_time_key_s",
        "time_s_min",
        "time_s_max",
        "candidate_count",
        "retained_count",
        "source_count",
        "branch_count",
        "retained_source_count",
        "retained_branch_count",
    ]
    if reservoir.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, Any]] = []
    for (sequence_id, time_key_s), group in reservoir.groupby(["sequence_id", "reservoir_time_key_s"], sort=True):
        records.append(
            {
                "sequence_id": str(sequence_id),
                "reservoir_time_key_s": float(time_key_s),
                "time_s_min": float(pd.to_numeric(group["time_s"], errors="coerce").min()),
                "time_s_max": float(pd.to_numeric(group["time_s"], errors="coerce").max()),
                "candidate_count": int(group["reservoir_input_candidate_count"].max()),
                "retained_count": int(len(group)),
                "source_count": int(group["source"].nunique(dropna=True)),
                "branch_count": int(group["candidate_branch"].nunique(dropna=True)),
                "retained_source_count": int(group["source"].nunique(dropna=True)),
                "retained_branch_count": int(group["candidate_branch"].nunique(dropna=True)),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _branch_summary(input_rows: pd.DataFrame, reservoir: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_branch",
        "source",
        "input_count",
        "retained_count",
        "retained_fraction",
        "mean_score_input",
        "mean_score_retained",
    ]
    if input_rows.empty:
        return pd.DataFrame(columns=columns)
    input_summary = (
        input_rows.groupby(["candidate_branch", "source"], dropna=False)["_reservoir_score"]
        .agg(input_count="size", mean_score_input="mean")
        .reset_index()
    )
    if reservoir.empty:
        input_summary["retained_count"] = 0
        input_summary["mean_score_retained"] = np.nan
    else:
        retained_summary = (
            reservoir.groupby(["candidate_branch", "source"], dropna=False)["_reservoir_score"]
            .agg(retained_count="size", mean_score_retained="mean")
            .reset_index()
        )
        input_summary = input_summary.merge(
            retained_summary,
            on=["candidate_branch", "source"],
            how="left",
        )
        input_summary["retained_count"] = input_summary["retained_count"].fillna(0).astype(int)
    input_summary["retained_fraction"] = input_summary["retained_count"] / input_summary["input_count"]
    return input_summary[columns]


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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
