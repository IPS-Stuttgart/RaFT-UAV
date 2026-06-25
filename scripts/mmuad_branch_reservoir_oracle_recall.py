#!/usr/bin/env python
"""Build branch-preserving MMUAD candidate reservoirs and oracle-recall tables.

This experiment runner is intended for MMUAD top-3 development.  It keeps
multiple candidate streams (for example raw/static, dynamic, source-translated,
and cross-sensor-merged candidates) as explicit branches, then reports whether a
bounded per-frame reservoir still contains a candidate close to the official
truth timestamps.  The diagnostic is truth-aware and must not be used for hidden
or validation inference; it is meant to decide whether the next bottleneck is
candidate generation/retention or the trajectory smoother.
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

from raft_uav.mmuad.evaluator import load_evaluation_truth_file  # noqa: E402
from raft_uav.mmuad.io import load_candidate_file  # noqa: E402
from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns  # noqa: E402


FRAME_ROWS_CSV = "mmuad_branch_reservoir_oracle_frame_rows.csv"
POOLED_CSV = "mmuad_branch_reservoir_oracle_pooled.csv"
BY_SEQUENCE_CSV = "mmuad_branch_reservoir_oracle_by_sequence.csv"
RESERVOIR_CSV = "mmuad_branch_reservoir_candidates.csv"
PROVENANCE_JSON = "mmuad_branch_reservoir_oracle_provenance.json"


@dataclass(frozen=True)
class CandidateInput:
    branch: str
    path: Path


def parse_candidate_input(value: str) -> CandidateInput:
    """Parse ``BRANCH=path`` or plain path candidate arguments."""

    if "=" in value:
        branch, path_text = value.split("=", 1)
        branch = _safe_label(branch) or _safe_label(Path(path_text).stem)
        return CandidateInput(branch=branch, path=Path(path_text))
    path = Path(value)
    return CandidateInput(branch=_safe_label(path.stem) or "candidate", path=path)


def build_branch_reservoir_oracle_tables(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 0.5,
    top_k_values: Iterable[int] = (1, 3, 5, 10, 20),
    per_source_top_n: int = 3,
    per_branch_top_n: int = 3,
    global_top_n: int = 20,
    score_column: str = "ranker_score",
    score_floor_quantile: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return frame, pooled, by-sequence, and retained-candidate tables."""

    candidate_rows = _finite_candidate_rows(candidates)
    truth_rows = _finite_truth_rows(truth)
    top_ks = tuple(sorted({int(k) for k in top_k_values if int(k) > 0}))
    if not top_ks:
        raise ValueError("at least one positive top-K value is required")
    if truth_rows.empty:
        empty_frames = pd.DataFrame(columns=_frame_columns())
        return empty_frames, _summarize(empty_frames, by_sequence=False), _summarize(empty_frames, by_sequence=True), candidate_rows.iloc[0:0].copy()

    frame_records: list[dict[str, Any]] = []
    reservoir_frames: list[pd.DataFrame] = []
    for sequence_id, truth_group in truth_rows.groupby("sequence_id", sort=True):
        sequence_candidates = candidate_rows.loc[candidate_rows["sequence_id"] == sequence_id]
        sequence_candidates = sequence_candidates.sort_values(["time_s", "_reservoir_score"], ascending=[True, False])
        for _, truth_row in truth_group.sort_values("time_s").iterrows():
            nearby = _nearby_candidates(
                sequence_candidates,
                time_s=float(truth_row["time_s"]),
                max_time_delta_s=max_time_delta_s,
            )
            reservoir = _select_reservoir(
                nearby,
                per_source_top_n=per_source_top_n,
                per_branch_top_n=per_branch_top_n,
                global_top_n=global_top_n,
                score_floor_quantile=score_floor_quantile,
            )
            if not reservoir.empty:
                reservoir = reservoir.copy()
                reservoir["truth_sequence_id"] = str(truth_row["sequence_id"])
                reservoir["truth_time_s"] = float(truth_row["time_s"])
                reservoir["truth_x_m"] = float(truth_row["x_m"])
                reservoir["truth_y_m"] = float(truth_row["y_m"])
                reservoir["truth_z_m"] = float(truth_row["z_m"])
                reservoir_frames.append(reservoir)
            frame_records.extend(
                _frame_records_for_truth(
                    truth_row,
                    nearby=nearby,
                    reservoir=reservoir,
                    top_ks=top_ks,
                    score_column=score_column,
                )
            )

    frame_rows = pd.DataFrame.from_records(frame_records, columns=_frame_columns())
    pooled = _summarize(frame_rows, by_sequence=False)
    by_sequence = _summarize(frame_rows, by_sequence=True)
    reservoir_rows = (
        pd.concat(reservoir_frames, ignore_index=True, sort=False)
        if reservoir_frames
        else candidate_rows.iloc[0:0].copy()
    )
    return frame_rows, pooled, by_sequence, reservoir_rows


def load_branch_candidate_inputs(inputs: Iterable[CandidateInput]) -> pd.DataFrame:
    """Load and concatenate normalized candidate rows with branch labels."""

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-file", type=Path, required=True)
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        metavar="BRANCH=PATH",
        help="candidate CSV to include; may be repeated; plain PATH uses file stem as branch",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-time-delta-s", type=float, default=0.5)
    parser.add_argument("--top-k", default="1,3,5,10,20")
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--write-reservoir-candidates", action="store_true")
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv BRANCH=PATH")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = tuple(parse_candidate_input(value) for value in args.candidate_csv)
    candidates = load_branch_candidate_inputs(inputs)
    truth = load_evaluation_truth_file(args.truth_file).rows
    top_ks = _parse_top_k(args.top_k)
    frame_rows, pooled, by_sequence, reservoir = build_branch_reservoir_oracle_tables(
        candidates,
        truth,
        max_time_delta_s=float(args.max_time_delta_s),
        top_k_values=top_ks,
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        global_top_n=int(args.global_top_n),
        score_column=str(args.score_column),
        score_floor_quantile=args.score_floor_quantile,
    )

    frame_path = args.output_dir / FRAME_ROWS_CSV
    pooled_path = args.output_dir / POOLED_CSV
    by_sequence_path = args.output_dir / BY_SEQUENCE_CSV
    provenance_path = args.output_dir / PROVENANCE_JSON
    frame_rows.to_csv(frame_path, index=False)
    pooled.to_csv(pooled_path, index=False)
    by_sequence.to_csv(by_sequence_path, index=False)
    if args.write_reservoir_candidates:
        reservoir.to_csv(args.output_dir / RESERVOIR_CSV, index=False)
    provenance = {
        "truth_file": str(args.truth_file),
        "candidate_inputs": [{"branch": item.branch, "path": str(item.path)} for item in inputs],
        "max_time_delta_s": float(args.max_time_delta_s),
        "top_k": list(top_ks),
        "per_source_top_n": int(args.per_source_top_n),
        "per_branch_top_n": int(args.per_branch_top_n),
        "global_top_n": int(args.global_top_n),
        "score_column": str(args.score_column),
        "score_floor_quantile": args.score_floor_quantile,
        "frame_rows_csv": str(frame_path),
        "pooled_csv": str(pooled_path),
        "by_sequence_csv": str(by_sequence_path),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print("mmuad_branch_reservoir_oracle_recall=ok")
    print(f"frame_rows_csv={frame_path}")
    print(f"pooled_csv={pooled_path}")
    print(f"by_sequence_csv={by_sequence_path}")
    if not pooled.empty:
        best = pooled.sort_values(["oracle_mse_m2", "top_k"], na_position="last").iloc[0]
        print(f"best_top_k={best['top_k']}")
        print(f"best_oracle_mse_m2={best['oracle_mse_m2']}")
    return 0


def _finite_candidate_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = normalize_candidate_columns(pd.DataFrame(candidates)).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s", "source", "track_id", "x_m", "y_m", "z_m", "confidence", "candidate_branch", "_reservoir_score"])
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
    score = _candidate_score(rows)
    rows["_reservoir_score"] = score
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    rows = rows.loc[finite].copy().reset_index(drop=True)
    rows["_candidate_row_id"] = np.arange(len(rows), dtype=int)
    return rows


def _finite_truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    rows = normalize_truth_columns(pd.DataFrame(truth)).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s", "x_m", "y_m", "z_m"])
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _candidate_score(rows: pd.DataFrame) -> pd.Series:
    for column in ("ranker_score", "confidence", "score"):
        if column in rows.columns:
            score = pd.to_numeric(rows[column], errors="coerce")
            if np.isfinite(score.to_numpy(float)).any():
                return score.fillna(float(np.nanmin(score.to_numpy(float))))
    return pd.Series(np.ones(len(rows), dtype=float), index=rows.index)


def _nearby_candidates(rows: pd.DataFrame, *, time_s: float, max_time_delta_s: float) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    deltas = (pd.to_numeric(rows["time_s"], errors="coerce") - float(time_s)).abs()
    finite = np.isfinite(deltas.to_numpy(float))
    nearby = rows.loc[finite & (deltas <= float(max_time_delta_s))].copy()
    if nearby.empty:
        return nearby
    nearby["truth_time_delta_s"] = pd.to_numeric(nearby["time_s"], errors="coerce") - float(time_s)
    return nearby.sort_values(["_reservoir_score", "time_s"], ascending=[False, True])


def _select_reservoir(
    nearby: pd.DataFrame,
    *,
    per_source_top_n: int,
    per_branch_top_n: int,
    global_top_n: int,
    score_floor_quantile: float | None,
) -> pd.DataFrame:
    if nearby.empty:
        return nearby.copy()
    selected_ids: set[int] = set()
    if per_source_top_n > 0 and "source" in nearby.columns:
        for _, group in nearby.groupby("source", sort=True):
            selected_ids.update(_top_ids(group, per_source_top_n))
    if per_branch_top_n > 0 and "candidate_branch" in nearby.columns:
        for _, group in nearby.groupby("candidate_branch", sort=True):
            selected_ids.update(_top_ids(group, per_branch_top_n))
    if global_top_n > 0:
        selected_ids.update(_top_ids(nearby, global_top_n))
    if score_floor_quantile is not None:
        q = float(score_floor_quantile)
        if not 0.0 <= q <= 1.0:
            raise ValueError("score_floor_quantile must be in [0, 1]")
        threshold = float(np.nanquantile(nearby["_reservoir_score"].to_numpy(float), q))
        selected_ids.update(nearby.loc[nearby["_reservoir_score"] >= threshold, "_candidate_row_id"].astype(int))
    reservoir = nearby.loc[nearby["_candidate_row_id"].astype(int).isin(selected_ids)].copy()
    return reservoir.sort_values(["_reservoir_score", "time_s"], ascending=[False, True]).reset_index(drop=True)


def _top_ids(rows: pd.DataFrame, n: int) -> set[int]:
    if rows.empty or n <= 0:
        return set()
    top = rows.sort_values(["_reservoir_score", "time_s"], ascending=[False, True]).head(int(n))
    return set(top["_candidate_row_id"].astype(int))


def _frame_records_for_truth(
    truth_row: pd.Series,
    *,
    nearby: pd.DataFrame,
    reservoir: pd.DataFrame,
    top_ks: tuple[int, ...],
    score_column: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for top_k in top_ks:
        subset = reservoir.head(top_k)
        records.append(_frame_record(truth_row, nearby=nearby, reservoir=reservoir, subset=subset, top_k=top_k, score_column=score_column))
    records.append(_frame_record(truth_row, nearby=nearby, reservoir=reservoir, subset=reservoir, top_k="all", score_column=score_column))
    return records


def _frame_record(
    truth_row: pd.Series,
    *,
    nearby: pd.DataFrame,
    reservoir: pd.DataFrame,
    subset: pd.DataFrame,
    top_k: int | str,
    score_column: str,
) -> dict[str, Any]:
    best = _best_candidate_to_truth(subset, truth_row)
    error = _candidate_error(best, truth_row)
    return {
        "sequence_id": str(truth_row["sequence_id"]),
        "time_s": float(truth_row["time_s"]),
        "top_k": top_k,
        "candidate_count_window": int(len(nearby)),
        "reservoir_count": int(len(reservoir)),
        "oracle_candidate_found": best is not None,
        "oracle_error_m": error,
        "oracle_squared_error_m2": float(error * error) if np.isfinite(error) else np.nan,
        "oracle_candidate_source": _candidate_text(best, "source"),
        "oracle_candidate_branch": _candidate_text(best, "candidate_branch"),
        "oracle_candidate_track_id": _candidate_text(best, "track_id"),
        "oracle_candidate_score": _candidate_value(best, "_reservoir_score"),
        "score_column": score_column,
    }


def _best_candidate_to_truth(rows: pd.DataFrame, truth_row: pd.Series) -> pd.Series | None:
    if rows.empty:
        return None
    truth_xyz = truth_row[["x_m", "y_m", "z_m"]].to_numpy(float)
    candidate_xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(float)
    distances = np.linalg.norm(candidate_xyz - truth_xyz, axis=1)
    if not np.isfinite(distances).any():
        return None
    return rows.iloc[int(np.nanargmin(distances))]


def _candidate_error(row: pd.Series | None, truth_row: pd.Series) -> float:
    if row is None:
        return np.nan
    truth_xyz = truth_row[["x_m", "y_m", "z_m"]].to_numpy(float)
    candidate_xyz = row[["x_m", "y_m", "z_m"]].to_numpy(float)
    return float(np.linalg.norm(candidate_xyz - truth_xyz))


def _candidate_text(row: pd.Series | None, column: str) -> str:
    if row is None or column not in row.index or pd.isna(row[column]):
        return ""
    return str(row[column])


def _candidate_value(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row.index:
        return np.nan
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else np.nan


def _summarize(frame_rows: pd.DataFrame, *, by_sequence: bool) -> pd.DataFrame:
    if frame_rows.empty:
        columns = ["top_k", "frame_count", "oracle_mse_m2", "oracle_rmse_m", "oracle_mean_m", "oracle_p95_m", "oracle_max_m"]
        if by_sequence:
            columns.insert(0, "sequence_id")
        return pd.DataFrame(columns=columns)
    keys = ["top_k"]
    if by_sequence:
        keys = ["sequence_id", "top_k"]
    records: list[dict[str, Any]] = []
    for group_key, group in frame_rows.groupby(keys, sort=True, dropna=False):
        errors = pd.to_numeric(group["oracle_error_m"], errors="coerce")
        errors = errors[np.isfinite(errors.to_numpy(float))]
        record: dict[str, Any] = {}
        if by_sequence:
            sequence_id, top_k = group_key if isinstance(group_key, tuple) else ("", group_key)
            record["sequence_id"] = sequence_id
            record["top_k"] = top_k
        else:
            record["top_k"] = group_key
        record["frame_count"] = int(len(group))
        record["candidate_found_count"] = int(group["oracle_candidate_found"].astype(bool).sum())
        record["mean_reservoir_count"] = float(pd.to_numeric(group["reservoir_count"], errors="coerce").mean())
        if errors.empty:
            record.update(
                {
                    "oracle_mse_m2": np.nan,
                    "oracle_rmse_m": np.nan,
                    "oracle_mean_m": np.nan,
                    "oracle_p50_m": np.nan,
                    "oracle_p95_m": np.nan,
                    "oracle_max_m": np.nan,
                }
            )
        else:
            squared = errors.to_numpy(float) ** 2
            record.update(
                {
                    "oracle_mse_m2": float(np.mean(squared)),
                    "oracle_rmse_m": float(np.sqrt(np.mean(squared))),
                    "oracle_mean_m": float(errors.mean()),
                    "oracle_p50_m": float(np.percentile(errors, 50)),
                    "oracle_p95_m": float(np.percentile(errors, 95)),
                    "oracle_max_m": float(errors.max()),
                }
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _frame_columns() -> list[str]:
    return [
        "sequence_id",
        "time_s",
        "top_k",
        "candidate_count_window",
        "reservoir_count",
        "oracle_candidate_found",
        "oracle_error_m",
        "oracle_squared_error_m2",
        "oracle_candidate_source",
        "oracle_candidate_branch",
        "oracle_candidate_track_id",
        "oracle_candidate_score",
        "score_column",
    ]


def _parse_top_k(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return tuple(values)


def _safe_label(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return text


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
