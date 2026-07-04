"""Attribute MMUAD candidate-oracle wins to sources and branches.

The branch-preserving reservoir and score-offset grid help keep candidate
hypotheses alive. This module answers the next diagnostic question: when the
candidate pool contains a good oracle candidate, which source/branch supplied it
and how deeply was it buried by the current score?  The resulting tables are
useful before running expensive mixture-MAP sweeps over large candidate pools.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import load_candidate_inputs
from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns

_DEFAULT_TOP_K = (1, 3, 5, 10, 20)


def build_candidate_oracle_attribution_tables(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    top_k_values: Sequence[int] = _DEFAULT_TOP_K,
    score_column: str = "candidate_reservoir_score",
    fallback_score_column: str = "ranker_score",
    max_truth_time_delta_s: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return frame, pooled, branch, and source oracle-attribution tables."""

    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    if rows.empty or truth_rows.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    rows = rows.copy()
    if "source" not in rows.columns:
        rows["source"] = "unknown"
    if "candidate_branch" not in rows.columns:
        rows["candidate_branch"] = rows["source"].fillna("candidate").astype(str)
    if "track_id" not in rows.columns:
        rows["track_id"] = np.arange(len(rows), dtype=int).astype(str)
    rows["candidate_oracle_score"] = _score_column(
        rows,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
    )
    top_k_tuple = tuple(sorted({int(value) for value in top_k_values if int(value) > 0}))
    truth_by_sequence = {
        str(sequence_id): group.sort_values("time_s").reset_index(drop=True)
        for sequence_id, group in truth_rows.groupby("sequence_id", sort=True)
    }
    frame_records: list[dict[str, Any]] = []
    for (sequence_id, time_s), group in rows.groupby(["sequence_id", "time_s"], sort=True):
        seq_truth = truth_by_sequence.get(str(sequence_id))
        if seq_truth is None or seq_truth.empty:
            continue
        truth_t = seq_truth["time_s"].to_numpy(float)
        nearest_idx = int(np.argmin(np.abs(truth_t - float(time_s))))
        truth_dt = float(time_s) - float(truth_t[nearest_idx])
        if abs(truth_dt) > float(max_truth_time_delta_s):
            continue
        truth_xyz = seq_truth.iloc[nearest_idx][["x_m", "y_m", "z_m"]].to_numpy(float)
        ranked = group.sort_values("candidate_oracle_score", ascending=False).reset_index(drop=True)
        candidate_xyz = ranked[["x_m", "y_m", "z_m"]].to_numpy(float)
        distances = np.linalg.norm(candidate_xyz - truth_xyz, axis=1)
        best_pos = int(np.argmin(distances))
        best_row = ranked.iloc[best_pos]
        record: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "time_s": float(time_s),
            "truth_time_delta_s": truth_dt,
            "candidate_count": int(len(ranked)),
            "oracle_all_3d_m": float(distances[best_pos]),
            "oracle_all_rank": int(best_pos + 1),
            "oracle_all_rank_fraction": float((best_pos + 1) / max(len(ranked), 1)),
            "oracle_all_candidate_score": float(best_row["candidate_oracle_score"]),
            "oracle_all_candidate_source": str(best_row.get("source", "unknown")),
            "oracle_all_candidate_branch": str(best_row.get("candidate_branch", "candidate")),
            "oracle_all_candidate_track_id": str(best_row.get("track_id", "")),
        }
        for top_k in top_k_tuple:
            bounded_k = min(int(top_k), len(distances))
            top_distances = distances[:bounded_k]
            top_best_pos = int(np.argmin(top_distances))
            top_row = ranked.iloc[top_best_pos]
            record[f"oracle_top{top_k}_3d_m"] = float(top_distances[top_best_pos])
            record[f"oracle_in_top{top_k}"] = bool(best_pos < bounded_k)
            record[f"oracle_top{top_k}_candidate_source"] = str(top_row.get("source", "unknown"))
            record[f"oracle_top{top_k}_candidate_branch"] = str(
                top_row.get("candidate_branch", "candidate"),
            )
        frame_records.append(record)
    frame_rows = pd.DataFrame.from_records(frame_records)
    if frame_rows.empty:
        empty = pd.DataFrame()
        return frame_rows, empty, empty, empty
    pooled = pd.DataFrame.from_records([_pooled_summary(frame_rows, top_k_values=top_k_tuple)])
    branch_summary = _group_summary(
        frame_rows,
        group_column="oracle_all_candidate_branch",
        label_column="candidate_branch",
    )
    source_summary = _group_summary(
        frame_rows,
        group_column="oracle_all_candidate_source",
        label_column="source",
    )
    return frame_rows, pooled, branch_summary, source_summary


def write_candidate_oracle_attribution_outputs(
    *,
    output_dir: Path,
    frame_rows: pd.DataFrame,
    pooled_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    source_summary: pd.DataFrame,
) -> dict[str, str]:
    """Write oracle-attribution artifacts and return their paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "frame_csv": output_dir / "mmuad_candidate_oracle_attribution_frames.csv",
        "pooled_csv": output_dir / "mmuad_candidate_oracle_attribution_pooled.csv",
        "branch_csv": output_dir / "mmuad_candidate_oracle_attribution_by_branch.csv",
        "source_csv": output_dir / "mmuad_candidate_oracle_attribution_by_source.csv",
        "summary_json": output_dir / "mmuad_candidate_oracle_attribution_summary.json",
    }
    frame_rows.to_csv(paths["frame_csv"], index=False)
    pooled_summary.to_csv(paths["pooled_csv"], index=False)
    branch_summary.to_csv(paths["branch_csv"], index=False)
    source_summary.to_csv(paths["source_csv"], index=False)
    summary_json = {
        "pooled": pooled_summary.to_dict(orient="records"),
        "by_branch": branch_summary.to_dict(orient="records"),
        "by_source": source_summary.to_dict(orient="records"),
    }
    paths["summary_json"].write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-oracle-attribution",
        description="attribute candidate-oracle wins to MMUAD sources and branches",
    )
    parser.add_argument("--candidate", action="append", default=[], help="candidate CSV as BRANCH=path")
    parser.add_argument("--candidate-csv", action="append", default=[], help="alias for --candidate")
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-column", default="candidate_reservoir_score")
    parser.add_argument("--fallback-score-column", default="ranker_score")
    parser.add_argument("--top-k", action="append", type=int, default=None)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    candidate_specs = [*args.candidate, *args.candidate_csv]
    candidates = load_candidate_inputs(candidate_specs)
    if candidates.empty:
        raise ValueError("at least one non-empty --candidate BRANCH=PATH CSV is required")
    truth = pd.read_csv(args.truth_csv)
    frame_rows, pooled, branch_summary, source_summary = build_candidate_oracle_attribution_tables(
        candidates,
        truth,
        top_k_values=tuple(args.top_k) if args.top_k is not None else _DEFAULT_TOP_K,
        score_column=args.score_column,
        fallback_score_column=args.fallback_score_column,
        max_truth_time_delta_s=args.max_truth_time_delta_s,
    )
    paths = write_candidate_oracle_attribution_outputs(
        output_dir=args.output_dir,
        frame_rows=frame_rows,
        pooled_summary=pooled,
        branch_summary=branch_summary,
        source_summary=source_summary,
    )
    print("mmuad_candidate_oracle_attribution=ok")
    print(f"frame_count={len(frame_rows)}")
    if not pooled.empty:
        print(f"oracle_all_mse={pooled.loc[0, 'oracle_all_3d_m_mse']}")
        print(f"oracle_rank_p50={pooled.loc[0, 'oracle_all_rank_p50']}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _score_column(
    rows: pd.DataFrame,
    *,
    score_column: str,
    fallback_score_column: str,
) -> pd.Series:
    primary = _numeric_column(rows, score_column, default=np.nan)
    fallback = _numeric_column(rows, fallback_score_column, default=np.nan)
    confidence = _numeric_column(rows, "confidence", default=1.0)
    return primary.fillna(fallback).fillna(confidence).fillna(0.0).astype(float)


def _numeric_column(rows: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce")
    return pd.Series(default, index=rows.index, dtype=float)


def _pooled_summary(frame_rows: pd.DataFrame, *, top_k_values: tuple[int, ...]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "frame_count": int(len(frame_rows)),
        "candidate_count_mean": float(pd.to_numeric(frame_rows["candidate_count"]).mean()),
        "oracle_all_rank_mean": float(pd.to_numeric(frame_rows["oracle_all_rank"]).mean()),
        "oracle_all_rank_p50": float(pd.to_numeric(frame_rows["oracle_all_rank"]).quantile(0.5)),
        "oracle_all_rank_p95": float(pd.to_numeric(frame_rows["oracle_all_rank"]).quantile(0.95)),
    }
    _add_error_stats(record, frame_rows, "oracle_all_3d_m")
    for top_k in top_k_values:
        _add_error_stats(record, frame_rows, f"oracle_top{top_k}_3d_m")
        column = f"oracle_in_top{top_k}"
        if column in frame_rows.columns:
            record[f"oracle_in_top{top_k}_fraction"] = float(frame_rows[column].mean())
    return record


def _group_summary(frame_rows: pd.DataFrame, *, group_column: str, label_column: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for label, group in frame_rows.groupby(group_column, sort=True, dropna=False):
        record: dict[str, Any] = {
            label_column: str(label),
            "winning_frames": int(len(group)),
            "winning_frame_fraction": float(len(group) / max(len(frame_rows), 1)),
            "oracle_rank_mean": float(pd.to_numeric(group["oracle_all_rank"]).mean()),
            "oracle_rank_p50": float(pd.to_numeric(group["oracle_all_rank"]).quantile(0.5)),
        }
        _add_error_stats(record, group, "oracle_all_3d_m")
        records.append(record)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).sort_values(
        ["winning_frames", label_column],
        ascending=[False, True],
    ).reset_index(drop=True)


def _add_error_stats(record: dict[str, Any], rows: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    if values.empty:
        record[f"{column}_mse"] = float("nan")
        record[f"{column}_rmse"] = float("nan")
        record[f"{column}_p95"] = float("nan")
        record[f"{column}_max"] = float("nan")
        return
    arr = values.to_numpy(float)
    mse = float(np.mean(arr**2))
    record[f"{column}_mse"] = mse
    record[f"{column}_rmse"] = float(np.sqrt(mse))
    record[f"{column}_p95"] = float(values.quantile(0.95))
    record[f"{column}_max"] = float(values.max())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
