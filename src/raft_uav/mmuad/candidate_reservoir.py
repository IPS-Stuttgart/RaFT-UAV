"""Branch-preserving candidate reservoir utilities for MMUAD experiments.

The current MMUAD pose gap is often caused by early candidate pruning: a single
ranker score may bury useful raw or calibrated candidates before the trajectory
smoother can use them. This module builds a conservative per-frame reservoir
that keeps a global top-N plus top candidates per source and per branch. It can
also write oracle-recall diagnostics when a validation/reference file is
available.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns


@dataclass(frozen=True)
class ReservoirConfig:
    """Configuration for branch-preserving MMUAD candidate reservoirs."""

    global_top_n: int = 20
    per_source_top_n: int = 3
    per_branch_top_n: int = 3
    max_candidates_per_frame: int = 40
    score_column: str = "ranker_score"
    fallback_score_column: str = "confidence"
    score_floor_quantile: float | None = None


def build_candidate_reservoir(
    candidates: pd.DataFrame,
    *,
    config: ReservoirConfig | None = None,
) -> pd.DataFrame:
    """Return a branch/source-aware per-frame candidate reservoir.

    The reservoir keeps the union of global top-N candidates, top-N candidates
    per source, top-N candidates per candidate branch, and optional score-floor
    candidates. This preserves low-ranked candidates from raw/dynamic/calibrated
    branches while still bounding per-frame candidate count for mixture-MAP
    experiments.
    """

    config = config or ReservoirConfig()
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        return rows.assign(
            candidate_branch=pd.Series(dtype=str),
            candidate_reservoir_score=pd.Series(dtype=float),
            candidate_reservoir_rank=pd.Series(dtype=float),
            candidate_reservoir_reason=pd.Series(dtype=str),
        )
    rows = rows.copy().reset_index(drop=True)
    if "candidate_branch" not in rows.columns:
        rows["candidate_branch"] = rows["source"].fillna("candidate").astype(str)
    rows["candidate_branch"] = rows["candidate_branch"].fillna("candidate").astype(str)
    rows["_candidate_original_row"] = np.arange(len(rows), dtype=int)
    rows["candidate_reservoir_score"] = _candidate_score(rows, config=config)

    selected_indices: set[int] = set()
    reasons: dict[int, set[str]] = {}
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=False):
        frame = frame.copy()
        _add_selected(
            frame,
            selected_indices=selected_indices,
            reasons=reasons,
            count=config.global_top_n,
            reason="global_top_n",
        )
        if config.per_source_top_n > 0:
            for source, group in frame.groupby("source", sort=False):
                _add_selected(
                    group,
                    selected_indices=selected_indices,
                    reasons=reasons,
                    count=config.per_source_top_n,
                    reason=f"source:{source}",
                )
        if config.per_branch_top_n > 0:
            for branch, group in frame.groupby("candidate_branch", sort=False):
                _add_selected(
                    group,
                    selected_indices=selected_indices,
                    reasons=reasons,
                    count=config.per_branch_top_n,
                    reason=f"branch:{branch}",
                )
        if config.score_floor_quantile is not None:
            quantile = float(np.clip(config.score_floor_quantile, 0.0, 1.0))
            floor = float(frame["candidate_reservoir_score"].quantile(quantile))
            floor_rows = frame.loc[frame["candidate_reservoir_score"] >= floor]
            _add_selected(
                floor_rows,
                selected_indices=selected_indices,
                reasons=reasons,
                count=len(floor_rows),
                reason=f"score_floor_q{quantile:g}",
            )

    if not selected_indices:
        return rows.iloc[0:0].drop(columns=["_candidate_original_row"], errors="ignore")
    out = rows.loc[sorted(selected_indices)].copy()
    out["candidate_reservoir_reason"] = [
        ";".join(sorted(reasons.get(int(row_id), set())))
        for row_id in out["_candidate_original_row"]
    ]
    out = _cap_per_frame(out, max_candidates_per_frame=config.max_candidates_per_frame)
    out = out.sort_values(
        ["sequence_id", "time_s", "candidate_reservoir_rank", "source"],
    ).reset_index(drop=True)
    return out.drop(columns=["_candidate_original_row"], errors="ignore")


def build_oracle_recall_tables(
    reservoir: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    top_k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
    max_truth_time_delta_s: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return frame rows plus pooled and per-sequence oracle recall tables."""

    rows = normalize_candidate_columns(pd.DataFrame(reservoir).copy())
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    if rows.empty or truth_rows.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    if "candidate_reservoir_score" not in rows.columns:
        rows["candidate_reservoir_score"] = _candidate_score(rows, config=ReservoirConfig())
    rows["candidate_reservoir_score"] = pd.to_numeric(
        rows["candidate_reservoir_score"],
        errors="coerce",
    ).fillna(float("-inf"))
    top_k_values = tuple(sorted({int(value) for value in top_k_values if int(value) > 0}))
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
        ranked = group.sort_values(["candidate_reservoir_score"], ascending=[False]).reset_index(
            drop=True,
        )
        distances = np.linalg.norm(
            ranked[["x_m", "y_m", "z_m"]].to_numpy(float) - truth_xyz,
            axis=1,
        )
        record: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "time_s": float(time_s),
            "candidate_count": int(len(ranked)),
            "truth_time_delta_s": truth_dt,
            "oracle_all_3d_m": float(np.min(distances)),
        }
        for top_k in top_k_values:
            bounded_k = min(int(top_k), len(distances))
            record[f"oracle_top{top_k}_3d_m"] = float(np.min(distances[:bounded_k]))
        frame_records.append(record)
    frame_rows = pd.DataFrame.from_records(frame_records)
    if frame_rows.empty:
        empty = pd.DataFrame()
        return frame_rows, empty, empty
    pooled = _oracle_summary(frame_rows, sequence_id="__pooled__", top_k_values=top_k_values)
    by_sequence = pd.DataFrame.from_records(
        [
            _oracle_summary(group, sequence_id=str(sequence_id), top_k_values=top_k_values)
            for sequence_id, group in frame_rows.groupby("sequence_id", sort=True)
        ],
    )
    return frame_rows, pd.DataFrame.from_records([pooled]), by_sequence


def _candidate_score(rows: pd.DataFrame, *, config: ReservoirConfig) -> pd.Series:
    primary = _numeric_column(rows, config.score_column, default=np.nan)
    fallback = _numeric_column(rows, config.fallback_score_column, default=1.0)
    return primary.fillna(fallback).fillna(0.0).astype(float)


def _numeric_column(rows: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce")
    return pd.Series(default, index=rows.index, dtype=float)


def _add_selected(
    frame: pd.DataFrame,
    *,
    selected_indices: set[int],
    reasons: dict[int, set[str]],
    count: int,
    reason: str,
) -> None:
    if count <= 0 or frame.empty:
        return
    ranked = frame.sort_values("candidate_reservoir_score", ascending=False).head(int(count))
    for row_id in ranked["_candidate_original_row"].astype(int):
        selected_indices.add(int(row_id))
        reasons.setdefault(int(row_id), set()).add(str(reason))


def _cap_per_frame(rows: pd.DataFrame, *, max_candidates_per_frame: int) -> pd.DataFrame:
    if max_candidates_per_frame <= 0 or rows.empty:
        out = rows.copy()
        out["candidate_reservoir_rank"] = 1.0
        return out
    parts: list[pd.DataFrame] = []
    for _, group in rows.groupby(["sequence_id", "time_s"], sort=False):
        capped = group.sort_values("candidate_reservoir_score", ascending=False).head(
            int(max_candidates_per_frame),
        ).copy()
        capped["candidate_reservoir_rank"] = np.arange(1, len(capped) + 1, dtype=float)
        parts.append(capped)
    return pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0].copy()


def _oracle_summary(
    frame_rows: pd.DataFrame,
    *,
    sequence_id: str,
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sequence_id": sequence_id,
        "frame_count": int(len(frame_rows)),
        "candidate_count_mean": float(pd.to_numeric(frame_rows["candidate_count"]).mean()),
    }
    for column in ["oracle_all_3d_m"] + [f"oracle_top{k}_3d_m" for k in top_k_values]:
        values = pd.to_numeric(frame_rows[column], errors="coerce").dropna()
        if values.empty:
            record[f"{column}_mse"] = float("nan")
            record[f"{column}_rmse"] = float("nan")
            record[f"{column}_p95"] = float("nan")
            record[f"{column}_max"] = float("nan")
            continue
        record[f"{column}_mse"] = float(np.mean(values.to_numpy(float) ** 2))
        record[f"{column}_rmse"] = float(np.sqrt(record[f"{column}_mse"]))
        record[f"{column}_p95"] = float(values.quantile(0.95))
        record[f"{column}_max"] = float(values.max())
    return record


def _load_candidate_specs(specs: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in specs:
        if "=" in spec:
            branch, path_text = spec.split("=", 1)
        else:
            path_text = spec
            branch = Path(spec).stem
        rows = pd.read_csv(Path(path_text))
        rows = normalize_candidate_columns(rows)
        rows["candidate_branch"] = str(branch)
        frames.append(rows)
    if not frames:
        raise ValueError("at least one --candidate BRANCH=PATH entry is required")
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-reservoir",
        description="build branch-preserving MMUAD candidate reservoirs",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="candidate CSV as BRANCH=path; may be repeated",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--oracle-frame-csv", type=Path)
    parser.add_argument("--oracle-summary-csv", type=Path)
    parser.add_argument("--oracle-by-sequence-csv", type=Path)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--fallback-score-column", default="confidence")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--top-k", type=int, action="append", default=[1, 3, 5, 10, 20])
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    candidates = _load_candidate_specs(list(args.candidate))
    reservoir = build_candidate_reservoir(
        candidates,
        config=ReservoirConfig(
            global_top_n=args.global_top_n,
            per_source_top_n=args.per_source_top_n,
            per_branch_top_n=args.per_branch_top_n,
            max_candidates_per_frame=args.max_candidates_per_frame,
            score_column=args.score_column,
            fallback_score_column=args.fallback_score_column,
            score_floor_quantile=args.score_floor_quantile,
        ),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    reservoir.to_csv(args.output_csv, index=False)
    print("mmuad_candidate_reservoir=ok")
    print(f"candidate_rows={len(candidates)}")
    print(f"reservoir_rows={len(reservoir)}")
    print(f"output_csv={args.output_csv}")

    if args.truth_csv is not None:
        truth = normalize_truth_columns(pd.read_csv(args.truth_csv))
        frame_rows, pooled, by_sequence = build_oracle_recall_tables(
            reservoir,
            truth,
            top_k_values=tuple(args.top_k),
            max_truth_time_delta_s=args.max_truth_time_delta_s,
        )
        if args.oracle_frame_csv is not None:
            args.oracle_frame_csv.parent.mkdir(parents=True, exist_ok=True)
            frame_rows.to_csv(args.oracle_frame_csv, index=False)
        if args.oracle_summary_csv is not None:
            args.oracle_summary_csv.parent.mkdir(parents=True, exist_ok=True)
            pooled.to_csv(args.oracle_summary_csv, index=False)
        if args.oracle_by_sequence_csv is not None:
            args.oracle_by_sequence_csv.parent.mkdir(parents=True, exist_ok=True)
            by_sequence.to_csv(args.oracle_by_sequence_csv, index=False)
        print(f"oracle_frames={len(frame_rows)}")
        if not pooled.empty:
            print(f"oracle_all_mse={pooled.loc[0, 'oracle_all_3d_m_mse']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
