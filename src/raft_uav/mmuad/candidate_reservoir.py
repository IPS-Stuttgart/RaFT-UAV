"""Branch-preserving candidate-reservoir diagnostics for MMUAD tracking.

The Track 5 pose experiments exposed a failure mode where useful raw/static or
source-calibrated candidates can be pruned before mixture smoothing sees them.
This module keeps multiple candidate branches alive and measures whether a
bounded per-source/per-branch reservoir preserves the oracle candidate ceiling.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import load_jsonable, normalize_candidate_columns, normalize_truth_columns


DEFAULT_TOP_K = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class ReservoirConfig:
    """Configuration for frame-level branch reservoir construction."""

    per_source_top_n: int = 3
    per_branch_top_n: int = 3
    global_top_n: int = 20
    score_column: str = "confidence"
    score_floor_quantile: float | None = None


def parse_branch_candidate_spec(spec: str) -> tuple[str, Path]:
    """Parse ``BRANCH=path.csv`` or derive the branch from the file stem."""

    if "=" in spec:
        branch, path = spec.split("=", 1)
        branch = branch.strip()
        if not branch:
            raise ValueError(f"candidate spec has an empty branch label: {spec!r}")
        return branch, Path(path)
    path = Path(spec)
    return path.stem, path


def load_branch_candidate_specs(specs: Iterable[str]) -> pd.DataFrame:
    """Load and tag repeated branch candidate CSV specifications."""

    frames: list[pd.DataFrame] = []
    for spec in specs:
        branch, path = parse_branch_candidate_spec(str(spec))
        frame = load_candidate_file(path)
        rows = tag_candidate_branch(frame.rows, branch=branch)
        rows["candidate_branch_file"] = str(path)
        frames.append(rows)
    if not frames:
        return normalize_candidate_columns(pd.DataFrame())
    return normalize_candidate_columns(pd.concat(frames, ignore_index=True))


def tag_candidate_branch(candidates: pd.DataFrame, *, branch: str) -> pd.DataFrame:
    """Return normalized candidates with branch and original-coordinate metadata."""

    rows = normalize_candidate_columns(pd.DataFrame(candidates)).copy()
    if rows.empty:
        rows["candidate_branch"] = pd.Series(dtype=object)
        return rows
    branch_text = str(branch).strip() or "candidate"
    rows["candidate_branch"] = branch_text
    for column in ("x_m", "y_m", "z_m"):
        original_column = f"original_{column}"
        if original_column not in rows.columns:
            rows[original_column] = rows[column]
    rows["candidate_branch_row_id"] = np.arange(len(rows), dtype=int)
    return rows


def build_branch_reservoir(
    candidates: pd.DataFrame,
    *,
    config: ReservoirConfig | None = None,
) -> pd.DataFrame:
    """Keep a bounded union of per-source, per-branch, and global top candidates.

    The returned table preserves candidates that a single global score ordering
    could discard, which is useful before running expensive mixture-MAP variants.
    """

    config = config or ReservoirConfig()
    rows = normalize_candidate_columns(pd.DataFrame(candidates)).copy()
    if rows.empty:
        return rows
    if "candidate_branch" not in rows.columns:
        rows["candidate_branch"] = rows["source"].astype(str)
    rows["_reservoir_row_id"] = np.arange(len(rows), dtype=int)
    rows["_reservoir_score"] = _score_values(rows, config.score_column)

    selected: list[pd.DataFrame] = []
    group_columns = ["sequence_id", "time_s"]
    for _, frame_rows in rows.groupby(group_columns, sort=True, dropna=False):
        frame = frame_rows.copy()
        frame = _apply_score_floor(frame, config.score_floor_quantile)
        selected_ids: set[int] = set()
        selected_ids.update(_top_row_ids(frame, config.global_top_n))
        if config.per_source_top_n > 0:
            for _, source_rows in frame.groupby("source", sort=False, dropna=False):
                selected_ids.update(_top_row_ids(source_rows, config.per_source_top_n))
        if config.per_branch_top_n > 0:
            for _, branch_rows in frame.groupby("candidate_branch", sort=False, dropna=False):
                selected_ids.update(_top_row_ids(branch_rows, config.per_branch_top_n))
        if selected_ids:
            selected.append(frame.loc[frame["_reservoir_row_id"].isin(selected_ids)])
    if not selected:
        return rows.iloc[0:0].drop(columns=["_reservoir_score", "_reservoir_row_id"])
    reservoir = pd.concat(selected, ignore_index=True)
    return reservoir.drop(columns=["_reservoir_score", "_reservoir_row_id"])


def build_topk_oracle_recall(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    top_k: tuple[int, ...] = DEFAULT_TOP_K,
    max_time_delta_s: float | None = 0.5,
    score_column: str = "confidence",
    candidate_pool: str = "candidate_pool",
) -> pd.DataFrame:
    """Compute per-timestamp top-K oracle distances for a candidate pool."""

    candidate_rows = normalize_candidate_columns(pd.DataFrame(candidates)).copy()
    truth_rows = normalize_truth_columns(pd.DataFrame(truth)).copy()
    columns = _oracle_row_columns(top_k)
    if candidate_rows.empty or truth_rows.empty:
        return pd.DataFrame(columns=columns)
    candidate_rows["_oracle_score"] = _score_values(candidate_rows, score_column)
    if "candidate_branch" not in candidate_rows.columns:
        candidate_rows["candidate_branch"] = candidate_rows["source"].astype(str)
    records: list[dict[str, Any]] = []
    for sequence_id, sequence_truth in truth_rows.groupby("sequence_id", sort=True):
        sequence_candidates = candidate_rows.loc[candidate_rows["sequence_id"] == sequence_id]
        sequence_candidates = sequence_candidates.sort_values("time_s").reset_index(drop=True)
        for _, truth_row in sequence_truth.sort_values("time_s").iterrows():
            nearest = _nearest_time_rows(
                sequence_candidates,
                float(truth_row["time_s"]),
                max_time_delta_s=max_time_delta_s,
            )
            records.append(
                _oracle_record(
                    truth_row,
                    nearest,
                    top_k=top_k,
                    candidate_pool=candidate_pool,
                )
            )
    return pd.DataFrame.from_records(records, columns=columns)


def summarize_oracle_recall(rows: pd.DataFrame, *, by_sequence: bool = False) -> pd.DataFrame:
    """Summarize oracle rows into long-form MSE/RMSE/P95 tables."""

    if rows.empty:
        return pd.DataFrame(columns=_summary_columns(by_sequence=by_sequence))
    group_columns = ["candidate_pool"]
    if by_sequence:
        group_columns.append("sequence_id")
    records: list[dict[str, Any]] = []
    error_columns = [column for column in rows.columns if column.startswith("oracle_")]
    for group_key, group in rows.groupby(group_columns, sort=True, dropna=False):
        group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        base: dict[str, Any] = {"candidate_pool": group_values[0]}
        if by_sequence:
            base["sequence_id"] = group_values[1]
        for column in error_columns:
            label = column.removeprefix("oracle_").removesuffix("_error_m")
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            record = dict(base)
            record["oracle_k"] = label
            record["truth_rows"] = int(len(group))
            record["matched_rows"] = int(len(finite))
            record["matched_fraction"] = float(len(finite) / len(group)) if len(group) else 0.0
            record.update(_error_stats(finite))
            records.append(record)
    return pd.DataFrame.from_records(records, columns=_summary_columns(by_sequence=by_sequence))


def write_branch_reservoir_artifacts(
    *,
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    output_dir: Path,
    config: ReservoirConfig,
    top_k: tuple[int, ...] = DEFAULT_TOP_K,
    max_time_delta_s: float | None = 0.5,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write reservoir candidates, oracle recall rows, summaries, and metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_pool = normalize_candidate_columns(pd.DataFrame(candidates)).copy()
    full_pool["candidate_pool"] = "full_pool"
    reservoir = build_branch_reservoir(full_pool, config=config)
    reservoir["candidate_pool"] = "branch_reservoir"
    truth_rows = normalize_truth_columns(pd.DataFrame(truth))

    full_rows = build_topk_oracle_recall(
        full_pool,
        truth_rows,
        top_k=top_k,
        max_time_delta_s=max_time_delta_s,
        score_column=config.score_column,
        candidate_pool="full_pool",
    )
    reservoir_rows = build_topk_oracle_recall(
        reservoir,
        truth_rows,
        top_k=top_k,
        max_time_delta_s=max_time_delta_s,
        score_column=config.score_column,
        candidate_pool="branch_reservoir",
    )
    oracle_rows = pd.concat([full_rows, reservoir_rows], ignore_index=True)
    pooled = summarize_oracle_recall(oracle_rows)
    by_sequence = summarize_oracle_recall(oracle_rows, by_sequence=True)

    paths = {
        "reservoir_candidates_csv": output_dir / "mmuad_branch_reservoir_candidates.csv",
        "oracle_rows_csv": output_dir / "mmuad_branch_reservoir_oracle_rows.csv",
        "oracle_pooled_csv": output_dir / "mmuad_branch_reservoir_oracle_pooled.csv",
        "oracle_by_sequence_csv": output_dir / "mmuad_branch_reservoir_oracle_by_sequence.csv",
        "provenance_json": output_dir / "mmuad_branch_reservoir_provenance.json",
    }
    reservoir.to_csv(paths["reservoir_candidates_csv"], index=False)
    oracle_rows.to_csv(paths["oracle_rows_csv"], index=False)
    pooled.to_csv(paths["oracle_pooled_csv"], index=False)
    by_sequence.to_csv(paths["oracle_by_sequence_csv"], index=False)
    metadata = {
        "config": asdict(config),
        "top_k": list(top_k),
        "max_time_delta_s": max_time_delta_s,
        "full_pool_rows": int(len(full_pool)),
        "reservoir_rows": int(len(reservoir)),
        "truth_rows": int(len(truth_rows)),
        "provenance": provenance or {},
    }
    paths["provenance_json"].write_text(json.dumps(load_jsonable(metadata), indent=2), encoding="utf-8")
    return paths


def _score_values(rows: pd.DataFrame, score_column: str) -> pd.Series:
    if score_column in rows.columns:
        score = pd.to_numeric(rows[score_column], errors="coerce")
    elif "confidence" in rows.columns:
        score = pd.to_numeric(rows["confidence"], errors="coerce")
    else:
        score = pd.Series(1.0, index=rows.index)
    return score.fillna(float("-inf"))


def _apply_score_floor(frame: pd.DataFrame, score_floor_quantile: float | None) -> pd.DataFrame:
    if score_floor_quantile is None or frame.empty:
        return frame
    q = min(max(float(score_floor_quantile), 0.0), 1.0)
    finite_scores = frame["_reservoir_score"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_scores.empty:
        return frame
    threshold = float(finite_scores.quantile(q))
    return frame.loc[frame["_reservoir_score"] >= threshold].copy()


def _top_row_ids(frame: pd.DataFrame, count: int) -> set[int]:
    if count <= 0 or frame.empty:
        return set()
    ordered = frame.sort_values(
        ["_reservoir_score", "_reservoir_row_id"],
        ascending=[False, True],
    )
    return set(ordered.head(int(count))["_reservoir_row_id"].astype(int))


def _nearest_time_rows(
    rows: pd.DataFrame,
    time_s: float,
    *,
    max_time_delta_s: float | None,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    deltas = (pd.to_numeric(rows["time_s"], errors="coerce") - float(time_s)).abs()
    finite = np.isfinite(deltas.to_numpy(float))
    if not finite.any():
        return rows.iloc[0:0].copy()
    best_delta = float(deltas.loc[finite].min())
    if max_time_delta_s is not None and best_delta > float(max_time_delta_s):
        return rows.iloc[0:0].copy()
    return rows.loc[finite & (np.abs(deltas - best_delta) <= 1.0e-9)].copy()


def _oracle_record(
    truth_row: pd.Series,
    candidates: pd.DataFrame,
    *,
    top_k: tuple[int, ...],
    candidate_pool: str,
) -> dict[str, Any]:
    base = {
        "candidate_pool": candidate_pool,
        "sequence_id": str(truth_row["sequence_id"]),
        "time_s": float(truth_row["time_s"]),
        "candidate_count": int(len(candidates)),
        "matched_time_s": np.nan,
        "matched_time_delta_s": np.nan,
    }
    if candidates.empty:
        for k in top_k:
            base[f"oracle_top{k}_error_m"] = np.nan
        base["oracle_all_error_m"] = np.nan
        return base
    ordered = candidates.sort_values(["_oracle_score"], ascending=[False]).reset_index(drop=True)
    base["matched_time_s"] = float(ordered["time_s"].iloc[0])
    base["matched_time_delta_s"] = float(base["matched_time_s"] - float(truth_row["time_s"]))
    truth_xyz = truth_row[["x_m", "y_m", "z_m"]].to_numpy(float)
    candidate_xyz = ordered[["x_m", "y_m", "z_m"]].to_numpy(float)
    distances = np.linalg.norm(candidate_xyz - truth_xyz, axis=1)
    for k in top_k:
        base[f"oracle_top{k}_error_m"] = _min_distance(distances[: int(k)])
    base["oracle_all_error_m"] = _min_distance(distances)
    return base


def _min_distance(distances: np.ndarray) -> float:
    finite = distances[np.isfinite(distances)]
    return float(np.min(finite)) if len(finite) else np.nan


def _error_stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {
            "oracle_mse": np.nan,
            "oracle_rmse_m": np.nan,
            "oracle_mean_m": np.nan,
            "oracle_p95_m": np.nan,
            "oracle_max_m": np.nan,
        }
    return {
        "oracle_mse": float(np.mean(np.square(values))),
        "oracle_rmse_m": float(np.sqrt(np.mean(np.square(values)))),
        "oracle_mean_m": float(np.mean(values)),
        "oracle_p95_m": float(np.percentile(values, 95)),
        "oracle_max_m": float(np.max(values)),
    }


def _oracle_row_columns(top_k: tuple[int, ...]) -> list[str]:
    columns = [
        "candidate_pool",
        "sequence_id",
        "time_s",
        "candidate_count",
        "matched_time_s",
        "matched_time_delta_s",
    ]
    columns.extend(f"oracle_top{k}_error_m" for k in top_k)
    columns.append("oracle_all_error_m")
    return columns


def _summary_columns(*, by_sequence: bool) -> list[str]:
    columns = ["candidate_pool"]
    if by_sequence:
        columns.append("sequence_id")
    columns.extend(
        [
            "oracle_k",
            "truth_rows",
            "matched_rows",
            "matched_fraction",
            "oracle_mse",
            "oracle_rmse_m",
            "oracle_mean_m",
            "oracle_p95_m",
            "oracle_max_m",
        ]
    )
    return columns


def _parse_top_k(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(value).split(",")]
    top_k = tuple(sorted({int(part) for part in parts if part}))
    if not top_k:
        raise argparse.ArgumentTypeError("--top-k must contain at least one positive integer")
    if any(k <= 0 for k in top_k):
        raise argparse.ArgumentTypeError("--top-k values must be positive")
    return top_k


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-branch-reservoir",
        description="write branch-preserving reservoir oracle-recall diagnostics for MMUAD",
    )
    parser.add_argument("--truth-file", type=Path, required=True)
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        help="candidate CSV, optionally as BRANCH=path; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-column", default="confidence")
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--max-time-delta-s", type=float, default=0.5)
    parser.add_argument("--top-k", type=_parse_top_k, default=DEFAULT_TOP_K)
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv")
    config = ReservoirConfig(
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        global_top_n=int(args.global_top_n),
        score_column=str(args.score_column),
        score_floor_quantile=args.score_floor_quantile,
    )
    truth = load_evaluation_truth_file(args.truth_file).rows
    candidates = load_branch_candidate_specs(args.candidate_csv)
    paths = write_branch_reservoir_artifacts(
        candidates=candidates,
        truth=truth,
        output_dir=args.output_dir,
        config=config,
        top_k=tuple(args.top_k),
        max_time_delta_s=float(args.max_time_delta_s),
        provenance={"candidate_csv": list(args.candidate_csv), "truth_file": str(args.truth_file)},
    )
    print("mmuad_branch_reservoir=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
