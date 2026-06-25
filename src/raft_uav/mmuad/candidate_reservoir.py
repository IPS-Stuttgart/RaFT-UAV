"""Branch-preserving candidate reservoirs for MMUAD mixture experiments.

The MMUAD pose diagnostics showed that early pruning can destroy the oracle
ceiling: a translated/scored stream can be easier to rank but may no longer
contain the best raw candidate. This module builds a frame-wise reservoir from
multiple candidate branches so later mixture-MAP or smoothing experiments can
preserve raw, dynamic, translated, and merged hypotheses instead of committing to
one branch too early.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import normalize_candidate_columns

DEFAULT_SCORE_COLUMNS = (
    "ranker_score",
    "confidence",
    "score",
    "probability",
)


@dataclass(frozen=True)
class ReservoirConfig:
    """Configuration for frame-wise branch-preserving candidate selection."""

    per_source_top_n: int = 3
    per_branch_top_n: int = 3
    global_top_n: int = 20
    score_floor_quantile: float | None = None
    max_candidates_per_frame: int | None = 40
    score_columns: tuple[str, ...] = DEFAULT_SCORE_COLUMNS


def load_branch_candidate_csvs(branch_specs: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Load repeated ``name=path`` candidate-branch specifications."""

    frames: dict[str, pd.DataFrame] = {}
    for spec in branch_specs:
        name, path = _parse_branch_spec(spec)
        frames[name] = pd.read_csv(path)
    if not frames:
        raise ValueError("at least one --branch name=path candidate table is required")
    return frames


def build_candidate_reservoir(
    branch_frames: dict[str, pd.DataFrame],
    *,
    config: ReservoirConfig | None = None,
) -> pd.DataFrame:
    """Return a frame-wise candidate reservoir from named candidate branches.

    Rows are selected by the union of top-N per source, top-N per branch, top-N
    per frame, and an optional frame-wise score-floor.  The final frame cap is a
    hard cap and deliberately removes lower-scoring rows after those union rules.
    """

    cfg = config or ReservoirConfig()
    _validate_config(cfg)
    rows = _prepare_branch_rows(branch_frames, cfg.score_columns)
    if rows.empty:
        return rows

    selected_by: dict[int, set[str]] = {int(idx): set() for idx in rows.index}
    for _, group in rows.groupby(["sequence_id", "time_s"], sort=False):
        frame_indices: set[int] = set()
        if cfg.per_source_top_n > 0:
            for source, source_group in group.groupby("source", sort=False):
                idxs = _top_indices(source_group, cfg.per_source_top_n)
                _mark_selected(selected_by, idxs, f"per_source_top{cfg.per_source_top_n}:{source}")
                frame_indices.update(idxs)
        if cfg.per_branch_top_n > 0:
            for branch, branch_group in group.groupby("candidate_branch", sort=False):
                idxs = _top_indices(branch_group, cfg.per_branch_top_n)
                _mark_selected(selected_by, idxs, f"per_branch_top{cfg.per_branch_top_n}:{branch}")
                frame_indices.update(idxs)
        if cfg.global_top_n > 0:
            idxs = _top_indices(group, cfg.global_top_n)
            _mark_selected(selected_by, idxs, f"global_top{cfg.global_top_n}")
            frame_indices.update(idxs)
        if cfg.score_floor_quantile is not None:
            finite = group["reservoir_score"].replace([np.inf, -np.inf], np.nan).dropna()
            if not finite.empty:
                threshold = float(finite.quantile(float(cfg.score_floor_quantile)))
                idxs = set(group.index[group["reservoir_score"] >= threshold].astype(int))
                _mark_selected(selected_by, idxs, f"score_floor_q{cfg.score_floor_quantile:g}")
                frame_indices.update(idxs)
        if cfg.max_candidates_per_frame is not None and len(frame_indices) > cfg.max_candidates_per_frame:
            kept = _top_indices(rows.loc[list(frame_indices)], cfg.max_candidates_per_frame)
            dropped = frame_indices.difference(kept)
            for idx in dropped:
                selected_by[int(idx)].clear()
                selected_by[int(idx)].add("dropped_by_frame_cap")
            _mark_selected(selected_by, kept, f"frame_cap{cfg.max_candidates_per_frame}")

    kept_indices = [
        idx
        for idx, labels in selected_by.items()
        if labels and "dropped_by_frame_cap" not in labels
    ]
    out = rows.loc[kept_indices].copy()
    if out.empty:
        return rows.iloc[0:0].copy()
    out["reservoir_selected_by"] = [";".join(sorted(selected_by[int(idx)])) for idx in out.index]
    out["reservoir_candidate_count_in_frame"] = (
        out.groupby(["sequence_id", "time_s"])["sequence_id"].transform("size").astype(int)
    )
    out = out.sort_values(
        ["sequence_id", "time_s", "reservoir_score", "candidate_branch", "source"],
        ascending=[True, True, False, True, True],
    )
    return out.reset_index(drop=True)


def summarize_candidate_reservoir(reservoir: pd.DataFrame, *, config: ReservoirConfig) -> dict[str, Any]:
    """Return JSON-serializable reservoir summary statistics."""

    if reservoir.empty:
        return {
            "rows": 0,
            "frames": 0,
            "branches": [],
            "sources": [],
            "config": _config_payload(config),
        }
    frame_counts = reservoir.groupby(["sequence_id", "time_s"]).size()
    by_branch = reservoir.groupby("candidate_branch").size().sort_values(ascending=False)
    by_source = reservoir.groupby("source").size().sort_values(ascending=False)
    return {
        "rows": int(len(reservoir)),
        "frames": int(len(frame_counts)),
        "sequence_count": int(reservoir["sequence_id"].nunique()),
        "candidate_count_per_frame_mean": float(frame_counts.mean()),
        "candidate_count_per_frame_p95": float(frame_counts.quantile(0.95)),
        "candidate_count_per_frame_max": int(frame_counts.max()),
        "branches": [str(item) for item in sorted(reservoir["candidate_branch"].unique())],
        "sources": [str(item) for item in sorted(reservoir["source"].unique())],
        "rows_by_branch": {str(k): int(v) for k, v in by_branch.items()},
        "rows_by_source": {str(k): int(v) for k, v in by_source.items()},
        "config": _config_payload(config),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-reservoir",
        description="build a branch-preserving MMUAD candidate reservoir",
    )
    parser.add_argument(
        "--branch",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="candidate branch specification; repeat for raw/dynamic/translated streams",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument(
        "--score-column",
        action="append",
        dest="score_columns",
        help=(
            "score column priority; may be repeated; defaults to "
            "ranker_score/confidence/score/probability"
        ),
    )
    args = parser.parse_args(argv)

    score_columns = tuple(args.score_columns) if args.score_columns else DEFAULT_SCORE_COLUMNS
    config = ReservoirConfig(
        per_source_top_n=args.per_source_top_n,
        per_branch_top_n=args.per_branch_top_n,
        global_top_n=args.global_top_n,
        score_floor_quantile=args.score_floor_quantile,
        max_candidates_per_frame=args.max_candidates_per_frame,
        score_columns=score_columns,
    )
    reservoir = build_candidate_reservoir(load_branch_candidate_csvs(args.branch), config=config)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    reservoir.to_csv(args.output_csv, index=False)
    summary = summarize_candidate_reservoir(reservoir, config=config)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("mmuad_candidate_reservoir=ok")
    print(f"output_csv={args.output_csv}")
    print(f"rows={summary['rows']}")
    print(f"frames={summary['frames']}")
    return 0


def _prepare_branch_rows(
    branch_frames: dict[str, pd.DataFrame],
    score_columns: tuple[str, ...],
) -> pd.DataFrame:
    prepared: list[pd.DataFrame] = []
    for branch, frame in branch_frames.items():
        if not branch or "=" in branch:
            raise ValueError(f"invalid candidate branch name: {branch!r}")
        rows = normalize_candidate_columns(frame, default_source=branch).copy()
        if rows.empty:
            continue
        rows["candidate_branch"] = str(branch)
        rows["candidate_original_row"] = np.arange(len(rows), dtype=int)
        rows["reservoir_score"] = _first_available_score(rows, score_columns)
        rows["reservoir_score_source"] = _score_source(rows, score_columns)
        prepared.append(rows)
    if not prepared:
        return pd.DataFrame()
    out = pd.concat(prepared, ignore_index=True, sort=False)
    out["reservoir_row_id"] = np.arange(len(out), dtype=int)
    out = out.set_index("reservoir_row_id", drop=False)
    for column in ("sequence_id", "source", "candidate_branch"):
        out[column] = out[column].fillna("").astype(str)
    out["time_s"] = pd.to_numeric(out["time_s"], errors="coerce")
    out["reservoir_score"] = pd.to_numeric(out["reservoir_score"], errors="coerce").fillna(0.0)
    return out.loc[np.isfinite(out["time_s"])].copy()


def _first_available_score(rows: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    score = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in columns:
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        score = score.where(score.notna(), values)
    return score.fillna(0.0)


def _score_source(rows: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    source = pd.Series("constant_zero", index=rows.index, dtype=object)
    remaining = pd.Series(True, index=rows.index)
    for column in columns:
        if column not in rows.columns:
            continue
        valid = pd.to_numeric(rows[column], errors="coerce").notna()
        use = remaining & valid
        source.loc[use] = column
        remaining &= ~use
    return source


def _top_indices(group: pd.DataFrame, n: int) -> set[int]:
    if n <= 0 or group.empty:
        return set()
    ordered = group.sort_values(
        ["reservoir_score", "candidate_branch", "source", "reservoir_row_id"],
        ascending=[False, True, True, True],
    )
    return set(int(idx) for idx in ordered.head(int(n)).index)


def _mark_selected(selected_by: dict[int, set[str]], indices: Iterable[int], label: str) -> None:
    for idx in indices:
        selected_by[int(idx)].add(label)


def _parse_branch_spec(spec: str) -> tuple[str, Path]:
    if "=" not in str(spec):
        raise ValueError(f"branch spec must be NAME=CSV, got {spec!r}")
    name, path = str(spec).split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"branch spec has empty branch name: {spec!r}")
    return name, Path(path)


def _validate_config(config: ReservoirConfig) -> None:
    if config.per_source_top_n < 0 or config.per_branch_top_n < 0 or config.global_top_n < 0:
        raise ValueError("top-N reservoir settings must be non-negative")
    if config.max_candidates_per_frame is not None and config.max_candidates_per_frame <= 0:
        raise ValueError("max_candidates_per_frame must be positive or None")
    if config.score_floor_quantile is not None and not 0.0 <= config.score_floor_quantile <= 1.0:
        raise ValueError("score_floor_quantile must be in [0, 1]")


def _config_payload(config: ReservoirConfig) -> dict[str, Any]:
    return {
        "per_source_top_n": int(config.per_source_top_n),
        "per_branch_top_n": int(config.per_branch_top_n),
        "global_top_n": int(config.global_top_n),
        "score_floor_quantile": config.score_floor_quantile,
        "max_candidates_per_frame": config.max_candidates_per_frame,
        "score_columns": list(config.score_columns),
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
