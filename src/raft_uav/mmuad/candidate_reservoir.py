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
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns

_REQUIRED_COLUMNS = ("sequence_id", "time_s", "x_m", "y_m", "z_m")


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
    cap_reason_bonus: float = 0.0


def load_candidate_inputs(specs: Sequence[str]) -> pd.DataFrame:
    """Load candidate CSV specs and attach branch metadata."""

    frames: list[pd.DataFrame] = []
    for spec in specs:
        branch, path = _split_candidate_spec(spec)
        rows = normalize_candidate_columns(pd.read_csv(path))
        if rows.empty:
            continue
        _validate_required_columns(rows, path)
        rows = rows.copy()
        if "source" not in rows.columns:
            rows["source"] = "unknown"
        if "track_id" not in rows.columns:
            rows["track_id"] = np.arange(len(rows), dtype=int).astype(str)
        if "candidate_branch" not in rows.columns:
            rows["candidate_branch"] = branch
        else:
            rows["candidate_branch"] = rows["candidate_branch"].fillna(branch).astype(str)
            rows.loc[rows["candidate_branch"].str.len() == 0, "candidate_branch"] = branch
        if "original_x_m" not in rows.columns:
            rows["original_x_m"] = pd.to_numeric(rows["x_m"], errors="coerce")
            rows["original_y_m"] = pd.to_numeric(rows["y_m"], errors="coerce")
            rows["original_z_m"] = pd.to_numeric(rows["z_m"], errors="coerce")
        rows["candidate_branch_input_path"] = str(path)
        frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=[*_REQUIRED_COLUMNS, "source", "candidate_branch"])
    return pd.concat(frames, ignore_index=True)


def build_candidate_reservoir(
    candidates: pd.DataFrame,
    *,
    config: ReservoirConfig | None = None,
    top_per_source: int | None = None,
    top_per_branch: int | None = None,
    global_top_n: int | None = None,
    max_candidates_per_frame: int | None = None,
    score_columns: Sequence[str] | None = None,
    score_floor_quantile: float | None = None,
) -> pd.DataFrame:
    """Return a branch/source-aware per-frame candidate reservoir.

    The reservoir keeps the union of global top-N candidates, top-N candidates
    per source, top-N candidates per candidate branch, and optional score-floor
    candidates. This preserves low-ranked candidates from raw/dynamic/calibrated
    branches while still bounding per-frame candidate count for mixture-MAP
    experiments.
    """

    if config is None:
        score_column = score_columns[0] if score_columns else "ranker_score"
        fallback_score_column = score_columns[1] if score_columns and len(score_columns) > 1 else "confidence"
        config = ReservoirConfig(
            global_top_n=40 if global_top_n is None else int(global_top_n),
            per_source_top_n=3 if top_per_source is None else int(top_per_source),
            per_branch_top_n=3 if top_per_branch is None else int(top_per_branch),
            max_candidates_per_frame=40 if max_candidates_per_frame is None else int(max_candidates_per_frame),
            score_column=score_column,
            fallback_score_column=fallback_score_column,
            score_floor_quantile=score_floor_quantile,
        )
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        return rows.assign(
            candidate_branch=pd.Series(dtype=str),
            candidate_reservoir_score=pd.Series(dtype=float),
            candidate_reservoir_rank=pd.Series(dtype=float),
            candidate_reservoir_reason=pd.Series(dtype=str),
            candidate_reservoir_reason_count=pd.Series(dtype=int),
            candidate_reservoir_cap_score=pd.Series(dtype=float),
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
    out["candidate_reservoir_reasons"] = out["candidate_reservoir_reason"]
    out = _cap_per_frame(
        out,
        max_candidates_per_frame=config.max_candidates_per_frame,
        cap_reason_bonus=config.cap_reason_bonus,
    )
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


def build_reservoir_summary(candidates: pd.DataFrame, reservoir: pd.DataFrame) -> dict[str, Any]:
    """Build a compact JSON-serializable reservoir summary."""

    input_counts = _frame_counts(candidates)
    reservoir_counts = _frame_counts(reservoir)
    reason_counts = _reservoir_reason_count_series(reservoir)
    return {
        "input_candidate_rows": int(len(candidates)),
        "reservoir_candidate_rows": int(len(reservoir)),
        "input_frame_count": int(len(input_counts)),
        "reservoir_frame_count": int(len(reservoir_counts)),
        "input_candidates_per_frame_mean": _safe_mean(input_counts),
        "reservoir_candidates_per_frame_mean": _safe_mean(reservoir_counts),
        "reservoir_candidates_per_frame_p95": _safe_quantile(reservoir_counts, 0.95),
        "reservoir_candidates_per_frame_max": _safe_max(reservoir_counts),
        "reservoir_reason_count_mean": _safe_mean(reason_counts),
        "reservoir_reason_count_p95": _safe_quantile(reason_counts, 0.95),
        "reservoir_reason_count_max": _safe_max(reason_counts),
        "source_counts": _value_counts(reservoir, "source"),
        "candidate_branch_counts": _value_counts(reservoir, "candidate_branch"),
        "reservoir_reason_counts": _reason_counts(reservoir),
    }


def write_reservoir_outputs(
    reservoir: pd.DataFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    input_candidates: pd.DataFrame | None = None,
) -> None:
    """Write reservoir CSV and optional summary JSON."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    reservoir.to_csv(output_csv, index=False)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = build_reservoir_summary(
            input_candidates if input_candidates is not None else reservoir,
            reservoir,
        )
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


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


def _cap_per_frame(
    rows: pd.DataFrame,
    *,
    max_candidates_per_frame: int,
    cap_reason_bonus: float = 0.0,
) -> pd.DataFrame:
    rows = _with_cap_score(rows, cap_reason_bonus=cap_reason_bonus)
    if max_candidates_per_frame <= 0 or rows.empty:
        out = rows.copy()
        out["candidate_reservoir_rank"] = 1.0
        return out
    parts: list[pd.DataFrame] = []
    for _, group in rows.groupby(["sequence_id", "time_s"], sort=False):
        capped = group.sort_values(
            ["candidate_reservoir_cap_score", "candidate_reservoir_score"],
            ascending=[False, False],
        ).head(int(max_candidates_per_frame)).copy()
        capped["candidate_reservoir_rank"] = np.arange(1, len(capped) + 1, dtype=float)
        parts.append(capped)
    return pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0].copy()


def _with_cap_score(rows: pd.DataFrame, *, cap_reason_bonus: float) -> pd.DataFrame:
    out = rows.copy()
    out["candidate_reservoir_reason_count"] = _reservoir_reason_count_series(out)
    base_score = pd.to_numeric(out["candidate_reservoir_score"], errors="coerce").fillna(float("-inf"))
    out["candidate_reservoir_cap_score"] = base_score + (
        float(cap_reason_bonus) * out["candidate_reservoir_reason_count"].astype(float)
    )
    return out


def _reservoir_reason_count_series(rows: pd.DataFrame) -> pd.Series:
    if "candidate_reservoir_reason_count" in rows.columns:
        return pd.to_numeric(rows["candidate_reservoir_reason_count"], errors="coerce").fillna(0)
    column = "candidate_reservoir_reason"
    if column not in rows.columns:
        column = "candidate_reservoir_reasons"
    if column not in rows.columns:
        return pd.Series(0, index=rows.index, dtype=int)
    return rows[column].fillna("").astype(str).map(_count_reason_tokens).astype(int)


def _count_reason_tokens(value: str) -> int:
    return sum(1 for token in value.replace(",", ";").split(";") if token.strip())


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
    candidates = load_candidate_inputs(specs)
    if candidates.empty:
        raise ValueError("at least one --candidate BRANCH=PATH entry is required")
    return candidates


def _split_candidate_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        branch, path_text = spec.split("=", 1)
    else:
        path_text = spec
        branch = Path(spec).stem
    branch = str(branch).strip().replace(" ", "_") or "candidate_branch"
    return branch, Path(path_text)


def _validate_required_columns(rows: pd.DataFrame, path: Path) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"candidate CSV {path} missing required columns: {missing}")


def _frame_counts(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=int)
    return rows.groupby(["sequence_id", "time_s"], dropna=False).size()


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else 0.0


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    return float(values.quantile(quantile)) if not values.empty else 0.0


def _safe_max(values: pd.Series) -> int:
    return int(values.max()) if not values.empty else 0


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns:
        return {}
    return {str(key): int(value) for key, value in rows[column].value_counts(dropna=False).items()}


def _reason_counts(rows: pd.DataFrame) -> dict[str, int]:
    column = "candidate_reservoir_reason"
    if column not in rows.columns:
        column = "candidate_reservoir_reasons"
    counts: dict[str, int] = {}
    if column not in rows.columns:
        return counts
    for value in rows[column].dropna().astype(str):
        for reason in value.replace(",", ";").split(";"):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return counts


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
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        help="alias for --candidate",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--oracle-frame-csv", type=Path)
    parser.add_argument("--oracle-summary-csv", type=Path)
    parser.add_argument("--oracle-by-sequence-csv", type=Path)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--top-per-source", type=int)
    parser.add_argument("--top-per-branch", type=int)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--fallback-score-column", default="confidence")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument(
        "--cap-reason-bonus",
        type=float,
        default=0.0,
        help="bonus added during final frame cap for each independent reservoir selection reason",
    )
    parser.add_argument("--top-k", type=int, action="append", default=[1, 3, 5, 10, 20])
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    candidate_specs = [*args.candidate, *args.candidate_csv]
    candidates = _load_candidate_specs(list(candidate_specs))
    per_source_top_n = args.per_source_top_n if args.top_per_source is None else args.top_per_source
    per_branch_top_n = args.per_branch_top_n if args.top_per_branch is None else args.top_per_branch
    reservoir = build_candidate_reservoir(
        candidates,
        config=ReservoirConfig(
            global_top_n=args.global_top_n,
            per_source_top_n=per_source_top_n,
            per_branch_top_n=per_branch_top_n,
            max_candidates_per_frame=args.max_candidates_per_frame,
            score_column=args.score_column,
            fallback_score_column=args.fallback_score_column,
            score_floor_quantile=args.score_floor_quantile,
            cap_reason_bonus=args.cap_reason_bonus,
        ),
    )
    write_reservoir_outputs(
        reservoir,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        input_candidates=candidates,
    )
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
