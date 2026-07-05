"""Contiguous failure-block diagnostics for MMUAD candidate-mixture assignments.

``raft-uav-mmuad-candidate-assignment-diagnostics`` explains each frame by
comparing the mixture assignment, state estimate, and local oracle candidate.
This module groups those frame-level labels into contiguous time blocks so the
next MMUAD runs can distinguish one-off misassignments from persistent intervals
where a good candidate is buried, missing, or ignored by the smoother.

Truth is required only in the upstream frame diagnostic.  This block summarizer
consumes that diagnostic table and does not read truth itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BLOCKS_CSV = "mmuad_candidate_assignment_blocks.csv"
SUMMARY_CSV = "mmuad_candidate_assignment_block_summary.csv"
SUMMARY_JSON = "mmuad_candidate_assignment_block_summary.json"


def build_candidate_assignment_block_tables(
    frame_rows: pd.DataFrame,
    *,
    max_gap_s: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return contiguous block and summary tables for assignment failure modes."""

    rows = pd.DataFrame(frame_rows).copy()
    if rows.empty:
        empty = pd.DataFrame()
        return empty, empty
    _require_columns(rows, ["sequence_id", "time_s", "assignment_failure_mode"])
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["assignment_failure_mode"] = rows["assignment_failure_mode"].astype(str)
    rows["time_s"] = pd.to_numeric(rows["time_s"], errors="coerce")
    rows = rows.dropna(subset=["sequence_id", "time_s", "assignment_failure_mode"]).sort_values(
        ["sequence_id", "time_s"],
    )
    block_records: list[dict[str, Any]] = []
    for sequence_id, sequence_rows in rows.groupby("sequence_id", sort=True):
        current: list[dict[str, Any]] = []
        previous_time: float | None = None
        previous_mode: str | None = None
        block_index = 0
        for record in sequence_rows.to_dict(orient="records"):
            time_s = float(record["time_s"])
            mode = str(record["assignment_failure_mode"])
            new_block = (
                previous_time is None
                or previous_mode != mode
                or time_s - previous_time > float(max_gap_s)
            )
            if new_block and current:
                block_records.append(
                    _summarize_block(
                        current,
                        sequence_id=str(sequence_id),
                        block_id=block_index,
                    )
                )
                block_index += 1
                current = []
            current.append(record)
            previous_time = time_s
            previous_mode = mode
        if current:
            block_records.append(
                _summarize_block(
                    current,
                    sequence_id=str(sequence_id),
                    block_id=block_index,
                )
            )
    blocks = pd.DataFrame.from_records(block_records)
    summary = build_candidate_assignment_block_summary(blocks)
    return blocks, summary


def build_candidate_assignment_block_summary(blocks: pd.DataFrame) -> pd.DataFrame:
    """Summarize assignment blocks by sequence and failure mode."""

    frame = pd.DataFrame(blocks).copy()
    if frame.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    records.append(_summary_record(frame, sequence_id="__pooled__", failure_mode="__all__"))
    for mode, group in frame.groupby("assignment_failure_mode", sort=True):
        records.append(_summary_record(group, sequence_id="__pooled__", failure_mode=str(mode)))
    for sequence_id, group in frame.groupby("sequence_id", sort=True):
        records.append(_summary_record(group, sequence_id=str(sequence_id), failure_mode="__all__"))
        for mode, mode_group in group.groupby("assignment_failure_mode", sort=True):
            records.append(
                _summary_record(mode_group, sequence_id=str(sequence_id), failure_mode=str(mode))
            )
    return pd.DataFrame.from_records(records)


def write_candidate_assignment_block_outputs(
    *,
    blocks: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    max_gap_s: float,
) -> dict[str, Path]:
    """Write block diagnostics and return artifact paths."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "blocks_csv": output / BLOCKS_CSV,
        "summary_csv": output / SUMMARY_CSV,
        "summary_json": output / SUMMARY_JSON,
    }
    blocks.to_csv(paths["blocks_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    payload = {
        "schema": "raft-uav-mmuad-candidate-assignment-blocks-v1",
        "max_gap_s": float(max_gap_s),
        "block_count": int(len(blocks)),
        "summary": summary.to_dict(orient="records"),
    }
    paths["summary_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-assignment-blocks",
        description="summarize contiguous MMUAD candidate-mixture assignment failures",
    )
    parser.add_argument("--frame-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gap-s", type=float, default=1.0)
    args = parser.parse_args(argv)

    frame_rows = pd.read_csv(args.frame_csv)
    blocks, summary = build_candidate_assignment_block_tables(
        frame_rows,
        max_gap_s=float(args.max_gap_s),
    )
    paths = write_candidate_assignment_block_outputs(
        blocks=blocks,
        summary=summary,
        output_dir=args.output_dir,
        max_gap_s=float(args.max_gap_s),
    )
    print("mmuad_candidate_assignment_blocks=ok")
    print(f"block_count={len(blocks)}")
    print(f"summary_rows={len(summary)}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _summarize_block(
    records: list[dict[str, Any]],
    *,
    sequence_id: str,
    block_id: int,
) -> dict[str, Any]:
    frame = pd.DataFrame.from_records(records)
    start_time_s = float(frame["time_s"].min())
    end_time_s = float(frame["time_s"].max())
    return {
        "sequence_id": str(sequence_id),
        "block_id": int(block_id),
        "assignment_failure_mode": str(frame["assignment_failure_mode"].iloc[0]),
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "duration_s": max(0.0, end_time_s - start_time_s),
        "frame_count": int(len(frame)),
        "state_error_3d_m_mean": _mean(frame.get("state_error_3d_m")),
        "state_error_3d_m_p95": _quantile(frame.get("state_error_3d_m"), 0.95),
        "state_error_3d_m_max": _max(frame.get("state_error_3d_m")),
        "oracle_error_3d_m_mean": _mean(frame.get("oracle_error_3d_m")),
        "oracle_error_3d_m_max": _max(frame.get("oracle_error_3d_m")),
        "state_regret_m_mean": _mean(frame.get("state_regret_m")),
        "state_regret_m_max": _max(frame.get("state_regret_m")),
        "dominant_regret_m_mean": _mean(frame.get("dominant_regret_m")),
        "weighted_regret_m_mean": _mean(frame.get("weighted_regret_m")),
        "oracle_mixture_weight_mean": _mean(frame.get("oracle_mixture_weight")),
        "oracle_weight_rank_p50": _quantile(frame.get("oracle_weight_rank"), 0.50),
        "oracle_weight_rank_p95": _quantile(frame.get("oracle_weight_rank"), 0.95),
        "dominant_matches_oracle_rate": _mean_bool(frame.get("dominant_is_oracle")),
        "oracle_in_topk_by_weight_rate": _mean_bool(frame.get("oracle_in_topk_by_weight")),
        "dominant_oracle_branch": _mode(frame.get("oracle_candidate_branch")),
        "dominant_oracle_source": _mode(frame.get("oracle_source")),
        "dominant_assigned_branch": _mode(frame.get("dominant_candidate_branch")),
        "dominant_assigned_source": _mode(frame.get("dominant_source")),
    }


def _summary_record(group: pd.DataFrame, *, sequence_id: str, failure_mode: str) -> dict[str, Any]:
    durations = pd.to_numeric(group.get("duration_s", pd.Series(dtype=float)), errors="coerce")
    frames = pd.to_numeric(group.get("frame_count", pd.Series(dtype=float)), errors="coerce")
    return {
        "sequence_id": str(sequence_id),
        "assignment_failure_mode": str(failure_mode),
        "block_count": int(len(group)),
        "frame_count": int(frames.sum()) if not frames.dropna().empty else 0,
        "duration_s_sum": float(durations.sum()) if not durations.dropna().empty else 0.0,
        "duration_s_max": _max(durations),
        "block_state_error_3d_m_max": _max(group.get("state_error_3d_m_max")),
        "block_state_error_3d_m_p95": _quantile(group.get("state_error_3d_m_max"), 0.95),
        "block_state_regret_m_max": _max(group.get("state_regret_m_max")),
        "mean_oracle_weight_rank_p50": _mean(group.get("oracle_weight_rank_p50")),
        "dominant_oracle_branch": _mode(group.get("dominant_oracle_branch")),
        "dominant_oracle_source": _mode(group.get("dominant_oracle_source")),
        "dominant_assigned_branch": _mode(group.get("dominant_assigned_branch")),
        "dominant_assigned_source": _mode(group.get("dominant_assigned_source")),
    }


def _require_columns(rows: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise ValueError(f"assignment diagnostic rows missing required columns: {missing}")


def _mean(values: pd.Series | None) -> float:
    if values is None:
        return float("nan")
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _mean_bool(values: pd.Series | None) -> float:
    if values is None:
        return float("nan")
    if values.dtype == bool:
        data = values.astype(float)
    else:
        data = values.astype(str).str.lower().isin({"true", "1", "yes", "y"}).astype(float)
    return _mean(data)


def _quantile(values: pd.Series | None, quantile: float) -> float:
    if values is None:
        return float("nan")
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(float(quantile))) if not values.empty else float("nan")


def _max(values: pd.Series | None) -> float:
    if values is None:
        return float("nan")
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.max()) if not values.empty else float("nan")


def _mode(values: pd.Series | None) -> str:
    if values is None:
        return ""
    cleaned = values.dropna().astype(str)
    cleaned = cleaned.loc[~cleaned.str.lower().isin({"", "nan", "none", "<na>"})]
    if cleaned.empty:
        return ""
    return str(cleaned.value_counts().index[0])


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
