"""Block diagnostics for MMUAD candidate-oracle failures.

The candidate-oracle attribution table is frame-level.  This module converts it
into contiguous time blocks so we can distinguish persistent missing-candidate
intervals from intervals where a good candidate exists but is buried below the
reservoir top-K.  The output is meant to guide the next reservoir/ranker/MAP
experiments without running an expensive tracker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_candidate_oracle_block_tables(
    frame_rows: pd.DataFrame,
    *,
    oracle_error_threshold_m: float = 5.0,
    top_k: int = 5,
    max_gap_s: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return contiguous block and summary tables for oracle failure modes.

    ``frame_rows`` should be the output of
    ``raft-uav-mmuad-candidate-oracle-attribution`` or contain compatible
    columns: ``sequence_id``, ``time_s``, ``oracle_all_3d_m``,
    ``oracle_all_rank``, and optionally ``oracle_in_top{K}``.
    """

    rows = pd.DataFrame(frame_rows).copy()
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    _require_columns(rows, ["sequence_id", "time_s", "oracle_all_3d_m"])
    top_k_column = f"oracle_in_top{int(top_k)}"
    if top_k_column not in rows.columns:
        rows[top_k_column] = pd.to_numeric(rows.get("oracle_all_rank", np.inf), errors="coerce") <= int(
            top_k,
        )
    rows["time_s"] = pd.to_numeric(rows["time_s"], errors="coerce")
    rows["oracle_all_3d_m"] = pd.to_numeric(rows["oracle_all_3d_m"], errors="coerce")
    rows["oracle_all_rank"] = pd.to_numeric(rows.get("oracle_all_rank", np.nan), errors="coerce")
    rows[top_k_column] = _to_bool_series(rows[top_k_column])
    rows = rows.dropna(subset=["sequence_id", "time_s"]).sort_values(
        ["sequence_id", "time_s"],
    )
    rows["oracle_failure_mode"] = _failure_mode(
        rows,
        top_k_column=top_k_column,
        oracle_error_threshold_m=float(oracle_error_threshold_m),
    )

    block_records: list[dict[str, Any]] = []
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        current: list[dict[str, Any]] = []
        previous_time: float | None = None
        previous_mode: str | None = None
        block_index = 0
        for record in group.to_dict(orient="records"):
            time_s = float(record["time_s"])
            mode = str(record["oracle_failure_mode"])
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
                        block_index=block_index,
                        top_k=top_k,
                    ),
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
                    block_index=block_index,
                    top_k=top_k,
                ),
            )
    blocks = pd.DataFrame.from_records(block_records)
    summary = _summary_table(blocks)
    return blocks, summary


def write_candidate_oracle_block_outputs(
    *,
    output_dir: Path,
    blocks: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, str]:
    """Write block diagnostics and return artifact paths."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "blocks_csv": output_dir / "mmuad_candidate_oracle_blocks.csv",
        "summary_csv": output_dir / "mmuad_candidate_oracle_block_summary.csv",
        "summary_json": output_dir / "mmuad_candidate_oracle_block_summary.json",
    }
    blocks.to_csv(paths["blocks_csv"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    paths["summary_json"].write_text(
        json.dumps(
            {
                "blocks": blocks.to_dict(orient="records"),
                "summary": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {key: str(value) for key, value in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-oracle-blocks",
        description="summarize contiguous MMUAD candidate-oracle failure blocks",
    )
    parser.add_argument("--frame-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oracle-error-threshold-m", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-gap-s", type=float, default=1.0)
    args = parser.parse_args(argv)

    frame_rows = pd.read_csv(args.frame_csv)
    blocks, summary = build_candidate_oracle_block_tables(
        frame_rows,
        oracle_error_threshold_m=args.oracle_error_threshold_m,
        top_k=args.top_k,
        max_gap_s=args.max_gap_s,
    )
    paths = write_candidate_oracle_block_outputs(
        output_dir=args.output_dir,
        blocks=blocks,
        summary=summary,
    )
    print("mmuad_candidate_oracle_blocks=ok")
    print(f"block_count={len(blocks)}")
    print(f"summary_rows={len(summary)}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _require_columns(rows: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise ValueError(f"frame rows missing required columns: {missing}")


def _to_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    text = values.fillna(False).astype(str).str.lower()
    return text.isin({"true", "1", "yes", "y"})


def _failure_mode(
    rows: pd.DataFrame,
    *,
    top_k_column: str,
    oracle_error_threshold_m: float,
) -> pd.Series:
    missing_candidate = rows["oracle_all_3d_m"] > float(oracle_error_threshold_m)
    buried_candidate = (~missing_candidate) & (~rows[top_k_column])
    return pd.Series(
        np.select(
            [missing_candidate, buried_candidate],
            ["missing_good_candidate", "good_candidate_buried"],
            default="covered_in_topk",
        ),
        index=rows.index,
    )


def _summarize_block(
    records: list[dict[str, Any]],
    *,
    sequence_id: str,
    block_index: int,
    top_k: int,
) -> dict[str, Any]:
    frame = pd.DataFrame.from_records(records)
    errors = pd.to_numeric(frame["oracle_all_3d_m"], errors="coerce").dropna()
    ranks = pd.to_numeric(frame.get("oracle_all_rank", pd.Series(dtype=float)), errors="coerce").dropna()
    start_time_s = float(frame["time_s"].min())
    end_time_s = float(frame["time_s"].max())
    return {
        "sequence_id": sequence_id,
        "block_id": int(block_index),
        "oracle_failure_mode": str(frame["oracle_failure_mode"].iloc[0]),
        "top_k": int(top_k),
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "duration_s": max(0.0, end_time_s - start_time_s),
        "frame_count": int(len(frame)),
        "oracle_all_3d_m_mean": _mean(errors),
        "oracle_all_3d_m_p95": _quantile(errors, 0.95),
        "oracle_all_3d_m_max": _max(errors),
        "oracle_all_rank_mean": _mean(ranks),
        "oracle_all_rank_p50": _quantile(ranks, 0.50),
        "oracle_all_rank_p95": _quantile(ranks, 0.95),
        "candidate_count_mean": _mean(pd.to_numeric(frame.get("candidate_count", np.nan), errors="coerce")),
        "dominant_oracle_branch": _mode(frame.get("oracle_all_candidate_branch")),
        "dominant_oracle_source": _mode(frame.get("oracle_all_candidate_source")),
    }


def _summary_table(blocks: pd.DataFrame) -> pd.DataFrame:
    if blocks.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for (sequence_id, mode), group in blocks.groupby(["sequence_id", "oracle_failure_mode"], sort=True):
        records.append(_summarize_block_group(group, sequence_id=str(sequence_id), mode=str(mode)))
    records.append(_summarize_block_group(blocks, sequence_id="__pooled__", mode="__all__"))
    for mode, group in blocks.groupby("oracle_failure_mode", sort=True):
        records.append(_summarize_block_group(group, sequence_id="__pooled__", mode=str(mode)))
    return pd.DataFrame.from_records(records)


def _summarize_block_group(group: pd.DataFrame, *, sequence_id: str, mode: str) -> dict[str, Any]:
    durations = pd.to_numeric(group["duration_s"], errors="coerce").dropna()
    frames = pd.to_numeric(group["frame_count"], errors="coerce").dropna()
    errors = pd.to_numeric(group["oracle_all_3d_m_max"], errors="coerce").dropna()
    return {
        "sequence_id": sequence_id,
        "oracle_failure_mode": mode,
        "block_count": int(len(group)),
        "frame_count": int(frames.sum()) if not frames.empty else 0,
        "duration_s_sum": float(durations.sum()) if not durations.empty else 0.0,
        "duration_s_max": _max(durations),
        "block_max_error_m_max": _max(errors),
        "block_max_error_m_p95": _quantile(errors, 0.95),
    }


def _mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def _quantile(values: pd.Series, quantile: float) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.quantile(quantile)) if not values.empty else float("nan")


def _max(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.max()) if not values.empty else float("nan")


def _mode(values: pd.Series | None) -> str:
    if values is None:
        return ""
    cleaned = values.dropna().astype(str)
    if cleaned.empty:
        return ""
    return str(cleaned.value_counts().index[0])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
