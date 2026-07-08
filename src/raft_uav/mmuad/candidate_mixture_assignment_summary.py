"""Summarize MMUAD candidate-mixture MAP assignment diagnostics.

The candidate-mixture MAP runner already writes row-level assignment weights.
For branch-preserving experiments, the next question is often which candidate
branches or sensors actually receive mixture mass, dominate frames, or carry
large residuals.  This module turns the row-level assignment CSV into compact
per-label and per-sequence summaries without requiring truth labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

DEFAULT_GROUP_COLUMNS = ("candidate_branch", "source")


def summarize_assignments(
    assignments: pd.DataFrame,
    *,
    group_columns: Iterable[str] = DEFAULT_GROUP_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pooled and per-sequence summaries for mixture assignment rows."""

    rows = pd.DataFrame(assignments).copy()
    if rows.empty:
        empty = pd.DataFrame()
        return empty, empty
    _ensure_assignment_columns(rows)
    pooled_parts: list[pd.DataFrame] = []
    sequence_parts: list[pd.DataFrame] = []
    for column in group_columns:
        if column not in rows.columns:
            continue
        pooled_parts.append(_summarize_for_group(rows, group_column=column))
        sequence_parts.append(_summarize_by_sequence(rows, group_column=column))
    pooled = _concat(pooled_parts)
    by_sequence = _concat(sequence_parts)
    return pooled, by_sequence


def write_assignment_summary_outputs(
    assignments: pd.DataFrame,
    *,
    output_dir: Path,
    group_columns: Iterable[str] = DEFAULT_GROUP_COLUMNS,
    pooled_csv: Path | None = None,
    by_sequence_csv: Path | None = None,
    summary_json: Path | None = None,
) -> dict[str, Path]:
    """Write assignment summary CSV/JSON artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pooled, by_sequence = summarize_assignments(assignments, group_columns=group_columns)
    paths = {
        "pooled_csv": pooled_csv or output / "mmuad_candidate_mixture_assignment_summary.csv",
        "by_sequence_csv": by_sequence_csv
        or output / "mmuad_candidate_mixture_assignment_by_sequence.csv",
        "summary_json": summary_json
        or output / "mmuad_candidate_mixture_assignment_summary.json",
    }
    for key in ("pooled_csv", "by_sequence_csv"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(paths["pooled_csv"], index=False)
    by_sequence.to_csv(paths["by_sequence_csv"], index=False)
    paths["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    payload = build_assignment_summary_payload(
        assignments,
        pooled=pooled,
        by_sequence=by_sequence,
        group_columns=tuple(group_columns),
    )
    paths["summary_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def build_assignment_summary_payload(
    assignments: pd.DataFrame,
    *,
    pooled: pd.DataFrame,
    by_sequence: pd.DataFrame,
    group_columns: Iterable[str],
) -> dict[str, Any]:
    """Build compact JSON metadata for assignment summaries."""

    rows = pd.DataFrame(assignments)
    frame_count = _frame_count(rows)
    dominant = rows.loc[_boolean_column(rows, "mixture_dominant")]
    return {
        "schema": "raft-uav-mmuad-candidate-mixture-assignment-summary-v1",
        "assignment_rows": int(len(rows)),
        "frame_count": int(frame_count),
        "sequence_count": int(rows["sequence_id"].nunique()) if "sequence_id" in rows else 0,
        "dominant_rows": int(len(dominant)),
        "group_columns": [str(column) for column in group_columns],
        "pooled_rows": int(len(pooled)),
        "by_sequence_rows": int(len(by_sequence)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-mixture-assignment-summary",
        description="summarize MMUAD candidate-mixture assignment diagnostics",
    )
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-column", action="append", default=[])
    parser.add_argument("--pooled-csv", type=Path)
    parser.add_argument("--by-sequence-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args(argv)

    assignments = pd.read_csv(args.assignments_csv)
    group_columns = tuple(args.group_column) or DEFAULT_GROUP_COLUMNS
    paths = write_assignment_summary_outputs(
        assignments,
        output_dir=args.output_dir,
        group_columns=group_columns,
        pooled_csv=args.pooled_csv,
        by_sequence_csv=args.by_sequence_csv,
        summary_json=args.summary_json,
    )
    print("mmuad_candidate_mixture_assignment_summary=ok")
    print(f"assignment_rows={len(assignments)}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _summarize_for_group(rows: pd.DataFrame, *, group_column: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    total_frames = max(_frame_count(rows), 1)
    total_mass = float(rows["mixture_final_weight"].sum())
    for label, group in rows.groupby(group_column, dropna=False, sort=True):
        records.append(_summary_record(group, group_column=group_column, label=label, total_frames=total_frames, total_mass=total_mass))
    return pd.DataFrame.from_records(records)


def _summarize_by_sequence(rows: pd.DataFrame, *, group_column: str) -> pd.DataFrame:
    if "sequence_id" not in rows.columns:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for sequence_id, sequence_rows in rows.groupby("sequence_id", dropna=False, sort=True):
        total_frames = max(_frame_count(sequence_rows), 1)
        total_mass = float(sequence_rows["mixture_final_weight"].sum())
        for label, group in sequence_rows.groupby(group_column, dropna=False, sort=True):
            record = _summary_record(
                group,
                group_column=group_column,
                label=label,
                total_frames=total_frames,
                total_mass=total_mass,
            )
            record["sequence_id"] = str(sequence_id)
            records.append(record)
    if not records:
        return pd.DataFrame()
    out = pd.DataFrame.from_records(records)
    leading = ["sequence_id", "group_column", "group_value"]
    return out[leading + [column for column in out.columns if column not in leading]]


def _summary_record(
    group: pd.DataFrame,
    *,
    group_column: str,
    label: Any,
    total_frames: int,
    total_mass: float,
) -> dict[str, Any]:
    weights = pd.to_numeric(group["mixture_final_weight"], errors="coerce").fillna(0.0)
    dominant = group.loc[_boolean_column(group, "mixture_dominant")]
    frames_with_label = _frame_count(group)
    mass = float(weights.sum())
    record = {
        "group_column": str(group_column),
        "group_value": str(label),
        "candidate_rows": int(len(group)),
        "frame_count": int(frames_with_label),
        "frame_fraction": float(frames_with_label / max(total_frames, 1)),
        "responsibility_sum": mass,
        "responsibility_fraction": float(mass / total_mass) if total_mass > 0.0 else 0.0,
        "responsibility_mean": float(weights.mean()) if len(weights) else 0.0,
        "responsibility_p95": _quantile(weights, 0.95),
        "dominant_count": int(len(dominant)),
        "dominant_fraction": float(len(dominant) / max(total_frames, 1)),
    }
    optional_stats = {
        "mixture_sigma_m": "sigma",
        "mixture_distance_to_state_m": "distance_to_state",
        "mixture_normalized_residual": "normalized_residual",
        "mixture_robust_cost": "robust_cost",
        "mixture_raw_score": "raw_score",
        "mixture_normalized_score": "normalized_score",
    }
    for column, prefix in optional_stats.items():
        if column not in group.columns:
            continue
        values = pd.to_numeric(group[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        record[f"{prefix}_mean"] = float(values.mean()) if not values.empty else float("nan")
        record[f"{prefix}_p95"] = float(values.quantile(0.95)) if not values.empty else float("nan")
    return record


def _ensure_assignment_columns(rows: pd.DataFrame) -> None:
    required = ["mixture_final_weight"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"assignment CSV missing required columns: {missing}")
    if "mixture_dominant" not in rows.columns:
        rows["mixture_dominant"] = False
    if "sequence_id" not in rows.columns:
        rows["sequence_id"] = "default"
    if "time_s" not in rows.columns:
        rows["time_s"] = np.arange(len(rows), dtype=float)


def _frame_count(rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    if {"sequence_id", "time_s"}.issubset(rows.columns):
        return int(rows.groupby(["sequence_id", "time_s"], dropna=False).ngroups)
    return int(len(rows))


def _boolean_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(False, index=rows.index)
    values = rows[column]
    if values.dtype == bool:
        return values.fillna(False)
    text = values.fillna(False).astype(str).str.lower()
    return text.isin({"true", "1", "yes", "y"})


def _quantile(values: pd.Series, quantile: float) -> float:
    if values.empty:
        return 0.0
    return float(values.quantile(quantile))


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
