"""Summarize MMUAD candidate-mixture assignment diagnostics by branch/source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BRANCH_SUMMARY_CSV = "mmuad_candidate_assignment_branch_summary.csv"
BRANCH_SUMMARY_JSON = "mmuad_candidate_assignment_branch_summary.json"
_GROUP_COLUMNS = (
    "assignment_failure_mode",
    "oracle_candidate_branch",
    "dominant_candidate_branch",
    "oracle_source",
    "dominant_source",
)


def build_candidate_assignment_branch_summary(frame_rows: pd.DataFrame) -> pd.DataFrame:
    rows = _normalized_frame_rows(frame_rows)
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    records.extend(_group_records(rows, sequence_id="__pooled__"))
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        records.extend(_group_records(group, sequence_id=str(sequence_id)))
    return pd.DataFrame.from_records(records).sort_values(
        ["sequence_id", "frame_count"],
        ascending=[True, False],
    )


def write_candidate_assignment_branch_summary(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    provenance: dict[str, Any] | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "branch_summary_csv": output / BRANCH_SUMMARY_CSV,
        "branch_summary_json": output / BRANCH_SUMMARY_JSON,
    }
    summary.to_csv(paths["branch_summary_csv"], index=False)
    payload = dict(provenance or {})
    payload.update(
        {
            "schema": "raft-uav-mmuad-candidate-assignment-branch-summary-v1",
            "row_count": int(len(summary)),
            "summary": summary.to_dict(orient="records"),
        }
    )
    paths["branch_summary_json"].write_text(
        json.dumps(_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    return {key: str(value) for key, value in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-assignment-branch-summary",
        description="summarize MMUAD candidate-mixture assignment diagnostics by branch/source",
    )
    parser.add_argument("--frame-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    frame_rows = pd.read_csv(args.frame_csv)
    summary = build_candidate_assignment_branch_summary(frame_rows)
    paths = write_candidate_assignment_branch_summary(
        output_dir=args.output_dir,
        summary=summary,
        provenance={"frame_csv": str(args.frame_csv)},
    )
    print("mmuad_candidate_assignment_branch_summary=ok")
    print(f"summary_rows={len(summary)}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _normalized_frame_rows(frame_rows: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(frame_rows).copy()
    if rows.empty:
        return rows
    if "sequence_id" not in rows.columns:
        raise ValueError("assignment frame rows missing required column 'sequence_id'")
    if "assignment_failure_mode" not in rows.columns:
        rows["assignment_failure_mode"] = "unknown"
    for column in _GROUP_COLUMNS:
        if column not in rows.columns:
            rows[column] = "unknown"
        rows[column] = _clean_text(rows[column])
    rows["sequence_id"] = _clean_text(rows["sequence_id"])
    for column in (
        "state_error_3d_m",
        "oracle_error_3d_m",
        "dominant_error_3d_m",
        "state_regret_m",
        "dominant_regret_m",
        "oracle_mixture_weight",
        "oracle_weight_rank",
        "candidate_count",
    ):
        if column not in rows.columns:
            rows[column] = np.nan
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    for column in ("dominant_is_oracle", "oracle_in_topk_by_weight"):
        if column not in rows.columns:
            rows[column] = False
        rows[column] = _bool_series(rows[column])
    return rows


def _group_records(rows: pd.DataFrame, *, sequence_id: str) -> list[dict[str, Any]]:
    records = [_summary_record(rows, sequence_id=sequence_id, group_label="__all__")]
    for keys, group in rows.groupby(list(_GROUP_COLUMNS), sort=True, dropna=False):
        records.append(
            _summary_record(
                group,
                sequence_id=sequence_id,
                group_label="branch_source_failure",
                group_values=dict(zip(_GROUP_COLUMNS, keys, strict=False)),
            )
        )
    return records


def _summary_record(
    rows: pd.DataFrame,
    *,
    sequence_id: str,
    group_label: str,
    group_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    group_values = group_values or {}
    record: dict[str, Any] = {
        "sequence_id": sequence_id,
        "group_label": group_label,
        "frame_count": int(len(rows)),
        "state_error_3d_m_mse": _mse(rows["state_error_3d_m"]),
        "oracle_error_3d_m_mse": _mse(rows["oracle_error_3d_m"]),
        "dominant_error_3d_m_mse": _mse(rows["dominant_error_3d_m"]),
        "state_regret_m_mean": _mean(rows["state_regret_m"]),
        "dominant_regret_m_mean": _mean(rows["dominant_regret_m"]),
        "state_error_3d_m_p95": _quantile(rows["state_error_3d_m"], 0.95),
        "oracle_mixture_weight_mean": _mean(rows["oracle_mixture_weight"]),
        "oracle_weight_rank_p50": _quantile(rows["oracle_weight_rank"], 0.50),
        "candidate_count_mean": _mean(rows["candidate_count"]),
        "dominant_matches_oracle_rate": _mean(rows["dominant_is_oracle"].astype(float)),
        "oracle_in_topk_by_weight_rate": _mean(rows["oracle_in_topk_by_weight"].astype(float)),
    }
    for column in _GROUP_COLUMNS:
        record[column] = str(group_values.get(column, "__all__"))
    return record


def _clean_text(values: pd.Series) -> pd.Series:
    text = values.where(values.notna(), "unknown").astype(str).str.strip()
    missing = text.eq("") | text.str.lower().isin({"nan", "none", "<na>"})
    return text.where(~missing, "unknown")


def _bool_series(values: pd.Series) -> pd.Series:
    series = pd.Series(values)
    if series.empty:
        return pd.Series(dtype=bool, index=series.index)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_truthy = numeric.notna() & (numeric != 0.0)
    text = series.where(series.notna(), "").astype(str).str.strip().str.lower()
    truthy_text = text.isin({"true", "t", "yes", "y", "1", "1.0"})
    falsy_text = text.isin(
        {"false", "f", "no", "n", "0", "0.0", "", "nan", "none", "<na>", "nat"}
    )
    return (truthy_text | (numeric_truthy & ~falsy_text)).astype(bool)


def _mse(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    return float(np.mean(np.square(series.to_numpy(float)))) if not series.empty else float("nan")


def _mean(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    return float(series.mean()) if not series.empty else float("nan")


def _quantile(values: pd.Series, quantile: float) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    return float(series.quantile(quantile)) if not series.empty else float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
