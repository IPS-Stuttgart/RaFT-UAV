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
_TRUE_BOOL_TEXT = frozenset({"true", "t", "yes", "y", "1", "1.0"})
_FALSE_BOOL_TEXT = frozenset(
    {
        "false",
        "f",
        "no",
        "n",
        "0",
        "0.0",
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }
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
        ["sequence_id", "assignment_priority_score", "frame_count"],
        ascending=[True, False, False],
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
        json.dumps(_jsonable(payload), indent=2, allow_nan=False),
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
        rows[column] = _bool_series(rows[column], column_name=column)
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
    state_mse = _mse(rows["state_error_3d_m"])
    oracle_mse = _mse(rows["oracle_error_3d_m"])
    dominant_mse = _mse(rows["dominant_error_3d_m"])
    state_regret_mean = _mean(rows["state_regret_m"])
    dominant_regret_mean = _mean(rows["dominant_regret_m"])
    oracle_weight_mean = _mean(rows["oracle_mixture_weight"])
    oracle_weight_deficit = _safe_difference(1.0, oracle_weight_mean)
    state_oracle_gap = _safe_difference(state_mse, oracle_mse)
    dominant_oracle_gap = _safe_difference(dominant_mse, oracle_mse)
    priority = _assignment_priority_score(
        frame_count=len(rows),
        state_regret_mean=state_regret_mean,
        oracle_weight_deficit=oracle_weight_deficit,
        state_oracle_mse_gap=state_oracle_gap,
    )
    record: dict[str, Any] = {
        "sequence_id": sequence_id,
        "group_label": group_label,
        "frame_count": int(len(rows)),
        "state_error_3d_m_mse": state_mse,
        "oracle_error_3d_m_mse": oracle_mse,
        "dominant_error_3d_m_mse": dominant_mse,
        "state_vs_oracle_mse_gap": state_oracle_gap,
        "dominant_vs_oracle_mse_gap": dominant_oracle_gap,
        "state_regret_m_mean": state_regret_mean,
        "dominant_regret_m_mean": dominant_regret_mean,
        "state_error_3d_m_p95": _quantile(rows["state_error_3d_m"], 0.95),
        "oracle_mixture_weight_mean": oracle_weight_mean,
        "oracle_weight_deficit_mean": oracle_weight_deficit,
        "oracle_weight_rank_p50": _quantile(rows["oracle_weight_rank"], 0.50),
        "candidate_count_mean": _mean(rows["candidate_count"]),
        "dominant_matches_oracle_rate": _mean(rows["dominant_is_oracle"].astype(float)),
        "oracle_in_topk_by_weight_rate": _mean(rows["oracle_in_topk_by_weight"].astype(float)),
        "assignment_priority_score": priority,
    }
    for column in _GROUP_COLUMNS:
        record[column] = str(group_values.get(column, "__all__"))
    return record


def _clean_text(values: pd.Series) -> pd.Series:
    text = values.where(values.notna(), "unknown").astype(str).str.strip()
    missing = text.eq("") | text.str.lower().isin({"nan", "none", "<na>"})
    return text.where(~missing, "unknown")


def _bool_series(values: pd.Series, *, column_name: str | None = None) -> pd.Series:
    input_name = getattr(values, "name", None)
    series = pd.Series(values, copy=False)
    if series.empty:
        return pd.Series(dtype=bool, index=series.index)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean").fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.casefold()
    truthy = (text.isin(_TRUE_BOOL_TEXT) | numeric.eq(1.0)).fillna(False)
    falsy = (
        series.isna() | text.isin(_FALSE_BOOL_TEXT) | numeric.eq(0.0)
    ).fillna(False)
    invalid = ~(truthy | falsy)
    if bool(invalid.any()):
        invalid_indices = invalid[invalid].index.tolist()
        invalid_values = series.loc[invalid_indices].tolist()
        label = column_name or (str(input_name) if input_name is not None else "Boolean values")
        raise ValueError(
            f"{label} contains invalid Boolean values at rows "
            f"{invalid_indices}: {invalid_values}"
        )
    return truthy.astype(bool)


def _finite_numeric(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    return numeric[np.isfinite(numeric)]


def _mse(values: pd.Series) -> float:
    finite = _finite_numeric(values)
    return float(np.mean(np.square(finite))) if finite.size else float("nan")


def _mean(values: pd.Series) -> float:
    finite = _finite_numeric(values)
    return float(np.mean(finite)) if finite.size else float("nan")


def _quantile(values: pd.Series, quantile: float) -> float:
    finite = _finite_numeric(values)
    return float(np.quantile(finite, quantile)) if finite.size else float("nan")


def _safe_difference(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return float(left - right)


def _positive_or_zero(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(max(value, 0.0))


def _assignment_priority_score(
    *,
    frame_count: int,
    state_regret_mean: float,
    oracle_weight_deficit: float,
    state_oracle_mse_gap: float,
) -> float:
    """Return an actionable ranking score for branch/source failure groups.

    The score favors persistent groups that have high state regret, low oracle
    assignment mass, and a large state-vs-oracle MSE gap.  It is diagnostic only:
    it helps choose the next branch/source reservoir experiment, not inference.
    """

    regret = _positive_or_zero(state_regret_mean)
    weight_deficit = _positive_or_zero(oracle_weight_deficit)
    mse_gap = _positive_or_zero(state_oracle_mse_gap)
    return float(frame_count) * regret * (1.0 + weight_deficit) * np.sqrt(1.0 + mse_gap)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
