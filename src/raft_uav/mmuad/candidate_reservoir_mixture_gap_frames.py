"""Frame-level reservoir-oracle vs mixture-achieved gap diagnostics for MMUAD.

The reservoir-mixture runner can now report pooled and per-sequence oracle gaps,
but the next MMUAD tuning loop needs to know *which frames* create the gap.  This
module joins mixture estimates with reservoir oracle-recall rows and writes a
frame table plus compact pooled/per-sequence summaries.  It is diagnostic only:
truth is needed only to compute or verify local public-validation errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.class_probability_csv import read_sequence_text_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.schema import normalize_truth_columns
from raft_uav.mmuad.tracker import add_truth_errors

ERROR_COLUMN_CANDIDATES = (
    "position_error_3d_m",
    "error_3d_m",
    "state_error_3d_m",
    "mixture_error_3d_m",
)
ORACLE_PREFIX = "oracle_"
ORACLE_SUFFIX = "_3d_m"
DEFAULT_GAP_THRESHOLDS_M = (1.0, 2.0, 5.0, 10.0, 20.0)
OPTIONAL_ESTIMATE_COLUMNS = (
    "state_x_m",
    "state_y_m",
    "state_z_m",
    "mixture_effective_candidate_count",
    "mixture_assignment_entropy",
    "mixture_dominant_weight",
    "mixture_effective_sigma_m",
)


def build_frame_gap_table(
    estimates: pd.DataFrame,
    oracle_frames: pd.DataFrame,
    *,
    truth: pd.DataFrame | None = None,
    time_round_decimals: int = 6,
    time_join_tolerance_s: float | None = None,
) -> pd.DataFrame:
    """Join mixture estimates to oracle rows and compute frame-level gaps.

    By default, the join preserves the historical rounded-time exact merge.  Set
    ``time_join_tolerance_s`` to use a nearest-neighbor per-sequence time join
    instead.  The tolerance mode is useful when reservoir-oracle diagnostics and
    mixture estimates were emitted by adjacent tools whose timestamps differ by
    small floating-point or template-resampling offsets.
    """

    estimate_rows = _with_mixture_error(estimates, truth=truth).copy()
    oracle_rows = pd.DataFrame(oracle_frames).copy()
    if estimate_rows.empty or oracle_rows.empty:
        return pd.DataFrame()
    _require_columns(
        estimate_rows,
        ("sequence_id", "time_s", "mixture_error_3d_m"),
        "estimates",
    )
    _require_columns(oracle_rows, ("sequence_id", "time_s"), "oracle frames")
    keep_estimate_cols = _estimate_keep_columns(estimate_rows)
    merged = _join_estimates_to_oracle(
        estimate_rows[keep_estimate_cols],
        oracle_rows,
        time_round_decimals=int(time_round_decimals),
        time_join_tolerance_s=time_join_tolerance_s,
    )
    if merged.empty:
        return merged.drop(
            columns=["_join_time_s", "_oracle_time_s", "_mixture_time_s"],
            errors="ignore",
        )
    merged = _with_join_time_columns(merged)
    mixture_error = pd.to_numeric(merged["mixture_error_3d_m"], errors="coerce")
    merged["mixture_mse_contribution_m2"] = mixture_error**2
    for column in _oracle_distance_columns(merged.columns):
        label = _oracle_label(column)
        oracle_error = pd.to_numeric(merged[column], errors="coerce")
        merged[f"{label}_mse_contribution_m2"] = oracle_error**2
        merged[f"gap_to_{label}_3d_m"] = mixture_error - oracle_error
        merged[f"gap_to_{label}_mse_contribution_m2"] = (
            merged["mixture_mse_contribution_m2"] - merged[f"{label}_mse_contribution_m2"]
        )
        merged[f"ratio_to_{label}_error"] = _safe_divide(mixture_error, oracle_error)
    return merged.drop(
        columns=["_join_time_s", "_oracle_time_s", "_mixture_time_s"],
        errors="ignore",
    ).sort_values(
        ["sequence_id", "time_s"],
    ).reset_index(drop=True)


def summarize_frame_gap(
    frame_gap: pd.DataFrame,
    *,
    group_column: str | None = None,
    gap_thresholds_m: Iterable[float] = DEFAULT_GAP_THRESHOLDS_M,
) -> pd.DataFrame:
    """Build pooled or grouped summary rows from a frame gap table."""

    rows = pd.DataFrame(frame_gap).copy()
    if rows.empty:
        return pd.DataFrame()
    if group_column is None:
        groups = [("__pooled__", rows)]
        label_column = "group"
    else:
        groups = [
            (str(key), group)
            for key, group in rows.groupby(group_column, sort=True)
        ]
        label_column = group_column
    records: list[dict[str, Any]] = []
    for label, group in groups:
        record: dict[str, Any] = {label_column: label, "frame_count": int(len(group))}
        mixture_errors = pd.to_numeric(
            group.get("mixture_error_3d_m"),
            errors="coerce",
        ).dropna()
        record.update(_error_summary("mixture", mixture_errors))
        for column in _oracle_distance_columns(group.columns):
            oracle_label = _oracle_label(column)
            oracle_errors = pd.to_numeric(group[column], errors="coerce").dropna()
            record.update(_error_summary(oracle_label, oracle_errors))
            gap_column = f"gap_to_{oracle_label}_mse_contribution_m2"
            if gap_column in group.columns:
                gaps = pd.to_numeric(group[gap_column], errors="coerce").dropna()
                record[f"gap_to_{oracle_label}_mse_3d_m2"] = _safe_mean(gaps)
            error_gap_column = f"gap_to_{oracle_label}_3d_m"
            if error_gap_column in group.columns:
                error_gaps = pd.to_numeric(
                    group[error_gap_column],
                    errors="coerce",
                ).dropna()
                record[f"gap_to_{oracle_label}_mean_3d_m"] = _safe_mean(error_gaps)
                record[f"gap_to_{oracle_label}_p95_3d_m"] = _safe_quantile(
                    error_gaps,
                    0.95,
                )
                for threshold in gap_thresholds_m:
                    record[f"frames_gap_to_{oracle_label}_gt_{threshold:g}m"] = int(
                        (error_gaps > float(threshold)).sum()
                    )
            oracle_mse = record.get(f"{oracle_label}_mse_3d_m2")
            mixture_mse = record.get("mixture_mse_3d_m2")
            record[f"ratio_to_{oracle_label}_mse"] = _safe_ratio(mixture_mse, oracle_mse)
        records.append(_jsonable(record))
    return pd.DataFrame.from_records(records)


def write_gap_outputs(
    frame_gap: pd.DataFrame,
    *,
    output_frame_csv: Path,
    output_summary_csv: Path,
    output_by_sequence_csv: Path | None = None,
    output_json: Path | None = None,
) -> dict[str, Path]:
    """Write frame, pooled, per-sequence, and optional JSON summaries."""

    paths: dict[str, Path] = {}
    output_frame_csv.parent.mkdir(parents=True, exist_ok=True)
    frame_gap.to_csv(output_frame_csv, index=False)
    paths["frame_gap_csv"] = output_frame_csv
    pooled = summarize_frame_gap(frame_gap)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(output_summary_csv, index=False)
    paths["summary_csv"] = output_summary_csv
    by_sequence = summarize_frame_gap(frame_gap, group_column="sequence_id")
    if output_by_sequence_csv is not None:
        output_by_sequence_csv.parent.mkdir(parents=True, exist_ok=True)
        by_sequence.to_csv(output_by_sequence_csv, index=False)
        paths["by_sequence_csv"] = output_by_sequence_csv
    if output_json is not None:
        payload = {
            "frame_count": int(len(frame_gap)),
            "pooled": _records(pooled),
            "by_sequence": _records(by_sequence),
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        paths["summary_json"] = output_json
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-reservoir-mixture-gap-frames",
        description="write frame-level mixture-vs-reservoir-oracle gap diagnostics",
    )
    parser.add_argument("--estimates-csv", type=Path, required=True)
    parser.add_argument("--oracle-frame-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--output-frame-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-by-sequence-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--time-round-decimals", type=int, default=6)
    parser.add_argument(
        "--time-join-tolerance-s",
        type=float,
        help=(
            "use nearest per-sequence estimate/oracle timestamp matching within this "
            "tolerance; omit to keep exact rounded-time matching"
        ),
    )
    args = parser.parse_args(argv)

    estimates = read_sequence_text_csv(args.estimates_csv)
    oracle_frames = read_sequence_text_csv(args.oracle_frame_csv)
    truth = None if args.truth_csv is None else load_evaluation_truth_file(args.truth_csv).rows
    frame_gap = build_frame_gap_table(
        estimates,
        oracle_frames,
        truth=truth,
        time_round_decimals=int(args.time_round_decimals),
        time_join_tolerance_s=args.time_join_tolerance_s,
    )
    paths = write_gap_outputs(
        frame_gap,
        output_frame_csv=args.output_frame_csv,
        output_summary_csv=args.output_summary_csv,
        output_by_sequence_csv=args.output_by_sequence_csv,
        output_json=args.output_json,
    )
    print("mmuad_reservoir_mixture_gap_frames=ok")
    print(f"frame_count={len(frame_gap)}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _with_mixture_error(estimates: pd.DataFrame, *, truth: pd.DataFrame | None) -> pd.DataFrame:
    rows = pd.DataFrame(estimates).copy()
    if rows.empty:
        return rows.assign(mixture_error_3d_m=pd.Series(dtype=float))
    for column in ERROR_COLUMN_CANDIDATES:
        if column in rows.columns:
            rows["mixture_error_3d_m"] = pd.to_numeric(rows[column], errors="coerce")
            return rows
    if truth is None:
        raise ValueError(
            "estimates must contain a 3D error column or --truth-csv must be supplied"
        )
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    rows = add_truth_errors(rows, truth_rows)
    for column in ERROR_COLUMN_CANDIDATES:
        if column in rows.columns:
            rows["mixture_error_3d_m"] = pd.to_numeric(rows[column], errors="coerce")
            return rows
    raise ValueError("could not compute mixture_error_3d_m from estimates and truth")


def _estimate_keep_columns(estimate_rows: pd.DataFrame) -> list[str]:
    keep_estimate_cols = [
        "sequence_id",
        "time_s",
        "mixture_error_3d_m",
    ]
    for column in OPTIONAL_ESTIMATE_COLUMNS:
        if column in estimate_rows.columns:
            keep_estimate_cols.append(column)
    return keep_estimate_cols


def _join_estimates_to_oracle(
    estimate_rows: pd.DataFrame,
    oracle_rows: pd.DataFrame,
    *,
    time_round_decimals: int,
    time_join_tolerance_s: float | None,
) -> pd.DataFrame:
    if time_join_tolerance_s is None:
        return _join_estimates_to_oracle_exact(
            estimate_rows,
            oracle_rows,
            time_round_decimals=time_round_decimals,
        )
    return _join_estimates_to_oracle_nearest(
        estimate_rows,
        oracle_rows,
        tolerance_s=float(time_join_tolerance_s),
    )


def _join_estimates_to_oracle_exact(
    estimate_rows: pd.DataFrame,
    oracle_rows: pd.DataFrame,
    *,
    time_round_decimals: int,
) -> pd.DataFrame:
    estimate_rows = estimate_rows.copy()
    oracle_rows = oracle_rows.copy()
    estimate_rows["_join_time_s"] = _rounded_time(estimate_rows["time_s"], time_round_decimals)
    oracle_rows["_join_time_s"] = _rounded_time(oracle_rows["time_s"], time_round_decimals)
    return oracle_rows.merge(
        estimate_rows,
        on=["sequence_id", "_join_time_s"],
        how="inner",
        suffixes=("_oracle", "_mixture"),
    )


def _join_estimates_to_oracle_nearest(
    estimate_rows: pd.DataFrame,
    oracle_rows: pd.DataFrame,
    *,
    tolerance_s: float,
) -> pd.DataFrame:
    if tolerance_s < 0.0:
        raise ValueError("time_join_tolerance_s must be non-negative")
    estimate_rows = estimate_rows.copy()
    oracle_rows = oracle_rows.copy()
    estimate_rows["time_s"] = pd.to_numeric(estimate_rows["time_s"], errors="coerce")
    oracle_rows["time_s"] = pd.to_numeric(oracle_rows["time_s"], errors="coerce")
    estimate_rows = estimate_rows.loc[estimate_rows["time_s"].notna()]
    oracle_rows = oracle_rows.loc[oracle_rows["time_s"].notna()]
    if estimate_rows.empty or oracle_rows.empty:
        return pd.DataFrame()
    estimate_rows["_sequence_key"] = estimate_rows["sequence_id"].astype(str)
    oracle_rows["_sequence_key"] = oracle_rows["sequence_id"].astype(str)
    parts: list[pd.DataFrame] = []
    for sequence_key, oracle_group in oracle_rows.groupby("_sequence_key", sort=True):
        estimate_group = estimate_rows.loc[estimate_rows["_sequence_key"] == str(sequence_key)]
        if estimate_group.empty:
            continue
        right = estimate_group.drop(columns=["sequence_id", "_sequence_key"], errors="ignore").copy()
        right["estimate_time_s"] = right["time_s"]
        merged = pd.merge_asof(
            oracle_group.sort_values("time_s"),
            right.sort_values("time_s"),
            on="time_s",
            direction="nearest",
            tolerance=float(tolerance_s),
            suffixes=("_oracle", "_mixture"),
        )
        merged = merged.loc[merged["mixture_error_3d_m"].notna()]
        parts.append(merged)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop(columns=["_sequence_key"], errors="ignore")


def _with_join_time_columns(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    if "time_s_oracle" in out.columns:
        out["oracle_time_s"] = pd.to_numeric(out["time_s_oracle"], errors="coerce")
    else:
        out["oracle_time_s"] = pd.to_numeric(out["time_s"], errors="coerce")
    if "time_s_mixture" in out.columns:
        out["mixture_time_s"] = pd.to_numeric(out["time_s_mixture"], errors="coerce")
    elif "estimate_time_s" in out.columns:
        out["mixture_time_s"] = pd.to_numeric(out["estimate_time_s"], errors="coerce")
    else:
        out["mixture_time_s"] = pd.to_numeric(out["time_s"], errors="coerce")
    out["time_s"] = out["oracle_time_s"]
    out["time_delta_s"] = out["mixture_time_s"] - out["oracle_time_s"]
    return out


def _require_columns(rows: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _rounded_time(values: pd.Series, decimals: int) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").round(int(decimals))


def _oracle_distance_columns(columns: Iterable[str]) -> list[str]:
    return [
        str(column)
        for column in columns
        if str(column).startswith(ORACLE_PREFIX) and str(column).endswith(ORACLE_SUFFIX)
    ]


def _oracle_label(column: str) -> str:
    return column.removesuffix(ORACLE_SUFFIX)


def _error_summary(prefix: str, errors: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(errors, errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        return {
            f"{prefix}_mean_3d_m": None,
            f"{prefix}_mse_3d_m2": None,
            f"{prefix}_rmse_3d_m": None,
            f"{prefix}_p95_3d_m": None,
            f"{prefix}_max_3d_m": None,
        }
    mse = float(np.mean(values**2))
    return {
        f"{prefix}_mean_3d_m": float(np.mean(values)),
        f"{prefix}_mse_3d_m2": mse,
        f"{prefix}_rmse_3d_m": float(np.sqrt(mse)),
        f"{prefix}_p95_3d_m": float(np.quantile(values, 0.95)),
        f"{prefix}_max_3d_m": float(np.max(values)),
    }


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(numeric.mean())


def _safe_quantile(values: pd.Series, q: float) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return None if numeric.empty else float(numeric.quantile(q))


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0.0, np.nan)
    return numerator / denominator


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        numerator_f = float(numerator)
        denominator_f = float(denominator)
    except (TypeError, ValueError):
        return None
    if (
        not np.isfinite(numerator_f)
        or not np.isfinite(denominator_f)
        or denominator_f == 0.0
    ):
        return None
    return numerator_f / denominator_f


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [_jsonable(record) for record in frame.to_dict(orient="records")]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
