"""Compare MMUAD candidate pools against a reference/full pool.

The MMUAD top-3 diagnostics showed that some transformed/scored streams lose
or bury oracle-quality candidates that still exist in the raw/full pool.  This
module compares one reference candidate pool against one or more candidate
streams at matched timestamps and reports where the candidate oracle ceiling was
lost, preserved, or merely pushed below the requested top-K.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_oracle_attribution import (
    build_candidate_oracle_attribution_tables,
)
from raft_uav.mmuad.candidate_reservoir import load_candidate_inputs
from raft_uav.mmuad.schema import normalize_truth_columns

_DEFAULT_TOP_K = (1, 3, 5, 10, 20)
_DEFAULT_SCORE_COLUMN = "candidate_reservoir_score"
_DEFAULT_FALLBACK_SCORE_COLUMN = "ranker_score"


def build_candidate_pool_compare_tables(
    reference_candidates: pd.DataFrame,
    candidate_pools: Mapping[str, pd.DataFrame],
    truth: pd.DataFrame,
    *,
    top_k_values: Sequence[int] = _DEFAULT_TOP_K,
    score_column: str = _DEFAULT_SCORE_COLUMN,
    fallback_score_column: str = _DEFAULT_FALLBACK_SCORE_COLUMN,
    max_truth_time_delta_s: float = 0.5,
    good_candidate_threshold_m: float = 5.0,
    loss_tolerance_m: float = 1.0e-6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return frame, pooled, sequence, and branch-delta comparison tables."""

    top_k_tuple = tuple(sorted({int(value) for value in top_k_values if int(value) > 0}))
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    reference_frames, _, _, _ = build_candidate_oracle_attribution_tables(
        reference_candidates,
        truth_rows,
        top_k_values=top_k_tuple,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        max_truth_time_delta_s=max_truth_time_delta_s,
    )
    if reference_frames.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    reference_subset = _reference_frame_subset(reference_frames, top_k_values=top_k_tuple)
    frame_parts: list[pd.DataFrame] = []
    for pool_label, pool_rows in candidate_pools.items():
        candidate_frames, _, _, _ = build_candidate_oracle_attribution_tables(
            pool_rows,
            truth_rows,
            top_k_values=top_k_tuple,
            score_column=score_column,
            fallback_score_column=fallback_score_column,
            max_truth_time_delta_s=max_truth_time_delta_s,
        )
        if candidate_frames.empty:
            candidate_compare = reference_subset.copy()
            candidate_compare["pool_label"] = str(pool_label)
            candidate_compare["pool_frame_present"] = False
            _fill_missing_candidate_columns(candidate_compare, top_k_values=top_k_tuple)
        else:
            candidate_subset = _candidate_frame_subset(candidate_frames, top_k_values=top_k_tuple)
            candidate_compare = reference_subset.merge(
                candidate_subset,
                on=["sequence_id", "time_s"],
                how="left",
                validate="one_to_one",
            )
            candidate_compare["pool_label"] = str(pool_label)
            candidate_compare["pool_frame_present"] = candidate_compare[
                "candidate_oracle_all_3d_m"
            ].notna()
        _add_delta_columns(
            candidate_compare,
            top_k_values=top_k_tuple,
            good_candidate_threshold_m=good_candidate_threshold_m,
            loss_tolerance_m=loss_tolerance_m,
        )
        frame_parts.append(candidate_compare)
    frame_rows = pd.concat(frame_parts, ignore_index=True) if frame_parts else pd.DataFrame()
    if frame_rows.empty:
        empty = pd.DataFrame()
        return frame_rows, empty, empty, empty
    pooled = _pooled_summary(frame_rows, top_k_values=top_k_tuple)
    by_sequence = _by_sequence_summary(frame_rows, top_k_values=top_k_tuple)
    by_branch = _by_reference_branch_summary(frame_rows, top_k_values=top_k_tuple)
    return frame_rows, pooled, by_sequence, by_branch


def write_candidate_pool_compare_outputs(
    *,
    output_dir: Path,
    frame_rows: pd.DataFrame,
    pooled_summary: pd.DataFrame,
    by_sequence: pd.DataFrame,
    by_reference_branch: pd.DataFrame,
) -> dict[str, str]:
    """Write candidate-pool comparison artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "frame_csv": output_dir / "mmuad_candidate_pool_compare_frames.csv",
        "pooled_csv": output_dir / "mmuad_candidate_pool_compare_pooled.csv",
        "by_sequence_csv": output_dir / "mmuad_candidate_pool_compare_by_sequence.csv",
        "by_reference_branch_csv": output_dir
        / "mmuad_candidate_pool_compare_by_reference_branch.csv",
        "summary_json": output_dir / "mmuad_candidate_pool_compare_summary.json",
    }
    frame_rows.to_csv(paths["frame_csv"], index=False)
    pooled_summary.to_csv(paths["pooled_csv"], index=False)
    by_sequence.to_csv(paths["by_sequence_csv"], index=False)
    by_reference_branch.to_csv(paths["by_reference_branch_csv"], index=False)
    summary = {
        "pooled": pooled_summary.to_dict(orient="records"),
        "by_sequence": by_sequence.to_dict(orient="records"),
        "by_reference_branch": by_reference_branch.to_dict(orient="records"),
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-pool-compare",
        description="compare MMUAD candidate pools against a reference/full pool oracle",
    )
    parser.add_argument(
        "--reference-candidate",
        action="append",
        default=[],
        help="reference/full-pool candidate CSV as BRANCH=path; may be repeated",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="candidate pool to compare as LABEL=path; may be repeated",
    )
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-column", default=_DEFAULT_SCORE_COLUMN)
    parser.add_argument("--fallback-score-column", default=_DEFAULT_FALLBACK_SCORE_COLUMN)
    parser.add_argument("--top-k", action="append", type=int, default=None)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--good-candidate-threshold-m", type=float, default=5.0)
    parser.add_argument("--loss-tolerance-m", type=float, default=1.0e-6)
    args = parser.parse_args(argv)

    if not args.reference_candidate:
        raise ValueError("at least one --reference-candidate BRANCH=PATH entry is required")
    if not args.candidate:
        raise ValueError("at least one --candidate LABEL=PATH entry is required")
    top_k_values = tuple(args.top_k) if args.top_k is not None else _DEFAULT_TOP_K
    reference_candidates = load_candidate_inputs(args.reference_candidate)
    if reference_candidates.empty:
        raise ValueError("reference candidate pool is empty")
    candidate_pools = _load_labeled_candidate_pools(args.candidate)
    truth = pd.read_csv(args.truth_csv)
    frame_rows, pooled, by_sequence, by_branch = build_candidate_pool_compare_tables(
        reference_candidates,
        candidate_pools,
        truth,
        top_k_values=top_k_values,
        score_column=args.score_column,
        fallback_score_column=args.fallback_score_column,
        max_truth_time_delta_s=args.max_truth_time_delta_s,
        good_candidate_threshold_m=args.good_candidate_threshold_m,
        loss_tolerance_m=args.loss_tolerance_m,
    )
    paths = write_candidate_pool_compare_outputs(
        output_dir=args.output_dir,
        frame_rows=frame_rows,
        pooled_summary=pooled,
        by_sequence=by_sequence,
        by_reference_branch=by_branch,
    )
    print("mmuad_candidate_pool_compare=ok")
    print(f"frame_rows={len(frame_rows)}")
    if not pooled.empty:
        best_row = pooled.sort_values("oracle_all_mse_delta").iloc[0]
        print(f"best_pool_label={best_row['pool_label']}")
        print(f"best_oracle_all_mse_delta={best_row['oracle_all_mse_delta']}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _load_labeled_candidate_pools(specs: Sequence[str]) -> dict[str, pd.DataFrame]:
    pools: dict[str, list[str]] = {}
    for spec in specs:
        label, path_text = _split_label_path(spec)
        pools.setdefault(label, []).append(f"{label}={path_text}")
    loaded = {label: load_candidate_inputs(label_specs) for label, label_specs in pools.items()}
    empty = [label for label, rows in loaded.items() if rows.empty]
    if empty:
        raise ValueError(f"candidate pools are empty: {empty}")
    return loaded


def _split_label_path(spec: str) -> tuple[str, str]:
    if "=" not in str(spec):
        path = Path(spec)
        return path.stem, str(path)
    label, path_text = str(spec).split("=", 1)
    label = label.strip().replace(" ", "_") or Path(path_text).stem
    return label, path_text


def _reference_frame_subset(
    reference_frames: pd.DataFrame,
    *,
    top_k_values: tuple[int, ...],
) -> pd.DataFrame:
    columns = [
        "sequence_id",
        "time_s",
        "truth_time_delta_s",
        "candidate_count",
        "oracle_all_3d_m",
        "oracle_all_rank",
        "oracle_all_candidate_source",
        "oracle_all_candidate_branch",
        "oracle_all_candidate_track_id",
    ]
    columns.extend([f"oracle_top{top_k}_3d_m" for top_k in top_k_values])
    columns.extend([f"oracle_in_top{top_k}" for top_k in top_k_values])
    subset = reference_frames[[column for column in columns if column in reference_frames.columns]].copy()
    return subset.rename(
        columns={
            column: f"reference_{column}"
            for column in subset.columns
            if column not in {"sequence_id", "time_s"}
        },
    )


def _candidate_frame_subset(
    candidate_frames: pd.DataFrame,
    *,
    top_k_values: tuple[int, ...],
) -> pd.DataFrame:
    columns = [
        "sequence_id",
        "time_s",
        "candidate_count",
        "oracle_all_3d_m",
        "oracle_all_rank",
        "oracle_all_candidate_source",
        "oracle_all_candidate_branch",
        "oracle_all_candidate_track_id",
    ]
    columns.extend([f"oracle_top{top_k}_3d_m" for top_k in top_k_values])
    columns.extend([f"oracle_in_top{top_k}" for top_k in top_k_values])
    subset = candidate_frames[[column for column in columns if column in candidate_frames.columns]].copy()
    return subset.rename(
        columns={
            column: f"candidate_{column}"
            for column in subset.columns
            if column not in {"sequence_id", "time_s"}
        },
    )


def _fill_missing_candidate_columns(rows: pd.DataFrame, *, top_k_values: tuple[int, ...]) -> None:
    candidate_columns = [
        "candidate_candidate_count",
        "candidate_oracle_all_3d_m",
        "candidate_oracle_all_rank",
        "candidate_oracle_all_candidate_source",
        "candidate_oracle_all_candidate_branch",
        "candidate_oracle_all_candidate_track_id",
    ]
    candidate_columns.extend([f"candidate_oracle_top{top_k}_3d_m" for top_k in top_k_values])
    candidate_columns.extend([f"candidate_oracle_in_top{top_k}" for top_k in top_k_values])
    for column in candidate_columns:
        if column not in rows.columns:
            rows[column] = np.nan


def _add_delta_columns(
    rows: pd.DataFrame,
    *,
    top_k_values: tuple[int, ...],
    good_candidate_threshold_m: float,
    loss_tolerance_m: float,
) -> None:
    rows["reference_has_good_candidate"] = (
        pd.to_numeric(rows["reference_oracle_all_3d_m"], errors="coerce")
        <= float(good_candidate_threshold_m)
    )
    rows["candidate_has_good_candidate"] = (
        pd.to_numeric(rows["candidate_oracle_all_3d_m"], errors="coerce")
        <= float(good_candidate_threshold_m)
    )
    rows["good_candidate_lost"] = rows["reference_has_good_candidate"] & (~rows["candidate_has_good_candidate"])
    rows["oracle_all_delta_m"] = (
        pd.to_numeric(rows["candidate_oracle_all_3d_m"], errors="coerce")
        - pd.to_numeric(rows["reference_oracle_all_3d_m"], errors="coerce")
    )
    rows["oracle_all_mse_delta_per_frame"] = (
        pd.to_numeric(rows["candidate_oracle_all_3d_m"], errors="coerce") ** 2
        - pd.to_numeric(rows["reference_oracle_all_3d_m"], errors="coerce") ** 2
    )
    rows["oracle_ceiling_worse"] = rows["oracle_all_delta_m"] > float(loss_tolerance_m)
    for top_k in top_k_values:
        ref_col = f"reference_oracle_top{top_k}_3d_m"
        cand_col = f"candidate_oracle_top{top_k}_3d_m"
        if ref_col in rows.columns and cand_col in rows.columns:
            rows[f"oracle_top{top_k}_delta_m"] = (
                pd.to_numeric(rows[cand_col], errors="coerce")
                - pd.to_numeric(rows[ref_col], errors="coerce")
            )
            rows[f"oracle_top{top_k}_mse_delta_per_frame"] = (
                pd.to_numeric(rows[cand_col], errors="coerce") ** 2
                - pd.to_numeric(rows[ref_col], errors="coerce") ** 2
            )
            rows[f"oracle_top{top_k}_worse"] = rows[f"oracle_top{top_k}_delta_m"] > float(
                loss_tolerance_m,
            )


def _pooled_summary(frame_rows: pd.DataFrame, *, top_k_values: tuple[int, ...]) -> pd.DataFrame:
    records = [
        _summarize_group(group, pool_label=str(pool_label), top_k_values=top_k_values)
        for pool_label, group in frame_rows.groupby("pool_label", sort=True)
    ]
    return pd.DataFrame.from_records(records).sort_values("oracle_all_mse_delta").reset_index(drop=True)


def _by_sequence_summary(frame_rows: pd.DataFrame, *, top_k_values: tuple[int, ...]) -> pd.DataFrame:
    records = []
    for (pool_label, sequence_id), group in frame_rows.groupby(["pool_label", "sequence_id"], sort=True):
        record = _summarize_group(group, pool_label=str(pool_label), top_k_values=top_k_values)
        record["sequence_id"] = str(sequence_id)
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values(
        ["oracle_all_mse_delta", "pool_label", "sequence_id"],
    ).reset_index(drop=True)


def _by_reference_branch_summary(
    frame_rows: pd.DataFrame,
    *,
    top_k_values: tuple[int, ...],
) -> pd.DataFrame:
    branch_col = "reference_oracle_all_candidate_branch"
    if branch_col not in frame_rows.columns:
        return pd.DataFrame()
    records = []
    for (pool_label, branch), group in frame_rows.groupby(["pool_label", branch_col], sort=True):
        record = _summarize_group(group, pool_label=str(pool_label), top_k_values=top_k_values)
        record["reference_candidate_branch"] = str(branch)
        records.append(record)
    return pd.DataFrame.from_records(records).sort_values(
        ["oracle_all_mse_delta", "pool_label", "reference_candidate_branch"],
    ).reset_index(drop=True)


def _summarize_group(
    group: pd.DataFrame,
    *,
    pool_label: str,
    top_k_values: tuple[int, ...],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "pool_label": pool_label,
        "frame_count": int(len(group)),
        "pool_frame_present_fraction": _mean_bool(group.get("pool_frame_present")),
        "reference_good_candidate_fraction": _mean_bool(group.get("reference_has_good_candidate")),
        "candidate_good_candidate_fraction": _mean_bool(group.get("candidate_has_good_candidate")),
        "good_candidate_lost_fraction": _mean_bool(group.get("good_candidate_lost")),
        "oracle_ceiling_worse_fraction": _mean_bool(group.get("oracle_ceiling_worse")),
        "reference_candidate_count_mean": _mean_numeric(group.get("reference_candidate_count")),
        "candidate_candidate_count_mean": _mean_numeric(group.get("candidate_candidate_count")),
    }
    _add_mse_pair(record, group, reference="reference_oracle_all_3d_m", candidate="candidate_oracle_all_3d_m", prefix="oracle_all")
    record["oracle_all_delta_m_mean"] = _mean_numeric(group.get("oracle_all_delta_m"))
    record["oracle_all_delta_m_p95"] = _quantile_numeric(group.get("oracle_all_delta_m"), 0.95)
    for top_k in top_k_values:
        _add_mse_pair(
            record,
            group,
            reference=f"reference_oracle_top{top_k}_3d_m",
            candidate=f"candidate_oracle_top{top_k}_3d_m",
            prefix=f"oracle_top{top_k}",
        )
        record[f"oracle_top{top_k}_worse_fraction"] = _mean_bool(
            group.get(f"oracle_top{top_k}_worse"),
        )
    return record


def _add_mse_pair(
    record: dict[str, Any],
    group: pd.DataFrame,
    *,
    reference: str,
    candidate: str,
    prefix: str,
) -> None:
    ref = pd.to_numeric(group.get(reference), errors="coerce") if reference in group else pd.Series(dtype=float)
    cand = pd.to_numeric(group.get(candidate), errors="coerce") if candidate in group else pd.Series(dtype=float)
    record[f"reference_{prefix}_mse"] = _mse(ref)
    record[f"candidate_{prefix}_mse"] = _mse(cand)
    record[f"{prefix}_mse_delta"] = record[f"candidate_{prefix}_mse"] - record[f"reference_{prefix}_mse"]


def _mse(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(np.mean(numeric.to_numpy(float) ** 2)) if not numeric.empty else float("nan")


def _mean_numeric(values: pd.Series | None) -> float:
    if values is None:
        return float("nan")
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else float("nan")


def _quantile_numeric(values: pd.Series | None, quantile: float) -> float:
    if values is None:
        return float("nan")
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(quantile)) if not numeric.empty else float("nan")


def _mean_bool(values: pd.Series | None) -> float:
    if values is None:
        return float("nan")
    if values.empty:
        return float("nan")
    return float(values.fillna(False).astype(bool).mean())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
