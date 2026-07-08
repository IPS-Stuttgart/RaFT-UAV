"""Ablate MMUAD candidate-pool branches against the full-pool oracle.

The MMUAD top-3 work repeatedly showed that a useful candidate can disappear
when raw, dynamic, source-translated, or merged streams are replaced too early.
This diagnostic starts from a single full candidate pool and automatically
compares the full-pool oracle with leave-one-branch/source and only-one-branch/
source pools.  It is validation/train-only when truth is supplied and helps
choose which candidate branch should be preserved, boosted, or repaired before
mixture-MAP smoothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from raft_uav.mmuad.candidate_pool_compare import _DEFAULT_FALLBACK_SCORE_COLUMN
from raft_uav.mmuad.candidate_pool_compare import _DEFAULT_SCORE_COLUMN
from raft_uav.mmuad.candidate_pool_compare import _DEFAULT_TOP_K
from raft_uav.mmuad.candidate_pool_compare import build_candidate_pool_compare_tables
from raft_uav.mmuad.candidate_pool_compare import write_candidate_pool_compare_outputs
from raft_uav.mmuad.candidate_reservoir import load_candidate_inputs

ABLATION_SUMMARY_JSON = "mmuad_candidate_pool_branch_ablation_summary.json"
ABLATION_MANIFEST_CSV = "mmuad_candidate_pool_branch_ablation_manifest.csv"
_GROUP_COLUMNS = ("candidate_branch", "source")
_TRUE_TEXT = {"1", "true", "t", "yes", "y"}
_FALSE_TEXT = {"0", "false", "f", "no", "n"}


def build_candidate_pool_branch_ablation_pools(
    candidates: pd.DataFrame,
    *,
    group_column: str = "candidate_branch",
    include_full_pool: bool = True,
    include_leave_one_out: bool = True,
    include_only_one: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return full/without/only candidate pools plus an ablation manifest."""

    rows = pd.DataFrame(candidates).copy()
    if rows.empty:
        return {}, pd.DataFrame()
    if group_column not in rows.columns:
        rows[group_column] = "unknown"
    rows[group_column] = _clean_label_series(rows[group_column])

    pools: dict[str, pd.DataFrame] = {}
    manifest_records: list[dict[str, Any]] = []
    if include_full_pool:
        label = "full_pool"
        pools[label] = rows.copy()
        manifest_records.append(
            _manifest_record(
                pool_label=label,
                ablation_type="full_pool",
                group_column=group_column,
                group_value="__all__",
                rows=pools[label],
            )
        )

    for group_value in sorted(rows[group_column].dropna().astype(str).unique().tolist()):
        slug = _slug(group_value)
        if include_leave_one_out:
            without = rows.loc[rows[group_column].astype(str) != group_value].copy()
            if not without.empty:
                label = f"without_{group_column}_{slug}"
                pools[label] = without
                manifest_records.append(
                    _manifest_record(
                        pool_label=label,
                        ablation_type="without_group",
                        group_column=group_column,
                        group_value=group_value,
                        rows=without,
                    )
                )
        if include_only_one:
            only = rows.loc[rows[group_column].astype(str) == group_value].copy()
            if not only.empty:
                label = f"only_{group_column}_{slug}"
                pools[label] = only
                manifest_records.append(
                    _manifest_record(
                        pool_label=label,
                        ablation_type="only_group",
                        group_column=group_column,
                        group_value=group_value,
                        rows=only,
                    )
                )
    return pools, pd.DataFrame.from_records(manifest_records)


def build_candidate_pool_branch_ablation_tables(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    group_column: str = "candidate_branch",
    top_k_values: Sequence[int] = _DEFAULT_TOP_K,
    score_column: str = _DEFAULT_SCORE_COLUMN,
    fallback_score_column: str = _DEFAULT_FALLBACK_SCORE_COLUMN,
    max_truth_time_delta_s: float = 0.5,
    good_candidate_threshold_m: float = 5.0,
    loss_tolerance_m: float = 1.0e-6,
    include_full_pool: bool = True,
    include_leave_one_out: bool = True,
    include_only_one: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build candidate-pool comparison tables for branch/source ablations."""

    candidate_pools, manifest = build_candidate_pool_branch_ablation_pools(
        candidates,
        group_column=group_column,
        include_full_pool=include_full_pool,
        include_leave_one_out=include_leave_one_out,
        include_only_one=include_only_one,
    )
    if not candidate_pools:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, manifest
    frame_rows, pooled, by_sequence, by_branch = build_candidate_pool_compare_tables(
        candidates,
        candidate_pools,
        truth,
        top_k_values=top_k_values,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        max_truth_time_delta_s=max_truth_time_delta_s,
        good_candidate_threshold_m=good_candidate_threshold_m,
        loss_tolerance_m=loss_tolerance_m,
    )
    pooled = _attach_manifest(pooled, manifest)
    by_sequence = _attach_manifest(by_sequence, manifest)
    by_branch = _attach_manifest(by_branch, manifest)
    frame_rows = _attach_manifest(frame_rows, manifest)
    return frame_rows, pooled, by_sequence, by_branch, manifest


def write_candidate_pool_branch_ablation_outputs(
    *,
    output_dir: Path,
    frame_rows: pd.DataFrame,
    pooled_summary: pd.DataFrame,
    by_sequence: pd.DataFrame,
    by_reference_branch: pd.DataFrame,
    manifest: pd.DataFrame,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write ablation tables and a compact JSON summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        key: Path(value)
        for key, value in write_candidate_pool_compare_outputs(
            output_dir=output,
            frame_rows=frame_rows,
            pooled_summary=pooled_summary,
            by_sequence=by_sequence,
            by_reference_branch=by_reference_branch,
        ).items()
    }
    manifest_path = output / ABLATION_MANIFEST_CSV
    manifest.to_csv(manifest_path, index=False)
    paths["ablation_manifest_csv"] = manifest_path
    summary_path = output / ABLATION_SUMMARY_JSON
    payload = {
        "schema": "raft-uav-mmuad-candidate-pool-branch-ablation-v1",
        "provenance": dict(provenance or {}),
        "pool_count": int(len(manifest)),
        "best_by_oracle_all_mse_delta": _best_record(pooled_summary, "oracle_all_mse_delta", ascending=True),
        "worst_by_oracle_all_mse_delta": _best_record(
            pooled_summary,
            "oracle_all_mse_delta",
            ascending=False,
        ),
        "manifest": manifest.to_dict(orient="records"),
        "pooled": pooled_summary.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    paths["ablation_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-pool-branch-ablation",
        description=(
            "run leave-one-out and only-one branch/source oracle ablations for an MMUAD "
            "candidate pool"
        ),
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
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-column", choices=_GROUP_COLUMNS, default="candidate_branch")
    parser.add_argument("--score-column", default=_DEFAULT_SCORE_COLUMN)
    parser.add_argument("--fallback-score-column", default=_DEFAULT_FALLBACK_SCORE_COLUMN)
    parser.add_argument("--top-k", action="append", type=int, default=None)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--good-candidate-threshold-m", type=float, default=5.0)
    parser.add_argument("--loss-tolerance-m", type=float, default=1.0e-6)
    parser.add_argument("--include-full-pool", default="true")
    parser.add_argument("--include-leave-one-out", default="true")
    parser.add_argument("--include-only-one", default="true")
    args = parser.parse_args(argv)

    candidate_specs = [*args.candidate, *args.candidate_csv]
    if not candidate_specs:
        raise ValueError("at least one --candidate BRANCH=PATH entry is required")
    top_k_values = tuple(args.top_k) if args.top_k is not None else _DEFAULT_TOP_K
    candidates = load_candidate_inputs(candidate_specs)
    if candidates.empty:
        raise ValueError("candidate pool is empty")
    truth = pd.read_csv(args.truth_csv)
    frame_rows, pooled, by_sequence, by_branch, manifest = (
        build_candidate_pool_branch_ablation_tables(
            candidates,
            truth,
            group_column=str(args.group_column),
            top_k_values=top_k_values,
            score_column=str(args.score_column),
            fallback_score_column=str(args.fallback_score_column),
            max_truth_time_delta_s=float(args.max_truth_time_delta_s),
            good_candidate_threshold_m=float(args.good_candidate_threshold_m),
            loss_tolerance_m=float(args.loss_tolerance_m),
            include_full_pool=_parse_bool(args.include_full_pool),
            include_leave_one_out=_parse_bool(args.include_leave_one_out),
            include_only_one=_parse_bool(args.include_only_one),
        )
    )
    paths = write_candidate_pool_branch_ablation_outputs(
        output_dir=args.output_dir,
        frame_rows=frame_rows,
        pooled_summary=pooled,
        by_sequence=by_sequence,
        by_reference_branch=by_branch,
        manifest=manifest,
        provenance={
            "candidate_specs": list(candidate_specs),
            "truth_csv": str(args.truth_csv),
            "group_column": str(args.group_column),
            "top_k_values": list(top_k_values),
        },
    )
    print("mmuad_candidate_pool_branch_ablation=ok")
    print(f"pool_count={len(manifest)}")
    print(f"frame_rows={len(frame_rows)}")
    if not pooled.empty and "oracle_all_mse_delta" in pooled.columns:
        best = pooled.sort_values("oracle_all_mse_delta").iloc[0]
        print(f"best_pool_label={best['pool_label']}")
        print(f"best_oracle_all_mse_delta={best['oracle_all_mse_delta']}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _attach_manifest(rows: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(rows).copy()
    if out.empty or manifest.empty or "pool_label" not in out.columns:
        return out
    keep_columns = [
        "pool_label",
        "ablation_type",
        "group_column",
        "group_value",
        "pool_candidate_rows",
        "pool_frame_count",
    ]
    return out.merge(
        manifest[[column for column in keep_columns if column in manifest.columns]],
        on="pool_label",
        how="left",
        validate="many_to_one",
    )


def _manifest_record(
    *,
    pool_label: str,
    ablation_type: str,
    group_column: str,
    group_value: str,
    rows: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "pool_label": str(pool_label),
        "ablation_type": str(ablation_type),
        "group_column": str(group_column),
        "group_value": str(group_value),
        "pool_candidate_rows": int(len(rows)),
        "pool_frame_count": int(_frame_count(rows)),
        "pool_source_count": int(rows["source"].nunique()) if "source" in rows.columns else 0,
        "pool_branch_count": int(rows["candidate_branch"].nunique())
        if "candidate_branch" in rows.columns
        else 0,
    }


def _frame_count(rows: pd.DataFrame) -> int:
    if rows.empty or not {"sequence_id", "time_s"}.issubset(rows.columns):
        return 0
    return int(len(rows[["sequence_id", "time_s"]].drop_duplicates()))


def _clean_label_series(values: pd.Series) -> pd.Series:
    text = values.where(values.notna(), "unknown").astype(str).str.strip()
    missing = text.eq("") | text.str.lower().isin({"nan", "none", "<na>", "nat"})
    return text.where(~missing, "unknown")


def _slug(value: str) -> str:
    text = str(value).strip().lower()
    out = []
    for char in text:
        if char.isalnum():
            out.append(char)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE_TEXT:
        return True
    if text in _FALSE_TEXT:
        return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def _best_record(rows: pd.DataFrame, column: str, *, ascending: bool) -> dict[str, Any]:
    if rows.empty or column not in rows.columns:
        return {}
    values = pd.to_numeric(rows[column], errors="coerce")
    if values.dropna().empty:
        return {}
    order = values.sort_values(ascending=ascending)
    return _jsonable(rows.loc[int(order.index[0])].to_dict())


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
