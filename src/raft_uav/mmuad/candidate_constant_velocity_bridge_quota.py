"""Reservoir quota for candidates with constant-velocity bridge support."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_constant_velocity_bridge import (
    ConstantVelocityBridgeConfig,
    attach_constant_velocity_bridge_features,
    validate_constant_velocity_bridge_config,
)
from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    _apply_frame_cap,
    build_candidate_reservoir,
    build_oracle_recall_tables,
    build_reservoir_summary,
    load_candidate_inputs,
)
from raft_uav.mmuad.schema import normalize_truth_columns

__all__ = [
    "ConstantVelocityBridgeConfig",
    "attach_constant_velocity_bridge_features",
    "build_constant_velocity_bridge_reservoir",
    "build_constant_velocity_bridge_summary",
    "main",
]


def build_constant_velocity_bridge_reservoir(
    candidates: pd.DataFrame,
    *,
    reservoir_config: ReservoirConfig | None = None,
    bridge_config: ConstantVelocityBridgeConfig | None = None,
) -> pd.DataFrame:
    """Add the strongest bridge-supported rows to a branch/source reservoir."""

    reservoir_config = reservoir_config or ReservoirConfig()
    bridge_config = bridge_config or ConstantVelocityBridgeConfig()
    validate_constant_velocity_bridge_config(bridge_config)
    annotated = attach_constant_velocity_bridge_features(candidates, config=bridge_config)
    if annotated.empty:
        return build_candidate_reservoir(annotated, config=reservoir_config)

    base = build_candidate_reservoir(annotated, config=reservoir_config)
    bridge = _select_bridge_quota(
        annotated,
        bridge_top_n=bridge_config.bridge_top_n,
        reason_prefix=bridge_config.reason_prefix,
        score_column=reservoir_config.score_column,
        fallback_score_column=reservoir_config.fallback_score_column,
    )
    combined = _combine_rows(base, bridge)
    preserve_prefixes = tuple(reservoir_config.preserve_reason_prefixes)
    if bridge_config.protect_bridge_quota:
        preserve_prefixes = (*preserve_prefixes, bridge_config.reason_prefix)
    combined = _apply_frame_cap(
        combined,
        max_candidates_per_frame=reservoir_config.max_candidates_per_frame,
        cap_reason_bonus=reservoir_config.cap_reason_bonus,
        preserve_reason_prefixes=preserve_prefixes,
    )
    combined = combined.sort_values(
        ["sequence_id", "time_s", "candidate_reservoir_rank", "source"],
    ).reset_index(drop=True)
    return combined.drop(columns=["_cv_bridge_input_row"], errors="ignore")


def build_constant_velocity_bridge_summary(
    candidates: pd.DataFrame,
    reservoir: pd.DataFrame,
    *,
    reservoir_config: ReservoirConfig,
    bridge_config: ConstantVelocityBridgeConfig,
) -> dict[str, Any]:
    """Return candidate-budget and bridge-support diagnostics."""

    summary = build_reservoir_summary(candidates, reservoir)
    supported = pd.Series(
        reservoir.get("candidate_cv_bridge_supported", pd.Series(dtype=bool))
    ).fillna(False).astype(bool)
    errors = pd.to_numeric(
        reservoir.get("candidate_cv_bridge_error_m", pd.Series(dtype=float)),
        errors="coerce",
    )
    reasons = reservoir.get("candidate_reservoir_reason", pd.Series(dtype=str))
    selected = pd.Series(reasons).fillna("").astype(str).str.contains(
        bridge_config.reason_prefix,
        regex=False,
    )
    finite_errors = errors.loc[supported & np.isfinite(errors)]
    summary.update(
        {
            "reservoir_config": _json_ready(asdict(reservoir_config)),
            "constant_velocity_bridge_config": _json_ready(asdict(bridge_config)),
            "cv_bridge_supported_rows": int(supported.sum()),
            "cv_bridge_quota_rows": int(selected.sum()),
            "cv_bridge_error_mean_m": (
                float(finite_errors.mean()) if len(finite_errors) else float("nan")
            ),
            "cv_bridge_error_p95_m": (
                float(finite_errors.quantile(0.95))
                if len(finite_errors)
                else float("nan")
            ),
        }
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.candidate_specs:
        raise ValueError("at least one --candidate-csv BRANCH=PATH entry is required")
    reservoir_config = ReservoirConfig(
        global_top_n=args.global_top_n,
        per_source_top_n=args.per_source_top_n,
        per_branch_top_n=args.per_branch_top_n,
        max_candidates_per_frame=args.max_candidates_per_frame,
        score_column=args.score_column,
        fallback_score_column=args.fallback_score_column,
        score_floor_quantile=args.score_floor_quantile,
        cap_reason_bonus=args.cap_reason_bonus,
    )
    bridge_config = ConstantVelocityBridgeConfig(
        bridge_top_n=args.bridge_top_n,
        max_frame_gap_s=args.max_frame_gap_s,
        max_speed_mps=args.max_speed_mps,
        max_interpolation_error_m=args.max_interpolation_error_m,
        interpolation_scale_m=args.interpolation_scale_m,
        max_neighbors_per_side=args.max_neighbors_per_side,
        require_same_source=args.require_same_source,
        require_same_branch=args.require_same_branch,
        protect_bridge_quota=not args.do_not_protect_bridge_quota,
    )
    candidates = load_candidate_inputs(args.candidate_specs)
    reservoir = build_constant_velocity_bridge_reservoir(
        candidates,
        reservoir_config=reservoir_config,
        bridge_config=bridge_config,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    reservoir.to_csv(args.output_csv, index=False)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = build_constant_velocity_bridge_summary(
            candidates,
            reservoir,
            reservoir_config=reservoir_config,
            bridge_config=bridge_config,
        )
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("mmuad_constant_velocity_bridge_reservoir=ok")
    print(f"candidate_rows={len(candidates)}")
    print(f"reservoir_rows={len(reservoir)}")
    print(f"output_csv={args.output_csv}")

    if args.truth_csv is not None:
        truth = normalize_truth_columns(pd.read_csv(args.truth_csv))
        top_k_values = tuple(args.top_k) if args.top_k else (1, 3, 5, 10, 20)
        frame_rows, pooled, by_sequence = build_oracle_recall_tables(
            reservoir,
            truth,
            top_k_values=top_k_values,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
        )
        _write_optional_csv(frame_rows, args.oracle_frame_csv)
        _write_optional_csv(pooled, args.oracle_summary_csv)
        _write_optional_csv(by_sequence, args.oracle_by_sequence_csv)
        if not pooled.empty:
            print(f"oracle_all_mse={pooled.loc[0, 'oracle_all_3d_m_mse']}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmuad-constant-velocity-bridge-reservoir",
        description="preserve candidates supported by bracketing constant-velocity motion",
    )
    parser.add_argument(
        "--candidate-csv",
        "--candidate",
        action="append",
        dest="candidate_specs",
        default=[],
        help="candidate CSV as BRANCH=path; may be repeated",
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
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--fallback-score-column", default="confidence")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--cap-reason-bonus", type=float, default=0.0)
    parser.add_argument("--bridge-top-n", type=int, default=2)
    parser.add_argument("--max-frame-gap-s", type=float, default=1.0)
    parser.add_argument("--max-speed-mps", type=float, default=60.0)
    parser.add_argument("--max-interpolation-error-m", type=float, default=5.0)
    parser.add_argument("--interpolation-scale-m", type=float, default=5.0)
    parser.add_argument("--max-neighbors-per-side", type=int, default=40)
    parser.add_argument("--require-same-source", action="store_true")
    parser.add_argument("--require-same-branch", action="store_true")
    parser.add_argument("--do-not-protect-bridge-quota", action="store_true")
    parser.add_argument("--top-k", type=int, action="append", default=None)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    return parser


def _select_bridge_quota(
    rows: pd.DataFrame,
    *,
    bridge_top_n: int,
    reason_prefix: str,
    score_column: str,
    fallback_score_column: str,
) -> pd.DataFrame:
    if bridge_top_n <= 0 or rows.empty:
        return rows.iloc[0:0].copy()
    work = rows.copy()
    work["_cv_bridge_base_score"] = _rowwise_score(
        work,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
    )
    work["_cv_bridge_stable_key"] = work.apply(_stable_key, axis=1)
    selected: list[pd.DataFrame] = []
    for _, frame in work.groupby(["sequence_id", "time_s"], sort=False, dropna=False):
        supported = frame.loc[
            frame["candidate_cv_bridge_supported"].fillna(False).astype(bool)
        ].copy()
        if supported.empty:
            continue
        supported = supported.sort_values(
            [
                "candidate_cv_bridge_score",
                "candidate_cv_bridge_error_m",
                "_cv_bridge_base_score",
                "_cv_bridge_stable_key",
            ],
            ascending=[False, True, False, True],
        ).head(int(bridge_top_n))
        supported["candidate_reservoir_score"] = supported["_cv_bridge_base_score"]
        supported["candidate_reservoir_reason"] = reason_prefix
        supported["candidate_reservoir_reasons"] = reason_prefix
        selected.append(supported)
    if not selected:
        return work.iloc[0:0]
    return pd.concat(selected, ignore_index=True).drop(
        columns=["_cv_bridge_base_score", "_cv_bridge_stable_key"],
        errors="ignore",
    )


def _combine_rows(base: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    if bridge.empty:
        return base.copy()
    combined = pd.concat([base, bridge], ignore_index=True, sort=False)
    if "_cv_bridge_input_row" not in combined.columns:
        raise ValueError("bridge reservoir rows require _cv_bridge_input_row provenance")
    records: list[pd.Series] = []
    for _, group in combined.groupby("_cv_bridge_input_row", sort=False, dropna=False):
        row = group.iloc[0].copy()
        reasons: set[str] = set()
        for value in group.get("candidate_reservoir_reason", pd.Series(dtype=str)):
            for token in str(value).replace(",", ";").split(";"):
                token = token.strip()
                if token and token.lower() != "nan":
                    reasons.add(token)
        row["candidate_reservoir_reason"] = ";".join(sorted(reasons))
        row["candidate_reservoir_reasons"] = row["candidate_reservoir_reason"]
        records.append(row)
    return pd.DataFrame(records).drop(
        columns=[
            "candidate_reservoir_reason_count",
            "candidate_reservoir_cap_score",
            "candidate_reservoir_protected",
            "candidate_reservoir_rank",
        ],
        errors="ignore",
    )


def _rowwise_score(
    rows: pd.DataFrame,
    *,
    score_column: str,
    fallback_score_column: str,
) -> pd.Series:
    primary = _numeric(rows, score_column, default=np.nan)
    fallback = _numeric(rows, fallback_score_column, default=0.0)
    primary = primary.where(np.isfinite(primary), np.nan)
    fallback = fallback.where(np.isfinite(fallback), np.nan)
    return primary.fillna(fallback).fillna(0.0).astype(float)


def _numeric(rows: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(default, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _stable_key(row: pd.Series) -> str:
    values = (
        row.get("source", ""),
        row.get("candidate_branch", ""),
        row.get("track_id", ""),
        row.get("x_m", ""),
        row.get("y_m", ""),
        row.get("z_m", ""),
    )
    return "|".join(str(value) for value in values)


def _write_optional_csv(rows: pd.DataFrame, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
