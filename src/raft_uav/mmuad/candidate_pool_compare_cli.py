"""Console entry point for MMUAD candidate pool comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_pool_compare import (
    _DEFAULT_FALLBACK_SCORE_COLUMN,
    _DEFAULT_SCORE_COLUMN,
    _DEFAULT_TOP_K,
    _load_labeled_candidate_pools,
    build_candidate_pool_compare_tables,
    write_candidate_pool_compare_outputs,
)
from raft_uav.mmuad.candidate_reservoir import load_candidate_inputs


def main(argv: list[str] | None = None) -> int:
    """Run candidate-pool comparison with explicit top-k replacement semantics."""

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


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
