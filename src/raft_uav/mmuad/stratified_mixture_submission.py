"""Installable stratified MMUAD mixture submission helper.

The repository script in ``scripts/mmuad_stratified_mixture_submission.py`` is
convenient from a checkout, but it is not available from a normal package
install.  This module exposes the same official Track 5 CSV/ZIP packaging path
inside ``raft_uav.mmuad`` so installed environments can run it with
``python -m raft_uav.mmuad.stratified_mixture_submission``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import (
    INITIALIZATION_CHOICES,
    LOSS_CHOICES,
    SCORE_NORMALIZATION_CHOICES,
    CandidateMixtureMapConfig,
)
from raft_uav.mmuad.candidate_mixture_map_stratified import (
    StratifiedMixtureTopKConfig,
    run_stratified_candidate_mixture_map,
    write_stratified_candidate_mixture_outputs,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.submission import (
    load_sequence_class_map,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)


def main(argv: list[str] | None = None) -> int:
    """Run stratified mixture smoothing and write official Track 5 artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="2")
    parser.add_argument("--official-results-csv", type=Path)
    parser.add_argument("--official-zip", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-per-branch", type=int, default=1)
    parser.add_argument("--min-per-source", type=int, default=1)
    parser.add_argument("--min-per-source-branch", type=int, default=0)
    parser.add_argument("--branch-column", default="candidate_branch")
    parser.add_argument("--score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--fallback-score-column", action="append", default=[])
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument(
        "--score-normalization",
        choices=SCORE_NORMALIZATION_CHOICES,
        default="minmax",
    )
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=3.0)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tolerance-m", type=float, default=1.0e-3)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_CHOICES,
        default="uncertainty-top1",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    official_results_csv = args.official_results_csv or output_dir / "mmaud_results.csv"
    official_zip = args.official_zip or output_dir / "ug2_submission.zip"
    fallback_columns = tuple(args.fallback_score_column) or ("ranker_score", "confidence")
    truth = None if args.truth_csv is None else load_evaluation_truth_file(args.truth_csv).rows
    initial = None if args.initial_estimates_csv is None else pd.read_csv(args.initial_estimates_csv)
    candidates = load_candidate_file(args.candidates_csv).rows
    mixture_config = CandidateMixtureMapConfig(
        top_k=int(args.top_k),
        score_column=str(args.score_column),
        fallback_score_columns=fallback_columns,
        sigma_column=str(args.sigma_column),
        default_sigma_m=float(args.default_sigma_m),
        sigma_min_m=float(args.sigma_min_m),
        sigma_max_m=float(args.sigma_max_m),
        score_normalization=str(args.score_normalization),
        score_weight=float(args.score_weight),
        temperature=float(args.temperature),
        sigma_log_weight=float(args.sigma_log_weight),
        loss=str(args.loss),
        huber_delta=float(args.huber_delta),
        smoothness_weight=float(args.smoothness_weight),
        iterations=int(args.iterations),
        tolerance_m=float(args.tolerance_m),
        uniform_weight_floor=float(args.uniform_weight_floor),
        initialization=str(args.initialization),
    )
    stratified_config = StratifiedMixtureTopKConfig(
        top_k=int(args.top_k),
        min_per_branch=int(args.min_per_branch),
        min_per_source=int(args.min_per_source),
        min_per_source_branch=int(args.min_per_source_branch),
        score_column=str(args.score_column),
        fallback_score_columns=fallback_columns,
        branch_column=str(args.branch_column),
        sigma_column=str(args.sigma_column),
    )
    result = run_stratified_candidate_mixture_map(
        candidates,
        stratified_config=stratified_config,
        mixture_config=mixture_config,
        initial_estimates=initial,
        truth=truth,
    )
    paths = write_stratified_candidate_mixture_outputs(result, output_dir)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    write_official_mmaud_results_csv(
        result.mixture_result.estimates,
        official_results_csv,
        classification=str(args.default_classification),
        class_map=class_map,
    )
    write_official_ug2_codabench_zip(
        result.mixture_result.estimates,
        official_zip,
        classification=str(args.default_classification),
        class_map=class_map,
    )
    paths["official_results_csv"] = official_results_csv
    paths["official_zip"] = official_zip

    print("mmuad_stratified_mixture_submission=ok")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    print(f"estimate_rows={len(result.mixture_result.estimates)}")
    pooled = result.mixture_result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
