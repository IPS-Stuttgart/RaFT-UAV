"""Origin-group-corrected mixture MAP over a branch-preserving MMUAD reservoir.

Candidate reservoirs preserve raw, calibrated, dynamic, and other branch-specific
hypotheses long enough for trajectory inference to use them.  Some of those rows
are alternative coordinate representations of the same physical observation.
A flat mixture gives a physical observation more prior mass merely because it has
more representations.  This module composes the maintained reservoir and grouped
mixture-MAP implementations so both protections are applied in one command.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import (
    INITIALIZATION_CHOICES,
    LOSS_CHOICES,
    SCORE_NORMALIZATION_CHOICES,
    CandidateMixtureMapConfig,
)
from raft_uav.mmuad.candidate_mixture_map_grouped import (
    MISSING_GROUP_POLICIES,
    GroupedCandidateMixtureMapResult,
    HypothesisGroupConfig,
    run_grouped_candidate_mixture_map,
    write_grouped_candidate_mixture_outputs,
)
from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    build_candidate_reservoir,
    build_reservoir_summary,
    load_candidate_inputs,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import (
    load_sequence_class_map,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)

GROUPED_RESERVOIR_CSV = "mmuad_grouped_reservoir_candidates.csv"
GROUPED_RESERVOIR_SUMMARY_JSON = "mmuad_grouped_reservoir_mixture_summary.json"


@dataclass(frozen=True)
class GroupedReservoirMixtureMapResult:
    """Reservoir rows, grouped mixture result, and combined provenance summary."""

    reservoir: pd.DataFrame
    grouped_result: GroupedCandidateMixtureMapResult
    summary: dict[str, Any]


def run_grouped_reservoir_mixture_map(
    candidates: pd.DataFrame,
    *,
    reservoir_config: ReservoirConfig | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> GroupedReservoirMixtureMapResult:
    """Preserve branch/source hypotheses, then remove representation-count bias.

    The reservoir is built before group correction because the correction must
    reflect the sibling representations that actually reach inference.  The
    mixture ``top_k`` is forced to zero: reservoir construction has already
    bounded each frame, and a second global score truncation would undo its
    branch/source guarantees.
    """

    reservoir_config = reservoir_config or ReservoirConfig()
    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()

    reservoir = build_candidate_reservoir(candidates, config=reservoir_config)
    effective_mixture_config = replace(mixture_config, top_k=0)
    grouped_result = run_grouped_candidate_mixture_map(
        reservoir,
        mixture_config=effective_mixture_config,
        group_config=group_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    summary = {
        "schema": "raft-uav-mmuad-grouped-reservoir-mixture-map-v1",
        "reservoir_config": asdict(reservoir_config),
        "mixture_config": asdict(effective_mixture_config),
        "hypothesis_group_config": asdict(group_config),
        "reservoir": build_reservoir_summary(candidates, reservoir),
        "hypothesis_grouping": grouped_result.grouping_summary,
        "mixture": grouped_result.mixture_result.summary,
        "truth_used_for_candidate_selection": False,
        "truth_used_for_hypothesis_grouping": False,
    }
    return GroupedReservoirMixtureMapResult(
        reservoir=reservoir,
        grouped_result=grouped_result,
        summary=_jsonable(summary),
    )


def write_grouped_reservoir_mixture_outputs(
    result: GroupedReservoirMixtureMapResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write reservoir, group-correction, mixture, and provenance artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = write_grouped_candidate_mixture_outputs(result.grouped_result, output)

    reservoir_path = output / GROUPED_RESERVOIR_CSV
    result.reservoir.to_csv(reservoir_path, index=False)
    paths["grouped_reservoir_csv"] = reservoir_path

    summary_path = output / GROUPED_RESERVOIR_SUMMARY_JSON
    summary_path.write_text(
        json.dumps(_jsonable(result.summary), indent=2),
        encoding="utf-8",
    )
    paths["grouped_reservoir_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-grouped-reservoir-mixture-map",
        description=(
            "run origin-group-corrected candidate-mixture MAP over a "
            "branch-preserving MMUAD reservoir"
        ),
    )
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        metavar="BRANCH=PATH",
        help="candidate CSV to include; may be repeated; plain PATH uses its stem",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--initial-estimates-csv", type=Path)

    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--reservoir-score-column", default="ranker_score")
    parser.add_argument("--reservoir-fallback-score-column", default="confidence")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--reservoir-cap-reason-bonus", type=float, default=0.0)

    parser.add_argument("--mixture-score-column", default="candidate_reservoir_score")
    parser.add_argument("--mixture-fallback-score-column", action="append", default=[])
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
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tolerance-m", type=float, default=1.0e-3)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument("--branch-balance", type=float, default=0.0)
    parser.add_argument("--source-balance", type=float, default=0.0)
    parser.add_argument("--responsibility-floor", type=float, default=0.0)
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_CHOICES,
        default="uncertainty-top1",
    )

    parser.add_argument("--hypothesis-group-column")
    parser.add_argument("--hypothesis-group-correction-strength", type=float, default=1.0)
    parser.add_argument(
        "--missing-hypothesis-group-policy",
        choices=MISSING_GROUP_POLICIES,
        default="unique",
    )

    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="2")
    parser.add_argument("--official-results-csv", type=Path)
    parser.add_argument("--official-zip", type=Path)
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv BRANCH=PATH")

    candidates = load_candidate_inputs(args.candidate_csv)
    truth = (
        None
        if args.truth_csv is None
        else load_evaluation_truth_file(args.truth_csv).rows
    )
    initial_estimates = (
        None
        if args.initial_estimates_csv is None
        else pd.read_csv(args.initial_estimates_csv)
    )
    reservoir_config = ReservoirConfig(
        global_top_n=int(args.global_top_n),
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        max_candidates_per_frame=int(args.max_candidates_per_frame),
        score_column=str(args.reservoir_score_column),
        fallback_score_column=str(args.reservoir_fallback_score_column),
        score_floor_quantile=args.score_floor_quantile,
        cap_reason_bonus=float(args.reservoir_cap_reason_bonus),
    )
    mixture_config = CandidateMixtureMapConfig(
        top_k=0,
        score_column=str(args.mixture_score_column),
        fallback_score_columns=tuple(args.mixture_fallback_score_column)
        or ("ranker_score", "confidence"),
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
        anchor_weight=float(args.anchor_weight),
        iterations=int(args.iterations),
        tolerance_m=float(args.tolerance_m),
        uniform_weight_floor=float(args.uniform_weight_floor),
        branch_balance=float(args.branch_balance),
        source_balance=float(args.source_balance),
        responsibility_floor=float(args.responsibility_floor),
        initialization=str(args.initialization),
    )
    group_config = HypothesisGroupConfig(
        group_column=args.hypothesis_group_column,
        correction_strength=float(args.hypothesis_group_correction_strength),
        missing_group_policy=str(args.missing_hypothesis_group_policy),
    )
    result = run_grouped_reservoir_mixture_map(
        candidates,
        reservoir_config=reservoir_config,
        mixture_config=mixture_config,
        group_config=group_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    paths = write_grouped_reservoir_mixture_outputs(result, args.output_dir)

    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    estimates = result.grouped_result.mixture_result.estimates
    if args.official_results_csv is not None:
        write_official_mmaud_results_csv(
            estimates,
            args.official_results_csv,
            classification=args.default_classification,
            class_map=class_map,
        )
        paths["official_results_csv"] = args.official_results_csv
    if args.official_zip is not None:
        write_official_ug2_codabench_zip(
            estimates,
            args.official_zip,
            classification=args.default_classification,
            class_map=class_map,
        )
        paths["official_zip"] = args.official_zip

    print("mmuad_grouped_reservoir_mixture_map=ok")
    print(f"reservoir_rows={len(result.reservoir)}")
    print(
        "duplicate_hypothesis_group_count="
        f"{result.grouped_result.grouping_summary['duplicate_hypothesis_group_count']}"
    )
    pooled = result.grouped_result.mixture_result.summary.get("metrics", {}).get(
        "pooled", {}
    )
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


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
