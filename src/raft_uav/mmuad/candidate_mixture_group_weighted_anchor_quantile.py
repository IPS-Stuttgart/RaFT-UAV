"""Reliability-quantile multi-anchor selection for MMUAD mixture-MAP.

Minimum-cost aggregation preserves a mode supported by any anchor, whereas a
weighted mean can over-penalize a mode that is contradicted by one poor anchor.
This module adds a reliability-weighted cost quantile between those extremes.
A quantile of zero reproduces the permissive minimum; a quantile of 0.5 is a
weighted median; a quantile of one requires agreement with every positive-
weight anchor.

The quantile affects only the finite physical-hypothesis group selection unary.
The final grouped learned-sigma / Huber mixture-MAP objective remains unchanged.
Ground truth is never used for selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
)
from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
    select_posterior_mass_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_group_multi_anchor_mass_topk import (
    MULTI_ANCHOR_UTILITY_COLUMN,
    MultiAnchorAggregationConfig,
    MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult,
    _jsonable,
    _load_anchor_specs,
    _mixture_config_from_args,
    _selection_config_from_args,
    _unique_anchor_slugs,
    add_multi_anchor_conditioned_selection_utility,
)
from raft_uav.mmuad.candidate_mixture_group_weighted_multi_anchor_mass_topk import (
    WEIGHTED_MULTI_ANCHOR_COST_COLUMN,
    WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN,
    _build_parser as _build_weighted_parser,
    _parse_anchor_reliability_specs,
    _resolve_anchor_weights,
    _safe_mean,
    _weighted_best_anchor_indices,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import (
    HypothesisGroupConfig,
    run_grouped_candidate_mixture_map,
    write_grouped_candidate_mixture_outputs,
)
from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file

WEIGHTED_QUANTILE_UTILITY_COLUMN = (
    "mixture_weighted_quantile_multi_anchor_conditioned_selection_utility"
)
WEIGHTED_QUANTILE_COST_COLUMN = "mixture_weighted_quantile_multi_anchor_aggregate_cost"


@dataclass(frozen=True)
class WeightedAnchorQuantileConfig:
    """Reliability weights and quantile used for anchor-cost aggregation."""

    cost_quantile: float = 0.5
    default_weight: float = 1.0


def add_weighted_quantile_multi_anchor_conditioned_selection_utility(
    candidates: pd.DataFrame,
    anchor_estimates: Mapping[str, pd.DataFrame],
    *,
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    quantile_config: WeightedAnchorQuantileConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach an inference-safe reliability-quantile anchor utility."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    anchor_config = anchor_config or AnchorConditioningConfig()
    quantile_config = quantile_config or WeightedAnchorQuantileConfig()
    _validate_quantile_config(quantile_config)

    labels = [str(label).strip() for label in anchor_estimates]
    weights = _resolve_anchor_weights(
        labels,
        anchor_reliability=anchor_reliability,
        default_weight=quantile_config.default_weight,
    )

    # The established multi-anchor scorer materializes all per-anchor distances
    # and bounded Huber costs. Its aggregate is ignored and replaced below.
    scored, normalized_anchors, base_summary = add_multi_anchor_conditioned_selection_utility(
        candidates,
        anchor_estimates,
        mixture_config=mixture_config,
        anchor_config=anchor_config,
        aggregation_config=MultiAnchorAggregationConfig(aggregation="minimum"),
    )

    slugs = _unique_anchor_slugs(labels)
    cost_matrix = np.column_stack(
        [
            pd.to_numeric(
                scored[f"mixture_multi_anchor_{slug}_cost"],
                errors="coerce",
            ).to_numpy(float)
            for slug in slugs
        ]
    )
    weight_vector = np.asarray([weights[label] for label in labels], dtype=float)
    aggregate_cost, matched_weight, effective_anchor_count = (
        aggregate_weighted_quantile_anchor_costs(
            cost_matrix,
            anchor_weights=weight_vector,
            quantile=quantile_config.cost_quantile,
        )
    )
    best_indices = _weighted_best_anchor_indices(cost_matrix, weight_vector)
    best_anchor = np.asarray(
        [labels[index] if index >= 0 else "" for index in best_indices],
        dtype=object,
    )

    total_weight = float(weight_vector.sum())
    scored[WEIGHTED_QUANTILE_COST_COLUMN] = aggregate_cost
    scored["mixture_weighted_quantile_multi_anchor_matched_weight"] = matched_weight
    scored["mixture_weighted_quantile_multi_anchor_matched_weight_fraction"] = (
        matched_weight / total_weight
    )
    scored["mixture_weighted_quantile_multi_anchor_effective_anchor_count"] = (
        effective_anchor_count
    )
    scored["mixture_weighted_quantile_multi_anchor_best_anchor"] = best_anchor
    scored[WEIGHTED_QUANTILE_UTILITY_COLUMN] = (
        pd.to_numeric(scored["mixture_multi_anchor_base_utility"], errors="coerce")
        - float(anchor_config.anchor_selection_weight) * aggregate_cost
    )

    # Preserve the established weighted and multi-anchor compatibility columns
    # so existing downstream selection and diagnostics can consume this result.
    scored[WEIGHTED_MULTI_ANCHOR_COST_COLUMN] = scored[WEIGHTED_QUANTILE_COST_COLUMN]
    scored[WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN] = scored[WEIGHTED_QUANTILE_UTILITY_COLUMN]
    scored[MULTI_ANCHOR_UTILITY_COLUMN] = scored[WEIGHTED_QUANTILE_UTILITY_COLUMN]

    normalized_anchors = normalized_anchors.copy()
    normalized_anchors["anchor_reliability_weight"] = normalized_anchors["anchor_name"].map(
        weights
    )

    summary = dict(base_summary)
    summary["schema"] = "raft-uav-mmuad-weighted-anchor-quantile-conditioning-v1"
    summary["anchor_reliability"] = {label: float(weights[label]) for label in labels}
    summary["quantile_config"] = asdict(quantile_config)
    summary["weighted_aggregation"] = "quantile"
    summary["mean_matched_anchor_weight"] = _safe_mean(matched_weight)
    summary["mean_effective_anchor_count"] = _safe_mean(effective_anchor_count)
    summary["weighted_best_anchor_counts"] = {
        str(key): int(value)
        for key, value in pd.Series(best_anchor).value_counts().items()
        if str(key)
    }
    summary["truth_used_for_weighting"] = False
    return scored, normalized_anchors, _jsonable(summary)


def select_weighted_quantile_posterior_mass_hypothesis_group_topk(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    quantile_config: WeightedAnchorQuantileConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select physical groups using reliability-quantile anchor costs."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    scored, normalized_anchors, quantile_summary = (
        add_weighted_quantile_multi_anchor_conditioned_selection_utility(
            candidates,
            anchor_estimates,
            anchor_reliability=anchor_reliability,
            mixture_config=mixture_config,
            anchor_config=anchor_config,
            quantile_config=quantile_config,
        )
    )
    selection_mixture_config = replace(
        mixture_config,
        score_column=WEIGHTED_QUANTILE_UTILITY_COLUMN,
        fallback_score_columns=(),
        score_normalization="none",
        score_weight=1.0,
        temperature=1.0,
        sigma_log_weight=0.0,
    )
    selected, summary = select_posterior_mass_hypothesis_group_topk(
        scored,
        mixture_config=selection_mixture_config,
        group_config=group_config,
        selection_config=selection_config,
    )
    summary = dict(summary)
    summary["schema"] = "raft-uav-mmuad-weighted-anchor-quantile-group-topk-v1"
    summary["weighted_anchor_quantile_conditioning"] = quantile_summary
    summary["selection_mixture_config"] = asdict(selection_mixture_config)
    summary["truth_used_for_selection"] = False
    return scored, selected, normalized_anchors, _jsonable(summary)


def run_weighted_quantile_posterior_mass_group_topk_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    quantile_config: WeightedAnchorQuantileConfig | None = None,
    final_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult:
    """Run quantile group selection followed by unchanged grouped mixture-MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    scored, selected, normalized_anchors, summary = (
        select_weighted_quantile_posterior_mass_hypothesis_group_topk(
            candidates,
            anchor_estimates=anchor_estimates,
            anchor_reliability=anchor_reliability,
            mixture_config=mixture_config,
            group_config=group_config,
            selection_config=selection_config,
            anchor_config=anchor_config,
            quantile_config=quantile_config,
        )
    )
    effective_mixture_config = mixture_config
    if int(selection_config.max_group_top_k) > 0:
        effective_mixture_config = replace(mixture_config, top_k=0)
    grouped = run_grouped_candidate_mixture_map(
        selected,
        mixture_config=effective_mixture_config,
        group_config=group_config,
        initial_estimates=final_initial_estimates,
        truth=truth,
    )
    summary["final_mixture_config"] = asdict(effective_mixture_config)
    summary["final_initial_estimates_supplied"] = final_initial_estimates is not None
    return MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult(
        scored_candidates=scored,
        selected_candidates=selected,
        normalized_anchors=normalized_anchors,
        grouped_result=grouped,
        selection_summary=_jsonable(summary),
    )


def aggregate_weighted_quantile_anchor_costs(
    cost_matrix: np.ndarray,
    *,
    anchor_weights: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate each cost row using a discrete reliability-weighted quantile."""

    costs = np.asarray(cost_matrix, dtype=float)
    weights = np.asarray(anchor_weights, dtype=float)
    if costs.ndim != 2:
        raise ValueError("anchor cost matrix must be two-dimensional")
    if weights.shape != (costs.shape[1],):
        raise ValueError("anchor weight count must match anchor cost columns")
    if not np.isfinite(weights).all() or (weights < 0.0).any() or not (weights > 0.0).any():
        raise ValueError("anchor weights must be finite, non-negative, and not all zero")
    quantile = float(quantile)
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("anchor cost quantile must be finite and in [0, 1]")

    aggregate = np.zeros(costs.shape[0], dtype=float)
    matched_weight = np.zeros(costs.shape[0], dtype=float)
    effective_count = np.zeros(costs.shape[0], dtype=float)
    for row_index, row in enumerate(costs):
        valid = np.isfinite(row) & (weights > 0.0)
        if not valid.any():
            continue
        values = row[valid]
        row_weights = weights[valid]
        matched_weight[row_index] = float(row_weights.sum())
        normalized_weights = row_weights / float(row_weights.sum())
        effective_count[row_index] = float(1.0 / np.sum(np.square(normalized_weights)))
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        cumulative_weight = np.cumsum(normalized_weights[order])
        selected_index = int(np.searchsorted(cumulative_weight, quantile, side="left"))
        selected_index = min(selected_index, len(sorted_values) - 1)
        aggregate[row_index] = float(sorted_values[selected_index])
    return aggregate, matched_weight, effective_count


def write_weighted_quantile_outputs(
    result: MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write quantile-selection diagnostics and grouped mixture outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored_path = output / "mmuad_weighted_anchor_quantile_scored_candidates.csv"
    selected_path = output / "mmuad_weighted_anchor_quantile_selected_candidates.csv"
    anchors_path = output / "mmuad_weighted_anchor_quantile_normalized_anchors.csv"
    summary_path = output / "mmuad_weighted_anchor_quantile_summary.json"
    result.scored_candidates.to_csv(scored_path, index=False)
    result.selected_candidates.to_csv(selected_path, index=False)
    result.normalized_anchors.to_csv(anchors_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2),
        encoding="utf-8",
    )
    paths = write_grouped_candidate_mixture_outputs(result.grouped_result, output)
    paths["weighted_anchor_quantile_scored_candidates_csv"] = scored_path
    paths["weighted_anchor_quantile_selected_candidates_csv"] = selected_path
    paths["weighted_anchor_quantile_normalized_anchors_csv"] = anchors_path
    paths["weighted_anchor_quantile_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.anchor_csv:
        parser.error("provide at least one --anchor-csv NAME=PATH")

    candidates = load_candidate_file(args.candidates_csv).rows
    anchors = _load_anchor_specs(args.anchor_csv)
    reliability = _parse_anchor_reliability_specs(args.anchor_reliability, set(anchors))
    final_initial = (
        None
        if args.final_initial_estimates_csv is None
        else read_estimate_csv(args.final_initial_estimates_csv)
    )
    truth = (
        None
        if args.truth_csv is None
        else load_evaluation_truth_file(args.truth_csv).rows
    )
    result = run_weighted_quantile_posterior_mass_group_topk_candidate_mixture_map(
        candidates,
        anchor_estimates=anchors,
        anchor_reliability=reliability,
        mixture_config=_mixture_config_from_args(args),
        group_config=HypothesisGroupConfig(
            group_column=args.hypothesis_group_column,
            correction_strength=args.hypothesis_group_correction_strength,
            missing_group_policy=args.missing_hypothesis_group_policy,
        ),
        selection_config=_selection_config_from_args(args),
        anchor_config=AnchorConditioningConfig(
            anchor_selection_weight=args.anchor_selection_weight,
            anchor_scale_m=args.anchor_scale_m,
            anchor_huber_delta=args.anchor_huber_delta,
            anchor_cost_cap=args.anchor_cost_cap,
            anchor_time_tolerance_s=args.anchor_time_tolerance_s,
            missing_anchor_policy=args.missing_anchor_policy,
        ),
        quantile_config=WeightedAnchorQuantileConfig(
            cost_quantile=args.anchor_cost_quantile,
            default_weight=args.default_anchor_reliability,
        ),
        final_initial_estimates=final_initial,
        truth=truth,
    )
    paths = write_weighted_quantile_outputs(result, args.output_dir)
    print("mmuad_weighted_anchor_quantile_group_topk=ok")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"anchor_count={len(anchors)}")
    print(f"anchor_cost_quantile={args.anchor_cost_quantile}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = _build_weighted_parser()
    parser.prog = (
        "python -m "
        "raft_uav.mmuad.candidate_mixture_group_weighted_anchor_quantile"
    )
    parser.description = (
        "select MMUAD physical groups with reliability-weighted anchor-cost quantiles"
    )
    for action in parser._actions:
        if action.dest == "aggregation":
            action.default = "minimum"
            action.choices = ("minimum",)
            action.help = argparse.SUPPRESS
    parser.add_argument(
        "--anchor-cost-quantile",
        type=float,
        default=0.5,
        help=(
            "reliability-weighted anchor-cost quantile in [0,1]; "
            "0=min, 0.5=weighted median, 1=max"
        ),
    )
    return parser


def _validate_quantile_config(config: WeightedAnchorQuantileConfig) -> None:
    quantile = float(config.cost_quantile)
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError("anchor cost quantile must be finite and in [0, 1]")
    default_weight = float(config.default_weight)
    if not np.isfinite(default_weight) or default_weight < 0.0:
        raise ValueError("default anchor reliability must be finite and non-negative")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
