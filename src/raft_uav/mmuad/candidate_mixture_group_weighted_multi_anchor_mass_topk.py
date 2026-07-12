"""Reliability-weighted multi-anchor group selection for MMUAD mixture-MAP.

The multi-anchor selector preserves physical hypotheses that agree with any
inference-time trajectory anchor.  That is useful for recall, but a weak anchor
can preserve an implausible mode as aggressively as a strong anchor.  This
module adds non-negative per-anchor reliability weights while keeping ground
truth out of inference.

For ``mean`` and ``softmin`` aggregation, anchor costs are combined with the
normalized reliability weights available for each candidate.  For ``minimum``,
zero-weight anchors are excluded and positive-weight anchors remain eligible;
this preserves the hard any-anchor interpretation.  The final grouped
mixture-MAP objective is unchanged and still uses the original ranker score,
learned uncertainty, robust loss, and smoothness.
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
    _build_parser as _build_multi_anchor_parser,
    _jsonable,
    _load_anchor_specs,
    _mixture_config_from_args,
    _selection_config_from_args,
    _unique_anchor_slugs,
    add_multi_anchor_conditioned_selection_utility,
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

WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN = (
    "mixture_weighted_multi_anchor_conditioned_selection_utility"
)
WEIGHTED_MULTI_ANCHOR_COST_COLUMN = "mixture_weighted_multi_anchor_aggregate_cost"


@dataclass(frozen=True)
class AnchorReliabilityConfig:
    """Reliability weights used to aggregate alternative trajectory anchors."""

    default_weight: float = 1.0


def add_weighted_multi_anchor_conditioned_selection_utility(
    candidates: pd.DataFrame,
    anchor_estimates: Mapping[str, pd.DataFrame],
    *,
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    aggregation_config: MultiAnchorAggregationConfig | None = None,
    reliability_config: AnchorReliabilityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach a multi-anchor utility with train-selectable anchor reliability."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    anchor_config = anchor_config or AnchorConditioningConfig()
    aggregation_config = aggregation_config or MultiAnchorAggregationConfig()
    reliability_config = reliability_config or AnchorReliabilityConfig()

    labels = [str(label).strip() for label in anchor_estimates]
    weights = _resolve_anchor_weights(
        labels,
        anchor_reliability=anchor_reliability,
        default_weight=reliability_config.default_weight,
    )
    scored, normalized_anchors, base_summary = add_multi_anchor_conditioned_selection_utility(
        candidates,
        anchor_estimates,
        mixture_config=mixture_config,
        anchor_config=anchor_config,
        aggregation_config=aggregation_config,
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
    aggregate_cost, matched_weight, effective_anchor_count = _aggregate_weighted_anchor_costs(
        cost_matrix,
        anchor_weights=weight_vector,
        aggregation_config=aggregation_config,
    )
    weighted_best_indices = _weighted_best_anchor_indices(cost_matrix, weight_vector)
    weighted_best_anchor = np.asarray(
        [labels[index] if index >= 0 else "" for index in weighted_best_indices],
        dtype=object,
    )

    scored[WEIGHTED_MULTI_ANCHOR_COST_COLUMN] = aggregate_cost
    scored["mixture_weighted_multi_anchor_matched_weight"] = matched_weight
    scored["mixture_weighted_multi_anchor_matched_weight_fraction"] = (
        matched_weight / float(weight_vector.sum())
    )
    scored["mixture_weighted_multi_anchor_effective_anchor_count"] = effective_anchor_count
    scored["mixture_weighted_multi_anchor_best_anchor"] = weighted_best_anchor
    scored[WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN] = (
        pd.to_numeric(scored["mixture_multi_anchor_base_utility"], errors="coerce")
        - float(anchor_config.anchor_selection_weight) * aggregate_cost
    )
    # Preserve compatibility for downstream helpers that inspect the established
    # multi-anchor utility name while selecting explicitly on the weighted column.
    scored[MULTI_ANCHOR_UTILITY_COLUMN] = scored[WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN]

    normalized_anchors = normalized_anchors.copy()
    normalized_anchors["anchor_reliability_weight"] = normalized_anchors["anchor_name"].map(
        weights
    )

    summary = dict(base_summary)
    summary["schema"] = "raft-uav-mmuad-weighted-multi-anchor-conditioning-v1"
    summary["anchor_reliability"] = {label: float(weights[label]) for label in labels}
    summary["reliability_config"] = asdict(reliability_config)
    summary["weighted_aggregation"] = aggregation_config.aggregation
    summary["mean_matched_anchor_weight"] = _safe_mean(matched_weight)
    summary["mean_effective_anchor_count"] = _safe_mean(effective_anchor_count)
    summary["weighted_best_anchor_counts"] = {
        str(key): int(value)
        for key, value in pd.Series(weighted_best_anchor).value_counts().items()
        if str(key)
    }
    summary["truth_used_for_weighting"] = False
    return scored, normalized_anchors, _jsonable(summary)


def select_weighted_multi_anchor_posterior_mass_hypothesis_group_topk(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    aggregation_config: MultiAnchorAggregationConfig | None = None,
    reliability_config: AnchorReliabilityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select physical groups using reliability-weighted multi-anchor costs."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    scored, normalized_anchors, weighted_summary = (
        add_weighted_multi_anchor_conditioned_selection_utility(
            candidates,
            anchor_estimates,
            anchor_reliability=anchor_reliability,
            mixture_config=mixture_config,
            anchor_config=anchor_config,
            aggregation_config=aggregation_config,
            reliability_config=reliability_config,
        )
    )
    selection_mixture_config = replace(
        mixture_config,
        score_column=WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN,
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
    summary["schema"] = "raft-uav-mmuad-weighted-multi-anchor-group-topk-v1"
    summary["weighted_multi_anchor_conditioning"] = weighted_summary
    summary["selection_mixture_config"] = asdict(selection_mixture_config)
    summary["truth_used_for_selection"] = False
    return scored, selected, normalized_anchors, _jsonable(summary)


def run_weighted_multi_anchor_posterior_mass_group_topk_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    aggregation_config: MultiAnchorAggregationConfig | None = None,
    reliability_config: AnchorReliabilityConfig | None = None,
    final_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult:
    """Run weighted group selection followed by the unchanged grouped MAP smoother."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    group_config = group_config or HypothesisGroupConfig()
    scored, selected, normalized_anchors, summary = (
        select_weighted_multi_anchor_posterior_mass_hypothesis_group_topk(
            candidates,
            anchor_estimates=anchor_estimates,
            anchor_reliability=anchor_reliability,
            mixture_config=mixture_config,
            group_config=group_config,
            selection_config=selection_config,
            anchor_config=anchor_config,
            aggregation_config=aggregation_config,
            reliability_config=reliability_config,
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


def write_weighted_multi_anchor_outputs(
    result: MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write weighted selection diagnostics and grouped mixture outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored_path = output / "mmuad_weighted_multi_anchor_scored_candidates.csv"
    selected_path = output / "mmuad_weighted_multi_anchor_selected_candidates.csv"
    anchors_path = output / "mmuad_weighted_multi_anchor_normalized_anchors.csv"
    summary_path = output / "mmuad_weighted_multi_anchor_summary.json"
    result.scored_candidates.to_csv(scored_path, index=False)
    result.selected_candidates.to_csv(selected_path, index=False)
    result.normalized_anchors.to_csv(anchors_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2),
        encoding="utf-8",
    )
    paths = write_grouped_candidate_mixture_outputs(result.grouped_result, output)
    paths["weighted_multi_anchor_scored_candidates_csv"] = scored_path
    paths["weighted_multi_anchor_selected_candidates_csv"] = selected_path
    paths["weighted_multi_anchor_normalized_anchors_csv"] = anchors_path
    paths["weighted_multi_anchor_summary_json"] = summary_path
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
    result = run_weighted_multi_anchor_posterior_mass_group_topk_candidate_mixture_map(
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
        aggregation_config=MultiAnchorAggregationConfig(
            aggregation=args.aggregation,
            softmin_temperature=args.softmin_temperature,
        ),
        reliability_config=AnchorReliabilityConfig(
            default_weight=args.default_anchor_reliability,
        ),
        final_initial_estimates=final_initial,
        truth=truth,
    )
    paths = write_weighted_multi_anchor_outputs(result, args.output_dir)
    print("mmuad_weighted_multi_anchor_posterior_mass_group_topk=ok")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"anchor_count={len(anchors)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = _build_multi_anchor_parser()
    parser.prog = (
        "python -m "
        "raft_uav.mmuad.candidate_mixture_group_weighted_multi_anchor_mass_topk"
    )
    parser.description = (
        "condition adaptive MMUAD group selection on reliability-weighted anchors"
    )
    parser.add_argument(
        "--anchor-reliability",
        action="append",
        default=[],
        metavar="NAME=WEIGHT",
        help=(
            "non-negative inference-time reliability for an anchor; may be repeated; "
            "unspecified anchors use --default-anchor-reliability"
        ),
    )
    parser.add_argument("--default-anchor-reliability", type=float, default=1.0)
    return parser


def _resolve_anchor_weights(
    labels: list[str],
    *,
    anchor_reliability: Mapping[str, float] | None,
    default_weight: float,
) -> dict[str, float]:
    if not np.isfinite(default_weight) or float(default_weight) < 0.0:
        raise ValueError("default anchor reliability must be finite and non-negative")
    supplied = dict(anchor_reliability or {})
    unknown = sorted(set(supplied) - set(labels))
    if unknown:
        raise ValueError(f"anchor reliability supplied for unknown anchors: {unknown}")
    result: dict[str, float] = {}
    for label in labels:
        weight = float(supplied.get(label, default_weight))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"anchor reliability for {label!r} must be finite and non-negative"
            )
        result[label] = weight
    if not any(weight > 0.0 for weight in result.values()):
        raise ValueError("at least one anchor reliability must be positive")
    return result


def _aggregate_weighted_anchor_costs(
    cost_matrix: np.ndarray,
    *,
    anchor_weights: np.ndarray,
    aggregation_config: MultiAnchorAggregationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cost_matrix.ndim != 2:
        raise ValueError("anchor cost matrix must be two-dimensional")
    weights = np.asarray(anchor_weights, dtype=float)
    if weights.shape != (cost_matrix.shape[1],):
        raise ValueError("anchor weight count must match anchor cost columns")
    if not np.isfinite(weights).all() or (weights < 0.0).any() or not (weights > 0.0).any():
        raise ValueError("anchor weights must be finite, non-negative, and not all zero")
    if aggregation_config.aggregation not in {"minimum", "mean", "softmin"}:
        raise ValueError(f"unsupported anchor aggregation {aggregation_config.aggregation!r}")
    temperature = float(aggregation_config.softmin_temperature)
    if aggregation_config.aggregation == "softmin" and (
        not np.isfinite(temperature) or temperature <= 0.0
    ):
        raise ValueError("softmin temperature must be finite and positive")

    result = np.zeros(cost_matrix.shape[0], dtype=float)
    matched_weight = np.zeros(cost_matrix.shape[0], dtype=float)
    effective_count = np.zeros(cost_matrix.shape[0], dtype=float)
    for row_index, row in enumerate(cost_matrix):
        valid = np.isfinite(row) & (weights > 0.0)
        if not valid.any():
            continue
        values = row[valid]
        row_weights = weights[valid]
        row_weights = row_weights / float(row_weights.sum())
        matched_weight[row_index] = float(weights[valid].sum())
        effective_count[row_index] = float(1.0 / np.sum(np.square(row_weights)))
        if aggregation_config.aggregation == "minimum":
            result[row_index] = float(np.min(values))
        elif aggregation_config.aggregation == "mean":
            result[row_index] = float(np.average(values, weights=row_weights))
        else:
            minimum = float(np.min(values))
            weighted_exp = float(
                np.sum(row_weights * np.exp(-(values - minimum) / temperature))
            )
            result[row_index] = minimum - temperature * float(np.log(weighted_exp))
    return result, matched_weight, effective_count


def _weighted_best_anchor_indices(
    cost_matrix: np.ndarray,
    anchor_weights: np.ndarray,
) -> np.ndarray:
    indices = np.full(cost_matrix.shape[0], -1, dtype=int)
    for row_index, values in enumerate(cost_matrix):
        valid = np.isfinite(values) & (anchor_weights > 0.0)
        if valid.any():
            valid_indices = np.flatnonzero(valid)
            indices[row_index] = int(valid_indices[np.argmin(values[valid])])
    return indices


def _parse_anchor_reliability_specs(
    specs: list[str],
    anchor_labels: set[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"invalid anchor reliability {spec!r}; expected NAME=WEIGHT"
            )
        label, value_text = spec.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"invalid empty anchor label in reliability spec {spec!r}")
        if label in result:
            raise ValueError(f"duplicate anchor reliability for {label!r}")
        if label not in anchor_labels:
            raise ValueError(f"anchor reliability supplied for unknown anchor {label!r}")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"invalid anchor reliability {spec!r}") from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"anchor reliability for {label!r} must be finite and non-negative"
            )
        result[label] = value
    return result


def _safe_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
