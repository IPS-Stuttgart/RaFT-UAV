"""Class-conditioned anchor reliability for MMUAD physical-group selection.

The fused UAV classifier is strong enough to provide useful context for pose,
but explicit class-conditioned trajectory smoothness did not improve the
maintained mixture-MAP result. This module uses classification at an earlier
and more targeted point: it blends per-class reliability profiles for
alternative inference-time trajectory anchors before weighted-quantile physical
group selection.

For sequence class probabilities ``p(c)`` and train-selected anchor reliability
``w(anchor, c)``, each candidate row receives

``w_eff(anchor) = (1-lambda) * w_global(anchor)
                  + lambda * sum_c p(c) * w(anchor, c)``.

The resulting row-wise weights affect only finite physical-hypothesis selection.
The final grouped learned-sigma / Huber mixture-MAP objective is unchanged, and
ground truth is never used at inference time.
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
from raft_uav.mmuad.candidate_mixture_group_weighted_anchor_quantile import (
    WEIGHTED_QUANTILE_COST_COLUMN,
    WEIGHTED_QUANTILE_UTILITY_COLUMN,
    _build_parser as _build_weighted_quantile_parser,
)
from raft_uav.mmuad.candidate_mixture_group_weighted_multi_anchor_mass_topk import (
    WEIGHTED_MULTI_ANCHOR_COST_COLUMN,
    WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN,
    _parse_anchor_reliability_specs,
    _resolve_anchor_weights,
    _safe_mean,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import (
    HypothesisGroupConfig,
    run_grouped_candidate_mixture_map,
    write_grouped_candidate_mixture_outputs,
)
from raft_uav.mmuad.class_probability_context import (
    FILL_MISSING_POLICIES,
    OFFICIAL_CLASS_LABELS,
    attach_class_probability_context,
)
from raft_uav.mmuad.class_probability_csv import read_class_probability_csv
from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file

CLASS_CONDITIONED_COST_COLUMN = (
    "mixture_class_conditioned_anchor_quantile_aggregate_cost"
)
CLASS_CONDITIONED_UTILITY_COLUMN = (
    "mixture_class_conditioned_anchor_quantile_selection_utility"
)


@dataclass(frozen=True)
class ClassConditionedAnchorReliabilityConfig:
    """Configuration for soft class-conditioned anchor reliability."""

    cost_quantile: float = 0.5
    default_weight: float = 1.0
    conditioning_strength: float = 1.0
    fill_missing_class_probabilities: str = "uniform"


def add_class_conditioned_anchor_quantile_selection_utility(
    candidates: pd.DataFrame,
    anchor_estimates: Mapping[str, pd.DataFrame],
    class_probabilities: pd.DataFrame,
    *,
    anchor_reliability: Mapping[str, float] | None = None,
    anchor_class_reliability: Mapping[str, Mapping[str, float]] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    reliability_config: ClassConditionedAnchorReliabilityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach an inference-safe class-conditioned weighted-quantile utility."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    anchor_config = anchor_config or AnchorConditioningConfig()
    reliability_config = (
        reliability_config or ClassConditionedAnchorReliabilityConfig()
    )
    _validate_reliability_config(reliability_config)

    labels = [str(label).strip() for label in anchor_estimates]
    base_weights = _resolve_anchor_weights(
        labels,
        anchor_reliability=anchor_reliability,
        default_weight=reliability_config.default_weight,
    )
    class_weights = resolve_anchor_class_reliability(
        labels,
        base_weights=base_weights,
        anchor_class_reliability=anchor_class_reliability,
    )

    scored, normalized_anchors, base_summary = (
        add_multi_anchor_conditioned_selection_utility(
            candidates,
            anchor_estimates,
            mixture_config=mixture_config,
            anchor_config=anchor_config,
            aggregation_config=MultiAnchorAggregationConfig(
                aggregation="minimum"
            ),
        )
    )
    scored, probability_matrix = _attach_normalized_class_probabilities(
        scored,
        class_probabilities,
        fill_missing=reliability_config.fill_missing_class_probabilities,
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
    row_weights, fallback_rows = class_conditioned_anchor_weight_matrix(
        probability_matrix,
        labels=labels,
        base_weights=base_weights,
        class_weights=class_weights,
        conditioning_strength=reliability_config.conditioning_strength,
    )
    aggregate_cost, matched_weight, effective_anchor_count = (
        aggregate_rowwise_weighted_quantile_anchor_costs(
            cost_matrix,
            anchor_weights=row_weights,
            quantile=reliability_config.cost_quantile,
        )
    )
    best_indices = _rowwise_best_anchor_indices(cost_matrix, row_weights)
    best_anchor = np.asarray(
        [labels[index] if index >= 0 else "" for index in best_indices],
        dtype=object,
    )

    weight_totals = row_weights.sum(axis=1)
    scored[CLASS_CONDITIONED_COST_COLUMN] = aggregate_cost
    scored[
        "mixture_class_conditioned_anchor_quantile_matched_weight"
    ] = matched_weight
    scored[
        "mixture_class_conditioned_anchor_quantile_matched_weight_fraction"
    ] = np.divide(
        matched_weight,
        weight_totals,
        out=np.zeros_like(matched_weight),
        where=weight_totals > 0.0,
    )
    scored[
        "mixture_class_conditioned_anchor_quantile_effective_anchor_count"
    ] = effective_anchor_count
    scored[
        "mixture_class_conditioned_anchor_quantile_best_anchor"
    ] = best_anchor
    scored[
        "mixture_class_conditioned_anchor_reliability_fallback"
    ] = fallback_rows.astype(float)
    for anchor_index, (label, slug) in enumerate(zip(labels, slugs, strict=True)):
        scored[
            f"mixture_class_conditioned_anchor_weight_{slug}"
        ] = row_weights[:, anchor_index]

    scored[CLASS_CONDITIONED_UTILITY_COLUMN] = (
        pd.to_numeric(
            scored["mixture_multi_anchor_base_utility"],
            errors="coerce",
        )
        - float(anchor_config.anchor_selection_weight) * aggregate_cost
    )

    # Preserve compatibility with the established weighted and multi-anchor
    # selection columns so downstream diagnostics need no special cases.
    scored[WEIGHTED_QUANTILE_COST_COLUMN] = scored[
        CLASS_CONDITIONED_COST_COLUMN
    ]
    scored[WEIGHTED_QUANTILE_UTILITY_COLUMN] = scored[
        CLASS_CONDITIONED_UTILITY_COLUMN
    ]
    scored[WEIGHTED_MULTI_ANCHOR_COST_COLUMN] = scored[
        CLASS_CONDITIONED_COST_COLUMN
    ]
    scored[WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN] = scored[
        CLASS_CONDITIONED_UTILITY_COLUMN
    ]
    scored[MULTI_ANCHOR_UTILITY_COLUMN] = scored[
        CLASS_CONDITIONED_UTILITY_COLUMN
    ]

    normalized_anchors = normalized_anchors.copy()
    normalized_anchors["anchor_reliability_weight"] = (
        normalized_anchors["anchor_name"].map(base_weights)
    )
    for class_label in OFFICIAL_CLASS_LABELS:
        normalized_anchors[
            f"anchor_class_reliability_{class_label}"
        ] = normalized_anchors["anchor_name"].map(
            {
                label: class_weights[label][class_label]
                for label in labels
            }
        )

    summary = dict(base_summary)
    summary["schema"] = (
        "raft-uav-mmuad-class-conditioned-anchor-quantile-conditioning-v1"
    )
    summary["anchor_reliability"] = {
        label: float(base_weights[label]) for label in labels
    }
    summary["anchor_class_reliability"] = {
        label: {
            class_label: float(class_weights[label][class_label])
            for class_label in OFFICIAL_CLASS_LABELS
        }
        for label in labels
    }
    summary["reliability_config"] = asdict(reliability_config)
    summary["mean_matched_anchor_weight"] = _safe_mean(matched_weight)
    summary["mean_effective_anchor_count"] = _safe_mean(
        effective_anchor_count
    )
    summary["class_conditioning_fallback_rows"] = int(fallback_rows.sum())
    summary["class_conditioning_fallback_rate"] = float(
        fallback_rows.mean() if len(fallback_rows) else 0.0
    )
    summary["mean_effective_anchor_weight"] = {
        label: float(np.mean(row_weights[:, index]))
        for index, label in enumerate(labels)
    }
    summary["best_anchor_counts"] = {
        str(key): int(value)
        for key, value in pd.Series(best_anchor).value_counts().items()
        if str(key)
    }
    summary["class_probability_confidence_mean"] = float(
        np.max(probability_matrix, axis=1).mean()
        if len(probability_matrix)
        else 0.0
    )
    summary["truth_used_for_weighting"] = False
    return scored, normalized_anchors, _jsonable(summary)


def select_class_conditioned_anchor_quantile_group_topk(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    class_probabilities: pd.DataFrame,
    anchor_reliability: Mapping[str, float] | None = None,
    anchor_class_reliability: Mapping[str, Mapping[str, float]] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    reliability_config: ClassConditionedAnchorReliabilityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select physical groups with class-conditioned anchor reliability."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    scored, normalized_anchors, conditioning_summary = (
        add_class_conditioned_anchor_quantile_selection_utility(
            candidates,
            anchor_estimates,
            class_probabilities,
            anchor_reliability=anchor_reliability,
            anchor_class_reliability=anchor_class_reliability,
            mixture_config=mixture_config,
            anchor_config=anchor_config,
            reliability_config=reliability_config,
        )
    )
    selection_mixture_config = replace(
        mixture_config,
        score_column=CLASS_CONDITIONED_UTILITY_COLUMN,
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
    summary["schema"] = (
        "raft-uav-mmuad-class-conditioned-anchor-quantile-group-topk-v1"
    )
    summary["class_conditioned_anchor_quantile"] = conditioning_summary
    summary["selection_mixture_config"] = asdict(
        selection_mixture_config
    )
    summary["truth_used_for_selection"] = False
    return scored, selected, normalized_anchors, _jsonable(summary)


def run_class_conditioned_anchor_quantile_group_topk_mixture_map(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    class_probabilities: pd.DataFrame,
    anchor_reliability: Mapping[str, float] | None = None,
    anchor_class_reliability: Mapping[str, Mapping[str, float]] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    reliability_config: ClassConditionedAnchorReliabilityConfig | None = None,
    final_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult:
    """Run class-conditioned selection and unchanged grouped mixture-MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    scored, selected, normalized_anchors, summary = (
        select_class_conditioned_anchor_quantile_group_topk(
            candidates,
            anchor_estimates=anchor_estimates,
            class_probabilities=class_probabilities,
            anchor_reliability=anchor_reliability,
            anchor_class_reliability=anchor_class_reliability,
            mixture_config=mixture_config,
            group_config=group_config,
            selection_config=selection_config,
            anchor_config=anchor_config,
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
    summary["final_initial_estimates_supplied"] = (
        final_initial_estimates is not None
    )
    return MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult(
        scored_candidates=scored,
        selected_candidates=selected,
        normalized_anchors=normalized_anchors,
        grouped_result=grouped,
        selection_summary=_jsonable(summary),
    )


def aggregate_rowwise_weighted_quantile_anchor_costs(
    cost_matrix: np.ndarray,
    *,
    anchor_weights: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate costs using one reliability vector per candidate row."""

    costs = np.asarray(cost_matrix, dtype=float)
    weights = np.asarray(anchor_weights, dtype=float)
    if costs.ndim != 2:
        raise ValueError("anchor cost matrix must be two-dimensional")
    if weights.shape != costs.shape:
        raise ValueError(
            "row-wise anchor weights must match the anchor cost matrix"
        )
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError(
            "row-wise anchor weights must be finite and non-negative"
        )
    quantile = float(quantile)
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError(
            "anchor cost quantile must be finite and in [0, 1]"
        )

    aggregate = np.zeros(costs.shape[0], dtype=float)
    matched_weight = np.zeros(costs.shape[0], dtype=float)
    effective_count = np.zeros(costs.shape[0], dtype=float)
    for row_index, row in enumerate(costs):
        row_weights = weights[row_index]
        valid = np.isfinite(row) & (row_weights > 0.0)
        if not valid.any():
            continue
        values = row[valid]
        valid_weights = row_weights[valid]
        matched_weight[row_index] = float(valid_weights.sum())
        normalized_weights = valid_weights / float(valid_weights.sum())
        effective_count[row_index] = float(
            1.0 / np.sum(np.square(normalized_weights))
        )
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        cumulative_weight = np.cumsum(normalized_weights[order])
        selected_index = int(
            np.searchsorted(cumulative_weight, quantile, side="left")
        )
        selected_index = min(
            selected_index,
            len(sorted_values) - 1,
        )
        aggregate[row_index] = float(sorted_values[selected_index])
    return aggregate, matched_weight, effective_count


def class_conditioned_anchor_weight_matrix(
    probability_matrix: np.ndarray,
    *,
    labels: list[str],
    base_weights: Mapping[str, float],
    class_weights: Mapping[str, Mapping[str, float]],
    conditioning_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend global and class-conditioned anchor reliability row by row."""

    probabilities = np.asarray(probability_matrix, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(
        OFFICIAL_CLASS_LABELS
    ):
        raise ValueError(
            "class probability matrix must have one column per official class"
        )
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError(
            "class probabilities must be finite and non-negative"
        )
    strength = float(conditioning_strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError(
            "class conditioning strength must be finite and in [0, 1]"
        )

    base_vector = np.asarray(
        [float(base_weights[label]) for label in labels],
        dtype=float,
    )
    class_matrix = np.asarray(
        [
            [
                float(class_weights[label][class_label])
                for label in labels
            ]
            for class_label in OFFICIAL_CLASS_LABELS
        ],
        dtype=float,
    )
    probability_sums = probabilities.sum(axis=1)
    normalized = np.divide(
        probabilities,
        probability_sums[:, None],
        out=np.zeros_like(probabilities),
        where=probability_sums[:, None] > 0.0,
    )
    expected = normalized @ class_matrix
    weights = (
        (1.0 - strength) * base_vector[None, :]
        + strength * expected
    )
    fallback = (probability_sums <= 0.0) | (
        np.sum(weights > 0.0, axis=1) == 0
    )
    if fallback.any():
        weights[fallback] = base_vector
    return weights, fallback


def resolve_anchor_class_reliability(
    labels: list[str],
    *,
    base_weights: Mapping[str, float],
    anchor_class_reliability: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, float]]:
    """Resolve a complete anchor-by-class reliability table."""

    supplied = {
        str(anchor): {
            str(class_label): float(weight)
            for class_label, weight in class_map.items()
        }
        for anchor, class_map in dict(
            anchor_class_reliability or {}
        ).items()
    }
    unknown_anchors = sorted(set(supplied) - set(labels))
    if unknown_anchors:
        raise ValueError(
            "class reliability supplied for unknown anchors: "
            f"{unknown_anchors}"
        )

    result: dict[str, dict[str, float]] = {}
    for label in labels:
        class_map = supplied.get(label, {})
        unknown_classes = sorted(
            set(class_map) - set(OFFICIAL_CLASS_LABELS)
        )
        if unknown_classes:
            raise ValueError(
                f"class reliability for {label!r} uses unsupported classes: "
                f"{unknown_classes}"
            )
        result[label] = {}
        for class_label in OFFICIAL_CLASS_LABELS:
            weight = float(
                class_map.get(class_label, base_weights[label])
            )
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"class reliability for anchor {label!r}, "
                    f"class {class_label!r} must be finite and non-negative"
                )
            result[label][class_label] = weight

    for class_label in OFFICIAL_CLASS_LABELS:
        if not any(
            result[label][class_label] > 0.0 for label in labels
        ):
            raise ValueError(
                f"at least one anchor reliability for class "
                f"{class_label!r} must be positive"
            )
    return result


def write_class_conditioned_anchor_quantile_outputs(
    result: MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write class-conditioned selection and grouped-MAP artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored_path = (
        output
        / "mmuad_class_conditioned_anchor_quantile_scored_candidates.csv"
    )
    selected_path = (
        output
        / "mmuad_class_conditioned_anchor_quantile_selected_candidates.csv"
    )
    anchors_path = (
        output
        / "mmuad_class_conditioned_anchor_quantile_normalized_anchors.csv"
    )
    summary_path = (
        output
        / "mmuad_class_conditioned_anchor_quantile_summary.json"
    )
    result.scored_candidates.to_csv(scored_path, index=False)
    result.selected_candidates.to_csv(selected_path, index=False)
    result.normalized_anchors.to_csv(anchors_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2),
        encoding="utf-8",
    )
    paths = write_grouped_candidate_mixture_outputs(
        result.grouped_result,
        output,
    )
    paths["class_conditioned_anchor_quantile_scored_candidates_csv"] = (
        scored_path
    )
    paths["class_conditioned_anchor_quantile_selected_candidates_csv"] = (
        selected_path
    )
    paths["class_conditioned_anchor_quantile_normalized_anchors_csv"] = (
        anchors_path
    )
    paths["class_conditioned_anchor_quantile_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.anchor_csv:
        parser.error("provide at least one --anchor-csv NAME=PATH")

    candidates = load_candidate_file(args.candidates_csv).rows
    anchors = _load_anchor_specs(args.anchor_csv)
    reliability = _parse_anchor_reliability_specs(
        args.anchor_reliability,
        set(anchors),
    )
    class_reliability = _parse_anchor_class_reliability_specs(
        args.anchor_class_reliability,
        set(anchors),
    )
    class_probabilities = read_class_probability_csv(
        args.class_probabilities_csv
    )
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
    result = (
        run_class_conditioned_anchor_quantile_group_topk_mixture_map(
            candidates,
            anchor_estimates=anchors,
            class_probabilities=class_probabilities,
            anchor_reliability=reliability,
            anchor_class_reliability=class_reliability,
            mixture_config=_mixture_config_from_args(args),
            group_config=HypothesisGroupConfig(
                group_column=args.hypothesis_group_column,
                correction_strength=(
                    args.hypothesis_group_correction_strength
                ),
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
            reliability_config=ClassConditionedAnchorReliabilityConfig(
                cost_quantile=args.anchor_cost_quantile,
                default_weight=args.default_anchor_reliability,
                conditioning_strength=args.class_conditioning_strength,
                fill_missing_class_probabilities=(
                    args.fill_missing_class_probabilities
                ),
            ),
            final_initial_estimates=final_initial,
            truth=truth,
        )
    )
    paths = write_class_conditioned_anchor_quantile_outputs(
        result,
        args.output_dir,
    )
    print("mmuad_class_conditioned_anchor_quantile_group_topk=ok")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"anchor_count={len(anchors)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = _build_weighted_quantile_parser()
    parser.prog = (
        "python -m "
        "raft_uav.mmuad."
        "candidate_mixture_group_class_conditioned_anchor_quantile"
    )
    parser.description = (
        "condition MMUAD physical-group selection on soft UAV class "
        "probabilities and per-class anchor reliability"
    )
    parser.add_argument(
        "--class-probabilities-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--anchor-class-reliability",
        action="append",
        default=[],
        metavar="NAME:CLASS=WEIGHT",
        help=(
            "per-class non-negative anchor reliability; may be repeated; "
            "unspecified anchor/classes fall back to global reliability"
        ),
    )
    parser.add_argument(
        "--class-conditioning-strength",
        type=float,
        default=1.0,
        help="blend in [0,1] between global and class-conditioned weights",
    )
    parser.add_argument(
        "--fill-missing-class-probabilities",
        choices=FILL_MISSING_POLICIES,
        default="uniform",
    )
    return parser


def _parse_anchor_class_reliability_specs(
    specs: list[str],
    known_anchors: set[str],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for spec in specs:
        if "=" not in spec or ":" not in spec.split("=", 1)[0]:
            raise ValueError(
                "anchor class reliability must use NAME:CLASS=WEIGHT"
            )
        left, weight_text = spec.split("=", 1)
        anchor, class_label = left.rsplit(":", 1)
        anchor = anchor.strip()
        class_label = class_label.strip()
        if anchor not in known_anchors:
            raise ValueError(
                f"class reliability supplied for unknown anchor {anchor!r}"
            )
        if class_label not in OFFICIAL_CLASS_LABELS:
            raise ValueError(
                f"unsupported Track 5 class label {class_label!r}"
            )
        try:
            weight = float(weight_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid class reliability weight in {spec!r}"
            ) from exc
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "anchor class reliability must be finite and non-negative"
            )
        result.setdefault(anchor, {})[class_label] = weight
    return result


def _attach_normalized_class_probabilities(
    scored: pd.DataFrame,
    class_probabilities: pd.DataFrame,
    *,
    fill_missing: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = scored.copy().reset_index(drop=True)
    rows["_class_conditioning_row_id"] = np.arange(len(rows), dtype=int)
    contextual = attach_class_probability_context(
        rows,
        class_probabilities,
        interaction_columns=(),
        fill_missing=fill_missing,
    ).rows
    contextual = contextual.sort_values(
        "_class_conditioning_row_id"
    ).reset_index(drop=True)
    if len(contextual) != len(rows):
        raise ValueError(
            "class probability merge changed the candidate row count"
        )
    probability_columns = [
        f"image_class_prob_{label}" for label in OFFICIAL_CLASS_LABELS
    ]
    probability_matrix = (
        contextual[probability_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    contextual = contextual.drop(
        columns=["_class_conditioning_row_id"],
        errors="ignore",
    )
    return contextual, probability_matrix


def _rowwise_best_anchor_indices(
    costs: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    result = np.full(costs.shape[0], -1, dtype=int)
    for row_index in range(costs.shape[0]):
        valid = np.isfinite(costs[row_index]) & (
            weights[row_index] > 0.0
        )
        if valid.any():
            valid_indices = np.flatnonzero(valid)
            result[row_index] = int(
                valid_indices[
                    np.argmin(costs[row_index, valid_indices])
                ]
            )
    return result


def _validate_reliability_config(
    config: ClassConditionedAnchorReliabilityConfig,
) -> None:
    quantile = float(config.cost_quantile)
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError(
            "anchor cost quantile must be finite and in [0, 1]"
        )
    default_weight = float(config.default_weight)
    if not np.isfinite(default_weight) or default_weight < 0.0:
        raise ValueError(
            "default anchor reliability must be finite and non-negative"
        )
    strength = float(config.conditioning_strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError(
            "class conditioning strength must be finite and in [0, 1]"
        )
    if (
        config.fill_missing_class_probabilities
        not in FILL_MISSING_POLICIES
    ):
        raise ValueError(
            "unsupported missing class probability policy "
            f"{config.fill_missing_class_probabilities!r}"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
