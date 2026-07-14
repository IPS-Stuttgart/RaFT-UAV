"""Confidence-adaptive class conditioning for MMUAD anchor reliability.

The fixed-strength class-conditioned selector can over-trust an uncertain UAV
class posterior.  This module scales the class-conditioning strength per
candidate row using an inference-time confidence statistic computed from the
soft class probabilities.  Confident posteriors retain the train-selected
class-specific anchor reliability profile, while ambiguous or uniform
posteriors fall back toward the global anchor reliability profile.

For maximum conditioning strength ``lambda`` and confidence ``q(p)``:

``lambda_eff = lambda * (floor + (1-floor) * q(p) ** power)``.

The effective strength is used only for finite physical-hypothesis selection.
The final grouped learned-sigma / Huber mixture-MAP objective is unchanged and
no ground truth is consumed at inference time.
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
from raft_uav.mmuad.candidate_mixture_group_class_conditioned_anchor_quantile import (
    CLASS_CONDITIONED_COST_COLUMN,
    CLASS_CONDITIONED_UTILITY_COLUMN,
    ClassConditionedAnchorReliabilityConfig,
    _build_parser as _build_class_conditioned_parser,
    _parse_anchor_class_reliability_specs,
    _rowwise_best_anchor_indices,
    _validate_reliability_config,
    add_class_conditioned_anchor_quantile_selection_utility,
    aggregate_rowwise_weighted_quantile_anchor_costs,
    resolve_anchor_class_reliability,
)
from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
    select_posterior_mass_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_group_multi_anchor_mass_topk import (
    MULTI_ANCHOR_UTILITY_COLUMN,
    MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult,
    _jsonable,
    _load_anchor_specs,
    _mixture_config_from_args,
    _selection_config_from_args,
    _unique_anchor_slugs,
)
from raft_uav.mmuad.candidate_mixture_group_weighted_anchor_quantile import (
    WEIGHTED_QUANTILE_COST_COLUMN,
    WEIGHTED_QUANTILE_UTILITY_COLUMN,
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
from raft_uav.mmuad.class_probability_context import OFFICIAL_CLASS_LABELS
from raft_uav.mmuad.class_probability_csv import read_class_probability_csv
from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file

CONFIDENCE_MODES = ("none", "entropy", "max-probability", "margin")
ADAPTIVE_COST_COLUMN = (
    "mixture_confidence_adaptive_class_anchor_quantile_aggregate_cost"
)
ADAPTIVE_UTILITY_COLUMN = (
    "mixture_confidence_adaptive_class_anchor_quantile_selection_utility"
)
CONFIDENCE_SCORE_COLUMN = "mixture_class_probability_confidence"
EFFECTIVE_STRENGTH_COLUMN = "mixture_class_conditioning_strength_effective"


@dataclass(frozen=True)
class ConfidenceAdaptiveClassConditioningConfig:
    """Configuration for confidence-adaptive class conditioning."""

    confidence_mode: str = "entropy"
    confidence_power: float = 1.0
    confidence_floor: float = 0.0


def class_probability_confidence(
    probability_matrix: np.ndarray,
    *,
    mode: str = "entropy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize class probabilities and return confidence plus missing mask.

    Confidence is in ``[0, 1]``.  ``entropy`` maps a uniform posterior to zero
    and a one-hot posterior to one.  ``max-probability`` removes the uniform
    baseline before scaling.  ``margin`` uses the largest-minus-second-largest
    probability.  ``none`` returns one for every non-missing row.
    """

    probabilities = np.asarray(probability_matrix, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(
        OFFICIAL_CLASS_LABELS
    ):
        raise ValueError(
            "class probability matrix must have one column per official class"
        )
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any():
        raise ValueError("class probabilities must be finite and non-negative")
    if mode not in CONFIDENCE_MODES:
        raise ValueError(f"unsupported class confidence mode {mode!r}")

    probability_sums = probabilities.sum(axis=1)
    missing = probability_sums <= 0.0
    normalized = np.divide(
        probabilities,
        probability_sums[:, None],
        out=np.zeros_like(probabilities),
        where=probability_sums[:, None] > 0.0,
    )
    confidence = np.zeros(len(normalized), dtype=float)
    valid = ~missing
    if not valid.any():
        return normalized, confidence, missing

    valid_probabilities = normalized[valid]
    if mode == "none":
        confidence[valid] = 1.0
    elif mode == "entropy":
        safe = np.where(valid_probabilities > 0.0, valid_probabilities, 1.0)
        entropy = -np.sum(
            np.where(
                valid_probabilities > 0.0,
                valid_probabilities * np.log(safe),
                0.0,
            ),
            axis=1,
        )
        confidence[valid] = 1.0 - entropy / np.log(len(OFFICIAL_CLASS_LABELS))
    elif mode == "max-probability":
        uniform = 1.0 / len(OFFICIAL_CLASS_LABELS)
        maximum = np.max(valid_probabilities, axis=1)
        confidence[valid] = (maximum - uniform) / (1.0 - uniform)
    else:
        ordered = np.sort(valid_probabilities, axis=1)
        confidence[valid] = ordered[:, -1] - ordered[:, -2]

    confidence = np.clip(confidence, 0.0, 1.0)
    return normalized, confidence, missing


def confidence_adaptive_class_conditioned_anchor_weight_matrix(
    probability_matrix: np.ndarray,
    *,
    labels: list[str],
    base_weights: Mapping[str, float],
    class_weights: Mapping[str, Mapping[str, float]],
    conditioning_strength: float,
    confidence_config: ConfidenceAdaptiveClassConditioningConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blend global and class-specific reliability with row-wise confidence."""

    config = confidence_config or ConfidenceAdaptiveClassConditioningConfig()
    _validate_confidence_config(config)
    maximum_strength = float(conditioning_strength)
    if not np.isfinite(maximum_strength) or not 0.0 <= maximum_strength <= 1.0:
        raise ValueError(
            "class conditioning strength must be finite and in [0, 1]"
        )

    normalized, confidence, missing = class_probability_confidence(
        probability_matrix,
        mode=config.confidence_mode,
    )
    confidence_factor = (
        float(config.confidence_floor)
        + (1.0 - float(config.confidence_floor))
        * np.power(confidence, float(config.confidence_power))
    )
    effective_strength = maximum_strength * confidence_factor
    effective_strength[missing] = 0.0

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
    expected = normalized @ class_matrix
    weights = (
        (1.0 - effective_strength[:, None]) * base_vector[None, :]
        + effective_strength[:, None] * expected
    )
    fallback = missing | (np.sum(weights > 0.0, axis=1) == 0)
    if fallback.any():
        weights[fallback] = base_vector
        effective_strength[fallback] = 0.0
    return weights, fallback, confidence, effective_strength


def add_confidence_adaptive_class_anchor_quantile_selection_utility(
    candidates: pd.DataFrame,
    anchor_estimates: Mapping[str, pd.DataFrame],
    class_probabilities: pd.DataFrame,
    *,
    anchor_reliability: Mapping[str, float] | None = None,
    anchor_class_reliability: Mapping[str, Mapping[str, float]] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    reliability_config: ClassConditionedAnchorReliabilityConfig | None = None,
    confidence_config: ConfidenceAdaptiveClassConditioningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach confidence-adaptive class-conditioned anchor utilities."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    anchor_config = anchor_config or AnchorConditioningConfig()
    reliability_config = (
        reliability_config or ClassConditionedAnchorReliabilityConfig()
    )
    confidence_config = (
        confidence_config or ConfidenceAdaptiveClassConditioningConfig()
    )
    _validate_reliability_config(reliability_config)
    _validate_confidence_config(confidence_config)

    # Reuse the established scorer to obtain normalized anchors, per-anchor
    # costs, class-probability context, and compatibility diagnostics.  A zero
    # base conditioning strength prevents the fixed-strength result from
    # affecting the adaptive recomputation below.
    base_reliability_config = replace(reliability_config, conditioning_strength=0.0)
    scored, normalized_anchors, base_summary = (
        add_class_conditioned_anchor_quantile_selection_utility(
            candidates,
            anchor_estimates,
            class_probabilities,
            anchor_reliability=anchor_reliability,
            anchor_class_reliability=anchor_class_reliability,
            mixture_config=mixture_config,
            anchor_config=anchor_config,
            reliability_config=base_reliability_config,
        )
    )

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
    probability_columns = [
        f"image_class_prob_{label}" for label in OFFICIAL_CLASS_LABELS
    ]
    probability_matrix = (
        scored[probability_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
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
    row_weights, fallback_rows, confidence, effective_strength = (
        confidence_adaptive_class_conditioned_anchor_weight_matrix(
            probability_matrix,
            labels=labels,
            base_weights=base_weights,
            class_weights=class_weights,
            conditioning_strength=reliability_config.conditioning_strength,
            confidence_config=confidence_config,
        )
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
    scored[ADAPTIVE_COST_COLUMN] = aggregate_cost
    scored[ADAPTIVE_UTILITY_COLUMN] = (
        pd.to_numeric(
            scored["mixture_multi_anchor_base_utility"],
            errors="coerce",
        )
        - float(anchor_config.anchor_selection_weight) * aggregate_cost
    )
    scored[CONFIDENCE_SCORE_COLUMN] = confidence
    scored[EFFECTIVE_STRENGTH_COLUMN] = effective_strength
    scored[
        "mixture_confidence_adaptive_class_anchor_quantile_matched_weight"
    ] = matched_weight
    scored[
        "mixture_confidence_adaptive_class_anchor_quantile_matched_weight_fraction"
    ] = np.divide(
        matched_weight,
        weight_totals,
        out=np.zeros_like(matched_weight),
        where=weight_totals > 0.0,
    )
    scored[
        "mixture_confidence_adaptive_class_anchor_quantile_effective_anchor_count"
    ] = effective_anchor_count
    scored[
        "mixture_confidence_adaptive_class_anchor_quantile_best_anchor"
    ] = best_anchor
    scored[
        "mixture_confidence_adaptive_class_anchor_reliability_fallback"
    ] = fallback_rows.astype(float)
    for anchor_index, slug in enumerate(slugs):
        scored[
            f"mixture_confidence_adaptive_class_anchor_weight_{slug}"
        ] = row_weights[:, anchor_index]
        # Keep the existing class-conditioned diagnostics aligned with the
        # adaptive method for downstream tools that already consume them.
        scored[f"mixture_class_conditioned_anchor_weight_{slug}"] = row_weights[
            :, anchor_index
        ]

    # Preserve established selector compatibility columns.
    scored[CLASS_CONDITIONED_COST_COLUMN] = aggregate_cost
    scored[CLASS_CONDITIONED_UTILITY_COLUMN] = scored[ADAPTIVE_UTILITY_COLUMN]
    scored[WEIGHTED_QUANTILE_COST_COLUMN] = aggregate_cost
    scored[WEIGHTED_QUANTILE_UTILITY_COLUMN] = scored[ADAPTIVE_UTILITY_COLUMN]
    scored[WEIGHTED_MULTI_ANCHOR_COST_COLUMN] = aggregate_cost
    scored[WEIGHTED_MULTI_ANCHOR_UTILITY_COLUMN] = scored[ADAPTIVE_UTILITY_COLUMN]
    scored[MULTI_ANCHOR_UTILITY_COLUMN] = scored[ADAPTIVE_UTILITY_COLUMN]
    scored[
        "mixture_class_conditioned_anchor_reliability_fallback"
    ] = fallback_rows.astype(float)

    summary = dict(base_summary)
    summary["schema"] = (
        "raft-uav-mmuad-confidence-adaptive-class-anchor-quantile-conditioning-v1"
    )
    summary["reliability_config"] = asdict(reliability_config)
    summary["confidence_config"] = asdict(confidence_config)
    summary["mean_class_probability_confidence"] = _safe_mean(confidence)
    summary["mean_effective_class_conditioning_strength"] = _safe_mean(
        effective_strength
    )
    summary["effective_class_conditioning_strength_p50"] = _safe_quantile(
        effective_strength,
        0.50,
    )
    summary["effective_class_conditioning_strength_p95"] = _safe_quantile(
        effective_strength,
        0.95,
    )
    summary["effective_class_conditioning_strength_max"] = _safe_max(
        effective_strength
    )
    summary["class_conditioning_fallback_rows"] = int(fallback_rows.sum())
    summary["class_conditioning_fallback_rate"] = float(
        fallback_rows.mean() if len(fallback_rows) else 0.0
    )
    summary["mean_matched_anchor_weight"] = _safe_mean(matched_weight)
    summary["mean_effective_anchor_count"] = _safe_mean(effective_anchor_count)
    summary["mean_effective_anchor_weight"] = {
        label: float(np.mean(row_weights[:, index]))
        for index, label in enumerate(labels)
    }
    summary["best_anchor_counts"] = {
        str(key): int(value)
        for key, value in pd.Series(best_anchor).value_counts().items()
        if str(key)
    }
    summary["truth_used_for_weighting"] = False
    return scored, normalized_anchors, _jsonable(summary)


def select_confidence_adaptive_class_anchor_quantile_group_topk(
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
    confidence_config: ConfidenceAdaptiveClassConditioningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select physical groups using confidence-adaptive class conditioning."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    scored, normalized_anchors, conditioning_summary = (
        add_confidence_adaptive_class_anchor_quantile_selection_utility(
            candidates,
            anchor_estimates,
            class_probabilities,
            anchor_reliability=anchor_reliability,
            anchor_class_reliability=anchor_class_reliability,
            mixture_config=mixture_config,
            anchor_config=anchor_config,
            reliability_config=reliability_config,
            confidence_config=confidence_config,
        )
    )
    selection_mixture_config = replace(
        mixture_config,
        score_column=ADAPTIVE_UTILITY_COLUMN,
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
        "raft-uav-mmuad-confidence-adaptive-class-anchor-quantile-group-topk-v1"
    )
    summary["confidence_adaptive_class_anchor_quantile"] = conditioning_summary
    summary["selection_mixture_config"] = asdict(selection_mixture_config)
    summary["truth_used_for_selection"] = False
    return scored, selected, normalized_anchors, _jsonable(summary)


def run_confidence_adaptive_class_anchor_quantile_group_topk_mixture_map(
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
    confidence_config: ConfidenceAdaptiveClassConditioningConfig | None = None,
    final_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult:
    """Run adaptive selection and the unchanged grouped mixture-MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    scored, selected, normalized_anchors, summary = (
        select_confidence_adaptive_class_anchor_quantile_group_topk(
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
            confidence_config=confidence_config,
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


def write_confidence_adaptive_class_anchor_quantile_outputs(
    result: MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write adaptive-selection and grouped-MAP artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored_path = output / "mmuad_confidence_adaptive_class_scored_candidates.csv"
    selected_path = output / "mmuad_confidence_adaptive_class_selected_candidates.csv"
    anchors_path = output / "mmuad_confidence_adaptive_class_normalized_anchors.csv"
    summary_path = output / "mmuad_confidence_adaptive_class_summary.json"
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
    paths["confidence_adaptive_class_scored_candidates_csv"] = scored_path
    paths["confidence_adaptive_class_selected_candidates_csv"] = selected_path
    paths["confidence_adaptive_class_normalized_anchors_csv"] = anchors_path
    paths["confidence_adaptive_class_summary_json"] = summary_path
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
        run_confidence_adaptive_class_anchor_quantile_group_topk_mixture_map(
            candidates,
            anchor_estimates=anchors,
            class_probabilities=class_probabilities,
            anchor_reliability=reliability,
            anchor_class_reliability=class_reliability,
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
            reliability_config=ClassConditionedAnchorReliabilityConfig(
                cost_quantile=args.anchor_cost_quantile,
                default_weight=args.default_anchor_reliability,
                conditioning_strength=args.class_conditioning_strength,
                fill_missing_class_probabilities=(
                    args.fill_missing_class_probabilities
                ),
            ),
            confidence_config=ConfidenceAdaptiveClassConditioningConfig(
                confidence_mode=args.class_confidence_mode,
                confidence_power=args.class_confidence_power,
                confidence_floor=args.class_confidence_floor,
            ),
            final_initial_estimates=final_initial,
            truth=truth,
        )
    )
    paths = write_confidence_adaptive_class_anchor_quantile_outputs(
        result,
        args.output_dir,
    )
    print("mmuad_confidence_adaptive_class_anchor_quantile_group_topk=ok")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"anchor_count={len(anchors)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = _build_class_conditioned_parser()
    parser.prog = (
        "python -m raft_uav.mmuad."
        "candidate_mixture_group_confidence_adaptive_class_anchor_quantile"
    )
    parser.description = (
        "adapt MMUAD class-conditioned anchor reliability to soft classifier "
        "confidence before physical-group selection"
    )
    parser.add_argument(
        "--class-confidence-mode",
        choices=CONFIDENCE_MODES,
        default="entropy",
        help=(
            "confidence statistic that scales class conditioning; entropy maps "
            "uniform probabilities to zero and one-hot probabilities to one"
        ),
    )
    parser.add_argument(
        "--class-confidence-power",
        type=float,
        default=1.0,
        help="positive exponent applied to the normalized class confidence",
    )
    parser.add_argument(
        "--class-confidence-floor",
        type=float,
        default=0.0,
        help=(
            "minimum fraction in [0,1] of the configured class-conditioning "
            "strength retained for non-missing probability rows"
        ),
    )
    return parser


def _validate_confidence_config(
    config: ConfidenceAdaptiveClassConditioningConfig,
) -> None:
    if config.confidence_mode not in CONFIDENCE_MODES:
        raise ValueError(
            f"unsupported class confidence mode {config.confidence_mode!r}"
        )
    power = float(config.confidence_power)
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("class confidence power must be finite and positive")
    floor = float(config.confidence_floor)
    if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("class confidence floor must be finite and in [0, 1]")


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.quantile(finite, quantile)) if finite.size else float("nan")


def _safe_max(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else float("nan")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
