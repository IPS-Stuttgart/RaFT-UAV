"""Multi-anchor adaptive hypothesis-group selection for MMUAD mixture-MAP.

A single trajectory anchor can improve candidate recall, but it can also prune a
second plausible mode before the robust trajectory smoother sees it. This module
conditions the finite physical-group budget on several inference-time trajectory
anchors and aggregates their bounded Huber costs without using ground truth.

The default ``minimum`` aggregation preserves a candidate that is coherent with
*any* supplied anchor. ``softmin`` offers a train-selectable compromise that
slightly rewards agreement with several anchors. Anchor-conditioned utilities are
used only for group selection; the final grouped mixture-MAP still uses the
original ranker score, learned uncertainty, robust loss, and smoothness objective.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from raft_uav.mmuad import candidate_mixture_map as core
from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
    add_anchor_conditioned_selection_utility,
)
from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
    select_posterior_mass_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_group_topk import GROUP_SCORE_MODES
from raft_uav.mmuad.candidate_mixture_map import (
    INITIALIZATION_CHOICES,
    LOSS_CHOICES,
    SCORE_NORMALIZATION_CHOICES,
    CandidateMixtureMapConfig,
)
from raft_uav.mmuad.candidate_mixture_map_grouped import (
    GroupedCandidateMixtureMapResult,
    HypothesisGroupConfig,
    run_grouped_candidate_mixture_map,
    write_grouped_candidate_mixture_outputs,
)
from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns

MULTI_ANCHOR_UTILITY_COLUMN = "mixture_multi_anchor_conditioned_selection_utility"
ANCHOR_AGGREGATION_CHOICES = ("minimum", "softmin", "mean")


@dataclass(frozen=True)
class MultiAnchorAggregationConfig:
    """Configuration for combining several anchor-conditioned costs."""

    aggregation: str = "minimum"
    softmin_temperature: float = 0.5


@dataclass(frozen=True)
class MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult:
    """Full multi-anchor scoring, selected rows, and final grouped MAP result."""

    scored_candidates: pd.DataFrame
    selected_candidates: pd.DataFrame
    normalized_anchors: pd.DataFrame
    grouped_result: GroupedCandidateMixtureMapResult
    selection_summary: dict[str, Any]


def add_multi_anchor_conditioned_selection_utility(
    candidates: pd.DataFrame,
    anchor_estimates: Mapping[str, pd.DataFrame],
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    aggregation_config: MultiAnchorAggregationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach a selection unary conditioned on several alternative trajectories."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    anchor_config = anchor_config or AnchorConditioningConfig()
    aggregation_config = aggregation_config or MultiAnchorAggregationConfig()
    _validate_aggregation_config(aggregation_config)
    if not anchor_estimates:
        raise ValueError("at least one anchor trajectory is required")

    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy()).reset_index(drop=True)
    labels = [str(label).strip() for label in anchor_estimates]
    if any(not label for label in labels):
        raise ValueError("anchor labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError("anchor labels must be unique")
    slugs = _unique_anchor_slugs(labels)

    neutral_anchor_config = replace(
        anchor_config,
        anchor_selection_weight=0.0,
        missing_anchor_policy="neutral",
    )
    cost_columns: list[np.ndarray] = []
    distance_columns: list[np.ndarray] = []
    matched_columns: list[np.ndarray] = []
    normalized_anchor_parts: list[pd.DataFrame] = []
    anchor_summaries: dict[str, Any] = {}
    scored: pd.DataFrame | None = None

    for label, slug in zip(labels, slugs, strict=True):
        anchor_scored, normalized, anchor_summary = add_anchor_conditioned_selection_utility(
            rows,
            anchor_estimates[label],
            mixture_config=mixture_config,
            anchor_config=neutral_anchor_config,
        )
        if scored is None:
            scored = rows.copy()
            scored["mixture_multi_anchor_base_raw_score"] = anchor_scored[
                "mixture_anchor_base_raw_score"
            ].to_numpy(float)
            scored["mixture_multi_anchor_sigma_m"] = anchor_scored[
                "mixture_anchor_sigma_m"
            ].to_numpy(float)
            scored["mixture_multi_anchor_base_utility"] = anchor_scored[
                "mixture_anchor_base_utility"
            ].to_numpy(float)
        matched = anchor_scored["mixture_anchor_matched"].astype(bool).to_numpy()
        distance = pd.to_numeric(
            anchor_scored["mixture_anchor_distance_m"], errors="coerce"
        ).to_numpy(float)
        cost = pd.to_numeric(anchor_scored["mixture_anchor_cost"], errors="coerce").to_numpy(
            float
        )
        cost = np.where(matched, cost, np.nan)
        distance = np.where(matched, distance, np.nan)
        cost_columns.append(cost)
        distance_columns.append(distance)
        matched_columns.append(matched)
        scored[f"mixture_multi_anchor_{slug}_matched"] = matched
        scored[f"mixture_multi_anchor_{slug}_time_delta_s"] = pd.to_numeric(
            anchor_scored["mixture_anchor_time_delta_s"], errors="coerce"
        ).to_numpy(float)
        scored[f"mixture_multi_anchor_{slug}_distance_m"] = distance
        scored[f"mixture_multi_anchor_{slug}_cost"] = cost
        normalized_part = normalized.copy()
        normalized_part.insert(0, "anchor_name", label)
        normalized_anchor_parts.append(normalized_part)
        anchor_summaries[label] = anchor_summary

    assert scored is not None
    cost_matrix = np.column_stack(cost_columns)
    distance_matrix = np.column_stack(distance_columns)
    matched_matrix = np.column_stack(matched_columns)
    matched_count = matched_matrix.sum(axis=1).astype(int)
    aggregate_cost = _aggregate_anchor_costs(
        cost_matrix,
        aggregation_config=aggregation_config,
    )
    best_index = _best_anchor_indices(cost_matrix)
    best_anchor = np.asarray(
        [labels[index] if index >= 0 else "" for index in best_index], dtype=object
    )
    best_distance = np.asarray(
        [distance_matrix[row, index] if index >= 0 else np.nan for row, index in enumerate(best_index)],
        dtype=float,
    )

    scored["mixture_multi_anchor_matched_count"] = matched_count
    scored["mixture_multi_anchor_matched_fraction"] = matched_count / float(len(labels))
    scored["mixture_multi_anchor_best_anchor"] = best_anchor
    scored["mixture_multi_anchor_best_distance_m"] = best_distance
    scored["mixture_multi_anchor_aggregate_cost"] = aggregate_cost
    scored[MULTI_ANCHOR_UTILITY_COLUMN] = (
        scored["mixture_multi_anchor_base_utility"].to_numpy(float)
        - float(anchor_config.anchor_selection_weight) * aggregate_cost
    )

    unmatched = matched_count == 0
    if unmatched.any() and anchor_config.missing_anchor_policy == "error":
        missing_frames = (
            scored.loc[unmatched, ["sequence_id", "time_s"]]
            .drop_duplicates()
            .head(5)
            .itertuples(index=False, name=None)
        )
        examples = ", ".join(f"{sequence}@{float(time_s):g}" for sequence, time_s in missing_frames)
        raise ValueError(
            "missing support from every anchor trajectory for candidate frames: " + examples
        )

    normalized_anchors = pd.concat(normalized_anchor_parts, ignore_index=True)
    summary = _multi_anchor_summary(
        scored,
        normalized_anchors,
        labels=labels,
        anchor_summaries=anchor_summaries,
        anchor_config=anchor_config,
        aggregation_config=aggregation_config,
    )
    return scored, normalized_anchors, summary


def select_multi_anchor_posterior_mass_hypothesis_group_topk(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    aggregation_config: MultiAnchorAggregationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select adaptive physical groups using a multi-anchor conditioned unary."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    scored, normalized_anchors, multi_anchor_summary = (
        add_multi_anchor_conditioned_selection_utility(
            candidates,
            anchor_estimates,
            mixture_config=mixture_config,
            anchor_config=anchor_config,
            aggregation_config=aggregation_config,
        )
    )
    selection_mixture_config = replace(
        mixture_config,
        score_column=MULTI_ANCHOR_UTILITY_COLUMN,
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
    summary["schema"] = "raft-uav-mmuad-multi-anchor-posterior-mass-group-topk-v1"
    summary["multi_anchor_conditioning"] = multi_anchor_summary
    summary["selection_mixture_config"] = asdict(selection_mixture_config)
    summary["truth_used_for_selection"] = False
    return scored, selected, normalized_anchors, _jsonable(summary)


def run_multi_anchor_posterior_mass_group_topk_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    aggregation_config: MultiAnchorAggregationConfig | None = None,
    final_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult:
    """Run multi-anchor group selection followed by grouped robust mixture-MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    group_config = group_config or HypothesisGroupConfig()
    scored, selected, normalized_anchors, summary = (
        select_multi_anchor_posterior_mass_hypothesis_group_topk(
            candidates,
            anchor_estimates=anchor_estimates,
            mixture_config=mixture_config,
            group_config=group_config,
            selection_config=selection_config,
            anchor_config=anchor_config,
            aggregation_config=aggregation_config,
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


def write_multi_anchor_posterior_mass_group_topk_outputs(
    result: MultiAnchorPosteriorMassGroupTopKCandidateMixtureResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write full scoring, selection, anchor, and grouped mixture diagnostics."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored_path = output / "mmuad_multi_anchor_posterior_mass_scored_candidates.csv"
    selected_path = output / "mmuad_multi_anchor_posterior_mass_selected_candidates.csv"
    anchors_path = output / "mmuad_multi_anchor_normalized_anchors.csv"
    summary_path = output / "mmuad_multi_anchor_posterior_mass_summary.json"
    result.scored_candidates.to_csv(scored_path, index=False)
    result.selected_candidates.to_csv(selected_path, index=False)
    result.normalized_anchors.to_csv(anchors_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2), encoding="utf-8"
    )
    paths = write_grouped_candidate_mixture_outputs(result.grouped_result, output)
    paths["multi_anchor_scored_candidates_csv"] = scored_path
    paths["multi_anchor_selected_candidates_csv"] = selected_path
    paths["multi_anchor_normalized_anchors_csv"] = anchors_path
    paths["multi_anchor_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m raft_uav.mmuad.candidate_mixture_group_multi_anchor_mass_topk",
        description="condition adaptive MMUAD physical-group selection on several anchors",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument(
        "--anchor-csv",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="alternative trajectory anchor; may be repeated",
    )
    parser.add_argument("--final-initial-estimates-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--aggregation", choices=ANCHOR_AGGREGATION_CHOICES, default="minimum")
    parser.add_argument("--softmin-temperature", type=float, default=0.5)
    parser.add_argument("--anchor-selection-weight", type=float, default=1.0)
    parser.add_argument("--anchor-scale-m", type=float, default=10.0)
    parser.add_argument("--anchor-huber-delta", type=float, default=1.0)
    parser.add_argument("--anchor-cost-cap", type=float, default=4.0)
    parser.add_argument("--anchor-time-tolerance-s", type=float, default=0.5)
    parser.add_argument(
        "--missing-anchor-policy", choices=("neutral", "error"), default="neutral"
    )
    parser.add_argument("--min-group-top-k", type=int, default=3)
    parser.add_argument("--max-group-top-k", type=int, default=20)
    parser.add_argument("--target-posterior-mass", type=float, default=0.95)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--uniform-posterior-blend", type=float, default=0.02)
    parser.add_argument("--max-siblings-per-group", type=int, default=2)
    parser.add_argument("--group-score-mode", choices=GROUP_SCORE_MODES, default="max")
    parser.add_argument("--diversity-weight", type=float, default=0.5)
    parser.add_argument("--diversity-scale-m", type=float, default=5.0)
    parser.add_argument("--diversity-cap-m", type=float, default=30.0)
    parser.add_argument("--row-top-k-when-disabled", type=int, default=20)
    parser.add_argument("--score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--fallback-score-column", action="append", default=[])
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument(
        "--score-normalization", choices=SCORE_NORMALIZATION_CHOICES, default="minmax"
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
        "--initialization", choices=INITIALIZATION_CHOICES, default="uncertainty-top1"
    )
    parser.add_argument("--hypothesis-group-column")
    parser.add_argument("--hypothesis-group-correction-strength", type=float, default=1.0)
    parser.add_argument(
        "--missing-hypothesis-group-policy", choices=("unique", "error"), default="unique"
    )
    args = parser.parse_args(argv)
    if not args.anchor_csv:
        parser.error("provide at least one --anchor-csv NAME=PATH")

    fallback = tuple(args.fallback_score_column) or ("ranker_score", "confidence")
    mixture_config = CandidateMixtureMapConfig(
        top_k=args.row_top_k_when_disabled,
        score_column=args.score_column,
        fallback_score_columns=fallback,
        sigma_column=args.sigma_column,
        default_sigma_m=args.default_sigma_m,
        sigma_min_m=args.sigma_min_m,
        sigma_max_m=args.sigma_max_m,
        score_normalization=args.score_normalization,
        score_weight=args.score_weight,
        temperature=args.temperature,
        sigma_log_weight=args.sigma_log_weight,
        loss=args.loss,
        huber_delta=args.huber_delta,
        smoothness_weight=args.smoothness_weight,
        anchor_weight=args.anchor_weight,
        iterations=args.iterations,
        tolerance_m=args.tolerance_m,
        uniform_weight_floor=args.uniform_weight_floor,
        branch_balance=args.branch_balance,
        source_balance=args.source_balance,
        responsibility_floor=args.responsibility_floor,
        initialization=args.initialization,
    )
    group_config = HypothesisGroupConfig(
        group_column=args.hypothesis_group_column,
        correction_strength=args.hypothesis_group_correction_strength,
        missing_group_policy=args.missing_hypothesis_group_policy,
    )
    selection_config = PosteriorMassGroupTopKConfig(
        min_group_top_k=args.min_group_top_k,
        max_group_top_k=args.max_group_top_k,
        target_posterior_mass=args.target_posterior_mass,
        posterior_temperature=args.posterior_temperature,
        uniform_posterior_blend=args.uniform_posterior_blend,
        max_siblings_per_group=args.max_siblings_per_group,
        group_score_mode=args.group_score_mode,
        diversity_weight=args.diversity_weight,
        diversity_scale_m=args.diversity_scale_m,
        diversity_cap_m=args.diversity_cap_m,
    )
    anchor_config = AnchorConditioningConfig(
        anchor_selection_weight=args.anchor_selection_weight,
        anchor_scale_m=args.anchor_scale_m,
        anchor_huber_delta=args.anchor_huber_delta,
        anchor_cost_cap=args.anchor_cost_cap,
        anchor_time_tolerance_s=args.anchor_time_tolerance_s,
        missing_anchor_policy=args.missing_anchor_policy,
    )
    aggregation_config = MultiAnchorAggregationConfig(
        aggregation=args.aggregation,
        softmin_temperature=args.softmin_temperature,
    )
    candidates = load_candidate_file(args.candidates_csv).rows
    anchors = _load_anchor_specs(args.anchor_csv)
    final_initial = (
        None
        if args.final_initial_estimates_csv is None
        else read_estimate_csv(args.final_initial_estimates_csv)
    )
    truth = (
        None if args.truth_csv is None else load_evaluation_truth_file(args.truth_csv).rows
    )
    result = run_multi_anchor_posterior_mass_group_topk_candidate_mixture_map(
        candidates,
        anchor_estimates=anchors,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
        anchor_config=anchor_config,
        aggregation_config=aggregation_config,
        final_initial_estimates=final_initial,
        truth=truth,
    )
    paths = write_multi_anchor_posterior_mass_group_topk_outputs(result, args.output_dir)
    print("mmuad_multi_anchor_posterior_mass_group_topk=ok")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"anchor_count={len(anchors)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    pooled = result.grouped_result.mixture_result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _aggregate_anchor_costs(
    cost_matrix: np.ndarray,
    *,
    aggregation_config: MultiAnchorAggregationConfig,
) -> np.ndarray:
    matched = np.isfinite(cost_matrix)
    result = np.zeros(cost_matrix.shape[0], dtype=float)
    for row_index in range(cost_matrix.shape[0]):
        values = cost_matrix[row_index, matched[row_index]]
        if len(values) == 0:
            continue
        if aggregation_config.aggregation == "minimum":
            result[row_index] = float(np.min(values))
        elif aggregation_config.aggregation == "mean":
            result[row_index] = float(np.mean(values))
        else:
            temperature = float(aggregation_config.softmin_temperature)
            minimum = float(np.min(values))
            shifted = np.exp(-(values - minimum) / temperature)
            result[row_index] = minimum - temperature * float(np.log(np.mean(shifted)))
    return result


def _best_anchor_indices(cost_matrix: np.ndarray) -> np.ndarray:
    indices = np.full(cost_matrix.shape[0], -1, dtype=int)
    for row_index, values in enumerate(cost_matrix):
        finite = np.isfinite(values)
        if finite.any():
            finite_indices = np.flatnonzero(finite)
            indices[row_index] = int(finite_indices[np.argmin(values[finite])])
    return indices


def _load_anchor_specs(specs: list[str]) -> dict[str, pd.DataFrame]:
    anchors: dict[str, pd.DataFrame] = {}
    for spec in specs:
        if "=" in spec:
            label, path_text = spec.split("=", 1)
        else:
            path_text = spec
            label = Path(spec).stem
        label = str(label).strip()
        if not label:
            raise ValueError(f"invalid empty anchor label in {spec!r}")
        if label in anchors:
            raise ValueError(f"duplicate anchor label {label!r}")
        anchors[label] = read_estimate_csv(Path(path_text))
    return anchors


def _unique_anchor_slugs(labels: list[str]) -> list[str]:
    result: list[str] = []
    used: set[str] = set()
    for label in labels:
        base = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "anchor"
        slug = base
        suffix = 2
        while slug in used:
            slug = f"{base}_{suffix}"
            suffix += 1
        used.add(slug)
        result.append(slug)
    return result


def _multi_anchor_summary(
    rows: pd.DataFrame,
    normalized_anchors: pd.DataFrame,
    *,
    labels: list[str],
    anchor_summaries: dict[str, Any],
    anchor_config: AnchorConditioningConfig,
    aggregation_config: MultiAnchorAggregationConfig,
) -> dict[str, Any]:
    frame_rows = rows[
        ["sequence_id", "time_s", "mixture_multi_anchor_matched_count"]
    ].drop_duplicates()
    matched_frames = int((frame_rows["mixture_multi_anchor_matched_count"] > 0).sum())
    best_counts = {
        str(key): int(value)
        for key, value in rows["mixture_multi_anchor_best_anchor"].value_counts().items()
        if str(key)
    }
    aggregate_cost = pd.to_numeric(
        rows["mixture_multi_anchor_aggregate_cost"], errors="coerce"
    )
    best_distance = pd.to_numeric(
        rows["mixture_multi_anchor_best_distance_m"], errors="coerce"
    ).dropna()
    return {
        "anchor_config": asdict(anchor_config),
        "aggregation_config": asdict(aggregation_config),
        "anchor_count": int(len(labels)),
        "anchor_labels": labels,
        "normalized_anchor_rows": int(len(normalized_anchors)),
        "candidate_rows": int(len(rows)),
        "frame_count": int(len(frame_rows)),
        "matched_frame_count": matched_frames,
        "matched_frame_fraction": (
            float(matched_frames / len(frame_rows)) if len(frame_rows) else 0.0
        ),
        "mean_matched_anchor_count": float(
            rows["mixture_multi_anchor_matched_count"].mean()
        )
        if len(rows)
        else 0.0,
        "aggregate_cost_mean": float(aggregate_cost.mean()) if len(rows) else None,
        "best_anchor_distance_mean_m": (
            float(best_distance.mean()) if len(best_distance) else None
        ),
        "best_anchor_distance_p95_m": (
            float(best_distance.quantile(0.95)) if len(best_distance) else None
        ),
        "best_anchor_candidate_counts": best_counts,
        "per_anchor": anchor_summaries,
    }


def _validate_aggregation_config(config: MultiAnchorAggregationConfig) -> None:
    if config.aggregation not in ANCHOR_AGGREGATION_CHOICES:
        raise ValueError(f"unsupported anchor aggregation {config.aggregation!r}")
    temperature = float(config.softmin_temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("softmin_temperature must be finite and positive")


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
