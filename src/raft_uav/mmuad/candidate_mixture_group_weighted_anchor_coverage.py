"""Reliability-prioritized multi-anchor coverage for MMUAD mixture-MAP.

Posterior-mass physical-group selection can still omit an anchor-supported mode
when the finite rescue budget is smaller than the number of plausible modes.
The existing bounded coverage rescue processes anchors in column order, so a
weak anchor can consume the budget before a stronger anchor is considered.

This module combines reliability-weighted anchor-cost quantile selection with a
global, framewise rescue proposal ranking. Rescue groups are merged across
anchors and prioritized by train-selectable anchor reliability and geometric
agreement before the unchanged grouped learned-sigma / Huber mixture-MAP runs.
Ground truth is never used for selection or rescue.
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
)
from raft_uav.mmuad.candidate_mixture_group_multi_anchor_mass_topk import (
    _jsonable,
    _load_anchor_specs,
    _mixture_config_from_args,
    _selection_config_from_args,
    _unique_anchor_slugs,
)
from raft_uav.mmuad.candidate_mixture_group_weighted_anchor_quantile import (
    WEIGHTED_QUANTILE_UTILITY_COLUMN,
    WeightedAnchorQuantileConfig,
    _build_parser as _build_quantile_parser,
    select_weighted_quantile_posterior_mass_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_group_weighted_multi_anchor_mass_topk import (
    _parse_anchor_reliability_specs,
    _resolve_anchor_weights,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import (
    GroupedCandidateMixtureMapResult,
    HypothesisGroupConfig,
    prepare_hypothesis_group_candidates,
    run_grouped_candidate_mixture_map,
    write_grouped_candidate_mixture_outputs,
)
from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns

PRIORITY_MODES = ("distance", "reliability", "reliability-distance")
PRIORITY_INPUT_ROW = "mixture_weighted_anchor_coverage_input_row"
PRIORITY_RESCUED = "mixture_weighted_anchor_coverage_rescued"
PRIORITY_RESCUE_ANCHORS = "mixture_weighted_anchor_coverage_rescue_anchors"
PRIORITY_RESCUE_DISTANCE = "mixture_weighted_anchor_coverage_rescue_distance_m"
PRIORITY_RESCUE_WEIGHT = "mixture_weighted_anchor_coverage_rescue_weight"
PRIORITY_RESCUE_SCORE = "mixture_weighted_anchor_coverage_rescue_priority_score"
PRIORITY_RESCUE_RANK = "mixture_weighted_anchor_coverage_rescue_priority_rank"


@dataclass(frozen=True)
class ReliabilityPriorityCoverageConfig:
    """Bounded global rescue configuration for reliability-weighted anchors."""

    enabled: bool = True
    max_anchor_distance_m: float = 25.0
    max_extra_groups_per_frame: int = 2
    max_siblings_per_rescued_group: int = 1
    priority_mode: str = "reliability-distance"
    distance_scale_m: float = 10.0


@dataclass(frozen=True)
class ReliabilityPriorityCoverageResult:
    """Candidate diagnostics, priority coverage rows, and grouped MAP output."""

    scored_candidates: pd.DataFrame
    selected_candidates: pd.DataFrame
    normalized_anchors: pd.DataFrame
    coverage_frames: pd.DataFrame
    grouped_result: GroupedCandidateMixtureMapResult
    selection_summary: dict[str, Any]


def select_reliability_priority_anchor_coverage(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    quantile_config: WeightedAnchorQuantileConfig | None = None,
    coverage_config: ReliabilityPriorityCoverageConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select physical groups, then globally prioritize missing anchor modes."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    anchor_config = anchor_config or AnchorConditioningConfig()
    quantile_config = quantile_config or WeightedAnchorQuantileConfig()
    coverage_config = coverage_config or ReliabilityPriorityCoverageConfig()
    _validate_coverage_config(coverage_config)

    labels = [str(label).strip() for label in anchor_estimates]
    weights = _resolve_anchor_weights(
        labels,
        anchor_reliability=anchor_reliability,
        default_weight=quantile_config.default_weight,
    )
    slugs = _unique_anchor_slugs(labels)
    label_by_slug = dict(zip(slugs, labels, strict=True))

    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy()).reset_index(drop=True)
    rows[PRIORITY_INPUT_ROW] = np.arange(len(rows), dtype=int)
    scored, selected, normalized_anchors, base_summary = (
        select_weighted_quantile_posterior_mass_hypothesis_group_topk(
            rows,
            anchor_estimates=anchor_estimates,
            anchor_reliability=weights,
            mixture_config=mixture_config,
            group_config=group_config,
            selection_config=selection_config,
            anchor_config=anchor_config,
            quantile_config=quantile_config,
        )
    )
    scored = scored.copy().reset_index(drop=True)
    selected = selected.copy().reset_index(drop=True)
    selected = _initialize_rescue_columns(selected)

    if (
        scored.empty
        or not coverage_config.enabled
        or int(coverage_config.max_extra_groups_per_frame) == 0
        or not labels
    ):
        coverage_frames = _empty_coverage_frames()
        summary = _build_summary(
            base_summary,
            coverage_config=coverage_config,
            anchor_weights=weights,
            selected_before=selected,
            selected_after=selected,
            coverage_frames=coverage_frames,
        )
        return scored, selected, normalized_anchors, coverage_frames, summary

    prepared, _, grouping_summary = prepare_hypothesis_group_candidates(
        scored,
        mixture_config=mixture_config,
        group_config=group_config,
    )
    group_by_input = prepared.set_index(PRIORITY_INPUT_ROW)["mixture_hypothesis_group"]
    selected["mixture_hypothesis_group"] = selected[PRIORITY_INPUT_ROW].map(group_by_input)
    selected_before = selected.copy()

    rescued_parts: list[pd.DataFrame] = []
    frame_records: list[dict[str, Any]] = []
    max_distance = float(coverage_config.max_anchor_distance_m)
    max_extra = int(coverage_config.max_extra_groups_per_frame)

    for (sequence_id, time_s), frame in prepared.groupby(
        ["sequence_id", "time_s"],
        sort=True,
        dropna=False,
    ):
        selected_mask = (
            selected["sequence_id"].astype(str).eq(str(sequence_id))
            & pd.to_numeric(selected["time_s"], errors="coerce").eq(float(time_s))
        )
        selected_frame = selected.loc[selected_mask]
        selected_groups = set(
            selected_frame["mixture_hypothesis_group"].dropna().astype(str).tolist()
        )
        selected_ids = set(
            pd.to_numeric(selected_frame[PRIORITY_INPUT_ROW], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

        proposals: dict[str, dict[str, Any]] = {}
        covered_before = 0
        covered_weight_before = 0.0
        unsupported = 0
        zero_weight = 0

        for slug in slugs:
            label = label_by_slug[slug]
            reliability = float(weights[label])
            if reliability <= 0.0:
                zero_weight += 1
                continue
            distance_column = f"mixture_multi_anchor_{slug}_distance_m"
            matched_column = f"mixture_multi_anchor_{slug}_matched"
            if distance_column not in frame.columns:
                unsupported += 1
                continue
            distance = pd.to_numeric(frame[distance_column], errors="coerce")
            eligible = np.isfinite(distance.to_numpy(float)) & (distance <= max_distance)
            if matched_column in frame.columns:
                eligible &= frame[matched_column].fillna(False).astype(bool).to_numpy()
            anchor_rows = frame.loc[eligible].copy()
            if anchor_rows.empty:
                unsupported += 1
                continue
            anchor_rows["_priority_anchor_distance_m"] = pd.to_numeric(
                anchor_rows[distance_column],
                errors="coerce",
            )
            anchor_rows = anchor_rows.sort_values(
                [
                    "_priority_anchor_distance_m",
                    WEIGHTED_QUANTILE_UTILITY_COLUMN,
                    "mixture_group_input_row",
                ],
                ascending=[True, False, True],
                kind="mergesort",
            )
            best = anchor_rows.iloc[0]
            group_value = str(best["mixture_hypothesis_group"])
            best_distance = float(best["_priority_anchor_distance_m"])
            best_utility = float(best[WEIGHTED_QUANTILE_UTILITY_COLUMN])
            if group_value in selected_groups:
                covered_before += 1
                covered_weight_before += reliability
                continue
            proposal = proposals.setdefault(
                group_value,
                {
                    "group": group_value,
                    "anchors": [],
                    "distances": [],
                    "weights": [],
                    "best_utility": best_utility,
                },
            )
            proposal["anchors"].append(label)
            proposal["distances"].append(best_distance)
            proposal["weights"].append(reliability)
            proposal["best_utility"] = max(float(proposal["best_utility"]), best_utility)

        ranked_proposals = _rank_proposals(
            proposals.values(),
            config=coverage_config,
        )
        selected_proposals = ranked_proposals[:max_extra]
        blocked_proposals = ranked_proposals[max_extra:]
        covered_by_rescue = sum(len(proposal["anchors"]) for proposal in selected_proposals)
        covered_weight_by_rescue = sum(
            float(proposal["support_weight"]) for proposal in selected_proposals
        )
        blocked_by_budget = sum(len(proposal["anchors"]) for proposal in blocked_proposals)
        blocked_weight = sum(float(proposal["support_weight"]) for proposal in blocked_proposals)

        for priority_rank, proposal in enumerate(selected_proposals, start=1):
            group_value = str(proposal["group"])
            siblings = frame.loc[
                frame["mixture_hypothesis_group"].astype(str).eq(group_value)
            ].copy()
            siblings = siblings.sort_values(
                [WEIGHTED_QUANTILE_UTILITY_COLUMN, "mixture_group_input_row"],
                ascending=[False, True],
                kind="mergesort",
            ).head(int(coverage_config.max_siblings_per_rescued_group))
            rescue_ids = (
                pd.to_numeric(siblings[PRIORITY_INPUT_ROW], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            rescue_ids = [row_id for row_id in rescue_ids if row_id not in selected_ids]
            if not rescue_ids:
                continue
            rescued = scored.loc[
                scored[PRIORITY_INPUT_ROW].astype(int).isin(rescue_ids)
            ].copy()
            rescued["mixture_hypothesis_group"] = rescued[PRIORITY_INPUT_ROW].map(
                group_by_input
            )
            rescued[PRIORITY_RESCUED] = True
            rescued[PRIORITY_RESCUE_ANCHORS] = ";".join(sorted(set(proposal["anchors"])))
            rescued[PRIORITY_RESCUE_DISTANCE] = float(proposal["min_distance_m"])
            rescued[PRIORITY_RESCUE_WEIGHT] = float(proposal["support_weight"])
            rescued[PRIORITY_RESCUE_SCORE] = float(proposal["priority_score"])
            rescued[PRIORITY_RESCUE_RANK] = int(priority_rank)
            rescued_parts.append(rescued)
            selected_ids.update(rescue_ids)

        frame_records.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "anchor_count": int(len(labels)),
                "positive_weight_anchor_count": int(sum(weights[label] > 0.0 for label in labels)),
                "selected_groups_before": int(
                    len(selected_frame["mixture_hypothesis_group"].dropna().unique())
                ),
                "proposed_groups": int(len(ranked_proposals)),
                "rescued_groups": int(len(selected_proposals)),
                "covered_anchors_before": int(covered_before),
                "covered_anchors_by_rescue": int(covered_by_rescue),
                "unsupported_anchors": int(unsupported),
                "zero_weight_anchors": int(zero_weight),
                "anchors_blocked_by_budget": int(blocked_by_budget),
                "covered_weight_before": float(covered_weight_before),
                "covered_weight_by_rescue": float(covered_weight_by_rescue),
                "weight_blocked_by_budget": float(blocked_weight),
            }
        )

    selected_after = pd.concat([selected, *rescued_parts], ignore_index=True, sort=False)
    selected_after = selected_after.drop_duplicates(subset=[PRIORITY_INPUT_ROW], keep="first")
    selected_after = selected_after.sort_values(
        ["sequence_id", "time_s", PRIORITY_RESCUED, PRIORITY_INPUT_ROW],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    coverage_frames = pd.DataFrame.from_records(frame_records)
    summary = _build_summary(
        base_summary,
        coverage_config=coverage_config,
        anchor_weights=weights,
        selected_before=selected_before,
        selected_after=selected_after,
        coverage_frames=coverage_frames,
    )
    summary["coverage_grouping"] = grouping_summary
    return scored, selected_after, normalized_anchors, coverage_frames, summary


def run_reliability_priority_anchor_coverage(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    anchor_reliability: Mapping[str, float] | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    quantile_config: WeightedAnchorQuantileConfig | None = None,
    coverage_config: ReliabilityPriorityCoverageConfig | None = None,
    final_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> ReliabilityPriorityCoverageResult:
    """Run priority coverage followed by unchanged grouped robust mixture-MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    group_config = group_config or HypothesisGroupConfig()
    scored, selected, anchors, coverage_frames, summary = (
        select_reliability_priority_anchor_coverage(
            candidates,
            anchor_estimates=anchor_estimates,
            anchor_reliability=anchor_reliability,
            mixture_config=mixture_config,
            group_config=group_config,
            selection_config=selection_config,
            anchor_config=anchor_config,
            quantile_config=quantile_config,
            coverage_config=coverage_config,
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
    return ReliabilityPriorityCoverageResult(
        scored_candidates=scored,
        selected_candidates=selected,
        normalized_anchors=anchors,
        coverage_frames=coverage_frames,
        grouped_result=grouped,
        selection_summary=_jsonable(summary),
    )


def write_reliability_priority_anchor_coverage_outputs(
    result: ReliabilityPriorityCoverageResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write priority-coverage diagnostics and grouped mixture artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored_path = output / "mmuad_weighted_anchor_coverage_scored_candidates.csv"
    selected_path = output / "mmuad_weighted_anchor_coverage_selected_candidates.csv"
    anchors_path = output / "mmuad_weighted_anchor_coverage_normalized_anchors.csv"
    frames_path = output / "mmuad_weighted_anchor_coverage_frames.csv"
    summary_path = output / "mmuad_weighted_anchor_coverage_summary.json"
    result.scored_candidates.to_csv(scored_path, index=False)
    result.selected_candidates.to_csv(selected_path, index=False)
    result.normalized_anchors.to_csv(anchors_path, index=False)
    result.coverage_frames.to_csv(frames_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2),
        encoding="utf-8",
    )
    paths = write_grouped_candidate_mixture_outputs(result.grouped_result, output)
    paths["weighted_anchor_coverage_scored_candidates_csv"] = scored_path
    paths["weighted_anchor_coverage_selected_candidates_csv"] = selected_path
    paths["weighted_anchor_coverage_normalized_anchors_csv"] = anchors_path
    paths["weighted_anchor_coverage_frames_csv"] = frames_path
    paths["weighted_anchor_coverage_summary_json"] = summary_path
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
    result = run_reliability_priority_anchor_coverage(
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
        coverage_config=ReliabilityPriorityCoverageConfig(
            enabled=not args.disable_reliability_priority_coverage,
            max_anchor_distance_m=args.anchor_coverage_max_distance_m,
            max_extra_groups_per_frame=args.anchor_coverage_max_extra_groups_per_frame,
            max_siblings_per_rescued_group=(
                args.anchor_coverage_max_siblings_per_rescued_group
            ),
            priority_mode=args.anchor_coverage_priority_mode,
            distance_scale_m=args.anchor_coverage_distance_scale_m,
        ),
        final_initial_estimates=final_initial,
        truth=truth,
    )
    paths = write_reliability_priority_anchor_coverage_outputs(result, args.output_dir)
    print("mmuad_weighted_anchor_priority_coverage=ok")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    rescued = result.selected_candidates[PRIORITY_RESCUED].fillna(False).astype(bool)
    print(f"rescued_candidate_rows={int(rescued.sum())}")
    pooled = result.grouped_result.mixture_result.summary.get("metrics", {}).get(
        "pooled", {}
    )
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = _build_quantile_parser()
    parser.prog = (
        "python -m "
        "raft_uav.mmuad.candidate_mixture_group_weighted_anchor_coverage"
    )
    parser.description = (
        "rescue MMUAD physical groups with reliability-prioritized multi-anchor coverage"
    )
    parser.add_argument("--disable-reliability-priority-coverage", action="store_true")
    parser.add_argument("--anchor-coverage-max-distance-m", type=float, default=25.0)
    parser.add_argument(
        "--anchor-coverage-max-extra-groups-per-frame",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--anchor-coverage-max-siblings-per-rescued-group",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--anchor-coverage-priority-mode",
        choices=PRIORITY_MODES,
        default="reliability-distance",
    )
    parser.add_argument("--anchor-coverage-distance-scale-m", type=float, default=10.0)
    return parser


def _rank_proposals(
    proposals: Any,
    *,
    config: ReliabilityPriorityCoverageConfig,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    distance_scale = float(config.distance_scale_m)
    for proposal_value in proposals:
        proposal = dict(proposal_value)
        distances = np.asarray(proposal["distances"], dtype=float)
        weights = np.asarray(proposal["weights"], dtype=float)
        support_weight = float(weights.sum())
        min_distance = float(np.min(distances))
        if config.priority_mode == "distance":
            priority_score = float(1.0 / (1.0 + min_distance / distance_scale))
        elif config.priority_mode == "reliability":
            priority_score = support_weight
        else:
            priority_score = float(np.sum(weights / (1.0 + distances / distance_scale)))
        proposal["support_weight"] = support_weight
        proposal["min_distance_m"] = min_distance
        proposal["priority_score"] = priority_score
        ranked.append(proposal)
    return sorted(
        ranked,
        key=lambda proposal: (
            -float(proposal["priority_score"]),
            float(proposal["min_distance_m"]),
            -float(proposal["best_utility"]),
            str(proposal["group"]),
        ),
    )


def _initialize_rescue_columns(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out[PRIORITY_RESCUED] = False
    out[PRIORITY_RESCUE_ANCHORS] = ""
    out[PRIORITY_RESCUE_DISTANCE] = np.nan
    out[PRIORITY_RESCUE_WEIGHT] = np.nan
    out[PRIORITY_RESCUE_SCORE] = np.nan
    out[PRIORITY_RESCUE_RANK] = np.nan
    return out


def _validate_coverage_config(config: ReliabilityPriorityCoverageConfig) -> None:
    distance = float(config.max_anchor_distance_m)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("max_anchor_distance_m must be finite and non-negative")
    if int(config.max_extra_groups_per_frame) < 0:
        raise ValueError("max_extra_groups_per_frame must be non-negative")
    if int(config.max_siblings_per_rescued_group) <= 0:
        raise ValueError("max_siblings_per_rescued_group must be positive")
    if config.priority_mode not in PRIORITY_MODES:
        raise ValueError(f"unsupported coverage priority mode {config.priority_mode!r}")
    distance_scale = float(config.distance_scale_m)
    if not np.isfinite(distance_scale) or distance_scale <= 0.0:
        raise ValueError("distance_scale_m must be finite and positive")


def _empty_coverage_frames() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sequence_id",
            "time_s",
            "anchor_count",
            "positive_weight_anchor_count",
            "selected_groups_before",
            "proposed_groups",
            "rescued_groups",
            "covered_anchors_before",
            "covered_anchors_by_rescue",
            "unsupported_anchors",
            "zero_weight_anchors",
            "anchors_blocked_by_budget",
            "covered_weight_before",
            "covered_weight_by_rescue",
            "weight_blocked_by_budget",
        ]
    )


def _build_summary(
    base_summary: dict[str, Any],
    *,
    coverage_config: ReliabilityPriorityCoverageConfig,
    anchor_weights: Mapping[str, float],
    selected_before: pd.DataFrame,
    selected_after: pd.DataFrame,
    coverage_frames: pd.DataFrame,
) -> dict[str, Any]:
    rescued = (
        selected_after[PRIORITY_RESCUED].fillna(False).astype(bool)
        if PRIORITY_RESCUED in selected_after.columns
        else pd.Series(False, index=selected_after.index)
    )
    rescued_group_count = 0
    if rescued.any() and "mixture_hypothesis_group" in selected_after.columns:
        rescued_group_count = int(
            selected_after.loc[rescued]
            .groupby(["sequence_id", "time_s"])["mixture_hypothesis_group"]
            .nunique()
            .sum()
        )
    summary = dict(base_summary)
    summary["schema"] = "raft-uav-mmuad-reliability-priority-anchor-coverage-v1"
    summary["coverage_config"] = asdict(coverage_config)
    summary["anchor_reliability"] = {
        str(label): float(weight) for label, weight in anchor_weights.items()
    }
    summary["selected_candidate_rows_before_coverage"] = int(len(selected_before))
    summary["selected_candidate_rows_after_coverage"] = int(len(selected_after))
    summary["rescued_candidate_rows"] = int(rescued.sum())
    summary["rescued_group_count"] = rescued_group_count
    summary["coverage_frame_count"] = int(len(coverage_frames))
    for column in (
        "covered_anchors_before",
        "covered_anchors_by_rescue",
        "unsupported_anchors",
        "zero_weight_anchors",
        "anchors_blocked_by_budget",
    ):
        summary[f"total_{column}"] = _sum_column(coverage_frames, column)
    for column in (
        "covered_weight_before",
        "covered_weight_by_rescue",
        "weight_blocked_by_budget",
    ):
        summary[f"total_{column}"] = _sum_float_column(coverage_frames, column)
    summary["truth_used_for_coverage"] = False
    return _jsonable(summary)


def _sum_column(rows: pd.DataFrame, column: str) -> int:
    if column not in rows.columns or rows.empty:
        return 0
    return int(pd.to_numeric(rows[column], errors="coerce").fillna(0).sum())


def _sum_float_column(rows: pd.DataFrame, column: str) -> float:
    if column not in rows.columns or rows.empty:
        return 0.0
    return float(pd.to_numeric(rows[column], errors="coerce").fillna(0.0).sum())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
