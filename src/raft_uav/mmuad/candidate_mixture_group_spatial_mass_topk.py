"""Posterior-mass-correct spatial group selection for MMUAD mixture-MAP.

The posterior-mass group selector first computes a budget from groups sorted by
posterior probability and then lets the spatial-diversity selector choose that
many groups.  With non-zero diversity, the selected prefix can differ from the
posterior-sorted prefix and retain substantially less mass than requested.

This module reverses those two operations: it obtains the spatially diverse
ordering up to ``max_group_top_k`` and chooses the smallest prefix of *that
actual ordering* whose posterior mass reaches the target.  The result preserves
spatial diversity while making the posterior-mass target operational rather
than merely diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
    _entropy,
    _jsonable,
    _softmax_probabilities,
    _validate_selection_config,
)
from raft_uav.mmuad.candidate_mixture_group_spatial_topk import (
    SpatialHypothesisGroupTopKConfig,
    _build_group_table,
    _candidate_unary_utility,
    select_spatial_hypothesis_group_topk,
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
    prepare_hypothesis_group_candidates,
    run_grouped_candidate_mixture_map,
    write_grouped_candidate_mixture_outputs,
)
from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns


@dataclass(frozen=True)
class SpatialPosteriorMassGroupTopKCandidateMixtureResult:
    """Selected candidates, grouped mixture result, and selection diagnostics."""

    selected_candidates: pd.DataFrame
    grouped_result: GroupedCandidateMixtureMapResult
    selection_summary: dict[str, Any]


def select_spatial_posterior_mass_hypothesis_group_topk(
    candidates: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select the smallest spatially ordered prefix that reaches target mass."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    _validate_selection_config(selection_config)

    original = normalize_candidate_columns(pd.DataFrame(candidates).copy()).reset_index(
        drop=True
    )
    enabled = int(selection_config.max_group_top_k) > 0
    if original.empty or not enabled:
        selected = original.copy()
        selected["mixture_spatial_mass_group_topk_selected"] = False
        return selected, _selection_summary(
            original,
            selected,
            selection_config=selection_config,
            enabled=enabled,
            frame_summaries=pd.DataFrame(),
        )

    prepared, _, grouping_summary = prepare_hypothesis_group_candidates(
        original,
        mixture_config=mixture_config,
        group_config=group_config,
    )
    prepared = prepared.copy()
    prepared["mixture_spatial_group_candidate_utility"] = _candidate_unary_utility(
        prepared,
        mixture_config=mixture_config,
    )

    selected_frames: list[pd.DataFrame] = []
    frame_records: list[dict[str, Any]] = []
    for (sequence_id, time_s), prepared_frame in prepared.groupby(
        ["sequence_id", "time_s"], sort=True, dropna=False
    ):
        groups = _build_group_table(
            prepared_frame,
            score_mode=selection_config.group_score_mode,
        )
        input_rows = pd.to_numeric(
            prepared_frame["mixture_group_input_row"], errors="raise"
        ).astype(int)
        original_frame = original.iloc[input_rows.to_numpy()].copy().reset_index(drop=True)

        # Ask the spatial selector for its complete bounded ordering.  We then
        # choose a prefix by the posterior mass of the groups that it actually
        # selected, rather than by the unattainable score-sorted ideal prefix.
        spatial_config = SpatialHypothesisGroupTopKConfig(
            group_top_k=min(int(selection_config.max_group_top_k), int(len(groups))),
            max_siblings_per_group=int(selection_config.max_siblings_per_group),
            group_score_mode=str(selection_config.group_score_mode),
            diversity_weight=float(selection_config.diversity_weight),
            diversity_scale_m=float(selection_config.diversity_scale_m),
            diversity_cap_m=float(selection_config.diversity_cap_m),
        )
        spatial_ordered, _ = select_spatial_hypothesis_group_topk(
            original_frame,
            mixture_config=mixture_config,
            group_config=group_config,
            selection_config=spatial_config,
        )
        ordered_group_ids = _ordered_unique_groups(spatial_ordered)
        budget = _spatial_order_mass_budget(
            groups,
            ordered_group_ids=ordered_group_ids,
            selection_config=selection_config,
        )
        selected_frame = spatial_ordered.loc[
            pd.to_numeric(
                spatial_ordered["mixture_spatial_group_rank"], errors="coerce"
            )
            <= int(budget["selected_group_budget"])
        ].copy()

        diagnostics = {
            "mixture_spatial_mass_group_topk_selected": True,
            "mixture_spatial_mass_group_budget": int(budget["selected_group_budget"]),
            "mixture_spatial_mass_group_ideal_budget": int(
                budget["ideal_selected_group_budget"]
            ),
            "mixture_spatial_mass_group_budget_expansion": int(
                budget["budget_expansion_vs_score_order"]
            ),
            "mixture_spatial_mass_group_available_groups": int(
                budget["available_groups"]
            ),
            "mixture_spatial_mass_group_target_posterior_mass": float(
                selection_config.target_posterior_mass
            ),
            "mixture_spatial_mass_group_retained_posterior_mass": float(
                budget["retained_posterior_mass"]
            ),
            "mixture_spatial_mass_group_ideal_retained_posterior_mass": float(
                budget["ideal_retained_posterior_mass"]
            ),
            "mixture_spatial_mass_group_target_reached": bool(
                budget["target_posterior_mass_reached"]
            ),
            "mixture_spatial_mass_group_posterior_shortfall": float(
                budget["posterior_mass_shortfall"]
            ),
            "mixture_spatial_mass_group_top1_posterior": float(
                budget["top1_posterior"]
            ),
            "mixture_spatial_mass_group_normalized_entropy": float(
                budget["normalized_entropy"]
            ),
            "mixture_spatial_mass_group_effective_count": float(
                budget["effective_count"]
            ),
        }
        for column, value in diagnostics.items():
            selected_frame[column] = value
        selected_frames.append(selected_frame)
        frame_records.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "input_rows": int(len(prepared_frame)),
                "selected_rows": int(len(selected_frame)),
                **budget,
            }
        )

    selected = pd.concat(selected_frames, ignore_index=True).sort_values(
        [
            "sequence_id",
            "time_s",
            "mixture_spatial_group_rank",
            "mixture_spatial_group_sibling_rank",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    summary = _selection_summary(
        original,
        selected,
        selection_config=selection_config,
        enabled=True,
        frame_summaries=pd.DataFrame.from_records(frame_records),
    )
    summary["hypothesis_grouping"] = grouping_summary
    return selected, summary


def run_spatial_posterior_mass_group_topk_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> SpatialPosteriorMassGroupTopKCandidateMixtureResult:
    """Run mass-correct spatial selection and grouped robust mixture-MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    selected, summary = select_spatial_posterior_mass_hypothesis_group_topk(
        candidates,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
    )
    effective_config = mixture_config
    if int(selection_config.max_group_top_k) > 0:
        effective_config = replace(mixture_config, top_k=0)
    grouped = run_grouped_candidate_mixture_map(
        selected,
        mixture_config=effective_config,
        group_config=group_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    return SpatialPosteriorMassGroupTopKCandidateMixtureResult(
        selected_candidates=selected,
        grouped_result=grouped,
        selection_summary=summary,
    )


def write_spatial_posterior_mass_group_topk_outputs(
    result: SpatialPosteriorMassGroupTopKCandidateMixtureResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write selected candidates, corrected mass diagnostics, and mixture outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "mmuad_spatial_posterior_mass_group_topk_candidates.csv"
    summary_path = output / "mmuad_spatial_posterior_mass_group_topk_summary.json"
    result.selected_candidates.to_csv(selected_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2), encoding="utf-8"
    )
    paths = write_grouped_candidate_mixture_outputs(result.grouped_result, output)
    paths["spatial_posterior_mass_group_topk_candidates_csv"] = selected_path
    paths["spatial_posterior_mass_group_topk_summary_json"] = summary_path
    return paths


def _ordered_unique_groups(selected: pd.DataFrame) -> list[str]:
    if selected.empty:
        return []
    ordered = selected.sort_values(
        ["mixture_spatial_group_rank", "mixture_spatial_group_sibling_rank"],
        kind="mergesort",
    )
    return ordered["mixture_hypothesis_group"].astype(str).drop_duplicates().tolist()


def _spatial_order_mass_budget(
    groups: pd.DataFrame,
    *,
    ordered_group_ids: list[str],
    selection_config: PosteriorMassGroupTopKConfig,
) -> dict[str, Any]:
    """Choose a budget from posterior mass along an actual spatial ordering."""

    count = int(len(groups))
    if count == 0:
        return {
            "available_groups": 0,
            "selected_group_budget": 0,
            "ideal_selected_group_budget": 0,
            "budget_expansion_vs_score_order": 0,
            "retained_posterior_mass": 0.0,
            "ideal_retained_posterior_mass": 0.0,
            "target_posterior_mass_reached": False,
            "posterior_mass_shortfall": float(selection_config.target_posterior_mass),
            "top1_posterior": float("nan"),
            "normalized_entropy": float("nan"),
            "effective_count": 0.0,
        }

    group_ids = groups["mixture_hypothesis_group"].astype(str).tolist()
    logits = pd.to_numeric(
        groups["mixture_spatial_group_score"], errors="coerce"
    ).to_numpy(float)
    probabilities = _softmax_probabilities(
        logits,
        temperature=selection_config.posterior_temperature,
    )
    blend = float(selection_config.uniform_posterior_blend)
    probabilities = (1.0 - blend) * probabilities + blend / count
    probability_by_group = dict(zip(group_ids, probabilities, strict=True))

    lower = min(int(selection_config.min_group_top_k), count)
    upper = min(int(selection_config.max_group_top_k), count)
    target = float(selection_config.target_posterior_mass)

    sorted_probabilities = np.sort(probabilities)[::-1]
    ideal_cumulative = np.cumsum(sorted_probabilities)
    ideal_required = int(np.searchsorted(ideal_cumulative, target, side="left") + 1)
    ideal_budget = min(max(ideal_required, lower), upper)

    ordered_probabilities = np.asarray(
        [probability_by_group.get(str(group_id), 0.0) for group_id in ordered_group_ids[:upper]],
        dtype=float,
    )
    actual_cumulative = np.cumsum(ordered_probabilities)
    if actual_cumulative.size and actual_cumulative[-1] >= target:
        actual_required = int(np.searchsorted(actual_cumulative, target, side="left") + 1)
    else:
        actual_required = upper
    budget = min(max(actual_required, lower), upper)
    retained = float(actual_cumulative[budget - 1]) if budget and actual_cumulative.size else 0.0
    ideal_retained = float(ideal_cumulative[budget - 1]) if budget else 0.0
    entropy = _entropy(probabilities)
    reached = bool(retained + 1.0e-12 >= target)

    return {
        "available_groups": count,
        "selected_group_budget": int(budget),
        "ideal_selected_group_budget": int(ideal_budget),
        "budget_expansion_vs_score_order": int(budget - ideal_budget),
        "retained_posterior_mass": retained,
        "ideal_retained_posterior_mass": ideal_retained,
        "target_posterior_mass_reached": reached,
        "posterior_mass_shortfall": float(max(target - retained, 0.0)),
        "top1_posterior": float(sorted_probabilities[0]),
        "normalized_entropy": float(entropy / np.log(count)) if count > 1 else 0.0,
        "effective_count": float(np.exp(entropy)),
    }


def _selection_summary(
    original: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    selection_config: PosteriorMassGroupTopKConfig,
    enabled: bool,
    frame_summaries: pd.DataFrame,
) -> dict[str, Any]:
    budgets = _numeric_column(frame_summaries, "selected_group_budget")
    ideal_budgets = _numeric_column(frame_summaries, "ideal_selected_group_budget")
    retained = _numeric_column(frame_summaries, "retained_posterior_mass")
    shortfall = _numeric_column(frame_summaries, "posterior_mass_shortfall")
    entropy = _numeric_column(frame_summaries, "normalized_entropy")
    reached = (
        frame_summaries.get("target_posterior_mass_reached", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
    )
    return {
        "schema": "raft-uav-mmuad-spatial-posterior-mass-group-topk-v1",
        "enabled": bool(enabled),
        "config": asdict(selection_config),
        "input_rows": int(len(original)),
        "selected_rows": int(len(selected)),
        "frame_count": int(
            original[["sequence_id", "time_s"]].drop_duplicates().shape[0]
        )
        if not original.empty
        else 0,
        "selected_group_budget_mean": _safe_stat(budgets, "mean"),
        "selected_group_budget_min": _safe_stat(budgets, "min"),
        "selected_group_budget_max": _safe_stat(budgets, "max"),
        "ideal_selected_group_budget_mean": _safe_stat(ideal_budgets, "mean"),
        "budget_expansion_mean": _safe_stat(budgets - ideal_budgets, "mean"),
        "retained_posterior_mass_mean": _safe_stat(retained, "mean"),
        "posterior_mass_shortfall_mean": _safe_stat(shortfall, "mean"),
        "target_posterior_mass_reached_fraction": float(reached.mean())
        if len(reached)
        else float("nan"),
        "normalized_entropy_mean": _safe_stat(entropy, "mean"),
        "frame_summaries": frame_summaries.to_dict(orient="records"),
        "truth_used_for_group_budget": False,
    }


def _numeric_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(rows[column], errors="coerce")
    return values.loc[np.isfinite(values)]


def _safe_stat(values: pd.Series, operation: str) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    if finite.empty:
        return float("nan")
    if operation == "mean":
        return float(finite.mean())
    if operation == "min":
        return float(finite.min())
    if operation == "max":
        return float(finite.max())
    raise ValueError(f"unsupported statistic operation={operation!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mmuad-candidate-mixture-group-spatial-mass-topk",
        description="select a spatial group prefix that actually reaches posterior mass",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--truth-csv", type=Path)
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

    fallback = tuple(args.fallback_score_column) or ("ranker_score", "confidence")
    mixture_config = CandidateMixtureMapConfig(
        top_k=0,
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
    candidates = load_candidate_file(args.candidates_csv).rows
    initial = (
        None
        if args.initial_estimates_csv is None
        else read_estimate_csv(args.initial_estimates_csv)
    )
    truth = (
        None if args.truth_csv is None else load_evaluation_truth_file(args.truth_csv).rows
    )
    result = run_spatial_posterior_mass_group_topk_candidate_mixture_map(
        candidates,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
        initial_estimates=initial,
        truth=truth,
    )
    paths = write_spatial_posterior_mass_group_topk_outputs(result, args.output_dir)
    print("mmuad_spatial_posterior_mass_group_topk=ok")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    print(
        "target_posterior_mass_reached_fraction="
        f"{result.selection_summary.get('target_posterior_mass_reached_fraction')}"
    )
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
