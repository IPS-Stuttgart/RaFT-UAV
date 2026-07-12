"""Guarantee bounded multi-anchor mode coverage before MMUAD mixture-MAP.

Multi-anchor minimum-cost selection protects candidates that agree with any
inference-time trajectory anchor, but a finite posterior-mass group budget can
still retain only one of several plausible anchor-supported modes. This module
adds a bounded post-selection rescue: for every supported anchor and frame, it
keeps the nearest physical hypothesis group when that group is close enough and
was not already selected.

The rescue is inference-safe. It uses only candidate metadata and trajectory
anchors; ground truth remains optional and is used only by downstream metrics.
The final grouped learned-sigma / Huber mixture-MAP objective is unchanged.
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
    MULTI_ANCHOR_UTILITY_COLUMN,
    MultiAnchorAggregationConfig,
    _build_parser as _build_multi_anchor_parser,
    _load_anchor_specs,
    _mixture_config_from_args,
    _selection_config_from_args,
    select_multi_anchor_posterior_mass_hypothesis_group_topk,
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

COVERAGE_INPUT_ROW = "mixture_multi_anchor_coverage_input_row"
COVERAGE_RESCUED = "mixture_multi_anchor_coverage_rescued"
COVERAGE_RESCUE_ANCHORS = "mixture_multi_anchor_coverage_rescue_anchors"
COVERAGE_RESCUE_DISTANCE = "mixture_multi_anchor_coverage_rescue_distance_m"


@dataclass(frozen=True)
class AnchorGroupCoverageConfig:
    """Bounded rescue configuration for anchor-supported physical groups."""

    enabled: bool = True
    max_anchor_distance_m: float = 25.0
    max_extra_groups_per_frame: int = 2
    max_siblings_per_rescued_group: int = 1


@dataclass(frozen=True)
class MultiAnchorCoverageCandidateMixtureResult:
    """Full scoring, coverage selection, diagnostics, and grouped MAP output."""

    scored_candidates: pd.DataFrame
    selected_candidates: pd.DataFrame
    normalized_anchors: pd.DataFrame
    coverage_frames: pd.DataFrame
    grouped_result: GroupedCandidateMixtureMapResult
    selection_summary: dict[str, Any]


def select_multi_anchor_coverage_hypothesis_group_topk(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    aggregation_config: MultiAnchorAggregationConfig | None = None,
    coverage_config: AnchorGroupCoverageConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select posterior-mass groups and rescue missing anchor-supported modes."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    coverage_config = coverage_config or AnchorGroupCoverageConfig()
    _validate_coverage_config(coverage_config)

    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy()).reset_index(drop=True)
    rows[COVERAGE_INPUT_ROW] = np.arange(len(rows), dtype=int)
    scored, selected, normalized_anchors, base_summary = (
        select_multi_anchor_posterior_mass_hypothesis_group_topk(
            rows,
            anchor_estimates=anchor_estimates,
            mixture_config=mixture_config,
            group_config=group_config,
            selection_config=selection_config,
            anchor_config=anchor_config,
            aggregation_config=aggregation_config,
        )
    )

    scored = scored.copy().reset_index(drop=True)
    selected = selected.copy().reset_index(drop=True)
    selected[COVERAGE_RESCUED] = False
    selected[COVERAGE_RESCUE_ANCHORS] = ""
    selected[COVERAGE_RESCUE_DISTANCE] = np.nan

    distance_columns = _anchor_distance_columns(scored)
    if (
        scored.empty
        or not bool(coverage_config.enabled)
        or int(coverage_config.max_extra_groups_per_frame) == 0
        or not distance_columns
    ):
        coverage_frames = _empty_coverage_frames()
        summary = _coverage_summary(
            base_summary,
            coverage_config=coverage_config,
            distance_columns=distance_columns,
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
    group_by_input = prepared.set_index(COVERAGE_INPUT_ROW)["mixture_hypothesis_group"]
    selected["mixture_hypothesis_group"] = selected[COVERAGE_INPUT_ROW].map(group_by_input)

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
            pd.to_numeric(selected_frame[COVERAGE_INPUT_ROW], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

        rescue_groups: dict[str, dict[str, Any]] = {}
        covered_before = 0
        covered_by_rescue = 0
        unsupported = 0
        blocked_by_budget = 0

        for anchor_slug, distance_column in distance_columns:
            matched_column = f"mixture_multi_anchor_{anchor_slug}_matched"
            distance = pd.to_numeric(frame[distance_column], errors="coerce")
            eligible = np.isfinite(distance.to_numpy(float)) & (distance <= max_distance)
            if matched_column in frame.columns:
                eligible &= frame[matched_column].fillna(False).astype(bool).to_numpy()
            anchor_rows = frame.loc[eligible].copy()
            if anchor_rows.empty:
                unsupported += 1
                continue
            anchor_rows["_coverage_anchor_distance_m"] = pd.to_numeric(
                anchor_rows[distance_column], errors="coerce"
            )
            anchor_rows = anchor_rows.sort_values(
                [
                    "_coverage_anchor_distance_m",
                    MULTI_ANCHOR_UTILITY_COLUMN,
                    "mixture_group_input_row",
                ],
                ascending=[True, False, True],
                kind="mergesort",
            )
            best = anchor_rows.iloc[0]
            group_value = str(best["mixture_hypothesis_group"])
            best_distance = float(best["_coverage_anchor_distance_m"])
            if group_value in rescue_groups:
                rescue_groups[group_value]["anchors"].append(anchor_slug)
                rescue_groups[group_value]["distance_m"] = min(
                    float(rescue_groups[group_value]["distance_m"]),
                    best_distance,
                )
                covered_by_rescue += 1
                continue
            if group_value in selected_groups:
                covered_before += 1
                continue
            if len(rescue_groups) >= max_extra:
                blocked_by_budget += 1
                continue
            rescue_groups[group_value] = {
                "anchors": [anchor_slug],
                "distance_m": best_distance,
            }
            selected_groups.add(group_value)
            covered_by_rescue += 1

        for group_value, rescue in rescue_groups.items():
            siblings = frame.loc[
                frame["mixture_hypothesis_group"].astype(str).eq(group_value)
            ].copy()
            siblings = siblings.sort_values(
                [MULTI_ANCHOR_UTILITY_COLUMN, "mixture_group_input_row"],
                ascending=[False, True],
                kind="mergesort",
            ).head(int(coverage_config.max_siblings_per_rescued_group))
            rescue_ids = (
                pd.to_numeric(siblings[COVERAGE_INPUT_ROW], errors="coerce")
                .dropna()
                .astype(int)
            )
            rescue_ids = [row_id for row_id in rescue_ids if row_id not in selected_ids]
            if not rescue_ids:
                continue
            rescued = scored.loc[
                scored[COVERAGE_INPUT_ROW].astype(int).isin(rescue_ids)
            ].copy()
            rescued["mixture_hypothesis_group"] = rescued[COVERAGE_INPUT_ROW].map(
                group_by_input
            )
            rescued[COVERAGE_RESCUED] = True
            rescued[COVERAGE_RESCUE_ANCHORS] = ";".join(sorted(set(rescue["anchors"])))
            rescued[COVERAGE_RESCUE_DISTANCE] = float(rescue["distance_m"])
            rescued_parts.append(rescued)
            selected_ids.update(rescue_ids)

        frame_records.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "anchor_count": int(len(distance_columns)),
                "selected_groups_before": int(len(selected_frame["mixture_hypothesis_group"].dropna().unique())),
                "rescued_groups": int(len(rescue_groups)),
                "covered_anchors_before": int(covered_before),
                "covered_anchors_by_rescue": int(covered_by_rescue),
                "unsupported_anchors": int(unsupported),
                "anchors_blocked_by_budget": int(blocked_by_budget),
            }
        )

    selected_after = pd.concat([selected, *rescued_parts], ignore_index=True, sort=False)
    selected_after = selected_after.drop_duplicates(subset=[COVERAGE_INPUT_ROW], keep="first")
    selected_after = selected_after.sort_values(
        ["sequence_id", "time_s", COVERAGE_RESCUED, COVERAGE_INPUT_ROW],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    coverage_frames = pd.DataFrame.from_records(frame_records)
    summary = _coverage_summary(
        base_summary,
        coverage_config=coverage_config,
        distance_columns=distance_columns,
        selected_before=selected_before,
        selected_after=selected_after,
        coverage_frames=coverage_frames,
    )
    summary["coverage_grouping"] = grouping_summary
    return scored, selected_after, normalized_anchors, coverage_frames, summary


def run_multi_anchor_coverage_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    anchor_estimates: Mapping[str, pd.DataFrame],
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    aggregation_config: MultiAnchorAggregationConfig | None = None,
    coverage_config: AnchorGroupCoverageConfig | None = None,
    final_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> MultiAnchorCoverageCandidateMixtureResult:
    """Run anchor-coverage selection followed by grouped robust mixture-MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    scored, selected, anchors, coverage_frames, summary = (
        select_multi_anchor_coverage_hypothesis_group_topk(
            candidates,
            anchor_estimates=anchor_estimates,
            mixture_config=mixture_config,
            group_config=group_config,
            selection_config=selection_config,
            anchor_config=anchor_config,
            aggregation_config=aggregation_config,
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
    return MultiAnchorCoverageCandidateMixtureResult(
        scored_candidates=scored,
        selected_candidates=selected,
        normalized_anchors=anchors,
        coverage_frames=coverage_frames,
        grouped_result=grouped,
        selection_summary=_jsonable(summary),
    )


def write_multi_anchor_coverage_outputs(
    result: MultiAnchorCoverageCandidateMixtureResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write coverage diagnostics plus the standard grouped mixture artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored_path = output / "mmuad_multi_anchor_coverage_scored_candidates.csv"
    selected_path = output / "mmuad_multi_anchor_coverage_selected_candidates.csv"
    anchors_path = output / "mmuad_multi_anchor_coverage_normalized_anchors.csv"
    frames_path = output / "mmuad_multi_anchor_coverage_frames.csv"
    summary_path = output / "mmuad_multi_anchor_coverage_summary.json"
    result.scored_candidates.to_csv(scored_path, index=False)
    result.selected_candidates.to_csv(selected_path, index=False)
    result.normalized_anchors.to_csv(anchors_path, index=False)
    result.coverage_frames.to_csv(frames_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2),
        encoding="utf-8",
    )
    paths = write_grouped_candidate_mixture_outputs(result.grouped_result, output)
    paths["multi_anchor_coverage_scored_candidates_csv"] = scored_path
    paths["multi_anchor_coverage_selected_candidates_csv"] = selected_path
    paths["multi_anchor_coverage_normalized_anchors_csv"] = anchors_path
    paths["multi_anchor_coverage_frames_csv"] = frames_path
    paths["multi_anchor_coverage_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.anchor_csv:
        parser.error("provide at least one --anchor-csv NAME=PATH")

    candidates = load_candidate_file(args.candidates_csv).rows
    anchors = _load_anchor_specs(args.anchor_csv)
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
    result = run_multi_anchor_coverage_candidate_mixture_map(
        candidates,
        anchor_estimates=anchors,
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
        coverage_config=AnchorGroupCoverageConfig(
            enabled=not args.disable_anchor_group_coverage,
            max_anchor_distance_m=args.anchor_coverage_max_distance_m,
            max_extra_groups_per_frame=args.anchor_coverage_max_extra_groups_per_frame,
            max_siblings_per_rescued_group=(
                args.anchor_coverage_max_siblings_per_rescued_group
            ),
        ),
        final_initial_estimates=final_initial,
        truth=truth,
    )
    paths = write_multi_anchor_coverage_outputs(result, args.output_dir)
    print("mmuad_multi_anchor_coverage_group_topk=ok")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    print(
        "rescued_candidate_rows="
        f"{int(result.selected_candidates[COVERAGE_RESCUED].fillna(False).sum())}"
    )
    pooled = result.grouped_result.mixture_result.summary.get("metrics", {}).get(
        "pooled", {}
    )
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = _build_multi_anchor_parser()
    parser.prog = "python -m raft_uav.mmuad.candidate_mixture_group_multi_anchor_coverage"
    parser.description = (
        "preserve bounded physical-group coverage for every supported MMUAD anchor"
    )
    parser.add_argument("--disable-anchor-group-coverage", action="store_true")
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
    return parser


def _anchor_distance_columns(rows: pd.DataFrame) -> list[tuple[str, str]]:
    prefix = "mixture_multi_anchor_"
    suffix = "_distance_m"
    result: list[tuple[str, str]] = []
    for column in rows.columns:
        if not column.startswith(prefix) or not column.endswith(suffix):
            continue
        slug = column[len(prefix) : -len(suffix)]
        if slug == "best" or not slug:
            continue
        matched_column = f"mixture_multi_anchor_{slug}_matched"
        if matched_column in rows.columns:
            result.append((slug, column))
    return result


def _validate_coverage_config(config: AnchorGroupCoverageConfig) -> None:
    distance = float(config.max_anchor_distance_m)
    if not np.isfinite(distance) or distance < 0.0:
        raise ValueError("max_anchor_distance_m must be finite and non-negative")
    if int(config.max_extra_groups_per_frame) < 0:
        raise ValueError("max_extra_groups_per_frame must be non-negative")
    if int(config.max_siblings_per_rescued_group) <= 0:
        raise ValueError("max_siblings_per_rescued_group must be positive")


def _empty_coverage_frames() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sequence_id",
            "time_s",
            "anchor_count",
            "selected_groups_before",
            "rescued_groups",
            "covered_anchors_before",
            "covered_anchors_by_rescue",
            "unsupported_anchors",
            "anchors_blocked_by_budget",
        ]
    )


def _coverage_summary(
    base_summary: dict[str, Any],
    *,
    coverage_config: AnchorGroupCoverageConfig,
    distance_columns: list[tuple[str, str]],
    selected_before: pd.DataFrame,
    selected_after: pd.DataFrame,
    coverage_frames: pd.DataFrame,
) -> dict[str, Any]:
    rescued = (
        selected_after[COVERAGE_RESCUED].fillna(False).astype(bool)
        if COVERAGE_RESCUED in selected_after.columns
        else pd.Series(False, index=selected_after.index)
    )
    rescue_group_count = 0
    if rescued.any() and "mixture_hypothesis_group" in selected_after.columns:
        rescue_group_count = int(
            selected_after.loc[rescued]
            .groupby(["sequence_id", "time_s"])["mixture_hypothesis_group"]
            .nunique(dropna=False)
            .sum()
        )
    frame_rescue_count = 0
    if not coverage_frames.empty and "rescued_groups" in coverage_frames.columns:
        frame_rescue_count = int((coverage_frames["rescued_groups"] > 0).sum())
    summary = dict(base_summary)
    summary["schema"] = "raft-uav-mmuad-multi-anchor-group-coverage-v1"
    summary["anchor_group_coverage"] = {
        "config": asdict(coverage_config),
        "anchor_slugs": [slug for slug, _ in distance_columns],
        "selected_candidate_rows_before": int(len(selected_before)),
        "selected_candidate_rows_after": int(len(selected_after)),
        "rescued_candidate_rows": int(rescued.sum()),
        "rescued_group_count": rescue_group_count,
        "frames_with_rescue": frame_rescue_count,
        "coverage_frame_count": int(len(coverage_frames)),
        "anchors_blocked_by_budget": int(
            coverage_frames.get("anchors_blocked_by_budget", pd.Series(dtype=int)).sum()
        ),
        "unsupported_anchors": int(
            coverage_frames.get("unsupported_anchors", pd.Series(dtype=int)).sum()
        ),
    }
    summary["truth_used_for_selection"] = False
    return _jsonable(summary)


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
