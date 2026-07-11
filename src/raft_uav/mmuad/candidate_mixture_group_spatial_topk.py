"""Select spatially diverse physical hypothesis groups before MMUAD mixture-MAP.

The branch-preserving MMUAD pool can contain different origin groups whose
representatives are almost co-located.  Even after sibling grouping, a pure
score top-K may spend most of its finite group budget on those near-duplicate
hypotheses and exclude a lower-scoring but geometrically distinct candidate.

This module adds a greedy, inference-safe diversity term to group-first top-K.
The first group is selected by score/uncertainty utility.  Later groups maximize

``normalized_group_score + diversity_weight * (1 - exp(-d_min / scale))``

where ``d_min`` is the distance to the nearest already selected group
representative.  Truth is optional and is used only by the downstream metric
reporter.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
class SpatialHypothesisGroupTopKConfig:
    """Configuration for spatially diverse group-first candidate selection."""

    group_top_k: int = 10
    max_siblings_per_group: int = 2
    group_score_mode: str = "max"
    diversity_weight: float = 0.5
    diversity_scale_m: float = 5.0
    diversity_cap_m: float = 30.0


@dataclass(frozen=True)
class SpatialGroupTopKCandidateMixtureResult:
    """Selected candidates, grouped mixture output, and selection diagnostics."""

    selected_candidates: pd.DataFrame
    grouped_result: GroupedCandidateMixtureMapResult
    selection_summary: dict[str, Any]


def select_spatial_hypothesis_group_topk(
    candidates: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: SpatialHypothesisGroupTopKConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select high-utility origin groups while discouraging spatial redundancy."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or SpatialHypothesisGroupTopKConfig()
    _validate_selection_config(selection_config)

    original = normalize_candidate_columns(pd.DataFrame(candidates).copy()).reset_index(
        drop=True
    )
    enabled = int(selection_config.group_top_k) > 0
    if original.empty or not enabled:
        selected = original.copy()
        selected["mixture_spatial_group_topk_selected"] = False
        return selected, _selection_summary(
            original,
            selected,
            selection_config=selection_config,
            enabled=enabled,
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

    selected_records: list[pd.DataFrame] = []
    frame_summaries: list[dict[str, Any]] = []
    group_selection_rows: list[dict[str, Any]] = []
    for (sequence_id, time_s), frame in prepared.groupby(
        ["sequence_id", "time_s"],
        sort=True,
        dropna=False,
    ):
        frame = frame.copy()
        groups = _build_group_table(
            frame,
            score_mode=selection_config.group_score_mode,
        )
        groups = _greedy_spatial_group_selection(
            groups,
            selection_config=selection_config,
        )
        selected_groups = groups.head(int(selection_config.group_top_k)).copy()
        group_diagnostics = selected_groups.set_index(
            "mixture_hypothesis_group"
        ).to_dict(orient="index")

        selected_parts: list[pd.DataFrame] = []
        for group_value in selected_groups["mixture_hypothesis_group"].astype(str):
            siblings = frame.loc[
                frame["mixture_hypothesis_group"].astype(str) == group_value
            ].copy()
            siblings = siblings.sort_values(
                [
                    "mixture_spatial_group_candidate_utility",
                    "mixture_group_input_row",
                ],
                ascending=[False, True],
                kind="mergesort",
            )
            if int(selection_config.max_siblings_per_group) > 0:
                siblings = siblings.head(int(selection_config.max_siblings_per_group))
            siblings["mixture_spatial_group_sibling_rank"] = np.arange(
                1,
                len(siblings) + 1,
                dtype=int,
            )
            diagnostics = group_diagnostics[group_value]
            for column, value in diagnostics.items():
                siblings[column] = value
            selected_parts.append(siblings)

        selected_frame = (
            pd.concat(selected_parts, ignore_index=True)
            if selected_parts
            else frame.iloc[0:0].copy()
        )
        selected_records.append(selected_frame)
        group_selection_rows.extend(selected_groups.to_dict(orient="records"))
        frame_summaries.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "input_rows": int(len(frame)),
                "input_groups": int(
                    frame["mixture_hypothesis_group"].nunique(dropna=False)
                ),
                "selected_rows": int(len(selected_frame)),
                "selected_groups": int(
                    selected_frame["mixture_hypothesis_group"].nunique(
                        dropna=False
                    )
                ),
            }
        )

    selected_prepared = pd.concat(selected_records, ignore_index=True)
    selected_ids = pd.to_numeric(
        selected_prepared["mixture_group_input_row"],
        errors="raise",
    ).astype(int)
    selected = original.iloc[selected_ids.to_numpy()].copy().reset_index(drop=True)
    diagnostic_columns = [
        "mixture_hypothesis_group",
        "mixture_spatial_group_candidate_utility",
        "mixture_spatial_group_score",
        "mixture_spatial_group_score_normalized",
        "mixture_spatial_group_size_before",
        "mixture_spatial_group_representative_x_m",
        "mixture_spatial_group_representative_y_m",
        "mixture_spatial_group_representative_z_m",
        "mixture_spatial_group_min_distance_m",
        "mixture_spatial_group_diversity_term",
        "mixture_spatial_group_selection_utility",
        "mixture_spatial_group_rank",
        "mixture_spatial_group_sibling_rank",
    ]
    for column in diagnostic_columns:
        selected[column] = selected_prepared[column].to_numpy()
    selected["mixture_spatial_group_topk_selected"] = True
    selected["mixture_spatial_group_top_k"] = int(selection_config.group_top_k)
    selected["mixture_spatial_group_max_siblings"] = int(
        selection_config.max_siblings_per_group
    )
    selected["mixture_spatial_group_score_mode"] = str(
        selection_config.group_score_mode
    )
    selected["mixture_spatial_group_diversity_weight"] = float(
        selection_config.diversity_weight
    )
    selected = selected.sort_values(
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
        frame_summaries=pd.DataFrame.from_records(frame_summaries),
        selected_groups=pd.DataFrame.from_records(group_selection_rows),
    )
    summary["hypothesis_grouping"] = grouping_summary
    return selected, summary


def run_spatial_group_topk_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: SpatialHypothesisGroupTopKConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> SpatialGroupTopKCandidateMixtureResult:
    """Run spatial group-first selection followed by grouped mixture-MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or SpatialHypothesisGroupTopKConfig()
    selected, selection_summary = select_spatial_hypothesis_group_topk(
        candidates,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
    )
    effective_mixture_config = mixture_config
    if int(selection_config.group_top_k) > 0:
        effective_mixture_config = replace(mixture_config, top_k=0)
    grouped = run_grouped_candidate_mixture_map(
        selected,
        mixture_config=effective_mixture_config,
        group_config=group_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    return SpatialGroupTopKCandidateMixtureResult(
        selected_candidates=selected,
        grouped_result=grouped,
        selection_summary=selection_summary,
    )


def write_spatial_group_topk_candidate_mixture_outputs(
    result: SpatialGroupTopKCandidateMixtureResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write spatial group-selection and standard grouped-mixture artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "mmuad_spatial_group_topk_candidates.csv"
    summary_path = output / "mmuad_spatial_group_topk_summary.json"
    result.selected_candidates.to_csv(selected_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2),
        encoding="utf-8",
    )
    paths = write_grouped_candidate_mixture_outputs(result.grouped_result, output)
    paths["spatial_group_topk_candidates_csv"] = selected_path
    paths["spatial_group_topk_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-spatial-group-topk",
        description="select spatially diverse MMUAD groups before mixture-MAP",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--group-top-k", type=int, default=10)
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
    parser.add_argument(
        "--hypothesis-group-correction-strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--missing-hypothesis-group-policy",
        choices=("unique", "error"),
        default="unique",
    )
    args = parser.parse_args(argv)

    fallback_columns = tuple(args.fallback_score_column) or (
        "ranker_score",
        "confidence",
    )
    mixture_config = CandidateMixtureMapConfig(
        top_k=args.row_top_k_when_disabled,
        score_column=args.score_column,
        fallback_score_columns=fallback_columns,
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
    selection_config = SpatialHypothesisGroupTopKConfig(
        group_top_k=args.group_top_k,
        max_siblings_per_group=args.max_siblings_per_group,
        group_score_mode=args.group_score_mode,
        diversity_weight=args.diversity_weight,
        diversity_scale_m=args.diversity_scale_m,
        diversity_cap_m=args.diversity_cap_m,
    )
    candidates = load_candidate_file(args.candidates_csv).rows
    initial_estimates = (
        None
        if args.initial_estimates_csv is None
        else read_estimate_csv(args.initial_estimates_csv)
    )
    truth = (
        None
        if args.truth_csv is None
        else load_evaluation_truth_file(args.truth_csv).rows
    )
    result = run_spatial_group_topk_candidate_mixture_map(
        candidates,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    paths = write_spatial_group_topk_candidate_mixture_outputs(
        result,
        args.output_dir,
    )
    print("mmuad_candidate_mixture_spatial_group_topk=ok")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    pooled = result.grouped_result.mixture_result.summary.get("metrics", {}).get(
        "pooled",
        {},
    )
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _build_group_table(frame: pd.DataFrame, *, score_mode: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group_value, siblings in frame.groupby(
        "mixture_hypothesis_group",
        sort=False,
        dropna=False,
    ):
        siblings = siblings.sort_values(
            ["mixture_spatial_group_candidate_utility", "mixture_group_input_row"],
            ascending=[False, True],
            kind="mergesort",
        )
        representative = siblings.iloc[0]
        utilities = siblings["mixture_spatial_group_candidate_utility"].to_numpy(float)
        records.append(
            {
                "mixture_hypothesis_group": str(group_value),
                "mixture_spatial_group_score": _aggregate_group_score(
                    utilities,
                    mode=score_mode,
                ),
                "mixture_spatial_group_size_before": int(len(siblings)),
                "mixture_spatial_group_representative_x_m": float(
                    representative["x_m"]
                ),
                "mixture_spatial_group_representative_y_m": float(
                    representative["y_m"]
                ),
                "mixture_spatial_group_representative_z_m": float(
                    representative["z_m"]
                ),
            }
        )
    groups = pd.DataFrame.from_records(records)
    groups["mixture_spatial_group_score_normalized"] = _minmax_scores(
        groups["mixture_spatial_group_score"].to_numpy(float)
    )
    return groups


def _greedy_spatial_group_selection(
    groups: pd.DataFrame,
    *,
    selection_config: SpatialHypothesisGroupTopKConfig,
) -> pd.DataFrame:
    available = groups.copy().reset_index(drop=True)
    selected_rows: list[dict[str, Any]] = []
    selected_xyz: list[np.ndarray] = []
    target_count = min(int(selection_config.group_top_k), len(available))

    while len(selected_rows) < target_count:
        scored: list[dict[str, Any]] = []
        for index, row in available.iterrows():
            xyz = row[
                [
                    "mixture_spatial_group_representative_x_m",
                    "mixture_spatial_group_representative_y_m",
                    "mixture_spatial_group_representative_z_m",
                ]
            ].to_numpy(float)
            min_distance = _minimum_distance(xyz, selected_xyz)
            diversity_term = _diversity_term(
                min_distance,
                scale_m=selection_config.diversity_scale_m,
                cap_m=selection_config.diversity_cap_m,
            )
            selection_utility = float(
                row["mixture_spatial_group_score_normalized"]
            ) + float(selection_config.diversity_weight) * diversity_term
            scored.append(
                {
                    "_available_index": int(index),
                    "mixture_spatial_group_min_distance_m": min_distance,
                    "mixture_spatial_group_diversity_term": diversity_term,
                    "mixture_spatial_group_selection_utility": selection_utility,
                    "mixture_spatial_group_score": float(
                        row["mixture_spatial_group_score"]
                    ),
                    "mixture_hypothesis_group": str(
                        row["mixture_hypothesis_group"]
                    ),
                }
            )
        ranked = pd.DataFrame.from_records(scored).sort_values(
            [
                "mixture_spatial_group_selection_utility",
                "mixture_spatial_group_score",
                "mixture_hypothesis_group",
            ],
            ascending=[False, False, True],
            kind="mergesort",
        )
        winner = ranked.iloc[0]
        available_index = int(winner["_available_index"])
        selected = available.loc[available_index].to_dict()
        selected["mixture_spatial_group_min_distance_m"] = float(
            winner["mixture_spatial_group_min_distance_m"]
        )
        selected["mixture_spatial_group_diversity_term"] = float(
            winner["mixture_spatial_group_diversity_term"]
        )
        selected["mixture_spatial_group_selection_utility"] = float(
            winner["mixture_spatial_group_selection_utility"]
        )
        selected["mixture_spatial_group_rank"] = len(selected_rows) + 1
        selected_rows.append(selected)
        selected_xyz.append(
            np.asarray(
                [
                    selected["mixture_spatial_group_representative_x_m"],
                    selected["mixture_spatial_group_representative_y_m"],
                    selected["mixture_spatial_group_representative_z_m"],
                ],
                dtype=float,
            )
        )
        available = available.drop(index=available_index)

    return pd.DataFrame.from_records(selected_rows)


def _candidate_unary_utility(
    prepared: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig,
) -> np.ndarray:
    normalized_score = pd.to_numeric(
        prepared["mixture_group_base_normalized_score"],
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    if mixture_config.sigma_column in prepared.columns:
        sigma = pd.to_numeric(
            prepared[mixture_config.sigma_column],
            errors="coerce",
        ).to_numpy(float)
    else:
        sigma = np.full(len(prepared), float(mixture_config.default_sigma_m))
    sigma = np.nan_to_num(
        sigma,
        nan=float(mixture_config.default_sigma_m),
        posinf=float(mixture_config.sigma_max_m),
        neginf=float(mixture_config.default_sigma_m),
    )
    sigma = np.clip(
        sigma,
        float(mixture_config.sigma_min_m),
        float(mixture_config.sigma_max_m),
    )
    temperature = max(float(mixture_config.temperature), 1.0e-12)
    return (
        float(mixture_config.score_weight) * normalized_score / temperature
        - float(mixture_config.sigma_log_weight) * np.log(sigma)
    )


def _aggregate_group_score(values: np.ndarray, *, mode: str) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("-inf")
    if mode == "max":
        return float(np.max(finite))
    if mode == "logmeanexp":
        maximum = float(np.max(finite))
        return float(maximum + np.log(np.mean(np.exp(finite - maximum))))
    raise ValueError(f"unsupported group_score_mode={mode!r}")


def _minmax_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.zeros(len(values), dtype=float)
    if not finite.any():
        return out
    minimum = float(np.min(values[finite]))
    maximum = float(np.max(values[finite]))
    if maximum - minimum <= 1.0e-12:
        out[finite] = 1.0
        return out
    out[finite] = (values[finite] - minimum) / (maximum - minimum)
    return out


def _minimum_distance(xyz: np.ndarray, selected_xyz: list[np.ndarray]) -> float:
    if not selected_xyz:
        return 0.0
    if not np.isfinite(xyz).all():
        return 0.0
    finite_selected = [item for item in selected_xyz if np.isfinite(item).all()]
    if not finite_selected:
        return 0.0
    return float(min(np.linalg.norm(xyz - item) for item in finite_selected))


def _diversity_term(distance_m: float, *, scale_m: float, cap_m: float) -> float:
    distance = max(float(distance_m), 0.0)
    if float(cap_m) > 0.0:
        distance = min(distance, float(cap_m))
    return float(1.0 - np.exp(-distance / float(scale_m)))


def _validate_selection_config(config: SpatialHypothesisGroupTopKConfig) -> None:
    if int(config.group_top_k) < 0:
        raise ValueError("group_top_k must be non-negative")
    if int(config.max_siblings_per_group) < 0:
        raise ValueError("max_siblings_per_group must be non-negative")
    if config.group_score_mode not in GROUP_SCORE_MODES:
        raise ValueError(f"unsupported group_score_mode={config.group_score_mode!r}")
    if float(config.diversity_weight) < 0.0:
        raise ValueError("diversity_weight must be non-negative")
    if float(config.diversity_scale_m) <= 0.0:
        raise ValueError("diversity_scale_m must be positive")
    if float(config.diversity_cap_m) < 0.0:
        raise ValueError("diversity_cap_m must be non-negative")


def _selection_summary(
    original: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    selection_config: SpatialHypothesisGroupTopKConfig,
    enabled: bool,
    frame_summaries: pd.DataFrame | None = None,
    selected_groups: pd.DataFrame | None = None,
) -> dict[str, Any]:
    frame_summaries = frame_summaries if frame_summaries is not None else pd.DataFrame()
    selected_groups = selected_groups if selected_groups is not None else pd.DataFrame()
    distances = pd.to_numeric(
        selected_groups.get("mixture_spatial_group_min_distance_m"),
        errors="coerce",
    )
    ranks = pd.to_numeric(
        selected_groups.get("mixture_spatial_group_rank"),
        errors="coerce",
    )
    if isinstance(distances, pd.Series) and isinstance(ranks, pd.Series):
        distances = distances.loc[(ranks > 1) & np.isfinite(distances)]
    else:
        distances = pd.Series(dtype=float)
    return {
        "schema": "raft-uav-mmuad-spatial-group-topk-v1",
        "enabled": bool(enabled),
        "config": asdict(selection_config),
        "input_candidate_rows": int(len(original)),
        "selected_candidate_rows": int(len(selected)),
        "frame_count": int(len(frame_summaries)),
        "input_groups_per_frame_mean": _column_mean(frame_summaries, "input_groups"),
        "selected_groups_per_frame_mean": _column_mean(
            frame_summaries,
            "selected_groups",
        ),
        "selected_rows_per_frame_mean": _column_mean(
            frame_summaries,
            "selected_rows",
        ),
        "selected_group_min_distance_mean_m": _series_mean(distances),
        "selected_group_min_distance_p50_m": _series_quantile(distances, 0.50),
        "selected_group_min_distance_p95_m": _series_quantile(distances, 0.95),
    }


def _column_mean(rows: pd.DataFrame, column: str) -> float:
    if rows.empty or column not in rows.columns:
        return float("nan")
    return _series_mean(pd.to_numeric(rows[column], errors="coerce"))


def _series_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else float("nan")


def _series_quantile(values: pd.Series, quantile: float) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.quantile(float(quantile))) if len(finite) else float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
