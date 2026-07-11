"""Select unique physical hypothesis groups before MMUAD mixture-MAP.

Raw and train-calibrated branches can contain several coordinate hypotheses for
one physical point-cloud cluster.  A flat row-level top-K can therefore spend
multiple slots on siblings from the same origin group and hide a geometrically
distinct cluster.  This module ranks origin groups first, keeps a configurable
number of siblings from every selected group, and then runs the maintained
origin-group-corrected robust candidate-mixture smoother.

The selector is inference-safe.  It uses candidate scores, learned uncertainty,
and candidate metadata only.  Optional truth is passed solely to the downstream
metric reporter.
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
    HypothesisGroupConfig,
    GroupedCandidateMixtureMapResult,
    prepare_hypothesis_group_candidates,
    run_grouped_candidate_mixture_map,
    write_grouped_candidate_mixture_outputs,
)
from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns

GROUP_SCORE_MODES = ("max", "logmeanexp")


@dataclass(frozen=True)
class HypothesisGroupTopKConfig:
    """Configuration for group-first candidate selection."""

    group_top_k: int = 10
    max_siblings_per_group: int = 2
    group_score_mode: str = "max"


@dataclass(frozen=True)
class GroupTopKCandidateMixtureResult:
    """Selected candidates, grouped mixture output, and selection diagnostics."""

    selected_candidates: pd.DataFrame
    grouped_result: GroupedCandidateMixtureMapResult
    selection_summary: dict[str, Any]


def select_hypothesis_group_topk(
    candidates: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: HypothesisGroupTopKConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select top origin groups per frame without sibling crowding.

    Group ranking uses the state-independent part of the maintained mixture unary
    term: normalized candidate score minus the learned-sigma log penalty.  The
    default group score is the best sibling utility.  ``logmeanexp`` provides a
    smooth, group-size-neutral alternative.
    """

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or HypothesisGroupTopKConfig()
    _validate_selection_config(selection_config)

    original = normalize_candidate_columns(pd.DataFrame(candidates).copy()).reset_index(drop=True)
    if original.empty:
        return original, _selection_summary(
            original,
            original,
            selection_config=selection_config,
            enabled=int(selection_config.group_top_k) > 0,
        )
    if int(selection_config.group_top_k) <= 0:
        passthrough = original.copy()
        passthrough["mixture_group_topk_selected"] = False
        return passthrough, _selection_summary(
            original,
            passthrough,
            selection_config=selection_config,
            enabled=False,
        )

    prepared, _, grouping_summary = prepare_hypothesis_group_candidates(
        original,
        mixture_config=mixture_config,
        group_config=group_config,
    )
    prepared = prepared.copy()
    prepared["mixture_group_topk_candidate_utility"] = _candidate_unary_utility(
        prepared,
        mixture_config=mixture_config,
    )

    selected_records: list[pd.DataFrame] = []
    frame_summaries: list[dict[str, Any]] = []
    for (sequence_id, time_s), frame in prepared.groupby(
        ["sequence_id", "time_s"],
        sort=True,
        dropna=False,
    ):
        frame = frame.copy()
        group_rows: list[dict[str, Any]] = []
        for group_value, siblings in frame.groupby(
            "mixture_hypothesis_group",
            sort=False,
            dropna=False,
        ):
            utilities = siblings["mixture_group_topk_candidate_utility"].to_numpy(float)
            group_rows.append(
                {
                    "mixture_hypothesis_group": str(group_value),
                    "mixture_group_topk_group_score": _aggregate_group_score(
                        utilities,
                        mode=selection_config.group_score_mode,
                    ),
                    "mixture_group_topk_group_size_before": int(len(siblings)),
                }
            )
        groups = pd.DataFrame.from_records(group_rows).sort_values(
            ["mixture_group_topk_group_score", "mixture_hypothesis_group"],
            ascending=[False, True],
            kind="mergesort",
        )
        groups["mixture_group_topk_group_rank"] = np.arange(1, len(groups) + 1, dtype=int)
        groups = groups.head(int(selection_config.group_top_k))
        group_diagnostics = groups.set_index("mixture_hypothesis_group").to_dict(orient="index")

        selected_parts: list[pd.DataFrame] = []
        for group_value in groups["mixture_hypothesis_group"].astype(str):
            siblings = frame.loc[
                frame["mixture_hypothesis_group"].astype(str) == group_value
            ].copy()
            siblings = siblings.sort_values(
                ["mixture_group_topk_candidate_utility", "mixture_group_input_row"],
                ascending=[False, True],
                kind="mergesort",
            )
            if int(selection_config.max_siblings_per_group) > 0:
                siblings = siblings.head(int(selection_config.max_siblings_per_group))
            siblings["mixture_group_topk_sibling_rank"] = np.arange(
                1,
                len(siblings) + 1,
                dtype=int,
            )
            diagnostics = group_diagnostics[group_value]
            for column, value in diagnostics.items():
                siblings[column] = value
            selected_parts.append(siblings)
        selected_frame = pd.concat(selected_parts, ignore_index=True) if selected_parts else frame.iloc[0:0]
        selected_records.append(selected_frame)
        frame_summaries.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "input_rows": int(len(frame)),
                "input_groups": int(frame["mixture_hypothesis_group"].nunique(dropna=False)),
                "selected_rows": int(len(selected_frame)),
                "selected_groups": int(
                    selected_frame["mixture_hypothesis_group"].nunique(dropna=False)
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
        "mixture_group_topk_candidate_utility",
        "mixture_group_topk_group_score",
        "mixture_group_topk_group_size_before",
        "mixture_group_topk_group_rank",
        "mixture_group_topk_sibling_rank",
    ]
    for column in diagnostic_columns:
        selected[column] = selected_prepared[column].to_numpy()
    selected["mixture_group_topk_selected"] = True
    selected["mixture_group_topk_group_top_k"] = int(selection_config.group_top_k)
    selected["mixture_group_topk_max_siblings_per_group"] = int(
        selection_config.max_siblings_per_group
    )
    selected["mixture_group_topk_group_score_mode"] = str(selection_config.group_score_mode)
    selected = selected.sort_values(
        [
            "sequence_id",
            "time_s",
            "mixture_group_topk_group_rank",
            "mixture_group_topk_sibling_rank",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    summary = _selection_summary(
        original,
        selected,
        selection_config=selection_config,
        enabled=True,
        frame_summaries=pd.DataFrame.from_records(frame_summaries),
    )
    summary["hypothesis_grouping"] = grouping_summary
    return selected, summary


def run_group_topk_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: HypothesisGroupTopKConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> GroupTopKCandidateMixtureResult:
    """Run group-first selection followed by grouped robust mixture-MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or HypothesisGroupTopKConfig()
    selected, selection_summary = select_hypothesis_group_topk(
        candidates,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
    )
    effective_mixture_config = mixture_config
    if int(selection_config.group_top_k) > 0:
        # Group-first selection already bounded every frame.  Disabling the core
        # row top-K prevents siblings from a selected group from crowding out a
        # lower-ranked selected group a second time.
        effective_mixture_config = replace(mixture_config, top_k=0)
    grouped = run_grouped_candidate_mixture_map(
        selected,
        mixture_config=effective_mixture_config,
        group_config=group_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    return GroupTopKCandidateMixtureResult(
        selected_candidates=selected,
        grouped_result=grouped,
        selection_summary=selection_summary,
    )


def write_group_topk_candidate_mixture_outputs(
    result: GroupTopKCandidateMixtureResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write group-selection artifacts and standard grouped-mixture outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "mmuad_group_topk_candidates.csv"
    summary_path = output / "mmuad_group_topk_summary.json"
    result.selected_candidates.to_csv(selected_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2),
        encoding="utf-8",
    )
    paths = write_grouped_candidate_mixture_outputs(result.grouped_result, output)
    paths["group_topk_candidates_csv"] = selected_path
    paths["group_topk_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mmuad-candidate-mixture-group-topk",
        description="select unique MMUAD hypothesis groups before robust mixture-MAP",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--group-top-k", type=int, default=10)
    parser.add_argument("--max-siblings-per-group", type=int, default=2)
    parser.add_argument("--group-score-mode", choices=GROUP_SCORE_MODES, default="max")
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
    parser.add_argument("--hypothesis-group-correction-strength", type=float, default=1.0)
    parser.add_argument(
        "--missing-hypothesis-group-policy",
        choices=("unique", "error"),
        default="unique",
    )
    args = parser.parse_args(argv)

    fallback_columns = tuple(args.fallback_score_column) or ("ranker_score", "confidence")
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
    selection_config = HypothesisGroupTopKConfig(
        group_top_k=args.group_top_k,
        max_siblings_per_group=args.max_siblings_per_group,
        group_score_mode=args.group_score_mode,
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
    result = run_group_topk_candidate_mixture_map(
        candidates,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    paths = write_group_topk_candidate_mixture_outputs(result, args.output_dir)
    print("mmuad_candidate_mixture_group_topk=ok")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    pooled = result.grouped_result.mixture_result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _candidate_unary_utility(
    prepared: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig,
) -> np.ndarray:
    normalized_score = pd.to_numeric(
        prepared["mixture_group_base_normalized_score"],
        errors="coerce",
    ).fillna(0.0).to_numpy(float)
    sigma = _candidate_sigmas(prepared, mixture_config=mixture_config)
    temperature = max(float(mixture_config.temperature), 1.0e-12)
    return (
        float(mixture_config.score_weight) * normalized_score / temperature
        - float(mixture_config.sigma_log_weight) * np.log(sigma)
    )


def _candidate_sigmas(
    rows: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig,
) -> np.ndarray:
    if mixture_config.sigma_column in rows.columns:
        sigma = pd.to_numeric(rows[mixture_config.sigma_column], errors="coerce").to_numpy(float)
    else:
        sigma = np.full(len(rows), float(mixture_config.default_sigma_m), dtype=float)
    valid = np.isfinite(sigma) & (sigma > 0.0)
    sigma = np.where(valid, sigma, float(mixture_config.default_sigma_m))
    return np.clip(
        sigma,
        max(float(mixture_config.sigma_min_m), 1.0e-9),
        max(float(mixture_config.sigma_max_m), float(mixture_config.sigma_min_m)),
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


def _validate_selection_config(config: HypothesisGroupTopKConfig) -> None:
    if int(config.group_top_k) < 0:
        raise ValueError("group_top_k must be non-negative")
    if int(config.max_siblings_per_group) < 0:
        raise ValueError("max_siblings_per_group must be non-negative")
    if config.group_score_mode not in GROUP_SCORE_MODES:
        raise ValueError(
            f"group_score_mode must be one of {GROUP_SCORE_MODES}, got {config.group_score_mode!r}"
        )


def _selection_summary(
    input_rows: pd.DataFrame,
    selected_rows: pd.DataFrame,
    *,
    selection_config: HypothesisGroupTopKConfig,
    enabled: bool,
    frame_summaries: pd.DataFrame | None = None,
) -> dict[str, Any]:
    frames = frame_summaries if frame_summaries is not None else pd.DataFrame()
    return {
        "enabled": bool(enabled),
        "config": asdict(selection_config),
        "input_candidate_rows": int(len(input_rows)),
        "selected_candidate_rows": int(len(selected_rows)),
        "dropped_candidate_rows": int(max(len(input_rows) - len(selected_rows), 0)),
        "input_frame_count": int(
            input_rows.groupby(["sequence_id", "time_s"], dropna=False).ngroups
            if not input_rows.empty
            else 0
        ),
        "selected_frame_count": int(
            selected_rows.groupby(["sequence_id", "time_s"], dropna=False).ngroups
            if not selected_rows.empty
            else 0
        ),
        "input_groups_per_frame_mean": _safe_mean(frames.get("input_groups")),
        "selected_groups_per_frame_mean": _safe_mean(frames.get("selected_groups")),
        "selected_rows_per_frame_mean": _safe_mean(frames.get("selected_rows")),
        "selected_rows_per_frame_max": _safe_max(frames.get("selected_rows")),
    }


def _safe_mean(values: Any) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    return None if numeric.empty else float(numeric.mean())


def _safe_max(values: Any) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    return None if numeric.empty else float(numeric.max())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
