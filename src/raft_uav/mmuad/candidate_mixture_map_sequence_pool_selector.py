"""Select MMUAD candidate-pool variants independently per sequence.

A branch-preserving reservoir keeps useful raw, dynamic, translated, and merged
candidates available to mixture-MAP.  Some branches can nevertheless be harmful
for particular sequences.  A single global branch ablation therefore either
keeps clutter everywhere or removes a useful branch everywhere.

This module evaluates the full pool and leave-one-group-out variants with the
same truth-free robust candidate-mixture objective, normalizes that objective
for the number of mixture components, and selects the best pool independently
for every sequence.  Optional truth is diagnostic only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad import candidate_mixture_map as core
from raft_uav.mmuad import candidate_mixture_map_multistart as multistart
from raft_uav.mmuad.io import load_candidate_csv, load_truth_csv
from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns
from raft_uav.mmuad.submission import (
    load_sequence_class_map,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.tracker import compute_metrics

POOL_SUMMARY_CSV = "mmuad_candidate_mixture_sequence_pool_summary.csv"
POOL_SUMMARY_JSON = "mmuad_candidate_mixture_sequence_pool_summary.json"
SELECTED_CANDIDATES_CSV = "mmuad_candidate_mixture_sequence_pool_candidates.csv"


@dataclass(frozen=True)
class CandidatePoolSequenceSelectorConfig:
    """Configuration for truth-free per-sequence candidate-pool selection."""

    group_column: str = "candidate_branch"
    include_full_pool: bool = True
    include_leave_one_out: bool = True
    max_leave_one_out: int = 8
    min_group_frame_fraction: float = 0.05
    restore_missing_frames: bool = True
    normalize_component_count: bool = True


@dataclass(frozen=True)
class CandidatePoolSequenceSelectorResult:
    """Selected mixture result plus candidate-pool diagnostics."""

    selected_pool_by_sequence: Mapping[str, str]
    selected_result: core.CandidateMixtureMapResult
    pool_summary: pd.DataFrame
    pool_candidates: Mapping[str, pd.DataFrame]
    selected_candidates: pd.DataFrame
    summary: dict[str, Any]


def run_sequence_pool_selector(
    candidates: pd.DataFrame,
    *,
    mixture_config: core.CandidateMixtureMapConfig | None = None,
    selector_config: CandidatePoolSequenceSelectorConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> CandidatePoolSequenceSelectorResult:
    """Select a full or leave-one-group-out candidate pool per sequence."""

    mixture_config = mixture_config or core.CandidateMixtureMapConfig()
    selector_config = selector_config or CandidatePoolSequenceSelectorConfig()
    _validate_selector_config(selector_config)

    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        raise ValueError("candidate-pool sequence selection requires candidate rows")
    rows = rows.reset_index(drop=True)
    group_column = str(selector_config.group_column)
    if group_column not in rows.columns:
        rows[group_column] = "unknown"
    rows[group_column] = _clean_labels(rows[group_column])

    pools = build_sequence_candidate_pool_variants(rows, config=selector_config)
    if not pools:
        raise ValueError("candidate-pool sequence selection produced no pool variants")

    pool_results: dict[str, core.CandidateMixtureMapResult] = {}
    records: list[dict[str, Any]] = []
    sequence_ids = sorted(rows["sequence_id"].astype(str).drop_duplicates().tolist())
    for pool_label, pool_rows in pools.items():
        result = core.run_candidate_mixture_map(
            pool_rows,
            config=mixture_config,
            initial_estimates=initial_estimates,
            truth=truth,
        )
        pool_results[pool_label] = result
        for sequence_id in sequence_ids:
            sequence_result = _slice_result(result, sequence_id=sequence_id)
            if sequence_result.estimates.empty:
                continue
            sequence_candidates = _slice_sequence(pool_rows, sequence_id=sequence_id)
            sequence_initial = _slice_initialization(
                initial_estimates,
                sequence_id=sequence_id,
            )
            objective = multistart.compute_candidate_mixture_selection_objective(
                sequence_result,
                mixture_config=mixture_config,
                candidates=sequence_candidates,
                initial_estimates=sequence_initial,
            )
            component_penalty = _component_count_penalty(sequence_result.assignments)
            normalized_objective = float(objective["selection_objective"])
            if selector_config.normalize_component_count:
                normalized_objective += component_penalty
            sequence_metrics = (
                result.summary.get("metrics", {}).get("sequences", {}).get(sequence_id, {})
            )
            records.append(
                {
                    "sequence_id": sequence_id,
                    "pool_label": pool_label,
                    "pool_type": "full" if pool_label == "full_pool" else "leave_one_out",
                    "removed_group": _removed_group(pool_label),
                    **objective,
                    "component_count_penalty": float(component_penalty),
                    "normalized_selection_objective": float(normalized_objective),
                    "candidate_rows": int(len(sequence_candidates)),
                    "candidate_frame_count": _frame_count(sequence_candidates),
                    "candidate_count_mean": _mean_candidates_per_frame(sequence_candidates),
                    "restored_frame_count": _restored_frame_count(sequence_candidates),
                    "estimate_rows": int(len(sequence_result.estimates)),
                    "assignment_rows": int(len(sequence_result.assignments)),
                    "diagnostic_mse_3d_m2": _optional_float(
                        sequence_metrics.get("mse_3d_m2")
                    ),
                    "diagnostic_rmse_3d_m": _optional_float(
                        sequence_metrics.get("rmse_3d_m")
                    ),
                    "diagnostic_p95_3d_m": _optional_float(
                        sequence_metrics.get("p95_3d_m")
                    ),
                    "diagnostic_max_3d_m": _optional_float(
                        sequence_metrics.get("max_3d_m")
                    ),
                }
            )

    ranked = _rank_pool_variants(pd.DataFrame.from_records(records))
    selected_rows = ranked.loc[ranked["selected"]].copy()
    selected_pool_by_sequence = {
        str(row.sequence_id): str(row.pool_label)
        for row in selected_rows.itertuples(index=False)
    }
    if set(selected_pool_by_sequence) != set(sequence_ids):
        missing = sorted(set(sequence_ids) - set(selected_pool_by_sequence))
        raise ValueError(f"candidate-pool selection missing sequences: {missing}")

    selected_candidates = _combine_selected_candidates(
        pools,
        selected_pool_by_sequence=selected_pool_by_sequence,
    )
    selected_result = _combine_selected_results(
        pool_results,
        selected_pool_by_sequence=selected_pool_by_sequence,
        mixture_config=mixture_config,
        truth=truth,
    )
    selected_objectives = pd.to_numeric(
        selected_rows["normalized_selection_objective"],
        errors="coerce",
    )
    summary = {
        "schema": "raft-uav-mmuad-candidate-mixture-sequence-pool-selector-v1",
        "selected_pool_by_sequence": selected_pool_by_sequence,
        "selected_pool_counts": {
            str(key): int(value)
            for key, value in pd.Series(
                list(selected_pool_by_sequence.values()),
                dtype=str,
            ).value_counts().items()
        },
        "sequence_count": int(len(sequence_ids)),
        "pool_count": int(len(pools)),
        "selected_objective_sum": (
            float(selected_objectives.sum()) if selected_objectives.notna().all() else None
        ),
        "mixture_config": asdict(mixture_config),
        "selector_config": asdict(selector_config),
        "truth_used_for_selection": False,
        "metrics": selected_result.summary.get("metrics", {}),
    }
    return CandidatePoolSequenceSelectorResult(
        selected_pool_by_sequence=selected_pool_by_sequence,
        selected_result=selected_result,
        pool_summary=ranked,
        pool_candidates=pools,
        selected_candidates=selected_candidates,
        summary=_jsonable(summary),
    )


def build_sequence_candidate_pool_variants(
    candidates: pd.DataFrame,
    *,
    config: CandidatePoolSequenceSelectorConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build full and leave-one-group-out candidate pools with frame coverage."""

    config = config or CandidatePoolSequenceSelectorConfig()
    _validate_selector_config(config)
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        return {}
    rows = rows.reset_index(drop=True)
    group_column = str(config.group_column)
    if group_column not in rows.columns:
        rows[group_column] = "unknown"
    rows[group_column] = _clean_labels(rows[group_column])
    rows["_pool_original_row"] = np.arange(len(rows), dtype=int)

    pools: dict[str, pd.DataFrame] = {}
    if config.include_full_pool:
        pools["full_pool"] = _annotate_pool(
            rows,
            pool_label="full_pool",
            removed_group=None,
        )
    if config.include_leave_one_out:
        for group_value in _eligible_groups(rows, config=config):
            pool_label = f"without_{_slug(group_column)}_{_slug(group_value)}"
            kept = rows.loc[rows[group_column].astype(str) != str(group_value)].copy()
            if kept.empty:
                continue
            if config.restore_missing_frames:
                kept = _restore_missing_frames(kept, full_rows=rows)
            if kept.empty:
                continue
            pools[pool_label] = _annotate_pool(
                kept,
                pool_label=pool_label,
                removed_group=group_value,
            )
    return {
        label: frame.drop(columns=["_pool_original_row"], errors="ignore").reset_index(drop=True)
        for label, frame in pools.items()
    }


def write_sequence_pool_selector_outputs(
    result: CandidatePoolSequenceSelectorResult,
    *,
    output_dir: Path,
    class_map: Mapping[str, str] | None = None,
    default_classification: int | str = 2,
    official_results_csv: Path | None = None,
    official_zip: Path | None = None,
) -> dict[str, Path]:
    """Write selected mixture artifacts and candidate-pool provenance."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = core.write_candidate_mixture_map_outputs(result.selected_result, output)

    summary_csv = output / POOL_SUMMARY_CSV
    result.pool_summary.to_csv(summary_csv, index=False)
    paths["sequence_pool_summary_csv"] = summary_csv

    selected_candidates_csv = output / SELECTED_CANDIDATES_CSV
    result.selected_candidates.to_csv(selected_candidates_csv, index=False)
    paths["selected_candidates_csv"] = selected_candidates_csv

    summary_json = output / POOL_SUMMARY_JSON
    summary_json.write_text(json.dumps(_jsonable(result.summary), indent=2), encoding="utf-8")
    paths["sequence_pool_summary_json"] = summary_json

    class_map = dict(class_map or {})
    if official_results_csv is not None:
        write_official_mmaud_results_csv(
            result.selected_result.estimates,
            official_results_csv,
            classification=default_classification,
            class_map=class_map,
        )
        paths["official_results_csv"] = Path(official_results_csv)
    if official_zip is not None:
        write_official_ug2_codabench_zip(
            result.selected_result.estimates,
            official_zip,
            classification=default_classification,
            class_map=class_map,
        )
        paths["official_zip"] = Path(official_zip)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-mixture-sequence-pool-selector",
        description=(
            "select full or leave-one-group-out MMUAD candidate pools per sequence "
            "with a truth-free mixture objective"
        ),
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--group-column", default="candidate_branch")
    parser.add_argument("--max-leave-one-out", type=int, default=8)
    parser.add_argument("--min-group-frame-fraction", type=float, default=0.05)
    parser.add_argument("--no-full-pool", action="store_true")
    parser.add_argument("--no-leave-one-out", action="store_true")
    parser.add_argument("--no-restore-missing-frames", action="store_true")
    parser.add_argument("--no-component-count-normalization", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--fallback-score-column", action="append", default=[])
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument(
        "--score-normalization",
        choices=core.SCORE_NORMALIZATION_CHOICES,
        default="minmax",
    )
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=3.0)
    parser.add_argument("--loss", choices=core.LOSS_CHOICES, default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tolerance-m", type=float, default=1.0e-3)
    parser.add_argument("--target-time-tolerance-s", type=float, default=0.5)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument("--branch-balance", type=float, default=0.0)
    parser.add_argument("--source-balance", type=float, default=0.0)
    parser.add_argument("--responsibility-floor", type=float, default=0.0)
    parser.add_argument(
        "--initialization",
        choices=core.INITIALIZATION_CHOICES,
        default="uncertainty-top1",
    )
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="2")
    parser.add_argument("--official-results-csv", type=Path)
    parser.add_argument("--official-zip", type=Path)
    args = parser.parse_args(argv)

    candidates = load_candidate_csv(args.candidates_csv).rows
    truth = None if args.truth_csv is None else load_truth_csv(args.truth_csv).rows
    initial_estimates = None
    if args.initial_estimates_csv is not None:
        initial_estimates = pd.read_csv(
            args.initial_estimates_csv,
            dtype=str,
            keep_default_na=False,
        )
        initial_estimates.columns = [str(column).strip() for column in initial_estimates.columns]

    fallback = tuple(args.fallback_score_column) or ("ranker_score", "confidence")
    mixture_config = core.CandidateMixtureMapConfig(
        top_k=int(args.top_k),
        score_column=str(args.score_column),
        fallback_score_columns=fallback,
        sigma_column=str(args.sigma_column),
        default_sigma_m=float(args.default_sigma_m),
        sigma_min_m=float(args.sigma_min_m),
        sigma_max_m=float(args.sigma_max_m),
        score_normalization=str(args.score_normalization),
        score_weight=float(args.score_weight),
        temperature=float(args.temperature),
        sigma_log_weight=float(args.sigma_log_weight),
        loss=str(args.loss),
        huber_delta=float(args.huber_delta),
        smoothness_weight=float(args.smoothness_weight),
        anchor_weight=float(args.anchor_weight),
        iterations=int(args.iterations),
        tolerance_m=float(args.tolerance_m),
        target_time_tolerance_s=float(args.target_time_tolerance_s),
        uniform_weight_floor=float(args.uniform_weight_floor),
        branch_balance=float(args.branch_balance),
        source_balance=float(args.source_balance),
        responsibility_floor=float(args.responsibility_floor),
        initialization=str(args.initialization),
    )
    selector_config = CandidatePoolSequenceSelectorConfig(
        group_column=str(args.group_column),
        include_full_pool=not bool(args.no_full_pool),
        include_leave_one_out=not bool(args.no_leave_one_out),
        max_leave_one_out=int(args.max_leave_one_out),
        min_group_frame_fraction=float(args.min_group_frame_fraction),
        restore_missing_frames=not bool(args.no_restore_missing_frames),
        normalize_component_count=not bool(args.no_component_count_normalization),
    )
    result = run_sequence_pool_selector(
        candidates,
        mixture_config=mixture_config,
        selector_config=selector_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_sequence_pool_selector_outputs(
        result,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        official_results_csv=args.official_results_csv,
        official_zip=args.official_zip,
    )
    print("mmuad_candidate_mixture_sequence_pool_selector=ok")
    print(f"sequence_count={len(result.selected_pool_by_sequence)}")
    print(f"selected_pool_count={len(set(result.selected_pool_by_sequence.values()))}")
    pooled = result.selected_result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"diagnostic_rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _eligible_groups(
    rows: pd.DataFrame,
    *,
    config: CandidatePoolSequenceSelectorConfig,
) -> list[str]:
    group_column = str(config.group_column)
    total_frames = max(_frame_count(rows), 1)
    records = []
    for group_value, group in rows.groupby(group_column, sort=False):
        frame_fraction = _frame_count(group) / float(total_frames)
        if frame_fraction < float(config.min_group_frame_fraction):
            continue
        records.append((str(group_value), frame_fraction, len(group)))
    records.sort(key=lambda item: (-item[1], -item[2], item[0]))
    if int(config.max_leave_one_out) > 0:
        records = records[: int(config.max_leave_one_out)]
    return [item[0] for item in records]


def _restore_missing_frames(kept: pd.DataFrame, *, full_rows: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["sequence_id", "time_s"]
    full_keys = full_rows[key_columns].drop_duplicates()
    kept_keys = kept[key_columns].drop_duplicates()
    missing = full_keys.merge(kept_keys, on=key_columns, how="left", indicator=True)
    missing = missing.loc[missing["_merge"] == "left_only", key_columns]
    out = kept.copy()
    out["candidate_pool_fallback_restored"] = False
    if missing.empty:
        return out
    fallback = full_rows.merge(missing, on=key_columns, how="inner")
    fallback = fallback.copy()
    fallback["candidate_pool_fallback_restored"] = True
    return pd.concat([out, fallback], ignore_index=True)


def _annotate_pool(
    rows: pd.DataFrame,
    *,
    pool_label: str,
    removed_group: str | None,
) -> pd.DataFrame:
    out = rows.copy()
    if "candidate_pool_fallback_restored" not in out.columns:
        out["candidate_pool_fallback_restored"] = False
    out["candidate_pool_variant"] = str(pool_label)
    out["candidate_pool_removed_group"] = "" if removed_group is None else str(removed_group)
    return out


def _component_count_penalty(assignments: pd.DataFrame) -> float:
    rows = pd.DataFrame(assignments)
    if rows.empty or not {"sequence_id", "time_s"}.issubset(rows.columns):
        return float("inf")
    counts = rows.groupby(["sequence_id", "time_s"], sort=False).size().to_numpy(float)
    if len(counts) == 0 or np.any(counts <= 0.0):
        return float("inf")
    return float(np.log(counts).sum())


def _rank_pool_variants(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        raise ValueError("candidate-pool sequence selection produced no sequence scores")
    ranked = rows.copy()
    ranked["pool_preference"] = np.where(ranked["pool_label"].eq("full_pool"), 0, 1)
    ranked = ranked.sort_values(
        [
            "sequence_id",
            "normalized_selection_objective",
            "selection_objective",
            "pool_preference",
            "pool_label",
        ],
        ascending=[True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["sequence_pool_rank"] = ranked.groupby("sequence_id", sort=False).cumcount() + 1
    ranked["selected"] = ranked["sequence_pool_rank"] == 1
    return ranked


def _combine_selected_candidates(
    pools: Mapping[str, pd.DataFrame],
    *,
    selected_pool_by_sequence: Mapping[str, str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for sequence_id, pool_label in sorted(selected_pool_by_sequence.items()):
        selected = _slice_sequence(pools[str(pool_label)], sequence_id=str(sequence_id))
        selected = selected.copy()
        selected["selected_candidate_pool"] = str(pool_label)
        parts.append(selected)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _combine_selected_results(
    pool_results: Mapping[str, core.CandidateMixtureMapResult],
    *,
    selected_pool_by_sequence: Mapping[str, str],
    mixture_config: core.CandidateMixtureMapConfig,
    truth: pd.DataFrame | None,
) -> core.CandidateMixtureMapResult:
    estimate_parts: list[pd.DataFrame] = []
    assignment_parts: list[pd.DataFrame] = []
    iteration_parts: list[pd.DataFrame] = []
    metrics_by_sequence: dict[str, Any] = {}
    truth_rows = None
    if truth is not None:
        truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())

    for sequence_id, pool_label in sorted(selected_pool_by_sequence.items()):
        result = pool_results[str(pool_label)]
        estimates = _slice_sequence(result.estimates, sequence_id=str(sequence_id))
        assignments = _slice_sequence(result.assignments, sequence_id=str(sequence_id))
        iterations = _slice_sequence(result.iteration_summary, sequence_id=str(sequence_id))
        for frame in (estimates, assignments, iterations):
            if not frame.empty:
                frame["selected_candidate_pool"] = str(pool_label)
        estimate_parts.append(estimates)
        assignment_parts.append(assignments)
        iteration_parts.append(iterations)
        sequence_truth = None
        if truth_rows is not None:
            sequence_truth = truth_rows.loc[
                truth_rows["sequence_id"].astype(str) == str(sequence_id)
            ]
        metrics_by_sequence[str(sequence_id)] = compute_metrics(estimates, sequence_truth)

    estimates_all = _concat(estimate_parts)
    assignments_all = _concat(assignment_parts)
    iterations_all = _concat(iteration_parts)
    pooled_metrics = compute_metrics(estimates_all, truth_rows)
    summary = {
        "schema": "raft-uav-mmuad-candidate-mixture-sequence-pool-selected-result-v1",
        "candidate_rows": int(len(assignments_all)),
        "sequence_count": int(len(selected_pool_by_sequence)),
        "estimate_rows": int(len(estimates_all)),
        "assignment_rows": int(len(assignments_all)),
        "config": asdict(mixture_config),
        "selected_pool_by_sequence": dict(selected_pool_by_sequence),
        "metrics": {
            "pooled": pooled_metrics,
            "sequences": metrics_by_sequence,
        },
    }
    return core.CandidateMixtureMapResult(
        estimates=estimates_all,
        assignments=assignments_all,
        iteration_summary=iterations_all,
        summary=_jsonable(summary),
    )


def _slice_result(
    result: core.CandidateMixtureMapResult,
    *,
    sequence_id: str,
) -> core.CandidateMixtureMapResult:
    return core.CandidateMixtureMapResult(
        estimates=_slice_sequence(result.estimates, sequence_id=sequence_id),
        assignments=_slice_sequence(result.assignments, sequence_id=sequence_id),
        iteration_summary=_slice_sequence(result.iteration_summary, sequence_id=sequence_id),
        summary={},
    )


def _slice_sequence(rows: pd.DataFrame, *, sequence_id: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if frame.empty or "sequence_id" not in frame.columns:
        return frame.iloc[0:0].copy()
    return frame.loc[frame["sequence_id"].astype(str) == str(sequence_id)].copy()


def _slice_initialization(
    rows: pd.DataFrame | None,
    *,
    sequence_id: str,
) -> pd.DataFrame | None:
    if rows is None:
        return None
    selected = _slice_sequence(rows, sequence_id=sequence_id)
    return selected if not selected.empty else None


def _frame_count(rows: pd.DataFrame) -> int:
    frame = pd.DataFrame(rows)
    if frame.empty or not {"sequence_id", "time_s"}.issubset(frame.columns):
        return 0
    return int(len(frame[["sequence_id", "time_s"]].drop_duplicates()))


def _mean_candidates_per_frame(rows: pd.DataFrame) -> float | None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None
    counts = frame.groupby(["sequence_id", "time_s"], sort=False).size()
    return float(counts.mean()) if not counts.empty else None


def _restored_frame_count(rows: pd.DataFrame) -> int:
    frame = pd.DataFrame(rows)
    column = "candidate_pool_fallback_restored"
    if frame.empty or column not in frame.columns:
        return 0
    restored = frame.loc[frame[column].fillna(False).astype(bool)]
    return _frame_count(restored)


def _removed_group(pool_label: str) -> str | None:
    if str(pool_label) == "full_pool":
        return None
    marker = "without_"
    text = str(pool_label)
    return text[len(marker) :] if text.startswith(marker) else None


def _clean_labels(values: pd.Series) -> pd.Series:
    text = values.where(values.notna(), "unknown").astype(str).str.strip()
    missing = text.eq("") | text.str.lower().isin({"nan", "none", "<na>", "nat"})
    return text.where(~missing, "unknown")


def _slug(value: str) -> str:
    text = str(value).strip().lower()
    chars = [char if char.isalnum() else "_" for char in text]
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _validate_selector_config(config: CandidatePoolSequenceSelectorConfig) -> None:
    if not str(config.group_column).strip():
        raise ValueError("group_column must not be empty")
    if int(config.max_leave_one_out) < 0:
        raise ValueError("max_leave_one_out must be non-negative")
    if not 0.0 <= float(config.min_group_frame_fraction) <= 1.0:
        raise ValueError("min_group_frame_fraction must be within [0, 1]")
    if not config.include_full_pool and not config.include_leave_one_out:
        raise ValueError("at least one candidate-pool variant must be enabled")


def _concat(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


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
