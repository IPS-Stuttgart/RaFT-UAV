"""Per-sequence restart selection for MMUAD candidate-mixture MAP.

The maintained multi-start smoother currently chooses one restart for an entire
validation/test batch.  MMUAD sequences are conditionally independent in the
trajectory objective, however, and different sequences can be explained by
different raw, translated, dynamic, or external initializations.  A single
batch-wide restart is therefore an unnecessary coupling.

This module evaluates every existing restart with the same truth-free robust
mixture objective, selects the best restart independently for each sequence,
then performs one final inference run with the combined sequence-specific
initialization.  Optional truth remains diagnostic only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from raft_uav.mmuad import candidate_mixture_map as core
from raft_uav.mmuad import candidate_mixture_map_multistart as multistart
from raft_uav.mmuad.io import load_candidate_csv, load_truth_csv
from raft_uav.mmuad.schema import normalize_candidate_columns
from raft_uav.mmuad.submission import (
    load_sequence_class_map,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)

SEQUENCE_SUMMARY_CSV = "mmuad_candidate_mixture_sequence_multistart_summary.csv"
SEQUENCE_SUMMARY_JSON = "mmuad_candidate_mixture_sequence_multistart_summary.json"
ALL_INITIALIZATIONS_CSV = "mmuad_candidate_mixture_sequence_multistart_initializations.csv"
SELECTED_INITIALIZATIONS_CSV = (
    "mmuad_candidate_mixture_sequence_multistart_selected_initializations.csv"
)


@dataclass(frozen=True)
class CandidateMixtureSequenceMultiStartResult:
    """Selected trajectory plus per-sequence restart diagnostics."""

    selected_start: str
    selected_start_by_sequence: Mapping[str, str]
    selected_result: core.CandidateMixtureMapResult
    start_summary: pd.DataFrame
    initializations: Mapping[str, pd.DataFrame | None]
    selected_initializations: pd.DataFrame
    summary: dict[str, Any]


def run_sequence_multistart_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    mixture_config: core.CandidateMixtureMapConfig | None = None,
    multistart_config: multistart.CandidateMixtureMultiStartConfig | None = None,
    external_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> CandidateMixtureSequenceMultiStartResult:
    """Select the lowest truth-free restart separately for every sequence."""

    mixture_config = mixture_config or core.CandidateMixtureMapConfig()
    multistart_config = multistart_config or multistart.CandidateMixtureMultiStartConfig()
    candidate_rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if candidate_rows.empty:
        raise ValueError("candidate-mixture sequence multi-start requires candidate rows")

    starts = multistart.build_candidate_mixture_initializations(
        candidate_rows,
        mixture_config=mixture_config,
        multistart_config=multistart_config,
        external_initial_estimates=external_initial_estimates,
    )
    if not starts:
        raise ValueError("candidate-mixture sequence multi-start produced no starts")

    results: dict[str, core.CandidateMixtureMapResult] = {}
    records: list[dict[str, Any]] = []
    for start_name, initial_estimates in starts.items():
        result = core.run_candidate_mixture_map(
            candidate_rows,
            config=mixture_config,
            initial_estimates=initial_estimates,
            truth=truth,
        )
        results[start_name] = result
        for sequence_id in _result_sequence_ids(result):
            sequence_result = _slice_result(result, sequence_id=sequence_id)
            sequence_candidates = _slice_sequence(candidate_rows, sequence_id=sequence_id)
            sequence_initial = _slice_initialization(initial_estimates, sequence_id=sequence_id)
            objective = multistart.compute_candidate_mixture_selection_objective(
                sequence_result,
                mixture_config=mixture_config,
                candidates=sequence_candidates,
                initial_estimates=sequence_initial,
            )
            sequence_metrics = (
                result.summary.get("metrics", {}).get("sequences", {}).get(str(sequence_id), {})
            )
            records.append(
                {
                    "sequence_id": str(sequence_id),
                    "start_name": str(start_name),
                    "start_type": str(start_name).split(":", 1)[0],
                    **objective,
                    "estimate_rows": int(len(sequence_result.estimates)),
                    "assignment_rows": int(len(sequence_result.assignments)),
                    "mean_assignment_entropy": _column_mean(
                        sequence_result.estimates,
                        "mixture_assignment_entropy",
                    ),
                    "mean_effective_sigma_m": _column_mean(
                        sequence_result.estimates,
                        "mixture_effective_sigma_m",
                    ),
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

    ranked = _rank_sequence_starts(pd.DataFrame.from_records(records))
    selected_rows = ranked.loc[ranked["selected"]].copy()
    selected_start_by_sequence = {
        str(row.sequence_id): str(row.start_name)
        for row in selected_rows.itertuples(index=False)
    }
    selected_initializations = _combine_selected_initializations(
        starts,
        selected_start_by_sequence=selected_start_by_sequence,
    )

    selected_names = sorted(set(selected_start_by_sequence.values()))
    if len(selected_names) == 1:
        selected_start = selected_names[0]
        selected_result = results[selected_start]
    else:
        selected_start = "per-sequence"
        selected_result = core.run_candidate_mixture_map(
            candidate_rows,
            config=mixture_config,
            initial_estimates=(
                selected_initializations if not selected_initializations.empty else None
            ),
            truth=truth,
        )

    selected_objective = pd.to_numeric(
        selected_rows["selection_objective"],
        errors="coerce",
    )
    summary = {
        "schema": "raft-uav-mmuad-candidate-mixture-sequence-multistart-v1",
        "selected_start": selected_start,
        "selected_start_by_sequence": selected_start_by_sequence,
        "selected_start_counts": {
            str(key): int(value)
            for key, value in pd.Series(
                list(selected_start_by_sequence.values()),
                dtype=str,
            ).value_counts().items()
        },
        "sequence_count": int(len(selected_start_by_sequence)),
        "start_count": int(len(starts)),
        "selected_objective_sum": (
            float(selected_objective.sum()) if selected_objective.notna().all() else None
        ),
        "mixture_config": asdict(mixture_config),
        "multistart_config": asdict(multistart_config),
        "truth_used_for_selection": False,
        "final_result_reused": bool(len(selected_names) == 1),
    }
    return CandidateMixtureSequenceMultiStartResult(
        selected_start=selected_start,
        selected_start_by_sequence=selected_start_by_sequence,
        selected_result=selected_result,
        start_summary=ranked,
        initializations=starts,
        selected_initializations=selected_initializations,
        summary=_jsonable(summary),
    )


def write_sequence_multistart_candidate_mixture_outputs(
    result: CandidateMixtureSequenceMultiStartResult,
    *,
    output_dir: Path,
    class_map: Mapping[str, str] | None = None,
    default_classification: int | str = 2,
    official_results_csv: Path | None = None,
    official_zip: Path | None = None,
) -> dict[str, Path]:
    """Write selected mixture artifacts and per-sequence restart provenance."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = core.write_candidate_mixture_map_outputs(result.selected_result, output)

    summary_csv = output / SEQUENCE_SUMMARY_CSV
    result.start_summary.to_csv(summary_csv, index=False)
    paths["sequence_multistart_summary_csv"] = summary_csv

    initialization_parts: list[pd.DataFrame] = []
    for start_name, rows in result.initializations.items():
        if rows is None or rows.empty:
            continue
        part = pd.DataFrame(rows).copy()
        part.insert(0, "start_name", str(start_name))
        initialization_parts.append(part)
    all_initializations = (
        pd.concat(initialization_parts, ignore_index=True)
        if initialization_parts
        else pd.DataFrame()
    )
    all_initializations_csv = output / ALL_INITIALIZATIONS_CSV
    all_initializations.to_csv(all_initializations_csv, index=False)
    paths["all_initializations_csv"] = all_initializations_csv

    selected_initializations_csv = output / SELECTED_INITIALIZATIONS_CSV
    result.selected_initializations.to_csv(selected_initializations_csv, index=False)
    paths["selected_initializations_csv"] = selected_initializations_csv

    summary_json = output / SEQUENCE_SUMMARY_JSON
    summary_json.write_text(json.dumps(_jsonable(result.summary), indent=2), encoding="utf-8")
    paths["sequence_multistart_summary_json"] = summary_json

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
        prog="python -m raft_uav.mmuad.candidate_mixture_map_sequence_multistart",
        description="select branch-seeded candidate-mixture restarts independently per MMUAD sequence",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--external-initial-estimates-csv", type=Path)
    parser.add_argument("--branch-column", default="candidate_branch")
    parser.add_argument("--max-branch-starts", type=int, default=8)
    parser.add_argument("--min-branch-frame-fraction", type=float, default=0.05)
    parser.add_argument("--no-score-top1-start", action="store_true")
    parser.add_argument("--no-frame-median-start", action="store_true")
    parser.add_argument("--no-branch-starts", action="store_true")
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
    external = None
    if args.external_initial_estimates_csv is not None:
        external = pd.read_csv(
            args.external_initial_estimates_csv,
            dtype=str,
            keep_default_na=False,
        )
        external.columns = [str(column).strip() for column in external.columns]

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
    multistart_config = multistart.CandidateMixtureMultiStartConfig(
        branch_column=str(args.branch_column),
        include_score_top1=not bool(args.no_score_top1_start),
        include_frame_median=not bool(args.no_frame_median_start),
        include_branch_starts=not bool(args.no_branch_starts),
        max_branch_starts=int(args.max_branch_starts),
        min_branch_frame_fraction=float(args.min_branch_frame_fraction),
    )
    result = run_sequence_multistart_candidate_mixture_map(
        candidates,
        mixture_config=mixture_config,
        multistart_config=multistart_config,
        external_initial_estimates=external,
        truth=truth,
    )
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_sequence_multistart_candidate_mixture_outputs(
        result,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        official_results_csv=args.official_results_csv,
        official_zip=args.official_zip,
    )
    print("mmuad_candidate_mixture_sequence_multistart=ok")
    print(f"selected_start={result.selected_start}")
    print(f"sequence_count={len(result.selected_start_by_sequence)}")
    print(f"selected_start_count={len(set(result.selected_start_by_sequence.values()))}")
    pooled = result.selected_result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"diagnostic_rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _rank_sequence_starts(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        raise ValueError("candidate-mixture sequence multi-start produced no sequence scores")
    ranked = rows.sort_values(
        ["sequence_id", "selection_objective", "mixture_data_nll", "start_name"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked["sequence_restart_rank"] = (
        ranked.groupby("sequence_id", sort=False).cumcount() + 1
    )
    ranked["selected"] = ranked["sequence_restart_rank"] == 1
    return ranked


def _result_sequence_ids(result: core.CandidateMixtureMapResult) -> list[str]:
    rows = pd.DataFrame(result.estimates)
    if rows.empty or "sequence_id" not in rows.columns:
        return []
    return sorted(rows["sequence_id"].astype(str).drop_duplicates().tolist())


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


def _combine_selected_initializations(
    starts: Mapping[str, pd.DataFrame | None],
    *,
    selected_start_by_sequence: Mapping[str, str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for sequence_id, start_name in sorted(selected_start_by_sequence.items()):
        rows = starts.get(str(start_name))
        if rows is None:
            continue
        selected = _slice_sequence(rows, sequence_id=str(sequence_id))
        if selected.empty:
            continue
        selected = selected.copy()
        selected["selected_start_name"] = str(start_name)
        parts.append(selected)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(
        ["sequence_id", "time_s"],
    ).reset_index(drop=True)


def _column_mean(rows: pd.DataFrame, column: str) -> float | None:
    frame = pd.DataFrame(rows)
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


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
