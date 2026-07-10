"""Branch-seeded multi-start candidate-mixture MAP for MMUAD.

Candidate-mixture MAP alternates between candidate responsibilities and a smooth
trajectory update, so it can converge to different local solutions from
different initial trajectories.  This module runs the maintained robust
candidate-mixture smoother from global, median, branch-specific, and optional
external starts, then chooses the winner without truth using the final robust
mixture evidence plus the core acceleration regularizer.
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
from raft_uav.mmuad.io import load_candidate_csv, load_truth_csv
from raft_uav.mmuad.schema import normalize_candidate_columns
from raft_uav.mmuad.submission import (
    load_sequence_class_map,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)

START_SUMMARY_CSV = "mmuad_candidate_mixture_multistart_summary.csv"
START_SUMMARY_JSON = "mmuad_candidate_mixture_multistart_summary.json"
INITIALIZATIONS_CSV = "mmuad_candidate_mixture_multistart_initializations.csv"


@dataclass(frozen=True)
class CandidateMixtureMultiStartConfig:
    """Configuration for inference-safe candidate-mixture restarts."""

    branch_column: str = "candidate_branch"
    include_score_top1: bool = True
    include_frame_median: bool = True
    include_branch_starts: bool = True
    max_branch_starts: int = 8
    min_branch_frame_fraction: float = 0.05


@dataclass(frozen=True)
class CandidateMixtureMultiStartResult:
    """Selected mixture result and restart diagnostics."""

    selected_start: str
    selected_result: core.CandidateMixtureMapResult
    start_summary: pd.DataFrame
    initializations: Mapping[str, pd.DataFrame | None]
    summary: dict[str, Any]


def build_candidate_mixture_initializations(
    candidates: pd.DataFrame,
    *,
    mixture_config: core.CandidateMixtureMapConfig | None = None,
    multistart_config: CandidateMixtureMultiStartConfig | None = None,
    external_initial_estimates: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | None]:
    """Build complete global and branch-specific initial trajectories."""

    mixture_config = mixture_config or core.CandidateMixtureMapConfig()
    multistart_config = multistart_config or CandidateMixtureMultiStartConfig()
    _validate_multistart_config(multistart_config)
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    starts: dict[str, pd.DataFrame | None] = {"core-default": None}
    if rows.empty:
        if external_initial_estimates is not None:
            starts["external"] = _normalize_external_initialization(external_initial_estimates)
        return starts

    branch_column = str(multistart_config.branch_column)
    rows = rows.reset_index(drop=True)
    if branch_column not in rows.columns:
        rows[branch_column] = (
            rows["source"].fillna("unknown").astype(str)
            if "source" in rows.columns
            else "unknown"
        )
    rows[branch_column] = rows[branch_column].fillna("unknown").astype(str)

    frame_cache: list[tuple[pd.DataFrame, pd.Series]] = []
    score_records: list[dict[str, Any]] = []
    median_records: list[dict[str, Any]] = []
    for (sequence_id, time_s), frame in rows.groupby(["sequence_id", "time_s"], sort=True):
        prepared = _prepare_initialization_frame(frame, mixture_config)
        fallback = prepared.iloc[int(np.argmax(prepared["_multistart_init_score"]))]
        score_row = prepared.iloc[int(np.argmax(prepared["_multistart_normalized_score"]))]
        frame_cache.append((prepared, fallback))
        score_records.append(_initial_estimate_record(score_row))
        median_records.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "state_x_m": float(pd.to_numeric(frame["x_m"], errors="coerce").median()),
                "state_y_m": float(pd.to_numeric(frame["y_m"], errors="coerce").median()),
                "state_z_m": float(pd.to_numeric(frame["z_m"], errors="coerce").median()),
            }
        )

    if multistart_config.include_score_top1:
        starts["score-top1"] = pd.DataFrame.from_records(score_records)
    if multistart_config.include_frame_median:
        starts["frame-median"] = pd.DataFrame.from_records(median_records)
    if multistart_config.include_branch_starts:
        branches = _eligible_branches(
            rows,
            frame_count=len(frame_cache),
            branch_column=branch_column,
            config=multistart_config,
        )
        for branch in branches:
            records = []
            for prepared, fallback in frame_cache:
                branch_rows = prepared.loc[prepared[branch_column].astype(str) == branch]
                chosen = fallback
                if not branch_rows.empty:
                    index = int(np.argmax(branch_rows["_multistart_init_score"].to_numpy(float)))
                    chosen = branch_rows.iloc[index]
                records.append(_initial_estimate_record(chosen))
            starts[f"branch:{branch}"] = pd.DataFrame.from_records(records)
    if external_initial_estimates is not None:
        starts["external"] = _normalize_external_initialization(external_initial_estimates)
    return starts


def run_multistart_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    mixture_config: core.CandidateMixtureMapConfig | None = None,
    multistart_config: CandidateMixtureMultiStartConfig | None = None,
    external_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> CandidateMixtureMultiStartResult:
    """Run every restart and select the lowest truth-free final objective."""

    mixture_config = mixture_config or core.CandidateMixtureMapConfig()
    multistart_config = multistart_config or CandidateMixtureMultiStartConfig()
    starts = build_candidate_mixture_initializations(
        candidates,
        mixture_config=mixture_config,
        multistart_config=multistart_config,
        external_initial_estimates=external_initial_estimates,
    )
    results: dict[str, core.CandidateMixtureMapResult] = {}
    records: list[dict[str, Any]] = []
    for start_name, initial_estimates in starts.items():
        result = core.run_candidate_mixture_map(
            candidates,
            config=mixture_config,
            initial_estimates=initial_estimates,
            truth=truth,
        )
        objective = compute_candidate_mixture_selection_objective(
            result,
            mixture_config=mixture_config,
        )
        pooled = result.summary.get("metrics", {}).get("pooled", {})
        results[start_name] = result
        records.append(
            {
                "start_name": start_name,
                "start_type": start_name.split(":", 1)[0],
                **objective,
                "final_quadratic_surrogate": _final_quadratic_surrogate(result),
                "estimate_rows": int(len(result.estimates)),
                "assignment_rows": int(len(result.assignments)),
                "mean_assignment_entropy": _column_mean(
                    result.estimates,
                    "mixture_assignment_entropy",
                ),
                "mean_effective_sigma_m": _column_mean(
                    result.estimates,
                    "mixture_effective_sigma_m",
                ),
                "diagnostic_mse_3d_m2": _optional_float(pooled.get("mse_3d_m2")),
                "diagnostic_rmse_3d_m": _optional_float(pooled.get("rmse_3d_m")),
                "diagnostic_p95_3d_m": _optional_float(pooled.get("p95_3d_m")),
                "diagnostic_max_3d_m": _optional_float(pooled.get("max_3d_m")),
            }
        )

    ranked = pd.DataFrame.from_records(records).sort_values(
        ["selection_objective", "mixture_data_nll", "start_name"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    if ranked.empty:
        raise ValueError("candidate-mixture multi-start produced no starts")
    selected_start = str(ranked.iloc[0]["start_name"])
    ranked["selected"] = ranked["start_name"].astype(str) == selected_start
    summary = {
        "schema": "raft-uav-mmuad-candidate-mixture-multistart-v1",
        "selected_start": selected_start,
        "start_count": int(len(ranked)),
        "mixture_config": asdict(mixture_config),
        "multistart_config": asdict(multistart_config),
        "selection": _jsonable(ranked.iloc[0].to_dict()),
        "truth_used_for_selection": False,
    }
    return CandidateMixtureMultiStartResult(
        selected_start=selected_start,
        selected_result=results[selected_start],
        start_summary=ranked,
        initializations=starts,
        summary=_jsonable(summary),
    )


def compute_candidate_mixture_selection_objective(
    result: core.CandidateMixtureMapResult,
    *,
    mixture_config: core.CandidateMixtureMapConfig,
) -> dict[str, float]:
    """Evaluate final robust mixture evidence plus acceleration regularization."""

    assignments = pd.DataFrame(result.assignments).copy()
    if assignments.empty or "mixture_log_weight" not in assignments.columns:
        return {
            "selection_objective": float("inf"),
            "mixture_data_nll": float("inf"),
            "smoothness_penalty": float("inf"),
        }
    data_nll = 0.0
    for _, frame in assignments.groupby(["sequence_id", "time_s"], sort=False):
        values = pd.to_numeric(frame["mixture_log_weight"], errors="coerce").to_numpy(float)
        data_nll -= _logsumexp(values)
    smoothness = _trajectory_smoothness_penalty(
        result.estimates,
        smoothness_weight=float(mixture_config.smoothness_weight),
    )
    return {
        "selection_objective": float(data_nll + smoothness),
        "mixture_data_nll": float(data_nll),
        "smoothness_penalty": float(smoothness),
    }


def write_multistart_candidate_mixture_outputs(
    result: CandidateMixtureMultiStartResult,
    *,
    output_dir: Path,
    class_map: Mapping[str, str] | None = None,
    default_classification: int | str = 2,
    official_results_csv: Path | None = None,
    official_zip: Path | None = None,
) -> dict[str, Path]:
    """Write the selected core run plus restart diagnostics and submissions."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = core.write_candidate_mixture_map_outputs(result.selected_result, output)
    summary_csv = output / START_SUMMARY_CSV
    result.start_summary.to_csv(summary_csv, index=False)
    paths["multistart_summary_csv"] = summary_csv
    initialization_parts = []
    for start_name, rows in result.initializations.items():
        if rows is None or rows.empty:
            continue
        part = pd.DataFrame(rows).copy()
        part.insert(0, "start_name", start_name)
        initialization_parts.append(part)
    initializations = (
        pd.concat(initialization_parts, ignore_index=True)
        if initialization_parts
        else pd.DataFrame()
    )
    initialization_csv = output / INITIALIZATIONS_CSV
    initializations.to_csv(initialization_csv, index=False)
    paths["initializations_csv"] = initialization_csv
    summary_json = output / START_SUMMARY_JSON
    summary_json.write_text(json.dumps(_jsonable(result.summary), indent=2), encoding="utf-8")
    paths["multistart_summary_json"] = summary_json
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
        prog="python -m raft_uav.mmuad.candidate_mixture_map_multistart",
        description="run branch-seeded multi-start MMUAD candidate-mixture MAP",
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
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tolerance-m", type=float, default=1.0e-3)
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
        iterations=int(args.iterations),
        tolerance_m=float(args.tolerance_m),
        uniform_weight_floor=float(args.uniform_weight_floor),
        branch_balance=float(args.branch_balance),
        source_balance=float(args.source_balance),
        responsibility_floor=float(args.responsibility_floor),
        initialization=str(args.initialization),
    )
    multistart_config = CandidateMixtureMultiStartConfig(
        branch_column=str(args.branch_column),
        include_score_top1=not bool(args.no_score_top1_start),
        include_frame_median=not bool(args.no_frame_median_start),
        include_branch_starts=not bool(args.no_branch_starts),
        max_branch_starts=int(args.max_branch_starts),
        min_branch_frame_fraction=float(args.min_branch_frame_fraction),
    )
    result = run_multistart_candidate_mixture_map(
        candidates,
        mixture_config=mixture_config,
        multistart_config=multistart_config,
        external_initial_estimates=external,
        truth=truth,
    )
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_multistart_candidate_mixture_outputs(
        result,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        official_results_csv=args.official_results_csv,
        official_zip=args.official_zip,
    )
    print("mmuad_candidate_mixture_multistart=ok")
    print(f"selected_start={result.selected_start}")
    print(f"start_count={len(result.start_summary)}")
    print(f"selection_objective={result.summary['selection']['selection_objective']}")
    pooled = result.selected_result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"diagnostic_rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _prepare_initialization_frame(
    frame: pd.DataFrame,
    config: core.CandidateMixtureMapConfig,
) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    raw_score = core._candidate_scores(out, config=config)
    normalized = core._normalize_scores(raw_score.to_numpy(float), mode=config.score_normalization)
    sigma = core._candidate_sigmas(out, config=config)
    out["_multistart_normalized_score"] = normalized
    out["_multistart_init_score"] = (
        float(config.score_weight) * normalized / float(config.temperature)
        - float(config.sigma_log_weight) * np.log(sigma.to_numpy(float))
    )
    return out


def _eligible_branches(
    rows: pd.DataFrame,
    *,
    frame_count: int,
    branch_column: str,
    config: CandidateMixtureMultiStartConfig,
) -> list[str]:
    presence = (
        rows[["sequence_id", "time_s", branch_column]]
        .drop_duplicates()
        .groupby(branch_column, dropna=False)
        .size()
    )
    row_count = rows.groupby(branch_column, dropna=False).size()
    minimum = max(1, int(np.ceil(float(config.min_branch_frame_fraction) * frame_count)))
    branches = [str(branch) for branch, count in presence.items() if int(count) >= minimum]
    branches.sort(key=lambda branch: (-int(presence.get(branch, 0)), -int(row_count.get(branch, 0)), branch))
    return branches[: int(config.max_branch_starts)] if int(config.max_branch_starts) > 0 else branches


def _initial_estimate_record(row: pd.Series) -> dict[str, Any]:
    return {
        "sequence_id": str(row["sequence_id"]),
        "time_s": float(row["time_s"]),
        "state_x_m": float(row["x_m"]),
        "state_y_m": float(row["y_m"]),
        "state_z_m": float(row["z_m"]),
    }


def _normalize_external_initialization(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(rows).copy()
    out.columns = [str(column).strip() for column in out.columns]
    for source, target in {"x_m": "state_x_m", "y_m": "state_y_m", "z_m": "state_z_m"}.items():
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
    if "sequence_id" not in out.columns:
        out["sequence_id"] = "default"
    required = ["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"external initial estimates missing required columns: {missing}")
    for column in required[1:]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    finite = np.isfinite(out[required[1:]].to_numpy(float)).all(axis=1)
    return out.loc[finite, required].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _trajectory_smoothness_penalty(estimates: pd.DataFrame, *, smoothness_weight: float) -> float:
    if smoothness_weight <= 0.0:
        return 0.0
    rows = pd.DataFrame(estimates).copy()
    if rows.empty:
        return float("inf")
    total = 0.0
    for _, sequence in rows.groupby("sequence_id", sort=False):
        ordered = sequence.sort_values("time_s")
        times = pd.to_numeric(ordered["time_s"], errors="coerce").to_numpy(float)
        state = ordered[["state_x_m", "state_y_m", "state_z_m"]].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(float)
        finite = np.isfinite(times) & np.isfinite(state).all(axis=1)
        matrix = core._second_derivative_matrix(times[finite])
        if matrix.size:
            total += float(smoothness_weight) * float(np.sum((matrix @ state[finite]) ** 2))
    return float(total)


def _logsumexp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    if not finite.any():
        return float("-inf")
    maximum = float(np.max(array[finite]))
    shifted = np.clip(array[finite] - maximum, -700.0, 0.0)
    return maximum + float(np.log(np.sum(np.exp(shifted))))


def _final_quadratic_surrogate(result: core.CandidateMixtureMapResult) -> float | None:
    rows = pd.DataFrame(result.iteration_summary).copy()
    if rows.empty or "quadratic_objective" not in rows.columns:
        return None
    final_rows = rows.sort_values("iteration").groupby("sequence_id", sort=False).tail(1)
    values = pd.to_numeric(final_rows["quadratic_objective"], errors="coerce").dropna()
    return float(values.sum()) if not values.empty else None


def _column_mean(rows: pd.DataFrame, column: str) -> float | None:
    if column not in rows.columns:
        return None
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _validate_multistart_config(config: CandidateMixtureMultiStartConfig) -> None:
    if int(config.max_branch_starts) < 0:
        raise ValueError("max_branch_starts must be non-negative")
    if not 0.0 <= float(config.min_branch_frame_fraction) <= 1.0:
        raise ValueError("min_branch_frame_fraction must be within [0, 1]")


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
