"""Grid search for Track 5 consensus-ensemble hyperparameters.

The consensus ensemble is inference-safe, but its radius/fallback settings should
be selected on labeled train folds before being applied to validation or hidden
test submissions.  This helper sweeps those settings on a supplied labeled split,
writes pooled and by-sequence diagnostics, and optionally emits the best
upload-ready Track 5 ZIP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput, parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble_weight_search import (
    _jsonable,
    _normalize_truth_for_exact_template,
    _score_template_estimates,
    _score_template_estimates_by_sequence,
    _selection_objective_value,
    _sequence_objective_metrics,
    SELECTION_OBJECTIVES,
)
from raft_uav.mmuad.track5_estimate_consensus_ensemble import (
    FALLBACK_POLICIES,
    build_track5_consensus_estimate_ensemble,
    write_track5_consensus_ensemble_outputs,
)

CONSENSUS_GRID_CSV = "mmuad_track5_consensus_ensemble_grid.csv"
CONSENSUS_GRID_BY_SEQUENCE_CSV = "mmuad_track5_consensus_ensemble_grid_by_sequence.csv"
BEST_CONFIG_JSON = "mmuad_track5_consensus_ensemble_best_config.json"
BEST_OUTPUT_DIR = "best_consensus_ensemble"


def search_track5_consensus_ensemble_grid(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    consensus_radius_m: Iterable[float] = (2.0, 5.0, 10.0),
    min_consensus_weight_fraction: Iterable[float] = (0.0, 0.5),
    fallback_policy: Iterable[str] = ("max-weight", "weighted-mean"),
    max_nearest_time_delta_s: float | None = None,
    selection_objective: str = "pooled-mse",
    sequence_objective_weight: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate consensus-ensemble settings and return grid plus best config."""

    if selection_objective not in SELECTION_OBJECTIVES:
        raise ValueError(
            f"unsupported selection_objective {selection_objective!r}; allowed={SELECTION_OBJECTIVES}"
        )
    inputs = tuple(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    loaded = [(item.label, pd.read_csv(item.path), float(item.weight)) for item in inputs]
    truth_rows = _normalize_truth_for_exact_template(truth)
    grid_records: list[dict[str, Any]] = []
    by_sequence_records: list[dict[str, Any]] = []
    row_index = 0
    for radius in _float_grid(consensus_radius_m, name="consensus_radius_m"):
        for min_fraction in _float_grid(
            min_consensus_weight_fraction,
            name="min_consensus_weight_fraction",
        ):
            if not 0.0 <= min_fraction <= 1.0:
                raise ValueError("min_consensus_weight_fraction values must be in [0, 1]")
            for policy in fallback_policy:
                if policy not in FALLBACK_POLICIES:
                    raise ValueError(f"unsupported fallback_policy {policy!r}; allowed={FALLBACK_POLICIES}")
                estimates, diagnostics = build_track5_consensus_estimate_ensemble(
                    loaded,
                    template,
                    consensus_radius_m=radius,
                    fallback_policy=policy,
                    min_consensus_weight_fraction=min_fraction,
                    max_nearest_time_delta_s=max_nearest_time_delta_s,
                )
                metrics = _score_template_estimates(estimates, truth_rows)
                by_sequence = _score_template_estimates_by_sequence(estimates, truth_rows)
                sequence_metrics = _sequence_objective_metrics(by_sequence)
                selection_value = _selection_objective_value(
                    metrics,
                    sequence_metrics,
                    selection_objective=selection_objective,
                    sequence_objective_weight=float(sequence_objective_weight),
                )
                record = {
                    "grid_index": int(row_index),
                    "consensus_radius_m": float(radius),
                    "min_consensus_weight_fraction": float(min_fraction),
                    "fallback_policy": policy,
                    "selection_objective": selection_objective,
                    "selection_objective_value": selection_value,
                    "sequence_objective_weight": float(sequence_objective_weight),
                    "valid_input_count_mean": _safe_mean(
                        diagnostics.get("valid_input_count", pd.Series(dtype=float))
                    ),
                    "selected_input_count_mean": _safe_mean(
                        diagnostics.get("selected_input_count", pd.Series(dtype=float))
                    ),
                    "fallback_fraction": _safe_mean(
                        diagnostics.get("fallback_applied", pd.Series(dtype=float)).astype(float)
                        if "fallback_applied" in diagnostics.columns
                        else pd.Series(dtype=float)
                    ),
                    **metrics,
                    **sequence_metrics,
                }
                for item in inputs:
                    record[f"weight_{item.label}"] = float(item.weight)
                grid_records.append(record)
                for _, sequence_row in by_sequence.iterrows():
                    by_sequence_records.append(
                        {
                            "grid_index": int(row_index),
                            "consensus_radius_m": float(radius),
                            "min_consensus_weight_fraction": float(min_fraction),
                            "fallback_policy": policy,
                            "selection_objective": selection_objective,
                            **sequence_row.to_dict(),
                        }
                    )
                row_index += 1
    grid = pd.DataFrame.from_records(grid_records)
    if grid.empty:
        raise ValueError("consensus grid produced no rows")
    by_sequence_grid = pd.DataFrame.from_records(by_sequence_records)
    best_row = grid.sort_values(
        ["selection_objective_value", "pose_mse_m2", "pose_p95_m", "pose_max_m"],
        na_position="last",
    ).iloc[0]
    best_index = int(best_row["grid_index"])
    best_by_sequence = by_sequence_grid.loc[by_sequence_grid["grid_index"] == best_index]
    best = {
        "schema": "raft-uav-mmuad-track5-consensus-ensemble-grid-v1",
        "best_grid_index": best_index,
        "selection_objective": selection_objective,
        "sequence_objective_weight": float(sequence_objective_weight),
        "selection_objective_value": _jsonable(best_row["selection_objective_value"]),
        "consensus_radius_m": float(best_row["consensus_radius_m"]),
        "min_consensus_weight_fraction": float(best_row["min_consensus_weight_fraction"]),
        "fallback_policy": str(best_row["fallback_policy"]),
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in inputs
        ],
        "metrics": {
            name: _jsonable(best_row[name])
            for name in (
                "pose_mse_m2",
                "pose_rmse_m",
                "pose_mean_m",
                "pose_p95_m",
                "pose_max_m",
                "matched_rows",
                "mean_sequence_mse_m2",
                "max_sequence_mse_m2",
                "p95_sequence_mse_m2",
            )
            if name in best_row.index
        },
        "by_sequence_metrics": _jsonable(best_by_sequence.to_dict(orient="records")),
    }
    grid.attrs["by_sequence"] = by_sequence_grid
    return grid, _jsonable(best)


def write_consensus_grid_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    truth: pd.DataFrame,
    output_dir: Path,
    consensus_radius_m: Iterable[float] = (2.0, 5.0, 10.0),
    min_consensus_weight_fraction: Iterable[float] = (0.0, 0.5),
    fallback_policy: Iterable[str] = ("max-weight", "weighted-mean"),
    max_nearest_time_delta_s: float | None = None,
    selection_objective: str = "pooled-mse",
    sequence_objective_weight: float = 0.25,
    write_best_submission: bool = False,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
) -> dict[str, Path]:
    """Write consensus grid, by-sequence diagnostics, best config, and optional ZIP."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_list = list(estimate_inputs)
    grid, best = search_track5_consensus_ensemble_grid(
        input_list,
        template=template,
        truth=truth,
        consensus_radius_m=consensus_radius_m,
        min_consensus_weight_fraction=min_consensus_weight_fraction,
        fallback_policy=fallback_policy,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        selection_objective=selection_objective,
        sequence_objective_weight=sequence_objective_weight,
    )
    by_sequence = pd.DataFrame(grid.attrs.get("by_sequence", pd.DataFrame()))
    paths = {
        "grid_csv": output / CONSENSUS_GRID_CSV,
        "grid_by_sequence_csv": output / CONSENSUS_GRID_BY_SEQUENCE_CSV,
        "best_config_json": output / BEST_CONFIG_JSON,
    }
    grid.to_csv(paths["grid_csv"], index=False)
    by_sequence.to_csv(paths["grid_by_sequence_csv"], index=False)
    paths["best_config_json"].write_text(json.dumps(_jsonable(best), indent=2), encoding="utf-8")
    if write_best_submission:
        best_paths = write_track5_consensus_ensemble_outputs(
            estimate_inputs=input_list,
            template=template,
            output_dir=output / BEST_OUTPUT_DIR,
            class_map=class_map or {},
            default_classification=default_classification,
            consensus_radius_m=float(best["consensus_radius_m"]),
            fallback_policy=str(best["fallback_policy"]),
            min_consensus_weight_fraction=float(best["min_consensus_weight_fraction"]),
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        paths.update({f"best_{name}": path for name, path in best_paths.items()})
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-consensus-ensemble-grid",
        description="select Track 5 consensus-ensemble settings on a labeled split",
    )
    parser.add_argument(
        "--estimate-csv",
        action="append",
        default=[],
        metavar="LABEL=PATH[@WEIGHT]",
        help="estimate trajectory to include; may be repeated",
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--consensus-radius-m", default="2,5,10")
    parser.add_argument("--min-consensus-weight-fraction", default="0,0.5")
    parser.add_argument("--fallback-policy", action="append", default=[])
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument(
        "--selection-objective",
        choices=SELECTION_OBJECTIVES,
        default="pooled-mse",
    )
    parser.add_argument("--sequence-objective-weight", type=float, default=0.25)
    parser.add_argument("--write-best-submission", action="store_true")
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv")
    estimates = [parse_estimate_spec(value) for value in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    truth = load_evaluation_truth_file(args.truth_csv).rows
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_consensus_grid_outputs(
        estimate_inputs=estimates,
        template=template,
        truth=truth,
        output_dir=args.output_dir,
        consensus_radius_m=_parse_float_list(args.consensus_radius_m),
        min_consensus_weight_fraction=_parse_float_list(args.min_consensus_weight_fraction),
        fallback_policy=args.fallback_policy or FALLBACK_POLICIES,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        selection_objective=args.selection_objective,
        sequence_objective_weight=args.sequence_objective_weight,
        write_best_submission=args.write_best_submission,
        class_map=class_map,
        default_classification=args.default_classification,
    )
    print("mmuad_track5_consensus_ensemble_grid=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).replace(";", ",").split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def _float_grid(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if not out:
        raise ValueError(f"{name} must contain at least one value")
    if not all(np.isfinite(value) for value in out):
        raise ValueError(f"{name} values must be finite")
    return out


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(numeric.to_numpy(float)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
