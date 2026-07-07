"""Train/validation weight and policy search for MMUAD Track 5 estimate ensembles.

The upload-time estimate ensemble accepts explicit weights and several robust
aggregation policies.  This companion module evaluates a small simplex grid
against a supplied truth/template file so weights and, optionally, the ensemble
policy can be selected upstream on train folds before a hidden-test submission.
It is a development/selection utility: it uses truth only for scoring the grid,
then writes the best upload-ready ensemble artifact for the same template.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import evaluate_mmaud_results, load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_estimate_ensemble import ENSEMBLE_POLICIES
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import build_track5_estimate_ensemble
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble import write_track5_estimate_ensemble_outputs

GRID_SUMMARY_CSV = "mmuad_track5_estimate_ensemble_weight_grid.csv"
GRID_BY_SEQUENCE_CSV = "mmuad_track5_estimate_ensemble_weight_grid_by_sequence.csv"
GRID_MANIFEST_JSON = "mmuad_track5_estimate_ensemble_weight_grid_manifest.json"
BEST_CONFIG_JSON = "mmuad_track5_estimate_ensemble_best_config.json"
BEST_OUTPUT_DIR = "best_ensemble"


@dataclass(frozen=True)
class EnsembleGridRow:
    """One candidate set of ensemble weights/policy and its score."""

    weights: tuple[float, ...]
    aggregation_policy: str
    trim_fraction: float
    pose_mse: float
    rmse_m: float
    mean_error_m: float
    p95_error_m: float
    max_error_m: float
    class_accuracy: float | None
    matched_count: int


def generate_simplex_weight_grid(
    n_inputs: int,
    *,
    step: float = 0.25,
    include_singletons: bool = True,
) -> list[tuple[float, ...]]:
    """Return non-negative weights that sum to one on a regular grid."""

    if n_inputs <= 0:
        raise ValueError("n_inputs must be positive")
    if not 0.0 < float(step) <= 1.0:
        raise ValueError("step must be in (0, 1]")
    units = int(round(1.0 / float(step)))
    if not np.isclose(units * float(step), 1.0):
        raise ValueError("step must divide 1.0 exactly, e.g. 0.5, 0.25, 0.1")
    grid: list[tuple[float, ...]] = []
    for values in product(range(units + 1), repeat=n_inputs):
        if sum(values) != units:
            continue
        if not include_singletons and sum(value > 0 for value in values) == 1:
            continue
        grid.append(tuple(float(value) / float(units) for value in values))
    return sorted(set(grid), reverse=True)


def evaluate_estimate_ensemble_weight_grid(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    weight_grid: Iterable[tuple[float, ...]],
    class_map_path: Path | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
    aggregation_policies: Iterable[str] = ("weighted-mean",),
    trim_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, ...]]:
    """Score weight/policy grid and return summary, by-sequence rows, and best weights."""

    inputs = tuple(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    policies = _normalize_aggregation_policies(aggregation_policies)
    loaded = [(item.label, read_estimate_csv(item.path), 1.0) for item in inputs]
    class_map = load_sequence_class_map(class_map_path) if class_map_path is not None else {}
    summary_records: list[dict[str, Any]] = []
    sequence_records: list[dict[str, Any]] = []
    best_row: EnsembleGridRow | None = None
    for policy in policies:
        for weights in weight_grid:
            if len(weights) != len(inputs):
                raise ValueError(
                    f"weight vector length {len(weights)} does not match inputs {len(inputs)}"
                )
            weighted_loaded = [
                (label, estimates, float(weight))
                for (label, estimates, _), weight in zip(loaded, weights, strict=True)
            ]
            ensemble, _ = build_track5_estimate_ensemble(
                weighted_loaded,
                template,
                max_nearest_time_delta_s=max_nearest_time_delta_s,
                aggregation_policy=policy,
                trim_fraction=float(trim_fraction),
            )
            results = _ensemble_results_frame(
                ensemble,
                class_map=class_map,
                default_classification=default_classification,
            )
            try:
                evaluation = evaluate_mmaud_results(
                    results,
                    truth,
                    metric_protocol="public-track5",
                    class_map_path=class_map_path,
                )
                row = _grid_row(
                    weights,
                    evaluation,
                    aggregation_policy=policy,
                    trim_fraction=float(trim_fraction),
                )
            except ValueError as exc:
                if "contains no finite trajectory rows" not in str(exc):
                    raise
                evaluation = {}
                row = _failed_grid_row(
                    weights,
                    aggregation_policy=policy,
                    trim_fraction=float(trim_fraction),
                )
            summary_records.append(_summary_record(inputs, row))
            sequence_records.extend(_sequence_records(inputs, row.weights, evaluation, row))
            if best_row is None or _row_sort_key(row) < _row_sort_key(best_row):
                best_row = row
    summary = pd.DataFrame.from_records(summary_records).sort_values(
        ["pose_mse", "p95_error_m", "max_error_m"],
        na_position="last",
    )
    by_sequence = pd.DataFrame.from_records(sequence_records)
    if best_row is None:
        raise ValueError("weight/policy grid produced no rows")
    return summary.reset_index(drop=True), by_sequence, best_row.weights


def write_estimate_ensemble_weight_grid_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    truth: pd.DataFrame,
    weight_grid: Iterable[tuple[float, ...]],
    output_dir: Path,
    class_map_path: Path | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
    aggregation_policies: Iterable[str] = ("weighted-mean",),
    trim_fraction: float = 0.2,
) -> dict[str, Path]:
    """Score a grid and write the best leaderboard-ready ensemble artifact."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inputs = tuple(estimate_inputs)
    summary, by_sequence, best_weights = evaluate_estimate_ensemble_weight_grid(
        inputs,
        template=template,
        truth=truth,
        weight_grid=weight_grid,
        class_map_path=class_map_path,
        default_classification=default_classification,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        aggregation_policies=aggregation_policies,
        trim_fraction=trim_fraction,
    )
    summary_csv = output / GRID_SUMMARY_CSV
    by_sequence_csv = output / GRID_BY_SEQUENCE_CSV
    manifest_json = output / GRID_MANIFEST_JSON
    best_config_json = output / BEST_CONFIG_JSON
    summary.to_csv(summary_csv, index=False)
    by_sequence.to_csv(by_sequence_csv, index=False)
    best_summary = summary.iloc[0].to_dict() if not summary.empty else {}
    best_policy = str(best_summary.get("aggregation_policy", "weighted-mean"))
    best_trim_fraction = float(best_summary.get("trim_fraction", trim_fraction))
    best_config = _best_weight_config(
        inputs,
        best_weights,
        aggregation_policy=best_policy,
        trim_fraction=best_trim_fraction,
        best_summary=best_summary,
        class_map_path=class_map_path,
        default_classification=default_classification,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    best_config_json.write_text(json.dumps(_jsonable(best_config), indent=2), encoding="utf-8")
    best_inputs = [
        EstimateInput(label=item.label, path=item.path, weight=float(weight))
        for item, weight in zip(inputs, best_weights, strict=True)
    ]
    class_map = load_sequence_class_map(class_map_path) if class_map_path is not None else {}
    best_paths = write_track5_estimate_ensemble_outputs(
        estimate_inputs=best_inputs,
        template=template,
        output_dir=output / BEST_OUTPUT_DIR,
        class_map=class_map,
        default_classification=default_classification,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        aggregation_policy=best_policy,
        trim_fraction=best_trim_fraction,
    )
    manifest = {
        "schema": "raft-uav-mmuad-track5-estimate-ensemble-weight-grid-v3",
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path)} for item in inputs
        ],
        "class_map_path": str(class_map_path) if class_map_path is not None else None,
        "default_classification": str(default_classification),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "aggregation_policies": list(_normalize_aggregation_policies(aggregation_policies)),
        "trim_fraction": float(trim_fraction),
        "grid_row_count": int(len(summary)),
        "best_weights": list(best_weights),
        "best_weight_config_json": str(best_config_json),
        "best_aggregation_policy": best_policy,
        "best_trim_fraction": best_trim_fraction,
        "best": best_summary,
        "paths": {
            "summary_csv": str(summary_csv),
            "by_sequence_csv": str(by_sequence_csv),
            "best_config_json": str(best_config_json),
            "best_output_dir": str(output / BEST_OUTPUT_DIR),
            **{f"best_{name}": str(path) for name, path in best_paths.items()},
        },
    }
    manifest_json.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return {
        "summary_csv": summary_csv,
        "by_sequence_csv": by_sequence_csv,
        "manifest_json": manifest_json,
        "best_config_json": best_config_json,
        **{f"best_{name}": path for name, path in best_paths.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-estimate-ensemble-grid",
        description="select Track 5 estimate-ensemble weights/policy on a scored template",
    )
    parser.add_argument(
        "--estimate-csv",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="estimate trajectory to include; may be repeated; input weights are ignored",
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--exclude-singletons", action="store_true")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument(
        "--aggregation-policy",
        choices=(*ENSEMBLE_POLICIES, "grid"),
        default="weighted-mean",
        help="ensemble aggregation policy to score; use grid to score all available policies",
    )
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH")
    inputs = tuple(parse_estimate_spec(value) for value in args.estimate_csv)
    template = load_official_track5_template_file(args.template)
    truth = load_evaluation_truth_file(args.truth).rows
    weight_grid = generate_simplex_weight_grid(
        len(inputs),
        step=float(args.step),
        include_singletons=not bool(args.exclude_singletons),
    )
    aggregation_policies = ENSEMBLE_POLICIES if args.aggregation_policy == "grid" else (args.aggregation_policy,)
    paths = write_estimate_ensemble_weight_grid_outputs(
        estimate_inputs=inputs,
        template=template,
        truth=truth,
        weight_grid=weight_grid,
        output_dir=args.output_dir,
        class_map_path=args.class_map,
        default_classification=args.default_classification,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        aggregation_policies=aggregation_policies,
        trim_fraction=float(args.trim_fraction),
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_estimate_ensemble_grid=ok")
    print(f"grid_row_count={manifest['grid_row_count']}")
    print(f"best_weights={manifest['best_weights']}")
    print(f"best_aggregation_policy={manifest['best_aggregation_policy']}")
    print(f"summary_csv={paths['summary_csv']}")
    print(f"manifest_json={paths['manifest_json']}")
    print(f"best_config_json={paths['best_config_json']}")
    print(f"best_official_zip={paths['best_official_zip']}")
    return 0


def _ensemble_results_frame(
    ensemble: pd.DataFrame,
    *,
    class_map: dict[str, str],
    default_classification: int | str,
) -> pd.DataFrame:
    rows = ensemble.copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["timestamp"] = pd.to_numeric(rows["time_s"], errors="coerce")
    rows["x"] = pd.to_numeric(rows["state_x_m"], errors="coerce")
    rows["y"] = pd.to_numeric(rows["state_y_m"], errors="coerce")
    rows["z"] = pd.to_numeric(rows["state_z_m"], errors="coerce")
    rows["uav_type"] = rows["sequence_id"].map(class_map).fillna(str(default_classification))
    rows["score"] = 1.0
    return rows[["sequence_id", "timestamp", "x", "y", "z", "uav_type", "score"]]


def _grid_row(
    weights: tuple[float, ...],
    evaluation: dict[str, Any],
    *,
    aggregation_policy: str,
    trim_fraction: float,
) -> EnsembleGridRow:
    summary = evaluation.get("summary", evaluation)
    pooled = summary.get("pooled", summary)
    return EnsembleGridRow(
        weights=tuple(float(weight) for weight in weights),
        aggregation_policy=str(aggregation_policy),
        trim_fraction=float(trim_fraction),
        pose_mse=float(
            pooled.get("pose_mse_loss_m2", pooled.get("mean_square_loss_m2", np.nan))
        ),
        rmse_m=float(pooled.get("rmse_3d_m", np.nan)),
        mean_error_m=float(pooled.get("mean_3d_m", np.nan)),
        p95_error_m=float(pooled.get("p95_3d_m", np.nan)),
        max_error_m=float(pooled.get("max_3d_m", np.nan)),
        class_accuracy=_optional_float(pooled.get("uav_type_accuracy")),
        matched_count=int(summary.get("matched_count", pooled.get("count", 0)) or 0),
    )


def _failed_grid_row(
    weights: tuple[float, ...],
    *,
    aggregation_policy: str,
    trim_fraction: float,
) -> EnsembleGridRow:
    return EnsembleGridRow(
        weights=tuple(float(weight) for weight in weights),
        aggregation_policy=str(aggregation_policy),
        trim_fraction=float(trim_fraction),
        pose_mse=float("inf"),
        rmse_m=float("inf"),
        mean_error_m=float("inf"),
        p95_error_m=float("inf"),
        max_error_m=float("inf"),
        class_accuracy=None,
        matched_count=0,
    )


def _summary_record(inputs: tuple[EstimateInput, ...], row: EnsembleGridRow) -> dict[str, Any]:
    record: dict[str, Any] = {
        "weights": ";".join(str(weight) for weight in row.weights),
        "aggregation_policy": row.aggregation_policy,
        "trim_fraction": row.trim_fraction,
        "pose_mse": row.pose_mse,
        "rmse_m": row.rmse_m,
        "mean_error_m": row.mean_error_m,
        "p95_error_m": row.p95_error_m,
        "max_error_m": row.max_error_m,
        "class_accuracy": row.class_accuracy,
        "matched_count": row.matched_count,
    }
    for item, weight in zip(inputs, row.weights, strict=True):
        record[f"weight_{item.label}"] = float(weight)
    return record


def _sequence_records(
    inputs: tuple[EstimateInput, ...],
    weights: tuple[float, ...],
    evaluation: dict[str, Any],
    row: EnsembleGridRow,
) -> list[dict[str, Any]]:
    rows = pd.DataFrame(evaluation.get("rows", pd.DataFrame()))
    if rows.empty or "sequence_id" not in rows.columns:
        return []
    records: list[dict[str, Any]] = []
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        matched = group.loc[group["matched"].astype(bool)] if "matched" in group else group
        errors = pd.to_numeric(matched.get("error_3d_m", pd.Series(dtype=float)), errors="coerce")
        errors = errors[np.isfinite(errors.to_numpy(float))]
        if errors.empty:
            continue
        record: dict[str, Any] = {
            "sequence_id": str(sequence_id),
            "weights": ";".join(str(weight) for weight in weights),
            "aggregation_policy": row.aggregation_policy,
            "trim_fraction": row.trim_fraction,
            "pose_mse": float(np.mean(errors.to_numpy(float) ** 2)),
            "rmse_m": float(np.sqrt(np.mean(errors.to_numpy(float) ** 2))),
            "mean_error_m": float(errors.mean()),
            "p95_error_m": float(np.percentile(errors, 95)),
            "max_error_m": float(errors.max()),
            "matched_count": int(len(errors)),
        }
        for item, weight in zip(inputs, weights, strict=True):
            record[f"weight_{item.label}"] = float(weight)
        records.append(record)
    return records


def _best_weight_config(
    inputs: tuple[EstimateInput, ...],
    weights: tuple[float, ...],
    *,
    aggregation_policy: str,
    trim_fraction: float,
    best_summary: dict[str, Any],
    class_map_path: Path | None,
    default_classification: int | str,
    max_nearest_time_delta_s: float | None,
) -> dict[str, Any]:
    return {
        "schema": "raft-uav-mmuad-track5-estimate-ensemble-config-v1",
        "weights": {
            item.label: float(weight) for item, weight in zip(inputs, weights, strict=True)
        },
        "aggregation_policy": aggregation_policy,
        "trim_fraction": float(trim_fraction),
        "class_map_path": str(class_map_path) if class_map_path is not None else None,
        "default_classification": str(default_classification),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path)} for item in inputs
        ],
        "selection_metric": "pose_mse,p95_error_m,max_error_m",
        "best": best_summary,
    }


def _row_sort_key(row: EnsembleGridRow) -> tuple[float, float, float]:
    return (
        _finite_sort_metric(row.pose_mse),
        _finite_sort_metric(row.p95_error_m),
        _finite_sort_metric(row.max_error_m),
    )


def _finite_sort_metric(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else float("inf")


def _normalize_aggregation_policies(values: Iterable[str]) -> tuple[str, ...]:
    policies = tuple(dict.fromkeys(str(value) for value in values))
    if not policies:
        raise ValueError("at least one aggregation policy is required")
    unsupported = [policy for policy in policies if policy not in ENSEMBLE_POLICIES]
    if unsupported:
        raise ValueError(f"unsupported aggregation policies: {unsupported}")
    return policies


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
