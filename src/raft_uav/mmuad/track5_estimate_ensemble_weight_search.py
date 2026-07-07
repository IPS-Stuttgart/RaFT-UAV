"""Train-fold weight search for MMUAD Track 5 estimate ensembles.

The estimate-level ensemble CLI accepts explicit weights.  This module provides a
truth-aware development helper to choose those weights on a labeled split (for
example train folds), then writes a JSON config that can be reused unchanged for
validation or hidden-test submissions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import build_track5_estimate_ensemble
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_estimate_ensemble import write_track5_estimate_ensemble_outputs

WEIGHT_GRID_CSV = "mmuad_track5_ensemble_weight_grid.csv"
WEIGHT_GRID_BY_SEQUENCE_CSV = "mmuad_track5_ensemble_weight_grid_by_sequence.csv"
BEST_WEIGHTS_JSON = "mmuad_track5_ensemble_best_weights.json"
BEST_OUTPUT_DIR = "best_weighted_ensemble"
SELECTION_OBJECTIVES = (
    "pooled-mse",
    "mean-sequence-mse",
    "max-sequence-mse",
    "pooled-plus-max-sequence-mse",
)


def search_track5_estimate_ensemble_weights(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    weight_step: float = 0.1,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
    max_nearest_time_delta_s: float | None = None,
    selection_objective: str = "pooled-mse",
    sequence_objective_weight: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate simplex weight grid and return grid rows plus best config.

    The returned grid carries a ``by_sequence`` attribute containing the same
    weight grid expanded by sequence.  ``selection_objective`` chooses the row
    that is written to the reusable best-weight config.  Sequence-balanced
    objectives are useful for leaderboard work because they avoid selecting a
    weight vector that wins pooled MSE by sacrificing one hard sequence.
    """

    if selection_objective not in SELECTION_OBJECTIVES:
        raise ValueError(
            f"unsupported selection_objective {selection_objective!r}; "
            f"allowed={SELECTION_OBJECTIVES}"
        )
    inputs = tuple(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    if weight_step <= 0.0 or weight_step > 1.0:
        raise ValueError("weight_step must be in (0, 1]")
    loaded = [(item.label, pd.read_csv(item.path), 1.0) for item in inputs]
    truth_rows = _normalize_truth_for_exact_template(truth)
    records: list[dict[str, Any]] = []
    by_sequence_records: list[dict[str, Any]] = []
    for weight_index, weights in enumerate(_simplex_weight_grid(len(inputs), step=float(weight_step))):
        weighted_inputs = [
            (label, rows, float(weight))
            for (label, rows, _), weight in zip(loaded, weights, strict=True)
        ]
        estimates, diagnostics = build_track5_estimate_ensemble(
            weighted_inputs,
            template,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
            aggregation_policy=aggregation_policy,
            trim_fraction=trim_fraction,
        )
        metrics = _score_template_estimates(estimates, truth_rows)
        by_sequence = _score_template_estimates_by_sequence(estimates, truth_rows)
        objective_metrics = _sequence_objective_metrics(by_sequence)
        selection_value = _selection_objective_value(
            metrics,
            objective_metrics,
            selection_objective=selection_objective,
            sequence_objective_weight=float(sequence_objective_weight),
        )
        record: dict[str, Any] = {
            "weight_grid_index": int(weight_index),
            "aggregation_policy": aggregation_policy,
            "trim_fraction": float(trim_fraction),
            "weight_step": float(weight_step),
            "selection_objective": selection_objective,
            "selection_objective_value": selection_value,
            "sequence_objective_weight": float(sequence_objective_weight),
            "valid_input_count_mean": _safe_mean(
                diagnostics.get("valid_input_count", pd.Series(dtype=float))
            ),
            **metrics,
            **objective_metrics,
        }
        weights_payload = {f"weight_{item.label}": float(weight) for item, weight in zip(inputs, weights, strict=True)}
        record.update(weights_payload)
        records.append(record)
        for _, sequence_row in by_sequence.iterrows():
            by_sequence_records.append(
                {
                    "weight_grid_index": int(weight_index),
                    "aggregation_policy": aggregation_policy,
                    "trim_fraction": float(trim_fraction),
                    "weight_step": float(weight_step),
                    "selection_objective": selection_objective,
                    **weights_payload,
                    **sequence_row.to_dict(),
                }
            )
    grid = pd.DataFrame.from_records(records)
    if grid.empty:
        raise ValueError("weight grid produced no rows")
    by_sequence_grid = pd.DataFrame.from_records(by_sequence_records)
    best_row = grid.sort_values(
        ["selection_objective_value", "pose_mse_m2", "pose_p95_m", "pose_max_m"],
        na_position="last",
    ).iloc[0]
    best_weights = {item.label: float(best_row[f"weight_{item.label}"]) for item in inputs}
    best_index = int(best_row["weight_grid_index"])
    best_by_sequence = by_sequence_grid.loc[
        by_sequence_grid.get("weight_grid_index", pd.Series(dtype=int)) == best_index
    ]
    best = {
        "schema": "raft-uav-mmuad-track5-estimate-ensemble-weight-search-v1",
        "aggregation_policy": aggregation_policy,
        "trim_fraction": float(trim_fraction),
        "weight_step": float(weight_step),
        "selection_objective": selection_objective,
        "sequence_objective_weight": float(sequence_objective_weight),
        "selection_objective_value": _jsonable(best_row["selection_objective_value"]),
        "best_weight_grid_index": best_index,
        "weights": best_weights,
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
            )
            if name in best_row.index
        },
        "by_sequence_metrics": _jsonable(best_by_sequence.to_dict(orient="records")),
        "estimate_inputs": [asdict(item) | {"path": str(item.path)} for item in inputs],
    }
    grid.attrs["by_sequence"] = by_sequence_grid
    return grid, _jsonable(best)


def write_weight_search_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    truth: pd.DataFrame,
    output_dir: Path,
    weight_step: float = 0.1,
    aggregation_policy: str = "weighted-mean",
    trim_fraction: float = 0.2,
    max_nearest_time_delta_s: float | None = None,
    write_best_submission: bool = False,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    selection_objective: str = "pooled-mse",
    sequence_objective_weight: float = 0.25,
) -> dict[str, Path]:
    """Run weight search and write grid, best config, by-sequence metrics, and optional ZIP."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_list = list(estimate_inputs)
    grid, best = search_track5_estimate_ensemble_weights(
        input_list,
        template=template,
        truth=truth,
        weight_step=weight_step,
        aggregation_policy=aggregation_policy,
        trim_fraction=trim_fraction,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        selection_objective=selection_objective,
        sequence_objective_weight=sequence_objective_weight,
    )
    by_sequence = pd.DataFrame(grid.attrs.get("by_sequence", pd.DataFrame()))
    paths = {
        "weight_grid_csv": output / WEIGHT_GRID_CSV,
        "weight_grid_by_sequence_csv": output / WEIGHT_GRID_BY_SEQUENCE_CSV,
        "best_weights_json": output / BEST_WEIGHTS_JSON,
    }
    grid.to_csv(paths["weight_grid_csv"], index=False)
    by_sequence.to_csv(paths["weight_grid_by_sequence_csv"], index=False)
    paths["best_weights_json"].write_text(json.dumps(_jsonable(best), indent=2), encoding="utf-8")
    if write_best_submission:
        best_weight_map = best["weights"]
        best_inputs = [
            EstimateInput(item.label, item.path, float(best_weight_map[item.label]))
            for item in input_list
        ]
        best_paths = write_track5_estimate_ensemble_outputs(
            estimate_inputs=best_inputs,
            template=template,
            output_dir=output / BEST_OUTPUT_DIR,
            class_map=class_map or {},
            default_classification=default_classification,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
            aggregation_policy=aggregation_policy,
            trim_fraction=trim_fraction,
        )
        paths.update({f"best_{name}": path for name, path in best_paths.items()})
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-ensemble-weight-search",
        description="select Track 5 estimate-ensemble weights on a labeled split",
    )
    parser.add_argument(
        "--estimate-csv",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="estimate trajectory to include; may be repeated; input weights are ignored",
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--aggregation-policy", default="weighted-mean")
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument(
        "--selection-objective",
        choices=SELECTION_OBJECTIVES,
        default="pooled-mse",
        help="objective used to choose the best weight row written to JSON",
    )
    parser.add_argument(
        "--sequence-objective-weight",
        type=float,
        default=0.25,
        help="weight for the max-sequence term in pooled-plus-max-sequence-mse",
    )
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
    paths = write_weight_search_outputs(
        estimate_inputs=estimates,
        template=template,
        truth=truth,
        output_dir=args.output_dir,
        weight_step=args.weight_step,
        aggregation_policy=args.aggregation_policy,
        trim_fraction=args.trim_fraction,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        write_best_submission=args.write_best_submission,
        class_map=class_map,
        default_classification=args.default_classification,
        selection_objective=args.selection_objective,
        sequence_objective_weight=args.sequence_objective_weight,
    )
    print("mmuad_track5_ensemble_weight_search=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _simplex_weight_grid(count: int, *, step: float) -> list[tuple[float, ...]]:
    if count <= 0:
        return []
    if count == 1:
        return [(1.0,)]
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0, atol=1.0e-9):
        raise ValueError("weight_step must divide 1.0 evenly, e.g. 0.5, 0.25, 0.1")
    raw = _simplex_integer_grid(count, units)
    return [tuple(value / units for value in row) for row in raw]


def _simplex_integer_grid(count: int, total: int) -> list[tuple[int, ...]]:
    if count == 1:
        return [(total,)]
    rows: list[tuple[int, ...]] = []
    for value in range(total + 1):
        for tail in _simplex_integer_grid(count - 1, total - value):
            rows.append((value, *tail))
    return rows


def _normalize_truth_for_exact_template(truth: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(truth).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s", "x_m", "y_m", "z_m"])
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["_time_token"] = _time_token(rows["time_s"])
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite, ["sequence_id", "_time_token", "x_m", "y_m", "z_m"]].copy()


def _score_template_estimates(estimates: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    rows = _merge_template_estimates_to_truth(estimates, truth)
    if rows.empty:
        return _empty_metrics()
    estimated_xyz = rows[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    truth_xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(float)
    finite = np.isfinite(estimated_xyz).all(axis=1) & np.isfinite(truth_xyz).all(axis=1)
    if not finite.any():
        return _empty_metrics()
    return _metrics_from_errors(np.linalg.norm(estimated_xyz[finite] - truth_xyz[finite], axis=1))


def _score_template_estimates_by_sequence(estimates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows = _merge_template_estimates_to_truth(estimates, truth)
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", *list(_empty_metrics())])
    records: list[dict[str, Any]] = []
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        estimated_xyz = group[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
        truth_xyz = group[["x_m", "y_m", "z_m"]].to_numpy(float)
        finite = np.isfinite(estimated_xyz).all(axis=1) & np.isfinite(truth_xyz).all(axis=1)
        if not finite.any():
            metrics = _empty_metrics()
        else:
            metrics = _metrics_from_errors(np.linalg.norm(estimated_xyz[finite] - truth_xyz[finite], axis=1))
        records.append({"sequence_id": str(sequence_id), **metrics})
    return pd.DataFrame.from_records(records)


def _merge_template_estimates_to_truth(estimates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(estimates).copy()
    if rows.empty or truth.empty:
        return pd.DataFrame()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["_time_token"] = _time_token(pd.to_numeric(rows["time_s"], errors="coerce"))
    return rows.merge(truth, on=["sequence_id", "_time_token"], how="inner", suffixes=("", "_truth"))


def _sequence_objective_metrics(by_sequence: pd.DataFrame) -> dict[str, Any]:
    if by_sequence.empty or "pose_mse_m2" not in by_sequence.columns:
        return {
            "mean_sequence_mse_m2": np.nan,
            "max_sequence_mse_m2": np.nan,
            "p95_sequence_mse_m2": np.nan,
        }
    sequence_mse = pd.to_numeric(by_sequence["pose_mse_m2"], errors="coerce")
    sequence_mse = sequence_mse[np.isfinite(sequence_mse.to_numpy(float))]
    if sequence_mse.empty:
        return {
            "mean_sequence_mse_m2": np.nan,
            "max_sequence_mse_m2": np.nan,
            "p95_sequence_mse_m2": np.nan,
        }
    values = sequence_mse.to_numpy(float)
    return {
        "mean_sequence_mse_m2": float(np.mean(values)),
        "max_sequence_mse_m2": float(np.max(values)),
        "p95_sequence_mse_m2": float(np.percentile(values, 95)),
    }


def _selection_objective_value(
    pooled_metrics: dict[str, Any],
    sequence_metrics: dict[str, Any],
    *,
    selection_objective: str,
    sequence_objective_weight: float,
) -> float:
    pooled_mse = float(pooled_metrics.get("pose_mse_m2", np.nan))
    mean_sequence = float(sequence_metrics.get("mean_sequence_mse_m2", np.nan))
    max_sequence = float(sequence_metrics.get("max_sequence_mse_m2", np.nan))
    if selection_objective == "pooled-mse":
        return pooled_mse
    if selection_objective == "mean-sequence-mse":
        return mean_sequence
    if selection_objective == "max-sequence-mse":
        return max_sequence
    if selection_objective == "pooled-plus-max-sequence-mse":
        return pooled_mse + float(sequence_objective_weight) * max_sequence
    raise ValueError(f"unsupported selection_objective: {selection_objective}")


def _metrics_from_errors(errors: np.ndarray) -> dict[str, Any]:
    squared = errors**2
    return {
        "matched_rows": int(len(errors)),
        "pose_mse_m2": float(np.mean(squared)),
        "pose_rmse_m": float(np.sqrt(np.mean(squared))),
        "pose_mean_m": float(np.mean(errors)),
        "pose_p95_m": float(np.percentile(errors, 95)),
        "pose_max_m": float(np.max(errors)),
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "matched_rows": 0,
        "pose_mse_m2": np.nan,
        "pose_rmse_m": np.nan,
        "pose_mean_m": np.nan,
        "pose_p95_m": np.nan,
        "pose_max_m": np.nan,
    }


def _time_token(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    rounded = np.round(numeric.to_numpy(dtype=float, na_value=np.nan), 9)
    rounded[np.isfinite(rounded) & (rounded == 0.0)] = 0.0
    tokens = [f"{value:.9f}" if np.isfinite(value) else "" for value in rounded]
    return pd.Series(tokens, index=values.index, dtype="object")


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(numeric.to_numpy(float)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(name): _jsonable(item) for name, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
