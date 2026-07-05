"""Train-fold weight search for Track 5 uncertainty-weighted ensembles.

The uncertainty ensemble already uses per-row inverse-variance weights from each
pose pipeline's predicted sigma column.  This helper searches the remaining
global pipeline weights on a labeled split, then writes a reusable JSON config
for validation or hidden-test Codabench submissions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput, parse_estimate_spec
from raft_uav.mmuad.track5_uncertainty_ensemble import build_track5_uncertainty_ensemble
from raft_uav.mmuad.track5_uncertainty_ensemble import write_track5_uncertainty_ensemble_outputs

WEIGHT_GRID_CSV = "mmuad_track5_uncertainty_ensemble_weight_grid.csv"
BEST_WEIGHTS_JSON = "mmuad_track5_uncertainty_ensemble_best_weights.json"
BEST_OUTPUT_DIR = "best_uncertainty_weighted_ensemble"


def search_track5_uncertainty_ensemble_weights(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    uncertainty_column: str = "predicted_sigma_m",
    fallback_sigma_m: float = 30.0,
    sigma_min_m: float = 1.0,
    sigma_max_m: float = 100.0,
    max_nearest_time_delta_s: float | None = None,
    weight_step: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate global-weight simplex grid for uncertainty ensembles."""

    inputs = tuple(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    if weight_step <= 0.0 or weight_step > 1.0:
        raise ValueError("weight_step must be in (0, 1]")
    truth_rows = _normalize_truth_for_exact_template(truth)
    records: list[dict[str, Any]] = []
    for weights in _simplex_weight_grid(len(inputs), step=float(weight_step)):
        weighted_inputs = [
            EstimateInput(item.label, item.path, float(weight))
            for item, weight in zip(inputs, weights, strict=True)
        ]
        estimates, diagnostics = build_track5_uncertainty_ensemble(
            weighted_inputs,
            template=template,
            uncertainty_column=uncertainty_column,
            fallback_sigma_m=float(fallback_sigma_m),
            sigma_min_m=float(sigma_min_m),
            sigma_max_m=float(sigma_max_m),
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        metrics = _score_template_estimates(estimates, truth_rows)
        record: dict[str, Any] = {
            "weight_step": float(weight_step),
            "uncertainty_column": uncertainty_column,
            "fallback_sigma_m": float(fallback_sigma_m),
            "sigma_min_m": float(sigma_min_m),
            "sigma_max_m": float(sigma_max_m),
            "effective_sigma_mean_m": _safe_mean(
                diagnostics.get("effective_sigma_m", pd.Series(dtype=float))
            ),
            "position_spread_mean_m": _safe_mean(
                diagnostics.get("position_spread_m", pd.Series(dtype=float))
            ),
            **metrics,
        }
        for item, weight in zip(inputs, weights, strict=True):
            record[f"weight_{item.label}"] = float(weight)
        records.append(record)
    grid = pd.DataFrame.from_records(records)
    if grid.empty:
        raise ValueError("weight grid produced no rows")
    best_row = grid.sort_values(
        ["pose_mse_m2", "pose_p95_m", "pose_max_m"],
        na_position="last",
    ).iloc[0]
    best_weights = {item.label: float(best_row[f"weight_{item.label}"]) for item in inputs}
    best = {
        "schema": "raft-uav-mmuad-track5-uncertainty-ensemble-weight-search-v1",
        "weights": best_weights,
        "weight_step": float(weight_step),
        "uncertainty_column": uncertainty_column,
        "fallback_sigma_m": float(fallback_sigma_m),
        "sigma_min_m": float(sigma_min_m),
        "sigma_max_m": float(sigma_max_m),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "metrics": {
            key: _jsonable(best_row[key])
            for key in (
                "pose_mse_m2",
                "pose_rmse_m",
                "pose_mean_m",
                "pose_p95_m",
                "pose_max_m",
                "matched_rows",
            )
            if key in best_row.index
        },
        "estimate_inputs": [asdict(item) | {"path": str(item.path)} for item in inputs],
    }
    return grid, _jsonable(best)


def write_uncertainty_weight_search_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    truth: pd.DataFrame,
    output_dir: Path,
    uncertainty_column: str = "predicted_sigma_m",
    fallback_sigma_m: float = 30.0,
    sigma_min_m: float = 1.0,
    sigma_max_m: float = 100.0,
    max_nearest_time_delta_s: float | None = None,
    weight_step: float = 0.1,
    write_best_submission: bool = False,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
) -> dict[str, Path]:
    """Run uncertainty-ensemble weight search and write artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_list = list(estimate_inputs)
    grid, best = search_track5_uncertainty_ensemble_weights(
        input_list,
        template=template,
        truth=truth,
        uncertainty_column=uncertainty_column,
        fallback_sigma_m=fallback_sigma_m,
        sigma_min_m=sigma_min_m,
        sigma_max_m=sigma_max_m,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        weight_step=weight_step,
    )
    paths = {
        "weight_grid_csv": output / WEIGHT_GRID_CSV,
        "best_weights_json": output / BEST_WEIGHTS_JSON,
    }
    grid.to_csv(paths["weight_grid_csv"], index=False)
    paths["best_weights_json"].write_text(json.dumps(_jsonable(best), indent=2), encoding="utf-8")
    if write_best_submission:
        best_weight_map = best["weights"]
        best_inputs = [
            EstimateInput(item.label, item.path, float(best_weight_map[item.label]))
            for item in input_list
        ]
        best_paths = write_track5_uncertainty_ensemble_outputs(
            estimate_inputs=best_inputs,
            template=template,
            output_dir=output / BEST_OUTPUT_DIR,
            class_map=class_map or {},
            default_classification=default_classification,
            uncertainty_column=uncertainty_column,
            fallback_sigma_m=fallback_sigma_m,
            sigma_min_m=sigma_min_m,
            sigma_max_m=sigma_max_m,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        paths.update({f"best_{name}": path for name, path in best_paths.items()})
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-uncertainty-ensemble-weight-search",
        description="select global weights for uncertainty-weighted Track 5 estimate ensembles",
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
    parser.add_argument("--uncertainty-column", default="predicted_sigma_m")
    parser.add_argument("--fallback-sigma-m", type=float, default=30.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=100.0)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--weight-step", type=float, default=0.1)
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
    paths = write_uncertainty_weight_search_outputs(
        estimate_inputs=estimates,
        template=template,
        truth=truth,
        output_dir=args.output_dir,
        uncertainty_column=args.uncertainty_column,
        fallback_sigma_m=float(args.fallback_sigma_m),
        sigma_min_m=float(args.sigma_min_m),
        sigma_max_m=float(args.sigma_max_m),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        weight_step=float(args.weight_step),
        write_best_submission=args.write_best_submission,
        class_map=class_map,
        default_classification=args.default_classification,
    )
    print("mmuad_track5_uncertainty_ensemble_weight_search=ok")
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
    rows["_time_key"] = _time_key(rows["time_s"])
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite, ["sequence_id", "_time_key", "x_m", "y_m", "z_m"]].copy()


def _score_template_estimates(estimates: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    rows = pd.DataFrame(estimates).copy()
    if rows.empty or truth.empty:
        return _empty_metrics()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["_time_key"] = _time_key(pd.to_numeric(rows["time_s"], errors="coerce"))
    merged = rows.merge(truth, on=["sequence_id", "_time_key"], how="inner", suffixes=("", "_truth"))
    if merged.empty:
        return _empty_metrics()
    estimated_xyz = merged[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    truth_xyz = merged[["x_m", "y_m", "z_m"]].to_numpy(float)
    finite = np.isfinite(estimated_xyz).all(axis=1) & np.isfinite(truth_xyz).all(axis=1)
    if not finite.any():
        return _empty_metrics()
    errors = np.linalg.norm(estimated_xyz[finite] - truth_xyz[finite], axis=1)
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


def _time_key(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").round(9).astype(str)


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(numeric.to_numpy(float)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
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
