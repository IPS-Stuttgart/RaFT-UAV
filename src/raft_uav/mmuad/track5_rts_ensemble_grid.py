"""Train-side grid search for MMUAD Track 5 RTS ensemble parameters.

The RTS ensemble is inference-safe once its smoothing parameters are fixed, but
its process noise, measurement noise, and disagreement inflation should be chosen
on labeled train folds rather than tuned on a hidden Codabench submission.  This
module evaluates a small explicit parameter grid against labeled Track 5 truth,
writes the ranked table, and can optionally export the best-parameter submission
artifact for a public-validation or train-fold run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput, parse_estimate_spec
from raft_uav.mmuad.track5_rts_ensemble import build_track5_rts_ensemble
from raft_uav.mmuad.track5_rts_ensemble import write_track5_rts_ensemble_outputs

GRID_RESULTS_CSV = "mmuad_track5_rts_ensemble_grid.csv"
GRID_BEST_JSON = "mmuad_track5_rts_ensemble_grid_best.json"
GRID_SUMMARY_JSON = "mmuad_track5_rts_ensemble_grid_summary.json"
BEST_OUTPUT_DIR = "best_rts_ensemble"
DEFAULT_MEASUREMENT_SIGMA_GRID = (5.0, 10.0, 15.0, 20.0)
DEFAULT_PROCESS_ACCEL_GRID = (1.0, 3.0, 5.0, 8.0, 12.0)
DEFAULT_SPREAD_VARIANCE_SCALE_GRID = (0.0, 0.5, 1.0, 2.0)


def run_track5_rts_ensemble_grid_search(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    measurement_sigma_grid: Iterable[float] = DEFAULT_MEASUREMENT_SIGMA_GRID,
    process_accel_grid: Iterable[float] = DEFAULT_PROCESS_ACCEL_GRID,
    spread_variance_scale_grid: Iterable[float] = DEFAULT_SPREAD_VARIANCE_SCALE_GRID,
    initial_position_std_m: float = 100.0,
    initial_velocity_std_mps: float = 25.0,
    max_nearest_time_delta_s: float | None = None,
    score_time_tolerance_s: float = 1.0e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate RTS ensemble smoothing parameters and return ranked rows."""

    input_list = list(estimate_inputs)
    if not input_list:
        raise ValueError("at least one estimate input is required")
    loaded_inputs = [
        (item.label, read_estimate_csv(item.path), float(item.weight)) for item in input_list
    ]
    truth_rows = _normalize_truth(truth)
    if truth_rows.empty:
        raise ValueError("truth contains no finite rows")
    grid_records: list[dict[str, Any]] = []
    for measurement_sigma_m in _positive_grid(measurement_sigma_grid, "measurement_sigma_grid"):
        for process_accel_std_mps2 in _nonnegative_grid(process_accel_grid, "process_accel_grid"):
            for spread_variance_scale in _nonnegative_grid(
                spread_variance_scale_grid,
                "spread_variance_scale_grid",
            ):
                estimates, diagnostics = build_track5_rts_ensemble(
                    loaded_inputs,
                    template,
                    measurement_sigma_m=float(measurement_sigma_m),
                    process_accel_std_mps2=float(process_accel_std_mps2),
                    initial_position_std_m=float(initial_position_std_m),
                    initial_velocity_std_mps=float(initial_velocity_std_mps),
                    spread_variance_scale=float(spread_variance_scale),
                    max_nearest_time_delta_s=max_nearest_time_delta_s,
                )
                metrics = _score_estimates(
                    estimates,
                    truth_rows,
                    score_time_tolerance_s=float(score_time_tolerance_s),
                )
                grid_records.append(
                    {
                        "measurement_sigma_m": float(measurement_sigma_m),
                        "process_accel_std_mps2": float(process_accel_std_mps2),
                        "spread_variance_scale": float(spread_variance_scale),
                        "initial_position_std_m": float(initial_position_std_m),
                        "initial_velocity_std_mps": float(initial_velocity_std_mps),
                        "max_nearest_time_delta_s": max_nearest_time_delta_s,
                        "diagnostic_valid_input_count_mean": _safe_mean(
                            diagnostics.get("valid_input_count", pd.Series(dtype=float))
                        ),
                        "diagnostic_input_spread_m_mean": _safe_mean(
                            diagnostics.get("input_spread_m", pd.Series(dtype=float))
                        ),
                        **metrics,
                    }
                )
    grid = pd.DataFrame.from_records(grid_records)
    if grid.empty:
        raise ValueError("parameter grid produced no rows")
    grid = grid.sort_values(
        ["pose_mse_m2", "rmse_3d_m", "mean_3d_m", "process_accel_std_mps2"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    best = _jsonable(
        {
            "schema": "raft-uav-mmuad-track5-rts-ensemble-grid-best-v1",
            "selection_metric": "pose_mse_m2",
            "best": grid.iloc[0].to_dict(),
            "estimate_inputs": [asdict(item) | {"path": str(item.path)} for item in input_list],
        }
    )
    return grid, best


def write_track5_rts_ensemble_grid_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    truth: pd.DataFrame,
    output_dir: Path,
    measurement_sigma_grid: Iterable[float] = DEFAULT_MEASUREMENT_SIGMA_GRID,
    process_accel_grid: Iterable[float] = DEFAULT_PROCESS_ACCEL_GRID,
    spread_variance_scale_grid: Iterable[float] = DEFAULT_SPREAD_VARIANCE_SCALE_GRID,
    initial_position_std_m: float = 100.0,
    initial_velocity_std_mps: float = 25.0,
    max_nearest_time_delta_s: float | None = None,
    score_time_tolerance_s: float = 1.0e-6,
    write_best_submission: bool = False,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
) -> dict[str, Path]:
    """Write ranked grid results and optional best-parameter Track 5 outputs."""

    input_list = list(estimate_inputs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grid, best = run_track5_rts_ensemble_grid_search(
        input_list,
        template=template,
        truth=truth,
        measurement_sigma_grid=measurement_sigma_grid,
        process_accel_grid=process_accel_grid,
        spread_variance_scale_grid=spread_variance_scale_grid,
        initial_position_std_m=initial_position_std_m,
        initial_velocity_std_mps=initial_velocity_std_mps,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        score_time_tolerance_s=score_time_tolerance_s,
    )
    paths = {
        "grid_csv": output / GRID_RESULTS_CSV,
        "best_json": output / GRID_BEST_JSON,
        "summary_json": output / GRID_SUMMARY_JSON,
    }
    grid.to_csv(paths["grid_csv"], index=False)
    paths["best_json"].write_text(json.dumps(best, indent=2), encoding="utf-8")
    summary = {
        "schema": "raft-uav-mmuad-track5-rts-ensemble-grid-summary-v1",
        "grid_row_count": int(len(grid)),
        "best": best["best"],
        "write_best_submission": bool(write_best_submission),
    }
    paths["summary_json"].write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    if write_best_submission:
        best_row = best["best"]
        best_paths = write_track5_rts_ensemble_outputs(
            estimate_inputs=input_list,
            template=template,
            output_dir=output / BEST_OUTPUT_DIR,
            class_map=class_map or {},
            default_classification=default_classification,
            measurement_sigma_m=float(best_row["measurement_sigma_m"]),
            process_accel_std_mps2=float(best_row["process_accel_std_mps2"]),
            initial_position_std_m=float(best_row["initial_position_std_m"]),
            initial_velocity_std_mps=float(best_row["initial_velocity_std_mps"]),
            spread_variance_scale=float(best_row["spread_variance_scale"]),
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        paths.update({f"best_{name}": path for name, path in best_paths.items()})
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-rts-ensemble-grid",
        description="grid-search train-side parameters for Track 5 RTS estimate ensembles",
    )
    parser.add_argument("--estimate-csv", action="append", default=[], metavar="LABEL=PATH[@WEIGHT]")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--measurement-sigma-grid", default=_grid_to_text(DEFAULT_MEASUREMENT_SIGMA_GRID))
    parser.add_argument("--process-accel-grid", default=_grid_to_text(DEFAULT_PROCESS_ACCEL_GRID))
    parser.add_argument(
        "--spread-variance-scale-grid",
        default=_grid_to_text(DEFAULT_SPREAD_VARIANCE_SCALE_GRID),
    )
    parser.add_argument("--initial-position-std-m", type=float, default=100.0)
    parser.add_argument("--initial-velocity-std-mps", type=float, default=25.0)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--score-time-tolerance-s", type=float, default=1.0e-6)
    parser.add_argument("--write-best-submission", action="store_true")
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    estimate_inputs = [parse_estimate_spec(value) for value in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    truth = load_evaluation_truth_file(args.truth_csv).rows
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_rts_ensemble_grid_outputs(
        estimate_inputs=estimate_inputs,
        template=template,
        truth=truth,
        output_dir=args.output_dir,
        measurement_sigma_grid=_parse_grid(args.measurement_sigma_grid),
        process_accel_grid=_parse_grid(args.process_accel_grid),
        spread_variance_scale_grid=_parse_grid(args.spread_variance_scale_grid),
        initial_position_std_m=float(args.initial_position_std_m),
        initial_velocity_std_mps=float(args.initial_velocity_std_mps),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        score_time_tolerance_s=float(args.score_time_tolerance_s),
        write_best_submission=bool(args.write_best_submission),
        class_map=class_map,
        default_classification=args.default_classification,
    )
    best = json.loads(paths["best_json"].read_text(encoding="utf-8"))["best"]
    print("mmuad_track5_rts_ensemble_grid=ok")
    print(f"best_pose_mse_m2={best['pose_mse_m2']}")
    print(f"best_measurement_sigma_m={best['measurement_sigma_m']}")
    print(f"best_process_accel_std_mps2={best['process_accel_std_mps2']}")
    print(f"best_spread_variance_scale={best['spread_variance_scale']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _score_estimates(
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    score_time_tolerance_s: float,
) -> dict[str, Any]:
    estimate_rows = _normalize_estimates(estimates)
    truth_by_sequence = {
        str(sequence): group.sort_values("time_s").reset_index(drop=True)
        for sequence, group in truth.groupby("sequence_id", sort=True)
    }
    errors: list[float] = []
    for _, row in estimate_rows.iterrows():
        truth_group = truth_by_sequence.get(str(row["sequence_id"]))
        if truth_group is None or truth_group.empty:
            continue
        times = truth_group["time_s"].to_numpy(float)
        index = int(np.argmin(np.abs(times - float(row["time_s"]))))
        delta = abs(float(times[index]) - float(row["time_s"]))
        if delta > float(score_time_tolerance_s):
            continue
        truth_xyz = truth_group.iloc[index][["x_m", "y_m", "z_m"]].to_numpy(float)
        estimate_xyz = row[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
        errors.append(float(np.linalg.norm(estimate_xyz - truth_xyz)))
    if not errors:
        return {
            "matched_row_count": 0,
            "pose_mse_m2": float("inf"),
            "rmse_3d_m": float("inf"),
            "mean_3d_m": float("inf"),
            "p95_3d_m": float("inf"),
            "max_3d_m": float("inf"),
        }
    values = np.asarray(errors, dtype=float)
    return {
        "matched_row_count": int(len(values)),
        "pose_mse_m2": float(np.mean(values**2)),
        "rmse_3d_m": float(np.sqrt(np.mean(values**2))),
        "mean_3d_m": float(np.mean(values)),
        "p95_3d_m": float(np.percentile(values, 95)),
        "max_3d_m": float(np.max(values)),
    }


def _normalize_estimates(estimates: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(estimates).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"])
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite].reset_index(drop=True)


def _normalize_truth(truth: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(truth).copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite].reset_index(drop=True)


def _parse_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def _grid_to_text(values: Iterable[float]) -> str:
    return ",".join(str(float(value)) for value in values)


def _positive_grid(values: Iterable[float], name: str) -> tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if not out or any((not np.isfinite(value)) or value <= 0.0 for value in out):
        raise ValueError(f"{name} must contain positive finite values")
    return out


def _nonnegative_grid(values: Iterable[float], name: str) -> tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if not out or any((not np.isfinite(value)) or value < 0.0 for value in out):
        raise ValueError(f"{name} must contain non-negative finite values")
    return out


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
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
