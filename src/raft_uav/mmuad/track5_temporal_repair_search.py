"""Parameter search for Track 5 temporal spike repair on labeled splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission
from raft_uav.mmuad.track5_temporal_repair import repair_track5_temporal_spikes
from raft_uav.mmuad.track5_temporal_repair import write_track5_temporal_repair_outputs

SEARCH_GRID_CSV = "mmuad_track5_temporal_repair_search_grid.csv"
BEST_CONFIG_JSON = "mmuad_track5_temporal_repair_best_config.json"
BEST_OUTPUT_DIR = "best_temporal_repair"


def search_track5_temporal_repair_parameters(
    submission: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_speed_grid: Iterable[float] = (40.0, 60.0, 80.0, 120.0),
    interpolation_residual_grid: Iterable[float] = (5.0, 10.0, 20.0, 30.0),
    iterations_grid: Iterable[int] = (1, 2, 3),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth_rows = _normalize_truth(truth)
    records: list[dict[str, Any]] = []
    for max_speed_mps in max_speed_grid:
        for max_interpolation_residual_m in interpolation_residual_grid:
            for iterations in iterations_grid:
                repaired, diagnostics = repair_track5_temporal_spikes(
                    submission,
                    max_speed_mps=float(max_speed_mps),
                    max_interpolation_residual_m=float(max_interpolation_residual_m),
                    iterations=int(iterations),
                )
                metrics = _score_estimates(repaired, truth_rows)
                repaired_count = int(diagnostics["repaired"].astype(bool).sum()) if not diagnostics.empty else 0
                records.append(
                    {
                        "max_speed_mps": float(max_speed_mps),
                        "max_interpolation_residual_m": float(max_interpolation_residual_m),
                        "iterations": int(iterations),
                        "repaired_row_count": repaired_count,
                        "repaired_fraction": float(repaired_count / len(repaired)) if len(repaired) else 0.0,
                        "max_repair_displacement_m": _safe_max(diagnostics.get("repair_displacement_m", pd.Series(dtype=float))),
                        **metrics,
                    }
                )
    grid = pd.DataFrame.from_records(records)
    if grid.empty:
        raise ValueError("temporal repair search grid produced no rows")
    best_row = grid.sort_values(
        ["pose_mse_m2", "pose_p95_m", "pose_max_m", "repaired_row_count"],
        na_position="last",
    ).iloc[0]
    best = {
        "schema": "raft-uav-mmuad-track5-temporal-repair-search-v1",
        "max_speed_mps": float(best_row["max_speed_mps"]),
        "max_interpolation_residual_m": float(best_row["max_interpolation_residual_m"]),
        "iterations": int(best_row["iterations"]),
        "metrics": {
            key: _jsonable(best_row[key])
            for key in (
                "matched_rows",
                "pose_mse_m2",
                "pose_rmse_m",
                "pose_mean_m",
                "pose_p95_m",
                "pose_max_m",
                "repaired_row_count",
                "repaired_fraction",
                "max_repair_displacement_m",
            )
            if key in best_row.index
        },
    }
    return grid, _jsonable(best)


def write_temporal_repair_search_outputs(
    *,
    submission: pd.DataFrame,
    truth: pd.DataFrame,
    output_dir: Path,
    input_submission_path: Path,
    max_speed_grid: Iterable[float] = (40.0, 60.0, 80.0, 120.0),
    interpolation_residual_grid: Iterable[float] = (5.0, 10.0, 20.0, 30.0),
    iterations_grid: Iterable[int] = (1, 2, 3),
    write_best_submission: bool = False,
    template: pd.DataFrame | None = None,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grid, best = search_track5_temporal_repair_parameters(
        submission,
        truth,
        max_speed_grid=max_speed_grid,
        interpolation_residual_grid=interpolation_residual_grid,
        iterations_grid=iterations_grid,
    )
    paths = {
        "grid_csv": output / SEARCH_GRID_CSV,
        "best_config_json": output / BEST_CONFIG_JSON,
    }
    grid.to_csv(paths["grid_csv"], index=False)
    paths["best_config_json"].write_text(json.dumps(best, indent=2), encoding="utf-8")
    if write_best_submission:
        repaired, diagnostics = repair_track5_temporal_spikes(
            submission,
            max_speed_mps=float(best["max_speed_mps"]),
            max_interpolation_residual_m=float(best["max_interpolation_residual_m"]),
            iterations=int(best["iterations"]),
        )
        best_paths = write_track5_temporal_repair_outputs(
            repaired=repaired,
            diagnostics=diagnostics,
            output_dir=output / BEST_OUTPUT_DIR,
            input_submission_path=input_submission_path,
            template=template,
            manifest={"selected_by": str(paths["best_config_json"])},
            require_leaderboard_ready=require_leaderboard_ready,
        )
        paths.update({f"best_{name}": path for name, path in best_paths.items()})
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raft-uav-mmuad-track5-temporal-repair-search")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-speed-grid", default="40,60,80,120")
    parser.add_argument("--interpolation-residual-grid", default="5,10,20,30")
    parser.add_argument("--iterations-grid", default="1,2,3")
    parser.add_argument("--write-best-submission", action="store_true")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    template = None if args.template is None else load_official_track5_template_file(args.template)
    paths = write_temporal_repair_search_outputs(
        submission=load_track5_submission(args.submission),
        truth=load_evaluation_truth_file(args.truth_csv).rows,
        output_dir=args.output_dir,
        input_submission_path=args.submission,
        max_speed_grid=_parse_float_grid(args.max_speed_grid),
        interpolation_residual_grid=_parse_float_grid(args.interpolation_residual_grid),
        iterations_grid=_parse_int_grid(args.iterations_grid),
        write_best_submission=bool(args.write_best_submission),
        template=template,
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    print("mmuad_track5_temporal_repair_search=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _normalize_truth(truth: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(truth).copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["_time_key"] = _time_key(rows["time_s"])
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite, ["sequence_id", "_time_key", "x_m", "y_m", "z_m"]].copy()


def _score_estimates(estimates: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    rows = pd.DataFrame(estimates).copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["_time_key"] = _time_key(pd.to_numeric(rows["time_s"], errors="coerce"))
    merged = rows.merge(truth, on=["sequence_id", "_time_key"], how="inner")
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


def _parse_float_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).replace(";", ",").split(",") if item.strip()]


def _parse_int_grid(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).replace(";", ",").split(",") if item.strip()]


def _safe_max(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.max(numeric.to_numpy(float)))


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
