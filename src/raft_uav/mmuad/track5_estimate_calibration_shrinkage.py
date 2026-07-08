"""Shrink train-fitted Track 5 estimate calibration before submission.

Global estimate calibration can overcorrect easy sequences even when it helps hard
ones.  This module applies a previously fitted calibration JSON with a scalar
shrinkage factor

    x_shrunk = x_raw + alpha * (x_calibrated - x_raw)

and optionally searches ``alpha`` on a labeled train fold.  The search path is
truth-aware; the apply path is inference-safe and uses only estimates, an
official template, a fitted calibration JSON, and optional class labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    load_sequence_class_map,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_estimate_calibration import apply_track5_estimate_calibration
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

SHRUNK_ESTIMATES_CSV = "mmuad_track5_calibration_shrunk_estimates.csv"
SHRINKAGE_GRID_CSV = "mmuad_track5_calibration_shrinkage_grid.csv"
BEST_ALPHA_JSON = "mmuad_track5_calibration_shrinkage_best_alpha.json"
DIAGNOSTICS_CSV = "mmuad_track5_calibration_shrinkage_diagnostics.csv"
MANIFEST_JSON = "mmuad_track5_calibration_shrinkage_manifest.json"
VALIDATION_JSON = "mmuad_track5_calibration_shrinkage_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_calibration_shrinkage_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"


def apply_track5_estimate_calibration_shrinkage(
    estimates: pd.DataFrame,
    *,
    template: pd.DataFrame,
    calibration: dict[str, Any],
    alpha: float,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a fitted calibration with scalar shrinkage ``alpha``."""

    alpha = _validate_alpha(alpha)
    raw, raw_diagnostics = resample_estimates_to_track5_template(
        estimates,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    full, full_diagnostics = apply_track5_estimate_calibration(
        estimates,
        template=template,
        calibration=calibration,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    out = raw.copy()
    raw_xyz = _xyz_array(raw)
    full_xyz = _xyz_array(full)
    finite = np.isfinite(raw_xyz).all(axis=1) & np.isfinite(full_xyz).all(axis=1)
    shrunk = np.full_like(raw_xyz, np.nan, dtype=float)
    shrunk[finite] = raw_xyz[finite] + alpha * (full_xyz[finite] - raw_xyz[finite])
    out["state_x_m"] = shrunk[:, 0]
    out["state_y_m"] = shrunk[:, 1]
    out["state_z_m"] = shrunk[:, 2]
    out["track5_estimate_calibration_shrunk"] = True
    out["track5_estimate_calibration_alpha"] = float(alpha)
    out["track5_estimate_calibration_mode"] = str(calibration.get("mode", "unknown"))
    diagnostics = raw_diagnostics.copy()
    diagnostics["calibration_alpha"] = float(alpha)
    diagnostics["raw_row_valid"] = _finite_xyz(raw)
    diagnostics["full_calibrated_row_valid"] = _finite_xyz(full)
    diagnostics["shrunk_row_valid"] = np.isfinite(shrunk).all(axis=1)
    if "calibrated_row_valid" in full_diagnostics.columns:
        diagnostics["full_apply_reported_valid"] = full_diagnostics["calibrated_row_valid"].astype(bool)
    return out, diagnostics


def search_track5_estimate_calibration_alpha(
    estimates: pd.DataFrame,
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    calibration: dict[str, Any],
    alpha_values: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score alpha candidates on a labeled split and return grid plus best config."""

    truth_rows = _normalize_truth_for_exact_template(truth)
    records: list[dict[str, Any]] = []
    for alpha in alpha_values:
        alpha = _validate_alpha(alpha)
        shrunk, diagnostics = apply_track5_estimate_calibration_shrinkage(
            estimates,
            template=template,
            calibration=calibration,
            alpha=alpha,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        records.append(
            {
                "alpha": float(alpha),
                "valid_row_count": int(diagnostics["shrunk_row_valid"].sum()),
                **_score_template_estimates(shrunk, truth_rows),
            }
        )
    grid = pd.DataFrame.from_records(records)
    if grid.empty:
        raise ValueError("alpha grid produced no rows")
    best_row = grid.sort_values(["pose_mse_m2", "pose_p95_m", "pose_max_m"], na_position="last").iloc[0]
    best = {
        "schema": "raft-uav-mmuad-track5-calibration-shrinkage-v1",
        "alpha": float(best_row["alpha"]),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "calibration_schema": calibration.get("schema"),
        "calibration_mode": calibration.get("mode"),
        "metrics": {
            key: _jsonable(best_row[key])
            for key in ("pose_mse_m2", "pose_rmse_m", "pose_mean_m", "pose_p95_m", "pose_max_m", "matched_rows")
            if key in best_row.index
        },
    }
    return grid, _jsonable(best)


def write_track5_estimate_calibration_shrinkage_outputs(
    *,
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    calibration: dict[str, Any],
    output_dir: Path,
    alpha: float,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Apply alpha shrinkage and write official Track 5 artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shrunk, diagnostics = apply_track5_estimate_calibration_shrinkage(
        estimates,
        template=template,
        calibration=calibration,
        alpha=alpha,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "shrunk_estimates_csv": output / SHRUNK_ESTIMATES_CSV,
        "diagnostics_csv": output / DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / VALIDATION_JSON,
        "validation_rows_csv": output / VALIDATION_ROWS_CSV,
        "manifest_json": output / MANIFEST_JSON,
    }
    shrunk.to_csv(paths["shrunk_estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    class_map = class_map or {}
    write_official_mmaud_results_csv(
        shrunk,
        paths["official_results_csv"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        shrunk,
        paths["official_zip"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    validation = validate_official_track5_submission(
        paths["official_zip"],
        template=template,
        require_zip=True,
    )
    paths["validation_json"].write_text(json.dumps(_jsonable(validation.summary), indent=2), encoding="utf-8")
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    manifest = {
        "schema": "raft-uav-mmuad-track5-calibration-shrinkage-apply-v1",
        "alpha": float(alpha),
        "calibration_schema": calibration.get("schema"),
        "calibration_mode": calibration.get("mode"),
        "row_count": int(len(shrunk)),
        "valid_shrunk_rows": int(_finite_xyz(shrunk).sum()),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def write_track5_estimate_calibration_shrinkage_search_outputs(
    *,
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    calibration: dict[str, Any],
    output_dir: Path,
    alpha_values: Iterable[float],
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Search alpha values and write grid plus best-alpha JSON."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    grid, best = search_track5_estimate_calibration_alpha(
        estimates,
        template=template,
        truth=truth,
        calibration=calibration,
        alpha_values=alpha_values,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "alpha_grid_csv": output / SHRINKAGE_GRID_CSV,
        "best_alpha_json": output / BEST_ALPHA_JSON,
    }
    grid.to_csv(paths["alpha_grid_csv"], index=False)
    paths["best_alpha_json"].write_text(json.dumps(best, indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-calibration-shrinkage",
        description="apply or train-select shrinkage for fitted Track 5 estimate calibration",
    )
    parser.add_argument("--estimates-csv", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--write-apply", action="store_true")
    parser.add_argument("--use-best-alpha", action="store_true")
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates = read_estimate_csv(args.estimates_csv)
    template = load_official_track5_template_file(args.template)
    calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))
    paths: dict[str, Path] = {}
    alpha = float(args.alpha)
    if args.truth_csv is not None:
        search_paths = write_track5_estimate_calibration_shrinkage_search_outputs(
            estimates=estimates,
            template=template,
            truth=load_evaluation_truth_file(args.truth_csv).rows,
            calibration=calibration,
            output_dir=output / "search",
            alpha_values=_parse_alpha_grid(args.alpha_grid),
            max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        )
        paths.update(search_paths)
        if args.use_best_alpha:
            alpha = float(json.loads(search_paths["best_alpha_json"].read_text(encoding="utf-8"))["alpha"])
    if args.write_apply or args.truth_csv is None:
        class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
        paths.update(
            write_track5_estimate_calibration_shrinkage_outputs(
                estimates=estimates,
                template=template,
                calibration=calibration,
                output_dir=output / "apply",
                alpha=alpha,
                class_map=class_map,
                default_classification=args.default_classification,
                max_nearest_time_delta_s=args.max_nearest_time_delta_s,
            )
        )
    print("mmuad_track5_calibration_shrinkage=ok")
    print(f"alpha={alpha}")
    for name, path in paths.items():
        print(f"{name}={path}")
    if args.require_leaderboard_ready:
        validation_path = paths.get("validation_json")
        if validation_path is None:
            raise SystemExit("leaderboard readiness requires --write-apply or apply-only mode")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not validation.get("leaderboard_ready", False):
            reasons = ", ".join(validation.get("leaderboard_blocking_reasons", []))
            raise SystemExit(f"shrunk calibrated upload is not leaderboard-ready: {reasons}")
    return 0


def _xyz_array(rows: pd.DataFrame) -> np.ndarray:
    return rows[["state_x_m", "state_y_m", "state_z_m"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def _finite_xyz(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool)
    return pd.Series(np.isfinite(_xyz_array(rows)).all(axis=1), index=rows.index)


def _validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"calibration alpha must be finite and in [0, 1]: {alpha}")
    return alpha


def _parse_alpha_grid(text: str) -> tuple[float, ...]:
    values = tuple(_validate_alpha(float(item.strip())) for item in str(text).replace(";", ",").split(",") if item.strip())
    if not values:
        raise ValueError("alpha grid must contain at least one value")
    return values


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
