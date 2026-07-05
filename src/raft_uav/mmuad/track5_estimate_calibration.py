"""Train-fitted pose calibration for MMUAD Track 5 estimate trajectories.

This module fits a light-weight correction from a labeled split and applies the
fixed correction to validation or hidden-test estimates. It is intended for
Codabench / UG2+ Track 5 pose pipelines whose final trajectories have residual
systematic bias after source calibration, mixture smoothing, or ensembling.

The fit path is truth-aware and must be used only on train folds or local public
validation diagnostics. The apply path is inference-safe: it uses only estimate
rows, an official timestamp template, and a previously fitted calibration JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    load_sequence_class_map,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

CalibrationMode = Literal["identity", "translation", "diagonal-affine", "affine"]
CALIBRATION_MODES = ("identity", "translation", "diagonal-affine", "affine")
CALIBRATED_ESTIMATES_CSV = "mmuad_track5_calibrated_estimates.csv"
CALIBRATION_FIT_PAIRS_CSV = "mmuad_track5_estimate_calibration_fit_pairs.csv"
CALIBRATION_JSON = "mmuad_track5_estimate_calibration.json"
CALIBRATION_APPLY_DIAGNOSTICS_CSV = "mmuad_track5_estimate_calibration_apply_diagnostics.csv"
CALIBRATION_APPLY_MANIFEST_JSON = "mmuad_track5_estimate_calibration_apply_manifest.json"
VALIDATION_JSON = "mmuad_track5_estimate_calibration_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_estimate_calibration_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"


def fit_track5_estimate_calibration(
    estimates: pd.DataFrame,
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    mode: CalibrationMode = "translation",
    robust: bool = True,
    ridge_lambda: float = 1.0e-6,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit a global Track 5 estimate correction and return config plus pairs."""

    mode = _normalize_mode(mode)
    resampled, _ = resample_estimates_to_track5_template(
        estimates,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    pairs = _fit_pairs(resampled, truth)
    if pairs.empty:
        raise ValueError("no finite estimate/truth pairs available for calibration")
    estimated = pairs[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    truth_xyz = pairs[["truth_x_m", "truth_y_m", "truth_z_m"]].to_numpy(float)
    transform = _fit_transform(
        estimated,
        truth_xyz,
        mode=mode,
        robust=bool(robust),
        ridge_lambda=float(ridge_lambda),
    )
    before_error = np.linalg.norm(estimated - truth_xyz, axis=1)
    after_xyz = _apply_transform_array(estimated, transform)
    after_error = np.linalg.norm(after_xyz - truth_xyz, axis=1)
    config = {
        "schema": "raft-uav-mmuad-track5-estimate-calibration-v1",
        "mode": mode,
        "robust": bool(robust),
        "ridge_lambda": float(ridge_lambda),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "transform": transform,
        "fit_summary": _error_summary(before_error, after_error, len(pairs)),
    }
    pairs = pairs.copy()
    pairs["before_error_m"] = before_error
    pairs["after_error_m"] = after_error
    return _jsonable(config), pairs


def apply_track5_estimate_calibration(
    estimates: pd.DataFrame,
    *,
    template: pd.DataFrame,
    calibration: dict[str, Any],
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a fitted calibration JSON to estimates after template resampling."""

    calibration_time_gate = calibration.get("max_nearest_time_delta_s")
    resampled, diagnostics = resample_estimates_to_track5_template(
        estimates,
        template,
        max_nearest_time_delta_s=(
            max_nearest_time_delta_s
            if max_nearest_time_delta_s is not None
            else calibration_time_gate
        ),
    )
    out = resampled.copy()
    xyz = (
        out[["state_x_m", "state_y_m", "state_z_m"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    finite = np.isfinite(xyz).all(axis=1)
    calibrated = np.full_like(xyz, np.nan, dtype=float)
    if finite.any():
        calibrated[finite] = _apply_transform_array(xyz[finite], calibration["transform"])
    out["state_x_m"] = calibrated[:, 0]
    out["state_y_m"] = calibrated[:, 1]
    out["state_z_m"] = calibrated[:, 2]
    out["track5_estimate_calibrated"] = True
    out["track5_estimate_calibration_mode"] = str(calibration.get("mode", "unknown"))
    diagnostics = diagnostics.copy()
    diagnostics["track5_estimate_calibration_mode"] = str(calibration.get("mode", "unknown"))
    diagnostics["calibrated_row_valid"] = np.isfinite(calibrated).all(axis=1)
    return out, diagnostics


def write_track5_estimate_calibration_fit_outputs(
    *,
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    output_dir: Path,
    mode: CalibrationMode = "translation",
    robust: bool = True,
    ridge_lambda: float = 1.0e-6,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Fit calibration and write JSON plus fit-pair diagnostics."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config, pairs = fit_track5_estimate_calibration(
        estimates,
        template=template,
        truth=truth,
        mode=mode,
        robust=robust,
        ridge_lambda=ridge_lambda,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "calibration_json": output / CALIBRATION_JSON,
        "fit_pairs_csv": output / CALIBRATION_FIT_PAIRS_CSV,
    }
    paths["calibration_json"].write_text(json.dumps(config, indent=2), encoding="utf-8")
    pairs.to_csv(paths["fit_pairs_csv"], index=False)
    return paths


def write_track5_estimate_calibration_apply_outputs(
    *,
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    calibration: dict[str, Any],
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Apply calibration and write official Track 5 artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    calibrated, diagnostics = apply_track5_estimate_calibration(
        estimates,
        template=template,
        calibration=calibration,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "calibrated_estimates_csv": output / CALIBRATED_ESTIMATES_CSV,
        "diagnostics_csv": output / CALIBRATION_APPLY_DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / VALIDATION_JSON,
        "validation_rows_csv": output / VALIDATION_ROWS_CSV,
        "manifest_json": output / CALIBRATION_APPLY_MANIFEST_JSON,
    }
    calibrated.to_csv(paths["calibrated_estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    class_map = class_map or {}
    write_official_mmaud_results_csv(
        calibrated,
        paths["official_results_csv"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        calibrated,
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
    paths["validation_json"].write_text(
        json.dumps(_jsonable(validation.summary), indent=2),
        encoding="utf-8",
    )
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    manifest = {
        "schema": "raft-uav-mmuad-track5-estimate-calibration-apply-v1",
        "calibration_schema": calibration.get("schema"),
        "mode": calibration.get("mode"),
        "row_count": int(len(calibrated)),
        "valid_calibrated_rows": int(_finite_xyz(calibrated).sum()),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "paths": {
            name: str(path) for name, path in paths.items() if name != "manifest_json"
        },
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-estimate-calibration",
        description="fit/apply train-derived calibration for Track 5 estimate trajectories",
    )
    parser.add_argument("--fit-estimates-csv", type=Path)
    parser.add_argument("--apply-estimates-csv", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=CALIBRATION_MODES, default="translation")
    parser.add_argument("--no-robust", action="store_true")
    parser.add_argument("--ridge-lambda", type=float, default=1.0e-6)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    template = load_official_track5_template_file(args.template)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if args.fit_estimates_csv is not None:
        if args.truth_csv is None:
            parser.error("--truth-csv is required with --fit-estimates-csv")
        paths.update(
            write_track5_estimate_calibration_fit_outputs(
                estimates=pd.read_csv(args.fit_estimates_csv),
                template=template,
                truth=load_evaluation_truth_file(args.truth_csv).rows,
                output_dir=output / "fit",
                mode=args.mode,
                robust=not bool(args.no_robust),
                ridge_lambda=float(args.ridge_lambda),
                max_nearest_time_delta_s=args.max_nearest_time_delta_s,
            )
        )
        args.calibration_json.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_json.write_text(
            paths["calibration_json"].read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        paths["calibration_json"] = args.calibration_json
    if args.apply_estimates_csv is not None:
        calibration = json.loads(args.calibration_json.read_text(encoding="utf-8"))
        class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
        paths.update(
            write_track5_estimate_calibration_apply_outputs(
                estimates=pd.read_csv(args.apply_estimates_csv),
                template=template,
                calibration=calibration,
                output_dir=output / "apply",
                class_map=class_map,
                default_classification=args.default_classification,
                max_nearest_time_delta_s=args.max_nearest_time_delta_s,
            )
        )
    if args.fit_estimates_csv is None and args.apply_estimates_csv is None:
        parser.error("provide --fit-estimates-csv, --apply-estimates-csv, or both")
    print("mmuad_track5_estimate_calibration=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    if args.require_leaderboard_ready:
        validation_path = paths.get("validation_json")
        if validation_path is None:
            raise SystemExit("leaderboard readiness requires --apply-estimates-csv")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not validation.get("leaderboard_ready", False):
            reasons = ", ".join(validation.get("leaderboard_blocking_reasons", []))
            raise SystemExit(f"calibrated upload is not leaderboard-ready: {reasons}")
    return 0


def _fit_pairs(estimates: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    estimate_rows = pd.DataFrame(estimates).copy()
    truth_rows = pd.DataFrame(truth).copy()
    if estimate_rows.empty or truth_rows.empty:
        return pd.DataFrame()
    truth_rows["sequence_id"] = truth_rows["sequence_id"].astype(str)
    estimate_rows["sequence_id"] = estimate_rows["sequence_id"].astype(str)
    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m"):
        estimate_rows[column] = pd.to_numeric(estimate_rows[column], errors="coerce")
    for column in ("time_s", "x_m", "y_m", "z_m"):
        truth_rows[column] = pd.to_numeric(truth_rows[column], errors="coerce")
    estimate_time_finite = np.isfinite(estimate_rows["time_s"].to_numpy(float))
    truth_time_finite = np.isfinite(truth_rows["time_s"].to_numpy(float))
    estimate_rows = estimate_rows.loc[estimate_time_finite].copy()
    truth_rows = truth_rows.loc[truth_time_finite].copy()
    if estimate_rows.empty or truth_rows.empty:
        return pd.DataFrame()
    estimate_rows["_time_key"] = _time_key(estimate_rows["time_s"])
    truth_rows["_time_key"] = _time_key(truth_rows["time_s"])
    pairs = estimate_rows.merge(
        truth_rows[["sequence_id", "_time_key", "x_m", "y_m", "z_m"]],
        on=["sequence_id", "_time_key"],
        how="inner",
    )
    pairs = pairs.rename(
        columns={"x_m": "truth_x_m", "y_m": "truth_y_m", "z_m": "truth_z_m"}
    )
    value_columns = [
        "state_x_m",
        "state_y_m",
        "state_z_m",
        "truth_x_m",
        "truth_y_m",
        "truth_z_m",
    ]
    finite = np.isfinite(pairs[value_columns].to_numpy(float)).all(axis=1)
    return pairs.loc[finite].reset_index(drop=True)


def _fit_transform(
    estimated: np.ndarray,
    truth: np.ndarray,
    *,
    mode: CalibrationMode,
    robust: bool,
    ridge_lambda: float,
) -> dict[str, Any]:
    residual = truth - estimated
    if mode == "identity":
        matrix = np.eye(3)
        offset = np.zeros(3)
    elif mode == "translation":
        matrix = np.eye(3)
        offset = np.median(residual, axis=0) if robust else np.mean(residual, axis=0)
    elif mode == "diagonal-affine":
        matrix = np.eye(3)
        offset = np.zeros(3)
        for axis in range(3):
            x = estimated[:, axis]
            y = truth[:, axis]
            design = np.column_stack([x, np.ones_like(x)])
            beta = _ridge_solve(design, y, ridge_lambda=float(ridge_lambda))
            matrix[axis, axis] = float(beta[0])
            offset[axis] = float(beta[1])
    elif mode == "affine":
        design = np.column_stack([estimated, np.ones(len(estimated))])
        beta = _ridge_solve(design, truth, ridge_lambda=float(ridge_lambda))
        matrix = beta[:3, :].T
        offset = beta[3, :]
    else:  # pragma: no cover
        raise ValueError(f"unsupported calibration mode: {mode}")
    return {
        "matrix": np.asarray(matrix, dtype=float).tolist(),
        "offset_m": np.asarray(offset, dtype=float).tolist(),
    }


def _apply_transform_array(xyz: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    matrix = np.asarray(transform.get("matrix", np.eye(3)), dtype=float)
    offset = np.asarray(transform.get("offset_m", np.zeros(3)), dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("calibration matrix must be 3x3")
    if offset.shape != (3,):
        raise ValueError("calibration offset_m must have length 3")
    return xyz @ matrix.T + offset


def _ridge_solve(design: np.ndarray, target: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    lhs = design.T @ design
    penalty = float(ridge_lambda) * np.eye(lhs.shape[0])
    penalty[-1, -1] = 0.0
    rhs = design.T @ target
    return np.linalg.solve(lhs + penalty, rhs)


def _error_summary(before_error: np.ndarray, after_error: np.ndarray, count: int) -> dict[str, Any]:
    return {
        "fit_pair_count": int(count),
        "before_mse_m2": float(np.mean(before_error**2)),
        "before_rmse_m": float(np.sqrt(np.mean(before_error**2))),
        "before_mean_m": float(np.mean(before_error)),
        "before_p95_m": float(np.percentile(before_error, 95)),
        "after_mse_m2": float(np.mean(after_error**2)),
        "after_rmse_m": float(np.sqrt(np.mean(after_error**2))),
        "after_mean_m": float(np.mean(after_error)),
        "after_p95_m": float(np.percentile(after_error, 95)),
        "mean_delta_m": float(np.mean(after_error) - np.mean(before_error)),
        "mse_delta_m2": float(np.mean(after_error**2) - np.mean(before_error**2)),
    }


def _normalize_mode(mode: str) -> CalibrationMode:
    mode = str(mode)
    if mode not in set(CALIBRATION_MODES):
        raise ValueError(f"unsupported calibration mode: {mode}")
    return cast(CalibrationMode, mode)


def _finite_xyz(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool)
    xyz = (
        rows[["state_x_m", "state_y_m", "state_z_m"]]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    return pd.Series(np.isfinite(xyz).all(axis=1), index=rows.index)


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
