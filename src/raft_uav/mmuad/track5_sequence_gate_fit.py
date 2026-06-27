"""Fit sequence-level blend weights for MMUAD/UG2+ Track 5 submissions.

The companion :mod:`raft_uav.mmuad.track5_sequence_gate` module applies a
precomputed sequence -> blend weight table.  This module creates those tables
from a base submission, an alternate submission, and a reference file.  It is
intended for diagnostics and train-only selection runs: fitting on public
validation truth is useful for estimating ceilings, but is not hidden-test
evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_submission_ensemble import _jsonable
from raft_uav.mmuad.track5_submission_ensemble import _submission_keys
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission

SUMMARY_CSV = "mmuad_track5_sequence_gate_fit_summary.csv"
SUMMARY_JSON = "mmuad_track5_sequence_gate_fit_summary.json"
SEQUENCE_FEATURES_CSV = "mmuad_track5_sequence_gate_fit_features.csv"
ORACLE_WEIGHTS_CSV = "mmuad_track5_sequence_gate_oracle_weights.csv"
SAME_SPLIT_WEIGHTS_CSV = "mmuad_track5_sequence_gate_same_split_weights.csv"
LOSO_WEIGHTS_CSV = "mmuad_track5_sequence_gate_loso_weights.csv"
APPLY_FEATURES_CSV = "mmuad_track5_sequence_gate_apply_features.csv"
APPLY_WEIGHTS_CSV = "mmuad_track5_sequence_gate_apply_weights.csv"
FEATURE_SHIFT_CSV = "mmuad_track5_sequence_gate_feature_shift.csv"
APPLY_SEQUENCE_SHIFT_CSV = "mmuad_track5_sequence_gate_apply_sequence_shift.csv"


@dataclass(frozen=True)
class SequenceGateFitResult:
    """Sequence-gate fit diagnostics and predicted weight tables."""

    summary: pd.DataFrame
    sequence_features: pd.DataFrame
    oracle_weights: pd.DataFrame
    same_split_weights: pd.DataFrame
    loso_weights: pd.DataFrame
    best_model: str
    apply_sequence_features: pd.DataFrame | None = None
    apply_weights: pd.DataFrame | None = None
    feature_shift: pd.DataFrame | None = None
    apply_sequence_shift: pd.DataFrame | None = None


def fit_track5_sequence_gate(
    *,
    base_submission: pd.DataFrame,
    alternate_submission: pd.DataFrame,
    truth: pd.DataFrame,
    weight_grid: np.ndarray | None = None,
    models: tuple[str, ...] = ("ridge", "tree_d3_leaf1", "tree_d4_leaf1"),
    apply_base_submission: pd.DataFrame | None = None,
    apply_alternate_submission: pd.DataFrame | None = None,
    random_state: int = 13,
) -> SequenceGateFitResult:
    """Fit per-sequence blend weights and return same-split/LOSO diagnostics.

    ``base_submission``, ``alternate_submission``, and ``truth`` must be
    normalized frames from :func:`load_track5_submission` and share
    ``sequence_id,time_s`` keys.  If ``apply_base_submission`` and
    ``apply_alternate_submission`` are provided, the best LOSO-selected model is
    refit on the training split and used to predict sequence weights for the
    apply split without using apply truth.  The returned model rows are sorted
    by LOSO MSE first, then same-split MSE.
    """

    if (apply_base_submission is None) != (apply_alternate_submission is None):
        raise ValueError("apply-base-submission and apply-alternate-submission must be paired")

    base, alternate, truth_rows = _aligned_frames(base_submission, alternate_submission, truth)
    grid = _weight_grid(weight_grid)
    oracle_weights = _oracle_sequence_weights(base, alternate, truth_rows, grid)
    sequence_features = _sequence_feature_table(base, alternate).merge(
        oracle_weights,
        on="sequence_id",
        how="inner",
        validate="one_to_one",
    )
    feature_columns = _feature_columns(sequence_features)
    if not feature_columns:
        raise ValueError("no finite sequence-gate feature columns available")
    summary_records: list[dict[str, Any]] = []
    same_split_predictions: dict[str, pd.DataFrame] = {}
    loso_predictions: dict[str, pd.DataFrame] = {}
    for model_name in models:
        same_weights = _predict_same_split_weights(
            model_name,
            sequence_features,
            feature_columns,
            random_state=random_state,
            min_weight=float(grid.min()),
            max_weight=float(grid.max()),
        )
        loso_weights = _predict_loso_weights(
            model_name,
            sequence_features,
            feature_columns,
            random_state=random_state,
            min_weight=float(grid.min()),
            max_weight=float(grid.max()),
        )
        same_metrics = _score_weight_table(base, alternate, truth_rows, same_weights)
        loso_metrics = _score_weight_table(base, alternate, truth_rows, loso_weights)
        same_split_predictions[model_name] = same_weights
        loso_predictions[model_name] = loso_weights
        summary_records.append(
            {
                "model": model_name,
                **{f"same_split_{key}": value for key, value in same_metrics.items()},
                **{f"loso_{key}": value for key, value in loso_metrics.items()},
                "same_split_weights": _weights_text(same_weights),
                "loso_weights": _weights_text(loso_weights),
            }
        )
    summary = pd.DataFrame.from_records(summary_records).sort_values(
        ["loso_mse", "same_split_mse", "model"],
        ascending=[True, True, True],
    )
    best_model = str(summary.iloc[0]["model"])
    apply_sequence_features = None
    apply_weights = None
    feature_shift = None
    apply_sequence_shift = None
    if apply_base_submission is not None and apply_alternate_submission is not None:
        apply_base, apply_alternate = _aligned_submission_frames(
            apply_base_submission,
            apply_alternate_submission,
        )
        apply_sequence_features = _sequence_feature_table(apply_base, apply_alternate)
        feature_shift = _feature_shift_table(
            sequence_features,
            apply_sequence_features,
            feature_columns,
        )
        apply_sequence_shift = _apply_sequence_shift_table(
            sequence_features,
            apply_sequence_features,
            feature_columns,
        )
        apply_weights = _predict_apply_weights(
            best_model,
            sequence_features,
            apply_sequence_features,
            feature_columns,
            random_state=random_state,
            min_weight=float(grid.min()),
            max_weight=float(grid.max()),
        )
    return SequenceGateFitResult(
        summary=summary.reset_index(drop=True),
        sequence_features=sequence_features.reset_index(drop=True),
        oracle_weights=oracle_weights.reset_index(drop=True),
        same_split_weights=same_split_predictions[best_model].reset_index(drop=True),
        loso_weights=loso_predictions[best_model].reset_index(drop=True),
        best_model=best_model,
        apply_sequence_features=None
        if apply_sequence_features is None
        else apply_sequence_features.reset_index(drop=True),
        apply_weights=None if apply_weights is None else apply_weights.reset_index(drop=True),
        feature_shift=None if feature_shift is None else feature_shift.reset_index(drop=True),
        apply_sequence_shift=None
        if apply_sequence_shift is None
        else apply_sequence_shift.reset_index(drop=True),
    )


def write_track5_sequence_gate_fit_outputs(
    *,
    result: SequenceGateFitResult,
    output_dir: Path,
    base_submission_path: Path,
    alternate_submission_path: Path,
    truth_path: Path,
    apply_base_submission_path: Path | None = None,
    apply_alternate_submission_path: Path | None = None,
    weight_grid: np.ndarray,
    protocol: str,
) -> dict[str, Path]:
    """Write fit summaries and weight tables."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": output / SUMMARY_CSV,
        "summary_json": output / SUMMARY_JSON,
        "sequence_features_csv": output / SEQUENCE_FEATURES_CSV,
        "oracle_weights_csv": output / ORACLE_WEIGHTS_CSV,
        "same_split_weights_csv": output / SAME_SPLIT_WEIGHTS_CSV,
        "loso_weights_csv": output / LOSO_WEIGHTS_CSV,
    }
    if result.apply_weights is not None:
        paths["apply_weights_csv"] = output / APPLY_WEIGHTS_CSV
    if result.apply_sequence_features is not None:
        paths["apply_features_csv"] = output / APPLY_FEATURES_CSV
    if result.feature_shift is not None:
        paths["feature_shift_csv"] = output / FEATURE_SHIFT_CSV
    if result.apply_sequence_shift is not None:
        paths["apply_sequence_shift_csv"] = output / APPLY_SEQUENCE_SHIFT_CSV
    result.summary.to_csv(paths["summary_csv"], index=False)
    result.sequence_features.to_csv(paths["sequence_features_csv"], index=False)
    result.oracle_weights.to_csv(paths["oracle_weights_csv"], index=False)
    result.same_split_weights.to_csv(paths["same_split_weights_csv"], index=False)
    result.loso_weights.to_csv(paths["loso_weights_csv"], index=False)
    if result.apply_weights is not None:
        result.apply_weights.to_csv(paths["apply_weights_csv"], index=False)
    if result.apply_sequence_features is not None:
        result.apply_sequence_features.to_csv(paths["apply_features_csv"], index=False)
    if result.feature_shift is not None:
        result.feature_shift.to_csv(paths["feature_shift_csv"], index=False)
    if result.apply_sequence_shift is not None:
        result.apply_sequence_shift.to_csv(paths["apply_sequence_shift_csv"], index=False)
    best_row = result.summary.iloc[0].to_dict()
    payload = {
        "schema": "raft-uav-mmuad-track5-sequence-gate-fit-v1",
        "protocol": protocol,
        "base_submission": str(base_submission_path),
        "alternate_submission": str(alternate_submission_path),
        "truth": str(truth_path),
        "apply_base_submission": None
        if apply_base_submission_path is None
        else str(apply_base_submission_path),
        "apply_alternate_submission": None
        if apply_alternate_submission_path is None
        else str(apply_alternate_submission_path),
        "weight_min": float(np.min(weight_grid)),
        "weight_max": float(np.max(weight_grid)),
        "weight_count": int(len(weight_grid)),
        "best_model": result.best_model,
        "best_row": best_row,
        "apply_sequence_count": 0
        if result.apply_weights is None
        else int(result.apply_weights["sequence_id"].nunique()),
        "apply_feature_shift": _feature_shift_summary(result.feature_shift),
        "apply_sequence_shift": _apply_sequence_shift_summary(result.apply_sequence_shift),
        "paths": {name: str(path) for name, path in paths.items() if name != "summary_json"},
    }
    paths["summary_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-sequence-gate-fit",
        description="fit Track 5 sequence-gate blend weights from base/alternate submissions",
    )
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--alternate-submission", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--apply-base-submission", type=Path)
    parser.add_argument("--apply-alternate-submission", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-min", type=float, default=0.0)
    parser.add_argument("--weight-max", type=float, default=0.5)
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="model name; repeatable. Supported: ridge, tree_d{depth}_leaf{leaf}, "
        "rf_depth{depth}, extra_depth{depth}",
    )
    parser.add_argument(
        "--protocol",
        default="diagnostic_truth_fit_not_hidden_test_evidence",
        help="free-form provenance string written to the summary JSON",
    )
    parser.add_argument("--random-state", type=int, default=13)
    args = parser.parse_args(argv)

    grid = _grid_from_args(args.weight_min, args.weight_max, args.weight_step)
    result = fit_track5_sequence_gate(
        base_submission=_load_track5_gate_rows(args.base_submission),
        alternate_submission=_load_track5_gate_rows(args.alternate_submission),
        truth=_load_track5_gate_rows(args.truth),
        apply_base_submission=None
        if args.apply_base_submission is None
        else _load_track5_gate_rows(args.apply_base_submission),
        apply_alternate_submission=None
        if args.apply_alternate_submission is None
        else _load_track5_gate_rows(args.apply_alternate_submission),
        weight_grid=grid,
        models=tuple(args.model)
        or (
            "ridge",
            "tree_d2_leaf1",
            "tree_d3_leaf1",
            "tree_d4_leaf1",
            "rf_depth2",
            "extra_depth2",
        ),
        random_state=args.random_state,
    )
    paths = write_track5_sequence_gate_fit_outputs(
        result=result,
        output_dir=args.output_dir,
        base_submission_path=args.base_submission,
        alternate_submission_path=args.alternate_submission,
        truth_path=args.truth,
        apply_base_submission_path=args.apply_base_submission,
        apply_alternate_submission_path=args.apply_alternate_submission,
        weight_grid=grid,
        protocol=args.protocol,
    )
    print("mmuad_track5_sequence_gate_fit=ok")
    print(f"best_model={result.best_model}")
    print(f"summary_csv={paths['summary_csv']}")
    print(f"same_split_weights_csv={paths['same_split_weights_csv']}")
    print(f"loso_weights_csv={paths['loso_weights_csv']}")
    if "apply_weights_csv" in paths:
        print(f"apply_weights_csv={paths['apply_weights_csv']}")
    return 0


def _load_track5_gate_rows(path: Path) -> pd.DataFrame:
    """Load official Track 5 rows or normalized train/reference rows."""

    try:
        return load_track5_submission(path)
    except ValueError:
        path = Path(path)
        if path.suffix.lower() == ".zip":
            raise
        rows = pd.read_csv(path)
    columns = {str(column).lower(): str(column) for column in rows.columns}
    required = {
        "sequence_id": columns.get("sequence_id"),
        "time_s": columns.get("time_s"),
        "state_x_m": columns.get("state_x_m") or columns.get("x_m"),
        "state_y_m": columns.get("state_y_m") or columns.get("y_m"),
        "state_z_m": columns.get("state_z_m") or columns.get("z_m"),
    }
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise ValueError(f"{path} missing official or normalized columns: {missing}")
    normalized = pd.DataFrame(
        {
            "sequence_id": rows[required["sequence_id"]].astype(str),
            "time_s": pd.to_numeric(rows[required["time_s"]], errors="raise"),
            "state_x_m": pd.to_numeric(rows[required["state_x_m"]], errors="raise"),
            "state_y_m": pd.to_numeric(rows[required["state_y_m"]], errors="raise"),
            "state_z_m": pd.to_numeric(rows[required["state_z_m"]], errors="raise"),
        }
    )
    class_column = (
        columns.get("classification") or columns.get("uav_type") or columns.get("class_id")
    )
    if class_column is not None:
        normalized["Classification"] = pd.to_numeric(rows[class_column], errors="coerce")
    return normalized.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _aligned_frames(
    base_submission: pd.DataFrame,
    alternate_submission: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base, alternate = _aligned_submission_frames(base_submission, alternate_submission)
    truth_rows = pd.DataFrame(truth).copy().sort_values(["sequence_id", "time_s"])
    if _submission_keys(base) != _submission_keys(truth_rows):
        raise ValueError("truth does not match submission sequence/timestamp keys")
    return (
        base.reset_index(drop=True),
        alternate.reset_index(drop=True),
        truth_rows.reset_index(drop=True),
    )


def _aligned_submission_frames(
    base_submission: pd.DataFrame,
    alternate_submission: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.DataFrame(base_submission).copy().sort_values(["sequence_id", "time_s"])
    alternate = pd.DataFrame(alternate_submission).copy().sort_values(["sequence_id", "time_s"])
    if _submission_keys(base) != _submission_keys(alternate):
        raise ValueError("base and alternate submissions do not match sequence/timestamp keys")
    return base.reset_index(drop=True), alternate.reset_index(drop=True)


def _grid_from_args(weight_min: float, weight_max: float, weight_step: float) -> np.ndarray:
    if weight_step <= 0:
        raise ValueError("weight-step must be positive")
    if weight_max < weight_min:
        raise ValueError("weight-max must be at least weight-min")
    count = int(np.floor((weight_max - weight_min) / weight_step)) + 1
    values = weight_min + np.arange(count + 1, dtype=float) * weight_step
    values = values[values <= weight_max + 1.0e-12]
    values[-1] = min(values[-1], weight_max)
    return _weight_grid(values)


def _weight_grid(weight_grid: np.ndarray | None) -> np.ndarray:
    if weight_grid is None:
        weight_grid = np.linspace(0.0, 0.5, 51)
    grid = np.asarray(weight_grid, dtype=float)
    grid = grid[np.isfinite(grid)]
    if grid.size == 0:
        raise ValueError("weight grid is empty")
    if float(grid.min()) < 0.0 or float(grid.max()) > 1.0:
        raise ValueError("sequence-gate weights must be in [0, 1]")
    return np.unique(np.sort(grid))


def _oracle_sequence_weights(
    base: pd.DataFrame,
    alternate: pd.DataFrame,
    truth: pd.DataFrame,
    weight_grid: np.ndarray,
) -> pd.DataFrame:
    base_xyz = base[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    alt_xyz = alternate[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    truth_xyz = truth[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    work = base[["sequence_id", "time_s"]].copy()
    work["_row_index"] = np.arange(len(work), dtype=int)
    records: list[dict[str, Any]] = []
    for sequence_id, group in work.groupby("sequence_id", sort=True):
        indices = group["_row_index"].to_numpy(int)
        metrics = []
        for weight in weight_grid:
            xyz = (1.0 - float(weight)) * base_xyz[indices] + float(weight) * alt_xyz[indices]
            errors = np.linalg.norm(xyz - truth_xyz[indices], axis=1)
            metrics.append((float(weight), _pose_metrics(errors)))
        best_weight, best_metrics = min(metrics, key=lambda item: item[1]["mse"])
        base_errors = np.linalg.norm(base_xyz[indices] - truth_xyz[indices], axis=1)
        alt_errors = np.linalg.norm(alt_xyz[indices] - truth_xyz[indices], axis=1)
        records.append(
            {
                "sequence_id": str(sequence_id),
                "oracle_weight": float(best_weight),
                **{f"oracle_{key}": value for key, value in best_metrics.items()},
                **{f"base_{key}": value for key, value in _pose_metrics(base_errors).items()},
                **{f"alternate_{key}": value for key, value in _pose_metrics(alt_errors).items()},
            }
        )
    return pd.DataFrame.from_records(records)


def _sequence_feature_table(base: pd.DataFrame, alternate: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sequence_id, base_group in base.groupby("sequence_id", sort=True):
        alt_group = alternate.loc[alternate["sequence_id"].astype(str) == str(sequence_id)].copy()
        base_group = base_group.sort_values("time_s")
        alt_group = alt_group.sort_values("time_s")
        base_xyz = base_group[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
        alt_xyz = alt_group[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
        diff = np.linalg.norm(alt_xyz - base_xyz, axis=1)
        z_diff = np.abs(alt_xyz[:, 2] - base_xyz[:, 2])
        base_speed, base_accel, base_path = _trajectory_motion_features(
            base_group["time_s"].to_numpy(float),
            base_xyz,
        )
        alt_speed, alt_accel, alt_path = _trajectory_motion_features(
            alt_group["time_s"].to_numpy(float),
            alt_xyz,
        )
        records.append(
            {
                "sequence_id": str(sequence_id),
                "row_count": int(len(base_group)),
                "diff_mean": _safe_mean(diff),
                "diff_p50": _safe_percentile(diff, 50.0),
                "diff_p95": _safe_percentile(diff, 95.0),
                "diff_max": _safe_max(diff),
                "diff_std": _safe_std(diff),
                "z_diff_mean": _safe_mean(z_diff),
                "base_speed_mean": _safe_mean(base_speed),
                "base_speed_p95": _safe_percentile(base_speed, 95.0),
                "base_speed_max": _safe_max(base_speed),
                "base_accel_p95": _safe_percentile(base_accel, 95.0),
                "base_path_len": float(base_path),
                "alternate_speed_mean": _safe_mean(alt_speed),
                "alternate_speed_p95": _safe_percentile(alt_speed, 95.0),
                "alternate_speed_max": _safe_max(alt_speed),
                "alternate_accel_p95": _safe_percentile(alt_accel, 95.0),
                "alternate_path_len": float(alt_path),
                "speed_p95_ratio_alternate_base": _safe_ratio(
                    _safe_percentile(alt_speed, 95.0),
                    _safe_percentile(base_speed, 95.0),
                ),
                "path_len_ratio_alternate_base": _safe_ratio(float(alt_path), float(base_path)),
            }
        )
    return pd.DataFrame.from_records(records)


def _trajectory_motion_features(
    times: np.ndarray,
    xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(xyz) < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), 0.0
    dt = np.diff(times.astype(float))
    step = np.linalg.norm(np.diff(xyz.astype(float), axis=0), axis=1)
    valid = dt > 1.0e-9
    speed = np.divide(step, dt, out=np.full_like(step, np.nan), where=valid)
    path_len = float(np.nansum(step))
    if len(speed) < 2:
        return speed, np.asarray([], dtype=float), path_len
    mid_dt = (dt[1:] + dt[:-1]) / 2.0
    accel_step = np.abs(np.diff(speed))
    valid_accel = mid_dt > 1.0e-9
    accel = np.divide(
        accel_step,
        mid_dt,
        out=np.full_like(accel_step, np.nan),
        where=valid_accel,
    )
    return speed, accel, path_len


def _feature_columns(rows: pd.DataFrame) -> list[str]:
    excluded = {
        "sequence_id",
        "oracle_weight",
        "oracle_mse",
        "oracle_rmse",
        "oracle_mean",
        "oracle_p95",
        "oracle_max",
        "base_mse",
        "base_rmse",
        "base_mean",
        "base_p95",
        "base_max",
        "alternate_mse",
        "alternate_rmse",
        "alternate_mean",
        "alternate_p95",
        "alternate_max",
    }
    columns: list[str] = []
    for column in rows.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        if np.isfinite(values.to_numpy(float)).any():
            columns.append(str(column))
    return columns


def _feature_shift_table(
    train_rows: pd.DataFrame,
    apply_rows: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Summarize train/apply feature distribution shift for gate inputs."""

    records: list[dict[str, Any]] = []
    for column in feature_columns:
        if column not in train_rows.columns or column not in apply_rows.columns:
            continue
        train = _finite_series(train_rows[column])
        apply = _finite_series(apply_rows[column])
        if train.size == 0 or apply.size == 0:
            continue
        train_min = float(np.min(train))
        train_max = float(np.max(train))
        train_p05 = float(np.percentile(train, 5.0))
        train_p50 = float(np.percentile(train, 50.0))
        train_p95 = float(np.percentile(train, 95.0))
        train_iqr = float(np.percentile(train, 75.0) - np.percentile(train, 25.0))
        robust_scale = train_iqr if train_iqr > 1.0e-12 else float(np.std(train))
        if robust_scale <= 1.0e-12 or not np.isfinite(robust_scale):
            robust_scale = 1.0
        below_min = int(np.count_nonzero(apply < train_min))
        above_max = int(np.count_nonzero(apply > train_max))
        robust_z = np.abs((apply - train_p50) / robust_scale)
        records.append(
            {
                "feature": str(column),
                "train_count": int(train.size),
                "apply_count": int(apply.size),
                "train_min": train_min,
                "train_p05": train_p05,
                "train_p50": train_p50,
                "train_p95": train_p95,
                "train_max": train_max,
                "apply_min": float(np.min(apply)),
                "apply_p05": float(np.percentile(apply, 5.0)),
                "apply_p50": float(np.percentile(apply, 50.0)),
                "apply_p95": float(np.percentile(apply, 95.0)),
                "apply_max": float(np.max(apply)),
                "apply_below_train_min_count": below_min,
                "apply_above_train_max_count": above_max,
                "apply_outside_train_range_count": below_min + above_max,
                "apply_outside_train_range_fraction": float((below_min + above_max) / apply.size),
                "apply_mean_abs_robust_z": float(np.mean(robust_z)),
                "apply_max_abs_robust_z": float(np.max(robust_z)),
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "feature",
                "train_count",
                "apply_count",
                "apply_outside_train_range_count",
                "apply_outside_train_range_fraction",
                "apply_mean_abs_robust_z",
                "apply_max_abs_robust_z",
            ]
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["apply_outside_train_range_fraction", "apply_max_abs_robust_z", "feature"],
        ascending=[False, False, True],
    )


def _apply_sequence_shift_table(
    train_rows: pd.DataFrame,
    apply_rows: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Summarize how far each apply sequence is from the train feature envelope."""

    if apply_rows.empty:
        return pd.DataFrame(
            columns=[
                "sequence_id",
                "feature_count",
                "outside_train_range_count",
                "outside_train_range_fraction",
                "mean_abs_robust_z",
                "max_abs_robust_z",
                "worst_shift_feature",
            ]
        )
    train_stats: dict[str, dict[str, float]] = {}
    for column in feature_columns:
        if column not in train_rows.columns or column not in apply_rows.columns:
            continue
        train = _finite_series(train_rows[column])
        if train.size == 0:
            continue
        iqr = float(np.percentile(train, 75.0) - np.percentile(train, 25.0))
        scale = iqr if iqr > 1.0e-12 else float(np.std(train))
        if scale <= 1.0e-12 or not np.isfinite(scale):
            scale = 1.0
        train_stats[column] = {
            "min": float(np.min(train)),
            "max": float(np.max(train)),
            "median": float(np.percentile(train, 50.0)),
            "scale": float(scale),
        }
    records: list[dict[str, Any]] = []
    for row in apply_rows.itertuples(index=False):
        sequence_id = str(getattr(row, "sequence_id"))
        outside_count = 0
        robust_z_values: list[float] = []
        worst_feature = ""
        worst_shift = -np.inf
        for column, stats in train_stats.items():
            value = getattr(row, column, np.nan)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(numeric):
                continue
            if numeric < stats["min"] or numeric > stats["max"]:
                outside_count += 1
            robust_z = abs((numeric - stats["median"]) / stats["scale"])
            robust_z_values.append(float(robust_z))
            if robust_z > worst_shift:
                worst_shift = float(robust_z)
                worst_feature = str(column)
        feature_count = len(robust_z_values)
        records.append(
            {
                "sequence_id": sequence_id,
                "feature_count": int(feature_count),
                "outside_train_range_count": int(outside_count),
                "outside_train_range_fraction": float(outside_count / feature_count)
                if feature_count
                else float("nan"),
                "mean_abs_robust_z": float(np.mean(robust_z_values))
                if robust_z_values
                else float("nan"),
                "max_abs_robust_z": float(np.max(robust_z_values))
                if robust_z_values
                else float("nan"),
                "worst_shift_feature": worst_feature,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["outside_train_range_fraction", "max_abs_robust_z", "sequence_id"],
        ascending=[False, False, True],
    )


def _feature_shift_summary(feature_shift: pd.DataFrame | None) -> dict[str, Any]:
    if feature_shift is None or feature_shift.empty:
        return {
            "feature_count": 0,
            "shifted_feature_count": 0,
            "max_outside_train_range_fraction": 0.0,
            "max_abs_robust_z": 0.0,
        }
    outside = pd.to_numeric(
        feature_shift["apply_outside_train_range_fraction"],
        errors="coerce",
    ).fillna(0.0)
    robust_z = pd.to_numeric(feature_shift["apply_max_abs_robust_z"], errors="coerce").fillna(0.0)
    shifted = feature_shift.loc[outside > 0.0, "feature"].astype(str).tolist()
    return {
        "feature_count": int(len(feature_shift)),
        "shifted_feature_count": int(len(shifted)),
        "shifted_features": shifted,
        "max_outside_train_range_fraction": float(outside.max()),
        "max_abs_robust_z": float(robust_z.max()),
    }


def _apply_sequence_shift_summary(apply_shift: pd.DataFrame | None) -> dict[str, Any]:
    if apply_shift is None or apply_shift.empty:
        return {
            "sequence_count": 0,
            "shifted_sequence_count": 0,
            "max_outside_train_range_fraction": 0.0,
            "max_abs_robust_z": 0.0,
        }
    outside = pd.to_numeric(
        apply_shift["outside_train_range_fraction"],
        errors="coerce",
    ).fillna(0.0)
    robust_z = pd.to_numeric(apply_shift["max_abs_robust_z"], errors="coerce").fillna(0.0)
    shifted = apply_shift.loc[outside > 0.0, "sequence_id"].astype(str).tolist()
    return {
        "sequence_count": int(len(apply_shift)),
        "shifted_sequence_count": int(len(shifted)),
        "shifted_sequences": shifted,
        "max_outside_train_range_fraction": float(outside.max()),
        "max_abs_robust_z": float(robust_z.max()),
    }


def _finite_series(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return numeric[np.isfinite(numeric)]


def _predict_same_split_weights(
    model_name: str,
    rows: pd.DataFrame,
    feature_columns: list[str],
    *,
    random_state: int,
    min_weight: float,
    max_weight: float,
) -> pd.DataFrame:
    model = _make_model(model_name, random_state=random_state)
    x = _feature_matrix(rows, feature_columns)
    y = rows["oracle_weight"].to_numpy(float)
    model.fit(x, y)
    predicted = model.predict(x)
    return _weight_table(rows["sequence_id"], predicted, min_weight=min_weight, max_weight=max_weight)


def _predict_loso_weights(
    model_name: str,
    rows: pd.DataFrame,
    feature_columns: list[str],
    *,
    random_state: int,
    min_weight: float,
    max_weight: float,
) -> pd.DataFrame:
    predictions: list[dict[str, Any]] = []
    work = rows.reset_index(drop=True)
    for idx, row in work.iterrows():
        train_rows = work.drop(index=idx)
        if train_rows.empty:
            value = float(row["oracle_weight"])
        else:
            model = _make_model(model_name, random_state=random_state)
            model.fit(_feature_matrix(train_rows, feature_columns), train_rows["oracle_weight"])
            value = float(model.predict(_feature_matrix(pd.DataFrame([row]), feature_columns))[0])
        predictions.append({"sequence_id": str(row["sequence_id"]), "blend_weight": value})
    return _weight_table(
        pd.Series([row["sequence_id"] for row in predictions]),
        np.asarray([row["blend_weight"] for row in predictions], dtype=float),
        min_weight=min_weight,
        max_weight=max_weight,
    )


def _predict_apply_weights(
    model_name: str,
    train_rows: pd.DataFrame,
    apply_rows: pd.DataFrame,
    feature_columns: list[str],
    *,
    random_state: int,
    min_weight: float,
    max_weight: float,
) -> pd.DataFrame:
    if apply_rows.empty:
        return pd.DataFrame(columns=["sequence_id", "blend_weight"])
    model = _make_model(model_name, random_state=random_state)
    model.fit(_feature_matrix(train_rows, feature_columns), train_rows["oracle_weight"])
    predicted = model.predict(_feature_matrix(apply_rows, feature_columns))
    return _weight_table(
        apply_rows["sequence_id"],
        np.asarray(predicted, dtype=float),
        min_weight=min_weight,
        max_weight=max_weight,
    )


def _make_model(model_name: str, *, random_state: int) -> Any:
    try:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeRegressor
    except ModuleNotFoundError:
        return _make_numpy_model(model_name, random_state=random_state)

    name = str(model_name)
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name.startswith("tree_d"):
        depth, leaf = _parse_depth_leaf(name, prefix="tree")
        return DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=random_state,
        )
    if name.startswith("rf_depth"):
        depth = int(name.removeprefix("rf_depth"))
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=depth,
            min_samples_leaf=1,
            random_state=random_state,
        )
    if name.startswith("extra_depth"):
        depth = int(name.removeprefix("extra_depth"))
        return ExtraTreesRegressor(
            n_estimators=200,
            max_depth=depth,
            min_samples_leaf=1,
            random_state=random_state,
        )
    raise ValueError(f"unsupported sequence-gate model {model_name!r}")


def _make_numpy_model(model_name: str, *, random_state: int) -> Any:
    name = str(model_name)
    if name == "ridge":
        return _NumpyRidgeRegressor(alpha=1.0)
    if name.startswith("tree_d"):
        depth, leaf = _parse_depth_leaf(name, prefix="tree")
        return _NumpyDecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf)
    if name.startswith("rf_depth"):
        depth = int(name.removeprefix("rf_depth"))
        return _NumpyForestRegressor(max_depth=depth, random_state=random_state, extra=False)
    if name.startswith("extra_depth"):
        depth = int(name.removeprefix("extra_depth"))
        return _NumpyForestRegressor(max_depth=depth, random_state=random_state, extra=True)
    raise ValueError(f"unsupported sequence-gate model {model_name!r}")


@dataclass
class _NumpyRidgeRegressor:
    alpha: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_NumpyRidgeRegressor":
        matrix = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        self._x_mean = np.mean(matrix, axis=0)
        self._x_scale = np.std(matrix, axis=0)
        self._x_scale = np.where(self._x_scale > 1.0e-12, self._x_scale, 1.0)
        self._y_mean = float(np.mean(target))
        design = (matrix - self._x_mean) / self._x_scale
        gram = design.T @ design
        penalty = float(self.alpha) * np.eye(gram.shape[0], dtype=float)
        rhs = design.T @ (target - self._y_mean)
        self._coef = np.linalg.pinv(gram + penalty) @ rhs
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=float)
        design = (matrix - self._x_mean) / self._x_scale
        return np.asarray(self._y_mean + design @ self._coef, dtype=float)


@dataclass
class _TreeNode:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None


@dataclass
class _NumpyDecisionTreeRegressor:
    max_depth: int
    min_samples_leaf: int = 1

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_NumpyDecisionTreeRegressor":
        matrix = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        self._root = self._fit_node(matrix, target, depth=int(self.max_depth))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        matrix = np.asarray(x, dtype=float)
        return np.asarray([self._predict_row(row, self._root) for row in matrix], dtype=float)

    def _fit_node(self, x: np.ndarray, y: np.ndarray, *, depth: int) -> _TreeNode:
        value = float(np.mean(y)) if y.size else 0.0
        if depth <= 0 or len(y) < 2 * int(self.min_samples_leaf) or np.allclose(y, value):
            return _TreeNode(value=value)
        split = self._best_split(x, y)
        if split is None:
            return _TreeNode(value=value)
        feature, threshold = split
        mask = x[:, feature] <= threshold
        return _TreeNode(
            value=value,
            feature=int(feature),
            threshold=float(threshold),
            left=self._fit_node(x[mask], y[mask], depth=depth - 1),
            right=self._fit_node(x[~mask], y[~mask], depth=depth - 1),
        )

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
        best_loss = _squared_error_sum(y)
        best: tuple[int, float] | None = None
        for feature in range(x.shape[1]):
            values = np.unique(x[:, feature])
            if values.size < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                mask = x[:, feature] <= threshold
                left_count = int(np.count_nonzero(mask))
                right_count = int(len(mask) - left_count)
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue
                loss = _squared_error_sum(y[mask]) + _squared_error_sum(y[~mask])
                if loss < best_loss - 1.0e-12:
                    best_loss = float(loss)
                    best = (int(feature), float(threshold))
        return best

    def _predict_row(self, row: np.ndarray, node: _TreeNode) -> float:
        current = node
        while current.feature is not None and current.threshold is not None:
            child = current.left if row[current.feature] <= current.threshold else current.right
            if child is None:
                break
            current = child
        return float(current.value)


@dataclass
class _NumpyForestRegressor:
    max_depth: int
    random_state: int
    extra: bool = False
    n_estimators: int = 31

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_NumpyForestRegressor":
        matrix = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        rng = np.random.default_rng(int(self.random_state))
        self._trees: list[_NumpyDecisionTreeRegressor] = []
        for _ in range(int(self.n_estimators)):
            indices = rng.integers(0, len(target), size=len(target)) if len(target) else []
            sample_x = matrix[indices]
            if self.extra and sample_x.size:
                sample_x = sample_x + rng.normal(0.0, 1.0e-9, size=sample_x.shape)
            tree = _NumpyDecisionTreeRegressor(
                max_depth=int(self.max_depth),
                min_samples_leaf=1,
            )
            tree.fit(sample_x, target[indices])
            self._trees.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._trees:
            return np.zeros(len(x), dtype=float)
        stacked = np.vstack([tree.predict(x) for tree in self._trees])
        return np.mean(stacked, axis=0)


def _squared_error_sum(values: np.ndarray) -> float:
    target = np.asarray(values, dtype=float)
    if target.size == 0:
        return 0.0
    centered = target - float(np.mean(target))
    return float(np.sum(centered**2))


def _parse_depth_leaf(name: str, *, prefix: str) -> tuple[int, int]:
    # tree_d4_leaf2 -> (4, 2)
    rest = name.removeprefix(f"{prefix}_d")
    depth_text, leaf_text = rest.split("_leaf", 1)
    return int(depth_text), int(leaf_text)


def _feature_matrix(rows: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    matrix = rows[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if matrix.ndim != 2:
        matrix = matrix.reshape(len(rows), -1)
    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    row_idx, col_idx = np.where(~np.isfinite(matrix))
    matrix[row_idx, col_idx] = medians[col_idx]
    return matrix


def _weight_table(
    sequences: pd.Series,
    weights: np.ndarray,
    *,
    min_weight: float,
    max_weight: float,
) -> pd.DataFrame:
    values = np.clip(np.asarray(weights, dtype=float), min_weight, max_weight)
    return pd.DataFrame(
        {
            "sequence_id": pd.Series(sequences).astype(str).to_numpy(),
            "blend_weight": values.astype(float),
        }
    ).sort_values("sequence_id")


def _score_weight_table(
    base: pd.DataFrame,
    alternate: pd.DataFrame,
    truth: pd.DataFrame,
    weights: pd.DataFrame,
) -> dict[str, float]:
    weight_map = {
        str(row.sequence_id): float(row.blend_weight)
        for row in weights.itertuples(index=False)
    }
    weight_values = np.asarray([weight_map[str(sequence)] for sequence in base["sequence_id"]])
    base_xyz = base[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    alt_xyz = alternate[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    truth_xyz = truth[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    xyz = (1.0 - weight_values[:, None]) * base_xyz + weight_values[:, None] * alt_xyz
    errors = np.linalg.norm(xyz - truth_xyz, axis=1)
    return _pose_metrics(errors)


def _pose_metrics(errors: np.ndarray) -> dict[str, float]:
    values = np.asarray(errors, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mse": np.nan, "rmse": np.nan, "mean": np.nan, "p95": np.nan, "max": np.nan}
    mse = float(np.mean(values**2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def _weights_text(weights: pd.DataFrame) -> str:
    return ";".join(
        f"{row.sequence_id}:{float(row.blend_weight):.6g}"
        for row in weights.sort_values("sequence_id").itertuples(index=False)
    )


def _safe_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _safe_std(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.std(finite)) if finite.size else float("nan")


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _safe_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1.0e-12:
        return float("nan")
    return float(numerator / denominator)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
