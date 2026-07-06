"""Fit sequence-level blend weights for Track 5 estimate trajectories.

The existing estimate sequence gate applies a sequence -> blend-weight table to
combine two pose trajectories.  This module creates those weights on a labeled
split and can predict weights for an unlabeled apply split from non-truth
trajectory-disagreement features.  It is intended for train-fold selection before
Codabench/UG2+ Track 5 submission generation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_estimate_sequence_gate import (
    blend_track5_estimate_sequence_gate,
    write_track5_estimate_sequence_gate_outputs,
)
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

SUMMARY_CSV = "mmuad_track5_estimate_sequence_gate_fit_summary.csv"
SUMMARY_JSON = "mmuad_track5_estimate_sequence_gate_fit_summary.json"
TRAIN_FEATURES_CSV = "mmuad_track5_estimate_sequence_gate_train_features.csv"
ORACLE_WEIGHTS_CSV = "mmuad_track5_estimate_sequence_gate_oracle_weights.csv"
LOSO_WEIGHTS_CSV = "mmuad_track5_estimate_sequence_gate_loso_weights.csv"
APPLY_FEATURES_CSV = "mmuad_track5_estimate_sequence_gate_apply_features.csv"
APPLY_WEIGHTS_CSV = "mmuad_track5_estimate_sequence_gate_apply_weights.csv"
APPLY_OUTPUT_DIR = "estimate_sequence_gate_apply_submission"
DEFAULT_WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FEATURE_COLUMNS = (
    "row_count",
    "valid_pair_fraction",
    "mean_displacement_m",
    "p50_displacement_m",
    "p95_displacement_m",
    "max_displacement_m",
    "std_displacement_m",
    "mean_base_step_m",
    "mean_alternate_step_m",
    "base_path_length_m",
    "alternate_path_length_m",
    "path_length_ratio",
    "time_span_s",
)


@dataclass(frozen=True)
class EstimateSequenceGateFitResult:
    """Artifacts from estimate-level sequence-gate fitting."""

    summary: pd.DataFrame
    train_features: pd.DataFrame
    oracle_weights: pd.DataFrame
    loso_weights: pd.DataFrame
    apply_features: pd.DataFrame | None = None
    apply_weights: pd.DataFrame | None = None


def fit_track5_estimate_sequence_gate(
    *,
    base_estimates: pd.DataFrame,
    alternate_estimates: pd.DataFrame,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    weight_grid: np.ndarray | None = None,
    apply_base_estimates: pd.DataFrame | None = None,
    apply_alternate_estimates: pd.DataFrame | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> EstimateSequenceGateFitResult:
    """Fit sequence blend weights and optional apply-split predictions.

    ``truth`` is used only for the labeled training split.  If apply estimates
    are supplied, weights are predicted by nearest-neighbor matching in
    non-truth trajectory feature space using the train oracle weights.
    """

    if (apply_base_estimates is None) != (apply_alternate_estimates is None):
        raise ValueError("apply base/alternate estimates must be provided together")
    grid = _parse_weight_grid(weight_grid)
    base, alternate = _resampled_pair(
        base_estimates,
        alternate_estimates,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    truth_rows = _normalize_truth_rows(truth)
    train_features = _sequence_feature_table(base, alternate)
    oracle_weights = _oracle_sequence_weights(base, alternate, truth_rows, grid)
    train_features = train_features.merge(oracle_weights, on="sequence_id", how="left")
    loso_weights = _loso_nearest_neighbor_weights(train_features)
    oracle_metrics = _score_weight_table(base, alternate, truth_rows, oracle_weights)
    loso_metrics = _score_weight_table(base, alternate, truth_rows, loso_weights)
    summary = pd.DataFrame.from_records(
        [
            {
                "model": "oracle_same_split",
                "weight_grid": ",".join(f"{value:g}" for value in grid),
                **{f"metric_{key}": value for key, value in oracle_metrics.items()},
            },
            {
                "model": "nearest_neighbor_loso",
                "weight_grid": ",".join(f"{value:g}" for value in grid),
                **{f"metric_{key}": value for key, value in loso_metrics.items()},
            },
        ]
    )
    apply_features = None
    apply_weights = None
    if apply_base_estimates is not None and apply_alternate_estimates is not None:
        apply_base, apply_alt = _resampled_pair(
            apply_base_estimates,
            apply_alternate_estimates,
            template,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        apply_features = _sequence_feature_table(apply_base, apply_alt)
        apply_weights = _predict_apply_nearest_neighbor(train_features, apply_features)
    return EstimateSequenceGateFitResult(
        summary=summary,
        train_features=train_features.sort_values("sequence_id").reset_index(drop=True),
        oracle_weights=oracle_weights.sort_values("sequence_id").reset_index(drop=True),
        loso_weights=loso_weights.sort_values("sequence_id").reset_index(drop=True),
        apply_features=None if apply_features is None else apply_features.sort_values("sequence_id").reset_index(drop=True),
        apply_weights=None if apply_weights is None else apply_weights.sort_values("sequence_id").reset_index(drop=True),
    )


def write_track5_estimate_sequence_gate_fit_outputs(
    *,
    result: EstimateSequenceGateFitResult,
    output_dir: Path,
    base_estimates_path: Path,
    alternate_estimates_path: Path,
    template_path: Path,
    truth_path: Path,
    weight_grid: np.ndarray,
    apply_base_estimates_path: Path | None = None,
    apply_alternate_estimates_path: Path | None = None,
    write_apply_submission: bool = False,
    template: pd.DataFrame | None = None,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    """Write fit diagnostics, weight tables, and optional apply submission."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": output / SUMMARY_CSV,
        "summary_json": output / SUMMARY_JSON,
        "train_features_csv": output / TRAIN_FEATURES_CSV,
        "oracle_weights_csv": output / ORACLE_WEIGHTS_CSV,
        "loso_weights_csv": output / LOSO_WEIGHTS_CSV,
    }
    result.summary.to_csv(paths["summary_csv"], index=False)
    result.train_features.to_csv(paths["train_features_csv"], index=False)
    result.oracle_weights.to_csv(paths["oracle_weights_csv"], index=False)
    result.loso_weights.to_csv(paths["loso_weights_csv"], index=False)
    if result.apply_features is not None:
        paths["apply_features_csv"] = output / APPLY_FEATURES_CSV
        result.apply_features.to_csv(paths["apply_features_csv"], index=False)
    if result.apply_weights is not None:
        paths["apply_weights_csv"] = output / APPLY_WEIGHTS_CSV
        result.apply_weights.to_csv(paths["apply_weights_csv"], index=False)
    if write_apply_submission:
        if apply_base_estimates_path is None or apply_alternate_estimates_path is None:
            raise ValueError("apply estimates are required to write an apply submission")
        if result.apply_weights is None:
            raise ValueError("apply weights are unavailable")
        if template is None:
            template = load_official_track5_template_file(template_path)
        estimates, diagnostics, weights = blend_track5_estimate_sequence_gate(
            base_estimates=pd.read_csv(apply_base_estimates_path),
            alternate_estimates=pd.read_csv(apply_alternate_estimates_path),
            template=template,
            sequence_weights=result.apply_weights,
            default_weight=0.0,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        apply_paths = write_track5_estimate_sequence_gate_outputs(
            estimates=estimates,
            diagnostics=diagnostics,
            sequence_weights=weights,
            output_dir=output / APPLY_OUTPUT_DIR,
            base_estimates_path=apply_base_estimates_path,
            alternate_estimates_path=apply_alternate_estimates_path,
            sequence_weights_path=paths["apply_weights_csv"],
            template=template,
            class_map=class_map or {},
            default_classification=default_classification,
            require_leaderboard_ready=require_leaderboard_ready,
        )
        paths.update({f"apply_{name}": path for name, path in apply_paths.items()})
    payload = {
        "schema": "raft-uav-mmuad-track5-estimate-sequence-gate-fit-v1",
        "protocol": "train truth selects oracle sequence weights; nearest-neighbor feature matching predicts apply weights",
        "base_estimates": str(base_estimates_path),
        "alternate_estimates": str(alternate_estimates_path),
        "apply_base_estimates": None if apply_base_estimates_path is None else str(apply_base_estimates_path),
        "apply_alternate_estimates": None if apply_alternate_estimates_path is None else str(apply_alternate_estimates_path),
        "template": str(template_path),
        "truth": str(truth_path),
        "weight_grid": [float(value) for value in weight_grid],
        "feature_columns": list(FEATURE_COLUMNS),
        "train_sequence_count": int(result.train_features["sequence_id"].nunique()),
        "apply_sequence_count": 0 if result.apply_weights is None else int(result.apply_weights["sequence_id"].nunique()),
        "summary": _jsonable(result.summary.to_dict(orient="records")),
        "paths": {name: str(path) for name, path in paths.items() if name != "summary_json"},
    }
    paths["summary_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-estimate-sequence-gate-fit",
        description="fit Track 5 estimate-level sequence-gate weights on a labeled split",
    )
    parser.add_argument("--base-estimates", type=Path, required=True)
    parser.add_argument("--alternate-estimates", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-grid", default=",".join(f"{value:g}" for value in DEFAULT_WEIGHT_GRID))
    parser.add_argument("--apply-base-estimates", type=Path)
    parser.add_argument("--apply-alternate-estimates", type=Path)
    parser.add_argument("--write-apply-submission", action="store_true")
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    template = load_official_track5_template_file(args.template)
    result = fit_track5_estimate_sequence_gate(
        base_estimates=pd.read_csv(args.base_estimates),
        alternate_estimates=pd.read_csv(args.alternate_estimates),
        template=template,
        truth=load_evaluation_truth_file(args.truth_csv).rows,
        weight_grid=_parse_weight_grid_text(args.weight_grid),
        apply_base_estimates=None if args.apply_base_estimates is None else pd.read_csv(args.apply_base_estimates),
        apply_alternate_estimates=None if args.apply_alternate_estimates is None else pd.read_csv(args.apply_alternate_estimates),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_estimate_sequence_gate_fit_outputs(
        result=result,
        output_dir=args.output_dir,
        base_estimates_path=args.base_estimates,
        alternate_estimates_path=args.alternate_estimates,
        apply_base_estimates_path=args.apply_base_estimates,
        apply_alternate_estimates_path=args.apply_alternate_estimates,
        template_path=args.template,
        truth_path=args.truth_csv,
        weight_grid=_parse_weight_grid_text(args.weight_grid),
        write_apply_submission=bool(args.write_apply_submission),
        template=template,
        class_map=class_map,
        default_classification=args.default_classification,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    print("mmuad_track5_estimate_sequence_gate_fit=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _resampled_pair(
    base_estimates: pd.DataFrame,
    alternate_estimates: pd.DataFrame,
    template: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base, _ = resample_estimates_to_track5_template(
        base_estimates,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    alternate, _ = resample_estimates_to_track5_template(
        alternate_estimates,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    base = _rename_state_columns(base, prefix="base")
    alternate = _rename_state_columns(alternate, prefix="alternate")
    keys = ["sequence_id", "time_s"]
    merged = base.merge(alternate, on=keys, how="inner")
    if len(merged) != len(base) or len(merged) != len(alternate):
        raise ValueError("base and alternate estimates do not align after template resampling")
    base_cols = keys + ["base_x_m", "base_y_m", "base_z_m"]
    alt_cols = keys + ["alternate_x_m", "alternate_y_m", "alternate_z_m"]
    return merged[base_cols].copy(), merged[alt_cols].copy()


def _rename_state_columns(rows: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    out = rows.copy()
    return out.rename(
        columns={
            "state_x_m": f"{prefix}_x_m",
            "state_y_m": f"{prefix}_y_m",
            "state_z_m": f"{prefix}_z_m",
        }
    )


def _sequence_feature_table(base: pd.DataFrame, alternate: pd.DataFrame) -> pd.DataFrame:
    rows = base.merge(alternate, on=["sequence_id", "time_s"], how="inner")
    records: list[dict[str, Any]] = []
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        base_xyz = group[["base_x_m", "base_y_m", "base_z_m"]].to_numpy(float)
        alt_xyz = group[["alternate_x_m", "alternate_y_m", "alternate_z_m"]].to_numpy(float)
        valid = np.isfinite(base_xyz).all(axis=1) & np.isfinite(alt_xyz).all(axis=1)
        displacement = np.linalg.norm(base_xyz[valid] - alt_xyz[valid], axis=1)
        base_steps = _step_lengths(base_xyz[valid])
        alt_steps = _step_lengths(alt_xyz[valid])
        time_values = pd.to_numeric(group["time_s"], errors="coerce").to_numpy(float)
        record = {
            "sequence_id": str(sequence_id),
            "row_count": int(len(group)),
            "valid_pair_count": int(valid.sum()),
            "valid_pair_fraction": float(valid.mean()) if len(valid) else 0.0,
            "mean_displacement_m": _safe_mean_array(displacement),
            "p50_displacement_m": _safe_percentile_array(displacement, 50),
            "p95_displacement_m": _safe_percentile_array(displacement, 95),
            "max_displacement_m": _safe_max_array(displacement),
            "std_displacement_m": _safe_std_array(displacement),
            "mean_base_step_m": _safe_mean_array(base_steps),
            "mean_alternate_step_m": _safe_mean_array(alt_steps),
            "base_path_length_m": _safe_sum_array(base_steps),
            "alternate_path_length_m": _safe_sum_array(alt_steps),
            "time_span_s": _time_span(time_values),
        }
        denom = record["base_path_length_m"]
        record["path_length_ratio"] = (
            float(record["alternate_path_length_m"] / denom)
            if denom is not None and np.isfinite(denom) and denom > 0.0
            else np.nan
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _oracle_sequence_weights(
    base: pd.DataFrame,
    alternate: pd.DataFrame,
    truth: pd.DataFrame,
    weight_grid: np.ndarray,
) -> pd.DataFrame:
    rows = _aligned_truth_frame(base, alternate, truth)
    records: list[dict[str, Any]] = []
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        best: dict[str, Any] | None = None
        for weight in weight_grid:
            metrics = _metrics_for_weight(group, float(weight))
            candidate = {"sequence_id": str(sequence_id), "sequence_gate_weight": float(weight), **metrics}
            if best is None or candidate["pose_mse_m2"] < best["pose_mse_m2"]:
                best = candidate
        if best is not None:
            records.append(best)
    return pd.DataFrame.from_records(records)


def _loso_nearest_neighbor_weights(train_features: pd.DataFrame) -> pd.DataFrame:
    if train_features["sequence_id"].nunique() < 2:
        raise ValueError("LOSO estimate sequence-gate fit needs at least two sequences")
    records: list[dict[str, Any]] = []
    for held_out in train_features["sequence_id"].astype(str):
        train = train_features.loc[train_features["sequence_id"].astype(str) != held_out]
        apply = train_features.loc[train_features["sequence_id"].astype(str) == held_out]
        predicted = _predict_apply_nearest_neighbor(train, apply).iloc[0].to_dict()
        predicted["sequence_id"] = held_out
        predicted["prediction_protocol"] = "loso-nearest-neighbor"
        records.append(predicted)
    return pd.DataFrame.from_records(records)


def _predict_apply_nearest_neighbor(train_features: pd.DataFrame, apply_features: pd.DataFrame) -> pd.DataFrame:
    train = train_features.copy()
    apply = apply_features.copy()
    columns = [column for column in FEATURE_COLUMNS if column in train.columns and column in apply.columns]
    if not columns:
        raise ValueError("no shared sequence-gate feature columns available")
    train_matrix = train[columns].apply(pd.to_numeric, errors="coerce")
    apply_matrix = apply[columns].apply(pd.to_numeric, errors="coerce")
    means = train_matrix.mean(axis=0).fillna(0.0)
    stds = train_matrix.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    train_z = ((train_matrix.fillna(means) - means) / stds).to_numpy(float)
    apply_z = ((apply_matrix.fillna(means) - means) / stds).to_numpy(float)
    records: list[dict[str, Any]] = []
    for index, sequence_id in enumerate(apply["sequence_id"].astype(str)):
        distances = np.linalg.norm(train_z - apply_z[index][None, :], axis=1)
        nearest_index = int(np.nanargmin(distances))
        nearest = train.iloc[nearest_index]
        records.append(
            {
                "sequence_id": sequence_id,
                "sequence_gate_weight": float(nearest["sequence_gate_weight"]),
                "nearest_train_sequence_id": str(nearest["sequence_id"]),
                "nearest_train_distance": float(distances[nearest_index]),
                "prediction_protocol": "nearest-neighbor-sequence-features",
            }
        )
    return pd.DataFrame.from_records(records)


def _score_weight_table(
    base: pd.DataFrame,
    alternate: pd.DataFrame,
    truth: pd.DataFrame,
    weights: pd.DataFrame,
) -> dict[str, Any]:
    rows = _aligned_truth_frame(base, alternate, truth).merge(
        weights[["sequence_id", "sequence_gate_weight"]],
        on="sequence_id",
        how="inner",
    )
    if rows.empty:
        return _empty_metrics()
    blended = _blend_rows(rows, rows["sequence_gate_weight"].to_numpy(float))
    truth_xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(float)
    finite = np.isfinite(blended).all(axis=1) & np.isfinite(truth_xyz).all(axis=1)
    if not finite.any():
        return _empty_metrics()
    errors = np.linalg.norm(blended[finite] - truth_xyz[finite], axis=1)
    return _error_metrics(errors)


def _aligned_truth_frame(base: pd.DataFrame, alternate: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    truth_rows = _normalize_truth_rows(truth)
    rows = base.merge(alternate, on=["sequence_id", "time_s"], how="inner")
    rows["_time_key"] = _time_key(rows["time_s"])
    truth_rows["_time_key"] = _time_key(truth_rows["time_s"])
    return rows.merge(
        truth_rows[["sequence_id", "_time_key", "x_m", "y_m", "z_m"]],
        on=["sequence_id", "_time_key"],
        how="inner",
    )


def _metrics_for_weight(rows: pd.DataFrame, weight: float) -> dict[str, Any]:
    blended = _blend_rows(rows, np.full(len(rows), float(weight)))
    truth_xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(float)
    finite = np.isfinite(blended).all(axis=1) & np.isfinite(truth_xyz).all(axis=1)
    if not finite.any():
        return _empty_metrics()
    errors = np.linalg.norm(blended[finite] - truth_xyz[finite], axis=1)
    return _error_metrics(errors)


def _blend_rows(rows: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    base_xyz = rows[["base_x_m", "base_y_m", "base_z_m"]].to_numpy(float)
    alt_xyz = rows[["alternate_x_m", "alternate_y_m", "alternate_z_m"]].to_numpy(float)
    return (1.0 - weights[:, None]) * base_xyz + weights[:, None] * alt_xyz


def _normalize_truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(truth).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s", "x_m", "y_m", "z_m"])
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite, ["sequence_id", "time_s", "x_m", "y_m", "z_m"]].reset_index(drop=True)


def _parse_weight_grid_text(text: str) -> np.ndarray:
    return _parse_weight_grid(np.asarray([float(item.strip()) for item in str(text).split(",") if item.strip()]))


def _parse_weight_grid(values: np.ndarray | None) -> np.ndarray:
    grid = np.asarray(DEFAULT_WEIGHT_GRID if values is None else values, dtype=float)
    if grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("weight grid must contain finite values")
    if (grid < 0.0).any() or (grid > 1.0).any():
        raise ValueError("weight grid values must be in [0, 1]")
    return np.unique(grid)


def _step_lengths(xyz: np.ndarray) -> np.ndarray:
    if len(xyz) < 2:
        return np.asarray([], dtype=float)
    return np.linalg.norm(np.diff(xyz, axis=0), axis=1)


def _error_metrics(errors: np.ndarray) -> dict[str, Any]:
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


def _time_span(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.max(values) - np.min(values))


def _safe_mean_array(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def _safe_percentile_array(values: np.ndarray, percentile: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if len(values) else np.nan


def _safe_max_array(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.max(values)) if len(values) else np.nan


def _safe_std_array(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.std(values)) if len(values) else np.nan


def _safe_sum_array(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.sum(values)) if len(values) else np.nan


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
