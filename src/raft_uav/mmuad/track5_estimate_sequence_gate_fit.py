"""Fit estimate-level sequence-gate weights for MMUAD Track 5.

The estimate sequence gate applies a sequence -> blend-weight table to combine
base and alternate pose trajectories.  This module creates such tables from a
labeled split and can predict weights for an unlabeled apply split using only
trajectory-disagreement features.  Use it on train folds before Codabench / UG2+
Track 5 validation or hidden-test submission generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

SUMMARY_CSV = "mmuad_track5_estimate_sequence_gate_fit_summary.csv"
TRAIN_FEATURES_CSV = "mmuad_track5_estimate_sequence_gate_train_features.csv"
ORACLE_WEIGHTS_CSV = "mmuad_track5_estimate_sequence_gate_oracle_weights.csv"
LOSO_WEIGHTS_CSV = "mmuad_track5_estimate_sequence_gate_loso_weights.csv"
APPLY_FEATURES_CSV = "mmuad_track5_estimate_sequence_gate_apply_features.csv"
APPLY_WEIGHTS_CSV = "mmuad_track5_estimate_sequence_gate_apply_weights.csv"
SUMMARY_JSON = "mmuad_track5_estimate_sequence_gate_fit_summary.json"
SEQUENCE_ALIASES = ("sequence_id", "Sequence", "sequence", "seq")
FEATURE_COLUMNS = (
    "row_count",
    "valid_pair_fraction",
    "mean_displacement_m",
    "p95_displacement_m",
    "max_displacement_m",
    "mean_base_step_m",
    "mean_alternate_step_m",
    "base_path_length_m",
    "alternate_path_length_m",
    "path_length_ratio",
)
DEFAULT_WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def fit_estimate_sequence_gate_weights(
    *,
    base_estimates: pd.DataFrame,
    alternate_estimates: pd.DataFrame,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    weight_grid: np.ndarray | None = None,
    apply_base_estimates: pd.DataFrame | None = None,
    apply_alternate_estimates: pd.DataFrame | None = None,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Fit sequence blend weights and optional apply-split predictions."""

    if (apply_base_estimates is None) != (apply_alternate_estimates is None):
        raise ValueError("apply base/alternate estimates must be provided together")
    grid = _weight_grid(weight_grid)
    base, alt = _resampled_pair(
        base_estimates,
        alternate_estimates,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    truth_rows = _truth_rows(truth)
    train_features = _sequence_features(base, alt)
    oracle_weights = _oracle_weights(base, alt, truth_rows, grid)
    train_features = train_features.merge(oracle_weights, on="sequence_id", how="left")
    loso_weights = _loso_weights(train_features)
    summary = pd.DataFrame.from_records(
        [
            {
                "model": "oracle_same_split",
                "weight_grid": ",".join(f"{w:g}" for w in grid),
                **{f"metric_{key}": value for key, value in _score_weights(base, alt, truth_rows, oracle_weights).items()},
            },
            {
                "model": "nearest_neighbor_loso",
                "weight_grid": ",".join(f"{w:g}" for w in grid),
                **{f"metric_{key}": value for key, value in _score_weights(base, alt, truth_rows, loso_weights).items()},
            },
        ]
    )
    result = {
        "summary": summary,
        "train_features": train_features.sort_values("sequence_id").reset_index(drop=True),
        "oracle_weights": oracle_weights.sort_values("sequence_id").reset_index(drop=True),
        "loso_weights": loso_weights.sort_values("sequence_id").reset_index(drop=True),
    }
    if apply_base_estimates is not None and apply_alternate_estimates is not None:
        apply_template = _template_for_apply_estimates(
            template,
            apply_base_estimates,
            apply_alternate_estimates,
        )
        apply_base, apply_alt = _resampled_pair(
            apply_base_estimates,
            apply_alternate_estimates,
            apply_template,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        apply_features = _sequence_features(apply_base, apply_alt)
        result["apply_features"] = apply_features.sort_values("sequence_id").reset_index(drop=True)
        result["apply_weights"] = _nearest_neighbor_predict(train_features, apply_features).sort_values(
            "sequence_id"
        ).reset_index(drop=True)
    return result


def _template_for_apply_estimates(
    template: pd.DataFrame,
    apply_base_estimates: pd.DataFrame,
    apply_alternate_estimates: pd.DataFrame,
) -> pd.DataFrame:
    base_sequences = _estimate_sequence_ids(apply_base_estimates, label="apply base estimates")
    alternate_sequences = _estimate_sequence_ids(
        apply_alternate_estimates,
        label="apply alternate estimates",
    )
    common_sequences = base_sequences.intersection(alternate_sequences)
    if not common_sequences:
        raise ValueError("apply base/alternate estimates must share at least one sequence")
    rows = pd.DataFrame(template).copy()
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    if sequence_column is None:
        raise ValueError("template must contain a sequence column")
    filtered = rows.loc[rows[sequence_column].astype(str).isin(common_sequences)].copy()
    if filtered.empty:
        raise ValueError("template has no rows for apply estimate sequences")
    return filtered


def _estimate_sequence_ids(estimates: pd.DataFrame, *, label: str) -> set[str]:
    rows = pd.DataFrame(estimates)
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    if sequence_column is None:
        raise ValueError(f"{label} must contain a sequence column")
    return set(rows[sequence_column].dropna().astype(str))


def _first_present(rows: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in rows.columns:
            return alias
    return None


def write_estimate_sequence_gate_fit_outputs(
    *,
    result: dict[str, pd.DataFrame],
    output_dir: Path,
    base_estimates_path: Path,
    alternate_estimates_path: Path,
    template_path: Path,
    truth_path: Path,
    weight_grid: np.ndarray,
    apply_base_estimates_path: Path | None = None,
    apply_alternate_estimates_path: Path | None = None,
) -> dict[str, Path]:
    """Write train features, oracle/LOSO weights, optional apply weights, and manifest."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": out / SUMMARY_CSV,
        "train_features_csv": out / TRAIN_FEATURES_CSV,
        "oracle_weights_csv": out / ORACLE_WEIGHTS_CSV,
        "loso_weights_csv": out / LOSO_WEIGHTS_CSV,
        "summary_json": out / SUMMARY_JSON,
    }
    result["summary"].to_csv(paths["summary_csv"], index=False)
    result["train_features"].to_csv(paths["train_features_csv"], index=False)
    result["oracle_weights"].to_csv(paths["oracle_weights_csv"], index=False)
    result["loso_weights"].to_csv(paths["loso_weights_csv"], index=False)
    if "apply_features" in result:
        paths["apply_features_csv"] = out / APPLY_FEATURES_CSV
        result["apply_features"].to_csv(paths["apply_features_csv"], index=False)
    if "apply_weights" in result:
        paths["apply_weights_csv"] = out / APPLY_WEIGHTS_CSV
        result["apply_weights"].to_csv(paths["apply_weights_csv"], index=False)
    payload = {
        "schema": "raft-uav-mmuad-track5-estimate-sequence-gate-fit-v1",
        "protocol": "train truth selects oracle sequence weights; nearest-neighbor sequence features predict apply weights",
        "base_estimates": str(base_estimates_path),
        "alternate_estimates": str(alternate_estimates_path),
        "apply_base_estimates": None if apply_base_estimates_path is None else str(apply_base_estimates_path),
        "apply_alternate_estimates": None
        if apply_alternate_estimates_path is None
        else str(apply_alternate_estimates_path),
        "template": str(template_path),
        "truth": str(truth_path),
        "weight_grid": [float(value) for value in weight_grid],
        "feature_columns": list(FEATURE_COLUMNS),
        "train_sequence_count": int(result["train_features"]["sequence_id"].nunique()),
        "apply_sequence_count": 0
        if "apply_weights" not in result
        else int(result["apply_weights"]["sequence_id"].nunique()),
        "summary": _jsonable(result["summary"].to_dict(orient="records")),
        "paths": {key: str(value) for key, value in paths.items() if key != "summary_json"},
    }
    paths["summary_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-estimate-sequence-gate-fit",
        description="fit estimate-level Track 5 sequence-gate weights",
    )
    parser.add_argument("--base-estimates", type=Path, required=True)
    parser.add_argument("--alternate-estimates", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weight-grid", default=",".join(f"{value:g}" for value in DEFAULT_WEIGHT_GRID))
    parser.add_argument("--apply-base-estimates", type=Path)
    parser.add_argument("--apply-alternate-estimates", type=Path)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    args = parser.parse_args(argv)

    template = load_official_track5_template_file(args.template)
    grid = _weight_grid_text(args.weight_grid)
    result = fit_estimate_sequence_gate_weights(
        base_estimates=pd.read_csv(args.base_estimates),
        alternate_estimates=pd.read_csv(args.alternate_estimates),
        template=template,
        truth=load_evaluation_truth_file(args.truth_csv).rows,
        weight_grid=grid,
        apply_base_estimates=None if args.apply_base_estimates is None else pd.read_csv(args.apply_base_estimates),
        apply_alternate_estimates=None
        if args.apply_alternate_estimates is None
        else pd.read_csv(args.apply_alternate_estimates),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    paths = write_estimate_sequence_gate_fit_outputs(
        result=result,
        output_dir=args.output_dir,
        base_estimates_path=args.base_estimates,
        alternate_estimates_path=args.alternate_estimates,
        apply_base_estimates_path=args.apply_base_estimates,
        apply_alternate_estimates_path=args.apply_alternate_estimates,
        template_path=args.template,
        truth_path=args.truth_csv,
        weight_grid=grid,
    )
    print("mmuad_track5_estimate_sequence_gate_fit=ok")
    for key, path in paths.items():
        print(f"{key}={path}")
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
    alt, _ = resample_estimates_to_track5_template(
        alternate_estimates,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    base = base.rename(columns={"state_x_m": "base_x_m", "state_y_m": "base_y_m", "state_z_m": "base_z_m"})
    alt = alt.rename(
        columns={
            "state_x_m": "alternate_x_m",
            "state_y_m": "alternate_y_m",
            "state_z_m": "alternate_z_m",
        }
    )
    merged = base[["sequence_id", "time_s", "base_x_m", "base_y_m", "base_z_m"]].merge(
        alt[["sequence_id", "time_s", "alternate_x_m", "alternate_y_m", "alternate_z_m"]],
        on=["sequence_id", "time_s"],
        how="inner",
    )
    if len(merged) != len(base) or len(merged) != len(alt):
        raise ValueError("base and alternate estimates do not align after template resampling")
    return (
        merged[["sequence_id", "time_s", "base_x_m", "base_y_m", "base_z_m"]].copy(),
        merged[["sequence_id", "time_s", "alternate_x_m", "alternate_y_m", "alternate_z_m"]].copy(),
    )


def _sequence_features(base: pd.DataFrame, alt: pd.DataFrame) -> pd.DataFrame:
    rows = base.merge(alt, on=["sequence_id", "time_s"], how="inner")
    records: list[dict[str, Any]] = []
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        b = group[["base_x_m", "base_y_m", "base_z_m"]].to_numpy(float)
        a = group[["alternate_x_m", "alternate_y_m", "alternate_z_m"]].to_numpy(float)
        valid = np.isfinite(b).all(axis=1) & np.isfinite(a).all(axis=1)
        disp = np.linalg.norm(b[valid] - a[valid], axis=1)
        b_steps = _steps(b[valid])
        a_steps = _steps(a[valid])
        base_path = _safe_sum(b_steps)
        alt_path = _safe_sum(a_steps)
        records.append(
            {
                "sequence_id": str(sequence_id),
                "row_count": int(len(group)),
                "valid_pair_fraction": float(valid.mean()) if len(valid) else 0.0,
                "mean_displacement_m": _safe_mean(disp),
                "p95_displacement_m": _safe_percentile(disp, 95),
                "max_displacement_m": _safe_max(disp),
                "mean_base_step_m": _safe_mean(b_steps),
                "mean_alternate_step_m": _safe_mean(a_steps),
                "base_path_length_m": base_path,
                "alternate_path_length_m": alt_path,
                "path_length_ratio": alt_path / base_path if np.isfinite(base_path) and base_path > 0 else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def _oracle_weights(
    base: pd.DataFrame,
    alt: pd.DataFrame,
    truth: pd.DataFrame,
    grid: np.ndarray,
) -> pd.DataFrame:
    rows = _truth_aligned(base, alt, truth)
    records: list[dict[str, Any]] = []
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        best: dict[str, Any] | None = None
        for weight in grid:
            metrics = _metrics_for_weight(group, float(weight))
            row = {"sequence_id": str(sequence_id), "sequence_gate_weight": float(weight), **metrics}
            if best is None or row["pose_mse_m2"] < best["pose_mse_m2"]:
                best = row
        if best is not None:
            records.append(best)
    return pd.DataFrame.from_records(records)


def _loso_weights(features: pd.DataFrame) -> pd.DataFrame:
    if features["sequence_id"].nunique() < 2:
        raise ValueError("at least two sequences are required for LOSO weights")
    records: list[dict[str, Any]] = []
    for sequence_id in features["sequence_id"].astype(str):
        train = features.loc[features["sequence_id"].astype(str) != sequence_id]
        holdout = features.loc[features["sequence_id"].astype(str) == sequence_id]
        predicted = _nearest_neighbor_predict(train, holdout).iloc[0].to_dict()
        predicted["sequence_id"] = sequence_id
        predicted["prediction_protocol"] = "loso-nearest-neighbor"
        records.append(predicted)
    return pd.DataFrame.from_records(records)


def _nearest_neighbor_predict(train: pd.DataFrame, apply: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in FEATURE_COLUMNS if column in train.columns and column in apply.columns]
    train_values = train[columns].apply(pd.to_numeric, errors="coerce")
    apply_values = apply[columns].apply(pd.to_numeric, errors="coerce")
    means = train_values.mean().fillna(0.0)
    stds = train_values.std().replace(0.0, 1.0).fillna(1.0)
    train_z = ((train_values.fillna(means) - means) / stds).to_numpy(float)
    apply_z = ((apply_values.fillna(means) - means) / stds).to_numpy(float)
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


def _score_weights(base: pd.DataFrame, alt: pd.DataFrame, truth: pd.DataFrame, weights: pd.DataFrame) -> dict[str, Any]:
    rows = _truth_aligned(base, alt, truth).merge(
        weights[["sequence_id", "sequence_gate_weight"]],
        on="sequence_id",
        how="inner",
    )
    if rows.empty:
        return _empty_metrics()
    w = rows["sequence_gate_weight"].to_numpy(float)
    blended = _blend(rows, w)
    truth_xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(float)
    finite = np.isfinite(blended).all(axis=1) & np.isfinite(truth_xyz).all(axis=1)
    return _error_metrics(np.linalg.norm(blended[finite] - truth_xyz[finite], axis=1))


def _truth_aligned(base: pd.DataFrame, alt: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows = base.merge(alt, on=["sequence_id", "time_s"], how="inner")
    rows["_time_key"] = _time_key(rows["time_s"])
    ref = _truth_rows(truth).copy()
    ref["_time_key"] = _time_key(ref["time_s"])
    return rows.merge(ref[["sequence_id", "_time_key", "x_m", "y_m", "z_m"]], on=["sequence_id", "_time_key"], how="inner")


def _metrics_for_weight(rows: pd.DataFrame, weight: float) -> dict[str, Any]:
    blended = _blend(rows, np.full(len(rows), float(weight)))
    truth_xyz = rows[["x_m", "y_m", "z_m"]].to_numpy(float)
    finite = np.isfinite(blended).all(axis=1) & np.isfinite(truth_xyz).all(axis=1)
    return _error_metrics(np.linalg.norm(blended[finite] - truth_xyz[finite], axis=1))


def _blend(rows: pd.DataFrame, weight: np.ndarray) -> np.ndarray:
    base_xyz = rows[["base_x_m", "base_y_m", "base_z_m"]].to_numpy(float)
    alt_xyz = rows[["alternate_x_m", "alternate_y_m", "alternate_z_m"]].to_numpy(float)
    return (1.0 - weight[:, None]) * base_xyz + weight[:, None] * alt_xyz


def _truth_rows(truth: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(truth).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s", "x_m", "y_m", "z_m"])
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite, ["sequence_id", "time_s", "x_m", "y_m", "z_m"]]


def _weight_grid_text(text: str) -> np.ndarray:
    return _weight_grid(np.asarray([float(item.strip()) for item in str(text).split(",") if item.strip()]))


def _weight_grid(values: np.ndarray | None) -> np.ndarray:
    grid = np.asarray(DEFAULT_WEIGHT_GRID if values is None else values, dtype=float)
    if grid.size == 0 or not np.isfinite(grid).all() or (grid < 0).any() or (grid > 1).any():
        raise ValueError("weight grid must contain finite values in [0, 1]")
    return np.unique(grid)


def _steps(xyz: np.ndarray) -> np.ndarray:
    if len(xyz) < 2:
        return np.asarray([], dtype=float)
    return np.linalg.norm(np.diff(xyz, axis=0), axis=1)


def _error_metrics(errors: np.ndarray) -> dict[str, Any]:
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return _empty_metrics()
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
    return {"matched_rows": 0, "pose_mse_m2": np.nan, "pose_rmse_m": np.nan, "pose_mean_m": np.nan, "pose_p95_m": np.nan, "pose_max_m": np.nan}


def _time_key(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").round(9).astype(str)


def _safe_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile)) if len(values) else np.nan


def _safe_max(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.max(values)) if len(values) else np.nan


def _safe_sum(values: np.ndarray) -> float:
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
