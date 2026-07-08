"""Geometric-median Track 5 estimate ensemble for MMUAD submissions.

Weighted means are useful when independent pose pipelines are unbiased, but they
are fragile when one candidate stream catastrophically follows a wrong branch.
This module builds an inference-safe Track 5 ensemble by resampling each estimate
trajectory to the official Sequence/Timestamp template and taking a weighted 3D
geometric median at every template row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    load_sequence_class_map,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput, parse_estimate_spec
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

GEOMEDIAN_ESTIMATES_CSV = "mmuad_track5_geomedian_estimates.csv"
GEOMEDIAN_DIAGNOSTICS_CSV = "mmuad_track5_geomedian_diagnostics.csv"
GEOMEDIAN_MANIFEST_JSON = "mmuad_track5_geomedian_manifest.json"
VALIDATION_JSON = "mmuad_track5_geomedian_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_geomedian_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
TEMPLATE_TIME_MATCH_ATOL_S = 1.0e-9


def build_track5_geometric_median_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
    template: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None = None,
    max_iterations: int = 64,
    tolerance_m: float = 1.0e-4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return geometric-median estimates and per-template diagnostics."""

    template_rows = _normalize_template_rows(template)
    if template_rows.empty:
        empty = pd.DataFrame(columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"])
        return empty, pd.DataFrame()

    resampled_parts: list[pd.DataFrame] = []
    input_summaries: list[dict[str, Any]] = []
    for label, estimates, weight in estimate_inputs:
        safe_label = _safe_label(label)
        safe_weight = _validate_weight(weight, label=safe_label)
        resampled, resample_diagnostics = resample_estimates_to_track5_template(
            estimates,
            template_rows,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        part = resampled[["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]].copy()
        part["geomedian_label"] = safe_label
        part["geomedian_weight"] = safe_weight
        part["geomedian_valid"] = _finite_xyz(part) & resampled.get(
            "template_resample_valid",
            pd.Series(True, index=resampled.index),
        ).astype(bool)
        resampled_parts.append(part)
        input_summaries.append(
            {
                "label": safe_label,
                "weight": safe_weight,
                "input_estimate_rows": int(len(estimates)),
                "valid_resampled_rows": int(part["geomedian_valid"].sum()),
                "mean_abs_nearest_time_delta_s": _safe_mean_abs(
                    resample_diagnostics.get("nearest_time_delta_s", pd.Series(dtype=float))
                ),
            }
        )
    if not resampled_parts:
        raise ValueError("at least one estimate input is required")
    stacked = pd.concat(resampled_parts, ignore_index=True, sort=False)

    estimate_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    for _, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        time_s = float(template_row["time_s"])
        rows = stacked.loc[
            (stacked["sequence_id"].astype(str) == sequence_id)
            & _template_time_matches(stacked["time_s"], time_s)
        ]
        valid = rows.loc[rows["geomedian_valid"].astype(bool) & (rows["geomedian_weight"] > 0.0)]
        if valid.empty:
            center = np.asarray([np.nan, np.nan, np.nan], dtype=float)
            mean_center = center.copy()
            iterations = 0
            displacement = np.nan
            labels = ""
            weight_sum = 0.0
            spread = np.nan
        else:
            xyz = valid[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
            weights = valid["geomedian_weight"].to_numpy(float)
            center, iterations, displacement = weighted_geometric_median(
                xyz,
                weights,
                max_iterations=max_iterations,
                tolerance_m=tolerance_m,
            )
            mean_center = np.sum(weights[:, None] * xyz, axis=0) / float(np.sum(weights))
            labels = ";".join(valid["geomedian_label"].astype(str).tolist())
            weight_sum = float(np.sum(weights))
            spread = _weighted_spread_m(xyz, weights, center)
        estimate_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "state_x_m": float(center[0]) if np.isfinite(center[0]) else np.nan,
                "state_y_m": float(center[1]) if np.isfinite(center[1]) else np.nan,
                "state_z_m": float(center[2]) if np.isfinite(center[2]) else np.nan,
                "track5_geometric_median_ensemble": True,
                "geomedian_source_count": int(len(valid)),
                "geomedian_weight_sum": weight_sum,
                "geomedian_labels": labels,
                "geomedian_iterations": int(iterations),
                "geomedian_final_displacement_m": displacement,
                "geomedian_position_spread_m": spread,
            }
        )
        diagnostic_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "candidate_input_count": int(len(rows)),
                "valid_input_count": int(len(valid)),
                "weight_sum": weight_sum,
                "labels": labels,
                "position_spread_m": spread,
                "geomedian_iterations": int(iterations),
                "geomedian_final_displacement_m": displacement,
                "geomedian_to_weighted_mean_m": float(np.linalg.norm(center - mean_center))
                if np.isfinite(center).all() and np.isfinite(mean_center).all()
                else np.nan,
            }
        )
    estimates = pd.DataFrame.from_records(estimate_records)
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    diagnostics.attrs["input_summaries"] = input_summaries
    return estimates, diagnostics


def weighted_geometric_median(
    xyz: np.ndarray,
    weights: np.ndarray,
    *,
    max_iterations: int = 64,
    tolerance_m: float = 1.0e-4,
) -> tuple[np.ndarray, int, float]:
    """Compute a weighted 3D geometric median with Weiszfeld iterations."""

    points = np.asarray(xyz, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(weights) & (weights > 0.0)
    points = points[finite]
    weights = weights[finite]
    if len(points) == 0:
        return np.asarray([np.nan, np.nan, np.nan], dtype=float), 0, np.nan
    if len(points) == 1:
        return points[0].astype(float), 0, 0.0
    center = np.sum(weights[:, None] * points, axis=0) / float(np.sum(weights))
    last_displacement = np.inf
    eps = 1.0e-9
    for iteration in range(1, int(max_iterations) + 1):
        distances = np.linalg.norm(points - center[None, :], axis=1)
        inv_dist = weights / np.maximum(distances, eps)
        updated = np.sum(inv_dist[:, None] * points, axis=0) / float(np.sum(inv_dist))
        last_displacement = float(np.linalg.norm(updated - center))
        center = updated
        if last_displacement <= float(tolerance_m):
            return center.astype(float), iteration, last_displacement
    return center.astype(float), int(max_iterations), last_displacement


def write_track5_geometric_median_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
    max_iterations: int = 64,
    tolerance_m: float = 1.0e-4,
) -> dict[str, Path]:
    """Write geometric-median estimates, official CSV/ZIP, and validation."""

    inputs = list(estimate_inputs)
    loaded_inputs = [(item.label, read_estimate_csv(item.path), float(item.weight)) for item in inputs]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates, diagnostics = build_track5_geometric_median_ensemble(
        loaded_inputs,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        max_iterations=max_iterations,
        tolerance_m=tolerance_m,
    )
    paths = {
        "estimates_csv": output / GEOMEDIAN_ESTIMATES_CSV,
        "diagnostics_csv": output / GEOMEDIAN_DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / VALIDATION_JSON,
        "validation_rows_csv": output / VALIDATION_ROWS_CSV,
        "manifest_json": output / GEOMEDIAN_MANIFEST_JSON,
    }
    estimates.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    class_map = class_map or {}
    write_official_mmaud_results_csv(
        estimates,
        paths["official_results_csv"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        estimates,
        paths["official_zip"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    validation = validate_official_track5_submission(paths["official_zip"], template=template, require_zip=True)
    paths["validation_json"].write_text(json.dumps(_jsonable(validation.summary), indent=2), encoding="utf-8")
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    manifest = {
        "schema": "raft-uav-mmuad-track5-geometric-median-ensemble-v1",
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in inputs
        ],
        "input_summaries": diagnostics.attrs.get("input_summaries", []),
        "row_count": int(len(estimates)),
        "valid_estimate_rows": int(_finite_xyz(estimates).sum()),
        "mean_position_spread_m": _safe_mean(diagnostics.get("position_spread_m", pd.Series(dtype=float))),
        "p95_position_spread_m": _safe_percentile(diagnostics.get("position_spread_m", pd.Series(dtype=float)), 95),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "max_iterations": int(max_iterations),
        "tolerance_m": float(tolerance_m),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-geomedian-ensemble",
        description="build a geometric-median ensemble for MMUAD Track 5 estimate CSVs",
    )
    parser.add_argument("--estimate-csv", action="append", default=[], metavar="LABEL=PATH[@WEIGHT]")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--max-iterations", type=int, default=64)
    parser.add_argument("--tolerance-m", type=float, default=1.0e-4)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    estimate_inputs = [parse_estimate_spec(spec) for spec in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_geometric_median_outputs(
        estimate_inputs=estimate_inputs,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
        max_iterations=int(args.max_iterations),
        tolerance_m=float(args.tolerance_m),
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_geomedian_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"geometric-median ensemble upload is not leaderboard-ready: {reasons}")
    return 0


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    sequence_column = _first_present(rows, ("sequence_id", "Sequence", "sequence", "seq"))
    time_column = _first_present(rows, ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"))
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].astype(str),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
        }
    )
    finite = np.isfinite(out["time_s"].to_numpy(float))
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _template_time_matches(values: pd.Series, target: float) -> np.ndarray:
    return np.isclose(pd.to_numeric(values, errors="coerce").to_numpy(float), float(target), rtol=0.0, atol=TEMPLATE_TIME_MATCH_ATOL_S)


def _finite_xyz(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool)
    xyz = rows[["state_x_m", "state_y_m", "state_z_m"]].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(xyz.to_numpy(float)).all(axis=1), index=rows.index)


def _validate_weight(value: float, *, label: str) -> float:
    weight = float(value)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(f"estimate weight must be finite and non-negative for {label}: {weight}")
    return weight


def _weighted_spread_m(xyz: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
    if len(xyz) == 0 or not np.isfinite(center).all():
        return np.nan
    distances = np.linalg.norm(xyz - center[None, :], axis=1)
    return float(np.sum(weights * distances) / np.sum(weights))


def _safe_mean_abs(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(np.abs(numeric.to_numpy(float))))


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(numeric.to_numpy(float)))


def _safe_percentile(values: pd.Series, percentile: float) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.percentile(numeric.to_numpy(float), percentile))


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(column).lower(): str(column) for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = lower.get(name.lower())
        if found is not None:
            return found
    return None


def _safe_label(value: Any) -> str:
    label = str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return label or "estimate"


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
