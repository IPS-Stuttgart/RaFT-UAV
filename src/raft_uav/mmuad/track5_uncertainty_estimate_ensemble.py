"""Uncertainty-weighted Track 5 estimate ensembling for MMUAD submissions.

This helper combines multiple estimate trajectories after resampling them to an
official Track 5 timestamp template.  Unlike the plain estimate ensemble, row
weights are adjusted by per-estimate uncertainty columns when available, using
inverse-variance weighting.  The path is inference-safe: uncertainty columns are
model outputs or constants supplied with the estimates, and the template only
provides requested Sequence/Timestamp rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    load_sequence_class_map,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble import parse_estimate_spec
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

ENSEMBLED_ESTIMATES_CSV = "mmuad_track5_uncertainty_ensemble_estimates.csv"
DIAGNOSTICS_CSV = "mmuad_track5_uncertainty_ensemble_diagnostics.csv"
MANIFEST_JSON = "mmuad_track5_uncertainty_ensemble_manifest.json"
VALIDATION_JSON = "mmuad_track5_uncertainty_ensemble_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_uncertainty_ensemble_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
DEFAULT_SIGMA_COLUMNS = (
    "state_sigma_m",
    "position_sigma_m",
    "predicted_sigma_m",
    "sigma_m",
    "rmse_m",
)
SEQUENCE_ALIASES = ("sequence_id", "Sequence", "sequence", "seq")
TIME_ALIASES = ("time_s", "Timestamp", "timestamp", "timestamp_s", "time")


@dataclass(frozen=True)
class UncertaintyEstimateInput:
    """One estimate input with a global weight and optional sigma column."""

    label: str
    path: Path
    weight: float = 1.0
    sigma_column: str | None = None


def parse_uncertainty_estimate_spec(value: str) -> UncertaintyEstimateInput:
    """Parse ``LABEL=PATH[@WEIGHT][:SIGMA_COLUMN]`` specs."""

    estimate_part = value
    sigma_column = None
    if ":" in value:
        estimate_part, sigma_column = value.rsplit(":", 1)
        sigma_column = sigma_column.strip() or None
    estimate = parse_estimate_spec(estimate_part)
    return UncertaintyEstimateInput(
        label=estimate.label,
        path=estimate.path,
        weight=estimate.weight,
        sigma_column=sigma_column,
    )


def build_track5_uncertainty_estimate_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float, str | None]],
    template: pd.DataFrame,
    *,
    default_sigma_m: float = 10.0,
    sigma_min_m: float = 1.0,
    sigma_max_m: float = 100.0,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return inverse-variance weighted estimates and diagnostics."""

    if default_sigma_m <= 0.0 or sigma_min_m <= 0.0 or sigma_max_m < sigma_min_m:
        raise ValueError("sigma bounds must satisfy 0 < sigma_min <= sigma_max and default > 0")
    template_rows = _normalize_template_rows(template)
    resampled_parts: list[pd.DataFrame] = []
    input_summaries: list[dict[str, Any]] = []
    for label, estimates, base_weight, sigma_column in estimate_inputs:
        label = _safe_label(label)
        base_weight = _validate_weight(base_weight, label=label)
        estimates = pd.DataFrame(estimates).copy()
        resampled, diagnostics = resample_estimates_to_track5_template(
            estimates,
            template_rows,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        sigma_source = _choose_sigma_column(estimates, sigma_column)
        sigma = _resample_sigma_to_template(
            estimates,
            template_rows,
            sigma_column=sigma_source,
            default_sigma_m=default_sigma_m,
            sigma_min_m=sigma_min_m,
            sigma_max_m=sigma_max_m,
        )
        part = resampled[["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]].copy()
        part["ensemble_label"] = label
        part["ensemble_base_weight"] = base_weight
        part["ensemble_sigma_m"] = sigma
        part["ensemble_effective_weight"] = base_weight / np.square(sigma)
        part["ensemble_valid"] = _finite_xyz(part) & resampled.get(
            "template_resample_valid",
            pd.Series(True, index=resampled.index),
        ).astype(bool)
        resampled_parts.append(part)
        input_summaries.append(
            {
                "label": label,
                "base_weight": base_weight,
                "sigma_column": sigma_source,
                "input_estimate_rows": int(len(estimates)),
                "template_rows": int(len(template_rows)),
                "valid_resampled_rows": int(part["ensemble_valid"].sum()),
                "mean_sigma_m": _safe_mean(part["ensemble_sigma_m"]),
                "mean_abs_nearest_time_delta_s": _safe_mean_abs(
                    diagnostics.get("nearest_time_delta_s", pd.Series(dtype=float))
                ),
            }
        )
    if not resampled_parts:
        raise ValueError("at least one estimate input is required")
    stacked = pd.concat(resampled_parts, ignore_index=True, sort=False)
    records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    for _, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        time_s = float(template_row["time_s"])
        rows = stacked.loc[
            (stacked["sequence_id"].astype(str) == sequence_id)
            & _template_time_matches(stacked["time_s"], time_s)
        ]
        valid = rows.loc[rows["ensemble_valid"].astype(bool) & (rows["ensemble_effective_weight"] > 0.0)]
        if valid.empty:
            xyz = np.asarray([np.nan, np.nan, np.nan], dtype=float)
            weight_sum = 0.0
            labels = ""
            spread = np.nan
        else:
            weights = valid["ensemble_effective_weight"].to_numpy(float)
            xyz_values = valid[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
            weight_sum = float(np.sum(weights))
            xyz = np.sum(weights[:, None] * xyz_values, axis=0) / weight_sum
            labels = ";".join(valid["ensemble_label"].astype(str).tolist())
            spread = _weighted_spread_m(xyz_values, weights, xyz)
        records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "state_x_m": float(xyz[0]) if np.isfinite(xyz[0]) else np.nan,
                "state_y_m": float(xyz[1]) if np.isfinite(xyz[1]) else np.nan,
                "state_z_m": float(xyz[2]) if np.isfinite(xyz[2]) else np.nan,
                "track5_uncertainty_ensemble": True,
                "ensemble_source_count": int(len(valid)),
                "ensemble_effective_weight_sum": weight_sum,
                "ensemble_labels": labels,
                "ensemble_position_spread_m": spread,
            }
        )
        diagnostic_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "candidate_input_count": int(len(rows)),
                "valid_input_count": int(len(valid)),
                "effective_weight_sum": weight_sum,
                "labels": labels,
                "position_spread_m": spread,
                "mean_sigma_m": _safe_mean(valid["ensemble_sigma_m"]) if not valid.empty else np.nan,
            }
        )
    estimates = pd.DataFrame.from_records(records)
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    diagnostics.attrs["input_summaries"] = input_summaries
    return estimates, diagnostics


def write_track5_uncertainty_estimate_ensemble_outputs(
    *,
    estimate_inputs: Iterable[UncertaintyEstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    default_sigma_m: float = 10.0,
    sigma_min_m: float = 1.0,
    sigma_max_m: float = 100.0,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Write uncertainty ensemble, official artifacts, validation, and manifest."""

    estimate_input_list = list(estimate_inputs)
    loaded = [
        (item.label, pd.read_csv(item.path), float(item.weight), item.sigma_column)
        for item in estimate_input_list
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates, diagnostics = build_track5_uncertainty_estimate_ensemble(
        loaded,
        template,
        default_sigma_m=default_sigma_m,
        sigma_min_m=sigma_min_m,
        sigma_max_m=sigma_max_m,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "ensemble_estimates_csv": output / ENSEMBLED_ESTIMATES_CSV,
        "diagnostics_csv": output / DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / VALIDATION_JSON,
        "validation_rows_csv": output / VALIDATION_ROWS_CSV,
        "manifest_json": output / MANIFEST_JSON,
    }
    estimates.to_csv(paths["ensemble_estimates_csv"], index=False)
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
        "schema": "raft-uav-mmuad-track5-uncertainty-estimate-ensemble-v1",
        "estimate_inputs": [
            {
                "label": item.label,
                "path": str(item.path),
                "weight": float(item.weight),
                "sigma_column": item.sigma_column,
            }
            for item in estimate_input_list
        ],
        "input_summaries": diagnostics.attrs.get("input_summaries", []),
        "row_count": int(len(estimates)),
        "valid_ensemble_rows": int(_finite_xyz(estimates).sum()),
        "default_sigma_m": float(default_sigma_m),
        "sigma_min_m": float(sigma_min_m),
        "sigma_max_m": float(sigma_max_m),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "mean_position_spread_m": _safe_mean(diagnostics.get("position_spread_m", pd.Series(dtype=float))),
        "p95_position_spread_m": _safe_percentile(
            diagnostics.get("position_spread_m", pd.Series(dtype=float)),
            95,
        ),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-uncertainty-ensemble",
        description="inverse-variance ensemble MMUAD Track 5 estimate trajectories",
    )
    parser.add_argument(
        "--estimate-csv",
        action="append",
        default=[],
        metavar="LABEL=PATH[@WEIGHT][:SIGMA_COLUMN]",
        help="estimate trajectory to include; may be repeated",
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=100.0)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT][:SIGMA_COLUMN]")
    inputs = [parse_uncertainty_estimate_spec(value) for value in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_uncertainty_estimate_ensemble_outputs(
        estimate_inputs=inputs,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        default_sigma_m=float(args.default_sigma_m),
        sigma_min_m=float(args.sigma_min_m),
        sigma_max_m=float(args.sigma_max_m),
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    validation = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_uncertainty_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={validation.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={validation.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not validation.get("leaderboard_ready", False):
        reasons = ", ".join(validation.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"uncertainty ensemble upload is not leaderboard-ready: {reasons}")
    return 0


def _choose_sigma_column(rows: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in rows.columns:
            raise ValueError(f"requested sigma column not found: {requested}")
        return requested
    for column in DEFAULT_SIGMA_COLUMNS:
        if column in rows.columns:
            return column
    return None


def _resample_sigma_to_template(
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    *,
    sigma_column: str | None,
    default_sigma_m: float,
    sigma_min_m: float,
    sigma_max_m: float,
) -> np.ndarray:
    template_rows = _normalize_template_rows(template)
    if sigma_column is None:
        return np.full(len(template_rows), float(default_sigma_m), dtype=float)
    rows = _normalize_sigma_rows(estimates, sigma_column=sigma_column)
    sigma_values: list[float] = []
    for _, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        time_s = float(template_row["time_s"])
        group = rows.loc[rows["sequence_id"] == sequence_id].sort_values("time_s")
        if group.empty:
            sigma_values.append(float(default_sigma_m))
            continue
        times = group["time_s"].to_numpy(float)
        sigmas = group["sigma_m"].to_numpy(float)
        if len(times) == 1:
            value = sigmas[0]
        else:
            value = np.interp(time_s, times, sigmas)
        sigma_values.append(float(np.clip(value, sigma_min_m, sigma_max_m)))
    return np.asarray(sigma_values, dtype=float)


def _normalize_sigma_rows(estimates: pd.DataFrame, *, sigma_column: str) -> pd.DataFrame:
    rows = pd.DataFrame(estimates).copy()
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    time_column = _first_present(rows, TIME_ALIASES)
    if sequence_column is None or time_column is None:
        raise ValueError("estimate sigma rows must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].astype(str),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
            "sigma_m": pd.to_numeric(rows[sigma_column], errors="coerce"),
        }
    )
    finite = np.isfinite(out[["time_s", "sigma_m"]].to_numpy(float)).all(axis=1)
    return out.loc[finite & (out["sigma_m"] > 0.0)].reset_index(drop=True)


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    time_column = _first_present(rows, TIME_ALIASES)
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


def _template_time_matches(values: pd.Series, time_s: float) -> np.ndarray:
    return np.isclose(
        pd.to_numeric(values, errors="coerce").to_numpy(float),
        float(time_s),
        rtol=0.0,
        atol=1.0e-9,
    )


def _finite_xyz(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool)
    xyz = rows[["state_x_m", "state_y_m", "state_z_m"]].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(xyz.to_numpy(float)).all(axis=1), index=rows.index)


def _weighted_spread_m(values: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
    if len(values) == 0 or not np.isfinite(center).all():
        return np.nan
    distances = np.linalg.norm(values - center[None, :], axis=1)
    weight_sum = np.sum(weights)
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return np.nan
    return float(np.sum(weights * distances) / weight_sum)


def _validate_weight(weight: float, *, label: str) -> float:
    weight = float(weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(f"estimate weight must be finite and non-negative for {label}: {weight}")
    return weight


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(column).lower(): str(column) for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = lower.get(name.lower())
        if found is not None:
            return found
    return None


def _safe_label(value: str) -> str:
    label = str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return label or "estimate"


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


def _safe_mean_abs(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(np.abs(numeric.to_numpy(float))))


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
