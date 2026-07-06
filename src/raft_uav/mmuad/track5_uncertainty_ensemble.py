"""Uncertainty-weighted Track 5 estimate ensembling.

This is an inference-safe companion to the fixed-weight estimate ensemble.  It
resamples several estimate trajectories onto an official Track 5 template, then
computes a row-wise inverse-variance weighted pose average using each estimate's
own predicted uncertainty column.  Global weights may still be supplied, but
ambiguous/high-sigma rows are downweighted automatically.
"""

from __future__ import annotations

import argparse
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
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput, parse_estimate_spec
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

UNCERTAINTY_ENSEMBLE_ESTIMATES_CSV = "mmuad_track5_uncertainty_ensemble_estimates.csv"
UNCERTAINTY_ENSEMBLE_DIAGNOSTICS_CSV = "mmuad_track5_uncertainty_ensemble_diagnostics.csv"
UNCERTAINTY_ENSEMBLE_MANIFEST_JSON = "mmuad_track5_uncertainty_ensemble_manifest.json"
UNCERTAINTY_ENSEMBLE_VALIDATION_JSON = "mmuad_track5_uncertainty_ensemble_validation.json"
UNCERTAINTY_ENSEMBLE_VALIDATION_ROWS_CSV = "mmuad_track5_uncertainty_ensemble_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"

SEQUENCE_ALIASES = ("sequence_id", "Sequence", "sequence", "seq")
TIME_ALIASES = ("time_s", "Timestamp", "timestamp", "timestamp_s", "time")
TEMPLATE_TIME_MATCH_ATOL_S = 1.0e-9


def build_track5_uncertainty_ensemble(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    uncertainty_column: str = "predicted_sigma_m",
    fallback_sigma_m: float = 30.0,
    sigma_min_m: float = 1.0,
    sigma_max_m: float = 100.0,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return inverse-variance ensembled estimates and diagnostics."""

    template_rows = _normalize_template_rows(template)
    inputs = list(estimate_inputs)
    if not inputs:
        raise ValueError("at least one estimate input is required")
    if template_rows.empty:
        return _empty_estimates(), _empty_diagnostics()
    fallback_sigma_m, sigma_min_m, sigma_max_m = _validate_sigma_parameters(
        fallback_sigma_m=fallback_sigma_m,
        sigma_min_m=sigma_min_m,
        sigma_max_m=sigma_max_m,
    )
    stacked_parts: list[pd.DataFrame] = []
    input_summary: list[dict[str, Any]] = []
    for item in inputs:
        rows = pd.read_csv(item.path)
        resampled, resample_diag = resample_estimates_to_track5_template(
            rows,
            template_rows,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        sigma = _resample_uncertainty_column(
            rows,
            template_rows,
            column=uncertainty_column,
            fallback_sigma_m=fallback_sigma_m,
            sigma_min_m=sigma_min_m,
            sigma_max_m=sigma_max_m,
        )
        part = resampled[["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]].copy()
        part["ensemble_label"] = item.label
        part["global_weight"] = float(item.weight)
        part["predicted_sigma_m"] = sigma.to_numpy(float)
        sigma_array = part["predicted_sigma_m"].to_numpy(float)
        part["inverse_variance_weight"] = np.where(
            np.isfinite(sigma_array) & (sigma_array > 0.0),
            float(item.weight) / np.square(sigma_array),
            0.0,
        )
        part["ensemble_valid"] = _finite_xyz(part) & resampled.get(
            "template_resample_valid",
            pd.Series(True, index=resampled.index),
        ).astype(bool)
        stacked_parts.append(part)
        input_summary.append(
            {
                "label": item.label,
                "path": str(item.path),
                "global_weight": float(item.weight),
                "input_rows": int(len(rows)),
                "valid_resampled_rows": int(part["ensemble_valid"].sum()),
                "mean_sigma_m": _safe_mean(part["predicted_sigma_m"]),
                "mean_abs_nearest_time_delta_s": _safe_mean_abs(
                    resample_diag.get("nearest_time_delta_s", pd.Series(dtype=float))
                ),
            }
        )
    stacked = pd.concat(stacked_parts, ignore_index=True, sort=False)
    estimates: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for _, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        time_s = float(template_row["time_s"])
        rows = stacked.loc[
            (stacked["sequence_id"].astype(str) == sequence_id)
            & _template_time_matches(stacked["time_s"], time_s)
        ]
        valid = rows.loc[
            rows["ensemble_valid"].astype(bool) & (rows["inverse_variance_weight"] > 0.0)
        ]
        if valid.empty:
            xyz = np.asarray([np.nan, np.nan, np.nan], dtype=float)
            weight_sum = 0.0
            labels = ""
            effective_sigma = np.nan
            spread = np.nan
        else:
            weights = valid["inverse_variance_weight"].to_numpy(float)
            xyz_rows = valid[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
            weight_sum = float(np.sum(weights))
            xyz = np.sum(weights[:, None] * xyz_rows, axis=0) / weight_sum
            labels = ";".join(valid["ensemble_label"].astype(str).tolist())
            effective_sigma = float(np.sqrt(1.0 / weight_sum)) if weight_sum > 0.0 else np.nan
            spread = _weighted_spread_m(xyz_rows, weights, xyz)
        estimates.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "state_x_m": float(xyz[0]) if np.isfinite(xyz[0]) else np.nan,
                "state_y_m": float(xyz[1]) if np.isfinite(xyz[1]) else np.nan,
                "state_z_m": float(xyz[2]) if np.isfinite(xyz[2]) else np.nan,
                "source": "track5-uncertainty-ensemble",
                "track_id": "track5-uncertainty-ensemble",
                "uncertainty_ensemble": True,
                "ensemble_source_count": int(len(valid)),
                "ensemble_weight_sum": weight_sum,
                "ensemble_effective_sigma_m": effective_sigma,
                "ensemble_position_spread_m": spread,
                "ensemble_labels": labels,
            }
        )
        diagnostics.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "candidate_input_count": int(len(rows)),
                "valid_input_count": int(len(valid)),
                "inverse_variance_weight_sum": weight_sum,
                "effective_sigma_m": effective_sigma,
                "position_spread_m": spread,
                "labels": labels,
            }
        )
    estimate_frame = pd.DataFrame.from_records(estimates)
    diagnostic_frame = pd.DataFrame.from_records(diagnostics)
    diagnostic_frame.attrs["input_summary"] = input_summary
    return estimate_frame, diagnostic_frame


def write_track5_uncertainty_ensemble_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    uncertainty_column: str = "predicted_sigma_m",
    fallback_sigma_m: float = 30.0,
    sigma_min_m: float = 1.0,
    sigma_max_m: float = 100.0,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Write uncertainty-ensemble estimates, official CSV/ZIP, and manifest."""

    input_list = list(estimate_inputs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates, diagnostics = build_track5_uncertainty_ensemble(
        input_list,
        template=template,
        uncertainty_column=uncertainty_column,
        fallback_sigma_m=fallback_sigma_m,
        sigma_min_m=sigma_min_m,
        sigma_max_m=sigma_max_m,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "ensemble_estimates_csv": output / UNCERTAINTY_ENSEMBLE_ESTIMATES_CSV,
        "diagnostics_csv": output / UNCERTAINTY_ENSEMBLE_DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / UNCERTAINTY_ENSEMBLE_VALIDATION_JSON,
        "validation_rows_csv": output / UNCERTAINTY_ENSEMBLE_VALIDATION_ROWS_CSV,
        "manifest_json": output / UNCERTAINTY_ENSEMBLE_MANIFEST_JSON,
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
        "schema": "raft-uav-mmuad-track5-uncertainty-ensemble-v1",
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in input_list
        ],
        "uncertainty_column": uncertainty_column,
        "fallback_sigma_m": float(fallback_sigma_m),
        "sigma_min_m": float(sigma_min_m),
        "sigma_max_m": float(sigma_max_m),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "input_summary": diagnostics.attrs.get("input_summary", []),
        "row_count": int(len(estimates)),
        "valid_ensemble_rows": int(_finite_xyz(estimates).sum()),
        "mean_effective_sigma_m": _safe_mean(
            diagnostics.get("effective_sigma_m", pd.Series(dtype=float))
        ),
        "mean_position_spread_m": _safe_mean(
            diagnostics.get("position_spread_m", pd.Series(dtype=float))
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
        description="inverse-variance ensemble Track 5 estimates on an official template",
    )
    parser.add_argument(
        "--estimate-csv",
        action="append",
        default=[],
        metavar="LABEL=PATH[@WEIGHT]",
        help="estimate trajectory to include; may be repeated",
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--uncertainty-column", default="predicted_sigma_m")
    parser.add_argument("--fallback-sigma-m", type=float, default=30.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=100.0)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    inputs = [parse_estimate_spec(spec) for spec in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_uncertainty_ensemble_outputs(
        estimate_inputs=inputs,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        uncertainty_column=args.uncertainty_column,
        fallback_sigma_m=float(args.fallback_sigma_m),
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
        raise SystemExit(f"uncertainty ensemble is not leaderboard-ready: {reasons}")
    return 0


def _validate_sigma_parameters(
    *,
    fallback_sigma_m: float,
    sigma_min_m: float,
    sigma_max_m: float,
) -> tuple[float, float, float]:
    fallback = _positive_finite(fallback_sigma_m, name="fallback_sigma_m")
    sigma_min = _positive_finite(sigma_min_m, name="sigma_min_m")
    sigma_max = _positive_finite(sigma_max_m, name="sigma_max_m")
    if sigma_max < sigma_min:
        raise ValueError("sigma_max_m must be greater than or equal to sigma_min_m")
    return fallback, sigma_min, sigma_max


def _positive_finite(value: float, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _resample_uncertainty_column(
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    *,
    column: str,
    fallback_sigma_m: float,
    sigma_min_m: float,
    sigma_max_m: float,
) -> pd.Series:
    rows = _normalize_uncertainty_rows(estimates, column=column)
    template_rows = _normalize_template_rows(template)
    output = pd.Series(fallback_sigma_m, index=template_rows.index, dtype=float)
    if rows.empty:
        return output.clip(lower=sigma_min_m, upper=sigma_max_m)
    for sequence_id, template_group in template_rows.groupby("sequence_id", sort=False):
        source = rows.loc[rows["sequence_id"] == sequence_id].sort_values("time_s")
        if source.empty:
            continue
        times = source["time_s"].to_numpy(float)
        values = source["sigma_m"].to_numpy(float)
        if len(times) == 1:
            interpolated = np.full(len(template_group), values[0], dtype=float)
        else:
            interpolated = np.interp(template_group["time_s"].to_numpy(float), times, values)
        output.loc[template_group.index] = interpolated
    return output.clip(lower=sigma_min_m, upper=sigma_max_m)


def _normalize_uncertainty_rows(estimates: pd.DataFrame, *, column: str) -> pd.DataFrame:
    rows = pd.DataFrame(estimates).copy()
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    time_column = _first_present(rows, TIME_ALIASES)
    if sequence_column is None or time_column is None or column not in rows.columns:
        return pd.DataFrame(columns=["sequence_id", "time_s", "sigma_m"])
    out = pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].astype(str),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
            "sigma_m": pd.to_numeric(rows[column], errors="coerce"),
        }
    )
    finite = np.isfinite(out[["time_s", "sigma_m"]].to_numpy(float)).all(axis=1)
    out = out.loc[finite & (out["sigma_m"] > 0.0)].copy()
    return out.drop_duplicates(["sequence_id", "time_s"], keep="last").reset_index(drop=True)


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


def _template_time_matches(values: pd.Series, target: float) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return np.isclose(numeric, float(target), rtol=0.0, atol=TEMPLATE_TIME_MATCH_ATOL_S)


def _finite_xyz(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool)
    xyz = rows[["state_x_m", "state_y_m", "state_z_m"]].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(xyz.to_numpy(float)).all(axis=1), index=rows.index)


def _weighted_spread_m(xyz: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
    if len(xyz) == 0:
        return np.nan
    distances = np.linalg.norm(xyz - center[None, :], axis=1)
    return float(np.sum(weights * distances) / np.sum(weights))


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(numeric.to_numpy(float)))


def _safe_mean_abs(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(np.abs(numeric.to_numpy(float))))


def _empty_estimates() -> pd.DataFrame:
    return pd.DataFrame(columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"])


def _empty_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(columns=["sequence_id", "time_s", "valid_input_count"])


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(column).lower(): str(column) for column in rows.columns}
    for name in names:
        if name in rows.columns:
            return name
        found = lower.get(name.lower())
        if found is not None:
            return found
    return None


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
