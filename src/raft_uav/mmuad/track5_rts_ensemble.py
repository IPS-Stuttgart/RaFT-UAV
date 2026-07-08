"""RTS-smoothed estimate ensembling for MMUAD/UG2+ Track 5.

Row-wise weighted ensembling is useful for Codabench submissions, but it does not
exploit temporal dynamics and can preserve frame-level jitter.  This module first
resamples multiple estimate trajectories onto the official Track 5 template,
then treats their weighted mean as a noisy position measurement and runs a
constant-velocity Rauch--Tung--Striebel smoother per sequence.

The inference path uses only estimate CSVs, explicit weights, and the official
Sequence/Timestamp template.  Truth is not used.
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
    parse_official_sequence_cell,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput, parse_estimate_spec
from raft_uav.mmuad.track5_template_resample import resample_estimates_to_track5_template

RTS_ESTIMATES_CSV = "mmuad_track5_rts_ensemble_estimates.csv"
RTS_DIAGNOSTICS_CSV = "mmuad_track5_rts_ensemble_diagnostics.csv"
RTS_MANIFEST_JSON = "mmuad_track5_rts_ensemble_manifest.json"
RTS_VALIDATION_JSON = "mmuad_track5_rts_ensemble_validation.json"
RTS_VALIDATION_ROWS_CSV = "mmuad_track5_rts_ensemble_validation_rows.csv"
OFFICIAL_RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
# Resampling copies requested template timestamps into each candidate row. Match those
# rows with an absolute tolerance only; NumPy's default relative tolerance is unsafe
# for epoch-style timestamps because seconds-scale differences can compare close.
TEMPLATE_TIME_ATOL_S = 1.0e-9


def build_track5_rts_ensemble(
    estimate_inputs: Iterable[tuple[str, pd.DataFrame, float]],
    template: pd.DataFrame,
    *,
    measurement_sigma_m: float = 10.0,
    process_accel_std_mps2: float = 5.0,
    initial_position_std_m: float = 100.0,
    initial_velocity_std_mps: float = 25.0,
    spread_variance_scale: float = 1.0,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return RTS-smoothed Track 5 estimates and row diagnostics."""

    measurement_sigma_m = _positive_finite(measurement_sigma_m, "measurement_sigma_m")
    process_accel_std_mps2 = _nonnegative_finite(
        process_accel_std_mps2,
        "process_accel_std_mps2",
    )
    initial_position_std_m = _positive_finite(initial_position_std_m, "initial_position_std_m")
    initial_velocity_std_mps = _positive_finite(
        initial_velocity_std_mps,
        "initial_velocity_std_mps",
    )
    spread_variance_scale = _nonnegative_finite(spread_variance_scale, "spread_variance_scale")
    template_rows = _normalize_template_rows(template)
    loaded = tuple(estimate_inputs)
    if not loaded:
        raise ValueError("at least one estimate input is required")
    if template_rows.empty:
        return _empty_estimates(), _empty_diagnostics()

    stacked_parts: list[pd.DataFrame] = []
    for order, (label_text, estimates, weight) in enumerate(loaded):
        label = _safe_label(label_text)
        weight = _positive_finite(weight, f"weight[{label}]")
        resampled, _ = resample_estimates_to_track5_template(
            estimates,
            template_rows,
            max_nearest_time_delta_s=max_nearest_time_delta_s,
        )
        part = resampled[["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]].copy()
        part["input_label"] = label
        part["input_order"] = int(order)
        part["input_weight"] = float(weight)
        part["input_valid"] = _finite_xyz(part) & resampled.get(
            "template_resample_valid",
            pd.Series(True, index=resampled.index),
        ).astype(bool)
        stacked_parts.append(part)
    stacked = pd.concat(stacked_parts, ignore_index=True, sort=False)

    estimate_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    for sequence_id, template_group in template_rows.groupby("sequence_id", sort=True):
        sequence_template = template_group.sort_values("time_s").reset_index(drop=True)
        times = sequence_template["time_s"].to_numpy(float)
        measurements = np.full((len(sequence_template), 3), np.nan, dtype=float)
        variances = np.full(len(sequence_template), np.inf, dtype=float)
        diagnostics: list[dict[str, Any]] = []
        for index, time_s in enumerate(times):
            rows = stacked.loc[
                (stacked["sequence_id"].astype(str) == str(sequence_id))
                & _time_matches(stacked["time_s"], float(time_s))
            ]
            valid = rows.loc[rows["input_valid"].astype(bool) & (rows["input_weight"] > 0.0)]
            if valid.empty:
                labels = ""
                weighted_xyz = np.asarray([np.nan, np.nan, np.nan], dtype=float)
                weight_sum = 0.0
                spread_m = np.nan
                measurement_variance = np.inf
            else:
                weights = valid["input_weight"].to_numpy(float) / (measurement_sigma_m**2)
                xyz = valid[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
                weight_sum = float(np.sum(weights))
                weighted_xyz = np.sum(weights[:, None] * xyz, axis=0) / weight_sum
                spread_variance = _weighted_spread_variance(xyz, weights, weighted_xyz)
                measurement_variance = (1.0 / weight_sum) + spread_variance_scale * spread_variance
                measurements[index] = weighted_xyz
                variances[index] = max(float(measurement_variance), 1.0e-9)
                spread_m = float(np.sqrt(max(spread_variance, 0.0)))
                labels = ";".join(valid["input_label"].astype(str).tolist())
            diagnostics.append(
                {
                    "sequence_id": str(sequence_id),
                    "time_s": float(time_s),
                    "valid_input_count": int(len(valid)),
                    "input_labels": labels,
                    "weighted_x_m": float(weighted_xyz[0]) if np.isfinite(weighted_xyz[0]) else np.nan,
                    "weighted_y_m": float(weighted_xyz[1]) if np.isfinite(weighted_xyz[1]) else np.nan,
                    "weighted_z_m": float(weighted_xyz[2]) if np.isfinite(weighted_xyz[2]) else np.nan,
                    "inverse_variance_weight_sum": weight_sum,
                    "measurement_variance_m2": measurement_variance,
                    "input_spread_m": spread_m,
                }
            )
        smoothed = _rts_smooth_positions(
            times,
            measurements,
            variances,
            process_accel_std_mps2=process_accel_std_mps2,
            initial_position_std_m=initial_position_std_m,
            initial_velocity_std_mps=initial_velocity_std_mps,
        )
        estimate = sequence_template.copy()
        estimate["source"] = "track5-rts-ensemble"
        estimate["track_id"] = "track5-rts-ensemble"
        estimate["state_x_m"] = smoothed[:, 0]
        estimate["state_y_m"] = smoothed[:, 1]
        estimate["state_z_m"] = smoothed[:, 2]
        estimate["track5_rts_ensemble"] = True
        estimate_parts.append(estimate)
        diag = pd.DataFrame.from_records(diagnostics)
        diag["smoothed_x_m"] = smoothed[:, 0]
        diag["smoothed_y_m"] = smoothed[:, 1]
        diag["smoothed_z_m"] = smoothed[:, 2]
        diag["smoothed_minus_weighted_m"] = np.linalg.norm(
            smoothed - measurements,
            axis=1,
        )
        diag.loc[~np.isfinite(measurements).all(axis=1), "smoothed_minus_weighted_m"] = np.nan
        diagnostic_parts.append(diag)
    estimates = pd.concat(estimate_parts, ignore_index=True, sort=False)
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True, sort=False)
    return estimates.sort_values(["sequence_id", "time_s"]).reset_index(drop=True), diagnostics


def write_track5_rts_ensemble_outputs(
    *,
    estimate_inputs: Iterable[EstimateInput],
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    measurement_sigma_m: float = 10.0,
    process_accel_std_mps2: float = 5.0,
    initial_position_std_m: float = 100.0,
    initial_velocity_std_mps: float = 25.0,
    spread_variance_scale: float = 1.0,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Write RTS ensemble estimates, official artifacts, validation, and manifest."""

    input_list = list(estimate_inputs)
    loaded = [(item.label, read_estimate_csv(item.path), float(item.weight)) for item in input_list]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    estimates, diagnostics = build_track5_rts_ensemble(
        loaded,
        template,
        measurement_sigma_m=measurement_sigma_m,
        process_accel_std_mps2=process_accel_std_mps2,
        initial_position_std_m=initial_position_std_m,
        initial_velocity_std_mps=initial_velocity_std_mps,
        spread_variance_scale=spread_variance_scale,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "estimates_csv": output / RTS_ESTIMATES_CSV,
        "diagnostics_csv": output / RTS_DIAGNOSTICS_CSV,
        "official_results_csv": output / OFFICIAL_RESULTS_CSV,
        "official_zip": output / OFFICIAL_ZIP,
        "validation_json": output / RTS_VALIDATION_JSON,
        "validation_rows_csv": output / RTS_VALIDATION_ROWS_CSV,
        "manifest_json": output / RTS_MANIFEST_JSON,
    }
    estimates.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    write_official_mmaud_results_csv(
        estimates,
        paths["official_results_csv"],
        classification=default_classification,
        class_map=class_map or {},
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        estimates,
        paths["official_zip"],
        classification=default_classification,
        class_map=class_map or {},
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
        "schema": "raft-uav-mmuad-track5-rts-ensemble-v1",
        "estimate_inputs": [
            {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
            for item in input_list
        ],
        "row_count": int(len(estimates)),
        "valid_rows": int(_finite_xyz(estimates).sum()),
        "measurement_sigma_m": float(measurement_sigma_m),
        "process_accel_std_mps2": float(process_accel_std_mps2),
        "initial_position_std_m": float(initial_position_std_m),
        "initial_velocity_std_mps": float(initial_velocity_std_mps),
        "spread_variance_scale": float(spread_variance_scale),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "mean_input_spread_m": _safe_mean(diagnostics.get("input_spread_m", pd.Series(dtype=float))),
        "mean_smoothed_minus_weighted_m": _safe_mean(
            diagnostics.get("smoothed_minus_weighted_m", pd.Series(dtype=float))
        ),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-rts-ensemble",
        description="RTS-smooth multiple MMUAD Track 5 estimate trajectories on an official template",
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
    parser.add_argument("--measurement-sigma-m", type=float, default=10.0)
    parser.add_argument("--process-accel-std-mps2", type=float, default=5.0)
    parser.add_argument("--initial-position-std-m", type=float, default=100.0)
    parser.add_argument("--initial-velocity-std-mps", type=float, default=25.0)
    parser.add_argument("--spread-variance-scale", type=float, default=1.0)
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if not args.estimate_csv:
        parser.error("provide at least one --estimate-csv LABEL=PATH[@WEIGHT]")
    estimate_inputs = [parse_estimate_spec(spec) for spec in args.estimate_csv]
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_rts_ensemble_outputs(
        estimate_inputs=estimate_inputs,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        measurement_sigma_m=args.measurement_sigma_m,
        process_accel_std_mps2=args.process_accel_std_mps2,
        initial_position_std_m=args.initial_position_std_m,
        initial_velocity_std_mps=args.initial_velocity_std_mps,
        spread_variance_scale=args.spread_variance_scale,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_rts_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"RTS ensemble upload is not leaderboard-ready: {reasons}")
    return 0


def _rts_smooth_positions(
    times: np.ndarray,
    measurements: np.ndarray,
    variances: np.ndarray,
    *,
    process_accel_std_mps2: float,
    initial_position_std_m: float,
    initial_velocity_std_mps: float,
) -> np.ndarray:
    out = np.full_like(measurements, np.nan, dtype=float)
    for axis in range(3):
        out[:, axis] = _rts_smooth_axis(
            times,
            measurements[:, axis],
            variances,
            process_accel_std_mps2=process_accel_std_mps2,
            initial_position_std_m=initial_position_std_m,
            initial_velocity_std_mps=initial_velocity_std_mps,
        )
    return out


def _rts_smooth_axis(
    times: np.ndarray,
    y: np.ndarray,
    variance: np.ndarray,
    *,
    process_accel_std_mps2: float,
    initial_position_std_m: float,
    initial_velocity_std_mps: float,
) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    y = np.asarray(y, dtype=float)
    variance = np.asarray(variance, dtype=float)
    finite = np.isfinite(times) & np.isfinite(y) & np.isfinite(variance) & (variance > 0.0)
    if not finite.any():
        return np.full(len(times), np.nan, dtype=float)
    n = len(times)
    x_filt = np.zeros((n, 2), dtype=float)
    p_filt = np.zeros((n, 2, 2), dtype=float)
    x_pred = np.zeros((n, 2), dtype=float)
    p_pred = np.zeros((n, 2, 2), dtype=float)
    transitions = np.zeros((n, 2, 2), dtype=float)

    first = int(np.flatnonzero(finite)[0])
    x = np.asarray([float(y[first]), 0.0], dtype=float)
    p = np.diag([float(initial_position_std_m) ** 2, float(initial_velocity_std_mps) ** 2])
    h = np.asarray([[1.0, 0.0]], dtype=float)
    identity = np.eye(2)
    for idx in range(n):
        if idx > 0:
            dt = max(float(times[idx] - times[idx - 1]), 1.0e-6)
            f = np.asarray([[1.0, dt], [0.0, 1.0]], dtype=float)
            q = _cv_process_noise(dt, float(process_accel_std_mps2))
            x = f @ x
            p = f @ p @ f.T + q
            transitions[idx] = f
        else:
            transitions[idx] = identity
        x_pred[idx] = x
        p_pred[idx] = p
        if finite[idx]:
            r = float(variance[idx])
            innovation = float(y[idx] - (h @ x)[0])
            s = float((h @ p @ h.T)[0, 0] + r)
            if s > 0.0:
                k = (p @ h.T / s).reshape(2)
                x = x + k * innovation
                p = (identity - np.outer(k, h.reshape(2))) @ p
        x_filt[idx] = x
        p_filt[idx] = p

    x_smooth = x_filt.copy()
    p_smooth = p_filt.copy()
    for idx in range(n - 2, -1, -1):
        f_next = transitions[idx + 1]
        gain = p_filt[idx] @ f_next.T @ np.linalg.pinv(p_pred[idx + 1])
        x_smooth[idx] = x_filt[idx] + gain @ (x_smooth[idx + 1] - x_pred[idx + 1])
        p_smooth[idx] = p_filt[idx] + gain @ (p_smooth[idx + 1] - p_pred[idx + 1]) @ gain.T
    return x_smooth[:, 0]


def _cv_process_noise(dt: float, accel_std: float) -> np.ndarray:
    q = float(accel_std) ** 2
    return q * np.asarray(
        [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
        dtype=float,
    )


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s"])
    seq_col = _first_present(rows, ("sequence_id", "Sequence", "sequence", "seq"))
    time_col = _first_present(rows, ("time_s", "Timestamp", "timestamp", "timestamp_s", "time"))
    if seq_col is None or time_col is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": rows[seq_col].map(_template_sequence_or_none),
            "time_s": pd.to_numeric(rows[time_col], errors="coerce"),
        }
    )
    finite = out["sequence_id"].notna() & np.isfinite(out["time_s"].to_numpy(float))
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _template_sequence_or_none(value: Any) -> str | None:
    try:
        return parse_official_sequence_cell(value)
    except ValueError:
        return None


def _finite_xyz(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=bool)
    xyz = rows[["state_x_m", "state_y_m", "state_z_m"]].apply(pd.to_numeric, errors="coerce")
    return pd.Series(np.isfinite(xyz.to_numpy(float)).all(axis=1), index=rows.index)


def _weighted_spread_variance(xyz: np.ndarray, weights: np.ndarray, center: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    center = np.asarray(center, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0 or xyz.size == 0:
        return float("nan")
    return float(np.sum(weights * np.sum((xyz - center) ** 2, axis=1) / 3.0) / total)


def _time_matches(values: pd.Series, time_s: float) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return np.isclose(numeric, float(time_s), rtol=0.0, atol=TEMPLATE_TIME_ATOL_S)


def _first_present(rows: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(column).strip().lower(): str(column) for column in rows.columns}
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


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return value


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(numeric.mean())


def _empty_estimates() -> pd.DataFrame:
    return pd.DataFrame(columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"])


def _empty_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(columns=["sequence_id", "time_s", "valid_input_count"])


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
