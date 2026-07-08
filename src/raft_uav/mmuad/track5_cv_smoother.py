"""Constant-velocity Kalman/RTS smoothing for MMUAD Track 5 submissions.

The local-linear smoother and speed/acceleration guards operate directly on
neighboring rows.  This module adds a compact physical trajectory prior for the
CVPR/UG2+ Track 5 official grid: per sequence, fit a constant-velocity Kalman
filter and Rauch--Tung--Striebel smoother to the submitted 3D positions, then
blend the smoothed path back into the input with an optional correction cap.

The procedure is inference-safe.  It uses no truth values and preserves the
official Sequence/Timestamp grid and Classification labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    load_official_track5_template_file,
    normalize_official_track5_results_frame,
    parse_official_position_cell,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_submission_ensemble import _jsonable
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission

CV_SMOOTHED_ESTIMATES_CSV = "mmuad_track5_cv_smoothed_estimates.csv"
CV_SMOOTHED_RESULTS_CSV = "mmaud_results_cv_smoothed.csv"
CV_SMOOTHED_ZIP = "ug2_submission_cv_smoothed.zip"
CV_DIAGNOSTICS_CSV = "mmuad_track5_cv_smoother_diagnostics.csv"
CV_MANIFEST_JSON = "mmuad_track5_cv_smoother_manifest.json"
CV_VALIDATION_JSON = "mmuad_track5_cv_smoother_validation.json"
CV_VALIDATION_ROWS_CSV = "mmuad_track5_cv_smoother_validation_rows.csv"


def smooth_track5_cv_submission(
    submission: pd.DataFrame,
    *,
    measurement_std_m: float = 8.0,
    acceleration_std_mps2: float = 6.0,
    initial_position_std_m: float = 25.0,
    initial_velocity_std_mps: float = 25.0,
    blend: float = 1.0,
    max_correction_m: float | None = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return CV-smoothed Track 5 rows and row-level diagnostics."""

    measurement_std_m = _positive_float(measurement_std_m, "measurement_std_m")
    acceleration_std_mps2 = _positive_float(acceleration_std_mps2, "acceleration_std_mps2")
    initial_position_std_m = _positive_float(initial_position_std_m, "initial_position_std_m")
    initial_velocity_std_mps = _positive_float(initial_velocity_std_mps, "initial_velocity_std_mps")
    blend = float(blend)
    if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("blend must be finite and in [0, 1]")
    if max_correction_m is not None:
        max_correction_m = _positive_float(max_correction_m, "max_correction_m")

    rows = _normalized_submission(submission)
    if rows.empty:
        return rows, pd.DataFrame(columns=_diagnostic_columns())

    smoothed_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    for sequence_id, group in rows.groupby("sequence_id", sort=True):
        smoothed, diagnostics = _smooth_sequence(
            group.sort_values("time_s").reset_index(drop=True),
            measurement_std_m=measurement_std_m,
            acceleration_std_mps2=acceleration_std_mps2,
            initial_position_std_m=initial_position_std_m,
            initial_velocity_std_mps=initial_velocity_std_mps,
            blend=blend,
            max_correction_m=max_correction_m,
        )
        smoothed_parts.append(smoothed)
        diagnostic_parts.append(diagnostics.assign(sequence_id=str(sequence_id)))
    return (
        pd.concat(smoothed_parts, ignore_index=True, sort=False)
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True),
        pd.concat(diagnostic_parts, ignore_index=True, sort=False),
    )


def write_track5_cv_smoother_outputs(
    *,
    smoothed: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    input_submission_path: Path,
    template: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    """Write CV-smoothed estimates, official CSV/ZIP, diagnostics, and manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "estimates_csv": output / CV_SMOOTHED_ESTIMATES_CSV,
        "results_csv": output / CV_SMOOTHED_RESULTS_CSV,
        "zip": output / CV_SMOOTHED_ZIP,
        "diagnostics_csv": output / CV_DIAGNOSTICS_CSV,
        "manifest_json": output / CV_MANIFEST_JSON,
    }
    smoothed.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    official_rows = smoothed.copy()
    official_rows["classification"] = official_rows["Classification"]
    write_official_mmaud_results_csv(
        official_rows,
        paths["results_csv"],
        classification=0,
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        official_rows,
        paths["zip"],
        classification=0,
        invalid_row_policy="raise",
    )
    validation_summary: dict[str, Any] | None = None
    if template is not None:
        validation = validate_official_track5_submission(paths["zip"], template=template, require_zip=True)
        paths["validation_json"] = output / CV_VALIDATION_JSON
        paths["validation_rows_csv"] = output / CV_VALIDATION_ROWS_CSV
        paths["validation_json"].write_text(
            json.dumps(_jsonable(validation.summary), indent=2),
            encoding="utf-8",
        )
        validation.rows.to_csv(paths["validation_rows_csv"], index=False)
        validation_summary = _jsonable(validation.summary)
        if require_leaderboard_ready and not validation.summary.get("leaderboard_ready", False):
            reasons = ", ".join(validation.summary.get("leaderboard_blocking_reasons", []))
            raise SystemExit(
                f"CV-smoothed submission is not leaderboard-ready: {reasons or 'unknown'}"
            )
    correction = pd.to_numeric(
        diagnostics.get("cv_smoother_applied_correction_m", pd.Series(dtype=float)),
        errors="coerce",
    )
    payload = dict(manifest or {})
    payload.update(
        {
            "schema": "raft-uav-mmuad-track5-cv-smoother-v1",
            "input_submission": str(input_submission_path),
            "row_count": int(len(smoothed)),
            "sequence_count": int(smoothed["sequence_id"].nunique()) if not smoothed.empty else 0,
            "mean_correction_m": _safe_mean(correction),
            "p95_correction_m": _safe_percentile(correction, 95),
            "max_correction_m": _safe_max(correction),
            "capped_rows": int(diagnostics.get("cv_smoother_capped", pd.Series(dtype=bool)).astype(bool).sum()),
            "validation": validation_summary,
            "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
        }
    )
    paths["manifest_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-cv-smoother",
        description="apply a CV Kalman/RTS smoother to an official MMUAD Track 5 submission",
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, help="optional official template for preflight validation")
    parser.add_argument("--measurement-std-m", type=float, default=8.0)
    parser.add_argument("--acceleration-std-mps2", type=float, default=6.0)
    parser.add_argument("--initial-position-std-m", type=float, default=25.0)
    parser.add_argument("--initial-velocity-std-mps", type=float, default=25.0)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--max-correction-m", type=float, default=10.0)
    parser.add_argument("--disable-correction-cap", action="store_true")
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if args.require_leaderboard_ready and args.template is None:
        raise SystemExit("--require-leaderboard-ready requires --template")
    submission = load_track5_submission(args.submission)
    cap = None if args.disable_correction_cap else float(args.max_correction_m)
    smoothed, diagnostics = smooth_track5_cv_submission(
        submission,
        measurement_std_m=float(args.measurement_std_m),
        acceleration_std_mps2=float(args.acceleration_std_mps2),
        initial_position_std_m=float(args.initial_position_std_m),
        initial_velocity_std_mps=float(args.initial_velocity_std_mps),
        blend=float(args.blend),
        max_correction_m=cap,
    )
    template = None if args.template is None else load_official_track5_template_file(args.template)
    paths = write_track5_cv_smoother_outputs(
        smoothed=smoothed,
        diagnostics=diagnostics,
        output_dir=args.output_dir,
        input_submission_path=args.submission,
        template=template,
        manifest={
            "measurement_std_m": float(args.measurement_std_m),
            "acceleration_std_mps2": float(args.acceleration_std_mps2),
            "initial_position_std_m": float(args.initial_position_std_m),
            "initial_velocity_std_mps": float(args.initial_velocity_std_mps),
            "blend": float(args.blend),
            "max_correction_m": cap,
        },
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_cv_smoother=ok")
    print(f"mean_correction_m={manifest['mean_correction_m']}")
    print(f"max_correction_m={manifest['max_correction_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _smooth_sequence(
    group: pd.DataFrame,
    *,
    measurement_std_m: float,
    acceleration_std_mps2: float,
    initial_position_std_m: float,
    initial_velocity_std_mps: float,
    blend: float,
    max_correction_m: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = group.copy().reset_index(drop=True)
    times = work["time_s"].to_numpy(float)
    xyz = work[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    if len(work) <= 1:
        out = _smoothed_sequence_rows(work, xyz, xyz, blend=blend, max_correction_m=max_correction_m)
        return out, _sequence_diagnostics(work, xyz, xyz, out)

    filtered_x, filtered_p, predicted_x, predicted_p, transitions = _kalman_filter_sequence(
        times,
        xyz,
        measurement_std_m=measurement_std_m,
        acceleration_std_mps2=acceleration_std_mps2,
        initial_position_std_m=initial_position_std_m,
        initial_velocity_std_mps=initial_velocity_std_mps,
    )
    smoothed_x = _rts_smooth(filtered_x, filtered_p, predicted_x, predicted_p, transitions)
    smoothed_xyz = smoothed_x[:, :3]
    out = _smoothed_sequence_rows(work, xyz, smoothed_xyz, blend=blend, max_correction_m=max_correction_m)
    diagnostics = _sequence_diagnostics(work, xyz, smoothed_xyz, out)
    return out, diagnostics


def _kalman_filter_sequence(
    times: np.ndarray,
    xyz: np.ndarray,
    *,
    measurement_std_m: float,
    acceleration_std_mps2: float,
    initial_position_std_m: float,
    initial_velocity_std_mps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(times)
    x = np.zeros(6, dtype=float)
    x[:3] = xyz[0]
    x[3:] = _initial_velocity(times, xyz)
    p = np.diag(
        [
            initial_position_std_m**2,
            initial_position_std_m**2,
            initial_position_std_m**2,
            initial_velocity_std_mps**2,
            initial_velocity_std_mps**2,
            initial_velocity_std_mps**2,
        ]
    )
    h = np.zeros((3, 6), dtype=float)
    h[:, :3] = np.eye(3)
    r = (measurement_std_m**2) * np.eye(3)
    filtered_x = np.zeros((n, 6), dtype=float)
    filtered_p = np.zeros((n, 6, 6), dtype=float)
    predicted_x = np.zeros((n, 6), dtype=float)
    predicted_p = np.zeros((n, 6, 6), dtype=float)
    transitions = np.zeros((n, 6, 6), dtype=float)
    predicted_x[0] = x
    predicted_p[0] = p
    transitions[0] = np.eye(6)
    for index in range(n):
        if index > 0:
            dt = max(float(times[index] - times[index - 1]), 1.0e-6)
            f = _cv_transition(dt)
            q = _cv_process_noise(dt, acceleration_std_mps2)
            x = f @ x
            p = f @ p @ f.T + q
            predicted_x[index] = x
            predicted_p[index] = p
            transitions[index] = f
        innovation = xyz[index] - h @ x
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.inv(s)
        x = x + k @ innovation
        p = (np.eye(6) - k @ h) @ p
        p = 0.5 * (p + p.T)
        filtered_x[index] = x
        filtered_p[index] = p
    return filtered_x, filtered_p, predicted_x, predicted_p, transitions


def _rts_smooth(
    filtered_x: np.ndarray,
    filtered_p: np.ndarray,
    predicted_x: np.ndarray,
    predicted_p: np.ndarray,
    transitions: np.ndarray,
) -> np.ndarray:
    out_x = filtered_x.copy()
    out_p = filtered_p.copy()
    for index in range(len(filtered_x) - 2, -1, -1):
        f_next = transitions[index + 1]
        gain = filtered_p[index] @ f_next.T @ np.linalg.pinv(predicted_p[index + 1])
        out_x[index] = filtered_x[index] + gain @ (out_x[index + 1] - predicted_x[index + 1])
        out_p[index] = filtered_p[index] + gain @ (out_p[index + 1] - predicted_p[index + 1]) @ gain.T
        out_p[index] = 0.5 * (out_p[index] + out_p[index].T)
    return out_x


def _smoothed_sequence_rows(
    rows: pd.DataFrame,
    original_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    blend: float,
    max_correction_m: float | None,
) -> pd.DataFrame:
    delta = target_xyz - original_xyz
    raw_correction = np.linalg.norm(delta, axis=1)
    capped = np.zeros(len(rows), dtype=bool)
    if max_correction_m is not None:
        mask = raw_correction > max_correction_m
        capped[mask] = True
        scale = np.ones(len(rows), dtype=float)
        scale[mask] = float(max_correction_m) / raw_correction[mask]
        delta = delta * scale[:, None]
    smoothed_xyz = original_xyz + float(blend) * delta
    out = rows.copy()
    out["input_state_x_m"] = original_xyz[:, 0]
    out["input_state_y_m"] = original_xyz[:, 1]
    out["input_state_z_m"] = original_xyz[:, 2]
    out["state_x_m"] = smoothed_xyz[:, 0]
    out["state_y_m"] = smoothed_xyz[:, 1]
    out["state_z_m"] = smoothed_xyz[:, 2]
    out["cv_smoother_raw_correction_m"] = raw_correction
    out["cv_smoother_applied_correction_m"] = np.linalg.norm(smoothed_xyz - original_xyz, axis=1)
    out["cv_smoother_capped"] = capped
    out["cv_smoother_applied"] = True
    return out


def _sequence_diagnostics(
    rows: pd.DataFrame,
    original_xyz: np.ndarray,
    target_xyz: np.ndarray,
    out: pd.DataFrame,
) -> pd.DataFrame:
    times = rows["time_s"].to_numpy(float)
    smoothed_xyz = out[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    records: list[dict[str, Any]] = []
    for index, row in rows.reset_index(drop=True).iterrows():
        records.append(
            {
                "sequence_id": str(row["sequence_id"]),
                "time_s": float(row["time_s"]),
                "input_speed_prev_mps": _speed_to_previous(original_xyz, times, index),
                "smoothed_speed_prev_mps": _speed_to_previous(smoothed_xyz, times, index),
                "target_correction_m": float(np.linalg.norm(target_xyz[index] - original_xyz[index])),
                "cv_smoother_raw_correction_m": float(out.iloc[index]["cv_smoother_raw_correction_m"]),
                "cv_smoother_applied_correction_m": float(out.iloc[index]["cv_smoother_applied_correction_m"]),
                "cv_smoother_capped": bool(out.iloc[index]["cv_smoother_capped"]),
            }
        )
    return pd.DataFrame.from_records(records)


def _initial_velocity(times: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    if len(times) < 2:
        return np.zeros(3, dtype=float)
    for index in range(1, len(times)):
        dt = float(times[index] - times[0])
        if dt > 1.0e-6:
            return (xyz[index] - xyz[0]) / dt
    return np.zeros(3, dtype=float)


def _cv_transition(dt: float) -> np.ndarray:
    f = np.eye(6, dtype=float)
    f[0, 3] = dt
    f[1, 4] = dt
    f[2, 5] = dt
    return f


def _cv_process_noise(dt: float, acceleration_std_mps2: float) -> np.ndarray:
    q = float(acceleration_std_mps2) ** 2
    block = np.asarray([[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]], dtype=float) * q
    out = np.zeros((6, 6), dtype=float)
    for axis in range(3):
        pos = axis
        vel = axis + 3
        out[pos, pos] = block[0, 0]
        out[pos, vel] = block[0, 1]
        out[vel, pos] = block[1, 0]
        out[vel, vel] = block[1, 1]
    return out


def _normalized_submission(submission: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(submission).copy()
    lower = {str(column).strip().lower() for column in rows.columns}
    if {"sequence", "timestamp", "position", "classification"}.issubset(lower):
        official = normalize_official_track5_results_frame(rows)
        positions = [parse_official_position_cell(value) for value in official["Position"]]
        xyz = pd.DataFrame(positions, columns=["state_x_m", "state_y_m", "state_z_m"], index=official.index)
        rows = pd.DataFrame(
            {
                "sequence_id": official["Sequence"].astype(str),
                "time_s": pd.to_numeric(official["Timestamp"], errors="coerce"),
                "state_x_m": xyz["state_x_m"],
                "state_y_m": xyz["state_y_m"],
                "state_z_m": xyz["state_z_m"],
                "Classification": official["Classification"],
            }
        )
    required = {"sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m", "Classification"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"submission missing normalized columns: {sorted(missing)}")
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m", "Classification"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)).all(axis=1)
    rows = rows.loc[finite].copy()
    return rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _speed_to_previous(xyz: np.ndarray, times: np.ndarray, index: int) -> float:
    if index <= 0:
        return np.nan
    dt = float(times[index] - times[index - 1])
    if dt <= 0.0:
        return np.nan
    return float(np.linalg.norm(xyz[index] - xyz[index - 1]) / dt)


def _positive_float(value: float, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


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


def _safe_max(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.max(numeric.to_numpy(float)))


def _diagnostic_columns() -> list[str]:
    return [
        "sequence_id",
        "time_s",
        "input_speed_prev_mps",
        "smoothed_speed_prev_mps",
        "target_correction_m",
        "cv_smoother_raw_correction_m",
        "cv_smoother_applied_correction_m",
        "cv_smoother_capped",
    ]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
