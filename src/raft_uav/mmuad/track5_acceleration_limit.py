"""Inference-safe acceleration-limit projection for MMUAD/UG2+ Track 5 submissions.

Track 5 predictions are single-UAV trajectories on a fixed template grid.  A
candidate or ensemble pipeline may satisfy a speed guard while still containing
short acceleration spikes that are implausible for a UAV and expensive under the
pose-MSE leaderboard metric.  This module projects an official submission onto a
locally constant-velocity trajectory with bounded acceleration, preserving the
Track 5 template and classification labels.

The method uses no truth values.  Select the acceleration limit on train folds or
from platform constraints before applying it to hidden-test submissions.
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
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_submission_ensemble import _jsonable
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission

ACCEL_LIMIT_ESTIMATES_CSV = "mmuad_track5_acceleration_limited_estimates.csv"
ACCEL_LIMIT_RESULTS_CSV = "mmaud_results_acceleration_limited.csv"
ACCEL_LIMIT_ZIP = "ug2_submission_acceleration_limited.zip"
ACCEL_LIMIT_DIAGNOSTICS_CSV = "mmuad_track5_acceleration_limit_diagnostics.csv"
ACCEL_LIMIT_MANIFEST_JSON = "mmuad_track5_acceleration_limit_manifest.json"
ACCEL_LIMIT_VALIDATION_JSON = "mmuad_track5_acceleration_limit_validation.json"
ACCEL_LIMIT_VALIDATION_ROWS_CSV = "mmuad_track5_acceleration_limit_validation_rows.csv"


def project_track5_acceleration_limit(
    submission: pd.DataFrame,
    *,
    max_acceleration_mps2: float = 20.0,
    iterations: int = 2,
    anchor_blend: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return acceleration-limited estimates and row-level diagnostics.

    The projection alternates forward and backward passes.  A forward pass clips
    each point to a ball around the constant-velocity prediction from the two
    previous points; the ball radius is ``0.5 * a_max * dt^2``.  The backward
    pass applies the same rule in reverse time.  Optional ``anchor_blend`` softly
    pulls the projected path back toward the input after each full iteration.
    """

    max_acceleration_mps2 = float(max_acceleration_mps2)
    if not np.isfinite(max_acceleration_mps2) or max_acceleration_mps2 <= 0.0:
        raise ValueError("max_acceleration_mps2 must be positive and finite")
    anchor_blend = float(anchor_blend)
    if not np.isfinite(anchor_blend) or not 0.0 <= anchor_blend < 1.0:
        raise ValueError("anchor_blend must be finite and in [0, 1)")
    normalized = _normalized_submission(submission)
    if normalized.empty:
        return normalized, pd.DataFrame(columns=_diagnostic_columns())

    limited_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    for sequence_id, group in normalized.groupby("sequence_id", sort=True):
        limited, diagnostics = _project_sequence(
            group.sort_values("time_s").reset_index(drop=True),
            max_acceleration_mps2=max_acceleration_mps2,
            iterations=max(1, int(iterations)),
            anchor_blend=anchor_blend,
        )
        limited_parts.append(limited)
        diagnostic_parts.append(diagnostics.assign(sequence_id=str(sequence_id)))
    limited_rows = pd.concat(limited_parts, ignore_index=True, sort=False)
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True, sort=False)
    return limited_rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True), diagnostics


def write_track5_acceleration_limit_outputs(
    *,
    limited: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    input_submission_path: Path,
    template: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    """Write acceleration-limited estimates, official CSV/ZIP, and manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "estimates_csv": output / ACCEL_LIMIT_ESTIMATES_CSV,
        "results_csv": output / ACCEL_LIMIT_RESULTS_CSV,
        "zip": output / ACCEL_LIMIT_ZIP,
        "diagnostics_csv": output / ACCEL_LIMIT_DIAGNOSTICS_CSV,
        "manifest_json": output / ACCEL_LIMIT_MANIFEST_JSON,
    }
    limited.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    official_rows = limited.copy()
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
        paths["validation_json"] = output / ACCEL_LIMIT_VALIDATION_JSON
        paths["validation_rows_csv"] = output / ACCEL_LIMIT_VALIDATION_ROWS_CSV
        paths["validation_json"].write_text(
            json.dumps(_jsonable(validation.summary), indent=2),
            encoding="utf-8",
        )
        validation.rows.to_csv(paths["validation_rows_csv"], index=False)
        validation_summary = _jsonable(validation.summary)
        if require_leaderboard_ready and not validation.summary.get("leaderboard_ready", False):
            reasons = ", ".join(validation.summary.get("leaderboard_blocking_reasons", []))
            raise SystemExit(
                f"acceleration-limited submission is not leaderboard-ready: {reasons or 'unknown'}"
            )
    changed = diagnostics.get("acceleration_limit_applied", pd.Series(dtype=bool)).astype(bool)
    corrections = pd.to_numeric(
        diagnostics.get("acceleration_limit_correction_m", pd.Series(dtype=float)),
        errors="coerce",
    )
    before_accel = pd.to_numeric(
        diagnostics.get("input_acceleration_mps2", pd.Series(dtype=float)),
        errors="coerce",
    )
    after_accel = pd.to_numeric(
        diagnostics.get("limited_acceleration_mps2", pd.Series(dtype=float)),
        errors="coerce",
    )
    payload = dict(manifest or {})
    payload.update(
        {
            "schema": "raft-uav-mmuad-track5-acceleration-limit-v1",
            "input_submission": str(input_submission_path),
            "row_count": int(len(limited)),
            "sequence_count": int(limited["sequence_id"].nunique()) if not limited.empty else 0,
            "changed_row_count": int(changed.sum()) if not diagnostics.empty else 0,
            "changed_fraction": float(changed.sum() / len(diagnostics)) if len(diagnostics) else 0.0,
            "max_correction_m": _safe_max(corrections),
            "mean_correction_m": _safe_mean(corrections),
            "input_acceleration_p95_mps2": _safe_percentile(before_accel, 95),
            "limited_acceleration_p95_mps2": _safe_percentile(after_accel, 95),
            "input_acceleration_max_mps2": _safe_max(before_accel),
            "limited_acceleration_max_mps2": _safe_max(after_accel),
            "validation": validation_summary,
            "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
        }
    )
    paths["manifest_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-acceleration-limit",
        description="project an official MMUAD Track 5 submission onto an acceleration-limited trajectory",
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, help="optional official template for preflight validation")
    parser.add_argument("--max-acceleration-mps2", type=float, default=20.0)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--anchor-blend", type=float, default=0.0)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if args.require_leaderboard_ready and args.template is None:
        raise SystemExit("--require-leaderboard-ready requires --template")
    submission = load_track5_submission(args.submission)
    limited, diagnostics = project_track5_acceleration_limit(
        submission,
        max_acceleration_mps2=float(args.max_acceleration_mps2),
        iterations=int(args.iterations),
        anchor_blend=float(args.anchor_blend),
    )
    template = None if args.template is None else load_official_track5_template_file(args.template)
    paths = write_track5_acceleration_limit_outputs(
        limited=limited,
        diagnostics=diagnostics,
        output_dir=args.output_dir,
        input_submission_path=args.submission,
        template=template,
        manifest={
            "max_acceleration_mps2": float(args.max_acceleration_mps2),
            "iterations": int(args.iterations),
            "anchor_blend": float(args.anchor_blend),
        },
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    manifest_payload = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_acceleration_limit=ok")
    print(f"changed_row_count={manifest_payload['changed_row_count']}")
    print(f"changed_fraction={manifest_payload['changed_fraction']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _project_sequence(
    group: pd.DataFrame,
    *,
    max_acceleration_mps2: float,
    iterations: int,
    anchor_blend: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = group.copy().reset_index(drop=True)
    original_xyz = work[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    xyz = original_xyz.copy()
    times = work["time_s"].to_numpy(float)
    for _ in range(iterations):
        xyz = _forward_acceleration_pass(xyz, times, max_acceleration_mps2=max_acceleration_mps2)
        xyz = _backward_acceleration_pass(xyz, times, max_acceleration_mps2=max_acceleration_mps2)
        if anchor_blend > 0.0:
            xyz = (1.0 - anchor_blend) * xyz + anchor_blend * original_xyz
    out = work.copy()
    out["input_state_x_m"] = original_xyz[:, 0]
    out["input_state_y_m"] = original_xyz[:, 1]
    out["input_state_z_m"] = original_xyz[:, 2]
    out["state_x_m"] = xyz[:, 0]
    out["state_y_m"] = xyz[:, 1]
    out["state_z_m"] = xyz[:, 2]
    correction = np.linalg.norm(xyz - original_xyz, axis=1)
    out["acceleration_limit_applied"] = correction > 1.0e-9
    out["acceleration_limit_correction_m"] = correction
    out["acceleration_limit_max_acceleration_mps2"] = float(max_acceleration_mps2)
    diagnostics = _sequence_diagnostics(work, xyz, original_xyz, max_acceleration_mps2=max_acceleration_mps2)
    return out, diagnostics


def _forward_acceleration_pass(
    xyz: np.ndarray,
    times: np.ndarray,
    *,
    max_acceleration_mps2: float,
) -> np.ndarray:
    out = xyz.copy()
    for index in range(2, len(out)):
        dt_prev = times[index - 1] - times[index - 2]
        dt_next = times[index] - times[index - 1]
        if not np.isfinite(dt_prev) or not np.isfinite(dt_next) or dt_prev <= 0.0 or dt_next <= 0.0:
            continue
        velocity = (out[index - 1] - out[index - 2]) / dt_prev
        prediction = out[index - 1] + velocity * dt_next
        out[index] = _clip_to_acceleration_ball(
            out[index],
            prediction,
            0.5 * max_acceleration_mps2 * dt_next * dt_next,
        )
    return out


def _backward_acceleration_pass(
    xyz: np.ndarray,
    times: np.ndarray,
    *,
    max_acceleration_mps2: float,
) -> np.ndarray:
    out = xyz.copy()
    for index in range(len(out) - 3, -1, -1):
        dt_next = times[index + 2] - times[index + 1]
        dt_prev = times[index + 1] - times[index]
        if not np.isfinite(dt_prev) or not np.isfinite(dt_next) or dt_prev <= 0.0 or dt_next <= 0.0:
            continue
        reverse_velocity = (out[index + 1] - out[index + 2]) / dt_next
        prediction = out[index + 1] + reverse_velocity * dt_prev
        out[index] = _clip_to_acceleration_ball(
            out[index],
            prediction,
            0.5 * max_acceleration_mps2 * dt_prev * dt_prev,
        )
    return out


def _clip_to_acceleration_ball(point: np.ndarray, prediction: np.ndarray, radius_m: float) -> np.ndarray:
    if not np.isfinite(radius_m) or radius_m < 0.0:
        return point
    delta = point - prediction
    norm = float(np.linalg.norm(delta))
    if not np.isfinite(norm) or norm <= radius_m or norm == 0.0:
        return point
    return prediction + delta * (radius_m / norm)


def _sequence_diagnostics(
    original_rows: pd.DataFrame,
    limited_xyz: np.ndarray,
    original_xyz: np.ndarray,
    *,
    max_acceleration_mps2: float,
) -> pd.DataFrame:
    times = original_rows["time_s"].to_numpy(float)
    correction = np.linalg.norm(limited_xyz - original_xyz, axis=1)
    input_accel = _acceleration_series(original_xyz, times)
    limited_accel = _acceleration_series(limited_xyz, times)
    diagnostics = pd.DataFrame(
        {
            "time_s": times,
            "Classification": original_rows["Classification"].to_numpy(),
            "input_acceleration_mps2": input_accel,
            "limited_acceleration_mps2": limited_accel,
            "acceleration_limit_applied": correction > 1.0e-9,
            "acceleration_limit_correction_m": correction,
            "max_acceleration_mps2": float(max_acceleration_mps2),
        }
    )
    return diagnostics


def _acceleration_series(xyz: np.ndarray, times: np.ndarray) -> np.ndarray:
    acceleration = np.full(len(xyz), np.nan, dtype=float)
    for index in range(1, len(xyz) - 1):
        dt_prev = times[index] - times[index - 1]
        dt_next = times[index + 1] - times[index]
        if not np.isfinite(dt_prev) or not np.isfinite(dt_next) or dt_prev <= 0.0 or dt_next <= 0.0:
            continue
        velocity_prev = (xyz[index] - xyz[index - 1]) / dt_prev
        velocity_next = (xyz[index + 1] - xyz[index]) / dt_next
        dt_mid = 0.5 * (dt_prev + dt_next)
        acceleration[index] = float(np.linalg.norm((velocity_next - velocity_prev) / dt_mid))
    return acceleration


def _normalized_submission(submission: pd.DataFrame) -> pd.DataFrame:
    rows = submission.copy()
    required = ["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"submission missing normalized columns: {missing}")
    if "Classification" not in rows.columns:
        rows["Classification"] = 0
    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows = rows.dropna(subset=["time_s", "state_x_m", "state_y_m", "state_z_m"])
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    rows["Classification"] = pd.to_numeric(rows["Classification"], errors="coerce").fillna(0).astype(int)
    return rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _diagnostic_columns() -> list[str]:
    return [
        "sequence_id",
        "time_s",
        "Classification",
        "input_acceleration_mps2",
        "limited_acceleration_mps2",
        "acceleration_limit_applied",
        "acceleration_limit_correction_m",
        "max_acceleration_mps2",
    ]


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values.to_numpy(float))]
    if values.empty:
        return None
    return float(values.mean())


def _safe_max(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values.to_numpy(float))]
    if values.empty:
        return None
    return float(values.max())


def _safe_percentile(series: pd.Series, percentile: float) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values.to_numpy(float))]
    if values.empty:
        return None
    return float(np.percentile(values, percentile))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
