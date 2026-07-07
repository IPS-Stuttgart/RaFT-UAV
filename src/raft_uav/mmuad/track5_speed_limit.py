"""Inference-safe speed-limit projection for MMUAD/UG2+ Track 5 submissions.

Official Track 5 submissions are single UAV trajectories on a fixed
``Sequence,Timestamp`` grid.  Independent candidate/mixture/ensemble pipelines can
still produce physically implausible short jumps.  This module projects a
submission onto a per-sequence speed-limited trajectory with forward/backward
passes, preserving the official template and Classification labels.

The method uses no truth values.  It is intended as a conservative leaderboard
post-processing guard selected on train folds or applied with known platform
speed limits.
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

SPEED_LIMIT_ESTIMATES_CSV = "mmuad_track5_speed_limited_estimates.csv"
SPEED_LIMIT_RESULTS_CSV = "mmaud_results_speed_limited.csv"
SPEED_LIMIT_ZIP = "ug2_submission_speed_limited.zip"
SPEED_LIMIT_DIAGNOSTICS_CSV = "mmuad_track5_speed_limit_diagnostics.csv"
SPEED_LIMIT_MANIFEST_JSON = "mmuad_track5_speed_limit_manifest.json"
SPEED_LIMIT_VALIDATION_JSON = "mmuad_track5_speed_limit_validation.json"
SPEED_LIMIT_VALIDATION_ROWS_CSV = "mmuad_track5_speed_limit_validation_rows.csv"


def project_track5_speed_limit(
    submission: pd.DataFrame,
    *,
    max_speed_mps: float = 60.0,
    iterations: int = 2,
    anchor_blend: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return speed-limited estimates and row-level diagnostics.

    The projection performs alternating forward/backward Lipschitz passes.  If a
    consecutive displacement exceeds ``max_speed_mps * dt``, the later point is
    moved onto the boundary of the feasible ball around its neighbor.  Optional
    ``anchor_blend`` softly pulls the projected path back toward the input after
    each full iteration; use zero for a strict projection.
    """

    max_speed_mps = float(max_speed_mps)
    if not np.isfinite(max_speed_mps) or max_speed_mps <= 0.0:
        raise ValueError("max_speed_mps must be positive and finite")
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
            max_speed_mps=max_speed_mps,
            iterations=max(1, int(iterations)),
            anchor_blend=anchor_blend,
        )
        limited_parts.append(limited)
        diagnostic_parts.append(diagnostics.assign(sequence_id=str(sequence_id)))
    limited_rows = pd.concat(limited_parts, ignore_index=True, sort=False)
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True, sort=False)
    return limited_rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True), diagnostics


def write_track5_speed_limit_outputs(
    *,
    limited: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    input_submission_path: Path,
    template: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    """Write speed-limited estimates, official CSV/ZIP, diagnostics, and manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "estimates_csv": output / SPEED_LIMIT_ESTIMATES_CSV,
        "results_csv": output / SPEED_LIMIT_RESULTS_CSV,
        "zip": output / SPEED_LIMIT_ZIP,
        "diagnostics_csv": output / SPEED_LIMIT_DIAGNOSTICS_CSV,
        "manifest_json": output / SPEED_LIMIT_MANIFEST_JSON,
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
        paths["validation_json"] = output / SPEED_LIMIT_VALIDATION_JSON
        paths["validation_rows_csv"] = output / SPEED_LIMIT_VALIDATION_ROWS_CSV
        paths["validation_json"].write_text(
            json.dumps(_jsonable(validation.summary), indent=2),
            encoding="utf-8",
        )
        validation.rows.to_csv(paths["validation_rows_csv"], index=False)
        validation_summary = _jsonable(validation.summary)
        if require_leaderboard_ready and not validation.summary.get("leaderboard_ready", False):
            reasons = ", ".join(validation.summary.get("leaderboard_blocking_reasons", []))
            raise SystemExit(
                f"speed-limited submission is not leaderboard-ready: {reasons or 'unknown'}"
            )
    changed = diagnostics.get("speed_limit_applied", pd.Series(dtype=bool)).astype(bool)
    applied = pd.to_numeric(diagnostics.get("speed_limit_correction_m", pd.Series(dtype=float)), errors="coerce")
    payload = dict(manifest or {})
    payload.update(
        {
            "schema": "raft-uav-mmuad-track5-speed-limit-v1",
            "input_submission": str(input_submission_path),
            "row_count": int(len(limited)),
            "sequence_count": int(limited["sequence_id"].nunique()) if not limited.empty else 0,
            "changed_row_count": int(changed.sum()) if not diagnostics.empty else 0,
            "changed_fraction": float(changed.sum() / len(diagnostics)) if len(diagnostics) else 0.0,
            "max_correction_m": _safe_max(applied),
            "mean_correction_m": _safe_mean(applied),
            "validation": validation_summary,
            "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
        }
    )
    paths["manifest_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-speed-limit",
        description="project an official MMUAD Track 5 submission onto a speed-limited trajectory",
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, help="optional official template for preflight validation")
    parser.add_argument("--max-speed-mps", type=float, default=60.0)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--anchor-blend", type=float, default=0.0)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if args.require_leaderboard_ready and args.template is None:
        raise SystemExit("--require-leaderboard-ready requires --template")
    submission = load_track5_submission(args.submission)
    limited, diagnostics = project_track5_speed_limit(
        submission,
        max_speed_mps=float(args.max_speed_mps),
        iterations=int(args.iterations),
        anchor_blend=float(args.anchor_blend),
    )
    template = None if args.template is None else load_official_track5_template_file(args.template)
    paths = write_track5_speed_limit_outputs(
        limited=limited,
        diagnostics=diagnostics,
        output_dir=args.output_dir,
        input_submission_path=args.submission,
        template=template,
        manifest={
            "max_speed_mps": float(args.max_speed_mps),
            "iterations": int(args.iterations),
            "anchor_blend": float(args.anchor_blend),
        },
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_speed_limit=ok")
    print(f"changed_row_count={manifest['changed_row_count']}")
    print(f"changed_fraction={manifest['changed_fraction']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _project_sequence(
    group: pd.DataFrame,
    *,
    max_speed_mps: float,
    iterations: int,
    anchor_blend: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = group.copy().reset_index(drop=True)
    original_xyz = work[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    xyz = original_xyz.copy()
    times = work["time_s"].to_numpy(float)
    for _ in range(iterations):
        xyz = _forward_speed_pass(xyz, times, max_speed_mps=max_speed_mps)
        xyz = _backward_speed_pass(xyz, times, max_speed_mps=max_speed_mps)
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
    out["speed_limit_applied"] = correction > 1.0e-9
    out["speed_limit_correction_m"] = correction
    out["speed_limit_max_speed_mps"] = float(max_speed_mps)
    diagnostics = _sequence_diagnostics(work, xyz, original_xyz, max_speed_mps=max_speed_mps)
    return out, diagnostics


def _forward_speed_pass(xyz: np.ndarray, times: np.ndarray, *, max_speed_mps: float) -> np.ndarray:
    out = xyz.copy()
    for index in range(1, len(out)):
        out[index] = _clip_to_speed_ball(
            out[index],
            out[index - 1],
            times[index] - times[index - 1],
            max_speed_mps,
        )
    return out


def _backward_speed_pass(xyz: np.ndarray, times: np.ndarray, *, max_speed_mps: float) -> np.ndarray:
    out = xyz.copy()
    for index in range(len(out) - 2, -1, -1):
        out[index] = _clip_to_speed_ball(
            out[index],
            out[index + 1],
            times[index + 1] - times[index],
            max_speed_mps,
        )
    return out


def _clip_to_speed_ball(point: np.ndarray, anchor: np.ndarray, dt_s: float, max_speed_mps: float) -> np.ndarray:
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        return point
    delta = point - anchor
    distance = float(np.linalg.norm(delta))
    max_distance = float(max_speed_mps) * float(dt_s)
    if distance <= max_distance or distance <= 0.0:
        return point
    return anchor + delta * (max_distance / distance)


def _sequence_diagnostics(
    rows: pd.DataFrame,
    xyz: np.ndarray,
    original_xyz: np.ndarray,
    *,
    max_speed_mps: float,
) -> pd.DataFrame:
    times = rows["time_s"].to_numpy(float)
    input_speed_prev = _previous_speeds(original_xyz, times)
    output_speed_prev = _previous_speeds(xyz, times)
    correction = np.linalg.norm(xyz - original_xyz, axis=1)
    return pd.DataFrame(
        {
            "sequence_id": rows["sequence_id"].astype(str),
            "time_s": times,
            "input_speed_prev_mps": input_speed_prev,
            "output_speed_prev_mps": output_speed_prev,
            "speed_limit_max_speed_mps": float(max_speed_mps),
            "speed_limit_applied": correction > 1.0e-9,
            "speed_limit_correction_m": correction,
        }
    )


def _previous_speeds(xyz: np.ndarray, times: np.ndarray) -> np.ndarray:
    speeds = np.full(len(xyz), np.nan, dtype=float)
    for index in range(1, len(xyz)):
        dt = times[index] - times[index - 1]
        if np.isfinite(dt) and dt > 0.0:
            speeds[index] = float(np.linalg.norm(xyz[index] - xyz[index - 1]) / dt)
    return speeds


def _normalized_submission(submission: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(submission).copy()
    required = {"sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m", "Classification"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"submission missing normalized columns: {sorted(missing)}")
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m", "Classification"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(rows[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _diagnostic_columns() -> list[str]:
    return [
        "sequence_id",
        "time_s",
        "input_speed_prev_mps",
        "output_speed_prev_mps",
        "speed_limit_max_speed_mps",
        "speed_limit_applied",
        "speed_limit_correction_m",
    ]


def _safe_mean(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite.to_numpy(float))]
    if finite.empty:
        return None
    return float(np.mean(finite.to_numpy(float)))


def _safe_max(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite.to_numpy(float))]
    if finite.empty:
        return None
    return float(np.max(finite.to_numpy(float)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
