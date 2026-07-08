"""Inference-safe acceleration-limit repair for MMUAD Track 5 submissions.

The existing speed-limit and isolated-spike repair utilities catch large jumps, but
leaderboard trajectories can still contain short high-acceleration kinks whose
adjacent speeds stay below a coarse speed gate.  This module repairs conservative
interior points by projecting them to the linear interpolation of their neighbors
when the implied local acceleration is implausible and the neighbor-to-neighbor
motion is still plausible.

The procedure uses no truth values and preserves the official Track 5
Sequence/Timestamp grid and Classification labels.
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

ACC_LIMIT_ESTIMATES_CSV = "mmuad_track5_acceleration_limited_estimates.csv"
ACC_LIMIT_RESULTS_CSV = "mmaud_results_acceleration_limited.csv"
ACC_LIMIT_ZIP = "ug2_submission_acceleration_limited.zip"
ACC_LIMIT_DIAGNOSTICS_CSV = "mmuad_track5_acceleration_limit_diagnostics.csv"
ACC_LIMIT_MANIFEST_JSON = "mmuad_track5_acceleration_limit_manifest.json"
ACC_LIMIT_VALIDATION_JSON = "mmuad_track5_acceleration_limit_validation.json"
ACC_LIMIT_VALIDATION_ROWS_CSV = "mmuad_track5_acceleration_limit_validation_rows.csv"


def repair_track5_acceleration_kinks(
    submission: pd.DataFrame,
    *,
    max_acceleration_mps2: float = 20.0,
    max_direct_speed_mps: float = 80.0,
    min_interpolation_residual_m: float = 1.0,
    iterations: int = 2,
    repair_blend: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return acceleration-limited estimates and row diagnostics.

    A row is repaired only when it is an interior point with finite neighbors,
    the local velocity change exceeds ``max_acceleration_mps2``, the direct
    neighbor-to-neighbor speed is at most ``max_direct_speed_mps``, and the point
    is at least ``min_interpolation_residual_m`` from linear neighbor
    interpolation.  ``repair_blend=1`` replaces the point by the interpolation;
    lower values move only partway, which is useful for conservative train-fold
    selected post-processing.
    """

    max_acceleration_mps2 = float(max_acceleration_mps2)
    max_direct_speed_mps = float(max_direct_speed_mps)
    min_interpolation_residual_m = float(min_interpolation_residual_m)
    repair_blend = float(repair_blend)
    if not np.isfinite(max_acceleration_mps2) or max_acceleration_mps2 <= 0.0:
        raise ValueError("max_acceleration_mps2 must be positive and finite")
    if not np.isfinite(max_direct_speed_mps) or max_direct_speed_mps <= 0.0:
        raise ValueError("max_direct_speed_mps must be positive and finite")
    if not np.isfinite(min_interpolation_residual_m) or min_interpolation_residual_m < 0.0:
        raise ValueError("min_interpolation_residual_m must be finite and non-negative")
    if not np.isfinite(repair_blend) or not 0.0 <= repair_blend <= 1.0:
        raise ValueError("repair_blend must be finite and in [0, 1]")

    rows = _normalized_submission(submission)
    if rows.empty:
        return rows, pd.DataFrame(columns=_diagnostic_columns())

    repaired_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    for _, group in rows.groupby("sequence_id", sort=True):
        repaired, diagnostics = _repair_sequence(
            group.sort_values("time_s").reset_index(drop=True),
            max_acceleration_mps2=max_acceleration_mps2,
            max_direct_speed_mps=max_direct_speed_mps,
            min_interpolation_residual_m=min_interpolation_residual_m,
            iterations=max(1, int(iterations)),
            repair_blend=repair_blend,
        )
        repaired_parts.append(repaired)
        diagnostic_parts.append(diagnostics)
    repaired_rows = pd.concat(repaired_parts, ignore_index=True, sort=False)
    diagnostics_rows = pd.concat(diagnostic_parts, ignore_index=True, sort=False)
    return repaired_rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True), diagnostics_rows


def write_track5_acceleration_limit_outputs(
    *,
    repaired: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    input_submission_path: Path,
    template: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    """Write acceleration-limited estimates, official CSV/ZIP, and diagnostics."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "estimates_csv": output / ACC_LIMIT_ESTIMATES_CSV,
        "results_csv": output / ACC_LIMIT_RESULTS_CSV,
        "zip": output / ACC_LIMIT_ZIP,
        "diagnostics_csv": output / ACC_LIMIT_DIAGNOSTICS_CSV,
        "manifest_json": output / ACC_LIMIT_MANIFEST_JSON,
    }
    repaired.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    official_rows = repaired.copy()
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
        paths["validation_json"] = output / ACC_LIMIT_VALIDATION_JSON
        paths["validation_rows_csv"] = output / ACC_LIMIT_VALIDATION_ROWS_CSV
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
    displacement = pd.to_numeric(
        diagnostics.get("acceleration_limit_displacement_m", pd.Series(dtype=float)),
        errors="coerce",
    )
    payload = dict(manifest or {})
    payload.update(
        {
            "schema": "raft-uav-mmuad-track5-acceleration-limit-v1",
            "input_submission": str(input_submission_path),
            "row_count": int(len(repaired)),
            "sequence_count": int(repaired["sequence_id"].nunique()) if not repaired.empty else 0,
            "changed_row_count": int(changed.sum()) if not diagnostics.empty else 0,
            "changed_fraction": float(changed.sum() / len(diagnostics)) if len(diagnostics) else 0.0,
            "max_correction_m": _safe_max(displacement),
            "mean_correction_m": _safe_mean(displacement),
            "validation": validation_summary,
            "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
        }
    )
    paths["manifest_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-acceleration-limit",
        description="repair high-acceleration kinks in an official MMUAD Track 5 submission",
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, help="optional official template for preflight validation")
    parser.add_argument("--max-acceleration-mps2", type=float, default=20.0)
    parser.add_argument("--max-direct-speed-mps", type=float, default=80.0)
    parser.add_argument("--min-interpolation-residual-m", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--repair-blend", type=float, default=1.0)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if args.require_leaderboard_ready and args.template is None:
        raise SystemExit("--require-leaderboard-ready requires --template")
    submission = load_track5_submission(args.submission)
    repaired, diagnostics = repair_track5_acceleration_kinks(
        submission,
        max_acceleration_mps2=float(args.max_acceleration_mps2),
        max_direct_speed_mps=float(args.max_direct_speed_mps),
        min_interpolation_residual_m=float(args.min_interpolation_residual_m),
        iterations=int(args.iterations),
        repair_blend=float(args.repair_blend),
    )
    template = None if args.template is None else load_official_track5_template_file(args.template)
    paths = write_track5_acceleration_limit_outputs(
        repaired=repaired,
        diagnostics=diagnostics,
        output_dir=args.output_dir,
        input_submission_path=args.submission,
        template=template,
        manifest={
            "max_acceleration_mps2": float(args.max_acceleration_mps2),
            "max_direct_speed_mps": float(args.max_direct_speed_mps),
            "min_interpolation_residual_m": float(args.min_interpolation_residual_m),
            "iterations": int(args.iterations),
            "repair_blend": float(args.repair_blend),
        },
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_acceleration_limit=ok")
    print(f"changed_row_count={manifest['changed_row_count']}")
    print(f"changed_fraction={manifest['changed_fraction']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


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


def _repair_sequence(
    group: pd.DataFrame,
    *,
    max_acceleration_mps2: float,
    max_direct_speed_mps: float,
    min_interpolation_residual_m: float,
    iterations: int,
    repair_blend: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = group.copy().reset_index(drop=True)
    work["acceleration_limit_applied"] = False
    work["acceleration_limit_iteration"] = 0
    work["acceleration_limit_displacement_m"] = 0.0
    for iteration in range(1, max(1, int(iterations)) + 1):
        diagnostics = _sequence_diagnostics(work, iteration=iteration)
        repair_mask = _repair_candidate_mask(
            diagnostics,
            max_acceleration_mps2=max_acceleration_mps2,
            max_direct_speed_mps=max_direct_speed_mps,
            min_interpolation_residual_m=min_interpolation_residual_m,
        )
        if not repair_mask.any():
            break
        xyz = work[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float, copy=True)
        for idx in np.flatnonzero(repair_mask):
            interpolated = diagnostics.loc[idx, ["interp_x_m", "interp_y_m", "interp_z_m"]].to_numpy(float)
            new_xyz = ((1.0 - repair_blend) * xyz[idx]) + (repair_blend * interpolated)
            displacement = float(np.linalg.norm(xyz[idx] - new_xyz))
            xyz[idx] = new_xyz
            work.loc[idx, ["state_x_m", "state_y_m", "state_z_m"]] = new_xyz
            work.loc[idx, "acceleration_limit_applied"] = True
            work.loc[idx, "acceleration_limit_iteration"] = iteration
            work.loc[idx, "acceleration_limit_displacement_m"] += displacement
    final = _sequence_diagnostics(work, iteration=max(1, int(iterations)) + 1)
    final["acceleration_limit_candidate"] = _repair_candidate_mask(
        final,
        max_acceleration_mps2=max_acceleration_mps2,
        max_direct_speed_mps=max_direct_speed_mps,
        min_interpolation_residual_m=min_interpolation_residual_m,
    )
    final["acceleration_limit_applied"] = work["acceleration_limit_applied"].to_numpy(bool)
    final["acceleration_limit_iteration"] = work["acceleration_limit_iteration"].to_numpy(int)
    final["acceleration_limit_displacement_m"] = work["acceleration_limit_displacement_m"].to_numpy(float)
    final["max_acceleration_mps2"] = float(max_acceleration_mps2)
    final["max_direct_speed_mps"] = float(max_direct_speed_mps)
    final["min_interpolation_residual_m"] = float(min_interpolation_residual_m)
    return work, final


def _sequence_diagnostics(group: pd.DataFrame, *, iteration: int) -> pd.DataFrame:
    xyz = group[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    times = group["time_s"].to_numpy(float)
    records: list[dict[str, Any]] = []
    for idx, row in group.iterrows():
        record = {
            "sequence_id": str(row["sequence_id"]),
            "time_s": float(row["time_s"]),
            "diagnostic_iteration": int(iteration),
            "local_acceleration_mps2": np.nan,
            "interpolation_residual_m": np.nan,
            "neighbor_direct_speed_mps": np.nan,
            "interp_x_m": np.nan,
            "interp_y_m": np.nan,
            "interp_z_m": np.nan,
        }
        if 0 < idx < len(group) - 1:
            prev_t, cur_t, next_t = times[idx - 1], times[idx], times[idx + 1]
            dt_prev = cur_t - prev_t
            dt_next = next_t - cur_t
            dt_direct = next_t - prev_t
            if dt_prev > 0.0 and dt_next > 0.0 and dt_direct > 0.0:
                alpha = (cur_t - prev_t) / dt_direct
                interp = (1.0 - alpha) * xyz[idx - 1] + alpha * xyz[idx + 1]
                residual = float(np.linalg.norm(xyz[idx] - interp))
                v_prev = (xyz[idx] - xyz[idx - 1]) / dt_prev
                v_next = (xyz[idx + 1] - xyz[idx]) / dt_next
                dt_avg = 0.5 * (dt_prev + dt_next)
                accel = float(np.linalg.norm(v_next - v_prev) / dt_avg)
                direct_speed = float(np.linalg.norm(xyz[idx + 1] - xyz[idx - 1]) / dt_direct)
                record.update(
                    {
                        "local_acceleration_mps2": accel,
                        "interpolation_residual_m": residual,
                        "neighbor_direct_speed_mps": direct_speed,
                        "interp_x_m": float(interp[0]),
                        "interp_y_m": float(interp[1]),
                        "interp_z_m": float(interp[2]),
                    }
                )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _repair_candidate_mask(
    diagnostics: pd.DataFrame,
    *,
    max_acceleration_mps2: float,
    max_direct_speed_mps: float,
    min_interpolation_residual_m: float,
) -> np.ndarray:
    accel = pd.to_numeric(diagnostics["local_acceleration_mps2"], errors="coerce").to_numpy(float)
    residual = pd.to_numeric(diagnostics["interpolation_residual_m"], errors="coerce").to_numpy(float)
    direct_speed = pd.to_numeric(diagnostics["neighbor_direct_speed_mps"], errors="coerce").to_numpy(float)
    return (
        np.isfinite(accel)
        & np.isfinite(residual)
        & np.isfinite(direct_speed)
        & (accel > float(max_acceleration_mps2))
        & (direct_speed <= float(max_direct_speed_mps))
        & (residual >= float(min_interpolation_residual_m))
    )


def _diagnostic_columns() -> list[str]:
    return [
        "sequence_id",
        "time_s",
        "diagnostic_iteration",
        "local_acceleration_mps2",
        "interpolation_residual_m",
        "neighbor_direct_speed_mps",
        "interp_x_m",
        "interp_y_m",
        "interp_z_m",
        "acceleration_limit_candidate",
        "acceleration_limit_applied",
        "acceleration_limit_iteration",
        "acceleration_limit_displacement_m",
    ]


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.mean(numeric.to_numpy(float)))


def _safe_max(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(np.max(numeric.to_numpy(float)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
