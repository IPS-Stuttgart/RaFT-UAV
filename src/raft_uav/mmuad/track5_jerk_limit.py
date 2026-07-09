"""Inference-safe jerk-limit repair for MMUAD Track 5 submissions.

Speed, acceleration, and Hampel post-processors catch jumps and isolated spikes,
but leaderboard trajectories can still contain short oscillatory kinks after
ensembling several candidate pipelines.  This module estimates a local jerk
proxy from four-point finite differences, computes a conservative jerk-penalized
smooth trajectory, and only repairs rows whose local jerk and correction size
exceed explicit thresholds.

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

JERK_LIMIT_ESTIMATES_CSV = "mmuad_track5_jerk_limited_estimates.csv"
JERK_LIMIT_RESULTS_CSV = "mmaud_results_jerk_limited.csv"
JERK_LIMIT_ZIP = "ug2_submission_jerk_limited.zip"
JERK_LIMIT_DIAGNOSTICS_CSV = "mmuad_track5_jerk_limit_diagnostics.csv"
JERK_LIMIT_MANIFEST_JSON = "mmuad_track5_jerk_limit_manifest.json"
JERK_LIMIT_VALIDATION_JSON = "mmuad_track5_jerk_limit_validation.json"
JERK_LIMIT_VALIDATION_ROWS_CSV = "mmuad_track5_jerk_limit_validation_rows.csv"


def repair_track5_jerk_kinks(
    submission: pd.DataFrame,
    *,
    max_jerk_mps3: float = 80.0,
    smoothness_weight: float = 10.0,
    min_correction_m: float = 1.0,
    max_correction_m: float | None = None,
    iterations: int = 1,
    repair_blend: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return jerk-limited estimates and row diagnostics.

    A row is repaired only when its four-point local jerk proxy exceeds
    ``max_jerk_mps3`` and the jerk-penalized smoother proposes a correction of at
    least ``min_correction_m``.  Endpoints are never changed.  This makes the
    repair suitable as a train-fold-selected final Codabench post-processor.
    """

    max_jerk_mps3 = float(max_jerk_mps3)
    smoothness_weight = float(smoothness_weight)
    min_correction_m = float(min_correction_m)
    repair_blend = float(repair_blend)
    if not np.isfinite(max_jerk_mps3) or max_jerk_mps3 <= 0.0:
        raise ValueError("max_jerk_mps3 must be positive and finite")
    if not np.isfinite(smoothness_weight) or smoothness_weight < 0.0:
        raise ValueError("smoothness_weight must be finite and non-negative")
    if not np.isfinite(min_correction_m) or min_correction_m < 0.0:
        raise ValueError("min_correction_m must be finite and non-negative")
    if max_correction_m is not None:
        max_correction_m = float(max_correction_m)
        if not np.isfinite(max_correction_m) or max_correction_m <= 0.0:
            raise ValueError("max_correction_m must be positive and finite")
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
            max_jerk_mps3=max_jerk_mps3,
            smoothness_weight=smoothness_weight,
            min_correction_m=min_correction_m,
            max_correction_m=max_correction_m,
            iterations=max(1, int(iterations)),
            repair_blend=repair_blend,
        )
        repaired_parts.append(repaired)
        diagnostic_parts.append(diagnostics)
    repaired_rows = pd.concat(repaired_parts, ignore_index=True, sort=False)
    diagnostics_rows = pd.concat(diagnostic_parts, ignore_index=True, sort=False)
    return (
        repaired_rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True),
        diagnostics_rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True),
    )


def write_track5_jerk_limit_outputs(
    *,
    repaired: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    input_submission_path: Path,
    template: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    """Write jerk-limited estimates, official CSV/ZIP, and diagnostics."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "estimates_csv": output / JERK_LIMIT_ESTIMATES_CSV,
        "results_csv": output / JERK_LIMIT_RESULTS_CSV,
        "zip": output / JERK_LIMIT_ZIP,
        "diagnostics_csv": output / JERK_LIMIT_DIAGNOSTICS_CSV,
        "manifest_json": output / JERK_LIMIT_MANIFEST_JSON,
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
        validation = validate_official_track5_submission(
            paths["zip"],
            template=template,
            require_zip=True,
        )
        paths["validation_json"] = output / JERK_LIMIT_VALIDATION_JSON
        paths["validation_rows_csv"] = output / JERK_LIMIT_VALIDATION_ROWS_CSV
        paths["validation_json"].write_text(
            json.dumps(_jsonable(validation.summary), indent=2),
            encoding="utf-8",
        )
        validation.rows.to_csv(paths["validation_rows_csv"], index=False)
        validation_summary = _jsonable(validation.summary)
        if require_leaderboard_ready and not validation.summary.get("leaderboard_ready", False):
            reasons = ", ".join(validation.summary.get("leaderboard_blocking_reasons", []))
            raise SystemExit(
                f"jerk-limited submission is not leaderboard-ready: {reasons or 'unknown'}"
            )
    changed = diagnostics.get("jerk_limit_applied", pd.Series(dtype=bool)).astype(bool)
    displacement = pd.to_numeric(
        diagnostics.get("jerk_limit_displacement_m", pd.Series(dtype=float)),
        errors="coerce",
    )
    jerk = pd.to_numeric(
        diagnostics.get("jerk_limit_mps3", pd.Series(dtype=float)),
        errors="coerce",
    )
    payload = dict(manifest or {})
    payload.update(
        {
            "schema": "raft-uav-mmuad-track5-jerk-limit-v1",
            "input_submission": str(input_submission_path),
            "row_count": int(len(repaired)),
            "sequence_count": int(repaired["sequence_id"].nunique()) if not repaired.empty else 0,
            "changed_row_count": int(changed.sum()) if not diagnostics.empty else 0,
            "changed_fraction": (
                float(changed.sum() / len(diagnostics)) if len(diagnostics) else 0.0
            ),
            "max_observed_jerk_mps3": _safe_max(jerk),
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
        prog="raft-uav-mmuad-track5-jerk-limit",
        description="repair high-jerk kinks in an official MMUAD Track 5 submission",
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        help="optional official template for preflight validation",
    )
    parser.add_argument("--max-jerk-mps3", type=float, default=80.0)
    parser.add_argument("--smoothness-weight", type=float, default=10.0)
    parser.add_argument("--min-correction-m", type=float, default=1.0)
    parser.add_argument("--max-correction-m", type=float)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--repair-blend", type=float, default=1.0)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if args.require_leaderboard_ready and args.template is None:
        raise SystemExit("--require-leaderboard-ready requires --template")
    submission = load_track5_submission(args.submission)
    repaired, diagnostics = repair_track5_jerk_kinks(
        submission,
        max_jerk_mps3=float(args.max_jerk_mps3),
        smoothness_weight=float(args.smoothness_weight),
        min_correction_m=float(args.min_correction_m),
        max_correction_m=args.max_correction_m,
        iterations=int(args.iterations),
        repair_blend=float(args.repair_blend),
    )
    template = None if args.template is None else load_official_track5_template_file(args.template)
    paths = write_track5_jerk_limit_outputs(
        repaired=repaired,
        diagnostics=diagnostics,
        output_dir=args.output_dir,
        input_submission_path=args.submission,
        template=template,
        manifest={
            "max_jerk_mps3": float(args.max_jerk_mps3),
            "smoothness_weight": float(args.smoothness_weight),
            "min_correction_m": float(args.min_correction_m),
            "max_correction_m": args.max_correction_m,
            "iterations": int(args.iterations),
            "repair_blend": float(args.repair_blend),
        },
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_jerk_limit=ok")
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
        xyz = pd.DataFrame(
            positions,
            columns=["state_x_m", "state_y_m", "state_z_m"],
            index=official.index,
        )
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
    finite = np.isfinite(
        rows[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    ).all(axis=1)
    rows = rows.loc[finite].copy()
    return rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _repair_sequence(
    group: pd.DataFrame,
    *,
    max_jerk_mps3: float,
    smoothness_weight: float,
    min_correction_m: float,
    max_correction_m: float | None,
    iterations: int,
    repair_blend: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = group.copy().reset_index(drop=True)
    cumulative_changed = np.zeros(len(work), dtype=bool)
    cumulative_displacement = np.zeros(len(work), dtype=float)
    final_jerk = np.full(len(work), np.nan, dtype=float)
    for _ in range(iterations):
        repaired, diagnostics = _repair_sequence_once(
            work,
            max_jerk_mps3=max_jerk_mps3,
            smoothness_weight=smoothness_weight,
            min_correction_m=min_correction_m,
            max_correction_m=max_correction_m,
            repair_blend=repair_blend,
        )
        applied = diagnostics["jerk_limit_applied"].to_numpy(bool)
        displacement = diagnostics["jerk_limit_displacement_m"].to_numpy(float)
        jerk = diagnostics["jerk_limit_mps3"].to_numpy(float)
        cumulative_changed |= applied
        cumulative_displacement += np.nan_to_num(displacement, nan=0.0)
        final_jerk = jerk
        work = repaired
        if not applied.any():
            break
    diagnostics = pd.DataFrame(
        {
            "sequence_id": work["sequence_id"].astype(str),
            "time_s": work["time_s"].astype(float),
            "jerk_limit_mps3": final_jerk,
            "jerk_limit_applied": cumulative_changed,
            "jerk_limit_displacement_m": cumulative_displacement,
        }
    )
    return work, diagnostics


def _repair_sequence_once(
    group: pd.DataFrame,
    *,
    max_jerk_mps3: float,
    smoothness_weight: float,
    min_correction_m: float,
    max_correction_m: float | None,
    repair_blend: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = group.copy().reset_index(drop=True)
    count = len(work)
    times = work["time_s"].to_numpy(float)
    xyz = work[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    jerk = _row_jerk_proxy(times, xyz)
    if count < 4 or not np.isfinite(xyz).all():
        diagnostics = _diagnostics_for(work, jerk, np.zeros(count, dtype=bool), np.zeros(count))
        return work, diagnostics
    smoothed = _smooth_positions(times, xyz, smoothness_weight=smoothness_weight)
    correction = smoothed - xyz
    correction_norm = np.linalg.norm(correction, axis=1)
    applied = (
        np.isfinite(jerk)
        & (jerk > float(max_jerk_mps3))
        & np.isfinite(correction_norm)
        & (correction_norm >= float(min_correction_m))
    )
    if len(applied):
        applied[0] = False
        applied[-1] = False
    if max_correction_m is not None:
        scale = np.ones(len(correction_norm), dtype=float)
        too_large = correction_norm > float(max_correction_m)
        scale[too_large] = float(max_correction_m) / np.maximum(correction_norm[too_large], 1.0e-12)
        correction = correction * scale[:, None]
        correction_norm = np.linalg.norm(correction, axis=1)
    repaired_xyz = xyz.copy()
    repaired_xyz[applied] = xyz[applied] + float(repair_blend) * correction[applied]
    actual_displacement = np.linalg.norm(repaired_xyz - xyz, axis=1)
    for axis, column in enumerate(("state_x_m", "state_y_m", "state_z_m")):
        work[column] = repaired_xyz[:, axis]
    diagnostics = _diagnostics_for(work, jerk, applied, np.where(applied, actual_displacement, 0.0))
    return work, diagnostics


def _third_derivative_matrix(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    rows: list[np.ndarray] = []
    for start in range(max(0, len(times) - 3)):
        window = times[start : start + 4]
        if not np.isfinite(window).all() or len(np.unique(window)) < 4:
            continue
        coeff = np.zeros(len(times), dtype=float)
        for local, t_i in enumerate(window):
            denom = 1.0
            for other, t_j in enumerate(window):
                if other != local:
                    denom *= float(t_i - t_j)
            if abs(denom) < 1.0e-12:
                coeff = np.zeros(len(times), dtype=float)
                break
            coeff[start + local] = 6.0 / denom
        if np.any(coeff):
            rows.append(coeff)
    if not rows:
        return np.zeros((0, len(times)), dtype=float)
    return np.vstack(rows)


def _row_jerk_proxy(times: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    count = len(times)
    row_jerk = np.full(count, np.nan, dtype=float)
    d3 = _third_derivative_matrix(times)
    if d3.size == 0:
        return row_jerk
    jerk_windows = d3 @ np.asarray(xyz, dtype=float)
    norms = np.linalg.norm(jerk_windows, axis=1)
    for window_index, norm in enumerate(norms):
        for row_index in range(window_index, window_index + 4):
            if row_index < count:
                if np.isnan(row_jerk[row_index]) or norm > row_jerk[row_index]:
                    row_jerk[row_index] = float(norm)
    return row_jerk


def _smooth_positions(
    times: np.ndarray,
    xyz: np.ndarray,
    *,
    smoothness_weight: float,
) -> np.ndarray:
    positions = np.asarray(xyz, dtype=float)
    if len(positions) < 4 or smoothness_weight <= 0.0:
        return positions.copy()
    d3 = _third_derivative_matrix(times)
    if d3.size == 0:
        return positions.copy()
    system = np.eye(len(positions), dtype=float) + float(smoothness_weight) * (d3.T @ d3)
    try:
        return np.linalg.solve(system, positions)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, positions, rcond=None)[0]


def _diagnostics_for(
    rows: pd.DataFrame,
    jerk: np.ndarray,
    applied: np.ndarray,
    displacement: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": rows["sequence_id"].astype(str),
            "time_s": rows["time_s"].astype(float),
            "jerk_limit_mps3": jerk,
            "jerk_limit_applied": applied.astype(bool),
            "jerk_limit_displacement_m": displacement.astype(float),
        }
    )


def _diagnostic_columns() -> list[str]:
    return [
        "sequence_id",
        "time_s",
        "jerk_limit_mps3",
        "jerk_limit_applied",
        "jerk_limit_displacement_m",
    ]


def _safe_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(numeric.mean())


def _safe_max(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return None
    return float(numeric.max())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
