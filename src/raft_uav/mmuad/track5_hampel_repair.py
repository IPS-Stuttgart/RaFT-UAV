"""Inference-safe Hampel spike repair for MMUAD/UG2+ Track 5 submissions.

Leaderboard submissions can contain isolated position spikes that are too small
or too local to be caught reliably by coarse speed and acceleration guards.  This
module detects per-sequence spatial outliers against a rolling local median and
moves only those rows toward the local consensus.  It preserves the official
``Sequence,Timestamp`` grid and classification labels and uses no truth values.
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

HAMPEL_ESTIMATES_CSV = "mmuad_track5_hampel_repaired_estimates.csv"
HAMPEL_RESULTS_CSV = "mmaud_results_hampel_repaired.csv"
HAMPEL_ZIP = "ug2_submission_hampel_repaired.zip"
HAMPEL_DIAGNOSTICS_CSV = "mmuad_track5_hampel_repair_diagnostics.csv"
HAMPEL_MANIFEST_JSON = "mmuad_track5_hampel_repair_manifest.json"
HAMPEL_VALIDATION_JSON = "mmuad_track5_hampel_repair_validation.json"
HAMPEL_VALIDATION_ROWS_CSV = "mmuad_track5_hampel_repair_validation_rows.csv"


def repair_track5_hampel_spikes(
    submission: pd.DataFrame,
    *,
    window_radius: int = 2,
    sigma_threshold: float = 3.0,
    min_scale_m: float = 1.0,
    min_residual_m: float = 3.0,
    repair_blend: float = 1.0,
    iterations: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Hampel-repaired estimates and row-level diagnostics.

    For each sequence, an interior row is compared with the median of nearby
    rows excluding the row itself.  The robust scale is the median absolute
    neighbor distance to that local median, multiplied by 1.4826 and floored by
    ``min_scale_m``.  A row is repaired when its residual exceeds both
    ``min_residual_m`` and ``sigma_threshold * scale``.  ``repair_blend=1``
    replaces the row by the local median; smaller values move partway.
    """

    window_radius = int(window_radius)
    iterations = int(iterations)
    sigma_threshold = float(sigma_threshold)
    min_scale_m = float(min_scale_m)
    min_residual_m = float(min_residual_m)
    repair_blend = float(repair_blend)
    if window_radius < 1:
        raise ValueError("window_radius must be at least 1")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if not np.isfinite(sigma_threshold) or sigma_threshold <= 0.0:
        raise ValueError("sigma_threshold must be positive and finite")
    if not np.isfinite(min_scale_m) or min_scale_m <= 0.0:
        raise ValueError("min_scale_m must be positive and finite")
    if not np.isfinite(min_residual_m) or min_residual_m < 0.0:
        raise ValueError("min_residual_m must be finite and non-negative")
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
            window_radius=window_radius,
            sigma_threshold=sigma_threshold,
            min_scale_m=min_scale_m,
            min_residual_m=min_residual_m,
            repair_blend=repair_blend,
            iterations=iterations,
        )
        repaired_parts.append(repaired)
        diagnostic_parts.append(diagnostics)
    repaired_rows = pd.concat(repaired_parts, ignore_index=True, sort=False)
    diagnostics_rows = pd.concat(diagnostic_parts, ignore_index=True, sort=False)
    return (
        repaired_rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True),
        diagnostics_rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True),
    )


def write_track5_hampel_repair_outputs(
    *,
    repaired: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    input_submission_path: Path,
    template: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    """Write repaired estimates, official CSV/ZIP, diagnostics, and manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "estimates_csv": output / HAMPEL_ESTIMATES_CSV,
        "results_csv": output / HAMPEL_RESULTS_CSV,
        "zip": output / HAMPEL_ZIP,
        "diagnostics_csv": output / HAMPEL_DIAGNOSTICS_CSV,
        "manifest_json": output / HAMPEL_MANIFEST_JSON,
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
        paths["validation_json"] = output / HAMPEL_VALIDATION_JSON
        paths["validation_rows_csv"] = output / HAMPEL_VALIDATION_ROWS_CSV
        paths["validation_json"].write_text(
            json.dumps(_jsonable(validation.summary), indent=2),
            encoding="utf-8",
        )
        validation.rows.to_csv(paths["validation_rows_csv"], index=False)
        validation_summary = _jsonable(validation.summary)
        if require_leaderboard_ready and not validation.summary.get("leaderboard_ready", False):
            reasons = ", ".join(validation.summary.get("leaderboard_blocking_reasons", []))
            raise SystemExit(f"Hampel-repaired submission is not leaderboard-ready: {reasons or 'unknown'}")
    changed = diagnostics.get("hampel_repair_applied", pd.Series(dtype=bool)).astype(bool)
    correction = pd.to_numeric(
        diagnostics.get("hampel_repair_correction_m", pd.Series(dtype=float)),
        errors="coerce",
    )
    payload = dict(manifest or {})
    payload.update(
        {
            "schema": "raft-uav-mmuad-track5-hampel-repair-v1",
            "input_submission": str(input_submission_path),
            "row_count": int(len(repaired)),
            "sequence_count": int(repaired["sequence_id"].nunique()) if not repaired.empty else 0,
            "changed_row_count": int(changed.sum()) if not diagnostics.empty else 0,
            "changed_fraction": float(changed.sum() / len(diagnostics)) if len(diagnostics) else 0.0,
            "max_correction_m": _safe_max(correction),
            "mean_correction_m": _safe_mean(correction),
            "validation": validation_summary,
            "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
        }
    )
    paths["manifest_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-hampel-repair",
        description="repair isolated local-median outliers in an official MMUAD Track 5 submission",
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, help="optional official template for preflight validation")
    parser.add_argument("--window-radius", type=int, default=2)
    parser.add_argument("--sigma-threshold", type=float, default=3.0)
    parser.add_argument("--min-scale-m", type=float, default=1.0)
    parser.add_argument("--min-residual-m", type=float, default=3.0)
    parser.add_argument("--repair-blend", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if args.require_leaderboard_ready and args.template is None:
        raise SystemExit("--require-leaderboard-ready requires --template")
    submission = load_track5_submission(args.submission)
    repaired, diagnostics = repair_track5_hampel_spikes(
        submission,
        window_radius=int(args.window_radius),
        sigma_threshold=float(args.sigma_threshold),
        min_scale_m=float(args.min_scale_m),
        min_residual_m=float(args.min_residual_m),
        repair_blend=float(args.repair_blend),
        iterations=int(args.iterations),
    )
    template = None if args.template is None else load_official_track5_template_file(args.template)
    paths = write_track5_hampel_repair_outputs(
        repaired=repaired,
        diagnostics=diagnostics,
        output_dir=args.output_dir,
        input_submission_path=args.submission,
        template=template,
        manifest={
            "window_radius": int(args.window_radius),
            "sigma_threshold": float(args.sigma_threshold),
            "min_scale_m": float(args.min_scale_m),
            "min_residual_m": float(args.min_residual_m),
            "repair_blend": float(args.repair_blend),
            "iterations": int(args.iterations),
        },
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_hampel_repair=ok")
    print(f"changed_row_count={manifest['changed_row_count']}")
    print(f"changed_fraction={manifest['changed_fraction']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _repair_sequence(
    group: pd.DataFrame,
    *,
    window_radius: int,
    sigma_threshold: float,
    min_scale_m: float,
    min_residual_m: float,
    repair_blend: float,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = group.copy().reset_index(drop=True)
    original_xyz = work[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    xyz = original_xyz.copy()
    all_diagnostics: pd.DataFrame | None = None
    for iteration in range(1, iterations + 1):
        xyz, diagnostics = _repair_xyz_once(
            work,
            xyz,
            original_xyz,
            window_radius=window_radius,
            sigma_threshold=sigma_threshold,
            min_scale_m=min_scale_m,
            min_residual_m=min_residual_m,
            repair_blend=repair_blend,
            iteration=iteration,
        )
        all_diagnostics = diagnostics
    out = work.copy()
    out["input_state_x_m"] = original_xyz[:, 0]
    out["input_state_y_m"] = original_xyz[:, 1]
    out["input_state_z_m"] = original_xyz[:, 2]
    out["state_x_m"] = xyz[:, 0]
    out["state_y_m"] = xyz[:, 1]
    out["state_z_m"] = xyz[:, 2]
    correction = np.linalg.norm(xyz - original_xyz, axis=1)
    out["hampel_repair_applied"] = correction > 1.0e-9
    out["hampel_repair_correction_m"] = correction
    diagnostics = all_diagnostics if all_diagnostics is not None else pd.DataFrame(columns=_diagnostic_columns())
    diagnostics["hampel_repair_correction_m"] = correction
    diagnostics["hampel_repair_applied"] = correction > 1.0e-9
    return out, diagnostics


def _repair_xyz_once(
    work: pd.DataFrame,
    xyz: np.ndarray,
    original_xyz: np.ndarray,
    *,
    window_radius: int,
    sigma_threshold: float,
    min_scale_m: float,
    min_residual_m: float,
    repair_blend: float,
    iteration: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    out = xyz.copy()
    diagnostics: list[dict[str, Any]] = []
    for index in range(len(xyz)):
        start = max(0, index - window_radius)
        stop = min(len(xyz), index + window_radius + 1)
        neighbor_indices = [item for item in range(start, stop) if item != index]
        neighbors = xyz[neighbor_indices]
        neighbors = neighbors[np.isfinite(neighbors).all(axis=1)]
        local_median = np.full(3, np.nan, dtype=float)
        robust_scale_m = np.nan
        residual_m = np.nan
        threshold_m = np.nan
        applied = False
        if len(neighbors) >= 2 and np.isfinite(xyz[index]).all():
            local_median = np.median(neighbors, axis=0)
            neighbor_residuals = np.linalg.norm(neighbors - local_median, axis=1)
            robust_scale_m = max(float(np.median(neighbor_residuals) * 1.4826), min_scale_m)
            residual_m = float(np.linalg.norm(xyz[index] - local_median))
            threshold_m = max(float(min_residual_m), float(sigma_threshold) * robust_scale_m)
            if residual_m > threshold_m:
                out[index] = (1.0 - repair_blend) * xyz[index] + repair_blend * local_median
                applied = True
        diagnostics.append(
            {
                "sequence_id": work.loc[index, "sequence_id"],
                "time_s": float(work.loc[index, "time_s"]),
                "iteration": int(iteration),
                "local_window_start_index": int(start),
                "local_window_stop_index": int(stop),
                "local_neighbor_count": int(len(neighbors)),
                "local_median_x_m": float(local_median[0]) if np.isfinite(local_median[0]) else np.nan,
                "local_median_y_m": float(local_median[1]) if np.isfinite(local_median[1]) else np.nan,
                "local_median_z_m": float(local_median[2]) if np.isfinite(local_median[2]) else np.nan,
                "hampel_residual_m": residual_m,
                "hampel_scale_m": robust_scale_m,
                "hampel_threshold_m": threshold_m,
                "hampel_iteration_applied": bool(applied),
                "original_state_x_m": float(original_xyz[index, 0]),
                "original_state_y_m": float(original_xyz[index, 1]),
                "original_state_z_m": float(original_xyz[index, 2]),
            }
        )
    return out, pd.DataFrame.from_records(diagnostics, columns=_diagnostic_columns())


def _normalized_submission(submission: pd.DataFrame) -> pd.DataFrame:
    rows = load_track5_submission_frame(submission)
    if rows.empty:
        return rows
    return rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def load_track5_submission_frame(submission: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(submission).copy()
    if {"sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"}.issubset(rows.columns):
        if "Classification" not in rows.columns:
            rows["Classification"] = rows.get("classification", 0)
        out = rows.copy()
    else:
        # The public loader already accepts official Track 5 CSV-style rows.
        out = rows
        if not {"sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"}.issubset(out.columns):
            # Reuse the path-based loader's normalizer indirectly by writing code
            # in terms of the same official result schema accepted by it.
            from raft_uav.mmuad.submission import normalize_official_track5_results_frame
            from raft_uav.mmuad.submission import parse_official_position_cell

            official = normalize_official_track5_results_frame(rows)
            positions = [parse_official_position_cell(value) for value in official["Position"]]
            xyz = pd.DataFrame(positions, columns=["state_x_m", "state_y_m", "state_z_m"])
            out = pd.DataFrame(
                {
                    "sequence_id": official["Sequence"].astype(str),
                    "time_s": pd.to_numeric(official["Timestamp"], errors="coerce"),
                    "state_x_m": xyz["state_x_m"],
                    "state_y_m": xyz["state_y_m"],
                    "state_z_m": xyz["state_z_m"],
                    "Classification": official["Classification"],
                }
            )
    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m", "Classification"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    finite = np.isfinite(out[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)).all(axis=1)
    return out.loc[finite].copy().reset_index(drop=True)


def _diagnostic_columns() -> list[str]:
    return [
        "sequence_id",
        "time_s",
        "iteration",
        "local_window_start_index",
        "local_window_stop_index",
        "local_neighbor_count",
        "local_median_x_m",
        "local_median_y_m",
        "local_median_z_m",
        "hampel_residual_m",
        "hampel_scale_m",
        "hampel_threshold_m",
        "hampel_iteration_applied",
        "original_state_x_m",
        "original_state_y_m",
        "original_state_z_m",
        "hampel_repair_correction_m",
        "hampel_repair_applied",
    ]


def _safe_max(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite.to_numpy(float))]
    if finite.empty:
        return None
    return float(finite.max())


def _safe_mean(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite.to_numpy(float))]
    if finite.empty:
        return None
    return float(finite.mean())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
