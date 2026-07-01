"""Inference-safe trajectory smoothing for MMUAD/UG2+ Track 5 submissions.

The Codabench metric is sensitive to moderate frame-level pose jitter after
candidate-mixture and reservoir selection.  This utility smooths an already
formed official Track 5 submission on the official ``Sequence,Timestamp`` grid
without using truth labels: for each sequence it fits a local weighted linear
model around every timestamp and optionally caps the correction from the input
trajectory.  Class labels are preserved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission

SMOOTHED_ESTIMATES_CSV = "mmuad_track5_smoothed_estimates.csv"
SMOOTHED_RESULTS_CSV = "mmaud_results_smoothed.csv"
SMOOTHED_ZIP = "ug2_submission_smoothed.zip"
DIAGNOSTICS_CSV = "mmuad_track5_trajectory_smooth_diagnostics.csv"
MANIFEST_JSON = "mmuad_track5_trajectory_smooth_manifest.json"
VALIDATION_JSON = "mmuad_track5_trajectory_smooth_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_trajectory_smooth_validation_rows.csv"


def smooth_track5_submission_rows(
    rows: pd.DataFrame,
    *,
    window_s: float = 15.0,
    bandwidth_s: float | None = None,
    blend: float = 1.0,
    max_correction_m: float | None = 10.0,
    min_neighbors: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Smooth normalized official Track 5 rows and return diagnostics.

    ``rows`` must contain normalized columns from ``load_track5_submission``:
    ``sequence_id``, ``time_s``, ``state_x_m``, ``state_y_m``, ``state_z_m``, and
    ``Classification``.  The smoother is local-linear, so it preserves constant
    velocity segments better than a moving average while attenuating isolated
    candidate-selection jitter.
    """

    if not 0.0 <= float(blend) <= 1.0:
        raise ValueError("blend must be in [0, 1]")
    if window_s <= 0.0:
        raise ValueError("window_s must be positive")
    if bandwidth_s is None:
        bandwidth_s = max(float(window_s) / 2.0, 1.0e-9)
    if bandwidth_s <= 0.0:
        raise ValueError("bandwidth_s must be positive")

    normalized = _normalized_estimate_rows(rows)
    smoothed_parts: list[pd.DataFrame] = []
    diagnostic_records: list[dict[str, Any]] = []
    for sequence_id, group in normalized.groupby("sequence_id", sort=True):
        work = group.sort_values("time_s").reset_index(drop=True)
        times = work["time_s"].to_numpy(float)
        xyz = work[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
        smoothed_xyz = np.empty_like(xyz)
        counts = np.zeros(len(work), dtype=int)
        raw_corrections = np.zeros(len(work), dtype=float)
        capped = np.zeros(len(work), dtype=bool)
        for index, time_s in enumerate(times):
            candidate, count = _local_linear_prediction(
                times,
                xyz,
                center_time=float(time_s),
                window_s=float(window_s),
                bandwidth_s=float(bandwidth_s),
                min_neighbors=int(min_neighbors),
            )
            counts[index] = count
            delta = candidate - xyz[index]
            raw_norm = float(np.linalg.norm(delta))
            raw_corrections[index] = raw_norm
            if max_correction_m is not None and raw_norm > float(max_correction_m) > 0.0:
                delta = delta * (float(max_correction_m) / raw_norm)
                capped[index] = True
            smoothed_xyz[index] = xyz[index] + float(blend) * delta
        out = work.copy()
        out["input_state_x_m"] = out["state_x_m"]
        out["input_state_y_m"] = out["state_y_m"]
        out["input_state_z_m"] = out["state_z_m"]
        out["state_x_m"] = smoothed_xyz[:, 0]
        out["state_y_m"] = smoothed_xyz[:, 1]
        out["state_z_m"] = smoothed_xyz[:, 2]
        out["trajectory_smoothed"] = True
        out["trajectory_smooth_window_s"] = float(window_s)
        out["trajectory_smooth_bandwidth_s"] = float(bandwidth_s)
        out["trajectory_smooth_blend"] = float(blend)
        out["trajectory_smooth_neighbor_count"] = counts
        out["trajectory_smooth_raw_correction_m"] = raw_corrections
        out["trajectory_smooth_capped"] = capped
        out["trajectory_smooth_applied_correction_m"] = np.linalg.norm(
            smoothed_xyz - xyz,
            axis=1,
        )
        smoothed_parts.append(out)
        diagnostic_records.extend(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(times[index]),
                "neighbor_count": int(counts[index]),
                "raw_correction_m": float(raw_corrections[index]),
                "applied_correction_m": float(out.iloc[index]["trajectory_smooth_applied_correction_m"]),
                "capped": bool(capped[index]),
                "input_speed_prev_mps": _speed_to_previous(xyz, times, index),
                "smoothed_speed_prev_mps": _speed_to_previous(smoothed_xyz, times, index),
            }
            for index in range(len(work))
        )
    smoothed = pd.concat(smoothed_parts, ignore_index=True, sort=False) if smoothed_parts else normalized
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    return smoothed.sort_values(["sequence_id", "time_s"]).reset_index(drop=True), diagnostics


def write_track5_trajectory_smooth_outputs(
    *,
    rows: pd.DataFrame,
    output_dir: Path,
    template: pd.DataFrame | None = None,
    window_s: float = 15.0,
    bandwidth_s: float | None = None,
    blend: float = 1.0,
    max_correction_m: float | None = 10.0,
    min_neighbors: int = 3,
) -> dict[str, Path]:
    """Write smoothed estimates, official CSV/ZIP, diagnostics, and manifest."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    smoothed, diagnostics = smooth_track5_submission_rows(
        rows,
        window_s=window_s,
        bandwidth_s=bandwidth_s,
        blend=blend,
        max_correction_m=max_correction_m,
        min_neighbors=min_neighbors,
    )
    paths = {
        "estimates_csv": output / SMOOTHED_ESTIMATES_CSV,
        "results_csv": output / SMOOTHED_RESULTS_CSV,
        "zip": output / SMOOTHED_ZIP,
        "diagnostics_csv": output / DIAGNOSTICS_CSV,
        "manifest_json": output / MANIFEST_JSON,
    }
    smoothed.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    official_rows = smoothed.copy()
    # Keep row-level official labels; class_map is a sequence-level override.
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
        paths["validation_json"] = output / VALIDATION_JSON
        paths["validation_rows_csv"] = output / VALIDATION_ROWS_CSV
        paths["validation_json"].write_text(
            json.dumps(_jsonable(validation.summary), indent=2),
            encoding="utf-8",
        )
        validation.rows.to_csv(paths["validation_rows_csv"], index=False)
        validation_summary = _jsonable(validation.summary)
    correction = pd.to_numeric(
        diagnostics.get("applied_correction_m", pd.Series(dtype=float)),
        errors="coerce",
    )
    payload = {
        "schema": "raft-uav-mmuad-track5-trajectory-smooth-v1",
        "row_count": int(len(smoothed)),
        "sequence_count": int(smoothed["sequence_id"].nunique()) if not smoothed.empty else 0,
        "window_s": float(window_s),
        "bandwidth_s": None if bandwidth_s is None else float(bandwidth_s),
        "blend": float(blend),
        "max_correction_m": max_correction_m,
        "min_neighbors": int(min_neighbors),
        "mean_applied_correction_m": _finite_mean(correction),
        "p95_applied_correction_m": _finite_percentile(correction, 95),
        "max_applied_correction_m": _finite_max(correction),
        "capped_rows": int(diagnostics.get("capped", pd.Series(dtype=bool)).astype(bool).sum()),
        "validation": validation_summary,
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-trajectory-smooth",
        description="smooth an official MMUAD/UG2+ Track 5 submission without truth labels",
    )
    parser.add_argument("--submission", type=Path, required=True, help="official CSV/ZIP submission")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, help="optional official template for validation")
    parser.add_argument("--window-s", type=float, default=15.0)
    parser.add_argument("--bandwidth-s", type=float)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--max-correction-m", type=float, default=10.0)
    parser.add_argument("--disable-correction-cap", action="store_true")
    parser.add_argument("--min-neighbors", type=int, default=3)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    rows = load_track5_submission(args.submission)
    template = None if args.template is None else pd.read_csv(args.template)
    paths = write_track5_trajectory_smooth_outputs(
        rows=rows,
        output_dir=args.output_dir,
        template=template,
        window_s=args.window_s,
        bandwidth_s=args.bandwidth_s,
        blend=args.blend,
        max_correction_m=None if args.disable_correction_cap else args.max_correction_m,
        min_neighbors=args.min_neighbors,
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    validation = manifest.get("validation") or {}
    print("mmuad_track5_trajectory_smooth=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    if validation:
        print(f"leaderboard_ready={validation.get('leaderboard_ready')}")
        print(f"codabench_upload_ready={validation.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and validation and not validation.get("leaderboard_ready", False):
        reasons = ", ".join(validation.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"smoothed submission is not leaderboard-ready: {reasons}")
    return 0


def _normalized_estimate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m", "Classification"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Track 5 rows missing normalized columns: {missing}")
    out = rows.copy()
    out["sequence_id"] = out["sequence_id"].astype(str)
    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    finite = np.isfinite(out[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)).all(axis=1)
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _local_linear_prediction(
    times: np.ndarray,
    xyz: np.ndarray,
    *,
    center_time: float,
    window_s: float,
    bandwidth_s: float,
    min_neighbors: int,
) -> tuple[np.ndarray, int]:
    dt = times - float(center_time)
    mask = np.abs(dt) <= float(window_s)
    count = int(mask.sum())
    original = xyz[int(np.argmin(np.abs(dt)))].astype(float)
    if count < int(min_neighbors):
        return original, count
    local_dt = dt[mask]
    local_xyz = xyz[mask]
    weights = np.exp(-0.5 * (local_dt / float(bandwidth_s)) ** 2)
    if not np.isfinite(weights).all() or float(np.sum(weights)) <= 0.0:
        return original, count
    design = np.column_stack([np.ones_like(local_dt), local_dt])
    weighted_design = design * np.sqrt(weights)[:, None]
    prediction = np.empty(3, dtype=float)
    for axis in range(3):
        target = local_xyz[:, axis] * np.sqrt(weights)
        try:
            beta, *_ = np.linalg.lstsq(weighted_design, target, rcond=None)
        except np.linalg.LinAlgError:
            return original, count
        prediction[axis] = float(beta[0])
    if not np.isfinite(prediction).all():
        return original, count
    return prediction, count


def _speed_to_previous(xyz: np.ndarray, times: np.ndarray, index: int) -> float | None:
    if index <= 0:
        return None
    dt = float(times[index] - times[index - 1])
    if dt <= 0.0 or not np.isfinite(dt):
        return None
    return float(np.linalg.norm(xyz[index] - xyz[index - 1]) / dt)


def _finite_mean(values: pd.Series) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.mean(finite))


def _finite_percentile(values: pd.Series, percentile: float) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.percentile(finite, percentile))


def _finite_max(values: pd.Series) -> float | None:
    finite = _finite_values(values)
    return None if finite.size == 0 else float(np.max(finite))


def _finite_values(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return array[np.isfinite(array)]


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
