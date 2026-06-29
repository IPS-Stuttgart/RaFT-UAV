"""Temporal spike repair for official MMUAD/UG2+ Track 5 submissions.

Leaderboard pose pipelines may produce occasional isolated position spikes after
candidate-mixture, reservoir, or ensemble steps.  This module repairs only
locally inconsistent interior points in an official Track 5 submission by
interpolating between neighboring timestamps when both incoming/outgoing speeds
are implausible but the direct neighbor-to-neighbor motion is plausible.

The procedure is inference-safe: it uses no truth values and preserves the
Sequence/Timestamp template and Classification labels.
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
from raft_uav.mmuad.track5_submission_ensemble import _jsonable
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission

REPAIRED_ESTIMATES_CSV = "mmuad_track5_temporal_repair_estimates.csv"
REPAIRED_RESULTS_CSV = "mmaud_results_temporal_repair.csv"
REPAIRED_ZIP = "ug2_submission_temporal_repair.zip"
DIAGNOSTICS_CSV = "mmuad_track5_temporal_repair_diagnostics.csv"
MANIFEST_JSON = "mmuad_track5_temporal_repair_manifest.json"
VALIDATION_JSON = "mmuad_track5_temporal_repair_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_temporal_repair_validation_rows.csv"


def repair_track5_temporal_spikes(
    submission: pd.DataFrame,
    *,
    max_speed_mps: float = 80.0,
    max_interpolation_residual_m: float = 25.0,
    iterations: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return repaired estimates and row-level diagnostics.

    A row is repaired when it is an interior point whose position is far from the
    linear interpolation of its immediate neighbors, both adjacent speeds exceed
    ``max_speed_mps``, and the direct neighbor-to-neighbor speed is still within
    the speed gate.  Multiple iterations handle short cascades while keeping the
    rule conservative.
    """

    rows = _normalized_submission(submission)
    if rows.empty:
        return rows, pd.DataFrame(columns=_diagnostic_columns())
    repaired_groups: list[pd.DataFrame] = []
    diagnostics_groups: list[pd.DataFrame] = []
    for _, group in rows.groupby("sequence_id", sort=True):
        repaired, diagnostics = _repair_sequence(
            group.sort_values("time_s").reset_index(drop=True),
            max_speed_mps=float(max_speed_mps),
            max_interpolation_residual_m=float(max_interpolation_residual_m),
            iterations=int(iterations),
        )
        repaired_groups.append(repaired)
        diagnostics_groups.append(diagnostics)
    repaired_rows = pd.concat(repaired_groups, ignore_index=True, sort=False)
    diagnostics_rows = pd.concat(diagnostics_groups, ignore_index=True, sort=False)
    return repaired_rows.sort_values(["sequence_id", "time_s"]).reset_index(drop=True), diagnostics_rows


def write_track5_temporal_repair_outputs(
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
        "estimates_csv": output / REPAIRED_ESTIMATES_CSV,
        "results_csv": output / REPAIRED_RESULTS_CSV,
        "zip": output / REPAIRED_ZIP,
        "diagnostics_csv": output / DIAGNOSTICS_CSV,
        "manifest_json": output / MANIFEST_JSON,
    }
    repaired.to_csv(paths["estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    class_map = {
        str(row.sequence_id): int(row.Classification)
        for row in repaired.drop_duplicates("sequence_id").itertuples(index=False)
    }
    write_official_mmaud_results_csv(
        repaired,
        paths["results_csv"],
        classification=0,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        repaired,
        paths["zip"],
        classification=0,
        class_map=class_map,
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
        if require_leaderboard_ready and not validation.summary.get("leaderboard_ready", False):
            reasons = ", ".join(validation.summary.get("leaderboard_blocking_reasons", [])) or "unknown"
            raise SystemExit(f"temporal-repaired submission is not leaderboard-ready: {reasons}")
    payload = dict(manifest or {})
    repaired_count = int(diagnostics["repaired"].astype(bool).sum()) if not diagnostics.empty else 0
    payload.update(
        {
            "schema": "raft-uav-mmuad-track5-temporal-repair-v1",
            "input_submission": str(input_submission_path),
            "row_count": int(len(repaired)),
            "sequence_count": int(repaired["sequence_id"].nunique()) if not repaired.empty else 0,
            "repaired_row_count": repaired_count,
            "repaired_fraction": float(repaired_count / len(repaired)) if len(repaired) else 0.0,
            "max_applied_repair_m": float(diagnostics["repair_displacement_m"].max())
            if "repair_displacement_m" in diagnostics and not diagnostics.empty
            else 0.0,
            "validation": validation_summary,
            "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
        }
    )
    paths["manifest_json"].write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-temporal-repair",
        description="repair isolated temporal spikes in an official MMUAD Track 5 submission",
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, help="optional official template for preflight validation")
    parser.add_argument("--max-speed-mps", type=float, default=80.0)
    parser.add_argument("--max-interpolation-residual-m", type=float, default=25.0)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    submission = load_track5_submission(args.submission)
    repaired, diagnostics = repair_track5_temporal_spikes(
        submission,
        max_speed_mps=float(args.max_speed_mps),
        max_interpolation_residual_m=float(args.max_interpolation_residual_m),
        iterations=int(args.iterations),
    )
    template = None if args.template is None else pd.read_csv(args.template)
    paths = write_track5_temporal_repair_outputs(
        repaired=repaired,
        diagnostics=diagnostics,
        output_dir=args.output_dir,
        input_submission_path=args.submission,
        template=template,
        manifest={
            "max_speed_mps": float(args.max_speed_mps),
            "max_interpolation_residual_m": float(args.max_interpolation_residual_m),
            "iterations": int(args.iterations),
        },
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_temporal_repair=ok")
    print(f"repaired_row_count={manifest['repaired_row_count']}")
    print(f"repaired_fraction={manifest['repaired_fraction']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _normalized_submission(submission: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(submission).copy()
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
    max_speed_mps: float,
    max_interpolation_residual_m: float,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = group.copy().reset_index(drop=True)
    work["temporal_repair_applied"] = False
    work["temporal_repair_iteration"] = 0
    work["temporal_repair_displacement_m"] = 0.0
    diagnostics: pd.DataFrame | None = None
    for iteration in range(1, max(1, int(iterations)) + 1):
        diagnostics = _sequence_diagnostics(work, iteration=iteration)
        repair_mask = diagnostics["repair_candidate"].to_numpy(bool)
        if not repair_mask.any():
            break
        xyz = work[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
        for idx in np.flatnonzero(repair_mask):
            interpolated = diagnostics.loc[idx, ["interp_x_m", "interp_y_m", "interp_z_m"]].to_numpy(float)
            displacement = float(np.linalg.norm(xyz[idx] - interpolated))
            xyz[idx] = interpolated
            work.loc[idx, ["state_x_m", "state_y_m", "state_z_m"]] = interpolated
            work.loc[idx, "temporal_repair_applied"] = True
            work.loc[idx, "temporal_repair_iteration"] = iteration
            work.loc[idx, "temporal_repair_displacement_m"] = displacement
    final_diagnostics = _sequence_diagnostics(work, iteration=max(1, int(iterations)) + 1)
    final_diagnostics["repaired"] = work["temporal_repair_applied"].to_numpy(bool)
    final_diagnostics["repair_iteration"] = work["temporal_repair_iteration"].to_numpy(int)
    final_diagnostics["repair_displacement_m"] = work["temporal_repair_displacement_m"].to_numpy(float)
    final_diagnostics["max_speed_mps"] = float(max_speed_mps)
    final_diagnostics["max_interpolation_residual_m"] = float(max_interpolation_residual_m)
    return work, final_diagnostics


def _sequence_diagnostics(group: pd.DataFrame, *, iteration: int) -> pd.DataFrame:
    n = len(group)
    times = group["time_s"].to_numpy(float)
    xyz = group[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    records: list[dict[str, Any]] = []
    for i in range(n):
        record = {
            "sequence_id": str(group.loc[i, "sequence_id"]),
            "time_s": float(times[i]),
            "iteration": int(iteration),
            "incoming_speed_mps": np.nan,
            "outgoing_speed_mps": np.nan,
            "neighbor_direct_speed_mps": np.nan,
            "interpolation_residual_m": np.nan,
            "interp_x_m": xyz[i, 0],
            "interp_y_m": xyz[i, 1],
            "interp_z_m": xyz[i, 2],
            "repair_candidate": False,
        }
        if 0 < i < n - 1:
            dt_in = times[i] - times[i - 1]
            dt_out = times[i + 1] - times[i]
            dt_direct = times[i + 1] - times[i - 1]
            if dt_in > 0.0 and dt_out > 0.0 and dt_direct > 0.0:
                alpha = dt_in / dt_direct
                interpolated = xyz[i - 1] + alpha * (xyz[i + 1] - xyz[i - 1])
                incoming_speed = float(np.linalg.norm(xyz[i] - xyz[i - 1]) / dt_in)
                outgoing_speed = float(np.linalg.norm(xyz[i + 1] - xyz[i]) / dt_out)
                direct_speed = float(np.linalg.norm(xyz[i + 1] - xyz[i - 1]) / dt_direct)
                residual = float(np.linalg.norm(xyz[i] - interpolated))
                record.update(
                    {
                        "incoming_speed_mps": incoming_speed,
                        "outgoing_speed_mps": outgoing_speed,
                        "neighbor_direct_speed_mps": direct_speed,
                        "interpolation_residual_m": residual,
                        "interp_x_m": float(interpolated[0]),
                        "interp_y_m": float(interpolated[1]),
                        "interp_z_m": float(interpolated[2]),
                        "repair_candidate": False,
                    }
                )
        records.append(record)
    diagnostics = pd.DataFrame.from_records(records)
    return diagnostics


def _diagnostic_columns() -> list[str]:
    return [
        "sequence_id",
        "time_s",
        "iteration",
        "incoming_speed_mps",
        "outgoing_speed_mps",
        "neighbor_direct_speed_mps",
        "interpolation_residual_m",
        "interp_x_m",
        "interp_y_m",
        "interp_z_m",
        "repair_candidate",
        "repaired",
        "repair_iteration",
        "repair_displacement_m",
        "max_speed_mps",
        "max_interpolation_residual_m",
    ]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
