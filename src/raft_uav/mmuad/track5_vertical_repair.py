"""Vertical-only spike repair for MMUAD/UG2+ Track 5 submissions.

Some leaderboard pipelines produce good horizontal trajectories but occasional
isolated altitude spikes after candidate assignment, calibration, or ensembling.
The generic temporal repair is intentionally conservative for full 3D outliers;
this module adds a vertical-only guard that preserves the official
Sequence/Timestamp grid, x/y positions, and Classification labels while replacing
only locally inconsistent z coordinates.

The repair is inference-safe: it uses no truth values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import load_official_track5_template_file
from raft_uav.mmuad.submission import validate_official_track5_submission
from raft_uav.mmuad.submission import write_official_mmaud_results_csv
from raft_uav.mmuad.submission import write_official_ug2_codabench_zip
from raft_uav.mmuad.track5_submission_ensemble import _jsonable
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission

REPAIRED_ESTIMATES_CSV = "mmuad_track5_vertical_repair_estimates.csv"
REPAIRED_RESULTS_CSV = "mmaud_results_vertical_repair.csv"
REPAIRED_ZIP = "ug2_submission_vertical_repair.zip"
DIAGNOSTICS_CSV = "mmuad_track5_vertical_repair_diagnostics.csv"
MANIFEST_JSON = "mmuad_track5_vertical_repair_manifest.json"
VALIDATION_JSON = "mmuad_track5_vertical_repair_validation.json"
VALIDATION_ROWS_CSV = "mmuad_track5_vertical_repair_validation_rows.csv"

_INVALID_BOOLEAN = object()
_TRUE_BOOLEAN_TOKENS = frozenset({"true", "t", "yes", "y", "on"})
_FALSE_BOOLEAN_TOKENS = frozenset({"false", "f", "no", "n", "off"})
_MISSING_BOOLEAN_TOKENS = frozenset({"", "nan", "none", "null", "<na>", "nat"})


def repair_track5_vertical_spikes(
    submission: pd.DataFrame,
    *,
    max_vertical_speed_mps: float = 20.0,
    max_neighbor_vertical_speed_mps: float = 10.0,
    max_vertical_residual_m: float = 15.0,
    max_horizontal_speed_mps: float | None = 80.0,
    iterations: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return vertically repaired estimates and per-row diagnostics.

    A point is repaired when its z value is far from linear interpolation between
    neighboring timestamps, both adjacent vertical speeds are implausibly high,
    the direct neighbor-to-neighbor vertical speed is plausible, and optional
    horizontal speeds remain plausible.  Only ``state_z_m`` is changed.
    """

    rows = _normalize_rows(submission)
    if rows.empty:
        return rows, pd.DataFrame(columns=_diagnostic_columns())
    repaired_parts: list[pd.DataFrame] = []
    diagnostics_parts: list[pd.DataFrame] = []
    for _, group in rows.groupby("sequence_id", sort=True):
        repaired, diagnostics = _repair_sequence(
            group.sort_values("time_s").reset_index(drop=True),
            max_vertical_speed_mps=float(max_vertical_speed_mps),
            max_neighbor_vertical_speed_mps=float(max_neighbor_vertical_speed_mps),
            max_vertical_residual_m=float(max_vertical_residual_m),
            max_horizontal_speed_mps=max_horizontal_speed_mps,
            iterations=int(iterations),
        )
        repaired_parts.append(repaired)
        diagnostics_parts.append(diagnostics)
    repaired_all = pd.concat(repaired_parts, ignore_index=True, sort=False)
    diagnostics_all = pd.concat(diagnostics_parts, ignore_index=True, sort=False)
    return (
        repaired_all.sort_values(["sequence_id", "time_s"]).reset_index(drop=True),
        diagnostics_all,
    )


def write_track5_vertical_repair_outputs(
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

    normalized_diagnostics = pd.DataFrame(diagnostics).copy()
    if "repaired" in normalized_diagnostics.columns:
        normalized_diagnostics["repaired"] = _normalized_boolean_column(
            normalized_diagnostics,
            "repaired",
        )
    diagnostics = normalized_diagnostics

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
        paths["validation_json"] = output / VALIDATION_JSON
        paths["validation_rows_csv"] = output / VALIDATION_ROWS_CSV
        paths["validation_json"].write_text(
            json.dumps(_jsonable(validation.summary), indent=2),
            encoding="utf-8",
        )
        validation.rows.to_csv(paths["validation_rows_csv"], index=False)
        validation_summary = _jsonable(validation.summary)
        if require_leaderboard_ready and not validation.summary.get(
            "leaderboard_ready",
            False,
        ):
            reasons = ", ".join(validation.summary.get("leaderboard_blocking_reasons", []))
            raise SystemExit(
                f"vertical-repaired submission is not leaderboard-ready: {reasons or 'unknown'}"
            )
    repaired_count = int(diagnostics["repaired"].sum()) if not diagnostics.empty else 0
    payload = dict(manifest or {})
    payload.update(
        {
            "schema": "raft-uav-mmuad-track5-vertical-repair-v1",
            "input_submission": str(input_submission_path),
            "row_count": int(len(repaired)),
            "sequence_count": int(repaired["sequence_id"].nunique())
            if not repaired.empty
            else 0,
            "repaired_row_count": repaired_count,
            "repaired_fraction": float(repaired_count / len(repaired))
            if len(repaired)
            else 0.0,
            "max_abs_vertical_repair_m": _safe_abs_max(
                diagnostics.get("vertical_repair_m", pd.Series(dtype=float))
            ),
            "validation": validation_summary,
            "paths": {
                name: str(path) for name, path in paths.items() if name != "manifest_json"
            },
        }
    )
    paths["manifest_json"].write_text(
        json.dumps(_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-vertical-repair",
        description="repair isolated altitude spikes in an official MMUAD Track 5 submission",
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        help="optional official template for preflight validation",
    )
    parser.add_argument("--max-vertical-speed-mps", type=float, default=20.0)
    parser.add_argument("--max-neighbor-vertical-speed-mps", type=float, default=10.0)
    parser.add_argument("--max-vertical-residual-m", type=float, default=15.0)
    parser.add_argument("--max-horizontal-speed-mps", type=float, default=80.0)
    parser.add_argument("--disable-horizontal-gate", action="store_true")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    submission = load_track5_submission(args.submission)
    repaired, diagnostics = repair_track5_vertical_spikes(
        submission,
        max_vertical_speed_mps=float(args.max_vertical_speed_mps),
        max_neighbor_vertical_speed_mps=float(args.max_neighbor_vertical_speed_mps),
        max_vertical_residual_m=float(args.max_vertical_residual_m),
        max_horizontal_speed_mps=None
        if args.disable_horizontal_gate
        else float(args.max_horizontal_speed_mps),
        iterations=int(args.iterations),
    )
    template = (
        None
        if args.template is None
        else load_official_track5_template_file(args.template)
    )
    paths = write_track5_vertical_repair_outputs(
        repaired=repaired,
        diagnostics=diagnostics,
        output_dir=args.output_dir,
        input_submission_path=args.submission,
        template=template,
        manifest={
            "max_vertical_speed_mps": float(args.max_vertical_speed_mps),
            "max_neighbor_vertical_speed_mps": float(args.max_neighbor_vertical_speed_mps),
            "max_vertical_residual_m": float(args.max_vertical_residual_m),
            "max_horizontal_speed_mps": None
            if args.disable_horizontal_gate
            else float(args.max_horizontal_speed_mps),
            "iterations": int(args.iterations),
        },
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    print("mmuad_track5_vertical_repair=ok")
    print(f"repaired_row_count={manifest['repaired_row_count']}")
    print(f"repaired_fraction={manifest['repaired_fraction']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _normalize_rows(submission: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(submission).copy()
    required = {
        "sequence_id",
        "time_s",
        "state_x_m",
        "state_y_m",
        "state_z_m",
        "Classification",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"submission missing normalized columns: {sorted(missing)}")
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "state_x_m", "state_y_m", "state_z_m", "Classification"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(
        rows[["time_s", "state_x_m", "state_y_m", "state_z_m", "Classification"]]
        .to_numpy(float)
    ).all(axis=1)
    return rows.loc[finite].copy().sort_values(["sequence_id", "time_s"]).reset_index(
        drop=True
    )


def _repair_sequence(
    group: pd.DataFrame,
    *,
    max_vertical_speed_mps: float,
    max_neighbor_vertical_speed_mps: float,
    max_vertical_residual_m: float,
    max_horizontal_speed_mps: float | None,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repaired = group.copy()
    repaired["vertical_repair_applied"] = False
    diagnostics_records: list[dict[str, Any]] = []
    for iteration in range(max(1, int(iterations))):
        changed = False
        for idx in range(1, len(repaired) - 1):
            prev = repaired.iloc[idx - 1]
            row = repaired.iloc[idx]
            nxt = repaired.iloc[idx + 1]
            decision = _repair_decision(
                prev,
                row,
                nxt,
                max_vertical_speed_mps=max_vertical_speed_mps,
                max_neighbor_vertical_speed_mps=max_neighbor_vertical_speed_mps,
                max_vertical_residual_m=max_vertical_residual_m,
                max_horizontal_speed_mps=max_horizontal_speed_mps,
            )
            repaired_flag = False
            old_z = float(row["state_z_m"])
            new_z = old_z
            if decision["should_repair"]:
                new_z = float(decision["interpolated_z_m"])
                repaired.at[idx, "state_z_m"] = new_z
                repaired.at[idx, "vertical_repair_applied"] = True
                changed = True
                repaired_flag = True
            diagnostics_records.append(
                {
                    "sequence_id": str(row["sequence_id"]),
                    "time_s": float(row["time_s"]),
                    "iteration": iteration + 1,
                    "repaired": repaired_flag,
                    "z_original_m": old_z,
                    "z_repaired_m": new_z,
                    "vertical_repair_m": float(new_z - old_z),
                    **decision,
                }
            )
        if not changed:
            break
    diagnostics = pd.DataFrame.from_records(
        diagnostics_records,
        columns=_diagnostic_columns(),
    )
    return repaired, diagnostics


def _repair_decision(
    prev: pd.Series,
    row: pd.Series,
    nxt: pd.Series,
    *,
    max_vertical_speed_mps: float,
    max_neighbor_vertical_speed_mps: float,
    max_vertical_residual_m: float,
    max_horizontal_speed_mps: float | None,
) -> dict[str, Any]:
    t0 = float(prev["time_s"])
    t1 = float(row["time_s"])
    t2 = float(nxt["time_s"])
    if not (t0 < t1 < t2):
        return _decision(False, "non_monotone_time")
    left_dt = max(t1 - t0, 1.0e-9)
    right_dt = max(t2 - t1, 1.0e-9)
    span_dt = max(t2 - t0, 1.0e-9)
    alpha = (t1 - t0) / span_dt
    interpolated_z = (
        (1.0 - alpha) * float(prev["state_z_m"])
        + alpha * float(nxt["state_z_m"])
    )
    residual = float(row["state_z_m"]) - interpolated_z
    left_vz = (float(row["state_z_m"]) - float(prev["state_z_m"])) / left_dt
    right_vz = (float(nxt["state_z_m"]) - float(row["state_z_m"])) / right_dt
    neighbor_vz = (float(nxt["state_z_m"]) - float(prev["state_z_m"])) / span_dt
    left_h = _horizontal_speed(prev, row, left_dt)
    right_h = _horizontal_speed(row, nxt, right_dt)
    horizontal_ok = True
    if max_horizontal_speed_mps is not None:
        horizontal_ok = (
            left_h <= float(max_horizontal_speed_mps)
            and right_h <= float(max_horizontal_speed_mps)
        )
    checks = {
        "interpolated_z_m": float(interpolated_z),
        "vertical_residual_m": float(residual),
        "abs_vertical_residual_m": abs(float(residual)),
        "left_vertical_speed_mps": float(left_vz),
        "right_vertical_speed_mps": float(right_vz),
        "neighbor_vertical_speed_mps": float(neighbor_vz),
        "left_horizontal_speed_mps": float(left_h),
        "right_horizontal_speed_mps": float(right_h),
        "horizontal_gate_ok": bool(horizontal_ok),
    }
    should_repair = (
        abs(residual) >= float(max_vertical_residual_m)
        and abs(left_vz) >= float(max_vertical_speed_mps)
        and abs(right_vz) >= float(max_vertical_speed_mps)
        and abs(neighbor_vz) <= float(max_neighbor_vertical_speed_mps)
        and horizontal_ok
    )
    reason = "repair" if should_repair else "kept"
    return {"should_repair": bool(should_repair), "repair_reason": reason, **checks}


def _decision(should_repair: bool, reason: str) -> dict[str, Any]:
    return {
        "should_repair": bool(should_repair),
        "repair_reason": reason,
        "interpolated_z_m": np.nan,
        "vertical_residual_m": np.nan,
        "abs_vertical_residual_m": np.nan,
        "left_vertical_speed_mps": np.nan,
        "right_vertical_speed_mps": np.nan,
        "neighbor_vertical_speed_mps": np.nan,
        "left_horizontal_speed_mps": np.nan,
        "right_horizontal_speed_mps": np.nan,
        "horizontal_gate_ok": False,
    }


def _horizontal_speed(a: pd.Series, b: pd.Series, dt: float) -> float:
    dx = float(b["state_x_m"]) - float(a["state_x_m"])
    dy = float(b["state_y_m"]) - float(a["state_y_m"])
    return float(np.hypot(dx, dy) / max(float(dt), 1.0e-9))


def _boolean_number(value: float) -> bool | object:
    """Parse one finite numeric Boolean representation."""

    if np.isnan(value):
        return False
    if not np.isfinite(value):
        return _INVALID_BOOLEAN
    if value == 0.0:
        return False
    if value == 1.0:
        return True
    return _INVALID_BOOLEAN


def _boolean_cell(value: object) -> bool | object:
    """Parse one persisted diagnostic flag without generic truthiness."""

    if value is None or value is pd.NA or np.ma.is_masked(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in _TRUE_BOOLEAN_TOKENS:
            return True
        if text in _FALSE_BOOLEAN_TOKENS or text in _MISSING_BOOLEAN_TOKENS:
            return False
        try:
            return _boolean_number(float(text))
        except (TypeError, ValueError, OverflowError):
            return _INVALID_BOOLEAN

    try:
        array = np.asanyarray(value)
    except (TypeError, ValueError):
        return _INVALID_BOOLEAN
    if array.ndim != 0 or np.iscomplexobj(array):
        return _INVALID_BOOLEAN
    if np.ma.isMaskedArray(array) and bool(np.ma.getmaskarray(array).any()):
        return False

    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    try:
        if bool(pd.isna(scalar)):
            return False
    except (TypeError, ValueError):
        return _INVALID_BOOLEAN
    try:
        return _boolean_number(float(scalar))
    except (TypeError, ValueError, OverflowError):
        return _INVALID_BOOLEAN


def _normalized_boolean_column(rows: pd.DataFrame, column: str) -> pd.Series:
    """Return strict Boolean diagnostics while preserving the row index."""

    values = pd.Series(rows[column], index=rows.index, copy=False)
    normalized: list[bool] = []
    invalid: list[tuple[object, object]] = []
    for index, value in values.items():
        parsed = _boolean_cell(value)
        if parsed is _INVALID_BOOLEAN:
            invalid.append((index, value))
            normalized.append(False)
        else:
            normalized.append(bool(parsed))

    if invalid:
        preview = ", ".join(
            f"index {index!r}: {value!r}" for index, value in invalid[:5]
        )
        suffix = "" if len(invalid) <= 5 else f"; plus {len(invalid) - 5} more"
        raise ValueError(f"{column} contains invalid Boolean values: {preview}{suffix}")
    return pd.Series(
        normalized,
        index=values.index,
        name=values.name,
        dtype=bool,
    )


def _safe_abs_max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(float))]
    if numeric.empty:
        return 0.0
    return float(np.max(np.abs(numeric.to_numpy(float))))


def _diagnostic_columns() -> list[str]:
    return [
        "sequence_id",
        "time_s",
        "iteration",
        "repaired",
        "z_original_m",
        "z_repaired_m",
        "vertical_repair_m",
        "should_repair",
        "repair_reason",
        "interpolated_z_m",
        "vertical_residual_m",
        "abs_vertical_residual_m",
        "left_vertical_speed_mps",
        "right_vertical_speed_mps",
        "neighbor_vertical_speed_mps",
        "left_horizontal_speed_mps",
        "right_horizontal_speed_mps",
        "horizontal_gate_ok",
    ]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
