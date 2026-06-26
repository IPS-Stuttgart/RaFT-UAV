"""Resample MMUAD estimates to an official Track 5 timestamp template.

Candidate-mixture and tracker outputs often live on sensor/candidate timestamps,
whereas Codabench submissions must contain exactly the requested
``Sequence,Timestamp`` rows.  This helper interpolates one trajectory per
sequence onto an official template, writes upload-ready Track 5 artifacts, and
runs the same local preflight validator used by the scorecard.
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
    load_sequence_class_map,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)

COORDINATE_COLUMN_SETS = (
    ("state_x_m", "state_y_m", "state_z_m"),
    ("x_m", "y_m", "z_m"),
    ("x", "y", "z"),
    ("east_m", "north_m", "up_m"),
)
SEQUENCE_ALIASES = ("sequence_id", "Sequence", "sequence", "seq")
TIME_ALIASES = ("time_s", "Timestamp", "timestamp", "timestamp_s", "time")


def resample_estimates_to_track5_template(
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Interpolate estimate coordinates onto official template timestamps.

    Returns ``(resampled_estimates, diagnostics)``.  Interpolation uses endpoint
    hold outside each sequence's estimate time span, matching the pragmatic
    leaderboard-submission need to produce one prediction for every requested
    timestamp while making extrapolated rows auditable through diagnostics.
    """

    estimate_rows = _normalize_estimate_rows(estimates)
    template_rows = _normalize_template_rows(template)
    if template_rows.empty:
        return _empty_resampled(), _empty_diagnostics()

    estimate_by_sequence = {
        sequence_id: group.sort_values("time_s").reset_index(drop=True)
        for sequence_id, group in estimate_rows.groupby("sequence_id", sort=True)
    }
    resampled_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    for _, template_row in template_rows.iterrows():
        sequence_id = str(template_row["sequence_id"])
        time_s = float(template_row["time_s"])
        group = estimate_by_sequence.get(sequence_id)
        if group is None or group.empty:
            record = _missing_estimate_record(sequence_id, time_s)
            resampled_records.append(record)
            diagnostic_records.append(
                _diagnostic_record(
                    sequence_id=sequence_id,
                    time_s=time_s,
                    nearest_time_delta_s=np.nan,
                    extrapolated=True,
                    source_row_count=0,
                    valid=False,
                )
            )
            continue
        interp, nearest_delta, extrapolated = _interpolated_position(group, time_s)
        valid = np.isfinite(interp).all()
        if max_nearest_time_delta_s is not None and np.isfinite(nearest_delta):
            valid = valid and abs(float(nearest_delta)) <= float(max_nearest_time_delta_s)
        resampled_records.append(
            {
                "sequence_id": sequence_id,
                "time_s": time_s,
                "state_x_m": float(interp[0]) if np.isfinite(interp[0]) else np.nan,
                "state_y_m": float(interp[1]) if np.isfinite(interp[1]) else np.nan,
                "state_z_m": float(interp[2]) if np.isfinite(interp[2]) else np.nan,
                "template_resampled": True,
                "template_nearest_time_delta_s": float(nearest_delta),
                "template_extrapolated": bool(extrapolated),
                "template_resample_valid": bool(valid),
            }
        )
        diagnostic_records.append(
            _diagnostic_record(
                sequence_id=sequence_id,
                time_s=time_s,
                nearest_time_delta_s=nearest_delta,
                extrapolated=extrapolated,
                source_row_count=len(group),
                valid=valid,
            )
        )
    resampled = pd.DataFrame.from_records(resampled_records)
    diagnostics = pd.DataFrame.from_records(diagnostic_records)
    return resampled, diagnostics


def write_track5_template_resample_outputs(
    *,
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
) -> dict[str, Path]:
    """Write resampled estimates, official CSV/ZIP, diagnostics, and validation."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resampled, diagnostics = resample_estimates_to_track5_template(
        estimates,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
    )
    paths = {
        "resampled_estimates_csv": output / "mmuad_template_resampled_estimates.csv",
        "diagnostics_csv": output / "mmuad_template_resample_diagnostics.csv",
        "official_results_csv": output / "mmaud_results.csv",
        "official_zip": output / "ug2_submission.zip",
        "validation_json": output / "mmuad_template_resample_validation.json",
        "validation_rows_csv": output / "mmuad_template_resample_validation_rows.csv",
        "manifest_json": output / "mmuad_template_resample_manifest.json",
    }
    resampled.to_csv(paths["resampled_estimates_csv"], index=False)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
    write_official_mmaud_results_csv(
        resampled,
        paths["official_results_csv"],
        classification=default_classification,
        class_map=class_map,
        invalid_row_policy="raise",
    )
    write_official_ug2_codabench_zip(
        resampled,
        paths["official_zip"],
        classification=default_classification,
        class_map=class_map,
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
        "schema": "raft-uav-mmuad-track5-template-resample-v1",
        "row_count": int(len(resampled)),
        "template_row_count": int(len(_normalize_template_rows(template))),
        "valid_resampled_rows": int(resampled.get("template_resample_valid", pd.Series(dtype=bool)).sum()),
        "extrapolated_rows": int(resampled.get("template_extrapolated", pd.Series(dtype=bool)).sum()),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "default_classification": str(default_classification),
        "max_nearest_time_delta_s": max_nearest_time_delta_s,
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-template-resample",
        description="interpolate MMUAD estimates onto a Track 5 timestamp template",
    )
    parser.add_argument("--estimates-csv", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="0")
    parser.add_argument("--max-nearest-time-delta-s", type=float)
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    estimates = pd.read_csv(args.estimates_csv)
    template = load_official_track5_template_file(args.template)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_track5_template_resample_outputs(
        estimates=estimates,
        template=template,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        max_nearest_time_delta_s=args.max_nearest_time_delta_s,
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_template_resample=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"template-resampled upload is not leaderboard-ready: {reasons}")
    return 0


def _normalize_estimate_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(estimates).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"])
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    time_column = _first_present(rows, TIME_ALIASES)
    coord_columns = _coordinate_columns(rows)
    if sequence_column is None or time_column is None:
        raise ValueError("estimates must contain sequence and time columns")
    out = pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].astype(str),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
            "state_x_m": pd.to_numeric(rows[coord_columns[0]], errors="coerce"),
            "state_y_m": pd.to_numeric(rows[coord_columns[1]], errors="coerce"),
            "state_z_m": pd.to_numeric(rows[coord_columns[2]], errors="coerce"),
        }
    )
    finite = np.isfinite(out[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)).all(axis=1)
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    rows = pd.DataFrame(template).copy()
    if rows.empty:
        return pd.DataFrame(columns=["sequence_id", "time_s"])
    sequence_column = _first_present(rows, SEQUENCE_ALIASES)
    time_column = _first_present(rows, TIME_ALIASES)
    if sequence_column is None or time_column is None:
        raise ValueError("template must contain sequence and timestamp columns")
    out = pd.DataFrame(
        {
            "sequence_id": rows[sequence_column].astype(str),
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
        }
    )
    finite = np.isfinite(out["time_s"].to_numpy(float))
    return out.loc[finite].sort_values(["sequence_id", "time_s"]).reset_index(drop=True)


def _coordinate_columns(rows: pd.DataFrame) -> tuple[str, str, str]:
    for columns in COORDINATE_COLUMN_SETS:
        if all(column in rows.columns for column in columns):
            return columns
    expected = " or ".join(",".join(columns) for columns in COORDINATE_COLUMN_SETS)
    raise ValueError(f"estimates must contain coordinate columns: {expected}")


def _interpolated_position(group: pd.DataFrame, time_s: float) -> tuple[np.ndarray, float, bool]:
    work = group.sort_values("time_s").drop_duplicates("time_s", keep="last")
    times = work["time_s"].to_numpy(float)
    xyz = work[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    nearest_idx = int(np.argmin(np.abs(times - float(time_s))))
    nearest_delta = float(float(time_s) - times[nearest_idx])
    extrapolated = bool(float(time_s) < times[0] or float(time_s) > times[-1])
    if len(times) == 1:
        return xyz[0].astype(float), nearest_delta, extrapolated
    interpolated = np.asarray([np.interp(float(time_s), times, xyz[:, axis]) for axis in range(3)])
    return interpolated, nearest_delta, extrapolated


def _missing_estimate_record(sequence_id: str, time_s: float) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "time_s": time_s,
        "state_x_m": np.nan,
        "state_y_m": np.nan,
        "state_z_m": np.nan,
        "template_resampled": True,
        "template_nearest_time_delta_s": np.nan,
        "template_extrapolated": True,
        "template_resample_valid": False,
    }


def _diagnostic_record(
    *,
    sequence_id: str,
    time_s: float,
    nearest_time_delta_s: float,
    extrapolated: bool,
    source_row_count: int,
    valid: bool,
) -> dict[str, Any]:
    return {
        "sequence_id": sequence_id,
        "time_s": time_s,
        "nearest_time_delta_s": nearest_time_delta_s,
        "abs_nearest_time_delta_s": abs(float(nearest_time_delta_s)) if np.isfinite(nearest_time_delta_s) else np.nan,
        "extrapolated": bool(extrapolated),
        "source_row_count": int(source_row_count),
        "valid": bool(valid),
    }


def _empty_resampled() -> pd.DataFrame:
    return pd.DataFrame(columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"])


def _empty_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(columns=["sequence_id", "time_s", "nearest_time_delta_s", "extrapolated", "source_row_count", "valid"])


def _first_present(rows: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {str(column).lower(): str(column) for column in rows.columns}
    for candidate in candidates:
        if candidate in rows.columns:
            return candidate
        found = lower.get(candidate.lower())
        if found is not None:
            return found
    return None


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
