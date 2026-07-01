#!/usr/bin/env python
"""Snap an official MMUAD Track 5 results table to a template grid.

Codabench submissions for the CVPR UG2+/MMUAD Track 5 task must contain exactly
one ``mmaud_results.csv`` row for each requested ``Sequence``/``Timestamp``.  A
tracker or mixture-MAP run can emit extra sensor-time rows, so this helper is a
truth-free final packaging step: it resamples an official-style result CSV/ZIP to
an official template, preserves sequence-level classifications, and writes an
upload-ready ZIP plus local preflight diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Literal
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.submission import (  # noqa: E402
    OFFICIAL_UG2_RESULT_COLUMNS,
    load_official_track5_results_frame,
    load_official_track5_template_file,
    parse_official_position_cell,
    validate_official_track5_submission,
)

RESAMPLE_METHODS = ("linear", "nearest")
CLASSIFICATION_POLICIES = ("sequence-mode", "nearest")
MISSING_POSITION_POLICIES = ("zero", "raise")
ResampleMethod = Literal["linear", "nearest"]
ClassificationPolicy = Literal["sequence-mode", "nearest"]
MissingPositionPolicy = Literal["zero", "raise"]

RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
DIAGNOSTICS_CSV = "mmuad_template_snap_diagnostics.csv"
VALIDATION_JSON = "mmuad_template_snap_validation.json"
VALIDATION_ROWS_CSV = "mmuad_template_snap_validation_rows.csv"
MANIFEST_JSON = "mmuad_template_snap_manifest.json"
DIAGNOSTIC_COLUMNS = (
    "template_row_index",
    "Sequence",
    "Timestamp",
    "source_row_count",
    "nearest_time_delta_s",
    "abs_nearest_time_delta_s",
    "extrapolated",
    "method",
    "interpolation_gap_s",
    "large_gap_fallback",
    "classification_policy",
    "valid",
)


def snap_official_results_to_template(
    results: pd.DataFrame,
    template: pd.DataFrame,
    *,
    resample_method: ResampleMethod = "linear",
    max_interpolation_gap_s: float | None = None,
    classification_policy: ClassificationPolicy = "sequence-mode",
    missing_position_policy: MissingPositionPolicy = "zero",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return official rows snapped to a template and per-row diagnostics."""

    method = _normalize_choice(resample_method, RESAMPLE_METHODS, "resample_method")
    class_policy = _normalize_choice(
        classification_policy,
        CLASSIFICATION_POLICIES,
        "classification_policy",
    )
    missing_policy = _normalize_choice(
        missing_position_policy,
        MISSING_POSITION_POLICIES,
        "missing_position_policy",
    )
    result_rows = _normalize_results_rows(results)
    template_rows = _normalize_template_rows(template)
    result_by_sequence = {
        sequence_id: group.sort_values("Timestamp").reset_index(drop=True)
        for sequence_id, group in result_rows.groupby("Sequence", sort=True)
    }

    output_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []
    for template_index, template_row in template_rows.iterrows():
        sequence_id = str(template_row["Sequence"])
        timestamp = float(template_row["Timestamp"])
        sequence_results = result_by_sequence.get(sequence_id)
        if sequence_results is None or sequence_results.empty:
            if missing_policy == "raise":
                raise ValueError(f"no source results for template sequence {sequence_id!r}")
            position = np.zeros(3, dtype=float)
            classification = 0
            diagnostic = _diagnostic_record(
                template_index=template_index,
                sequence_id=sequence_id,
                timestamp=timestamp,
                source_row_count=0,
                nearest_time_delta_s=np.nan,
                extrapolated=True,
                method="missing-zero",
                interpolation_gap_s=np.nan,
                large_gap_fallback=False,
                classification_policy=class_policy,
                valid=False,
            )
        else:
            position, interpolation_diagnostic = _resampled_position(
                sequence_results,
                timestamp,
                resample_method=method,
                max_interpolation_gap_s=max_interpolation_gap_s,
            )
            classification = _resampled_classification(
                sequence_results,
                timestamp,
                classification_policy=class_policy,
            )
            diagnostic = _diagnostic_record(
                template_index=template_index,
                sequence_id=sequence_id,
                timestamp=timestamp,
                source_row_count=len(sequence_results),
                nearest_time_delta_s=interpolation_diagnostic["nearest_time_delta_s"],
                extrapolated=interpolation_diagnostic["extrapolated"],
                method=interpolation_diagnostic["method"],
                interpolation_gap_s=interpolation_diagnostic["interpolation_gap_s"],
                large_gap_fallback=interpolation_diagnostic["large_gap_fallback"],
                classification_policy=class_policy,
                valid=bool(np.isfinite(position).all()),
            )
        output_records.append(
            {
                "Sequence": sequence_id,
                "Timestamp": timestamp,
                "Position": _format_position(position),
                "Classification": int(classification),
            }
        )
        diagnostic_records.append(diagnostic)

    output = pd.DataFrame.from_records(
        output_records,
        columns=list(OFFICIAL_UG2_RESULT_COLUMNS),
    )
    diagnostics = pd.DataFrame.from_records(diagnostic_records, columns=list(DIAGNOSTIC_COLUMNS))
    return output, diagnostics


def write_template_snapped_submission(
    *,
    results: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
    resample_method: ResampleMethod = "linear",
    max_interpolation_gap_s: float | None = None,
    classification_policy: ClassificationPolicy = "sequence-mode",
    missing_position_policy: MissingPositionPolicy = "zero",
) -> dict[str, Path]:
    """Write snapped official CSV/ZIP, validation rows, and a manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    snapped, diagnostics = snap_official_results_to_template(
        results,
        template,
        resample_method=resample_method,
        max_interpolation_gap_s=max_interpolation_gap_s,
        classification_policy=classification_policy,
        missing_position_policy=missing_position_policy,
    )
    paths = {
        "official_results_csv": output_dir / RESULTS_CSV,
        "official_zip": output_dir / OFFICIAL_ZIP,
        "diagnostics_csv": output_dir / DIAGNOSTICS_CSV,
        "validation_json": output_dir / VALIDATION_JSON,
        "validation_rows_csv": output_dir / VALIDATION_ROWS_CSV,
        "manifest_json": output_dir / MANIFEST_JSON,
    }
    csv_text = snapped.to_csv(index=False)
    paths["official_results_csv"].write_text(csv_text, encoding="utf-8")
    with ZipFile(paths["official_zip"], "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(RESULTS_CSV, csv_text)
    diagnostics.to_csv(paths["diagnostics_csv"], index=False)
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
        "schema": "raft-uav-mmuad-template-snap-v1",
        "row_count": int(len(snapped)),
        "template_row_count": int(len(_normalize_template_rows(template))),
        "source_result_rows": int(len(results)),
        "valid_snapped_rows": int(diagnostics["valid"].astype(bool).sum()),
        "invalid_snapped_rows": int((~diagnostics["valid"].astype(bool)).sum()),
        "extrapolated_rows": int(diagnostics["extrapolated"].astype(bool).sum()),
        "large_gap_fallback_rows": int(diagnostics["large_gap_fallback"].astype(bool).sum()),
        "leaderboard_ready": bool(validation.summary.get("leaderboard_ready", False)),
        "codabench_upload_ready": bool(validation.summary.get("codabench_upload_ready", False)),
        "resample_method": str(resample_method),
        "classification_policy": str(classification_policy),
        "missing_position_policy": str(missing_position_policy),
        "max_interpolation_gap_s": max_interpolation_gap_s,
        "paths": {name: str(path) for name, path in paths.items() if name != "manifest_json"},
    }
    paths["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="official CSV/ZIP to snap")
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="official Track 5 template CSV/ZIP",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resample-method", choices=RESAMPLE_METHODS, default="linear")
    parser.add_argument("--max-interpolation-gap-s", type=float)
    parser.add_argument(
        "--classification-policy",
        choices=CLASSIFICATION_POLICIES,
        default="sequence-mode",
    )
    parser.add_argument(
        "--missing-position-policy",
        choices=MISSING_POSITION_POLICIES,
        default="zero",
    )
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    results = load_official_track5_results_frame(args.results)
    template = load_official_track5_template_file(args.template)
    paths = write_template_snapped_submission(
        results=results,
        template=template,
        output_dir=args.output_dir,
        resample_method=args.resample_method,
        max_interpolation_gap_s=args.max_interpolation_gap_s,
        classification_policy=args.classification_policy,
        missing_position_policy=args.missing_position_policy,
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_template_snap=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"template-snapped upload is not leaderboard-ready: {reasons}")
    return 0


def _normalize_results_rows(results: pd.DataFrame) -> pd.DataFrame:
    rows = load_official_track5_results_frame_from_frame(results)
    if rows.empty:
        positions = np.empty((0, 3), dtype=float)
    else:
        positions = np.asarray(
            [parse_official_position_cell(value) for value in rows["Position"]],
            dtype=float,
        )
    rows = rows.copy()
    rows["x"] = positions[:, 0]
    rows["y"] = positions[:, 1]
    rows["z"] = positions[:, 2]
    rows["Classification"] = pd.to_numeric(rows["Classification"], errors="coerce").astype(int)
    return rows.sort_values(["Sequence", "Timestamp"]).reset_index(drop=True)


def load_official_track5_results_frame_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize an in-memory official Track 5 result-like frame."""

    lower = {str(column).strip().lower(): column for column in frame.columns}
    required = ("sequence", "timestamp", "position", "classification")
    missing = [column for column in required if column not in lower]
    if missing:
        raise ValueError(f"official Track 5 results missing columns: {missing}")
    rows = pd.DataFrame(
        {
            "Sequence": frame[lower["sequence"]].astype(str).str.strip(),
            "Timestamp": pd.to_numeric(frame[lower["timestamp"]], errors="coerce"),
            "Position": frame[lower["position"]],
            "Classification": pd.to_numeric(frame[lower["classification"]], errors="coerce"),
        }
    )
    finite = rows["Sequence"].ne("")
    finite &= rows["Timestamp"].notna()
    finite &= rows["Classification"].notna()
    rows = rows.loc[finite].copy()
    rows["Timestamp"] = rows["Timestamp"].astype(float)
    rows["Classification"] = rows["Classification"].astype(int)
    return rows.sort_values(["Sequence", "Timestamp"]).reset_index(drop=True)


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    lower = {str(column).strip().lower(): column for column in template.columns}
    sequence_col = lower.get("sequence") or lower.get("sequence_id")
    timestamp_col = lower.get("timestamp") or lower.get("time_s")
    if sequence_col is None or timestamp_col is None:
        raise ValueError("template must contain Sequence/Timestamp or sequence_id/time_s")
    rows = pd.DataFrame(
        {
            "Sequence": template[sequence_col].astype(str).str.strip(),
            "Timestamp": pd.to_numeric(template[timestamp_col], errors="coerce"),
        }
    )
    finite = rows["Sequence"].ne("") & rows["Timestamp"].notna()
    return rows.loc[finite].sort_values(["Sequence", "Timestamp"]).reset_index(drop=True)


def _resampled_position(
    sequence_results: pd.DataFrame,
    timestamp: float,
    *,
    resample_method: ResampleMethod,
    max_interpolation_gap_s: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    work = sequence_results.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")
    times = work["Timestamp"].to_numpy(float)
    xyz = work[["x", "y", "z"]].to_numpy(float)
    nearest_index = int(np.argmin(np.abs(times - float(timestamp))))
    nearest_delta = float(float(timestamp) - times[nearest_index])
    extrapolated = bool(float(timestamp) < float(times[0]) or float(timestamp) > float(times[-1]))
    interpolation_gap_s = _bracketing_gap_s(times, float(timestamp))
    large_gap_fallback = (
        max_interpolation_gap_s is not None
        and np.isfinite(interpolation_gap_s)
        and float(interpolation_gap_s) > float(max_interpolation_gap_s)
    )
    if len(times) == 1 or resample_method == "nearest" or large_gap_fallback:
        method_used = "nearest" if not large_gap_fallback else "nearest-large-gap-fallback"
        return xyz[nearest_index].astype(float), {
            "nearest_time_delta_s": nearest_delta,
            "extrapolated": extrapolated,
            "method": method_used,
            "interpolation_gap_s": interpolation_gap_s,
            "large_gap_fallback": bool(large_gap_fallback),
        }
    position = np.asarray(
        [np.interp(float(timestamp), times, xyz[:, axis]) for axis in range(3)],
        dtype=float,
    )
    return position, {
        "nearest_time_delta_s": nearest_delta,
        "extrapolated": extrapolated,
        "method": "linear",
        "interpolation_gap_s": interpolation_gap_s,
        "large_gap_fallback": False,
    }


def _resampled_classification(
    sequence_results: pd.DataFrame,
    timestamp: float,
    *,
    classification_policy: ClassificationPolicy,
) -> int:
    if classification_policy == "sequence-mode":
        mode = sequence_results["Classification"].mode(dropna=True)
        if not mode.empty:
            return int(mode.sort_values().iloc[0])
    times = sequence_results["Timestamp"].to_numpy(float)
    nearest_index = int(np.argmin(np.abs(times - float(timestamp))))
    return int(sequence_results["Classification"].iloc[nearest_index])


def _diagnostic_record(
    *,
    template_index: int,
    sequence_id: str,
    timestamp: float,
    source_row_count: int,
    nearest_time_delta_s: float,
    extrapolated: bool,
    method: str,
    interpolation_gap_s: float,
    large_gap_fallback: bool,
    classification_policy: str,
    valid: bool,
) -> dict[str, Any]:
    return {
        "template_row_index": int(template_index),
        "Sequence": str(sequence_id),
        "Timestamp": float(timestamp),
        "source_row_count": int(source_row_count),
        "nearest_time_delta_s": nearest_time_delta_s,
        "abs_nearest_time_delta_s": abs(float(nearest_time_delta_s))
        if np.isfinite(nearest_time_delta_s)
        else np.nan,
        "extrapolated": bool(extrapolated),
        "method": str(method),
        "interpolation_gap_s": interpolation_gap_s,
        "large_gap_fallback": bool(large_gap_fallback),
        "classification_policy": str(classification_policy),
        "valid": bool(valid),
    }


def _bracketing_gap_s(times: np.ndarray, timestamp: float) -> float:
    if len(times) < 2:
        return np.nan
    if float(timestamp) <= float(times[0]) or float(timestamp) >= float(times[-1]):
        return np.nan
    right = int(np.searchsorted(times, float(timestamp), side="right"))
    left = max(0, right - 1)
    if right >= len(times):
        return np.nan
    return float(times[right] - times[left])


def _format_position(position: np.ndarray) -> str:
    x, y, z = [float(value) for value in position]
    return f"({_format_float(x)},{_format_float(y)},{_format_float(z)})"


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _normalize_choice(value: str, choices: tuple[str, ...], name: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in choices:
        raise ValueError(f"{name} must be one of {choices}; got {value!r}")
    return normalized


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
