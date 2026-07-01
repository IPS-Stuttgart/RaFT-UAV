#!/usr/bin/env python
"""Snap an official MMUAD Track 5 results table to a template grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from raft_uav.mmuad.submission import (  # noqa: E402
    OFFICIAL_UG2_RESULT_COLUMNS,
    load_official_track5_results_frame,
    load_official_track5_template_file,
    normalize_official_track5_results_frame,
    parse_official_position_cell,
    validate_official_track5_submission,
)
from raft_uav.mmuad.track5_template_resample import (  # noqa: E402
    resample_estimates_to_track5_template,
)

RESAMPLE_METHODS = ("linear", "nearest")
CLASSIFICATION_POLICIES = ("sequence-mode", "nearest")
MISSING_POSITION_POLICIES = ("zero", "raise")
RESULTS_CSV = "mmaud_results.csv"
OFFICIAL_ZIP = "ug2_submission.zip"
DIAGNOSTICS_CSV = "mmuad_template_snap_diagnostics.csv"
VALIDATION_JSON = "mmuad_template_snap_validation.json"
VALIDATION_ROWS_CSV = "mmuad_template_snap_validation_rows.csv"
MANIFEST_JSON = "mmuad_template_snap_manifest.json"


def snap_official_results_to_template(
    results: pd.DataFrame,
    template: pd.DataFrame,
    *,
    resample_method: str = "linear",
    max_interpolation_gap_s: float | None = None,
    classification_policy: str = "sequence-mode",
    missing_position_policy: str = "zero",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return official rows snapped to a template and per-row diagnostics."""

    method = _choice(resample_method, RESAMPLE_METHODS, "resample_method")
    class_policy = _choice(classification_policy, CLASSIFICATION_POLICIES, "classification_policy")
    missing_policy = _choice(missing_position_policy, MISSING_POSITION_POLICIES, "missing_position_policy")
    source = _official_rows_as_estimates(results)
    resampled, raw_diag = resample_estimates_to_track5_template(
        source,
        template,
        resample_method=method,
        max_interpolation_gap_s=max_interpolation_gap_s,
    )
    if missing_policy == "raise":
        missing = raw_diag.loc[pd.to_numeric(raw_diag.get("source_row_count", 0), errors="coerce").fillna(0) == 0]
        if not missing.empty:
            sequence = str(missing["sequence_id"].iloc[0])
            raise ValueError(f"no source results for template sequence {sequence!r}")

    by_sequence = {
        sequence: group.sort_values("Timestamp").reset_index(drop=True)
        for sequence, group in source.groupby("Sequence", sort=True)
    }
    records: list[dict[str, Any]] = []
    diag_records: list[dict[str, Any]] = []
    for template_index, (row, diag) in enumerate(zip(resampled.to_dict("records"), raw_diag.to_dict("records"), strict=False)):
        sequence = str(row["sequence_id"])
        timestamp = float(row["time_s"])
        source_count = int(diag.get("source_row_count", 0) or 0)
        missing_source = source_count == 0
        position = np.array(
            [row.get("state_x_m", np.nan), row.get("state_y_m", np.nan), row.get("state_z_m", np.nan)],
            dtype=float,
        )
        if missing_source or not np.isfinite(position).all():
            position = np.zeros(3, dtype=float)
        classification = 0 if missing_source else _classification(by_sequence[sequence], timestamp, class_policy)
        large_gap = bool(diag.get("large_gap_fallback", False))
        method_used = str(diag.get("resample_method", method))
        if missing_source:
            method_used = "missing-zero"
        elif large_gap:
            method_used = "nearest-large-gap-fallback"
        nearest_delta = float(diag.get("nearest_time_delta_s", np.nan))
        valid = bool(diag.get("valid", False)) and not missing_source
        records.append(
            {
                "Sequence": sequence,
                "Timestamp": timestamp,
                "Position": _format_position(position),
                "Classification": int(classification),
            }
        )
        diag_records.append(
            {
                "template_row_index": int(template_index),
                "Sequence": sequence,
                "Timestamp": timestamp,
                "source_row_count": source_count,
                "nearest_time_delta_s": nearest_delta,
                "abs_nearest_time_delta_s": abs(nearest_delta) if np.isfinite(nearest_delta) else np.nan,
                "extrapolated": bool(diag.get("extrapolated", False)) or missing_source,
                "method": method_used,
                "interpolation_gap_s": float(diag.get("interpolation_gap_s", np.nan)),
                "large_gap_fallback": large_gap,
                "classification_policy": class_policy,
                "valid": valid,
            }
        )
    return (
        pd.DataFrame.from_records(records, columns=list(OFFICIAL_UG2_RESULT_COLUMNS)),
        pd.DataFrame.from_records(diag_records),
    )


def write_template_snapped_submission(
    *,
    results: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
    resample_method: str = "linear",
    max_interpolation_gap_s: float | None = None,
    classification_policy: str = "sequence-mode",
    missing_position_policy: str = "zero",
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
    validation = validate_official_track5_submission(paths["official_zip"], template=template, require_zip=True)
    paths["validation_json"].write_text(json.dumps(_jsonable(validation.summary), indent=2), encoding="utf-8")
    validation.rows.to_csv(paths["validation_rows_csv"], index=False)
    manifest = {
        "schema": "raft-uav-mmuad-template-snap-v1",
        "row_count": int(len(snapped)),
        "source_result_rows": int(len(results)),
        "valid_snapped_rows": int(diagnostics["valid"].astype(bool).sum()),
        "invalid_snapped_rows": int((~diagnostics["valid"].astype(bool)).sum()),
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


def load_official_track5_results_frame_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize an in-memory official Track 5 result-like frame."""

    return normalize_official_track5_results_frame(pd.DataFrame(frame))


def _official_rows_as_estimates(frame: pd.DataFrame) -> pd.DataFrame:
    rows = load_official_track5_results_frame_from_frame(frame)
    positions = np.asarray([parse_official_position_cell(value) for value in rows["Position"]], dtype=float).reshape(-1, 3)
    return pd.DataFrame(
        {
            "Sequence": rows["Sequence"].astype(str),
            "Timestamp": rows["Timestamp"].astype(float),
            "sequence_id": rows["Sequence"].astype(str),
            "time_s": rows["Timestamp"].astype(float),
            "state_x_m": positions[:, 0],
            "state_y_m": positions[:, 1],
            "state_z_m": positions[:, 2],
            "Classification": rows["Classification"].astype(int),
        }
    )


def _classification(rows: pd.DataFrame, timestamp: float, policy: str) -> int:
    if policy == "sequence-mode":
        mode = rows["Classification"].mode(dropna=True)
        if not mode.empty:
            return int(mode.sort_values().iloc[0])
    times = rows["Timestamp"].to_numpy(float)
    return int(rows["Classification"].iloc[int(np.argmin(np.abs(times - float(timestamp))))])


def _format_position(position: np.ndarray) -> str:
    return "(" + ",".join(f"{float(value):.12g}" for value in position) + ")"


def _choice(value: str, choices: tuple[str, ...], name: str) -> str:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="official CSV/ZIP to snap")
    parser.add_argument("--template", type=Path, required=True, help="official Track 5 template CSV/ZIP")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resample-method", choices=RESAMPLE_METHODS, default="linear")
    parser.add_argument("--max-interpolation-gap-s", type=float)
    parser.add_argument("--classification-policy", choices=CLASSIFICATION_POLICIES, default="sequence-mode")
    parser.add_argument("--missing-position-policy", choices=MISSING_POSITION_POLICIES, default="zero")
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)
    paths = write_template_snapped_submission(
        results=load_official_track5_results_frame(args.results),
        template=load_official_track5_template_file(args.template),
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
