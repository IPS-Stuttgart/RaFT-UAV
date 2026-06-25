"""Preserve raw and train-calibrated MMUAD candidate hypotheses.

A source transform can improve difficult sequences while degrading already-good
ones.  Replacing the raw stream therefore lowers candidate recall before a
branch-aware reservoir or mixture smoother can decide which hypothesis to use.
This module keeps the raw and calibrated coordinates as separate candidate
branches and records the applied displacement for downstream ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import build_candidate_reservoir
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns
from raft_uav.mmuad.source_calibration import (
    SOURCE_CALIBRATION_MODES,
    apply_source_calibration_payload,
    load_source_calibration_json,
)


DEFAULT_RAW_BRANCH = "raw"
ORIGIN_ROW_COLUMN = "mmuad_calibration_origin_row"
ORIGINAL_TRACK_ID_COLUMN = "mmuad_original_track_id"
ORIGINAL_XYZ_COLUMNS = (
    "mmuad_original_x_m",
    "mmuad_original_y_m",
    "mmuad_original_z_m",
)


def build_source_calibration_branch_union(
    candidates: CandidateFrame | pd.DataFrame,
    calibration_payload: dict[str, Any],
    *,
    mode: str | None = None,
    raw_branch: str = DEFAULT_RAW_BRANCH,
    calibrated_branch: str | None = None,
    keep_unapplied_calibrated: bool = False,
    branch_track_ids: bool = True,
) -> CandidateFrame:
    """Return raw and calibrated candidates as distinct hypothesis branches.

    The calibrated branch is added only for rows where a source transform was
    actually applied unless ``keep_unapplied_calibrated`` is enabled.  Original
    coordinates and displacement diagnostics remain on both branches.  Track IDs
    are branch-qualified by default so downstream temporal association does not
    treat two coordinate hypotheses as one physical observation.
    """

    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(rows)

    selected_mode = str(mode or calibration_payload.get("mode", "identity"))
    raw_label = _branch_label(raw_branch)
    calibrated_label = _branch_label(
        calibrated_branch or f"{selected_mode.replace('-', '_')}_calibrated"
    )
    prepared = rows.copy().reset_index(drop=True)
    prepared[ORIGIN_ROW_COLUMN] = np.arange(len(prepared), dtype=int)
    prepared[ORIGINAL_TRACK_ID_COLUMN] = prepared["track_id"]
    prepared[ORIGINAL_XYZ_COLUMNS[0]] = pd.to_numeric(prepared["x_m"], errors="coerce")
    prepared[ORIGINAL_XYZ_COLUMNS[1]] = pd.to_numeric(prepared["y_m"], errors="coerce")
    prepared[ORIGINAL_XYZ_COLUMNS[2]] = pd.to_numeric(prepared["z_m"], errors="coerce")

    raw_rows = _annotate_branch_rows(
        prepared,
        branch=raw_label,
        calibration_mode=selected_mode,
        is_calibrated=False,
        branch_track_ids=branch_track_ids,
    )
    calibrated = apply_source_calibration_payload(
        CandidateFrame(prepared),
        calibration_payload,
        mode=mode,
    ).rows
    calibrated = _annotate_branch_rows(
        calibrated,
        branch=calibrated_label,
        calibration_mode=selected_mode,
        is_calibrated=True,
        branch_track_ids=branch_track_ids,
    )
    if not keep_unapplied_calibrated:
        applied = calibrated.get("mmuad_source_calibration_applied")
        if applied is None:
            calibrated = calibrated.iloc[0:0].copy()
        else:
            calibrated = calibrated.loc[pd.Series(applied, index=calibrated.index).fillna(False)]

    combined = pd.concat([raw_rows, calibrated], ignore_index=True, sort=False)
    return CandidateFrame(normalize_candidate_columns(combined))


def source_calibration_branch_summary(rows: pd.DataFrame) -> dict[str, Any]:
    """Return a compact JSON-serializable branch-union summary."""

    frame = pd.DataFrame(rows).copy()
    displacement = pd.to_numeric(
        frame.get("mmuad_calibration_displacement_m", pd.Series(dtype=float)),
        errors="coerce",
    )
    finite_displacement = displacement[np.isfinite(displacement.to_numpy(float))]
    calibrated = pd.Series(
        frame.get("mmuad_candidate_branch_is_calibrated", False),
        index=frame.index,
    ).fillna(False).astype(bool)
    applied = pd.Series(
        frame.get("mmuad_source_calibration_applied", False),
        index=frame.index,
    ).fillna(False).astype(bool)
    return {
        "row_count": int(len(frame)),
        "raw_branch_row_count": int((~calibrated).sum()),
        "calibrated_branch_row_count": int(calibrated.sum()),
        "calibration_applied_row_count": int(applied.sum()),
        "candidate_branch_counts": _value_counts(frame, "candidate_branch"),
        "source_counts": _value_counts(frame, "source"),
        "calibration_displacement_mean_m": _safe_mean(finite_displacement),
        "calibration_displacement_p95_m": _safe_quantile(finite_displacement, 0.95),
        "calibration_displacement_max_m": _safe_max(finite_displacement),
        "distinct_origin_row_count": int(
            pd.to_numeric(frame.get(ORIGIN_ROW_COLUMN), errors="coerce").nunique()
            if ORIGIN_ROW_COLUMN in frame.columns
            else 0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-source-calibration-branches",
        description=(
            "preserve raw and train-calibrated MMUAD candidate hypotheses as "
            "separate branches"
        ),
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-candidates", type=Path, required=True)
    parser.add_argument("--mmuad-source-calibration-json", type=Path, required=True)
    parser.add_argument("--mmuad-source-calibration-mode", choices=SOURCE_CALIBRATION_MODES)
    parser.add_argument("--raw-branch", default=DEFAULT_RAW_BRANCH)
    parser.add_argument("--calibrated-branch")
    parser.add_argument("--keep-unapplied-calibrated", action="store_true")
    parser.add_argument("--keep-original-track-ids", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument(
        "--reservoir-output-csv",
        type=Path,
        help="optionally write a branch-aware bounded reservoir from the union",
    )
    parser.add_argument("--reservoir-global-top-n", type=int, default=20)
    parser.add_argument("--reservoir-per-source-top-n", type=int, default=3)
    parser.add_argument("--reservoir-per-branch-top-n", type=int, default=3)
    parser.add_argument("--reservoir-max-candidates-per-frame", type=int, default=40)
    args = parser.parse_args(argv)

    input_frame = load_candidate_file(args.candidates)
    payload = load_source_calibration_json(args.mmuad_source_calibration_json)
    union = build_source_calibration_branch_union(
        input_frame,
        payload,
        mode=args.mmuad_source_calibration_mode,
        raw_branch=args.raw_branch,
        calibrated_branch=args.calibrated_branch,
        keep_unapplied_calibrated=args.keep_unapplied_calibrated,
        branch_track_ids=not args.keep_original_track_ids,
    )
    args.output_candidates.parent.mkdir(parents=True, exist_ok=True)
    union.rows.to_csv(args.output_candidates, index=False)

    reservoir = None
    if args.reservoir_output_csv is not None:
        reservoir = build_candidate_reservoir(
            union.rows,
            global_top_n=args.reservoir_global_top_n,
            top_per_source=args.reservoir_per_source_top_n,
            top_per_branch=args.reservoir_per_branch_top_n,
            max_candidates_per_frame=args.reservoir_max_candidates_per_frame,
        )
        args.reservoir_output_csv.parent.mkdir(parents=True, exist_ok=True)
        reservoir.to_csv(args.reservoir_output_csv, index=False)

    summary = source_calibration_branch_summary(union.rows)
    summary.update(
        {
            "input_candidates": str(args.candidates),
            "source_calibration_json": str(args.mmuad_source_calibration_json),
            "output_candidates": str(args.output_candidates),
            "reservoir_output_csv": (
                str(args.reservoir_output_csv)
                if args.reservoir_output_csv is not None
                else None
            ),
            "reservoir_row_count": None if reservoir is None else int(len(reservoir)),
        }
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("mmuad_source_calibration_branches=ok")
    print(f"output_candidates={args.output_candidates}")
    print(f"rows={len(union.rows)}")
    print(f"calibrated_rows={summary['calibrated_branch_row_count']}")
    if args.reservoir_output_csv is not None:
        print(f"reservoir_output_csv={args.reservoir_output_csv}")
        print(f"reservoir_rows={len(reservoir)}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    return 0


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    rows = candidates.rows.copy() if isinstance(candidates, CandidateFrame) else pd.DataFrame(candidates)
    return normalize_candidate_columns(rows)


def _annotate_branch_rows(
    rows: pd.DataFrame,
    *,
    branch: str,
    calibration_mode: str,
    is_calibrated: bool,
    branch_track_ids: bool,
) -> pd.DataFrame:
    out = pd.DataFrame(rows).copy()
    out["candidate_branch"] = branch
    out["mmuad_source_calibration_branch"] = branch
    out["mmuad_candidate_branch_is_calibrated"] = bool(is_calibrated)
    out["mmuad_branch_calibration_mode"] = str(calibration_mode)
    original_xyz = out[list(ORIGINAL_XYZ_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    current_xyz = out[["x_m", "y_m", "z_m"]].apply(pd.to_numeric, errors="coerce")
    delta = current_xyz.to_numpy(float) - original_xyz.to_numpy(float)
    out["mmuad_calibration_dx_m"] = delta[:, 0]
    out["mmuad_calibration_dy_m"] = delta[:, 1]
    out["mmuad_calibration_dz_m"] = delta[:, 2]
    out["mmuad_calibration_displacement_m"] = np.linalg.norm(delta, axis=1)
    if not is_calibrated:
        out["mmuad_source_calibration_applied"] = False
    if branch_track_ids:
        out["track_id"] = [
            _qualified_track_id(original, branch, origin_row)
            for original, origin_row in zip(
                out[ORIGINAL_TRACK_ID_COLUMN],
                out[ORIGIN_ROW_COLUMN],
                strict=False,
            )
        ]
    return out


def _qualified_track_id(original: object, branch: str, origin_row: object) -> str:
    if original is None or pd.isna(original) or str(original).strip() == "":
        base = f"row-{int(origin_row)}"
    else:
        base = str(original)
    return f"{base}@{branch}"


def _branch_label(value: str) -> str:
    label = str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    if not label:
        raise ValueError("candidate branch label must not be empty")
    return label


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns or rows.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in rows[column].fillna("").astype(str).value_counts().sort_index().items()
    }


def _safe_mean(values: pd.Series) -> float | None:
    return None if values.empty else float(values.mean())


def _safe_quantile(values: pd.Series, quantile: float) -> float | None:
    return None if values.empty else float(values.quantile(float(quantile)))


def _safe_max(values: pd.Series) -> float | None:
    return None if values.empty else float(values.max())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
