"""Interpolate MMUAD candidate branches along a train-fitted calibration path.

A train-fitted source transform can under-correct or over-correct a candidate on
an unseen sequence.  Keeping only the raw and fully calibrated endpoints forces
downstream assignment to choose between those two hypotheses.  This module adds
optional intermediate coordinate hypotheses on the line from the raw candidate
to its calibrated counterpart while preserving a shared physical-origin ID.

The extra rows are intended for branch-preserving reservoirs followed by the
origin-group multiplicity correction and robust candidate-mixture inference.
No truth is used when constructing the ensemble.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import build_candidate_reservoir
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns
from raft_uav.mmuad.source_calibration import (
    SOURCE_CALIBRATION_MODES,
    load_source_calibration_json,
)
from raft_uav.mmuad.source_calibration_branches import (
    ORIGINAL_TRACK_ID_COLUMN,
    ORIGINAL_XYZ_COLUMNS,
    ORIGIN_ROW_COLUMN,
    build_source_calibration_branch_union,
    source_calibration_branch_summary,
)

DEFAULT_CALIBRATION_FRACTIONS = (0.0, 0.5, 1.0)
CALIBRATION_FRACTION_COLUMN = "mmuad_calibration_path_fraction"
EFFECTIVE_ALPHA_COLUMN = "mmuad_source_calibration_effective_alpha"
INTERPOLATED_COLUMN = "mmuad_calibration_path_interpolated"


def build_source_calibration_path_ensemble(
    candidates: CandidateFrame | pd.DataFrame,
    calibration_payload: dict[str, Any],
    *,
    fractions: Sequence[float] = DEFAULT_CALIBRATION_FRACTIONS,
    mode: str | None = None,
    raw_branch: str = "raw",
    calibrated_branch: str | None = None,
    intermediate_branch_prefix: str | None = None,
    keep_unapplied_calibrated: bool = False,
    branch_track_ids: bool = True,
) -> CandidateFrame:
    """Return raw, intermediate, and fully calibrated candidate hypotheses.

    Fractions are constrained to ``[0, 1]``.  Fraction zero is the raw candidate,
    fraction one is the train-fitted calibrated candidate, and interior values
    linearly interpolate the coordinates.  Every derived row retains the same
    ``mmuad_calibration_origin_row`` so downstream physical-group correction can
    remove representation-count bias without discarding coordinate hypotheses.
    """

    normalized_fractions = _normalize_fractions(fractions)
    union = build_source_calibration_branch_union(
        candidates,
        calibration_payload,
        mode=mode,
        raw_branch=raw_branch,
        calibrated_branch=calibrated_branch,
        keep_unapplied_calibrated=keep_unapplied_calibrated,
        branch_track_ids=False,
    ).rows
    if union.empty:
        return CandidateFrame(_empty_ensemble_rows(union))

    calibrated_mask = _boolean_series(
        union.get("mmuad_candidate_branch_is_calibrated", False),
        union.index,
    )
    raw_rows = union.loc[~calibrated_mask].copy()
    calibrated_rows = union.loc[calibrated_mask].copy()
    calibrated_label = _calibrated_label(calibrated_rows, calibration_payload, mode)
    prefix = _branch_label(intermediate_branch_prefix or f"{calibrated_label}_path")

    parts: list[pd.DataFrame] = []
    if _contains_fraction(normalized_fractions, 0.0):
        parts.append(_annotate_fraction(raw_rows, fraction=0.0, branch=None))
    for fraction in normalized_fractions:
        if np.isclose(fraction, 0.0) or np.isclose(fraction, 1.0):
            continue
        branch = f"{prefix}_f{_fraction_token(fraction)}"
        parts.append(
            _interpolate_calibrated_rows(
                calibrated_rows,
                fraction=float(fraction),
                branch=branch,
            )
        )
    if _contains_fraction(normalized_fractions, 1.0):
        parts.append(_annotate_fraction(calibrated_rows, fraction=1.0, branch=None))

    if not parts:
        return CandidateFrame(_empty_ensemble_rows(union))
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out.loc[
        np.isfinite(out[["x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    ].copy()
    if branch_track_ids:
        out["track_id"] = [
            _qualified_track_id(original, branch, origin)
            for original, branch, origin in zip(
                out[ORIGINAL_TRACK_ID_COLUMN],
                out["candidate_branch"],
                out[ORIGIN_ROW_COLUMN],
                strict=False,
            )
        ]
    return CandidateFrame(normalize_candidate_columns(out))


def source_calibration_path_summary(rows: pd.DataFrame) -> dict[str, Any]:
    """Return branch-union and calibration-path diagnostics."""

    frame = pd.DataFrame(rows).copy()
    summary = source_calibration_branch_summary(frame)
    fractions = pd.to_numeric(
        frame.get(CALIBRATION_FRACTION_COLUMN, pd.Series(dtype=float)),
        errors="coerce",
    )
    finite = fractions[np.isfinite(fractions.to_numpy(float))]
    interpolated = _boolean_series(frame.get(INTERPOLATED_COLUMN, False), frame.index)
    summary.update(
        {
            "calibration_path_fractions": sorted(
                {float(value) for value in finite.to_numpy(float)}
            ),
            "calibration_path_fraction_counts": {
                f"{float(key):g}": int(value)
                for key, value in finite.value_counts().sort_index().items()
            },
            "interpolated_branch_row_count": int(interpolated.sum()),
            "interpolated_branch_count": int(
                frame.loc[interpolated, "candidate_branch"].astype(str).nunique()
                if "candidate_branch" in frame.columns
                else 0
            ),
            "truth_used_for_calibration_path_ensemble": False,
        }
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-source-calibration-path-ensemble",
        description=(
            "preserve raw and calibrated MMUAD candidates plus optional "
            "intermediate calibration-path branches"
        ),
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-candidates", type=Path, required=True)
    parser.add_argument("--mmuad-source-calibration-json", type=Path, required=True)
    parser.add_argument("--mmuad-source-calibration-mode", choices=SOURCE_CALIBRATION_MODES)
    parser.add_argument(
        "--calibration-fractions",
        default=",".join(f"{value:g}" for value in DEFAULT_CALIBRATION_FRACTIONS),
        help="comma-separated raw-to-calibrated fractions in [0,1]",
    )
    parser.add_argument("--raw-branch", default="raw")
    parser.add_argument("--calibrated-branch")
    parser.add_argument("--intermediate-branch-prefix")
    parser.add_argument("--keep-unapplied-calibrated", action="store_true")
    parser.add_argument("--keep-original-track-ids", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--reservoir-output-csv", type=Path)
    parser.add_argument("--reservoir-global-top-n", type=int, default=20)
    parser.add_argument("--reservoir-per-source-top-n", type=int, default=3)
    parser.add_argument("--reservoir-per-branch-top-n", type=int, default=3)
    parser.add_argument("--reservoir-max-candidates-per-frame", type=int, default=40)
    args = parser.parse_args(argv)

    input_frame = load_candidate_file(args.candidates)
    payload = load_source_calibration_json(args.mmuad_source_calibration_json)
    fractions = _parse_fraction_text(args.calibration_fractions)
    ensemble = build_source_calibration_path_ensemble(
        input_frame,
        payload,
        fractions=fractions,
        mode=args.mmuad_source_calibration_mode,
        raw_branch=args.raw_branch,
        calibrated_branch=args.calibrated_branch,
        intermediate_branch_prefix=args.intermediate_branch_prefix,
        keep_unapplied_calibrated=args.keep_unapplied_calibrated,
        branch_track_ids=not args.keep_original_track_ids,
    )
    args.output_candidates.parent.mkdir(parents=True, exist_ok=True)
    ensemble.rows.to_csv(args.output_candidates, index=False)

    reservoir = None
    if args.reservoir_output_csv is not None:
        reservoir = build_candidate_reservoir(
            ensemble.rows,
            global_top_n=args.reservoir_global_top_n,
            top_per_source=args.reservoir_per_source_top_n,
            top_per_branch=args.reservoir_per_branch_top_n,
            max_candidates_per_frame=args.reservoir_max_candidates_per_frame,
        )
        args.reservoir_output_csv.parent.mkdir(parents=True, exist_ok=True)
        reservoir.to_csv(args.reservoir_output_csv, index=False)

    summary = source_calibration_path_summary(ensemble.rows)
    summary.update(
        {
            "input_candidates": str(args.candidates),
            "source_calibration_json": str(args.mmuad_source_calibration_json),
            "output_candidates": str(args.output_candidates),
            "requested_calibration_fractions": list(fractions),
            "reservoir_output_csv": (
                None
                if args.reservoir_output_csv is None
                else str(args.reservoir_output_csv)
            ),
            "reservoir_row_count": None if reservoir is None else int(len(reservoir)),
        }
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("mmuad_source_calibration_path_ensemble=ok")
    print(f"output_candidates={args.output_candidates}")
    print(f"rows={len(ensemble.rows)}")
    print(f"fractions={','.join(f'{value:g}' for value in fractions)}")
    if args.reservoir_output_csv is not None:
        print(f"reservoir_output_csv={args.reservoir_output_csv}")
        print(f"reservoir_rows={len(reservoir)}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    return 0


def _interpolate_calibrated_rows(
    calibrated_rows: pd.DataFrame,
    *,
    fraction: float,
    branch: str,
) -> pd.DataFrame:
    if calibrated_rows.empty:
        return calibrated_rows.copy()
    out = calibrated_rows.copy()
    original_xyz = out[list(ORIGINAL_XYZ_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    calibrated_xyz = out[["x_m", "y_m", "z_m"]].apply(pd.to_numeric, errors="coerce")
    interpolated_xyz = original_xyz.to_numpy(float) + float(fraction) * (
        calibrated_xyz.to_numpy(float) - original_xyz.to_numpy(float)
    )
    out[["x_m", "y_m", "z_m"]] = interpolated_xyz
    return _annotate_fraction(out, fraction=fraction, branch=branch)


def _annotate_fraction(
    rows: pd.DataFrame,
    *,
    fraction: float,
    branch: str | None,
) -> pd.DataFrame:
    out = rows.copy()
    if branch is not None:
        branch = _branch_label(branch)
        out["candidate_branch"] = branch
        out["mmuad_source_calibration_branch"] = branch
    out[CALIBRATION_FRACTION_COLUMN] = float(fraction)
    out[INTERPOLATED_COLUMN] = bool(0.0 < float(fraction) < 1.0)
    calibrated = bool(float(fraction) > 0.0)
    out["mmuad_candidate_branch_is_calibrated"] = calibrated
    if not calibrated:
        out["mmuad_source_calibration_applied"] = False

    original_xyz = out[list(ORIGINAL_XYZ_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    current_xyz = out[["x_m", "y_m", "z_m"]].apply(pd.to_numeric, errors="coerce")
    delta = current_xyz.to_numpy(float) - original_xyz.to_numpy(float)
    out["mmuad_calibration_dx_m"] = delta[:, 0]
    out["mmuad_calibration_dy_m"] = delta[:, 1]
    out["mmuad_calibration_dz_m"] = delta[:, 2]
    out["mmuad_calibration_displacement_m"] = np.linalg.norm(delta, axis=1)

    alpha = pd.to_numeric(
        out.get("mmuad_source_calibration_alpha", pd.Series(np.nan, index=out.index)),
        errors="coerce",
    )
    out[EFFECTIVE_ALPHA_COLUMN] = alpha.to_numpy(float) * float(fraction)
    if np.isclose(fraction, 0.0):
        out[EFFECTIVE_ALPHA_COLUMN] = 0.0
    return out


def _empty_ensemble_rows(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(rows).copy()
    out[CALIBRATION_FRACTION_COLUMN] = pd.Series(dtype=float)
    out[EFFECTIVE_ALPHA_COLUMN] = pd.Series(dtype=float)
    out[INTERPOLATED_COLUMN] = pd.Series(dtype=bool)
    return out


def _normalize_fractions(values: Iterable[float]) -> tuple[float, ...]:
    fractions = tuple(float(value) for value in values)
    if not fractions:
        raise ValueError("at least one calibration fraction is required")
    for value in fractions:
        if not np.isfinite(value):
            raise ValueError("calibration fractions must be finite")
        if value < 0.0 or value > 1.0:
            raise ValueError("calibration fractions must lie in [0, 1]")
    return tuple(sorted(set(fractions)))


def _parse_fraction_text(value: str) -> tuple[float, ...]:
    tokens = [token.strip() for token in str(value).split(",") if token.strip()]
    return _normalize_fractions(float(token) for token in tokens)


def _contains_fraction(values: Sequence[float], target: float) -> bool:
    return any(np.isclose(float(value), float(target)) for value in values)


def _calibrated_label(
    calibrated_rows: pd.DataFrame,
    payload: dict[str, Any],
    mode: str | None,
) -> str:
    if not calibrated_rows.empty and "candidate_branch" in calibrated_rows.columns:
        labels = calibrated_rows["candidate_branch"].dropna().astype(str)
        if not labels.empty:
            return _branch_label(labels.iloc[0])
    selected_mode = str(mode or payload.get("mode", "source-translation"))
    return _branch_label(f"{selected_mode.replace('-', '_')}_calibrated")


def _qualified_track_id(original: object, branch: object, origin_row: object) -> str:
    if original is None or pd.isna(original) or str(original).strip() == "":
        base = f"row-{int(origin_row)}"
    else:
        base = str(original)
    return f"{base}@{_branch_label(str(branch))}"


def _branch_label(value: str) -> str:
    label = str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    if not label:
        raise ValueError("candidate branch label must not be empty")
    return label


def _fraction_token(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def _boolean_series(values: Any, index: pd.Index) -> pd.Series:
    series = pd.Series(values, index=index)
    if series.empty:
        return pd.Series(False, index=index, dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0).ne(0.0)
    text = series.astype("string").str.strip().str.lower()
    truthy = text.isin({"1", "true", "t", "yes", "y"})
    falsey = text.isin({"0", "false", "f", "no", "n", "", "none", "null", "nan"})
    numeric = pd.to_numeric(text, errors="coerce").fillna(0.0).ne(0.0)
    return (truthy | (~falsey & numeric)).fillna(False).astype(bool)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
