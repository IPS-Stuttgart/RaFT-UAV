#!/usr/bin/env python
"""Write raw + source-calibrated MMUAD candidates while preserving input branches.

This helper keeps raw and calibrated coordinates as separate hypotheses with
labels such as ``static:raw`` and ``static:source_translation_calibrated``.  It
is intended for branch-preserving reservoirs/mixture-MAP experiments where
source calibration should be an additional candidate branch, not an early
replacement that can lower oracle recall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.candidate_reservoir import build_candidate_reservoir  # noqa: E402
from raft_uav.mmuad.io import load_candidate_file  # noqa: E402
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns  # noqa: E402
from raft_uav.mmuad.source_calibration import (  # noqa: E402
    SOURCE_CALIBRATION_MODES,
    apply_source_calibration_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-candidates", type=Path, required=True)
    parser.add_argument("--mmuad-source-calibration-json", type=Path, required=True)
    parser.add_argument("--mmuad-source-calibration-mode", choices=SOURCE_CALIBRATION_MODES)
    parser.add_argument("--raw-branch-suffix", default="raw")
    parser.add_argument("--calibrated-branch-suffix", default="source_translation_calibrated")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--reservoir-output-csv", type=Path)
    parser.add_argument("--reservoir-global-top-n", type=int, default=20)
    parser.add_argument("--reservoir-per-source-top-n", type=int, default=3)
    parser.add_argument("--reservoir-per-branch-top-n", type=int, default=3)
    parser.add_argument("--reservoir-max-candidates-per-frame", type=int, default=40)
    args = parser.parse_args(argv)

    raw = load_candidate_file(args.candidates).rows
    union = build_preserved_branch_union(
        raw,
        source_calibration_json=args.mmuad_source_calibration_json,
        mode=args.mmuad_source_calibration_mode,
        raw_branch_suffix=args.raw_branch_suffix,
        calibrated_branch_suffix=args.calibrated_branch_suffix,
    )
    args.output_candidates.parent.mkdir(parents=True, exist_ok=True)
    union.to_csv(args.output_candidates, index=False)

    reservoir = None
    if args.reservoir_output_csv is not None:
        reservoir = build_candidate_reservoir(
            union,
            global_top_n=args.reservoir_global_top_n,
            top_per_source=args.reservoir_per_source_top_n,
            top_per_branch=args.reservoir_per_branch_top_n,
            max_candidates_per_frame=args.reservoir_max_candidates_per_frame,
        )
        args.reservoir_output_csv.parent.mkdir(parents=True, exist_ok=True)
        reservoir.to_csv(args.reservoir_output_csv, index=False)

    summary = branch_union_summary(union)
    summary.update(
        {
            "input_candidates": str(args.candidates),
            "source_calibration_json": str(args.mmuad_source_calibration_json),
            "output_candidates": str(args.output_candidates),
            "reservoir_output_csv": None if args.reservoir_output_csv is None else str(args.reservoir_output_csv),
            "reservoir_row_count": None if reservoir is None else int(len(reservoir)),
        }
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("mmuad_preserve_input_branch_source_calibration=ok")
    print(f"output_candidates={args.output_candidates}")
    print(f"rows={len(union)}")
    print(f"candidate_branch_count={union['candidate_branch'].nunique()}")
    if args.reservoir_output_csv is not None:
        print(f"reservoir_output_csv={args.reservoir_output_csv}")
        print(f"reservoir_rows={len(reservoir)}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    return 0


def build_preserved_branch_union(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    source_calibration_json: Path,
    mode: str | None = None,
    raw_branch_suffix: str = "raw",
    calibrated_branch_suffix: str = "source_translation_calibrated",
) -> pd.DataFrame:
    """Return raw and calibrated candidates with input branches kept distinct."""

    raw = candidates.rows.copy() if isinstance(candidates, CandidateFrame) else pd.DataFrame(candidates)
    raw = normalize_candidate_columns(raw).reset_index(drop=True)
    if raw.empty:
        return raw
    if "track_id" not in raw.columns:
        raw["track_id"] = [f"row-{idx}" for idx in range(len(raw))]
    raw["mmuad_calibration_origin_row"] = np.arange(len(raw), dtype=int)
    raw["mmuad_input_candidate_branch"] = _input_branch(raw)
    raw["mmuad_original_track_id"] = raw["track_id"].astype(str)
    for source, target in (("x_m", "mmuad_original_x_m"), ("y_m", "mmuad_original_y_m"), ("z_m", "mmuad_original_z_m")):
        raw[target] = pd.to_numeric(raw[source], errors="coerce")

    calibrated = apply_source_calibration_json(CandidateFrame(raw), source_calibration_json, mode=mode).rows
    applied = pd.Series(calibrated.get("mmuad_source_calibration_applied", False), index=calibrated.index).fillna(False).astype(bool)
    calibrated = calibrated.loc[applied].copy()

    raw_branch = _annotate_branch(raw, suffix=raw_branch_suffix, calibrated=False)
    calibrated_branch = _annotate_branch(calibrated, suffix=calibrated_branch_suffix, calibrated=True)
    union = pd.concat([raw_branch, calibrated_branch], ignore_index=True, sort=False)
    return normalize_candidate_columns(union)


def branch_union_summary(rows: pd.DataFrame) -> dict[str, Any]:
    frame = pd.DataFrame(rows).copy()
    calibrated = pd.Series(frame.get("mmuad_candidate_branch_is_calibrated", False), index=frame.index).fillna(False).astype(bool)
    return {
        "row_count": int(len(frame)),
        "raw_row_count": int((~calibrated).sum()),
        "calibrated_row_count": int(calibrated.sum()),
        "candidate_branch_counts": _counts(frame, "candidate_branch"),
        "input_candidate_branch_counts": _counts(frame, "mmuad_input_candidate_branch"),
        "source_counts": _counts(frame, "source"),
    }


def _annotate_branch(rows: pd.DataFrame, *, suffix: str, calibrated: bool) -> pd.DataFrame:
    out = pd.DataFrame(rows).copy()
    out["mmuad_input_candidate_branch"] = _input_branch(out)
    branch_suffix = _safe_label(suffix)
    out["mmuad_source_calibration_branch_suffix"] = branch_suffix
    out["candidate_branch"] = [f"{input_branch}:{branch_suffix}" for input_branch in out["mmuad_input_candidate_branch"].astype(str)]
    out["mmuad_candidate_branch_is_calibrated"] = bool(calibrated)
    original = out[["mmuad_original_x_m", "mmuad_original_y_m", "mmuad_original_z_m"]].to_numpy(float)
    current = out[["x_m", "y_m", "z_m"]].to_numpy(float)
    delta = current - original
    out["mmuad_calibration_dx_m"] = delta[:, 0]
    out["mmuad_calibration_dy_m"] = delta[:, 1]
    out["mmuad_calibration_dz_m"] = delta[:, 2]
    out["mmuad_calibration_displacement_m"] = np.linalg.norm(delta, axis=1)
    out["track_id"] = [
        f"{track_id}@{branch}"
        for track_id, branch in zip(out["mmuad_original_track_id"].astype(str), out["candidate_branch"].astype(str), strict=False)
    ]
    if not calibrated:
        out["mmuad_source_calibration_applied"] = False
    return out


def _input_branch(rows: pd.DataFrame) -> pd.Series:
    if "mmuad_input_candidate_branch" in rows.columns:
        values = rows["mmuad_input_candidate_branch"]
    elif "candidate_branch" in rows.columns:
        values = rows["candidate_branch"]
    elif "source" in rows.columns:
        values = rows["source"]
    else:
        values = pd.Series("candidate", index=rows.index)
    return values.fillna("candidate").astype(str).map(_safe_label)


def _safe_label(value: object) -> str:
    text = str(value).strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
    return text or "candidate"


def _counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns:
        return {}
    return {str(key): int(value) for key, value in rows[column].fillna("").astype(str).value_counts().sort_index().items()}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
