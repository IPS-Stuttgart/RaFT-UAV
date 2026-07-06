"""Robust medoid ensemble for official MMUAD/UG2+ Track 5 submissions.

Weighted averaging can be hurt by one high-confidence but wrong trajectory.  This
module keeps the official Track 5 template unchanged but selects, for each
Sequence/Timestamp row, the submitted position closest to a robust consensus
center.  It is inference-safe: it uses only submitted predictions and an
optional template for preflight validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import validate_official_track5_submission
from raft_uav.mmuad.track5_submission_ensemble import SubmissionInput
from raft_uav.mmuad.track5_submission_ensemble import _classification_vote_margin
from raft_uav.mmuad.track5_submission_ensemble import _ensemble_classification
from raft_uav.mmuad.track5_submission_ensemble import _jsonable
from raft_uav.mmuad.track5_submission_ensemble import _submission_keys
from raft_uav.mmuad.track5_submission_ensemble import _weighted_spread_m
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission
from raft_uav.mmuad.track5_submission_ensemble import parse_submission_input
from raft_uav.mmuad.track5_submission_ensemble import write_track5_submission_ensemble_outputs

MEDOID_CENTER_POLICIES = ("weighted-median", "weighted-mean")


def ensemble_track5_submissions_medoid(
    submissions: Iterable[SubmissionInput],
    *,
    class_policy: str = "weighted-vote",
    center_policy: str = "weighted-median",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row-wise medoid estimates and diagnostics for official submissions."""

    if center_policy not in MEDOID_CENTER_POLICIES:
        raise ValueError(f"unsupported center_policy: {center_policy}")
    inputs = tuple(submissions)
    if not inputs:
        raise ValueError("at least one submission is required")
    frames: list[pd.DataFrame] = []
    for item in inputs:
        rows = load_track5_submission(item.path)
        rows["ensemble_input_label"] = item.label
        rows["ensemble_input_path"] = str(item.path)
        rows["ensemble_weight"] = float(item.weight)
        frames.append(rows)
    expected_count = len(frames[0])
    reference_keys = _submission_keys(frames[0])
    for item, rows in zip(inputs, frames, strict=True):
        if len(rows) != expected_count:
            raise ValueError(
                f"submission {item.label} has {len(rows)} rows; expected {expected_count}"
            )
        if _submission_keys(rows) != reference_keys:
            raise ValueError(f"submission {item.label} does not match the reference template keys")
    stacked = pd.concat(frames, ignore_index=True, sort=False)
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for (sequence_id, time_s), group in stacked.groupby(["sequence_id", "time_s"], sort=True):
        weights = group["ensemble_weight"].to_numpy(float)
        xyz = group[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
        center = _consensus_center(xyz, weights, policy=center_policy)
        selected_index = _medoid_index(xyz, weights, center)
        selected = group.iloc[int(selected_index)]
        selected_xyz = xyz[int(selected_index)]
        classification = _ensemble_classification(group, policy=class_policy)
        spread = _weighted_spread_m(xyz, weights, selected_xyz)
        center_spread = _weighted_spread_m(xyz, weights, center)
        records.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "source": "track5-submission-medoid-ensemble",
                "track_id": "track5-submission-medoid-ensemble",
                "state_x_m": float(selected_xyz[0]),
                "state_y_m": float(selected_xyz[1]),
                "state_z_m": float(selected_xyz[2]),
                "Classification": int(classification),
                "ensemble_input_count": int(len(group)),
                "ensemble_weight_sum": float(np.sum(weights)),
                "ensemble_position_spread_m": float(spread),
                "medoid_selected_label": str(selected["ensemble_input_label"]),
                "medoid_center_policy": center_policy,
            }
        )
        diagnostics.append(
            {
                "sequence_id": str(sequence_id),
                "time_s": float(time_s),
                "input_count": int(len(group)),
                "weight_sum": float(np.sum(weights)),
                "position_spread_m": float(spread),
                "center_position_spread_m": float(center_spread),
                "classification": int(classification),
                "classification_vote_margin": float(_classification_vote_margin(group)),
                "input_labels": ";".join(group["ensemble_input_label"].astype(str)),
                "selected_label": str(selected["ensemble_input_label"]),
                "center_policy": center_policy,
                "center_x_m": float(center[0]),
                "center_y_m": float(center[1]),
                "center_z_m": float(center[2]),
            }
        )
    estimates = pd.DataFrame.from_records(records).sort_values(["sequence_id", "time_s"])
    diagnostics_df = pd.DataFrame.from_records(diagnostics).sort_values(["sequence_id", "time_s"])
    return estimates.reset_index(drop=True), diagnostics_df.reset_index(drop=True)


def write_track5_submission_medoid_ensemble_outputs(
    *,
    estimates: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
    inputs: Iterable[SubmissionInput],
    center_policy: str,
    class_policy: str,
    template: pd.DataFrame | None = None,
    require_leaderboard_ready: bool = False,
) -> dict[str, Path]:
    """Write medoid ensemble artifacts and optional template validation."""

    paths = write_track5_submission_ensemble_outputs(
        estimates=estimates,
        diagnostics=diagnostics,
        output_dir=output_dir,
        template=template,
        manifest={
            "schema": "raft-uav-mmuad-track5-submission-medoid-ensemble-v1",
            "inputs": [
                {"label": item.label, "path": str(item.path), "weight": float(item.weight)}
                for item in inputs
            ],
            "class_policy": class_policy,
            "center_policy": center_policy,
            "selected_label_counts": diagnostics.get(
                "selected_label", pd.Series(dtype=str)
            ).value_counts().to_dict(),
        },
    )
    if require_leaderboard_ready:
        if template is None:
            raise SystemExit("--require-leaderboard-ready requires --template")
        validation = validate_official_track5_submission(paths["zip"], template=template, require_zip=True)
        if not validation.summary.get("leaderboard_ready", False):
            reasons = ", ".join(validation.summary.get("leaderboard_blocking_reasons", [])) or "unknown"
            raise SystemExit(f"medoid ensemble is not leaderboard-ready: {reasons}")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-medoid-track5-submissions",
        description="robust medoid ensemble of official MMUAD/UG2+ Track 5 submissions",
    )
    parser.add_argument(
        "--submission",
        action="append",
        default=[],
        metavar="LABEL=WEIGHT:PATH",
        help="official CSV/ZIP submission; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument(
        "--class-policy",
        choices=("weighted-vote", "first"),
        default="weighted-vote",
    )
    parser.add_argument(
        "--center-policy",
        choices=MEDOID_CENTER_POLICIES,
        default="weighted-median",
    )
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    if args.require_leaderboard_ready and args.template is None:
        parser.error(
            "--require-leaderboard-ready requires --template so leaderboard readiness can be validated"
        )
    if not args.submission:
        parser.error("provide at least one --submission")
    inputs = tuple(parse_submission_input(value) for value in args.submission)
    estimates, diagnostics = ensemble_track5_submissions_medoid(
        inputs,
        class_policy=args.class_policy,
        center_policy=args.center_policy,
    )
    template = None if args.template is None else pd.read_csv(args.template)
    paths = write_track5_submission_medoid_ensemble_outputs(
        estimates=estimates,
        diagnostics=diagnostics,
        output_dir=args.output_dir,
        inputs=inputs,
        class_policy=args.class_policy,
        center_policy=args.center_policy,
        template=template,
        require_leaderboard_ready=bool(args.require_leaderboard_ready),
    )
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    validation = manifest.get("validation") or {}
    print("mmuad_track5_submission_medoid_ensemble=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"center_policy={args.center_policy}")
    if validation:
        print(f"leaderboard_ready={validation.get('leaderboard_ready')}")
        print(f"codabench_upload_ready={validation.get('codabench_upload_ready')}")
    return 0


def _consensus_center(xyz: np.ndarray, weights: np.ndarray, *, policy: str) -> np.ndarray:
    if policy == "weighted-mean":
        return np.sum(weights[:, None] * xyz, axis=0) / float(np.sum(weights))
    if policy == "weighted-median":
        return np.asarray([_weighted_median(xyz[:, axis], weights) for axis in range(3)])
    raise ValueError(f"unsupported center_policy: {policy}")


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * float(np.sum(sorted_weights))
    return float(sorted_values[int(np.searchsorted(cumulative, cutoff, side="left"))])


def _medoid_index(xyz: np.ndarray, weights: np.ndarray, center: np.ndarray) -> int:
    distances = np.linalg.norm(xyz - center[None, :], axis=1)
    # Prefer high-weight inputs when two submissions are equally close to the
    # consensus center.
    order = np.lexsort((-weights, distances))
    return int(order[0])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
