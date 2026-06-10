"""CLI for experimental MMUAD tracking adapters."""

from __future__ import annotations

import argparse
from pathlib import Path

from raft_uav.mmuad.calibration import load_calibration_json, transform_candidate_frame
from raft_uav.mmuad.io import (
    load_candidate_csv,
    load_point_cloud_csv_as_candidates,
    load_truth_csv,
    merge_candidate_frames,
)
from raft_uav.mmuad.sequence import discover_sequence_paths, load_sequence_export
from raft_uav.mmuad.submission import (
    compute_trajectory_metrics,
    write_submission_csv,
    write_submission_json,
)
from raft_uav.mmuad.tracker import TrackerConfig, run_mmuad_tracker, write_tracker_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track",
        description="experimental CVPR UG2+/MMUAD tracking-by-detection adapter",
    )
    parser.add_argument("--candidate-csv", action="append", type=Path, default=[])
    parser.add_argument("--point-cloud-csv", action="append", type=Path, default=[])
    parser.add_argument("--sequence-root", type=Path)
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--no-apply-calibration", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-source", default="lidar-cluster")
    parser.add_argument("--voxel-size-m", type=float, default=0.75)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    parser.add_argument("--soft-anchor-cap-m", type=float, default=2.0)
    parser.add_argument("--secondary-covariance-scale", type=float, default=25.0)
    parser.add_argument("--acceleration-std-mps2", type=float, default=8.0)
    parser.add_argument("--submission-csv", type=Path)
    parser.add_argument("--submission-json", type=Path)
    parser.add_argument("--submission-track-id", default="raft_uav_pp")
    args = parser.parse_args(argv)

    if args.sequence_root is not None:
        output = _run_sequence_root(args)
    else:
        output = _run_explicit_files(args)
    paths = write_tracker_output(output, args.output_dir)
    if args.submission_csv is not None:
        paths["submission_csv"] = str(
            write_submission_csv(
                output.estimates,
                args.submission_csv,
                track_id=args.submission_track_id,
            )
        )
    if args.submission_json is not None:
        paths["submission_json"] = str(
            write_submission_json(
                output.estimates,
                args.submission_json,
                track_id=args.submission_track_id,
            )
        )
    if not output.estimates.empty:
        extra_metrics = compute_trajectory_metrics(output.estimates)
        metrics_extra_json = args.output_dir / "mmuad_trajectory_metrics.json"
        import json

        metrics_extra_json.write_text(json.dumps(extra_metrics, indent=2), encoding="utf-8")
        paths["trajectory_metrics_json"] = str(metrics_extra_json)
    print("mmuad_track=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    pooled = output.metrics.get("pooled", {})
    if "mean_3d_m" in pooled:
        print(f"pooled_mean_3d_m={pooled['mean_3d_m']}")
        print(f"pooled_p95_3d_m={pooled['p95_3d_m']}")
        print(f"pooled_max_3d_m={pooled['max_3d_m']}")
    return 0


def _run_explicit_files(args: argparse.Namespace):
    frames = [load_candidate_csv(path) for path in args.candidate_csv]
    frames.extend(
        load_point_cloud_csv_as_candidates(
            path,
            source=args.point_source,
            voxel_size_m=args.voxel_size_m,
            min_points=args.min_cluster_points,
        )
        for path in args.point_cloud_csv
    )
    if not frames:
        raise SystemExit(
            "provide --sequence-root or at least one --candidate-csv/--point-cloud-csv"
        )
    candidates = merge_candidate_frames(frames)
    if args.calibration_json is not None and not args.no_apply_calibration:
        calibration = load_calibration_json(args.calibration_json)
        candidates = transform_candidate_frame(candidates, calibration)
    truth = load_truth_csv(args.truth_csv) if args.truth_csv else None
    return run_mmuad_tracker(
        candidates,
        truth,
        config=TrackerConfig(
            acceleration_std_mps2=args.acceleration_std_mps2,
            soft_anchor_cap_m=args.soft_anchor_cap_m,
            secondary_covariance_scale=args.secondary_covariance_scale,
        ),
    )


def _run_sequence_root(args: argparse.Namespace):
    sequences = discover_sequence_paths(args.sequence_root, sequence_glob=args.sequence_glob)
    if not sequences:
        raise SystemExit(f"no MMUAD sequence exports found under {args.sequence_root}")
    candidate_frames = []
    truth_frames = []
    for paths in sequences:
        candidates, truth, _calibration = load_sequence_export(
            paths,
            apply_calibration=not args.no_apply_calibration,
            voxel_size_m=args.voxel_size_m,
            min_cluster_points=args.min_cluster_points,
        )
        candidate_frames.append(candidates)
        if truth is not None:
            truth_frames.append(truth)
    candidates = merge_candidate_frames(candidate_frames)
    truth = merge_truth_frames(truth_frames) if truth_frames else None
    return run_mmuad_tracker(
        candidates,
        truth,
        config=TrackerConfig(
            acceleration_std_mps2=args.acceleration_std_mps2,
            soft_anchor_cap_m=args.soft_anchor_cap_m,
            secondary_covariance_scale=args.secondary_covariance_scale,
        ),
    )


def merge_truth_frames(frames):
    import pandas as pd

    from raft_uav.mmuad.schema import TruthFrame, normalize_truth_columns

    rows = [frame.rows for frame in frames if not frame.rows.empty]
    if not rows:
        return None
    return TruthFrame(normalize_truth_columns(pd.concat(rows, ignore_index=True)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
