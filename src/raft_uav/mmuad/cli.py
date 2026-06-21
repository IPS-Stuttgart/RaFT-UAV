"""CLI for experimental MMUAD tracking adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.archive import extract_mmuad_archive, is_supported_archive
from raft_uav.mmuad.calibration import load_calibration_auto, transform_candidate_frame
from raft_uav.mmuad.camera import (
    load_camera_detections_csv_as_candidates,
    load_camera_models_from_files,
)
from raft_uav.mmuad.classification import (
    infer_sequence_class_map_from_candidates,
    load_sequence_classifier_model,
    predict_sequence_classes_from_model,
    sequence_class_map_from_predictions,
    sequence_classifier_provenance,
    sequence_features_from_rows,
    write_sequence_classifier_provenance,
    write_sequence_class_map,
)
from raft_uav.mmuad.completion import (
    complete_results_to_truth_timestamps,
    completion_summary,
)
from raft_uav.mmuad.cluster_ranker import (
    build_cluster_feature_table,
    load_cluster_ranker_model,
    merge_cross_sensor_candidate_clusters,
    predict_cluster_scores,
    score_cluster_candidates,
    write_ranker_diagnostics,
)
from raft_uav.mmuad.evaluate import evaluate_submission_csv
from raft_uav.mmuad.evaluator import (
    evaluate_mmaud_results,
    load_evaluation_truth_file,
    load_mmaud_results_csv,
    load_mmaud_results_file,
    write_evaluation_artifacts,
)
from raft_uav.mmuad.inspect import (
    inspect_sequence_root,
    write_layout_report as write_sequence_layout_report,
)
from raft_uav.mmuad.io import (
    load_candidate_file,
    load_candidate_csv,
    load_point_cloud_file_as_candidates,
    load_point_cloud_csv_as_candidates,
    load_truth_file,
    merge_candidate_frames,
)
from raft_uav.mmuad.mot import MultiObjectTrackerConfig, run_mmuad_multi_object_tracker
from raft_uav.mmuad.native_ros import extract_native_rosbag_topic_map
from raft_uav.mmuad.layout import (
    inspect_mmuad_layout,
    write_layout_report as write_mmuad_layout_report,
)
from raft_uav.mmuad.radar import load_radar_polar_csv_as_candidates
from raft_uav.mmuad.rosbag_bridge import (
    inspect_rosbag,
    load_topic_map_exports,
    write_topic_map_template,
)
from raft_uav.mmuad.sequence import discover_sequence_paths, load_sequence_export
from raft_uav.mmuad.sequence import official_track5_timestamp_template
from raft_uav.mmuad.schema import normalize_truth_columns
from raft_uav.mmuad.splits import (
    filter_sequences_by_split,
    filter_sequences_by_split_folder,
    load_split_manifest,
    split_manifest_summary,
)
from raft_uav.mmuad.submission import (
    compute_trajectory_metrics,
    estimates_to_mmaud_results_frame,
    load_official_track5_template_file,
    load_sequence_class_map,
    validate_official_track5_submission,
    verify_official_upload_manifest,
    write_normalized_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
    write_submission_csv,
    write_mmaud_results_csv,
    write_submission_json,
    write_submission_zip,
    write_ug2_codabench_zip,
)
from raft_uav.mmuad.tracker import TrackerConfig, run_mmuad_tracker, write_tracker_output
from raft_uav.mmuad.tracker import TrackerOutput, compute_metrics
from raft_uav.mmuad.trajectory_completion import (
    TrajectoryCompletionConfig,
    complete_and_smooth_estimates,
    write_trajectory_completion_diagnostics,
)


DEFAULT_UG2_OFFICIAL_CLASSIFICATION = "2"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track",
        description="experimental CVPR UG2+/MMUAD tracking-by-detection adapter",
    )
    parser.add_argument("--candidate-csv", action="append", type=Path, default=[])
    parser.add_argument("--candidate-file", action="append", type=Path, default=[])
    parser.add_argument("--inspect-root", type=Path)
    parser.add_argument("--layout-report-json", type=Path)
    parser.add_argument("--layout-report-csv", type=Path)
    parser.add_argument("--evaluate-submission-csv", type=Path)
    parser.add_argument("--validate-ug2-official-codabench-zip", type=Path)
    parser.add_argument("--normalize-ug2-official-submission", type=Path)
    parser.add_argument("--normalized-ug2-official-codabench-zip", type=Path)
    parser.add_argument("--normalized-ug2-official-results-csv", type=Path)
    parser.add_argument("--official-normalization-json", type=Path)
    parser.add_argument("--official-validation-json", type=Path)
    parser.add_argument("--official-validation-rows-csv", type=Path)
    parser.add_argument("--official-upload-manifest-json", type=Path)
    parser.add_argument("--verify-official-upload-manifest", type=Path)
    parser.add_argument("--official-upload-manifest-verification-json", type=Path)
    parser.add_argument("--official-validation-template-csv", type=Path)
    parser.add_argument("--official-validation-template-file", type=Path)
    parser.add_argument("--official-validation-timestamp-tolerance-s", type=float, default=1.0e-6)
    parser.add_argument("--evaluate-truth-csv", type=Path)
    parser.add_argument("--evaluate-truth-file", type=Path)
    parser.add_argument("--evaluation-json", type=Path)
    parser.add_argument("--evaluation-max-time-delta-s", type=float, default=0.5)
    parser.add_argument(
        "--evaluation-protocol",
        choices=("nearest-time", "public-track5"),
        default="nearest-time",
        help=(
            "local result metric protocol: nearest-time diagnostic or public "
            "Track 5 timestamp-aligned MSE/classification metrics"
        ),
    )
    parser.add_argument(
        "--evaluation-timestamp-tolerance-s",
        type=float,
        default=1.0e-6,
        help="timestamp tolerance for --evaluation-protocol public-track5",
    )
    parser.add_argument(
        "--evaluation-require-complete-track5",
        action="store_true",
        help=(
            "with --evaluation-protocol public-track5, exit nonzero when the "
            "metric grid or official upload package is not leaderboard-ready"
        ),
    )
    parser.add_argument("--point-cloud-csv", action="append", type=Path, default=[])
    parser.add_argument("--point-cloud-file", action="append", type=Path, default=[])
    parser.add_argument("--radar-polar-csv", action="append", type=Path, default=[])
    parser.add_argument("--radar-polar-file", action="append", type=Path, default=[])
    parser.add_argument("--radar-polar-source", default="radar-polar")
    parser.add_argument(
        "--radar-azimuth-convention",
        choices=(
            "north-clockwise",
            "east-counterclockwise",
            "east-clockwise",
            "x-forward-left-positive",
        ),
        default="north-clockwise",
    )
    parser.add_argument("--radar-angle-unit", choices=("deg", "rad"), default="deg")
    parser.add_argument("--radar-polar-range-std-m", type=float, default=2.0)
    parser.add_argument("--radar-polar-angle-std-deg", type=float, default=2.0)
    parser.add_argument("--radar-polar-z-std-m", type=float, default=5.0)
    parser.add_argument("--camera-detections-csv", action="append", type=Path, default=[])
    parser.add_argument("--camera-detections-file", action="append", type=Path, default=[])
    parser.add_argument("--camera-calibration-file", action="append", type=Path, default=[])
    parser.add_argument(
        "--camera-source",
        help="camera model/source id for explicit camera detection files without a source column",
    )
    parser.add_argument("--camera-fixed-depth-m", type=float)
    parser.add_argument("--camera-std-xy-m", type=float, default=5.0)
    parser.add_argument("--camera-std-z-m", type=float, default=10.0)
    parser.add_argument("--sequence-root", type=Path)
    parser.add_argument(
        "--sequence-root-archive-extract-dir",
        type=Path,
        help=(
            "directory for extracting ZIP/TAR sequence-root archives before "
            "normal MMUAD sequence discovery; defaults under --output-dir"
        ),
    )
    parser.add_argument("--inspect-layout-only", action="store_true")
    parser.add_argument("--rosbag-path", type=Path)
    parser.add_argument("--rosbag-report-json", type=Path)
    parser.add_argument("--topic-map-template-json", type=Path)
    parser.add_argument(
        "--topic-map-template-mode",
        choices=("export", "native"),
        default="export",
        help=(
            "write CSV-export topic-map templates or native extraction templates "
            "for --topic-map-template-json"
        ),
    )
    parser.add_argument(
        "--topic-map-file",
        "--topic-map-json",
        dest="topic_map_file",
        type=Path,
        help="topic-map metadata file in JSON or YAML form",
    )
    parser.add_argument("--topic-map-base-dir", type=Path)
    parser.add_argument("--native-ros-extract-output-dir", type=Path)
    parser.add_argument(
        "--native-ros-auto-topic-map",
        action="store_true",
        help=(
            "in sequence-root mode, inspect ROS-only sequence folders, write a "
            "native topic-map template, and run native extraction from it"
        ),
    )
    parser.add_argument(
        "--native-ros-auto-topic-map-dir",
        type=Path,
        help=(
            "directory for generated sequence-root native topic maps and ROS "
            "inspection reports; defaults under --output-dir"
        ),
    )
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split-name")
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--truth-file", type=Path)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--calibration-file", type=Path, help="JSON/YAML/TXT calibration interchange file")
    parser.add_argument("--no-apply-calibration", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tracker-mode", choices=("single-uav", "multi-object"), default="single-uav")
    parser.add_argument("--mot-max-association-distance-m", type=float, default=15.0)
    parser.add_argument("--mot-max-track-age-s", type=float, default=1.5)
    parser.add_argument("--point-source", default="lidar-cluster")
    parser.add_argument("--voxel-size-m", type=float, default=0.75)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    parser.add_argument("--cluster-ranker-model-json", type=Path)
    parser.add_argument("--cluster-ranker-previous-states-csv", type=Path)
    parser.add_argument("--cluster-ranker-image-evidence-csv", type=Path)
    parser.add_argument("--cluster-ranker-scored-candidates-csv", type=Path)
    parser.add_argument("--cluster-ranker-score-features-csv", type=Path)
    parser.add_argument("--cluster-ranker-merged-candidates-csv", type=Path)
    parser.add_argument("--cluster-ranker-keep-confidence", action="store_true")
    parser.add_argument("--cluster-ranker-cross-sensor-time-window-s", type=float, default=0.05)
    parser.add_argument("--cluster-ranker-cross-sensor-distance-gate-m", type=float, default=5.0)
    parser.add_argument(
        "--trajectory-completion-mode",
        choices=(
            "none",
            "gap-interpolation",
            "fixed-lag",
            "constant-velocity",
            "constant-acceleration",
        ),
        default="none",
    )
    parser.add_argument("--trajectory-completion-max-gap-s", type=float, default=1.0)
    parser.add_argument("--trajectory-smoothing-lag-s", type=float, default=1.0)
    parser.add_argument("--trajectory-smoothing-blend", type=float, default=1.0)
    parser.add_argument(
        "--trajectory-speed-gate-mps",
        type=float,
        default=0.0,
        help="mark isolated trajectory points that imply speeds above this gate; 0 disables",
    )
    parser.add_argument(
        "--trajectory-outlier-replacement",
        choices=("none", "local-linear"),
        default="none",
        help="replace speed-gated interior outliers before smoothing",
    )
    parser.add_argument("--trajectory-outlier-replacement-max-gap-s", type=float)
    parser.add_argument("--trajectory-completion-no-truth-timestamps", action="store_true")
    parser.add_argument("--trajectory-completion-no-infer-grid", action="store_true")
    parser.add_argument("--soft-anchor-cap-m", type=float, default=2.0)
    parser.add_argument("--secondary-covariance-scale", type=float, default=25.0)
    parser.add_argument("--acceleration-std-mps2", type=float, default=8.0)
    parser.add_argument(
        "--selection-motion-weight",
        type=float,
        default=1.0,
        help="greedy path cost weight for motion speed after the first selected candidate",
    )
    parser.add_argument(
        "--selection-confidence-weight",
        type=float,
        default=0.0,
        help="greedy path reward for per-frame normalized candidate confidence/ranker score",
    )
    parser.add_argument(
        "--selection-mobility-weight",
        type=float,
        default=0.0,
        help="greedy path reward for the unsupervised mobility prior after the first candidate",
    )
    parser.add_argument(
        "--selection-source-priority-weight",
        type=float,
        default=0.0,
        help="greedy path cost weight for configured source priority after the first candidate",
    )
    parser.add_argument(
        "--selection-speed-scale-mps",
        type=float,
        default=20.0,
        help="speed scale used to normalize motion cost in greedy path selection",
    )
    parser.add_argument("--submission-csv", type=Path)
    parser.add_argument("--submission-json", type=Path)
    parser.add_argument("--submission-zip", type=Path)
    parser.add_argument("--submission-track-id", default="raft_uav_pp")
    parser.add_argument("--ug2-results-csv", type=Path)
    parser.add_argument("--ug2-codabench-zip", type=Path)
    parser.add_argument("--ug2-official-results-csv", type=Path)
    parser.add_argument("--ug2-official-codabench-zip", type=Path)
    parser.add_argument(
        "--ug2-official-validate-on-write",
        action="store_true",
        help=(
            "validate the freshly written official Track 5 ZIP and write "
            "preflight diagnostics before returning"
        ),
    )
    parser.add_argument("--ug2-class-name", default="unknown")
    parser.add_argument(
        "--ug2-official-classification",
        default=DEFAULT_UG2_OFFICIAL_CLASSIFICATION,
        help=(
            "default integer Classification id for official Track 5 result rows "
            f"(default: {DEFAULT_UG2_OFFICIAL_CLASSIFICATION}, the public "
            "validation global-majority prior); numeric --ug2-class-map-file "
            "values override this per sequence"
        ),
    )
    parser.add_argument(
        "--ug2-official-invalid-row-policy",
        choices=("raise", "drop"),
        default="raise",
        help=(
            "how official Track 5 writers handle non-finite timestamps or "
            "positions; use drop only for diagnostic exports"
        ),
    )
    parser.add_argument(
        "--ug2-official-complete-to-sequence-timestamps",
        action="store_true",
        help=(
            "resample official Track 5 output rows to timestamps discovered in "
            "--sequence-root modality folders before writing official CSV/ZIP"
        ),
    )
    parser.add_argument(
        "--ug2-official-timestamp-source",
        choices=(
            "ground-truth-or-all",
            "all-modalities",
            "ground-truth",
            "image",
            "lidar-360",
            "livox-avia",
            "radar-enhance-pcl",
        ),
        default="ground-truth-or-all",
        help=(
            "which public Track 5 folder timestamps to use with "
            "--ug2-official-complete-to-sequence-timestamps"
        ),
    )
    parser.add_argument(
        "--ug2-class-map-file",
        "--ug2-class-map-csv",
        dest="ug2_class_map_file",
        type=Path,
        help="sequence-to-UAV-type class map in CSV, JSON, or YAML form",
    )
    parser.add_argument(
        "--sequence-classifier",
        type=Path,
        help=(
            "joblib model from raft-uav-mmuad-train-sequence-classifier; predicted "
            "sequence classes are copied to every Track 5 timestamp for that sequence"
        ),
    )
    parser.add_argument(
        "--sequence-classifier-predictions-csv",
        type=Path,
        help="optional CSV for per-sequence classifier predictions",
    )
    parser.add_argument(
        "--sequence-classifier-feature-report",
        type=Path,
        help="optional CSV for features used by --sequence-classifier at apply time",
    )
    parser.add_argument(
        "--sequence-classifier-provenance-json",
        type=Path,
        help="optional JSON with classifier provenance for scorecards",
    )
    parser.add_argument("--infer-ug2-class-map-from-candidates", action="store_true")
    parser.add_argument("--inferred-class-map-csv", type=Path)
    parser.add_argument("--classification-min-confidence", type=float, default=0.0)
    parser.add_argument("--complete-results-to-truth-csv", type=Path)
    parser.add_argument("--complete-results-to-truth-file", type=Path)
    parser.add_argument("--completed-results-csv", type=Path)
    parser.add_argument("--completed-results-diagnostics-csv", type=Path)
    parser.add_argument("--completed-ug2-codabench-zip", type=Path)
    parser.add_argument("--completion-max-interpolation-gap-s", type=float, default=1.0)
    parser.add_argument("--completion-extrapolation", choices=("hold", "nan"), default="hold")
    parser.add_argument("--evaluate-results-csv", type=Path)
    parser.add_argument("--evaluate-results-zip", type=Path)
    parser.add_argument("--evaluation-rows-csv", type=Path)
    parser.add_argument(
        "--evaluation-class-map-file",
        "--evaluation-class-map-csv",
        dest="evaluation_class_map_file",
        type=Path,
        help="sequence-to-UAV-type truth class map in CSV, JSON, or YAML form",
    )
    args = parser.parse_args(argv)

    if args.verify_official_upload_manifest is not None:
        return _run_official_upload_manifest_verification(args)

    if args.normalize_ug2_official_submission is not None:
        _maybe_prepare_sequence_root_archive(args)
        return _run_official_submission_normalization(args)

    if args.validate_ug2_official_codabench_zip is not None:
        _maybe_prepare_sequence_root_archive(args)
        return _run_official_submission_validation(args)

    if args.rosbag_path is not None:
        report = inspect_rosbag(args.rosbag_path)
        if args.rosbag_report_json is not None:
            args.rosbag_report_json.parent.mkdir(parents=True, exist_ok=True)
            args.rosbag_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.topic_map_template_json is not None:
            write_topic_map_template(
                report,
                args.topic_map_template_json,
                template_mode=args.topic_map_template_mode,
            )
        if args.native_ros_extract_output_dir is not None:
            if args.topic_map_file is None:
                raise SystemExit(
                    "--native-ros-extract-output-dir requires --topic-map-file/--topic-map-json"
                )
            extracted = extract_native_rosbag_topic_map(
                bag_path=args.rosbag_path,
                topic_map_json=args.topic_map_file,
                output_dir=args.native_ros_extract_output_dir,
                voxel_size_m=args.voxel_size_m,
                min_points=args.min_cluster_points,
            )
            _maybe_set_official_validation_template_from_truth(args, extracted.truth)
            _maybe_set_official_validation_template_from_native_images(args, extracted)
            if extracted.candidates is None or extracted.candidates.rows.empty:
                manifest_path = (
                    args.native_ros_extract_output_dir
                    / "native_ros_extraction_manifest.json"
                )
                raise SystemExit(
                    "native ROS extraction produced no candidate rows; inspect "
                    f"{manifest_path} and update the topic map to include "
                    "candidate-bearing topics"
                )
            if args.infer_ug2_class_map_from_candidates:
                args._inferred_class_map = infer_sequence_class_map_from_candidates(
                    extracted.candidates,
                    min_confidence=args.classification_min_confidence,
                    default_class=args.ug2_class_name,
                )
            output = _run_tracker_for_mode(args, extracted.candidates, extracted.truth)
            return _write_tracking_artifacts(
                args,
                output,
                extra_paths={
                    "native_ros_manifest_json": str(
                        args.native_ros_extract_output_dir
                        / "native_ros_extraction_manifest.json"
                    )
                },
            )
        if args.topic_map_file is None:
            print("mmuad_rosbag_inspection=ok")
            print(f"topic_count={len(report.get('topics', []))}")
            return 0

    evaluation_results = _evaluation_results_path(args)
    if evaluation_results is not None:
        evaluation_truth = _evaluation_truth_path(args)
        if evaluation_truth is None:
            raise SystemExit(
                "--evaluate-results-csv/--evaluate-results-zip requires "
                "--evaluate-truth-csv or --evaluate-truth-file"
            )
        truth_frame = load_evaluation_truth_file(evaluation_truth)
        result = evaluate_mmaud_results(
            load_mmaud_results_file(evaluation_results),
            truth_frame,
            max_time_delta_s=args.evaluation_max_time_delta_s,
            metric_protocol=args.evaluation_protocol,
            timestamp_tolerance_s=args.evaluation_timestamp_tolerance_s,
            class_map_path=args.evaluation_class_map_file,
        )
        if args.evaluation_protocol == "public-track5":
            _attach_public_track5_submission_validation(
                args,
                result,
                evaluation_results=evaluation_results,
                truth_frame=truth_frame,
            )
        paths = write_evaluation_artifacts(
            result,
            summary_json=args.evaluation_json or (args.output_dir / "mmuad_local_evaluation.json"),
            rows_csv=args.evaluation_rows_csv,
        )
        if args.evaluation_require_complete_track5:
            _require_complete_track5_evaluation(args, result["summary"])
        print("mmuad_local_evaluation=ok")
        for name, path in paths.items():
            print(f"{name}={path}")
        pooled = result["summary"].get("pooled", {})
        if "mean_3d_m" in pooled:
            print(f"mean_3d_m={pooled['mean_3d_m']}")
        return 0

    if args.inspect_root is not None:
        return _run_inspect(args)
    if args.evaluate_submission_csv is not None:
        return _run_submission_evaluation(args)
    if args.inspect_layout_only:
        if args.sequence_root is None:
            raise SystemExit("--inspect-layout-only requires --sequence-root")
        report_path = args.layout_report_json or (args.output_dir / "mmuad_layout_report.json")
        summary = inspect_mmuad_layout(args.sequence_root)
        written = write_mmuad_layout_report(summary, report_path)
        print("mmuad_layout_inspection=ok")
        print(f"layout_report_json={written}")
        print(f"file_count={summary['file_count']}")
        return 0

    if args.sequence_root is not None:
        _maybe_prepare_sequence_root_archive(args)
        output = _run_sequence_root(args)
    else:
        output = _run_explicit_files(args)
    return _write_tracking_artifacts(args, output)


def _write_tracking_artifacts(
    args: argparse.Namespace,
    output,
    *,
    extra_paths: dict[str, str] | None = None,
) -> int:
    paths = write_tracker_output(output, args.output_dir)
    if extra_paths:
        paths.update(extra_paths)
    archive_manifest = getattr(args, "_sequence_root_archive_manifest_json", None)
    if archive_manifest is not None:
        paths["sequence_root_archive_manifest_json"] = str(archive_manifest)
    native_sequence_manifests = getattr(args, "_native_ros_sequence_manifests_json", None)
    if native_sequence_manifests is not None:
        paths["native_ros_sequence_manifests_json"] = str(native_sequence_manifests)
    cluster_ranker_paths = getattr(args, "_cluster_ranker_paths", None)
    if cluster_ranker_paths:
        paths.update(cluster_ranker_paths)
    trajectory_completion_paths = getattr(args, "_trajectory_completion_paths", None)
    if trajectory_completion_paths:
        paths.update(trajectory_completion_paths)
    sequence_classifier_paths = getattr(args, "_sequence_classifier_paths", None)
    if sequence_classifier_paths:
        paths.update(sequence_classifier_paths)
    explicit_class_map = load_sequence_class_map(args.ug2_class_map_file)
    inferred_class_map = getattr(args, "_inferred_class_map", {})
    sequence_classifier_class_map = getattr(args, "_sequence_classifier_class_map", {})
    # Prefer explicit class-map files when multiple sources are provided.
    class_map = {**inferred_class_map, **sequence_classifier_class_map, **explicit_class_map}
    if args.inferred_class_map_csv is not None and inferred_class_map:
        paths["inferred_class_map_csv"] = str(
            write_sequence_class_map(inferred_class_map, args.inferred_class_map_csv)
        )
    official_output_estimates = output.estimates
    if args.ug2_official_complete_to_sequence_timestamps:
        completion, template = _complete_to_official_sequence_timestamps(
            args,
            output.estimates,
            class_map=class_map,
        )
        official_output_estimates = completion.rows
        diagnostics_path = args.output_dir / "mmuad_official_timestamp_completion_rows.csv"
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        completion.diagnostics.to_csv(diagnostics_path, index=False)
        summary = completion_summary(completion, requested_count=len(template))
        summary["timestamp_source"] = getattr(
            args,
            "_official_completion_template_source",
            args.ug2_official_timestamp_source,
        )
        summary_path = args.output_dir / "mmuad_official_timestamp_completion_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        paths["official_timestamp_completion_rows_csv"] = str(diagnostics_path)
        paths["official_timestamp_completion_summary_json"] = str(summary_path)
        args._official_validation_template = template
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
    if args.submission_zip is not None:
        paths["submission_zip"] = str(
            write_submission_zip(
                output.estimates,
                args.submission_zip,
                track_id=args.submission_track_id,
            )
        )
    if args.ug2_results_csv is not None:
        paths["ug2_results_csv"] = str(
            write_mmaud_results_csv(
                output.estimates,
                args.ug2_results_csv,
                class_name=args.ug2_class_name,
                class_map=class_map,
            )
        )
    if args.ug2_codabench_zip is not None:
        paths["ug2_codabench_zip"] = str(
            write_ug2_codabench_zip(
                output.estimates,
                args.ug2_codabench_zip,
                class_name=args.ug2_class_name,
                class_map=class_map,
            )
        )
    if args.ug2_official_results_csv is not None:
        paths["ug2_official_results_csv"] = str(
            write_official_mmaud_results_csv(
                official_output_estimates,
                args.ug2_official_results_csv,
                classification=args.ug2_official_classification,
                class_map=class_map,
                invalid_row_policy=args.ug2_official_invalid_row_policy,
            )
        )
    if args.ug2_official_codabench_zip is not None:
        paths["ug2_official_codabench_zip"] = str(
            write_official_ug2_codabench_zip(
                official_output_estimates,
                args.ug2_official_codabench_zip,
                classification=args.ug2_official_classification,
                class_map=class_map,
                invalid_row_policy=args.ug2_official_invalid_row_policy,
            )
        )
    if args.ug2_official_validate_on_write:
        if args.ug2_official_codabench_zip is None:
            raise SystemExit(
                "--ug2-official-validate-on-write requires "
                "--ug2-official-codabench-zip"
            )
        validation_paths = _validate_written_official_zip(args)
        paths.update(validation_paths)
    completion_truth_path = _completion_truth_path(args)
    if completion_truth_path is not None:
        if args.completed_results_csv is None and args.completed_ug2_codabench_zip is None:
            raise SystemExit(
                "--complete-results-to-truth-csv/--complete-results-to-truth-file "
                "requires --completed-results-csv or --completed-ug2-codabench-zip"
            )
        template_truth = load_truth_file(completion_truth_path)
        base_results = write_mmaud_results_csv(
            output.estimates,
            args.output_dir / "mmaud_results_for_completion.csv",
            class_name=args.ug2_class_name,
            class_map=class_map,
        )
        completion = complete_results_to_truth_timestamps(
            load_mmaud_results_csv(base_results),
            template_truth,
            max_interpolation_gap_s=args.completion_max_interpolation_gap_s,
            extrapolation=args.completion_extrapolation,
        )
        if args.completed_results_csv is not None:
            args.completed_results_csv.parent.mkdir(parents=True, exist_ok=True)
            completion.rows.to_csv(args.completed_results_csv, index=False)
            paths["completed_results_csv"] = str(args.completed_results_csv)
        if args.completed_results_diagnostics_csv is not None:
            args.completed_results_diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
            completion.diagnostics.to_csv(args.completed_results_diagnostics_csv, index=False)
            paths["completed_results_diagnostics_csv"] = str(
                args.completed_results_diagnostics_csv
            )
        if args.completed_ug2_codabench_zip is not None:
            args.completed_ug2_codabench_zip.parent.mkdir(parents=True, exist_ok=True)
            from zipfile import ZIP_DEFLATED, ZipFile

            with ZipFile(
                args.completed_ug2_codabench_zip, "w", compression=ZIP_DEFLATED
            ) as archive:
                archive.writestr("mmaud_results.csv", completion.rows.to_csv(index=False))
            paths["completed_ug2_codabench_zip"] = str(
                args.completed_ug2_codabench_zip
            )
        summary = completion_summary(
            completion, requested_count=len(template_truth.rows)
        )
        summary_path = args.output_dir / "mmuad_completion_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        paths["completion_summary_json"] = str(summary_path)
    if not output.estimates.empty:
        extra_metrics = compute_trajectory_metrics(output.estimates)
        metrics_extra_json = args.output_dir / "mmuad_trajectory_metrics.json"
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



def _require_complete_track5_evaluation(
    args: argparse.Namespace,
    summary: dict,
) -> None:
    if args.evaluation_protocol != "public-track5":
        raise SystemExit(
            "--evaluation-require-complete-track5 requires "
            "--evaluation-protocol public-track5"
        )
    if summary.get("leaderboard_ready") is True:
        return
    reasons = summary.get("leaderboard_blocking_reasons") or ["unknown"]
    raise SystemExit(
        "public Track 5 evaluation is not leaderboard-ready: "
        + ", ".join(str(reason) for reason in reasons)
    )


def _attach_public_track5_submission_validation(
    args: argparse.Namespace,
    result: dict,
    *,
    evaluation_results: Path,
    truth_frame,
) -> None:
    """Fold official Track 5 package preflight into public-metric readiness."""

    validation = validate_official_track5_submission(
        evaluation_results,
        template=truth_frame.rows,
        timestamp_tolerance_s=args.evaluation_timestamp_tolerance_s,
        require_zip=Path(evaluation_results).suffix.lower() == ".zip",
    )
    summary = result["summary"]
    validation_summary = validation.summary
    summary["official_submission_validation"] = validation_summary
    summary["official_submission_valid"] = bool(validation_summary.get("valid"))
    summary["codabench_upload_ready"] = bool(
        validation_summary.get("codabench_upload_ready")
    )
    if validation_summary.get("codabench_upload_ready") is True:
        return

    summary["leaderboard_ready"] = False
    if validation_summary.get("valid") is not True:
        summary["score_valid_for_leaderboard"] = False
    existing = list(summary.get("leaderboard_blocking_reasons", []))
    for reason in _official_submission_validation_blocking_reasons(validation_summary):
        if reason not in existing:
            existing.append(reason)
    summary["leaderboard_blocking_reasons"] = existing


def _official_submission_validation_blocking_reasons(summary: dict) -> list[str]:
    reasons = []
    if summary.get("valid") is not True:
        reasons.append("official_submission_validation_failed")
    if not summary.get("codabench_upload_ready", False):
        reasons.append("official_upload_package_not_ready")
    if summary.get("is_zip") and not summary.get("contains_only_mmaud_results_csv", False):
        reasons.append("official_zip_members_invalid")
    if list(summary.get("columns", [])) != list(summary.get("expected_columns", [])):
        reasons.append("official_columns_invalid")
    invalid_counts = (
        int(summary.get("invalid_sequence_count", 0)),
        int(summary.get("invalid_timestamp_count", 0)),
        int(summary.get("invalid_position_count", 0)),
        int(summary.get("invalid_classification_count", 0)),
    )
    if any(count > 0 for count in invalid_counts):
        reasons.append("official_invalid_rows")
    if int(summary.get("duplicate_prediction_count", 0)) > 0:
        reasons.append("official_duplicate_predictions")
    if int(summary.get("missing_template_timestamp_count", 0)) > 0:
        reasons.append("official_missing_template_timestamps")
    if int(summary.get("extra_prediction_count", 0)) > 0:
        reasons.append("official_extra_predictions")
    if summary.get("errors"):
        reasons.append("official_validation_errors")
    return reasons


def _run_inspect(args: argparse.Namespace) -> int:
    json_path = args.layout_report_json or (args.output_dir / "mmuad_layout_report.json")
    csv_path = args.layout_report_csv or (args.output_dir / "mmuad_layout_report_files.csv")
    report = inspect_sequence_root(args.inspect_root, sequence_glob=args.sequence_glob)
    write_sequence_layout_report(report, json_path=json_path, csv_path=csv_path)
    print("mmuad_inspect=ok")
    print(f"layout_report_json={json_path}")
    print(f"layout_report_csv={csv_path}")
    print(f"sequence_count={report['sequence_count']}")
    print(f"file_count={report['file_count']}")
    return 0


def _maybe_prepare_sequence_root_archive(args: argparse.Namespace) -> None:
    if args.sequence_root is None:
        return
    if not is_supported_archive(args.sequence_root):
        return
    extract_parent = args.sequence_root_archive_extract_dir or (
        args.output_dir / "mmuad_sequence_root_archive"
    )
    manifest = extract_mmuad_archive(args.sequence_root, extract_parent)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "mmuad_sequence_root_archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    args._sequence_root_archive_manifest_json = manifest_path
    args._sequence_root_archive_original = args.sequence_root
    args.sequence_root = Path(manifest["extract_root"])


def _run_submission_evaluation(args: argparse.Namespace) -> int:
    evaluation_truth = _evaluation_truth_path(args)
    if evaluation_truth is None:
        raise SystemExit(
            "--evaluate-truth-csv or --evaluate-truth-file is required with --evaluate-submission-csv"
        )
    output_json = args.evaluation_json or (args.output_dir / "mmuad_submission_eval.json")
    metrics = evaluate_submission_csv(
        args.evaluate_submission_csv,
        evaluation_truth,
        max_time_delta_s=args.evaluation_max_time_delta_s,
        metric_protocol=args.evaluation_protocol,
        timestamp_tolerance_s=args.evaluation_timestamp_tolerance_s,
        class_map_path=args.evaluation_class_map_file,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.evaluation_require_complete_track5:
        _require_complete_track5_evaluation(args, metrics)
    print("mmuad_evaluate=ok")
    print(f"evaluation_json={output_json}")
    pooled = metrics.get("pooled", {})
    if "mean_3d_m" in pooled:
        print(f"pooled_mean_3d_m={pooled['mean_3d_m']}")
        print(f"pooled_p95_3d_m={pooled['p95_3d_m']}")
        print(f"pooled_max_3d_m={pooled['max_3d_m']}")
    return 0


def _run_official_submission_validation(args: argparse.Namespace) -> int:
    template = _official_submission_validation_template(args)
    validation = validate_official_track5_submission(
        args.validate_ug2_official_codabench_zip,
        template=template,
        timestamp_tolerance_s=args.official_validation_timestamp_tolerance_s,
        require_zip=True,
    )
    json_path = args.official_validation_json or (
        args.output_dir / "mmuad_official_submission_validation.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(validation.summary, indent=2), encoding="utf-8")
    paths = {"official_validation_json": str(json_path)}
    if args.official_validation_rows_csv is not None:
        args.official_validation_rows_csv.parent.mkdir(parents=True, exist_ok=True)
        validation.rows.to_csv(args.official_validation_rows_csv, index=False)
        paths["official_validation_rows_csv"] = str(args.official_validation_rows_csv)
    manifest_path = _write_official_upload_manifest(
        args,
        validation.summary,
        artifact_path=args.validate_ug2_official_codabench_zip,
        validation_json_path=json_path,
        validation_rows_path=args.official_validation_rows_csv,
    )
    paths["official_upload_manifest_json"] = str(manifest_path)
    print("mmuad_official_submission_validation=ok")
    print(f"valid={validation.summary['valid']}")
    print(f"leaderboard_ready={validation.summary['leaderboard_ready']}")
    print(f"codabench_upload_ready={validation.summary['codabench_upload_ready']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0 if validation.summary["codabench_upload_ready"] else 1


def _run_official_submission_normalization(args: argparse.Namespace) -> int:
    output_zip = args.normalized_ug2_official_codabench_zip or (
        args.output_dir / "normalized_official_submission.zip"
    )
    normalization = write_normalized_official_track5_submission(
        args.normalize_ug2_official_submission,
        output_zip,
        results_csv_path=args.normalized_ug2_official_results_csv,
    )
    normalization_json = args.official_normalization_json or (
        args.output_dir / "mmuad_official_submission_normalization.json"
    )
    normalization_json.parent.mkdir(parents=True, exist_ok=True)
    normalization_json.write_text(json.dumps(normalization, indent=2), encoding="utf-8")

    template = _official_submission_validation_template(args)
    validation = validate_official_track5_submission(
        output_zip,
        template=template,
        timestamp_tolerance_s=args.official_validation_timestamp_tolerance_s,
        require_zip=True,
    )
    json_path = args.official_validation_json or (
        args.output_dir / "mmuad_official_submission_validation.json"
    )
    rows_path = args.official_validation_rows_csv or (
        args.output_dir / "mmuad_official_submission_validation_rows.csv"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(validation.summary, indent=2), encoding="utf-8")
    validation.rows.to_csv(rows_path, index=False)
    manifest_path = _write_official_upload_manifest(
        args,
        validation.summary,
        artifact_path=output_zip,
        validation_json_path=json_path,
        validation_rows_path=rows_path,
    )
    print("mmuad_official_submission_normalization=ok")
    print(f"normalized_zip={output_zip}")
    print(f"normalization_json={normalization_json}")
    print(f"valid={validation.summary['valid']}")
    print(f"leaderboard_ready={validation.summary['leaderboard_ready']}")
    print(f"codabench_upload_ready={validation.summary['codabench_upload_ready']}")
    print(f"official_validation_json={json_path}")
    print(f"official_validation_rows_csv={rows_path}")
    print(f"official_upload_manifest_json={manifest_path}")
    return 0 if validation.summary["valid"] else 1


def _run_official_upload_manifest_verification(args: argparse.Namespace) -> int:
    verification = verify_official_upload_manifest(args.verify_official_upload_manifest)
    json_path = args.official_upload_manifest_verification_json or (
        args.output_dir / "mmuad_official_upload_manifest_verification.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print("mmuad_official_upload_manifest_verification=ok")
    print(f"valid={verification['valid']}")
    print(f"codabench_upload_ready={verification['codabench_upload_ready']}")
    print(f"official_upload_manifest_verification_json={json_path}")
    return 0 if verification["valid"] else 1


def _validate_written_official_zip(args: argparse.Namespace) -> dict[str, str]:
    template = getattr(args, "_official_validation_template", None)
    if template is None:
        template = _official_submission_validation_template(args)
    validation = validate_official_track5_submission(
        args.ug2_official_codabench_zip,
        template=template,
        timestamp_tolerance_s=args.official_validation_timestamp_tolerance_s,
        require_zip=True,
    )
    json_path = args.official_validation_json or (
        args.output_dir / "mmuad_official_submission_validation.json"
    )
    rows_path = args.official_validation_rows_csv or (
        args.output_dir / "mmuad_official_submission_validation_rows.csv"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(validation.summary, indent=2), encoding="utf-8")
    validation.rows.to_csv(rows_path, index=False)
    manifest_path = _write_official_upload_manifest(
        args,
        validation.summary,
        artifact_path=args.ug2_official_codabench_zip,
        validation_json_path=json_path,
        validation_rows_path=rows_path,
    )
    paths = {
        "official_validation_json": str(json_path),
        "official_validation_rows_csv": str(rows_path),
        "official_upload_manifest_json": str(manifest_path),
    }
    if not validation.summary["codabench_upload_ready"]:
        errors = "; ".join(str(item) for item in validation.summary.get("errors", []))
        reasons = ", ".join(
            str(item)
            for item in validation.summary.get("leaderboard_blocking_reasons", [])
        )
        detail_items = [item for item in (errors, reasons) if item]
        detail = f": {'; '.join(detail_items)}" if detail_items else ""
        raise SystemExit(f"written official Track 5 ZIP failed validation{detail}")
    return paths


def _write_official_upload_manifest(
    args: argparse.Namespace,
    summary: dict,
    *,
    artifact_path: Path,
    validation_json_path: Path,
    validation_rows_path: Path | None,
) -> Path:
    manifest_path = args.official_upload_manifest_json or (
        args.output_dir / "mmuad_official_upload_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _official_upload_manifest_payload(
        summary,
        artifact_path=artifact_path,
        validation_json_path=validation_json_path,
        validation_rows_path=validation_rows_path,
        classification_provenance=getattr(args, "_sequence_classifier_provenance", None),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _official_upload_manifest_payload(
    summary: dict,
    *,
    artifact_path: Path,
    validation_json_path: Path,
    validation_rows_path: Path | None,
    classification_provenance: dict | None = None,
) -> dict:
    sequence_summaries = {
        str(sequence_id): _official_upload_sequence_manifest(row)
        for sequence_id, row in dict(summary.get("sequences", {})).items()
    }
    blocking_sequences = [
        sequence_id
        for sequence_id, row in sequence_summaries.items()
        if not row.get("leaderboard_ready", False)
    ]
    manifest = {
        "schema": "raft-uav-mmuad-official-upload-manifest-v1",
        "artifact_path": str(artifact_path),
        "validation_json": str(validation_json_path),
        "validation_rows_csv": (
            str(validation_rows_path) if validation_rows_path is not None else None
        ),
        "artifact_exists": bool(summary.get("artifact_exists", False)),
        "artifact_size_bytes": _optional_int(summary.get("artifact_size_bytes")),
        "artifact_sha256": summary.get("artifact_sha256"),
        "mmaud_results_csv_size_bytes": _optional_int(
            summary.get("mmaud_results_csv_size_bytes")
        ),
        "mmaud_results_csv_compressed_size_bytes": _optional_int(
            summary.get("mmaud_results_csv_compressed_size_bytes")
        ),
        "mmaud_results_csv_crc32": summary.get("mmaud_results_csv_crc32"),
        "mmaud_results_csv_sha256": summary.get("mmaud_results_csv_sha256"),
        "is_zip": bool(summary.get("is_zip", False)),
        "codabench_upload_ready": bool(summary.get("codabench_upload_ready", False)),
        "leaderboard_ready": bool(summary.get("leaderboard_ready", False)),
        "score_valid_for_leaderboard": bool(
            summary.get("score_valid_for_leaderboard", False)
        ),
        "valid": bool(summary.get("valid", False)),
        "leaderboard_blocking_reasons": list(
            summary.get("leaderboard_blocking_reasons", [])
        ),
        "row_count": int(summary.get("row_count", 0) or 0),
        "valid_row_count": int(summary.get("valid_row_count", 0) or 0),
        "template_checked": bool(summary.get("template_checked", False)),
        "template_timestamp_count": _optional_int(
            summary.get("template_timestamp_count")
        ),
        "timestamp_tolerance_s": float(summary.get("timestamp_tolerance_s", 0.0)),
        "members": list(summary.get("members", [])),
        "columns": list(summary.get("columns", [])),
        "expected_columns": list(summary.get("expected_columns", [])),
        "sequence_count": len(sequence_summaries),
        "ready_sequence_count": int(
            sum(1 for row in sequence_summaries.values() if row.get("leaderboard_ready"))
        ),
        "blocking_sequence_count": len(blocking_sequences),
        "blocking_sequences": blocking_sequences,
        "sequences": sequence_summaries,
    }
    if classification_provenance:
        for key in (
            "classification_model_path",
            "classification_method",
            "classification_train_sequences",
            "classification_feature_columns",
            "classification_class_map",
            "classification_prediction_mode",
            "train_data_available",
        ):
            manifest[key] = classification_provenance.get(key)
    return manifest


def _official_upload_sequence_manifest(row: dict) -> dict:
    return {
        "leaderboard_ready": bool(row.get("leaderboard_ready", False)),
        "score_valid_for_leaderboard": bool(
            row.get("score_valid_for_leaderboard", False)
        ),
        "leaderboard_blocking_reasons": list(
            row.get("leaderboard_blocking_reasons", [])
        ),
        "template_checked": bool(row.get("template_checked", False)),
        "template_timestamp_count": _optional_int(row.get("template_timestamp_count")),
        "prediction_count": int(row.get("prediction_count", 0) or 0),
        "valid_prediction_count": int(row.get("valid_prediction_count", 0) or 0),
        "covered_template_timestamp_count": int(
            row.get("covered_template_timestamp_count", 0) or 0
        ),
        "missing_template_timestamp_count": int(
            row.get("missing_template_timestamp_count", 0) or 0
        ),
        "extra_prediction_count": int(row.get("extra_prediction_count", 0) or 0),
        "duplicate_prediction_count": int(
            row.get("duplicate_prediction_count", 0) or 0
        ),
        "invalid_sequence_count": int(row.get("invalid_sequence_count", 0) or 0),
        "invalid_timestamp_count": int(row.get("invalid_timestamp_count", 0) or 0),
        "invalid_position_count": int(row.get("invalid_position_count", 0) or 0),
        "invalid_classification_count": int(
            row.get("invalid_classification_count", 0) or 0
        ),
    }


def _optional_int(value) -> int | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return int(value)


def _official_submission_validation_template(args: argparse.Namespace) -> pd.DataFrame | None:
    explicit_template = _official_validation_template_path(args)
    if explicit_template is not None:
        return _load_official_validation_template_file(explicit_template)
    if args.sequence_root is None:
        return None
    sequences = discover_sequence_paths(args.sequence_root, sequence_glob=args.sequence_glob)
    if args.split_file is not None:
        if not args.split_name:
            raise SystemExit("--split-name is required when --split-file is provided")
        manifest = load_split_manifest(args.split_file)
        sequences = filter_sequences_by_split(sequences, manifest, args.split_name)
    elif args.split_name:
        sequences = filter_sequences_by_split_folder(
            sequences,
            args.sequence_root,
            args.split_name,
        )
    frames = [
        official_track5_timestamp_template(
            paths,
            timestamp_source=args.ug2_official_timestamp_source,
        ).rows
        for paths in sequences
    ]
    rows = [frame for frame in frames if not frame.empty]
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def _load_official_validation_template_file(path: Path) -> pd.DataFrame:
    try:
        return load_truth_file(path).rows
    except ValueError:
        try:
            return load_official_track5_template_file(path)
        except Exception as template_error:
            raise ValueError(
                "official validation template must be a normalized truth/template "
                "file or an official Track 5 CSV/ZIP with Sequence and Timestamp"
            ) from template_error


def _official_validation_template_path(args: argparse.Namespace) -> Path | None:
    if (
        args.official_validation_template_csv is not None
        and args.official_validation_template_file is not None
    ):
        raise SystemExit(
            "provide only one of --official-validation-template-csv "
            "or --official-validation-template-file"
        )
    return args.official_validation_template_file or args.official_validation_template_csv


def _maybe_set_official_validation_template_from_truth(
    args: argparse.Namespace,
    truth,
) -> None:
    if truth is None or getattr(args, "_official_validation_template", None) is not None:
        return
    if _official_validation_template_path(args) is not None:
        return
    if getattr(truth, "rows", pd.DataFrame()).empty:
        return
    args._official_validation_template = truth.rows


def _maybe_set_official_validation_template_from_native_images(
    args: argparse.Namespace,
    extracted,
) -> None:
    template = _native_image_timestamp_template_frame(extracted)
    if template is None:
        return
    _maybe_set_official_validation_template_from_frames(args, [template])
    _maybe_set_official_native_image_template_from_frames(args, [template])


def _maybe_set_official_validation_template_from_frames(
    args: argparse.Namespace,
    frames: list[pd.DataFrame],
) -> None:
    if getattr(args, "_official_validation_template", None) is not None:
        return
    if _official_validation_template_path(args) is not None:
        return
    rows = [frame for frame in frames if frame is not None and not frame.empty]
    if not rows:
        return
    args._official_validation_template = normalize_truth_columns(
        pd.concat(rows, ignore_index=True)
        .drop_duplicates(subset=["sequence_id", "time_s"])
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )


def _maybe_set_official_native_image_template_from_frames(
    args: argparse.Namespace,
    frames: list[pd.DataFrame],
) -> None:
    rows = [frame for frame in frames if frame is not None and not frame.empty]
    if not rows:
        return
    args._official_native_image_timestamp_template = normalize_truth_columns(
        pd.concat(rows, ignore_index=True)
        .drop_duplicates(subset=["sequence_id", "time_s"])
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )


def _native_image_timestamp_template_frame(extracted) -> pd.DataFrame | None:
    image_timestamps = getattr(extracted, "image_timestamps", None)
    if image_timestamps is None or getattr(image_timestamps, "empty", True):
        return None
    if not {"sequence_id", "time_s"}.issubset(set(image_timestamps.columns)):
        return None
    template = image_timestamps[["sequence_id", "time_s"]].copy()
    template["x_m"] = 0.0
    template["y_m"] = 0.0
    template["z_m"] = 0.0
    return template


def _run_explicit_files(args: argparse.Namespace):
    frames = [load_candidate_csv(path) for path in args.candidate_csv]
    frames.extend(load_candidate_file(path) for path in args.candidate_file)
    frames.extend(
        load_point_cloud_csv_as_candidates(
            path,
            source=args.point_source,
            voxel_size_m=args.voxel_size_m,
            min_points=args.min_cluster_points,
        )
        for path in args.point_cloud_csv
    )
    frames.extend(
        load_point_cloud_file_as_candidates(
            path,
            source=args.point_source,
            voxel_size_m=args.voxel_size_m,
            min_points=args.min_cluster_points,
        )
        for path in args.point_cloud_file
    )
    radar_polar_files = list(args.radar_polar_csv) + list(args.radar_polar_file)
    frames.extend(
        load_radar_polar_csv_as_candidates(
            path,
            source=args.radar_polar_source,
            azimuth_convention=args.radar_azimuth_convention,
            angle_unit=args.radar_angle_unit,
            range_std_m=args.radar_polar_range_std_m,
            angle_std_deg=args.radar_polar_angle_std_deg,
            z_std_m=args.radar_polar_z_std_m,
        )
        for path in radar_polar_files
    )
    camera_detection_files = list(args.camera_detections_csv) + list(args.camera_detections_file)
    if camera_detection_files:
        if not args.camera_calibration_file:
            raise SystemExit(
                "--camera-detections-csv/--camera-detections-file "
                "requires --camera-calibration-file"
            )
        camera_models = load_camera_models_from_files(
            args.camera_calibration_file,
            source_hint_from_path=_camera_source_hint_from_path,
        )
        frames.extend(
            load_camera_detections_csv_as_candidates(
                path,
                camera_models=camera_models,
                source=args.camera_source,
                default_source=_camera_source_hint_from_path(path),
                fixed_depth_m=args.camera_fixed_depth_m,
                std_xy_m=args.camera_std_xy_m,
                std_z_m=args.camera_std_z_m,
            )
            for path in camera_detection_files
        )
    topic_truth = None
    if args.topic_map_file is not None:
        bundle = load_topic_map_exports(
            args.topic_map_file,
            base_dir=args.topic_map_base_dir,
        )
        frames.append(bundle.candidates)
        topic_truth = bundle.truth
    if not frames:
        raise SystemExit(
            "provide --sequence-root, --topic-map-file/--topic-map-json, or at least one "
            "--candidate-csv/--candidate-file/--point-cloud-csv"
        )
    candidates = merge_candidate_frames(frames)
    if args.infer_ug2_class_map_from_candidates:
        args._inferred_class_map = infer_sequence_class_map_from_candidates(
            candidates,
            min_confidence=args.classification_min_confidence,
            default_class=args.ug2_class_name,
        )
    calibration_path = args.calibration_file or args.calibration_json
    if calibration_path is not None and not args.no_apply_calibration:
        calibration = load_calibration_auto(calibration_path)
        candidates = transform_candidate_frame(candidates, calibration)
    _maybe_apply_sequence_classifier(args, candidates)
    candidates = _maybe_apply_cluster_ranker(args, candidates)
    truth_path = _explicit_truth_path(args)
    truth = load_truth_file(truth_path) if truth_path is not None else topic_truth
    _maybe_set_official_validation_template_from_truth(args, truth)
    return _run_tracker_for_mode(args, candidates, truth)


def _explicit_truth_path(args: argparse.Namespace) -> Path | None:
    if args.truth_csv is not None and args.truth_file is not None:
        raise SystemExit("provide only one of --truth-csv or --truth-file")
    return args.truth_file or args.truth_csv


def _evaluation_truth_path(args: argparse.Namespace) -> Path | None:
    if args.evaluate_truth_csv is not None and args.evaluate_truth_file is not None:
        raise SystemExit("provide only one of --evaluate-truth-csv or --evaluate-truth-file")
    return args.evaluate_truth_file or args.evaluate_truth_csv


def _evaluation_results_path(args: argparse.Namespace) -> Path | None:
    if args.evaluate_results_csv is not None and args.evaluate_results_zip is not None:
        raise SystemExit("provide only one of --evaluate-results-csv or --evaluate-results-zip")
    return args.evaluate_results_zip or args.evaluate_results_csv


def _completion_truth_path(args: argparse.Namespace) -> Path | None:
    if (
        args.complete_results_to_truth_csv is not None
        and args.complete_results_to_truth_file is not None
    ):
        raise SystemExit(
            "provide only one of --complete-results-to-truth-csv "
            "or --complete-results-to-truth-file"
        )
    return args.complete_results_to_truth_file or args.complete_results_to_truth_csv


def _complete_to_official_sequence_timestamps(
    args: argparse.Namespace,
    estimates,
    *,
    class_map: dict[str, str],
):
    template = _official_completion_template(args)
    base_results = estimates_to_mmaud_results_frame(
        estimates,
        class_name=args.ug2_class_name,
        class_map=class_map,
    )
    base_results["uav_type"] = [
        class_map.get(str(sequence_id), str(args.ug2_official_classification))
        for sequence_id in base_results["sequence_id"]
    ]
    completion = complete_results_to_truth_timestamps(
        base_results,
        template,
        max_interpolation_gap_s=args.completion_max_interpolation_gap_s,
        extrapolation=args.completion_extrapolation,
    )
    if completion.rows.empty:
        raise SystemExit(
            "official Track 5 timestamp completion produced no rows; "
            "relax --completion-extrapolation or provide candidate coverage"
        )
    return completion, template


def _official_completion_template(args: argparse.Namespace) -> pd.DataFrame:
    explicit_template = _official_validation_template_path(args)
    if explicit_template is not None:
        template = _load_official_validation_template_file(explicit_template)
        if template.empty:
            raise SystemExit(
                "official Track 5 completion template contains no usable timestamps"
            )
        args._official_completion_template_source = "official-validation-template"
        return template

    sequences = getattr(args, "_sequence_paths", None)
    if not sequences:
        native_template = getattr(args, "_official_native_image_timestamp_template", None)
        if native_template is not None and not native_template.empty:
            args._official_completion_template_source = "native-image-timestamps"
            return native_template
        raise SystemExit(
            "--ug2-official-complete-to-sequence-timestamps requires "
            "--sequence-root, --official-validation-template-file, or native image "
            "timestamp topics"
        )
    template_frames = [
        official_track5_timestamp_template(
            paths,
            timestamp_source=args.ug2_official_timestamp_source,
        )
        for paths in sequences
    ]
    rows = [frame.rows for frame in template_frames if not frame.rows.empty]
    if not rows:
        native_template = getattr(args, "_official_native_image_timestamp_template", None)
        if native_template is not None and not native_template.empty:
            args._official_completion_template_source = "native-image-timestamps"
            return native_template
        raise SystemExit(
            "no official Track 5 timestamps found in --sequence-root for "
            f"source {args.ug2_official_timestamp_source!r}"
        )
    args._official_completion_template_source = args.ug2_official_timestamp_source
    return pd.concat(rows, ignore_index=True)


def _camera_source_hint_from_path(path: Path) -> str | None:
    parent = Path(path).parent.name.replace(" ", "_").replace("-", "_")
    if _looks_like_camera_source_name(parent):
        return parent
    return None


def _looks_like_camera_source_name(name: str) -> bool:
    normalized = str(name).lower().replace("-", "_").replace(" ", "_")
    tokens = ("camera", "cam", "image", "images")
    for token in tokens:
        if (
            normalized == token
            or normalized.startswith(f"{token}_")
            or normalized.endswith(f"_{token}")
            or (normalized.startswith(token) and normalized[len(token) :].isdigit())
        ):
            return True
    return False


def _run_sequence_root(args: argparse.Namespace):
    sequences = discover_sequence_paths(args.sequence_root, sequence_glob=args.sequence_glob)
    if args.split_file is not None:
        if not args.split_name:
            raise SystemExit("--split-name is required when --split-file is provided")
        manifest = load_split_manifest(args.split_file)
        sequences = filter_sequences_by_split(sequences, manifest, args.split_name)
        split_summary = split_manifest_summary(manifest)
        print(f"split={args.split_name} sequences={split_summary[args.split_name]['count']}")
    elif args.split_name:
        sequences = filter_sequences_by_split_folder(
            sequences,
            args.sequence_root,
            args.split_name,
        )
        print(f"split={args.split_name} sequences={len(sequences)}")
    if not sequences:
        raise SystemExit(f"no MMUAD sequence exports found under {args.sequence_root}")
    args._sequence_paths = sequences
    candidate_frames = []
    truth_frames = []
    native_manifests = []
    native_image_template_frames = []
    for paths in sequences:
        if _should_run_native_sequence_export(args, paths):
            (
                candidates,
                truth,
                manifest_path,
                extracted,
                topic_map_file,
                auto_report_file,
                auto_generated,
            ) = _load_native_sequence_export(
                args,
                paths,
            )
            manifest_row = {
                "sequence_id": paths.sequence_id,
                "bag_path": str(paths.rosbag_paths[0]),
                "topic_map_file": str(topic_map_file),
                "manifest_json": str(manifest_path),
                "auto_topic_map_generated": bool(auto_generated),
            }
            if auto_report_file is not None:
                manifest_row["rosbag_report_json"] = str(auto_report_file)
            native_manifests.append(manifest_row)
            candidate_frames.append(candidates)
            if truth is not None:
                truth_frames.append(truth)
            image_template = _native_image_timestamp_template_frame(extracted)
            if image_template is not None:
                native_image_template_frames.append(image_template)
        if _has_exported_sequence_inputs(paths):
            candidates, truth, _calibration = load_sequence_export(
                paths,
                apply_calibration=not args.no_apply_calibration,
                voxel_size_m=args.voxel_size_m,
                min_cluster_points=args.min_cluster_points,
                radar_azimuth_convention=args.radar_azimuth_convention,
                radar_angle_unit=args.radar_angle_unit,
                radar_polar_range_std_m=args.radar_polar_range_std_m,
                radar_polar_angle_std_deg=args.radar_polar_angle_std_deg,
                radar_polar_z_std_m=args.radar_polar_z_std_m,
                camera_fixed_depth_m=args.camera_fixed_depth_m,
                camera_std_xy_m=args.camera_std_xy_m,
                camera_std_z_m=args.camera_std_z_m,
            )
            candidate_frames.append(candidates)
            if truth is not None:
                truth_frames.append(truth)
    if not candidate_frames:
        raise SystemExit(f"no MMUAD sequence exports found under {args.sequence_root}")
    candidates = merge_candidate_frames(candidate_frames)
    if args.infer_ug2_class_map_from_candidates:
        args._inferred_class_map = infer_sequence_class_map_from_candidates(
            candidates,
            min_confidence=args.classification_min_confidence,
            default_class=args.ug2_class_name,
        )
    truth = merge_truth_frames(truth_frames) if truth_frames else None
    _maybe_apply_sequence_classifier(args, candidates)
    candidates = _maybe_apply_cluster_ranker(args, candidates)
    if native_manifests:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_summary_path = args.output_dir / "native_ros_sequence_manifests.json"
        manifest_summary_path.write_text(json.dumps(native_manifests, indent=2), encoding="utf-8")
        args._native_ros_sequence_manifests_json = manifest_summary_path
        _maybe_set_official_validation_template_from_truth(args, truth)
        _maybe_set_official_validation_template_from_frames(
            args,
            native_image_template_frames,
        )
        _maybe_set_official_native_image_template_from_frames(
            args,
            native_image_template_frames,
        )
    return _run_tracker_for_mode(args, candidates, truth)


def _should_run_native_sequence_export(args: argparse.Namespace, paths) -> bool:
    if paths.native_topic_map_jsons:
        return True
    return bool(args.native_ros_auto_topic_map and paths.rosbag_paths)


def _has_exported_sequence_inputs(paths) -> bool:
    return any(
        (
            paths.candidate_csvs,
            paths.candidate_trajectory_files,
            paths.radar_polar_csvs,
            paths.camera_detection_csvs,
            paths.point_cloud_files,
            paths.topic_map_jsons,
        )
    )


def _load_native_sequence_export(args: argparse.Namespace, paths):
    if len(paths.rosbag_paths) != 1:
        raise SystemExit(
            f"sequence {paths.sequence_id!r} has {len(paths.rosbag_paths)} ROS recordings; "
            "use explicit --rosbag-path and --topic-map-file"
        )
    topic_map_json, auto_generated, auto_report_json = _native_sequence_topic_map(
        args,
        paths,
        bag_path=paths.rosbag_paths[0],
    )
    output_dir = _native_sequence_extract_output_dir(args, paths.sequence_id)
    extracted = extract_native_rosbag_topic_map(
        bag_path=paths.rosbag_paths[0],
        topic_map_json=topic_map_json,
        output_dir=output_dir,
        voxel_size_m=args.voxel_size_m,
        min_points=args.min_cluster_points,
    )
    if extracted.candidates is None or extracted.candidates.rows.empty:
        manifest_path = output_dir / "native_ros_extraction_manifest.json"
        raise SystemExit(
            "native ROS extraction produced no candidate rows for "
            f"sequence {paths.sequence_id!r}; inspect {manifest_path} and update "
            "the topic map to include candidate-bearing topics"
        )
    return (
        extracted.candidates,
        extracted.truth,
        output_dir / "native_ros_extraction_manifest.json",
        extracted,
        topic_map_json,
        auto_report_json,
        auto_generated,
    )


def _native_sequence_topic_map(
    args: argparse.Namespace,
    paths,
    *,
    bag_path: Path,
) -> tuple[Path, bool, Path | None]:
    if len(paths.native_topic_map_jsons) == 1:
        return paths.native_topic_map_jsons[0], False, None
    if len(paths.native_topic_map_jsons) > 1:
        raise SystemExit(
            f"sequence {paths.sequence_id!r} has {len(paths.native_topic_map_jsons)} "
            "native topic maps; use explicit --rosbag-path and --topic-map-file"
        )
    if not args.native_ros_auto_topic_map:
        raise SystemExit(
            f"sequence {paths.sequence_id!r} has no native topic map; provide one "
            "or pass --native-ros-auto-topic-map"
        )
    auto_dir = _native_sequence_auto_topic_map_dir(args, paths.sequence_id)
    auto_dir.mkdir(parents=True, exist_ok=True)
    report = inspect_rosbag(bag_path)
    if not report.get("topics"):
        reason = report.get("native_reader_error") or report.get("recommendation") or "no topics"
        raise SystemExit(
            "native ROS auto-topic-map found no inspectable topics for "
            f"sequence {paths.sequence_id!r}: {reason}"
        )
    report_path = auto_dir / "rosbag_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    topic_map_path = auto_dir / "topic_map_native.json"
    write_topic_map_template(report, topic_map_path, template_mode="native")
    return topic_map_path, True, report_path


def _native_sequence_auto_topic_map_dir(args: argparse.Namespace, sequence_id: str) -> Path:
    base = args.native_ros_auto_topic_map_dir or (
        args.output_dir / "native_ros_auto_topic_maps"
    )
    return base / _safe_path_component(sequence_id)


def _native_sequence_extract_output_dir(args: argparse.Namespace, sequence_id: str) -> Path:
    base = args.native_ros_extract_output_dir or (args.output_dir / "native_ros_extracted")
    return base / _safe_path_component(sequence_id)


def _safe_path_component(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value)
    )
    return safe or "sequence"


def _maybe_apply_sequence_classifier(args, candidates) -> None:
    if args.sequence_classifier is None:
        return
    model = load_sequence_classifier_model(args.sequence_classifier)
    features = sequence_features_from_rows(candidates.rows)
    predictions = predict_sequence_classes_from_model(model, features)
    class_map = sequence_class_map_from_predictions(predictions)
    provenance = sequence_classifier_provenance(
        model,
        model_path=args.sequence_classifier,
        class_map=class_map,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = args.sequence_classifier_predictions_csv or (
        args.output_dir / "mmuad_sequence_classifier_predictions.csv"
    )
    feature_report = args.sequence_classifier_feature_report or (
        args.output_dir / "mmuad_sequence_classifier_features.csv"
    )
    provenance_json = args.sequence_classifier_provenance_json or (
        args.output_dir / "mmuad_sequence_classifier_provenance.json"
    )
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_csv, index=False)
    feature_report.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(feature_report, index=False)
    write_sequence_classifier_provenance(provenance, provenance_json)
    args._sequence_classifier_class_map = class_map
    args._sequence_classifier_provenance = provenance
    args._sequence_classifier_paths = {
        "sequence_classifier_model": str(args.sequence_classifier),
        "sequence_classifier_predictions_csv": str(predictions_csv),
        "sequence_classifier_feature_report_csv": str(feature_report),
        "sequence_classifier_provenance_json": str(provenance_json),
    }


def _maybe_apply_cluster_ranker(args, candidates):
    if args.cluster_ranker_model_json is None:
        return candidates
    model = load_cluster_ranker_model(args.cluster_ranker_model_json)
    previous_states = (
        None
        if args.cluster_ranker_previous_states_csv is None
        else pd.read_csv(args.cluster_ranker_previous_states_csv)
    )
    image_evidence = (
        None
        if args.cluster_ranker_image_evidence_csv is None
        else pd.read_csv(args.cluster_ranker_image_evidence_csv)
    )
    merged = merge_cross_sensor_candidate_clusters(
        candidates,
        time_window_s=args.cluster_ranker_cross_sensor_time_window_s,
        distance_gate_m=args.cluster_ranker_cross_sensor_distance_gate_m,
    )
    frames = [candidates]
    if not merged.rows.empty:
        frames.append(merged)
    score_input = merge_candidate_frames(frames)
    paths: dict[str, str] = {
        "cluster_ranker_model_json": str(args.cluster_ranker_model_json),
    }
    if args.cluster_ranker_merged_candidates_csv is not None:
        args.cluster_ranker_merged_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.rows.to_csv(args.cluster_ranker_merged_candidates_csv, index=False)
        paths["cluster_ranker_merged_candidates_csv"] = str(
            args.cluster_ranker_merged_candidates_csv
        )
    scored = score_cluster_candidates(
        score_input,
        model,
        replace_confidence=not args.cluster_ranker_keep_confidence,
        previous_states=previous_states,
        image_evidence=image_evidence,
        cross_sensor_time_window_s=args.cluster_ranker_cross_sensor_time_window_s,
        cross_sensor_distance_gate_m=args.cluster_ranker_cross_sensor_distance_gate_m,
    )
    if args.cluster_ranker_scored_candidates_csv is not None:
        args.cluster_ranker_scored_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        scored.rows.to_csv(args.cluster_ranker_scored_candidates_csv, index=False)
        paths["cluster_ranker_scored_candidates_csv"] = str(
            args.cluster_ranker_scored_candidates_csv
        )
    if args.cluster_ranker_score_features_csv is not None:
        features = build_cluster_feature_table(
            score_input,
            previous_states=previous_states,
            image_evidence=image_evidence,
            cross_sensor_time_window_s=args.cluster_ranker_cross_sensor_time_window_s,
            cross_sensor_distance_gate_m=args.cluster_ranker_cross_sensor_distance_gate_m,
        )
        features["ranker_score"] = predict_cluster_scores(features, model)
        write_ranker_diagnostics(features, args.cluster_ranker_score_features_csv)
        paths["cluster_ranker_score_features_csv"] = str(
            args.cluster_ranker_score_features_csv
        )
    args._cluster_ranker_paths = paths
    return scored




def _run_tracker_for_mode(args, candidates, truth):
    if args.tracker_mode == "multi-object":
        output = run_mmuad_multi_object_tracker(
            candidates,
            truth,
            config=MultiObjectTrackerConfig(
                acceleration_std_mps2=args.acceleration_std_mps2,
                max_association_distance_m=args.mot_max_association_distance_m,
                max_track_age_s=args.mot_max_track_age_s,
            ),
        )
    else:
        output = run_mmuad_tracker(
            candidates,
            truth,
            config=TrackerConfig(
                acceleration_std_mps2=args.acceleration_std_mps2,
                soft_anchor_cap_m=args.soft_anchor_cap_m,
                secondary_covariance_scale=args.secondary_covariance_scale,
                selection_motion_weight=args.selection_motion_weight,
                selection_confidence_weight=args.selection_confidence_weight,
                selection_mobility_weight=args.selection_mobility_weight,
                selection_source_priority_weight=args.selection_source_priority_weight,
                selection_speed_scale_mps=args.selection_speed_scale_mps,
            ),
        )
    return _maybe_apply_trajectory_completion(args, output, truth)


def _maybe_apply_trajectory_completion(args, output, truth):
    if args.trajectory_completion_mode == "none" or output.estimates.empty:
        return output
    result = complete_and_smooth_estimates(
        output.estimates,
        None if truth is None else truth.rows,
        config=TrajectoryCompletionConfig(
            mode=args.trajectory_completion_mode,
            max_gap_s=args.trajectory_completion_max_gap_s,
            fixed_lag_s=args.trajectory_smoothing_lag_s,
            smoothing_blend=args.trajectory_smoothing_blend,
            include_truth_timestamps=not args.trajectory_completion_no_truth_timestamps,
            infer_missing_grid=not args.trajectory_completion_no_infer_grid,
            speed_gate_mps=args.trajectory_speed_gate_mps,
            outlier_replacement=args.trajectory_outlier_replacement,
            outlier_replacement_max_gap_s=args.trajectory_outlier_replacement_max_gap_s,
        ),
    )
    args._trajectory_completion_paths = write_trajectory_completion_diagnostics(
        result,
        args.output_dir,
    )
    metrics = _recompute_tracker_metrics(args, result.estimates, truth)
    return TrackerOutput(result.estimates, metrics, output.selected_tracklets)


def _recompute_tracker_metrics(args, estimates, truth):
    truth_rows = None if truth is None else truth.rows
    if args.tracker_mode == "multi-object":
        from raft_uav.mmuad.mot import compute_multi_object_metrics

        sequence_metrics = {}
        if not estimates.empty and "sequence_id" in estimates.columns:
            for sequence_id, sequence_estimates in estimates.groupby("sequence_id", sort=True):
                sequence_truth = None
                if truth_rows is not None and "sequence_id" in truth_rows.columns:
                    sequence_truth = truth_rows.loc[truth_rows["sequence_id"].astype(str) == str(sequence_id)]
                sequence_metrics[str(sequence_id)] = compute_multi_object_metrics(
                    sequence_estimates,
                    sequence_truth,
                )
        return {
            "sequences": sequence_metrics,
            "pooled": compute_multi_object_metrics(estimates, truth_rows),
        }
    sequence_metrics = {}
    if not estimates.empty and "sequence_id" in estimates.columns:
        for sequence_id, sequence_estimates in estimates.groupby("sequence_id", sort=True):
            sequence_metrics[str(sequence_id)] = compute_metrics(
                sequence_estimates,
                None if truth_rows is None else truth_rows.loc[truth_rows["sequence_id"].astype(str) == str(sequence_id)],
            )
    return {
        "sequences": sequence_metrics,
        "pooled": compute_metrics(estimates, truth_rows),
    }


def merge_truth_frames(frames):
    import pandas as pd

    from raft_uav.mmuad.schema import TruthFrame, normalize_truth_columns

    rows = [frame.rows for frame in frames if not frame.rows.empty]
    if not rows:
        return None
    return TruthFrame(normalize_truth_columns(pd.concat(rows, ignore_index=True)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
