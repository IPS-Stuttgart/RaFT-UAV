"""CLI for experimental MMUAD tracking adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.calibration import load_calibration_auto, transform_candidate_frame
from raft_uav.mmuad.camera import (
    load_camera_detections_csv_as_candidates,
    load_camera_models_from_files,
)
from raft_uav.mmuad.classification import (
    infer_sequence_class_map_from_candidates,
    write_sequence_class_map,
)
from raft_uav.mmuad.completion import (
    complete_results_to_truth_timestamps,
    completion_summary,
)
from raft_uav.mmuad.evaluate import evaluate_submission_csv
from raft_uav.mmuad.evaluator import (
    evaluate_mmaud_results,
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
from raft_uav.mmuad.splits import (
    filter_sequences_by_split,
    filter_sequences_by_split_folder,
    load_split_manifest,
    split_manifest_summary,
)
from raft_uav.mmuad.submission import (
    compute_trajectory_metrics,
    estimates_to_mmaud_results_frame,
    load_sequence_class_map,
    validate_official_track5_submission,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
    write_submission_csv,
    write_mmaud_results_csv,
    write_submission_json,
    write_submission_zip,
    write_ug2_codabench_zip,
)
from raft_uav.mmuad.tracker import TrackerConfig, run_mmuad_tracker, write_tracker_output


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
    parser.add_argument("--official-validation-json", type=Path)
    parser.add_argument("--official-validation-rows-csv", type=Path)
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
            "package is not leaderboard-ready"
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
    parser.add_argument("--inspect-layout-only", action="store_true")
    parser.add_argument("--rosbag-path", type=Path)
    parser.add_argument("--rosbag-report-json", type=Path)
    parser.add_argument("--topic-map-template-json", type=Path)
    parser.add_argument(
        "--topic-map-file",
        "--topic-map-json",
        dest="topic_map_file",
        type=Path,
        help="topic-map metadata file in JSON or YAML form",
    )
    parser.add_argument("--topic-map-base-dir", type=Path)
    parser.add_argument("--native-ros-extract-output-dir", type=Path)
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
    parser.add_argument("--soft-anchor-cap-m", type=float, default=2.0)
    parser.add_argument("--secondary-covariance-scale", type=float, default=25.0)
    parser.add_argument("--acceleration-std-mps2", type=float, default=8.0)
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
        default="0",
        help=(
            "default integer Classification id for official Track 5 result rows; "
            "numeric --ug2-class-map-file values override this per sequence"
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

    if args.validate_ug2_official_codabench_zip is not None:
        return _run_official_submission_validation(args)

    if args.rosbag_path is not None:
        report = inspect_rosbag(args.rosbag_path)
        if args.rosbag_report_json is not None:
            args.rosbag_report_json.parent.mkdir(parents=True, exist_ok=True)
            args.rosbag_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.topic_map_template_json is not None:
            write_topic_map_template(report, args.topic_map_template_json)
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
            if extracted.candidates is not None:
                output = _run_tracker_for_mode(args, extracted.candidates, extracted.truth)
                paths = write_tracker_output(output, args.output_dir)
                paths["native_ros_manifest_json"] = str(
                    args.native_ros_extract_output_dir / "native_ros_extraction_manifest.json"
                )
                print("mmuad_track=ok")
                for name, path in paths.items():
                    print(f"{name}={path}")
                return 0
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
        result = evaluate_mmaud_results(
            load_mmaud_results_file(evaluation_results),
            load_truth_file(evaluation_truth),
            max_time_delta_s=args.evaluation_max_time_delta_s,
            metric_protocol=args.evaluation_protocol,
            timestamp_tolerance_s=args.evaluation_timestamp_tolerance_s,
            class_map_path=args.evaluation_class_map_file,
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
        output = _run_sequence_root(args)
    else:
        output = _run_explicit_files(args)
    paths = write_tracker_output(output, args.output_dir)
    explicit_class_map = load_sequence_class_map(args.ug2_class_map_file)
    inferred_class_map = getattr(args, "_inferred_class_map", {})
    # Prefer explicit class-map files when both are provided.
    class_map = {**inferred_class_map, **explicit_class_map}
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
        summary["timestamp_source"] = args.ug2_official_timestamp_source
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
            )
        )
    if args.ug2_official_codabench_zip is not None:
        paths["ug2_official_codabench_zip"] = str(
            write_official_ug2_codabench_zip(
                official_output_estimates,
                args.ug2_official_codabench_zip,
                classification=args.ug2_official_classification,
                class_map=class_map,
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
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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
    print("mmuad_official_submission_validation=ok")
    print(f"valid={validation.summary['valid']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0 if validation.summary["valid"] else 1


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
    paths = {
        "official_validation_json": str(json_path),
        "official_validation_rows_csv": str(rows_path),
    }
    if not validation.summary["valid"]:
        errors = "; ".join(str(item) for item in validation.summary.get("errors", []))
        detail = f": {errors}" if errors else ""
        raise SystemExit(f"written official Track 5 ZIP failed validation{detail}")
    return paths


def _official_submission_validation_template(args: argparse.Namespace) -> pd.DataFrame | None:
    explicit_template = _official_validation_template_path(args)
    if explicit_template is not None:
        return load_truth_file(explicit_template).rows
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
    truth_path = _explicit_truth_path(args)
    truth = load_truth_file(truth_path) if truth_path is not None else topic_truth
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
    sequences = getattr(args, "_sequence_paths", None)
    if not sequences:
        raise SystemExit(
            "--ug2-official-complete-to-sequence-timestamps requires --sequence-root"
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
        raise SystemExit(
            "no official Track 5 timestamps found in --sequence-root for "
            f"source {args.ug2_official_timestamp_source!r}"
        )
    template = pd.concat(rows, ignore_index=True)
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
    for paths in sequences:
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
    candidates = merge_candidate_frames(candidate_frames)
    if args.infer_ug2_class_map_from_candidates:
        args._inferred_class_map = infer_sequence_class_map_from_candidates(
            candidates,
            min_confidence=args.classification_min_confidence,
            default_class=args.ug2_class_name,
        )
    truth = merge_truth_frames(truth_frames) if truth_frames else None
    return _run_tracker_for_mode(args, candidates, truth)




def _run_tracker_for_mode(args, candidates, truth):
    if args.tracker_mode == "multi-object":
        return run_mmuad_multi_object_tracker(
            candidates,
            truth,
            config=MultiObjectTrackerConfig(
                acceleration_std_mps2=args.acceleration_std_mps2,
                max_association_distance_m=args.mot_max_association_distance_m,
                max_track_age_s=args.mot_max_track_age_s,
            ),
        )
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
