#!/usr/bin/env python
"""One-command MMUAD train-to-validation experiment harness.

The script inventories the train layout, trains the existing MMUAD sequence
classifier and cluster ranker on train, applies both to public validation,
writes official Track 5 artifacts, scores them locally, and records provenance.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any
from zipfile import BadZipFile, ZipFile

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.classification import load_sequence_class_labels  # noqa: E402
from raft_uav.mmuad.cluster_ranker import _load_candidates_from_args  # noqa: E402
from raft_uav.mmuad.evaluator import load_evaluation_truth_file  # noqa: E402
from raft_uav.mmuad.io import load_truth_file  # noqa: E402
from raft_uav.mmuad.sequence import (  # noqa: E402
    _sequence_class_label,
    discover_sequence_paths,
)
from raft_uav.mmuad.source_calibration import (  # noqa: E402
    SOURCE_CALIBRATION_MODES,
    fit_source_calibration,
    write_source_calibration_json,
)
from raft_uav.mmuad.train_selected_config import (  # noqa: E402
    load_train_selected_config,
)


WORK_ROOT = Path("/mnt/lexar4tb/mmuad_realdata")
DEFAULT_VAL_ROOT = WORK_ROOT / "extracted/val-d2b4424284f3/val"
DEFAULT_VAL_REFERENCE = WORK_ROOT / "challenge_meta/validation_ref_new_for_your_ref.csv"
DEFAULT_OUTPUT_DIR = WORK_ROOT / "outputs/mmuad_train_to_val"

CLASS_NAME_TO_ID = {
    "0": "0",
    "mavic 3": "0",
    "mavic3": "0",
    "dji mavic 3": "0",
    "1": "1",
    "m30": "1",
    "matrice 30": "1",
    "dji m30": "1",
    "2": "2",
    "m300": "2",
    "matrice 300": "2",
    "dji m300": "2",
    "3": "3",
    "phantom 4": "3",
    "phantom4": "3",
    "dji phantom 4": "3",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-zip", type=Path, action="append", default=[])
    parser.add_argument("--train-reference", type=Path)
    parser.add_argument("--train-truth", type=Path)
    parser.add_argument("--val-root", type=Path, default=DEFAULT_VAL_ROOT)
    parser.add_argument("--val-reference", type=Path, default=DEFAULT_VAL_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selected-config-json", type=Path)
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument("--classifier-method", default="random-forest")
    parser.add_argument("--ranker-model-type", default="random-forest-classifier")
    parser.add_argument("--ranker-target-column", default="good_cluster_5m")
    parser.add_argument("--good-threshold-m", type=float, default=5.0)
    parser.add_argument("--selection-confidence-weight", type=float, default=64.0)
    parser.add_argument(
        "--source-calibration-mode",
        choices=SOURCE_CALIBRATION_MODES,
        default="identity",
    )
    parser.add_argument("--source-translation-alpha", type=float, default=1.0)
    parser.add_argument("--source-calibration-max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--source-calibration-max-pair-distance-m", type=float, default=120.0)
    parser.add_argument("--source-calibration-min-pairs-per-source", type=int, default=20)
    parser.add_argument("--mmuad-selection-mode", choices=("greedy", "viterbi"), default="greedy")
    parser.add_argument("--mmuad-viterbi-motion-weight", type=float, default=1.0)
    parser.add_argument("--mmuad-viterbi-ranker-weight", type=float, default=1.0)
    parser.add_argument("--mmuad-viterbi-source-switch-penalty", type=float, default=0.0)
    parser.add_argument("--mmuad-viterbi-max-speed-mps", type=float, default=60.0)
    parser.add_argument("--mmuad-viterbi-gap-penalty", type=float, default=0.0)
    parser.add_argument(
        "--trajectory-completion-mode",
        choices=("none", "gap-interpolation", "fixed-lag", "constant-velocity", "constant-acceleration"),
        default="none",
    )
    parser.add_argument("--trajectory-speed-gate-mps", type=float, default=0.0)
    parser.add_argument("--trajectory-smoothing-blend", type=float, default=1.0)
    parser.add_argument("--image-nonimage-fusion-weight", type=float, default=0.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--voxel-size-m", type=float, default=0.75)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    parser.add_argument("--timestamp-source", default="ground-truth-or-all")
    parser.add_argument("--no-apply-calibration", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    selected_config = apply_selected_config(args)

    if args.train_reference is not None and args.train_reference.resolve() == args.val_reference.resolve():
        raise ValueError("train and validation references must be different files")

    paths = artifact_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)
    started_at = utc_now()
    commands: list[dict[str, Any]] = []

    try:
        inventory = build_train_inventory(args)
        write_json(paths["train_inventory"], inventory)
        resolved = resolve_train_inputs(args, paths)
        if not args.dry_run:
            fit_selected_source_calibration(args, paths, resolved, commands)
        planned = command_plan(args, paths, resolved)
        if not args.dry_run:
            for step, command in planned:
                run_step(step, command, args.output_dir, commands)
        summary = build_summary(
            args,
            paths,
            resolved,
            commands,
            started_at,
            "dry_run" if args.dry_run else "ok",
            None,
            planned,
            selected_config,
        )
    except Exception as exc:
        summary = build_summary(
            args,
            paths,
            locals().get("resolved"),
            commands,
            started_at,
            "failed",
            str(exc),
            locals().get("planned"),
            selected_config,
        )
        write_json(paths["summary_json"], summary)
        write_summary_csv(paths["summary_csv"], summary)
        write_provenance(args, paths, summary, commands)
        raise

    write_json(paths["summary_json"], summary)
    write_summary_csv(paths["summary_csv"], summary)
    write_provenance(args, paths, summary, commands)
    print(f"mmuad_train_to_val_summary_json={paths['summary_json']}")
    print(f"mmuad_train_to_val_summary_csv={paths['summary_csv']}")
    print(f"train_inventory_json={paths['train_inventory']}")
    print(f"track5_scorecard_train_to_val_json={paths['scorecard_json']}")
    print(f"mmaud_results_train_to_val_csv={paths['official_results_csv']}")
    print(f"ug2_submission_train_to_val_zip={paths['official_zip']}")
    return 0


def artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "summary_json": output_dir / "mmuad_train_to_val_summary.json",
        "summary_csv": output_dir / "mmuad_train_to_val_summary.csv",
        "train_inventory": output_dir / "train_inventory.json",
        "scorecard_json": output_dir / "track5_scorecard_train_to_val.json",
        "official_results_csv": output_dir / "mmaud_results_train_to_val.csv",
        "official_zip": output_dir / "ug2_submission_train_to_val.zip",
        "provenance": output_dir / "mmuad_train_to_val_provenance.json",
        "auto_truth": output_dir / "train_truth_auto.csv",
        "auto_class_map": output_dir / "train_class_map_auto.csv",
        "classifier_model": output_dir / "mmuad_sequence_classifier_train.joblib",
        "classifier_features": output_dir / "mmuad_sequence_classifier_train_features.csv",
        "classifier_predictions": output_dir / "mmuad_sequence_classifier_train_predictions.csv",
        "classifier_metrics": output_dir / "mmuad_sequence_classifier_train_metrics.json",
        "ranker_model": output_dir / "mmuad_cluster_ranker_train.json",
        "ranker_features": output_dir / "mmuad_cluster_ranker_train_features.csv",
        "ranker_candidates": output_dir / "mmuad_cluster_ranker_train_candidates.csv",
        "source_calibration_json": output_dir / "mmuad_train_source_calibration.json",
        "source_calibration_pairs": output_dir / "mmuad_train_source_calibration_pairs.csv",
        "source_calibration_summary": output_dir / "mmuad_train_source_calibration_summary.csv",
        "ranker_train_source_calibrated": output_dir / "mmuad_cluster_ranker_train_source_calibrated_candidates.csv",
        "tracker_dir": output_dir / "tracker_train_to_val",
        "tracker_val_source_calibrated": output_dir / "mmuad_val_source_calibrated_candidates.csv",
        "ranker_val_scored": output_dir / "mmuad_cluster_ranker_val_scored_candidates.csv",
        "ranker_val_features": output_dir / "mmuad_cluster_ranker_val_score_features.csv",
        "ranker_val_merged": output_dir / "mmuad_cluster_ranker_val_merged_candidates.csv",
        "classifier_val_predictions": output_dir / "mmuad_sequence_classifier_val_predictions.csv",
        "classifier_val_features": output_dir / "mmuad_sequence_classifier_val_features.csv",
        "classifier_provenance": output_dir / "mmuad_sequence_classifier_provenance.json",
        "official_validation_json": output_dir / "mmuad_official_submission_validation_train_to_val.json",
        "official_validation_rows": output_dir / "mmuad_official_submission_validation_rows_train_to_val.csv",
        "official_manifest": output_dir / "mmuad_official_upload_manifest_train_to_val.json",
        "scorecard_csv": output_dir / "track5_scorecard_train_to_val.csv",
        "scorecard_validation_rows": output_dir / "track5_scorecard_validation_rows_train_to_val.csv",
        "scorecard_public_rows": output_dir / "track5_scorecard_public_rows_train_to_val.csv",
        "scorecard_nearest_rows": output_dir / "track5_scorecard_nearest_rows_train_to_val.csv",
    }


def apply_selected_config(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.selected_config_json is None:
        return None
    config = load_train_selected_config(args.selected_config_json)
    mapping = {
        "source_calibration_mode": "source_calibration_mode",
        "source_translation_alpha": "source_translation_alpha",
        "ranker_model_type": "ranker_model_type",
        "ranker_target_column": "ranker_target_column",
        "mmuad_selection_mode": "mmuad_selection_mode",
        "viterbi_motion_weight": "mmuad_viterbi_motion_weight",
        "viterbi_ranker_weight": "mmuad_viterbi_ranker_weight",
        "viterbi_source_switch_penalty": "mmuad_viterbi_source_switch_penalty",
        "viterbi_max_speed_mps": "mmuad_viterbi_max_speed_mps",
        "viterbi_gap_penalty": "mmuad_viterbi_gap_penalty",
        "smoothing_mode": "trajectory_completion_mode",
        "smoothing_speed_gate_mps": "trajectory_speed_gate_mps",
        "smoothing_blend": "trajectory_smoothing_blend",
        "classifier_method": "classifier_method",
        "image_nonimage_fusion_weight": "image_nonimage_fusion_weight",
    }
    for config_key, arg_name in mapping.items():
        setattr(args, arg_name, config[config_key])
    return config


def fit_selected_source_calibration(
    args: argparse.Namespace,
    paths: dict[str, Path],
    resolved: dict[str, Any],
    commands: list[dict[str, Any]],
) -> None:
    if args.source_calibration_mode == "identity":
        return
    started = time.time()
    candidates = _load_candidates_from_args(
        csv_path=None,
        sequence_root=args.train_root,
        sequence_glob=args.sequence_glob,
        apply_calibration=not args.no_apply_calibration,
        voxel_size_m=args.voxel_size_m,
        min_cluster_points=args.min_cluster_points,
    )
    truth = load_evaluation_truth_file(Path(resolved["train_truth"])).rows
    payload, pairs, fit_summary = fit_source_calibration(
        candidates,
        truth,
        mode=args.source_calibration_mode,
        max_truth_time_delta_s=args.source_calibration_max_truth_time_delta_s,
        max_pair_distance_m=args.source_calibration_max_pair_distance_m,
        min_pairs_per_source=args.source_calibration_min_pairs_per_source,
        source_translation_alpha_grid=[args.source_translation_alpha],
    )
    payload["provenance"] = {
        "protocol": "fit_on_train_only_before_public_validation_eval",
        "train_root": str(args.train_root),
        "train_truth": resolved["train_truth"],
        "selected_config_json": None
        if args.selected_config_json is None
        else str(args.selected_config_json),
    }
    write_source_calibration_json(payload, paths["source_calibration_json"])
    pairs.to_csv(paths["source_calibration_pairs"], index=False)
    fit_summary.to_csv(paths["source_calibration_summary"], index=False)
    commands.append(
        {
            "name": "fit_source_calibration",
            "command": [
                "in-process",
                "fit_source_calibration",
                "--mode",
                args.source_calibration_mode,
                "--source-translation-alpha",
                str(args.source_translation_alpha),
            ],
            "returncode": 0,
            "duration_s": round(time.time() - started, 3),
            "output_json": str(paths["source_calibration_json"]),
            "fit_pairs_csv": str(paths["source_calibration_pairs"]),
            "fit_summary_csv": str(paths["source_calibration_summary"]),
        }
    )


def build_train_inventory(args: argparse.Namespace) -> dict[str, Any]:
    sequences = discover_sequence_paths(args.train_root, sequence_glob=args.sequence_glob)
    rows = []
    for paths in sequences:
        child_dirs = {child.name.lower() for child in paths.root.iterdir() if child.is_dir()}
        label = normalize_class_label(_sequence_class_label(paths.class_files, sequence_id=paths.sequence_id))
        rows.append(
            {
                "sequence_id": paths.sequence_id,
                "root": str(paths.root),
                "has_image_dir": bool({"image", "images"} & child_dirs),
                "has_lidar_360_dir": "lidar_360" in child_dirs,
                "has_livox_avia_dir": "livox_avia" in child_dirs,
                "has_radar_enhance_pcl_dir": "radar_enhance_pcl" in child_dirs,
                "truth_file_count": len(paths.truth_files),
                "class_file_count": len(paths.class_files),
                "class_label": label,
                "truth_files_sample": [str(path) for path in paths.truth_files[:10]],
                "class_files_sample": [str(path) for path in paths.class_files[:10]],
            }
        )
    layout = {
        "sequence_count": len(rows),
        "with_image_dir": sum(row["has_image_dir"] for row in rows),
        "with_lidar_360_dir": sum(row["has_lidar_360_dir"] for row in rows),
        "with_livox_avia_dir": sum(row["has_livox_avia_dir"] for row in rows),
        "with_radar_enhance_pcl_dir": sum(row["has_radar_enhance_pcl_dir"] for row in rows),
        "with_truth_files": sum(row["truth_file_count"] > 0 for row in rows),
        "with_class_files": sum(row["class_file_count"] > 0 for row in rows),
        "with_class_label": sum(bool(row["class_label"]) for row in rows),
    }
    layout["has_expected_track5_train_layout"] = bool(
        layout["sequence_count"]
        and layout["with_image_dir"] == layout["sequence_count"]
        and layout["with_lidar_360_dir"] == layout["sequence_count"]
        and layout["with_livox_avia_dir"] == layout["sequence_count"]
        and layout["with_radar_enhance_pcl_dir"] == layout["sequence_count"]
    )
    zips = unique_paths([*args.train_zip, *(args.work_root.rglob("*train*.zip") if args.work_root.exists() else [])])
    return {
        "schema": "raft-uav-mmuad-train-inventory-v1",
        "created_at_utc": utc_now(),
        "train_root": str(args.train_root),
        "train_root_exists": args.train_root.exists(),
        "sequence_glob": args.sequence_glob,
        "layout": layout,
        "sequence_rows": rows,
        "zip_candidates": [zip_inventory(path) for path in zips],
    }


def resolve_train_inputs(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    auto_truth, auto_class_map = write_auto_refs(args.train_root, args.sequence_glob, paths["auto_truth"], paths["auto_class_map"])
    train_reference = args.train_reference or auto_class_map
    train_truth = args.train_truth or auto_truth
    if train_reference is None or not train_reference.exists():
        raise ValueError("could not resolve train labels; pass --train-reference or provide class files")
    if train_truth is None or not train_truth.exists():
        raise ValueError("could not resolve train truth; pass --train-truth or provide ground_truth files")
    labels = load_sequence_class_labels(train_reference)
    return {
        "train_reference": str(train_reference),
        "train_truth": str(train_truth),
        "auto_train_truth": str(auto_truth) if auto_truth else None,
        "auto_train_class_map": str(auto_class_map) if auto_class_map else None,
        "train_label_sequence_count": len(labels),
        "train_label_values": sorted({str(value) for value in labels.values()}),
    }


def write_auto_refs(train_root: Path, sequence_glob: str, truth_path: Path, class_map_path: Path) -> tuple[Path | None, Path | None]:
    truth_frames: list[pd.DataFrame] = []
    class_rows: list[dict[str, str]] = []
    for paths in discover_sequence_paths(train_root, sequence_glob=sequence_glob):
        label = normalize_class_label(_sequence_class_label(paths.class_files, sequence_id=paths.sequence_id))
        if label:
            class_rows.append({"sequence_id": paths.sequence_id, "uav_type": label})
        for truth_file in paths.truth_files:
            truth = load_truth_file(truth_file, default_sequence_id=paths.sequence_id).rows.copy()
            if label:
                truth["uav_type"] = label
            truth_frames.append(truth)
    wrote_truth = None
    wrote_class = None
    if truth_frames:
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(truth_frames, ignore_index=True).drop_duplicates(
            subset=["sequence_id", "time_s"]
        ).sort_values(["sequence_id", "time_s"]).to_csv(truth_path, index=False)
        wrote_truth = truth_path
    if class_rows:
        class_map_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(class_rows).drop_duplicates(subset=["sequence_id"]).sort_values(
            "sequence_id"
        ).to_csv(class_map_path, index=False)
        wrote_class = class_map_path
    return wrote_truth, wrote_class


def command_plan(args: argparse.Namespace, paths: dict[str, Path], resolved: dict[str, Any]) -> list[tuple[str, list[str]]]:
    common = ["--sequence-glob", args.sequence_glob, "--voxel-size-m", str(args.voxel_size_m), "--min-cluster-points", str(args.min_cluster_points)]
    if args.no_apply_calibration:
        common.append("--no-apply-calibration")
    source_calibration_args = []
    if args.source_calibration_mode != "identity":
        source_calibration_args = [
            "--mmuad-source-calibration-json",
            str(paths["source_calibration_json"]),
            "--mmuad-source-calibration-mode",
            args.source_calibration_mode,
        ]
    classifier = [
        sys.executable,
        "-m",
        "raft_uav.mmuad.train_sequence_classifier",
        str(args.train_root),
        "--reference",
        resolved["train_reference"],
        "--method",
        args.classifier_method,
        "--output",
        str(paths["classifier_model"]),
        "--feature-report",
        str(paths["classifier_features"]),
        "--predictions-csv",
        str(paths["classifier_predictions"]),
        "--metrics-json",
        str(paths["classifier_metrics"]),
        "--random-state",
        str(args.random_state),
        "--n-estimators",
        str(args.n_estimators),
        *common,
    ]
    ranker = [
        sys.executable,
        "-m",
        "raft_uav.mmuad.cluster_ranker",
        "--train-sequence-root",
        str(args.train_root),
        "--train-truth",
        resolved["train_truth"],
        "--model-json",
        str(paths["ranker_model"]),
        "--model-type",
        args.ranker_model_type,
        "--target-column",
        args.ranker_target_column,
        "--good-threshold-m",
        str(args.good_threshold_m),
        "--train-features-csv",
        str(paths["ranker_features"]),
        "--train-candidates-output-csv",
        str(paths["ranker_candidates"]),
        *source_calibration_args,
        *(
            [
                "--train-source-calibrated-candidates-csv",
                str(paths["ranker_train_source_calibrated"]),
            ]
            if source_calibration_args
            else []
        ),
        "--random-state",
        str(args.random_state),
        "--n-estimators",
        str(args.n_estimators),
        *common,
    ]
    tracker = [
        sys.executable,
        "-m",
        "raft_uav.mmuad.run",
        str(args.val_root),
        "--output-dir",
        str(paths["tracker_dir"]),
        "--cluster-ranker-model-json",
        str(paths["ranker_model"]),
        "--cluster-ranker-scored-candidates-csv",
        str(paths["ranker_val_scored"]),
        "--cluster-ranker-score-features-csv",
        str(paths["ranker_val_features"]),
        "--cluster-ranker-merged-candidates-csv",
        str(paths["ranker_val_merged"]),
        "--sequence-classifier",
        str(paths["classifier_model"]),
        "--sequence-classifier-predictions-csv",
        str(paths["classifier_val_predictions"]),
        "--sequence-classifier-feature-report",
        str(paths["classifier_val_features"]),
        "--sequence-classifier-provenance-json",
        str(paths["classifier_provenance"]),
        *source_calibration_args,
        *(
            [
                "--mmuad-source-calibrated-candidates-csv",
                str(paths["tracker_val_source_calibrated"]),
            ]
            if source_calibration_args
            else []
        ),
        "--selection-confidence-weight",
        str(args.selection_confidence_weight),
        "--mmuad-selection-mode",
        args.mmuad_selection_mode,
        "--mmuad-viterbi-motion-weight",
        str(args.mmuad_viterbi_motion_weight),
        "--mmuad-viterbi-ranker-weight",
        str(args.mmuad_viterbi_ranker_weight),
        "--mmuad-viterbi-source-switch-penalty",
        str(args.mmuad_viterbi_source_switch_penalty),
        "--mmuad-viterbi-max-speed-mps",
        str(args.mmuad_viterbi_max_speed_mps),
        "--mmuad-viterbi-gap-penalty",
        str(args.mmuad_viterbi_gap_penalty),
        "--trajectory-completion-mode",
        args.trajectory_completion_mode,
        "--trajectory-speed-gate-mps",
        str(args.trajectory_speed_gate_mps),
        "--trajectory-smoothing-blend",
        str(args.trajectory_smoothing_blend),
        "--ug2-official-results-csv",
        str(paths["official_results_csv"]),
        "--ug2-official-codabench-zip",
        str(paths["official_zip"]),
        "--ug2-official-complete-to-sequence-timestamps",
        "--ug2-official-timestamp-source",
        args.timestamp_source,
        "--official-validation-template-file",
        str(args.val_reference),
        "--ug2-official-validate-on-write",
        "--official-validation-json",
        str(paths["official_validation_json"]),
        "--official-validation-rows-csv",
        str(paths["official_validation_rows"]),
        "--official-upload-manifest-json",
        str(paths["official_manifest"]),
        *common,
    ]
    scorecard = [
        sys.executable,
        "-m",
        "raft_uav.mmuad.track5_scorecard_cli",
        "--results",
        str(paths["official_zip"]),
        "--truth",
        str(args.val_reference),
        "--template",
        str(args.val_reference),
        "--official-upload-manifest",
        str(paths["official_manifest"]),
        "--classification-provenance-json",
        str(paths["classifier_provenance"]),
        "--output-json",
        str(paths["scorecard_json"]),
        "--summary-csv",
        str(paths["scorecard_csv"]),
        "--validation-rows-csv",
        str(paths["scorecard_validation_rows"]),
        "--public-evaluation-rows-csv",
        str(paths["scorecard_public_rows"]),
        "--nearest-time-rows-csv",
        str(paths["scorecard_nearest_rows"]),
    ]
    return [
        ("train_sequence_classifier", classifier),
        ("train_cluster_ranker", ranker),
        ("run_validation_tracker", tracker),
        ("track5_scorecard", scorecard),
    ]


def run_step(name: str, command: list[str], output_dir: Path, records: list[dict[str, Any]]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    started = time.time()
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout = output_dir / "logs" / f"{name}.stdout.log"
    stderr = output_dir / "logs" / f"{name}.stderr.log"
    stdout.write_text(result.stdout, encoding="utf-8")
    stderr.write_text(result.stderr, encoding="utf-8")
    records.append(
        {
            "name": name,
            "command": command,
            "returncode": result.returncode,
            "duration_s": round(time.time() - started, 3),
            "stdout_log": str(stdout),
            "stderr_log": str(stderr),
        }
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)


def build_summary(
    args: argparse.Namespace,
    paths: dict[str, Path],
    resolved: dict[str, Any] | None,
    commands: list[dict[str, Any]],
    started_at: str,
    status: str,
    failure: str | None,
    planned: list[tuple[str, list[str]]] | None,
    selected_config: dict[str, Any] | None,
) -> dict[str, Any]:
    scorecard = read_json(paths["scorecard_json"])
    pooled = (scorecard.get("public_track5") or {}).get("pooled") or {}
    nearest = (scorecard.get("nearest_time") or {}).get("pooled") or {}
    summary = {
        "schema": "raft-uav-mmuad-train-to-val-summary-v1",
        "status": status,
        "failure": failure,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "train_root": str(args.train_root),
        "val_root": str(args.val_root),
        "val_reference": str(args.val_reference),
        "train_inventory_json": str(paths["train_inventory"]),
        "train_reference": None if resolved is None else resolved.get("train_reference"),
        "train_truth": None if resolved is None else resolved.get("train_truth"),
        "mmaud_results_train_to_val_csv": str(paths["official_results_csv"]),
        "ug2_submission_train_to_val_zip": str(paths["official_zip"]),
        "track5_scorecard_train_to_val_json": str(paths["scorecard_json"]),
        "selected_config_json": None
        if args.selected_config_json is None
        else str(args.selected_config_json),
        "selected_config": selected_config,
        "selection_protocol": (
            "frozen_train_selected_config"
            if selected_config is not None
            else "direct_cli_settings"
        ),
        "source_calibration_mode": args.source_calibration_mode,
        "source_translation_alpha": args.source_translation_alpha,
        "source_calibration_json": (
            str(paths["source_calibration_json"])
            if args.source_calibration_mode != "identity"
            else None
        ),
        "ranker_model_type": args.ranker_model_type,
        "ranker_target_column": args.ranker_target_column,
        "mmuad_selection_mode": args.mmuad_selection_mode,
        "viterbi_motion_weight": args.mmuad_viterbi_motion_weight,
        "viterbi_ranker_weight": args.mmuad_viterbi_ranker_weight,
        "viterbi_source_switch_penalty": args.mmuad_viterbi_source_switch_penalty,
        "viterbi_max_speed_mps": args.mmuad_viterbi_max_speed_mps,
        "viterbi_gap_penalty": args.mmuad_viterbi_gap_penalty,
        "smoothing_mode": args.trajectory_completion_mode,
        "smoothing_speed_gate_mps": args.trajectory_speed_gate_mps,
        "smoothing_blend": args.trajectory_smoothing_blend,
        "classifier_method": args.classifier_method,
        "image_nonimage_fusion_weight": args.image_nonimage_fusion_weight,
        "pose_mse_loss_m2": pooled.get("pose_mse_loss_m2"),
        "pose_rmse_m": nearest.get("rmse_3d_m"),
        "p95_3d_m": pooled.get("p95_3d_m") or nearest.get("p95_3d_m"),
        "classification_accuracy": pooled.get("classification_accuracy"),
        "uav_type_accuracy": pooled.get("uav_type_accuracy"),
        "commands": commands,
        "planned_commands": None if planned is None else [{"name": name, "command": command} for name, command in planned],
    }
    if resolved:
        summary.update(resolved)
    return summary


def write_provenance(args: argparse.Namespace, paths: dict[str, Path], summary: dict[str, Any], commands: list[dict[str, Any]]) -> None:
    payload = {
        "schema": "raft-uav-mmuad-train-to-val-provenance-v1",
        "created_at_utc": utc_now(),
        "repo_root": str(REPO_ROOT),
        "git": git_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary_json": str(paths["summary_json"]),
        "summary": summary,
        "commands": commands,
    }
    write_json(paths["provenance"], payload)


def zip_inventory(path: Path) -> dict[str, Any]:
    item: dict[str, Any] = {"zip": str(path), "exists": path.exists()}
    if not path.exists():
        return item
    item["size_bytes"] = path.stat().st_size
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
    except (BadZipFile, OSError) as exc:
        item["read_error"] = str(exc)
        return item
    item["entries"] = len(names)
    item["top_level"] = sorted({name.split("/", 1)[0] for name in names if name})
    item["first_50"] = names[:50]
    item["sample"] = names[:200]
    return item


def normalize_class_label(label: Any) -> str:
    if label is None:
        return ""
    text = str(label).strip()
    key = text.lower().replace("_", " ").replace("-", " ")
    return CLASS_NAME_TO_ID.get(key, CLASS_NAME_TO_ID.get(key.replace(" ", ""), text))


def unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return sorted(out)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2), encoding="utf-8")


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    row = {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in summary.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {"sha": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "dirty": bool(status)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
