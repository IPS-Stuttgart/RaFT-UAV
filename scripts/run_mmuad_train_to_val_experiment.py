#!/usr/bin/env python3
"""Run a leakage-guarded MMUAD train-to-public-validation experiment.

The script is intentionally an orchestration layer around the maintained MMUAD
CLIs.  It does not implement a new tracker; it inventories the train/validation
layout, trains models on the train reference, applies them to validation, writes
an official-style Track 5 submission, and scores it against the validation
reference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ExperimentPaths:
    """Canonical output locations for one train-to-val experiment."""

    output_dir: Path
    inventory_json: Path
    manifest_json: Path
    classifier_model: Path
    classifier_features_csv: Path
    classifier_metrics_json: Path
    ranker_model: Path
    ranker_train_features_csv: Path
    ranker_score_features_csv: Path
    ranker_scored_candidates_csv: Path
    tracker_output_dir: Path
    official_results_csv: Path
    official_zip: Path
    scorecard_json: Path
    scorecard_csv: Path
    pose_by_sequence_csv: Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--train-reference", type=Path, required=True)
    parser.add_argument("--val-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument("--classifier-method", default="random-forest")
    parser.add_argument("--ranker-model-type", default="random-forest-classifier")
    parser.add_argument("--ranker-target-column", default="good_cluster_5m")
    parser.add_argument("--ranker-good-threshold-m", type=float, default=5.0)
    parser.add_argument("--selection-confidence-weight", type=float, default=1.0)
    parser.add_argument("--selection-mobility-weight", type=float, default=0.5)
    parser.add_argument("--selection-source-priority-weight", type=float, default=0.25)
    parser.add_argument("--selection-motion-weight", type=float, default=1.0)
    parser.add_argument("--timestamp-source", default="ground-truth-or-all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = experiment_paths(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _guard_against_reference_leakage(args.train_reference, args.val_reference)
    inventory = inventory_layout(args.train_root, args.val_root, args.train_reference, args.val_reference)
    paths.inventory_json.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    commands = build_commands(args, paths)
    manifest = {
        "protocol": "MMUAD train-to-public-validation; validation reference is scoring-only",
        "train_root": str(args.train_root),
        "val_root": str(args.val_root),
        "train_reference": str(args.train_reference),
        "val_reference": str(args.val_reference),
        "output_dir": str(args.output_dir),
        "dry_run": bool(args.dry_run),
        "inventory_json": str(paths.inventory_json),
        "commands": [{"name": name, "command": command} for name, command in commands],
    }
    paths.manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        print(f"manifest_json={paths.manifest_json}")
        print(f"inventory_json={paths.inventory_json}")
        for name, command in commands:
            print(f"[{name}] {' '.join(command)}")
        return 0

    for name, command in commands:
        print(f"running={name}")
        subprocess.run(command, check=True)
    print("mmuad_train_to_val=ok")
    print(f"manifest_json={paths.manifest_json}")
    print(f"scorecard_json={paths.scorecard_json}")
    return 0


def experiment_paths(output_dir: Path) -> ExperimentPaths:
    output_dir = Path(output_dir)
    classifier_dir = output_dir / "sequence_classifier"
    ranker_dir = output_dir / "cluster_ranker"
    tracker_dir = output_dir / "validation_tracker"
    scorecard_dir = output_dir / "scorecard"
    return ExperimentPaths(
        output_dir=output_dir,
        inventory_json=output_dir / "train_to_val_inventory.json",
        manifest_json=output_dir / "mmuad_train_to_val_manifest.json",
        classifier_model=classifier_dir / "sequence_classifier.joblib",
        classifier_features_csv=classifier_dir / "train_sequence_features.csv",
        classifier_metrics_json=classifier_dir / "train_sequence_classifier_metrics.json",
        ranker_model=ranker_dir / "cluster_ranker.json",
        ranker_train_features_csv=ranker_dir / "train_cluster_features.csv",
        ranker_score_features_csv=ranker_dir / "val_cluster_features.csv",
        ranker_scored_candidates_csv=ranker_dir / "val_scored_candidates.csv",
        tracker_output_dir=tracker_dir,
        official_results_csv=tracker_dir / "mmaud_results.csv",
        official_zip=tracker_dir / "ug2_submission.zip",
        scorecard_json=scorecard_dir / "track5_scorecard_train_to_val.json",
        scorecard_csv=scorecard_dir / "track5_scorecard_train_to_val.csv",
        pose_by_sequence_csv=scorecard_dir / "mmuad_pose_by_sequence.csv",
    )


def build_commands(args: argparse.Namespace, paths: ExperimentPaths) -> list[tuple[str, list[str]]]:
    py = str(args.python)
    train_root = str(args.train_root)
    val_root = str(args.val_root)
    commands: list[tuple[str, list[str]]] = []
    commands.append(
        (
            "train_sequence_classifier",
            [
                py,
                "-m",
                "raft_uav.mmuad.train_sequence_classifier",
                train_root,
                "--reference",
                str(args.train_reference),
                "--method",
                str(args.classifier_method),
                "--output",
                str(paths.classifier_model),
                "--feature-report",
                str(paths.classifier_features_csv),
                "--metrics-json",
                str(paths.classifier_metrics_json),
                "--sequence-glob",
                str(args.sequence_glob),
            ],
        )
    )
    commands.append(
        (
            "train_and_score_cluster_ranker",
            [
                py,
                "-m",
                "raft_uav.mmuad.cluster_ranker",
                "--train-sequence-root",
                train_root,
                "--train-truth",
                str(args.train_reference),
                "--score-sequence-root",
                val_root,
                "--model-json",
                str(paths.ranker_model),
                "--model-type",
                str(args.ranker_model_type),
                "--target-column",
                str(args.ranker_target_column),
                "--good-threshold-m",
                str(args.ranker_good_threshold_m),
                "--train-features-csv",
                str(paths.ranker_train_features_csv),
                "--score-features-csv",
                str(paths.ranker_score_features_csv),
                "--scored-candidates-csv",
                str(paths.ranker_scored_candidates_csv),
                "--sequence-glob",
                str(args.sequence_glob),
            ],
        )
    )
    commands.append(
        (
            "run_validation_tracker",
            [
                py,
                "-m",
                "raft_uav.mmuad.run",
                val_root,
                "--output-dir",
                str(paths.tracker_output_dir),
                "--candidate-csv",
                str(paths.ranker_scored_candidates_csv),
                "--selection-confidence-weight",
                str(args.selection_confidence_weight),
                "--selection-mobility-weight",
                str(args.selection_mobility_weight),
                "--selection-source-priority-weight",
                str(args.selection_source_priority_weight),
                "--selection-motion-weight",
                str(args.selection_motion_weight),
                "--sequence-classifier",
                str(paths.classifier_model),
                "--sequence-classifier-provenance-json",
                str(paths.tracker_output_dir / "mmuad_sequence_classifier_provenance.json"),
                "--ug2-official-complete-to-sequence-timestamps",
                "--ug2-official-timestamp-source",
                str(args.timestamp_source),
                "--ug2-official-results-csv",
                str(paths.official_results_csv),
                "--ug2-official-codabench-zip",
                str(paths.official_zip),
                "--ug2-official-validate-on-write",
            ],
        )
    )
    commands.append(
        (
            "score_public_validation",
            [
                py,
                "-m",
                "raft_uav.mmuad.track5_scorecard_cli",
                "--results",
                str(paths.official_zip),
                "--truth",
                str(args.val_reference),
                "--sequence-root",
                val_root,
                "--timestamp-source",
                str(args.timestamp_source),
                "--classification-provenance-json",
                str(paths.tracker_output_dir / "mmuad_sequence_classifier_provenance.json"),
                "--selected-tracklets-csv",
                str(paths.tracker_output_dir / "mmuad_selected_tracklets.csv"),
                "--output-json",
                str(paths.scorecard_json),
                "--summary-csv",
                str(paths.scorecard_csv),
                "--pose-by-sequence-csv",
                str(paths.pose_by_sequence_csv),
                "--require-leaderboard-ready",
            ],
        )
    )
    return commands


def inventory_layout(
    train_root: Path,
    val_root: Path,
    train_reference: Path,
    val_reference: Path,
) -> dict[str, Any]:
    return {
        "train_root": _path_inventory(train_root),
        "val_root": _path_inventory(val_root),
        "train_reference": _path_inventory(train_reference),
        "val_reference": _path_inventory(val_reference),
        "leakage_guard": {
            "train_reference_equals_val_reference": train_reference.resolve()
            == val_reference.resolve(),
            "train_root_equals_val_root": train_root.resolve() == val_root.resolve(),
        },
    }


def _path_inventory(path: Path) -> dict[str, Any]:
    path = Path(path)
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }
    if not path.exists():
        return info
    stat = path.stat()
    info["size_bytes"] = int(stat.st_size)
    if path.is_dir():
        entries = sorted(path.iterdir(), key=lambda item: item.name)
        info["entry_count"] = len(entries)
        info["sample_entries"] = [entry.name for entry in entries[:50]]
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        info["zip_entry_count"] = len(names)
        info["zip_top_level"] = sorted({name.split("/", 1)[0] for name in names if name})
        info["zip_sample_entries"] = names[:100]
    return info


def _guard_against_reference_leakage(train_reference: Path, val_reference: Path) -> None:
    if Path(train_reference).resolve() == Path(val_reference).resolve():
        raise ValueError("train and validation references must be different files")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
