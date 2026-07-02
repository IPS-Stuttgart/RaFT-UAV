"""CLI for local UG2+/MMUAD Track 5 validation/evaluation scorecards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raft_uav.mmuad.archive import extract_mmuad_archive, is_supported_archive
from raft_uav.mmuad.schema import load_jsonable
from raft_uav.mmuad.sequence import OFFICIAL_TRACK5_TIMESTAMP_SOURCES
from raft_uav.mmuad.track5_scorecard import (
    build_track5_scorecard,
    write_track5_scorecard,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-scorecard",
        description=(
            "combine local Track 5 upload validation, public-metric evaluation, "
            "and nearest-time diagnostics"
        ),
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="mmaud_results.csv or ZIP",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        help="normalized or official Track 5 truth CSV/ZIP",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="official template/results file for timestamp coverage",
    )
    parser.add_argument(
        "--sequence-root",
        type=Path,
        help=(
            "public Track 5 sequence root used to build a timestamp template "
            "directly from Image/ground_truth/LiDAR folders; ZIP/TAR-family "
            "archives are safely extracted before inspection"
        ),
    )
    parser.add_argument(
        "--sequence-root-archive-extract-dir",
        type=Path,
        help=(
            "directory for extracting ZIP/TAR sequence roots before building "
            "the Track 5 scorecard template; defaults beside --output-json"
        ),
    )
    parser.add_argument(
        "--sequence-root-archive-manifest-json",
        type=Path,
        help=(
            "where to write the extraction manifest when --sequence-root is an "
            "archive; defaults beside --output-json"
        ),
    )
    parser.add_argument("--sequence-glob", default="*", help="sequence discovery glob")
    parser.add_argument(
        "--split-name",
        help="optional top-level split folder to score, e.g. val or test",
    )
    parser.add_argument(
        "--timestamp-source",
        choices=OFFICIAL_TRACK5_TIMESTAMP_SOURCES,
        default="ground-truth-or-all",
        help="modality folder used for --sequence-root timestamp-template discovery",
    )
    parser.add_argument("--class-map", type=Path, help="sequence-to-class map CSV/JSON/YAML")
    parser.add_argument(
        "--official-upload-manifest",
        type=Path,
        help="mmuad_official_upload_manifest.json to verify with the scorecard",
    )
    parser.add_argument(
        "--classification-provenance-json",
        type=Path,
        help="mmuad_sequence_classifier_provenance.json from a classifier-backed run",
    )
    parser.add_argument(
        "--selected-tracklets-csv",
        type=Path,
        help="mmuad_selected_tracklets.csv used to annotate pose-by-sequence sensor usage",
    )
    parser.add_argument(
        "--candidate-oracle-gap-csv",
        type=Path,
        help="mmuad_candidate_oracle_gap.csv used to summarize candidate regret",
    )
    parser.add_argument("--allow-csv-submission", action="store_true")
    parser.add_argument("--timestamp-tolerance-s", type=float, default=1.0e-6)
    parser.add_argument("--nearest-time-delta-s", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--validation-rows-csv", type=Path)
    parser.add_argument("--public-evaluation-rows-csv", type=Path)
    parser.add_argument("--nearest-time-rows-csv", type=Path)
    parser.add_argument("--pose-by-sequence-csv", type=Path)
    parser.add_argument("--candidate-regret-summary-csv", type=Path)
    parser.add_argument(
        "--require-leaderboard-ready",
        action="store_true",
        help=(
            "exit nonzero unless upload validation and public Track 5 evaluation "
            "are leaderboard-ready"
        ),
    )
    args = parser.parse_args(argv)

    sequence_root, archive_manifest_path = _prepare_scorecard_sequence_root(args)
    scorecard = build_track5_scorecard(
        results_path=args.results,
        truth_path=args.truth,
        template_path=args.template,
        sequence_root=sequence_root,
        sequence_glob=args.sequence_glob,
        split_name=args.split_name,
        timestamp_source=args.timestamp_source,
        class_map_path=args.class_map,
        upload_manifest_path=args.official_upload_manifest,
        classification_provenance_path=args.classification_provenance_json,
        selected_tracklets_path=args.selected_tracklets_csv,
        candidate_oracle_gap_path=args.candidate_oracle_gap_csv,
        require_zip=not args.allow_csv_submission,
        timestamp_tolerance_s=args.timestamp_tolerance_s,
        max_time_delta_s=args.nearest_time_delta_s,
    )
    if archive_manifest_path is not None:
        scorecard.summary["sequence_root_archive_manifest_json"] = str(archive_manifest_path)
    written_paths = write_track5_scorecard(
        scorecard,
        summary_json=args.output_json,
        summary_csv=args.summary_csv,
        validation_rows_csv=args.validation_rows_csv,
        public_evaluation_rows_csv=args.public_evaluation_rows_csv,
        nearest_time_rows_csv=args.nearest_time_rows_csv,
        pose_by_sequence_csv=args.pose_by_sequence_csv,
        candidate_regret_summary_csv=args.candidate_regret_summary_csv,
    )

    summary = scorecard.summary
    print("track5_scorecard=ok")
    for name, path in written_paths.items():
        print(f"{name}={path}")
    if archive_manifest_path is not None:
        print(f"sequence_root_archive_manifest_json={archive_manifest_path}")
    print(f"leaderboard_ready={summary['scorecard_leaderboard_ready']}")
    print(f"codabench_upload_ready={summary['codabench_upload_ready']}")
    if summary.get("sequence_root") is not None:
        template_count = summary.get("validation", {}).get("template_timestamp_count")
        print(f"sequence_root={summary['sequence_root']}")
        print(f"timestamp_source={summary['timestamp_source']}")
        print(f"template_timestamp_count={template_count}")
    if summary.get("upload_manifest_valid") is not None:
        print(f"upload_manifest_valid={summary['upload_manifest_valid']}")
    pooled = (summary.get("public_track5") or {}).get("pooled", {})
    if "pose_mse_loss_m2" in pooled:
        print(f"pose_mse_loss_m2={pooled['pose_mse_loss_m2']}")
    if "uav_type_accuracy" in pooled:
        print(f"uav_type_accuracy={pooled['uav_type_accuracy']}")
    if "classification_accuracy" in pooled:
        print(f"classification_accuracy={pooled['classification_accuracy']}")
    if args.require_leaderboard_ready and not summary["scorecard_leaderboard_ready"]:
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"Track 5 scorecard is not leaderboard-ready: {reasons}")
    return 0


def _prepare_scorecard_sequence_root(
    args: argparse.Namespace,
) -> tuple[Path | None, Path | None]:
    """Return a usable sequence root and optional extraction-manifest path."""

    sequence_root = args.sequence_root
    if sequence_root is None:
        if args.sequence_root_archive_extract_dir is not None:
            raise ValueError("--sequence-root-archive-extract-dir requires --sequence-root")
        if args.sequence_root_archive_manifest_json is not None:
            raise ValueError("--sequence-root-archive-manifest-json requires --sequence-root")
        return None, None
    if not is_supported_archive(sequence_root):
        if args.sequence_root_archive_extract_dir is not None:
            raise ValueError(
                "--sequence-root-archive-extract-dir requires an archive --sequence-root"
            )
        if args.sequence_root_archive_manifest_json is not None:
            raise ValueError(
                "--sequence-root-archive-manifest-json requires an archive --sequence-root"
            )
        return sequence_root, None

    extract_dir = args.sequence_root_archive_extract_dir or (
        args.output_json.parent / "mmuad_scorecard_sequence_root_archive"
    )
    manifest = extract_mmuad_archive(sequence_root, extract_dir)
    manifest_path = args.sequence_root_archive_manifest_json or (
        args.output_json.parent / "mmuad_scorecard_sequence_root_archive_manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(load_jsonable(manifest), indent=2),
        encoding="utf-8",
    )
    return Path(manifest["extract_root"]), manifest_path


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
