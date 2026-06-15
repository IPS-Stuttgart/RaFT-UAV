"""CLI for local UG2+/MMUAD Track 5 validation/evaluation scorecards."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument("--results", type=Path, required=True, help="mmaud_results.csv or ZIP")
    parser.add_argument("--truth", type=Path, help="normalized or official Track 5 truth CSV/ZIP")
    parser.add_argument(
        "--template",
        type=Path,
        help="official template/results file for timestamp coverage",
    )
    parser.add_argument("--class-map", type=Path, help="sequence-to-class map CSV/JSON/YAML")
    parser.add_argument(
        "--official-upload-manifest",
        type=Path,
        help="mmuad_official_upload_manifest.json to verify with the scorecard",
    )
    parser.add_argument("--allow-csv-submission", action="store_true")
    parser.add_argument("--timestamp-tolerance-s", type=float, default=1.0e-6)
    parser.add_argument("--nearest-time-delta-s", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--validation-rows-csv", type=Path)
    parser.add_argument("--public-evaluation-rows-csv", type=Path)
    parser.add_argument("--nearest-time-rows-csv", type=Path)
    parser.add_argument(
        "--require-leaderboard-ready",
        action="store_true",
        help=(
            "exit nonzero unless upload validation and public Track 5 evaluation "
            "are leaderboard-ready"
        ),
    )
    args = parser.parse_args(argv)

    scorecard = build_track5_scorecard(
        results_path=args.results,
        truth_path=args.truth,
        template_path=args.template,
        class_map_path=args.class_map,
        upload_manifest_path=args.official_upload_manifest,
        require_zip=not args.allow_csv_submission,
        timestamp_tolerance_s=args.timestamp_tolerance_s,
        max_time_delta_s=args.nearest_time_delta_s,
    )
    write_track5_scorecard(
        scorecard,
        summary_json=args.output_json,
        summary_csv=args.summary_csv,
        validation_rows_csv=args.validation_rows_csv,
        public_evaluation_rows_csv=args.public_evaluation_rows_csv,
        nearest_time_rows_csv=args.nearest_time_rows_csv,
    )

    summary = scorecard.summary
    print("track5_scorecard=ok")
    print(f"leaderboard_ready={summary['scorecard_leaderboard_ready']}")
    print(f"codabench_upload_ready={summary['codabench_upload_ready']}")
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
