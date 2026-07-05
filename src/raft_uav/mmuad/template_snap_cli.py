"""CLI for snapping official MMUAD Track 5 results to a template grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raft_uav.mmuad.submission import (
    load_official_track5_results_frame,
    load_official_track5_template_file,
)
from raft_uav.mmuad.template_snap_utils import (
    CLASSIFICATION_POLICIES,
    MISSING_POSITION_POLICIES,
    RESAMPLE_METHODS,
)
from raft_uav.mmuad.template_snap_write import write_template_snapped_submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="official CSV/ZIP to snap")
    parser.add_argument("--template", type=Path, required=True, help="official template CSV/ZIP")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resample-method", choices=RESAMPLE_METHODS, default="linear")
    parser.add_argument("--max-interpolation-gap-s", type=float)
    parser.add_argument(
        "--classification-policy",
        choices=CLASSIFICATION_POLICIES,
        default="sequence-mode",
    )
    parser.add_argument(
        "--missing-position-policy",
        choices=MISSING_POSITION_POLICIES,
        default="zero",
    )
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    args = parser.parse_args(argv)

    paths = write_template_snapped_submission(
        results=load_official_track5_results_frame(args.results),
        template=load_official_track5_template_file(args.template),
        output_dir=args.output_dir,
        resample_method=args.resample_method,
        max_interpolation_gap_s=args.max_interpolation_gap_s,
        classification_policy=args.classification_policy,
        missing_position_policy=args.missing_position_policy,
    )
    summary = json.loads(paths["validation_json"].read_text(encoding="utf-8"))
    print("mmuad_template_snap=ok")
    for name, path in paths.items():
        print(f"{name}={path}")
    print(f"leaderboard_ready={summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not summary.get("leaderboard_ready", False):
        reasons = ", ".join(summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"template-snapped upload is not leaderboard-ready: {reasons}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
