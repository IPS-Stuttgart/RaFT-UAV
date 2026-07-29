#!/usr/bin/env python
"""Run the Multi-UAV LTS baseline with RaFT-UAV competition tracker fixes.

This wrapper applies the version-guarded upstream BoT-SORT patch, exports the
competition configuration, and delegates detector inference and submission
packaging to ``run_multi_uav_lts_official_baseline.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.multi_uav_lts.upstream_patch import (  # noqa: E402
    UpstreamPatchReport,
    apply_upstream_tracker_patch,
)


DEFAULT_WORK_ROOT = Path("/mnt/lexar4tb/multi_uav_lts")


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _competition_environment(args: argparse.Namespace) -> dict[str, str]:
    return {
        "RAFT_UAV_LTS_PRESERVE_INITIAL_IDS": _bool_text(args.preserve_initial_ids),
        "RAFT_UAV_LTS_CONFIRMED_OUTPUT": _bool_text(args.confirmed_output),
        "RAFT_UAV_LTS_COAST_FRAMES": str(args.coast_frames),
        "RAFT_UAV_LTS_CLOSED_WORLD": _bool_text(args.closed_world),
        "RAFT_UAV_LTS_ASSOCIATION_MODE": args.association_mode,
        "RAFT_UAV_LTS_NWD_WEIGHT": str(args.nwd_weight),
        "RAFT_UAV_LTS_NWD_SCALE": str(args.nwd_scale),
        "RAFT_UAV_LTS_APPEARANCE_WEIGHT": str(args.appearance_weight),
        "RAFT_UAV_LTS_APPEARANCE_MIN_SIDE": str(args.appearance_min_side),
        "RAFT_UAV_LTS_MOTION_GATE": _bool_text(args.motion_gate),
    }


def _baseline_paths(argv: list[str]) -> tuple[Path, Path, Path, bool]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--botsort-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    known, _unknown = parser.parse_known_args(argv)
    work_root = known.work_root.expanduser()
    botsort_root = known.botsort_root or (
        work_root / "repos/YOLOv12-BoT-SORT-ReID/BoT-SORT"
    )
    output_dir = known.output_dir or work_root / "outputs/competition_tracker"
    return work_root, botsort_root, output_dir, known.output_dir is not None


def _load_baseline_main() -> Callable[[list[str] | None], int]:
    path = REPO_ROOT / "scripts/run_multi_uav_lts_official_baseline.py"
    spec = importlib.util.spec_from_file_location(
        "raft_uav_multi_uav_lts_official_baseline",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import baseline runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _write_run_metadata(
    path: Path,
    *,
    environment: dict[str, str],
    patch_report: UpstreamPatchReport | None,
    baseline_argv: list[str],
) -> None:
    payload = {
        "schema": "raft-uav-multi-uav-lts-competition-tracker-v1",
        "environment": environment,
        "upstream_patch": None if patch_report is None else asdict(patch_report),
        "baseline_argv": baseline_argv,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "All unrecognized arguments are forwarded to "
            "run_multi_uav_lts_official_baseline.py."
        ),
    )
    parser.add_argument(
        "--association-mode",
        choices=("gated-weighted", "legacy-min"),
        default="gated-weighted",
    )
    parser.add_argument("--nwd-weight", type=float, default=0.5)
    parser.add_argument("--nwd-scale", type=float, default=20.0)
    parser.add_argument("--appearance-weight", type=float, default=0.25)
    parser.add_argument("--appearance-min-side", type=float, default=16.0)
    parser.add_argument("--coast-frames", type=int, default=1)
    parser.add_argument(
        "--closed-world",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="restrict output identities to the first-frame identity bank",
    )
    parser.add_argument(
        "--confirmed-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="suppress detector-born tracks until they receive a second match",
    )
    parser.add_argument(
        "--preserve-initial-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the object-id column from first-frame labels",
    )
    parser.add_argument(
        "--motion-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reject associations outside the Kalman innovation gate",
    )
    parser.add_argument(
        "--no-upstream-patch",
        action="store_true",
        help="delegate without modifying the external checkout",
    )
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="apply the patch and write metadata without running inference",
    )
    parser.add_argument(
        "--verify-upstream-only",
        action="store_true",
        help="validate patch compatibility without modifying or running the checkout",
    )
    parser.add_argument("--patch-report-json", type=Path)
    return parser


def _validate_options(args: argparse.Namespace) -> None:
    if not 0.0 <= args.nwd_weight <= 1.0:
        raise ValueError("--nwd-weight must be in [0, 1]")
    if args.nwd_scale <= 0.0:
        raise ValueError("--nwd-scale must be positive")
    if not 0.0 <= args.appearance_weight <= 1.0:
        raise ValueError("--appearance-weight must be in [0, 1]")
    if args.appearance_min_side <= 0.0:
        raise ValueError("--appearance-min-side must be positive")
    if args.coast_frames < 0:
        raise ValueError("--coast-frames must be non-negative")
    if args.no_upstream_patch and (args.patch_only or args.verify_upstream_only):
        raise ValueError(
            "--no-upstream-patch cannot be combined with patch-only verification modes"
        )
    if args.patch_only and args.verify_upstream_only:
        raise ValueError("choose either --patch-only or --verify-upstream-only")


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args, baseline_argv = parser.parse_known_args(raw_argv)
    _validate_options(args)

    _work_root, botsort_root, output_dir, explicit_output_dir = _baseline_paths(
        baseline_argv
    )
    if not explicit_output_dir:
        baseline_argv.extend(["--output-dir", str(output_dir)])

    environment = _competition_environment(args)
    for name, value in environment.items():
        os.environ[name] = value

    patch_report: UpstreamPatchReport | None = None
    if not args.no_upstream_patch:
        patch_report = apply_upstream_tracker_patch(
            botsort_root,
            dry_run=args.verify_upstream_only,
        )

    report_path = args.patch_report_json or output_dir / "competition_tracker_config.json"
    _write_run_metadata(
        report_path,
        environment=environment,
        patch_report=patch_report,
        baseline_argv=baseline_argv,
    )
    print(f"multi_uav_lts_competition_config={report_path}")

    if args.patch_only or args.verify_upstream_only:
        return 0
    baseline_main = _load_baseline_main()
    return int(baseline_main(baseline_argv))


if __name__ == "__main__":
    raise SystemExit(main())
