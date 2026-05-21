"""Best non-oracle RaFT-UAV experiment preset.

This command intentionally stays away from the truth-gated/oracle association
paths.  It combines the strongest currently exposed non-oracle components in a
single reproducible entry point:

* range-covariance tracklet-Viterbi radar association,
* IMM replay of the selected RF/radar sequence,
* Student-t robust measurement updates, and
* fixed-lag RTS smoothing before metrics are computed.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from raft_uav import tracklet_viterbi_cli as _tracklet_cli

_DEFAULT_OUTPUT_DIR = Path("outputs/best-nontruth")
_DEFAULT_SMOOTHER_LAG_S = 20.0
_DEFAULT_ROBUST_UPDATE = "student-t"
_DEFAULT_RADAR_CATPROB_THRESHOLD = 0.4
_ROBUST_UPDATE_CHOICES = ("none", "nis-inflate", "student-t", "huber")
_TRACKLET_CATPROB_RETENTION_CHOICES = ("hard", "soft")
_LEARNED_CANDIDATE_SCORE_CHOICES = ("additive", "replace")


def main(argv: list[str] | None = None) -> int:
    """Run the best non-oracle preset or print its expanded command."""

    args = _parse_args(argv)
    forwarded_argv = build_tracklet_cli_argv(args)
    if args.dry_run:
        print("raft-uav " + shlex.join(forwarded_argv))
        return 0
    return _tracklet_cli.main(forwarded_argv)


def build_tracklet_cli_argv(args: argparse.Namespace) -> list[str]:
    """Return argv for :mod:`raft_uav.tracklet_viterbi_cli`.

    Keeping this as a pure function makes the preset testable without requiring
    the large AERPAW data files.
    """

    forwarded = [
        "--tracklet-variant",
        "range-covariance",
        "--tracklet-replay-tracker",
        "imm",
        "run-baseline",
        str(args.dataset_root),
        "--flight",
        args.flight,
        "--output-dir",
        args.output_dir.as_posix(),
        "--radar-catprob-threshold",
        _format_float(args.radar_catprob_threshold),
        "--acceleration-std",
        _format_float(args.acceleration_std),
        "--radar-association",
        "tracklet-viterbi",
        "--smoother",
        "fixed-lag",
        "--smoother-lag-s",
        _format_float(args.smoother_lag_s),
        "--max-eval-time-delta-s",
        _format_float(args.max_eval_time_delta_s),
        "--robust-update",
        args.robust_update,
        "--rf-gate-prob",
        _format_float(args.rf_gate_prob),
        "--radar-gate-prob",
        _format_float(args.radar_gate_prob),
        "--rf-inflation-alpha",
        _format_float(args.rf_inflation_alpha),
        "--radar-inflation-alpha",
        _format_float(args.radar_inflation_alpha),
        "--rf-safety-gate-prob",
        _format_float(args.rf_safety_gate_prob),
        "--radar-safety-gate-prob",
        _format_float(args.radar_safety_gate_prob),
        "--rf-max-residual-m",
        _format_float(args.rf_max_residual_m),
        "--radar-max-residual-m",
        _format_float(args.radar_max_residual_m),
        "--tracklet-catprob-retention-mode",
        args.tracklet_catprob_retention_mode,
        "--tracklet-max-candidates",
        str(args.tracklet_max_candidates),
        "--tracklet-max-candidate-pool-per-frame",
        str(args.tracklet_max_candidate_pool_per_frame),
        "--tracklet-max-candidates-per-track-id",
        str(args.tracklet_max_candidates_per_track_id),
        "--tracklet-missed-detection-cost",
        _format_float(args.tracklet_missed_detection_cost),
        "--tracklet-track-switch-cost",
        _format_float(args.tracklet_track_switch_cost),
        "--tracklet-anchor-nis-weight",
        _format_float(args.tracklet_anchor_nis_weight),
        "--tracklet-transition-nis-weight",
        _format_float(args.tracklet_transition_nis_weight),
        "--tracklet-velocity-nis-weight",
        _format_float(args.tracklet_velocity_nis_weight),
    ]
    if args.disable_association_safety_gate:
        forwarded.append("--disable-association-safety-gate")
    if args.calibration_bundle is not None:
        forwarded.extend(["--calibration-bundle", args.calibration_bundle.as_posix()])
    if args.learned_candidate_model is not None:
        forwarded.extend(
            [
                "--tracklet-learned-candidate-model",
                args.learned_candidate_model.as_posix(),
                "--tracklet-learned-candidate-score-mode",
                args.learned_candidate_score_mode,
            ]
        )
    if args.enable_radar_velocity_update:
        forwarded.extend(
            [
                "--enable-radar-velocity-update",
                "--radar-velocity-std-mps",
                _format_float(args.radar_velocity_std_mps),
            ]
        )
    return forwarded


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="raft-uav-best-non-oracle",
        description=(
            "Run the non-oracle result preset: range-covariance tracklet Viterbi, "
            "IMM replay, Student-t robust updates, and fixed-lag RTS smoothing."
        ),
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-bundle", type=Path, default=None)
    parser.add_argument("--acceleration-std", type=_positive_float, default=4.0)
    parser.add_argument("--radar-catprob-threshold", type=_probability, default=_DEFAULT_RADAR_CATPROB_THRESHOLD)
    parser.add_argument("--smoother-lag-s", type=_positive_float, default=_DEFAULT_SMOOTHER_LAG_S)
    parser.add_argument("--max-eval-time-delta-s", type=_positive_float, default=2.0)
    parser.add_argument(
        "--robust-update",
        choices=_ROBUST_UPDATE_CHOICES,
        default=_DEFAULT_ROBUST_UPDATE,
        help="robust update used by the preset; use 'none' for a pure Kalman update",
    )
    parser.add_argument("--rf-gate-prob", type=_probability, default=0.99)
    parser.add_argument("--radar-gate-prob", type=_probability, default=0.99)
    parser.add_argument("--rf-inflation-alpha", type=_positive_float, default=1.0)
    parser.add_argument("--radar-inflation-alpha", type=_positive_float, default=1.0)
    parser.add_argument("--rf-safety-gate-prob", type=_probability, default=0.9999999)
    parser.add_argument("--radar-safety-gate-prob", type=_probability, default=0.9999999)
    parser.add_argument("--rf-max-residual-m", type=_nonnegative_float, default=750.0)
    parser.add_argument("--radar-max-residual-m", type=_nonnegative_float, default=0.0)
    parser.add_argument(
        "--disable-association-safety-gate",
        action="store_true",
        help="forward the base CLI flag that disables the hard RF/radar safety gate",
    )
    parser.add_argument(
        "--learned-candidate-model",
        type=Path,
        default=None,
        help="LOFO-trained learned radar-candidate model used as a tracklet unary term",
    )
    parser.add_argument(
        "--learned-candidate-score-mode",
        choices=_LEARNED_CANDIDATE_SCORE_CHOICES,
        default="additive",
    )
    parser.add_argument(
        "--tracklet-catprob-retention-mode",
        choices=_TRACKLET_CATPROB_RETENTION_CHOICES,
        default="soft",
    )
    parser.add_argument("--tracklet-max-candidates", type=_positive_int, default=8)
    parser.add_argument("--tracklet-max-candidate-pool-per-frame", type=_positive_int, default=24)
    parser.add_argument("--tracklet-max-candidates-per-track-id", type=_nonnegative_int, default=1)
    parser.add_argument("--tracklet-missed-detection-cost", type=_nonnegative_float, default=7.0)
    parser.add_argument("--tracklet-track-switch-cost", type=_nonnegative_float, default=8.0)
    parser.add_argument("--tracklet-anchor-nis-weight", type=_nonnegative_float, default=0.35)
    parser.add_argument("--tracklet-transition-nis-weight", type=_nonnegative_float, default=1.0)
    parser.add_argument("--tracklet-velocity-nis-weight", type=_nonnegative_float, default=0.15)
    parser.add_argument(
        "--enable-radar-velocity-update",
        action="store_true",
        help="include Fortem velocity in radar replay updates for deliberate ablations",
    )
    parser.add_argument("--radar-velocity-std-mps", type=_positive_float, default=12.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the equivalent raft-uav command without reading the dataset",
    )
    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must satisfy 0 < p < 1")
    return parsed


def _format_float(value: float) -> str:
    return f"{value:g}"


if __name__ == "__main__":
    raise SystemExit(main())
