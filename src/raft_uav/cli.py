"""Command-line entry points for RaFT-UAV experiments."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import run_async_cv_baseline
from raft_uav.baselines.radar_association import (
    RADAR_ASSOCIATION_MODES,
    RADAR_COVARIANCE_MODELS,
    run_async_cv_baseline_with_radar_association,
)
from raft_uav.baselines.smoothing import SMOOTHER_MODES, smooth_tracking_records
from raft_uav.baselines.update_logic import (
    DEFAULT_HUBER_THRESHOLD,
    DEFAULT_STUDENT_T_DOF,
    ROBUST_UPDATE_MODES,
)
from raft_uav.calibration.bundle import apply_calibration_bundle, load_calibration_bundle
from raft_uav.calibration.time_offset import apply_time_offset
from raft_uav.evaluation.diagnostics import build_diagnostic_summary
from raft_uav.evaluation.metrics import (
    position_errors_m,
    sampled_position_errors_m,
    summarize_errors,
)
from raft_uav.io.aerpaw import (
    DEFAULT_RADAR_CLOCK_OFFSET_S,
    DEFAULT_RF_CLOCK_OFFSET_S,
    RADAR_SELECTION_MODES,
    discover_flights,
    normalize_radar,
    normalize_rf,
    normalize_truth,
    radar_measurements_to_enu,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    rf_measurements_to_enu,
    select_flight,
    select_radar_measurement_rows,
    summarize_flight_schema,
)
from raft_uav.numeric import optional_float as _optional_float
from raft_uav.numeric import optional_int as _optional_int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raft-uav")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="list discovered AERPAW flights")
    inspect_parser.add_argument("dataset_root", type=Path)
    inspect_parser.add_argument(
        "--flight",
        action="append",
        help="inspect only this flight; can be passed multiple times",
    )
    inspect_parser.add_argument(
        "--rf-clock-offset-s",
        type=float,
        default=DEFAULT_RF_CLOCK_OFFSET_S,
        help="RF raw timestamp offset added before truth-relative normalization",
    )
    inspect_parser.add_argument(
        "--radar-clock-offset-s",
        type=float,
        default=DEFAULT_RADAR_CLOCK_OFFSET_S,
        help="radar raw timestamp offset added before truth-relative normalization",
    )

    baseline_parser = subparsers.add_parser(
        "run-baseline", help="run the initial CV fusion baseline"
    )
    baseline_parser.add_argument("dataset_root", type=Path)
    baseline_parser.add_argument("--flight", required=True)
    baseline_parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    baseline_parser.add_argument("--acceleration-std", type=float, default=4.0)
    baseline_parser.add_argument(
        "--rf-clock-offset-s",
        type=float,
        default=DEFAULT_RF_CLOCK_OFFSET_S,
        help=(
            "RF raw timestamp offset, in seconds, added before truth-relative "
            "normalization"
        ),
    )
    baseline_parser.add_argument(
        "--radar-clock-offset-s",
        type=float,
        default=DEFAULT_RADAR_CLOCK_OFFSET_S,
        help=(
            "radar raw timestamp offset, in seconds, added before truth-relative "
            "normalization"
        ),
    )
    baseline_parser.add_argument(
        "--rf-time-offset-correction-s",
        type=float,
        default=0.0,
        help="residual calibrated correction added to normalized RF time_s",
    )
    baseline_parser.add_argument(
        "--radar-time-offset-correction-s",
        type=float,
        default=0.0,
        help="residual calibrated correction added to normalized radar time_s",
    )
    baseline_parser.add_argument(
        "--calibration-bundle",
        type=Path,
        default=None,
        help="LOFO calibration manifest with time offsets, bias, and uncertainty models",
    )
    baseline_parser.add_argument(
        "--radar-association",
        choices=["catprob", *RADAR_ASSOCIATION_MODES],
        default="catprob",
        help="radar association mode for choosing trackData rows before radar updates",
    )
    baseline_parser.add_argument(
        "--radar-selection",
        choices=RADAR_SELECTION_MODES,
        default=None,
        help="legacy radar row selection; overrides --radar-association when provided",
    )
    baseline_parser.add_argument("--radar-catprob-threshold", type=float, default=0.5)
    baseline_parser.add_argument(
        "--radar-covariance-model",
        choices=RADAR_COVARIANCE_MODELS,
        default="cartesian",
        help=(
            "Radar measurement covariance model for association modes. "
            "'cartesian' keeps the fixed ENU diagonal covariance; 'geometry' builds "
            "a row-wise line-of-sight/cross-range ENU covariance from radar range."
        ),
    )
    baseline_parser.add_argument(
        "--radar-range-std-m",
        type=float,
        default=12.0,
        help="Base radial radar standard deviation for --radar-covariance-model geometry.",
    )
    baseline_parser.add_argument(
        "--radar-range-std-fraction",
        type=float,
        default=0.005,
        help=(
            "Range-proportional radial radar standard deviation fraction for "
            "--radar-covariance-model geometry."
        ),
    )
    baseline_parser.add_argument(
        "--radar-crossrange-angle-std-deg",
        type=float,
        default=1.5,
        help=(
            "Angular cross-range radar standard deviation in degrees for "
            "--radar-covariance-model geometry."
        ),
    )
    baseline_parser.add_argument(
        "--radar-crossrange-min-std-m",
        type=float,
        default=5.0,
        help="Minimum cross-range standard deviation for geometry radar covariance.",
    )
    baseline_parser.add_argument(
        "--radar-crossrange-max-std-m",
        type=float,
        default=80.0,
        help="Maximum cross-range standard deviation for geometry radar covariance.",
    )
    baseline_parser.add_argument("--truth-gate-m", type=float, default=150.0)
    baseline_parser.add_argument("--truth-time-gate-s", type=float, default=1.0)
    baseline_parser.add_argument(
        "--track-switch-nis-ratio",
        type=float,
        default=0.5,
        help="track-continuity switches IDs only when best NIS is below this ratio",
    )
    baseline_parser.add_argument(
        "--geometry-velocity-std",
        type=float,
        default=12.0,
        help="velocity standard deviation used by --radar-association geometry-score",
    )
    baseline_parser.add_argument(
        "--geometry-velocity-weight",
        type=float,
        default=0.25,
        help="weight for the radar velocity consistency term in geometry-score",
    )
    baseline_parser.add_argument(
        "--geometry-switch-penalty",
        type=float,
        default=4.0,
        help="NIS-scale penalty for changing Fortem track IDs in geometry-score",
    )
    baseline_parser.add_argument(
        "--geometry-catprob-weight",
        type=float,
        default=2.0,
        help="weight for low UAV class-probability penalty in geometry-score",
    )
    baseline_parser.add_argument(
        "--rf-anchor-weight",
        type=float,
        default=0.35,
        help="RF anchor penalty weight for rf-anchored-nis/rf-gated-nis",
    )
    baseline_parser.add_argument(
        "--rf-anchor-time-gate-s",
        type=float,
        default=2.0,
        help="maximum age of latest RF update used by RF-anchor association modes",
    )
    baseline_parser.add_argument(
        "--rf-anchor-nis-cap",
        type=float,
        default=25.0,
        help="per-candidate RF anchor NIS cap used by RF-anchor association modes",
    )
    baseline_parser.add_argument(
        "--rf-anchor-gate-nis",
        type=float,
        default=25.0,
        help="hard RF-anchor NIS gate used by --radar-association rf-gated-nis",
    )
    baseline_parser.add_argument(
        "--pda-nis-temperature",
        type=float,
        default=1.0,
        help="softmax temperature for NIS likelihoods in pda-mixture association",
    )
    baseline_parser.add_argument(
        "--pda-catprob-exponent",
        type=float,
        default=1.0,
        help="exponent for UAV class-probability priors in pda-mixture association",
    )
    baseline_parser.add_argument("--track-bank-max-hypotheses", type=int, default=16)
    baseline_parser.add_argument("--track-bank-max-assignments", type=int, default=16)
    baseline_parser.add_argument("--track-bank-max-candidates", type=int, default=16)
    baseline_parser.add_argument("--track-bank-gate-prob", type=float, default=0.9999999)
    baseline_parser.add_argument("--track-bank-detection-prob", type=float, default=0.999)
    baseline_parser.add_argument("--track-bank-clutter-intensity", type=float, default=1.0e-12)
    baseline_parser.add_argument("--track-bank-prune-delta", type=float, default=80.0)
    baseline_parser.add_argument(
        "--stable-segment-min-frames",
        type=int,
        default=100,
        help="minimum contiguous rows for --radar-association stable-segments",
    )
    baseline_parser.add_argument(
        "--stable-segment-max-transition-speed-mps",
        type=float,
        default=65.0,
        help="maximum stitched-segment transition speed for stable-segments",
    )
    baseline_parser.add_argument(
        "--stable-segment-range-gate-m",
        type=float,
        default=800.0,
        help="radar range gate for stable-segments; <=0 disables the range gate",
    )
    baseline_parser.add_argument(
        "--stable-segment-interpolation-max-gap-s",
        type=float,
        default=5.0,
        help=(
            "maximum anchor-to-anchor gap for stable-segments-interpolated; "
            "<=0 disables the gap cap"
        ),
    )
    baseline_parser.add_argument(
        "--stable-segment-interpolation-max-speed-mps",
        type=float,
        default=65.0,
        help=(
            "maximum anchor-to-anchor speed for stable-segments-interpolated; "
            "<=0 disables the speed cap"
        ),
    )
    baseline_parser.add_argument(
        "--stable-segment-interpolation-std-scale",
        type=float,
        default=2.0,
        help=(
            "measurement standard-deviation scale applied to interpolated "
            "stable-segment radar updates"
        ),
    )
    baseline_parser.add_argument(
        "--stable-segment-interpolation-gap-std-mps",
        type=float,
        default=12.0,
        help=(
            "extra position standard deviation per second from the nearest "
            "real stable-segment anchor"
        ),
    )
    baseline_parser.add_argument(
        "--stable-segment-rf-score-weight",
        type=float,
        default=1.0,
        help="RF-NIS penalty weight used when stitching stable radar segments",
    )
    baseline_parser.add_argument(
        "--stable-segment-rf-time-gate-s",
        type=float,
        default=2.0,
        help="maximum RF-to-segment time distance used for stable segment scoring",
    )
    baseline_parser.add_argument(
        "--stable-segment-rf-nis-cap",
        type=float,
        default=25.0,
        help="per-RF NIS cap used in stable segment RF-consistency scoring",
    )
    baseline_parser.add_argument(
        "--smoother",
        choices=SMOOTHER_MODES,
        default="none",
        help="post-filter smoothing mode applied before metrics are computed",
    )
    baseline_parser.add_argument(
        "--smoother-lag-s",
        type=float,
        default=20.0,
        help="future horizon for --smoother fixed-lag",
    )
    baseline_parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    baseline_parser.add_argument(
        "--enable-gating",
        action="store_true",
        help="enable normalized-innovation-squared Mahalanobis gates before updates",
    )
    baseline_parser.add_argument(
        "--robust-update",
        choices=["none", *ROBUST_UPDATE_MODES],
        default="none",
        help=(
            "robust update rule; nis-inflate gates via NIS then inflates, "
            "student-t applies Student-t covariance scaling, and huber applies "
            "multivariate Huber covariance scaling"
        ),
    )
    baseline_parser.add_argument(
        "--rf-gate-prob",
        type=float,
        default=0.99,
        help=(
            "chi-square gate probability for 2D RF updates when gating is enabled, "
            "and the NIS threshold used by --robust-update nis-inflate"
        ),
    )
    baseline_parser.add_argument(
        "--radar-gate-prob",
        type=float,
        default=0.99,
        help=(
            "chi-square gate probability for 3D radar updates when gating is enabled, "
            "and the NIS threshold used by --robust-update nis-inflate"
        ),
    )
    baseline_parser.add_argument(
        "--disable-association-safety-gate",
        action="store_true",
        help="disable the hard RF/radar safety gate that turns impossible updates into misses",
    )
    baseline_parser.add_argument(
        "--rf-safety-gate-prob",
        type=float,
        default=0.9999999,
        help="hard chi-square gate probability for rejecting impossible 2D RF updates",
    )
    baseline_parser.add_argument(
        "--radar-safety-gate-prob",
        type=float,
        default=0.9999999,
        help="hard chi-square gate probability for rejecting impossible 3D radar updates",
    )
    baseline_parser.add_argument(
        "--rf-max-residual-m",
        type=float,
        default=750.0,
        help=(
            "Euclidean RF residual safety cap; with a safety NIS gate it rejects only "
            "statistically implausible updates, <=0 disables it"
        ),
    )
    baseline_parser.add_argument(
        "--radar-max-residual-m",
        type=float,
        default=0.0,
        help=(
            "Euclidean radar residual safety cap; with a safety NIS gate it rejects only "
            "statistically implausible updates, <=0 disables it"
        ),
    )
    baseline_parser.add_argument(
        "--rf-inflation-alpha",
        type=float,
        default=1.0,
        help="RF exponent for --robust-update nis-inflate covariance scaling",
    )
    baseline_parser.add_argument(
        "--radar-inflation-alpha",
        type=float,
        default=1.0,
        help="radar exponent for --robust-update nis-inflate covariance scaling",
    )
    baseline_parser.add_argument(
        "--enable-radar-velocity-update",
        action="store_true",
        help=(
            "include Fortem velocity components in radar measurement updates when "
            "they are available; use only as an explicit ablation"
        ),
    )
    baseline_parser.add_argument(
        "--radar-velocity-std-mps",
        type=float,
        default=12.0,
        help="radar velocity standard deviation used when --enable-radar-velocity-update is set",
    )

    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(
            args.dataset_root, args.flight, args.rf_clock_offset_s, args.radar_clock_offset_s
        )
    if args.command == "run-baseline":
        return _run_baseline(
            args.dataset_root,
            args.flight,
            args.output_dir,
            args.acceleration_std,
            args.radar_association,
            args.radar_selection,
            args.rf_clock_offset_s,
            args.radar_clock_offset_s,
            args.rf_time_offset_correction_s,
            args.radar_time_offset_correction_s,
            args.calibration_bundle,
            args.radar_catprob_threshold,
            args.radar_covariance_model,
            args.radar_range_std_m,
            args.radar_range_std_fraction,
            args.radar_crossrange_angle_std_deg,
            args.radar_crossrange_min_std_m,
            args.radar_crossrange_max_std_m,
            args.truth_gate_m,
            args.truth_time_gate_s,
            args.track_switch_nis_ratio,
            args.geometry_velocity_std,
            args.geometry_velocity_weight,
            args.geometry_switch_penalty,
            args.geometry_catprob_weight,
            args.rf_anchor_weight,
            args.rf_anchor_time_gate_s,
            args.rf_anchor_nis_cap,
            args.rf_anchor_gate_nis,
            args.pda_nis_temperature,
            args.pda_catprob_exponent,
            args.track_bank_max_hypotheses,
            args.track_bank_max_assignments,
            args.track_bank_max_candidates,
            args.track_bank_gate_prob,
            args.track_bank_detection_prob,
            args.track_bank_clutter_intensity,
            args.track_bank_prune_delta,
            args.stable_segment_min_frames,
            args.stable_segment_max_transition_speed_mps,
            args.stable_segment_range_gate_m,
            args.stable_segment_interpolation_max_gap_s,
            args.stable_segment_interpolation_max_speed_mps,
            args.stable_segment_interpolation_std_scale,
            args.stable_segment_interpolation_gap_std_mps,
            args.stable_segment_rf_score_weight,
            args.stable_segment_rf_time_gate_s,
            args.stable_segment_rf_nis_cap,
            args.smoother,
            args.smoother_lag_s,
            args.max_eval_time_delta_s,
            args.enable_gating,
            args.robust_update,
            args.rf_gate_prob,
            args.radar_gate_prob,
            not args.disable_association_safety_gate,
            args.rf_safety_gate_prob,
            args.radar_safety_gate_prob,
            args.rf_max_residual_m,
            args.radar_max_residual_m,
            args.rf_inflation_alpha,
            args.radar_inflation_alpha,
            args.enable_radar_velocity_update,
            args.radar_velocity_std_mps,
        )
    raise ValueError(args.command)


def _inspect(
    dataset_root: Path,
    requested_flights: list[str] | None,
    rf_clock_offset_s: float,
    radar_clock_offset_s: float,
) -> int:
    if requested_flights:
        flights = [select_flight(dataset_root, name) for name in requested_flights]
        discovered_count = len(discover_flights(dataset_root))
    else:
        flights = discover_flights(dataset_root)
        discovered_count = len(flights)

    print(f"discovered_flights={discovered_count}")
    for flight in flights:
        summary = summarize_flight_schema(
            flight,
            rf_clock_offset_s=rf_clock_offset_s,
            radar_clock_offset_s=radar_clock_offset_s,
        )
        print(f"\nflight={summary['flight']}")
        for modality in ("truth", "rf", "radar"):
            _print_modality_summary(modality, summary.get(modality))
    return 0


def _run_baseline(
    dataset_root: Path,
    flight_name: str,
    output_dir: Path,
    acceleration_std: float,
    radar_association: str,
    legacy_radar_selection: str | None,
    rf_clock_offset_s: float,
    radar_clock_offset_s: float,
    rf_time_offset_correction_s: float,
    radar_time_offset_correction_s: float,
    calibration_bundle_path: Path | None,
    radar_catprob_threshold: float,
    radar_covariance_model: str,
    radar_range_std_m: float,
    radar_range_std_fraction: float,
    radar_crossrange_angle_std_deg: float,
    radar_crossrange_min_std_m: float,
    radar_crossrange_max_std_m: float,
    truth_gate_m: float,
    truth_time_gate_s: float,
    track_switch_nis_ratio: float,
    geometry_velocity_std: float,
    geometry_velocity_weight: float,
    geometry_switch_penalty: float,
    geometry_catprob_weight: float,
    rf_anchor_weight: float,
    rf_anchor_time_gate_s: float,
    rf_anchor_nis_cap: float,
    rf_anchor_gate_nis: float,
    pda_nis_temperature: float,
    pda_catprob_exponent: float,
    track_bank_max_hypotheses: int,
    track_bank_max_assignments: int,
    track_bank_max_candidates: int,
    track_bank_gate_prob: float,
    track_bank_detection_prob: float,
    track_bank_clutter_intensity: float,
    track_bank_prune_delta: float,
    stable_segment_min_frames: int,
    stable_segment_max_transition_speed_mps: float,
    stable_segment_range_gate_m: float,
    stable_segment_interpolation_max_gap_s: float,
    stable_segment_interpolation_max_speed_mps: float,
    stable_segment_interpolation_std_scale: float,
    stable_segment_interpolation_gap_std_mps: float,
    stable_segment_rf_score_weight: float,
    stable_segment_rf_time_gate_s: float,
    stable_segment_rf_nis_cap: float,
    smoother: str,
    smoother_lag_s: float,
    max_eval_time_delta_s: float,
    enable_gating: bool,
    robust_update: str,
    rf_gate_prob: float,
    radar_gate_prob: float,
    enable_association_safety_gate: bool,
    rf_safety_gate_prob: float,
    radar_safety_gate_prob: float,
    rf_max_residual_m: float,
    radar_max_residual_m: float,
    rf_inflation_alpha: float,
    radar_inflation_alpha: float,
    enable_radar_velocity_update: bool,
    radar_velocity_std_mps: float,
) -> int:
    if enable_gating and robust_update != "none":
        raise ValueError("--enable-gating and --robust-update are mutually exclusive")
    if robust_update == "nis-inflate" and (
        rf_inflation_alpha <= 0.0 or radar_inflation_alpha <= 0.0
    ):
        raise ValueError("inflation alphas must be positive")
    if track_switch_nis_ratio <= 0.0:
        raise ValueError("track_switch_nis_ratio must be positive")
    if rf_anchor_weight < 0.0:
        raise ValueError("rf_anchor_weight must be nonnegative")
    if rf_anchor_time_gate_s < 0.0:
        raise ValueError("rf_anchor_time_gate_s must be nonnegative")
    if rf_anchor_nis_cap <= 0.0:
        raise ValueError("rf_anchor_nis_cap must be positive")
    if rf_anchor_gate_nis <= 0.0:
        raise ValueError("rf_anchor_gate_nis must be positive")
    if pda_nis_temperature <= 0.0:
        raise ValueError("pda_nis_temperature must be positive")
    if pda_catprob_exponent < 0.0:
        raise ValueError("pda_catprob_exponent must be nonnegative")
    if track_bank_max_hypotheses < 1:
        raise ValueError("track_bank_max_hypotheses must be positive")
    if track_bank_max_assignments < 1:
        raise ValueError("track_bank_max_assignments must be positive")
    if track_bank_max_candidates < 1:
        raise ValueError("track_bank_max_candidates must be positive")
    if stable_segment_min_frames < 1:
        raise ValueError("stable_segment_min_frames must be positive")
    if stable_segment_max_transition_speed_mps <= 0.0:
        raise ValueError("stable_segment_max_transition_speed_mps must be positive")
    if stable_segment_range_gate_m < 0.0:
        raise ValueError("stable_segment_range_gate_m must be nonnegative")
    if stable_segment_interpolation_max_gap_s < 0.0:
        raise ValueError("stable_segment_interpolation_max_gap_s must be nonnegative")
    if stable_segment_interpolation_max_speed_mps < 0.0:
        raise ValueError("stable_segment_interpolation_max_speed_mps must be nonnegative")
    if stable_segment_interpolation_std_scale <= 0.0:
        raise ValueError("stable_segment_interpolation_std_scale must be positive")
    if stable_segment_interpolation_gap_std_mps < 0.0:
        raise ValueError("stable_segment_interpolation_gap_std_mps must be nonnegative")
    if stable_segment_rf_score_weight < 0.0:
        raise ValueError("stable_segment_rf_score_weight must be nonnegative")
    if stable_segment_rf_time_gate_s < 0.0:
        raise ValueError("stable_segment_rf_time_gate_s must be nonnegative")
    if stable_segment_rf_nis_cap <= 0.0:
        raise ValueError("stable_segment_rf_nis_cap must be positive")
    if radar_covariance_model not in RADAR_COVARIANCE_MODELS:
        raise ValueError(f"--radar-covariance-model must be one of {RADAR_COVARIANCE_MODELS}")
    if radar_range_std_m <= 0.0:
        raise ValueError("--radar-range-std-m must be positive")
    if radar_range_std_fraction < 0.0:
        raise ValueError("--radar-range-std-fraction must be nonnegative")
    if radar_crossrange_angle_std_deg <= 0.0:
        raise ValueError("--radar-crossrange-angle-std-deg must be positive")
    if radar_crossrange_min_std_m <= 0.0:
        raise ValueError("--radar-crossrange-min-std-m must be positive")
    if radar_velocity_std_mps <= 0.0:
        raise ValueError("--radar-velocity-std-mps must be positive")
    if radar_crossrange_max_std_m < radar_crossrange_min_std_m:
        raise ValueError("--radar-crossrange-max-std-m must be >= --radar-crossrange-min-std-m")
    if smoother == "fixed-lag" and smoother_lag_s < 0.0:
        raise ValueError("smoother_lag_s must be nonnegative for fixed-lag smoothing")
    radar_mode = legacy_radar_selection or radar_association
    if radar_covariance_model != "cartesian" and radar_mode not in RADAR_ASSOCIATION_MODES:
        raise ValueError(
            "--radar-covariance-model geometry requires a radar association mode; "
            f"got {radar_mode!r}"
        )
    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")

    truth_raw = read_truth(flight.truth_txt)
    truth, projector, truth_origin_time = normalize_truth(truth_raw)

    rf = pd.DataFrame()
    radar = pd.DataFrame()
    selected_radar = pd.DataFrame()
    measurements = []
    rf_measurements = []
    calibration_summary: dict[str, Any] | None = None
    if flight.rf_csv is not None:
        rf = _inside_truth_window(
            _apply_time_offset_correction(
                normalize_rf(
                    read_rf_csv(flight.rf_csv),
                    projector,
                    truth_origin_time,
                    clock_offset_s=rf_clock_offset_s,
                ),
                rf_time_offset_correction_s,
            ),
            truth,
        )
    if flight.radar_json is not None:
        radar = _inside_truth_window(
            _apply_time_offset_correction(
                normalize_radar(
                    read_radar_tracks_json(flight.radar_json),
                    projector,
                    truth_origin_time,
                    clock_offset_s=radar_clock_offset_s,
                ),
                radar_time_offset_correction_s,
            ),
            truth,
        )

    if calibration_bundle_path is not None:
        bundle = load_calibration_bundle(calibration_bundle_path)
        rf, radar, calibration_summary = apply_calibration_bundle(
            rf=rf,
            radar=radar,
            bundle=bundle,
        )
        rf = _inside_truth_window(rf, truth)
        radar = _inside_truth_window(radar, truth)

    if not rf.empty:
        rf_measurements = rf_measurements_to_enu(rf)
        measurements.extend(rf_measurements)

    if enable_radar_velocity_update:
        os.environ["RAFT_UAV_RADAR_UPDATE_USES_VELOCITY"] = "1"
        os.environ["RAFT_UAV_RADAR_VELOCITY_STD_MPS"] = f"{radar_velocity_std_mps:g}"

    gate_probabilities = None
    safety_gate_probabilities = None
    max_residual_norms = None
    robust_updates = None
    inflation_alphas = None
    if enable_gating or robust_update == "nis-inflate":
        gate_probabilities = {"rf": rf_gate_prob, "radar": radar_gate_prob}
    if enable_association_safety_gate:
        safety_gate_probabilities = {
            "rf": rf_safety_gate_prob,
            "radar": radar_safety_gate_prob,
        }
        max_residual_norms = {
            "rf": None if rf_max_residual_m <= 0.0 else rf_max_residual_m,
            "radar": None if radar_max_residual_m <= 0.0 else radar_max_residual_m,
        }
    if robust_update != "none":
        robust_updates = {"rf": robust_update, "radar": robust_update}
    if robust_update == "nis-inflate":
        inflation_alphas = {"rf": rf_inflation_alpha, "radar": radar_inflation_alpha}

    if radar_mode in RADAR_ASSOCIATION_MODES:
        records, selected_radar = run_async_cv_baseline_with_radar_association(
            rf_measurements=rf_measurements,
            radar=radar,
            association=radar_mode,
            truth=truth,
            acceleration_std_mps2=acceleration_std,
            radar_covariance_model=radar_covariance_model,
            radar_range_std_m=radar_range_std_m,
            radar_range_std_fraction=radar_range_std_fraction,
            radar_crossrange_angle_std_deg=radar_crossrange_angle_std_deg,
            radar_crossrange_min_std_m=radar_crossrange_min_std_m,
            radar_crossrange_max_std_m=radar_crossrange_max_std_m,
            gate_probabilities_by_source=gate_probabilities,
            safety_gate_probabilities_by_source=safety_gate_probabilities,
            robust_update_by_source=robust_updates,
            inflation_alpha_by_source=inflation_alphas,
            max_residual_norms_by_source=max_residual_norms,
            track_switch_nis_ratio=track_switch_nis_ratio,
            candidate_catprob_threshold=radar_catprob_threshold,
            geometry_velocity_std_mps=geometry_velocity_std,
            geometry_velocity_weight=geometry_velocity_weight,
            geometry_switch_penalty=geometry_switch_penalty,
            geometry_catprob_weight=geometry_catprob_weight,
            rf_anchor_weight=rf_anchor_weight,
            rf_anchor_time_gate_s=rf_anchor_time_gate_s,
            rf_anchor_nis_cap=rf_anchor_nis_cap,
            rf_anchor_gate_nis=rf_anchor_gate_nis,
            pda_nis_temperature=pda_nis_temperature,
            pda_catprob_exponent=pda_catprob_exponent,
            track_bank_max_hypotheses=track_bank_max_hypotheses,
            track_bank_max_assignments=track_bank_max_assignments,
            track_bank_max_candidates=track_bank_max_candidates,
            track_bank_gate_probability=track_bank_gate_prob,
            track_bank_detection_probability=track_bank_detection_prob,
            track_bank_clutter_intensity=track_bank_clutter_intensity,
            track_bank_prune_log_weight_delta=track_bank_prune_delta,
            stable_segment_min_frames=stable_segment_min_frames,
            stable_segment_max_transition_speed_mps=stable_segment_max_transition_speed_mps,
            stable_segment_range_gate_m=(
                None if stable_segment_range_gate_m <= 0.0 else stable_segment_range_gate_m
            ),
            stable_segment_interpolation_max_gap_s=(
                None
                if stable_segment_interpolation_max_gap_s <= 0.0
                else stable_segment_interpolation_max_gap_s
            ),
            stable_segment_interpolation_max_speed_mps=(
                None
                if stable_segment_interpolation_max_speed_mps <= 0.0
                else stable_segment_interpolation_max_speed_mps
            ),
            stable_segment_interpolation_std_scale=stable_segment_interpolation_std_scale,
            stable_segment_interpolation_gap_std_mps=stable_segment_interpolation_gap_std_mps,
            stable_segment_rf_score_weight=stable_segment_rf_score_weight,
            stable_segment_rf_time_gate_s=stable_segment_rf_time_gate_s,
            stable_segment_rf_nis_cap=stable_segment_rf_nis_cap,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
        measurements = [
            *rf_measurements,
            *radar_measurements_to_enu(
                selected_radar,
                include_velocity=enable_radar_velocity_update,
                default_velocity_std_mps=radar_velocity_std_mps,
            ),
        ]
    else:
        selected_radar = select_radar_measurement_rows(
            radar,
            selection=radar_mode,
            truth=truth,
            catprob_threshold=radar_catprob_threshold,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
        measurements.extend(
            radar_measurements_to_enu(
                selected_radar,
                include_velocity=enable_radar_velocity_update,
                default_velocity_std_mps=radar_velocity_std_mps,
            )
        )
        records = run_async_cv_baseline(
            measurements,
            acceleration_std_mps2=acceleration_std,
            gate_probabilities_by_source=gate_probabilities,
            safety_gate_probabilities_by_source=safety_gate_probabilities,
            robust_update_by_source=robust_updates,
            inflation_alpha_by_source=inflation_alphas,
            max_residual_norms_by_source=max_residual_norms,
        )
    if not records:
        raise RuntimeError(f"{flight.name} produced no baseline posterior records")
    records = smooth_tracking_records(
        records,
        method=smoother,
        acceleration_std_mps2=acceleration_std,
        lag_s=smoother_lag_s,
    )

    estimate_frame = _records_to_frame(records)
    diagnostics_columns = [
        "time_s",
        "source",
        "measurement_dim",
        "accepted",
        "update_action",
        "nis",
        "gate_threshold",
        "safety_gate_threshold",
        "residual_gate_threshold_m",
        "covariance_scale",
        "inflation_alpha",
        "residual_norm_m",
    ]
    diagnostics_frame = estimate_frame[diagnostics_columns].copy()

    flight_output = output_dir / flight.name
    flight_output.mkdir(parents=True, exist_ok=True)

    estimates_path = flight_output / "estimates.csv"
    diagnostics_path = flight_output / "diagnostics.csv"
    selected_radar_path = flight_output / "selected_radar.csv"
    selected_radar_attempted = _attempted_selected_radar_frame(selected_radar)
    selected_radar_attempted_path = flight_output / "selected_radar_attempted.csv"
    hypotheses_path = flight_output / "hypotheses.csv"
    metrics_path = flight_output / "metrics.json"
    diagnostic_summary_path = flight_output / "diagnostic_summary.json"
    plot_path = flight_output / "trajectory.png"
    estimate_frame.to_csv(estimates_path, index=False)
    diagnostics_frame.to_csv(diagnostics_path, index=False)
    selected_radar.to_csv(selected_radar_path, index=False)
    selected_radar_attempted.to_csv(selected_radar_attempted_path, index=False)
    _hypotheses_to_frame(records).to_csv(hypotheses_path, index=False)

    metrics = _baseline_metrics(
        flight_name=flight.name,
        flight=flight,
        truth=truth,
        rf=rf,
        radar=radar,
        selected_radar=selected_radar,
        attempted_selected_radar=selected_radar_attempted,
        estimate_frame=estimate_frame,
        acceleration_std=acceleration_std,
        radar_association=radar_mode,
        radar_catprob_threshold=radar_catprob_threshold,
        radar_covariance_model=radar_covariance_model,
        radar_range_std_m=radar_range_std_m,
        radar_range_std_fraction=radar_range_std_fraction,
        radar_crossrange_angle_std_deg=radar_crossrange_angle_std_deg,
        radar_crossrange_min_std_m=radar_crossrange_min_std_m,
        radar_crossrange_max_std_m=radar_crossrange_max_std_m,
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
        track_switch_nis_ratio=track_switch_nis_ratio,
        geometry_velocity_std=geometry_velocity_std,
        geometry_velocity_weight=geometry_velocity_weight,
        geometry_switch_penalty=geometry_switch_penalty,
        geometry_catprob_weight=geometry_catprob_weight,
        rf_anchor_weight=rf_anchor_weight,
        rf_anchor_time_gate_s=rf_anchor_time_gate_s,
        rf_anchor_nis_cap=rf_anchor_nis_cap,
        rf_anchor_gate_nis=rf_anchor_gate_nis,
        pda_nis_temperature=pda_nis_temperature,
        pda_catprob_exponent=pda_catprob_exponent,
        track_bank_max_hypotheses=track_bank_max_hypotheses,
        track_bank_max_assignments=track_bank_max_assignments,
        track_bank_max_candidates=track_bank_max_candidates,
        track_bank_gate_prob=track_bank_gate_prob,
        track_bank_detection_prob=track_bank_detection_prob,
        track_bank_clutter_intensity=track_bank_clutter_intensity,
        track_bank_prune_delta=track_bank_prune_delta,
        stable_segment_min_frames=stable_segment_min_frames,
        stable_segment_max_transition_speed_mps=stable_segment_max_transition_speed_mps,
        stable_segment_range_gate_m=stable_segment_range_gate_m,
        stable_segment_interpolation_max_gap_s=stable_segment_interpolation_max_gap_s,
        stable_segment_interpolation_max_speed_mps=stable_segment_interpolation_max_speed_mps,
        stable_segment_interpolation_std_scale=stable_segment_interpolation_std_scale,
        stable_segment_interpolation_gap_std_mps=stable_segment_interpolation_gap_std_mps,
        stable_segment_rf_score_weight=stable_segment_rf_score_weight,
        stable_segment_rf_time_gate_s=stable_segment_rf_time_gate_s,
        stable_segment_rf_nis_cap=stable_segment_rf_nis_cap,
        smoother=smoother,
        smoother_lag_s=smoother_lag_s,
        max_eval_time_delta_s=max_eval_time_delta_s,
        enable_gating=enable_gating,
        robust_update=robust_update,
        rf_gate_prob=rf_gate_prob,
        radar_gate_prob=radar_gate_prob,
        enable_association_safety_gate=enable_association_safety_gate,
        rf_safety_gate_prob=rf_safety_gate_prob,
        radar_safety_gate_prob=radar_safety_gate_prob,
        rf_max_residual_m=rf_max_residual_m,
        radar_max_residual_m=radar_max_residual_m,
        rf_inflation_alpha=rf_inflation_alpha,
        radar_inflation_alpha=radar_inflation_alpha,
        calibration_summary=calibration_summary,
        enable_radar_velocity_update=enable_radar_velocity_update,
        radar_velocity_std_mps=radar_velocity_std_mps,
    )
    metrics_path.write_text(_json_dump_text(metrics), encoding="utf-8")
    diagnostic_summary = build_diagnostic_summary(
        estimate_frame=estimate_frame,
        selected_radar=selected_radar,
        truth=truth,
        max_eval_time_delta_s=max_eval_time_delta_s,
    )
    diagnostic_summary_path.write_text(
        _json_dump_text(diagnostic_summary),
        encoding="utf-8",
    )
    _write_trajectory_plot(plot_path, truth, rf, selected_radar, estimate_frame, flight.name)

    print(f"flight={flight.name}")
    print(f"measurements={len(measurements)}")
    print(f"posterior_records={len(records)}")
    print(f"accepted_measurements={metrics['accepted_measurements']}")
    print(f"rejected_measurements={metrics['rejected_measurements']}")
    print(f"reweighted_measurements={metrics['reweighted_measurements']}")
    print(f"rf_rows={len(rf)}")
    print(f"radar_rows={len(radar)}")
    print(f"rf_clock_offset_s={rf_clock_offset_s:.3f}")
    print(f"radar_clock_offset_s={radar_clock_offset_s:.3f}")
    print(f"rf_time_offset_correction_s={rf_time_offset_correction_s:.3f}")
    print(f"radar_time_offset_correction_s={radar_time_offset_correction_s:.3f}")
    print(f"radar_association={radar_mode}")
    print(f"selected_radar_rows={len(selected_radar)}")
    print(f"attempted_selected_radar_rows={len(selected_radar_attempted)}")
    print(f"selected_radar_track_ids={metrics['selected_radar_track_ids']}")
    print(f"smoother={smoother}")
    print(f"metrics_json={metrics_path}")
    print(f"diagnostic_summary_json={diagnostic_summary_path}")
    print(f"estimates_csv={estimates_path}")
    print(f"diagnostics_csv={diagnostics_path}")
    print(f"selected_radar_csv={selected_radar_path}")
    print(f"selected_radar_attempted_csv={selected_radar_attempted_path}")
    print(f"hypotheses_csv={hypotheses_path}")
    print(f"trajectory_png={plot_path}")
    print(f"mean_2d_m={_format_optional_metric(metrics['position_error_2d'].get('mean_m'))}")
    print(f"std_2d_m={_format_optional_metric(metrics['position_error_2d'].get('std_m'))}")
    print(f"rmse_2d_m={metrics['position_error_2d']['rmse_m']:.3f}")
    print(f"max_2d_m={_format_optional_metric(metrics['position_error_2d'].get('max_m'))}")
    print(f"mean_3d_m={_format_optional_metric(metrics['position_error_3d'].get('mean_m'))}")
    print(f"std_3d_m={_format_optional_metric(metrics['position_error_3d'].get('std_m'))}")
    print(f"rmse_3d_m={metrics['position_error_3d']['rmse_m']:.3f}")
    print(f"max_3d_m={_format_optional_metric(metrics['position_error_3d'].get('max_m'))}")
    return 0


def _records_to_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        state = np.asarray(record["state"], dtype=float).reshape(6)
        filtered_state = record.get("filtered_state")
        filtered = (
            np.asarray(filtered_state, dtype=float).reshape(6)
            if filtered_state is not None
            else None
        )
        rows.append(
            {
                "time_s": float(record["time_s"]),
                "source": str(record["source"]),
                "track_id": _optional_int(record.get("track_id")),
                "association_mode": _optional_str(record.get("association_mode")),
                "association_nis": _optional_float(record.get("association_nis")),
                "association_score": _optional_float(record.get("association_score")),
                "hypothesis_count": _optional_int(record.get("hypothesis_count")),
                "best_hypothesis_weight": _optional_float(record.get("best_hypothesis_weight")),
                "hypothesis_weight_margin": _optional_float(record.get("hypothesis_weight_margin")),
                "measurement_dim": int(record.get("measurement_dim", 0)),
                "accepted": bool(record.get("accepted", True)),
                "update_action": str(record.get("update_action", "updated")),
                "nis": _optional_float(record.get("nis")),
                "gate_threshold": _optional_float(record.get("gate_threshold")),
                "safety_gate_threshold": _optional_float(record.get("safety_gate_threshold")),
                "residual_gate_threshold_m": _optional_float(
                    record.get("residual_gate_threshold_m")
                ),
                "covariance_scale": _optional_float(record.get("covariance_scale")),
                "inflation_alpha": _optional_float(record.get("inflation_alpha")),
                "residual_norm_m": _optional_float(record.get("residual_norm_m")),
                "east_m": state[0],
                "north_m": state[1],
                "up_m": state[2],
                "v_east_mps": state[3],
                "v_north_mps": state[4],
                "v_up_mps": state[5],
                "filtered_east_m": None if filtered is None else filtered[0],
                "filtered_north_m": None if filtered is None else filtered[1],
                "filtered_up_m": None if filtered is None else filtered[2],
                "filtered_v_east_mps": None if filtered is None else filtered[3],
                "filtered_v_north_mps": None if filtered is None else filtered[4],
                "filtered_v_up_mps": None if filtered is None else filtered[5],
                "smoother_method": _optional_str(record.get("smoother_method")),
                "smoother_lag_s": _optional_float(record.get("smoother_lag_s")),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("time_s").reset_index(drop=True)


def _hypotheses_to_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        hypotheses = record.get("hypotheses")
        if not isinstance(hypotheses, list):
            continue
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                continue
            row = {"record_index": int(record_index), "source": str(record["source"])}
            row.update(hypothesis)
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _attempted_selected_radar_frame(selected_radar: pd.DataFrame) -> pd.DataFrame:
    """Return attempted radar selections when a runner exposes them via attrs."""

    attempted = selected_radar.attrs.get("attempted_selected_radar")
    if isinstance(attempted, pd.DataFrame):
        return attempted
    return selected_radar


def _baseline_metrics(
    *,
    flight_name: str,
    flight: Any,
    truth: pd.DataFrame,
    rf: pd.DataFrame,
    radar: pd.DataFrame,
    selected_radar: pd.DataFrame,
    attempted_selected_radar: pd.DataFrame | None = None,
    estimate_frame: pd.DataFrame,
    acceleration_std: float,
    radar_association: str,
    radar_catprob_threshold: float,
    radar_covariance_model: str = "cartesian",
    radar_range_std_m: float = 12.0,
    radar_range_std_fraction: float = 0.005,
    radar_crossrange_angle_std_deg: float = 1.5,
    radar_crossrange_min_std_m: float = 5.0,
    radar_crossrange_max_std_m: float = 80.0,
    truth_gate_m: float,
    truth_time_gate_s: float,
    track_switch_nis_ratio: float,
    geometry_velocity_std: float,
    geometry_velocity_weight: float,
    geometry_switch_penalty: float,
    geometry_catprob_weight: float,
    rf_anchor_weight: float,
    rf_anchor_time_gate_s: float,
    rf_anchor_nis_cap: float,
    rf_anchor_gate_nis: float,
    pda_nis_temperature: float,
    pda_catprob_exponent: float,
    track_bank_max_hypotheses: int,
    track_bank_max_assignments: int,
    track_bank_max_candidates: int,
    track_bank_gate_prob: float,
    track_bank_detection_prob: float,
    track_bank_clutter_intensity: float,
    track_bank_prune_delta: float,
    stable_segment_min_frames: int,
    stable_segment_max_transition_speed_mps: float,
    stable_segment_range_gate_m: float,
    stable_segment_interpolation_max_gap_s: float,
    stable_segment_interpolation_max_speed_mps: float,
    stable_segment_interpolation_std_scale: float,
    stable_segment_interpolation_gap_std_mps: float,
    stable_segment_rf_score_weight: float,
    stable_segment_rf_time_gate_s: float,
    stable_segment_rf_nis_cap: float,
    smoother: str,
    smoother_lag_s: float,
    max_eval_time_delta_s: float,
    enable_gating: bool,
    robust_update: str,
    rf_gate_prob: float,
    radar_gate_prob: float,
    enable_association_safety_gate: bool,
    rf_safety_gate_prob: float,
    radar_safety_gate_prob: float,
    rf_max_residual_m: float,
    radar_max_residual_m: float,
    rf_inflation_alpha: float,
    radar_inflation_alpha: float,
    calibration_summary: dict[str, Any] | None = None,
    enable_radar_velocity_update: bool = False,
    radar_velocity_std_mps: float = 12.0,
) -> dict[str, Any]:
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    estimate_times = estimate_frame["time_s"].to_numpy(dtype=float)
    estimate_positions = estimate_frame[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    error_2d = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=max_eval_time_delta_s,
        dimensions=2,
    )
    error_3d = position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=max_eval_time_delta_s,
        dimensions=3,
    )
    paper_sampled_error_2d = sampled_position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=max_eval_time_delta_s,
        dimensions=2,
    )
    paper_sampled_error_3d = sampled_position_errors_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=max_eval_time_delta_s,
        dimensions=3,
    )
    source_counts = Counter(str(value) for value in estimate_frame["source"])
    accepted_mask = estimate_frame["accepted"].astype(bool)
    accepted_by_source = Counter(
        str(value) for value in estimate_frame.loc[accepted_mask, "source"]
    )
    rejected_by_source = Counter(
        str(value) for value in estimate_frame.loc[~accepted_mask, "source"]
    )
    covariance_scale = (
        pd.to_numeric(estimate_frame["covariance_scale"], errors="coerce")
        .fillna(1.0)
        .to_numpy(dtype=float)
    )
    reweighted_mask = covariance_scale > 1.0
    reweighted_by_source = Counter(
        str(value) for value in estimate_frame.loc[reweighted_mask, "source"]
    )

    selected_ids = []
    if "track_id" in selected_radar.columns:
        selected_ids = sorted(int(value) for value in selected_radar["track_id"].dropna().unique())
    catprob_fallback_rows = 0
    if "association_catprob_fallback" in selected_radar.columns:
        catprob_fallback_rows = int(
            selected_radar["association_catprob_fallback"].fillna(False).astype(bool).sum()
        )
    if attempted_selected_radar is None:
        attempted_selected_radar = selected_radar

    return {
        "flight": flight_name,
        "files": {
            "truth": flight.truth_txt.name if flight.truth_txt else None,
            "rf": flight.rf_csv.name if flight.rf_csv else None,
            "radar": flight.radar_json.name if flight.radar_json else None,
        },
        "state": ["east", "north", "up", "v_east", "v_north", "v_up"],
        "acceleration_std_mps2": float(acceleration_std),
        "rf_covariance": "diag(CEP^2, CEP^2), default std 75 m",
        "radar_covariance": {
            "model": radar_covariance_model,
            "cartesian_fallback": "diag(25^2, 25^2, 35^2) m^2",
            "range_std_m": radar_range_std_m,
            "range_std_fraction": radar_range_std_fraction,
            "crossrange_angle_std_deg": radar_crossrange_angle_std_deg,
            "crossrange_min_std_m": radar_crossrange_min_std_m,
            "crossrange_max_std_m": radar_crossrange_max_std_m,
        },
        "radar_selection": radar_association,
        "radar_association": radar_association,
        "radar_catprob_threshold": float(radar_catprob_threshold),
        "truth_gate_m": float(truth_gate_m),
        "truth_time_gate_s": float(truth_time_gate_s),
        "track_switch_nis_ratio": float(track_switch_nis_ratio),
        "geometry_association": {
            "velocity_std_mps": float(geometry_velocity_std),
            "velocity_weight": float(geometry_velocity_weight),
            "switch_penalty": float(geometry_switch_penalty),
            "catprob_weight": float(geometry_catprob_weight),
        },
        "rf_anchor_association": {
            "weight": float(rf_anchor_weight),
            "time_gate_s": float(rf_anchor_time_gate_s),
            "nis_cap": float(rf_anchor_nis_cap),
            "gate_nis": float(rf_anchor_gate_nis),
        },
        "pda_association": {
            "nis_temperature": float(pda_nis_temperature),
            "catprob_exponent": float(pda_catprob_exponent),
        },
        "track_bank_association": {
            "max_hypotheses": int(track_bank_max_hypotheses),
            "max_assignments": int(track_bank_max_assignments),
            "max_candidates": int(track_bank_max_candidates),
            "gate_probability": float(track_bank_gate_prob),
            "detection_probability": float(track_bank_detection_prob),
            "clutter_intensity": float(track_bank_clutter_intensity),
            "prune_log_weight_delta": float(track_bank_prune_delta),
        },
        "stable_segment_association": {
            "min_frames": int(stable_segment_min_frames),
            "max_transition_speed_mps": float(stable_segment_max_transition_speed_mps),
            "range_gate_m": None
            if stable_segment_range_gate_m <= 0.0
            else float(stable_segment_range_gate_m),
            "interpolation_max_gap_s": None
            if stable_segment_interpolation_max_gap_s <= 0.0
            else float(stable_segment_interpolation_max_gap_s),
            "interpolation_max_speed_mps": None
            if stable_segment_interpolation_max_speed_mps <= 0.0
            else float(stable_segment_interpolation_max_speed_mps),
            "interpolation_std_scale": float(stable_segment_interpolation_std_scale),
            "interpolation_gap_std_mps": float(stable_segment_interpolation_gap_std_mps),
            "rf_score_weight": float(stable_segment_rf_score_weight),
            "rf_time_gate_s": float(stable_segment_rf_time_gate_s),
            "rf_nis_cap": float(stable_segment_rf_nis_cap),
        },
        "smoother": {
            "method": smoother,
            "lag_s": float(smoother_lag_s) if smoother == "fixed-lag" else None,
        },
        "max_eval_time_delta_s": float(max_eval_time_delta_s),
        "gating": {
            "enabled": bool(enable_gating),
            "test_statistic": "normalized innovation squared",
            "rf_gate_probability": float(rf_gate_prob) if enable_gating else None,
            "radar_gate_probability": float(radar_gate_prob) if enable_gating else None,
        },
        "robust_update": {
            "method": None if robust_update == "none" else robust_update,
            "test_statistic": "normalized innovation squared"
            if robust_update != "none"
            else None,
            "rf_gate_probability": float(rf_gate_prob)
            if robust_update == "nis-inflate"
            else None,
            "radar_gate_probability": float(radar_gate_prob)
            if robust_update == "nis-inflate"
            else None,
            "student_t_degrees_of_freedom": float(DEFAULT_STUDENT_T_DOF)
            if robust_update == "student-t"
            else None,
            "huber_threshold": float(DEFAULT_HUBER_THRESHOLD)
            if robust_update == "huber"
            else None,
            "rf_inflation_alpha": float(rf_inflation_alpha)
            if robust_update == "nis-inflate"
            else None,
            "radar_inflation_alpha": float(radar_inflation_alpha)
            if robust_update == "nis-inflate"
            else None,
        },
        "calibration_bundle": calibration_summary,
        "radar_velocity_update": {
            "enabled": bool(enable_radar_velocity_update),
            "velocity_std_mps": float(radar_velocity_std_mps)
            if enable_radar_velocity_update
            else None,
        },
        "association_safety_gate": {
            "enabled": bool(enable_association_safety_gate),
            "test_statistic": "normalized innovation squared",
            "rf_gate_probability": float(rf_safety_gate_prob)
            if enable_association_safety_gate
            else None,
            "radar_gate_probability": float(radar_safety_gate_prob)
            if enable_association_safety_gate
            else None,
            "rf_max_residual_m": float(rf_max_residual_m)
            if enable_association_safety_gate and rf_max_residual_m > 0.0
            else None,
            "radar_max_residual_m": float(radar_max_residual_m)
            if enable_association_safety_gate and radar_max_residual_m > 0.0
            else None,
        },
        "truth_rows": int(len(truth)),
        "rf_rows": int(len(rf)),
        "radar_rows": int(len(radar)),
        "selected_radar_rows": int(len(selected_radar)),
        "attempted_selected_radar_rows": int(len(attempted_selected_radar)),
        "radar_catprob_fallback_rows": catprob_fallback_rows,
        "selected_radar_track_ids": selected_ids,
        "posterior_records": int(len(estimate_frame)),
        "accepted_measurements": int(accepted_mask.sum()),
        "rejected_measurements": int((~accepted_mask).sum()),
        "reweighted_measurements": int(reweighted_mask.sum()),
        "source_counts": {key: int(value) for key, value in sorted(source_counts.items())},
        "accepted_by_source": {
            key: int(value) for key, value in sorted(accepted_by_source.items())
        },
        "rejected_by_source": {
            key: int(value) for key, value in sorted(rejected_by_source.items())
        },
        "reweighted_by_source": {
            key: int(value) for key, value in sorted(reweighted_by_source.items())
        },
        "nis_by_source": _summarize_nis_by_source(estimate_frame),
        "covariance_scale_by_source": _summarize_covariance_scale_by_source(estimate_frame),
        "time_range_s": {
            "truth_min": float(truth["time_s"].min()),
            "truth_max": float(truth["time_s"].max()),
            "estimate_min": float(estimate_frame["time_s"].min()),
            "estimate_max": float(estimate_frame["time_s"].max()),
        },
        "position_error_2d": summarize_errors(error_2d),
        "position_error_3d": summarize_errors(error_3d),
        "paper_sampled_position_error_2d": summarize_errors(paper_sampled_error_2d),
        "paper_sampled_position_error_3d": summarize_errors(paper_sampled_error_3d),
    }


def _summarize_nis_by_source(
    estimate_frame: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    summaries: dict[str, dict[str, float | None]] = {}
    for source, group in estimate_frame.groupby("source"):
        values = pd.to_numeric(group["nis"], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            summaries[str(source)] = {
                "count": 0.0,
                "mean": None,
                "p50": None,
                "p95": None,
            }
            continue
        summaries[str(source)] = {
            "count": float(values.size),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    return summaries


def _summarize_covariance_scale_by_source(
    estimate_frame: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    summaries: dict[str, dict[str, float | None]] = {}
    for source, group in estimate_frame.groupby("source"):
        values = (
            pd.to_numeric(group["covariance_scale"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if values.size == 0:
            summaries[str(source)] = {
                "count": 0.0,
                "mean": None,
                "p50": None,
                "p95": None,
                "max": None,
            }
            continue
        summaries[str(source)] = {
            "count": float(values.size),
            "mean": float(np.mean(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summaries


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _json_dump_text(payload: Any) -> str:
    """Serialize JSON artifacts without non-standard NaN/Infinity tokens."""

    return json.dumps(_json_safe(payload), indent=2, allow_nan=False)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return value


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    truth_min = float(truth["time_s"].min())
    truth_max = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= truth_min) & (frame["time_s"] <= truth_max)].copy()


def _apply_time_offset_correction(frame: pd.DataFrame, correction_s: float) -> pd.DataFrame:
    """Apply a residual calibrated time correction to an already-normalized frame."""

    correction = float(correction_s)
    if frame.empty or "time_s" not in frame.columns or correction == 0.0:
        return frame
    return apply_time_offset(frame, correction)


def _format_optional_metric(value: object) -> str:
    if value is None:
        return "nan"
    scalar = float(value)
    return f"{scalar:.3f}" if np.isfinite(scalar) else "nan"


def _write_trajectory_plot(
    path: Path,
    truth: pd.DataFrame,
    rf: pd.DataFrame,
    radar: pd.DataFrame,
    estimates: pd.DataFrame,
    flight_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    ax.plot(truth["east_m"], truth["north_m"], color="black", linewidth=1.8, label="truth")
    if not rf.empty:
        ax.scatter(
            rf["east_m"],
            rf["north_m"],
            s=14,
            color="#d95f02",
            alpha=0.55,
            linewidths=0,
            label="RF",
        )
    if not radar.empty:
        ax.scatter(
            radar["east_m"],
            radar["north_m"],
            s=10,
            color="#1b9e77",
            alpha=0.45,
            linewidths=0,
            label="radar",
        )
    ax.plot(
        estimates["east_m"],
        estimates["north_m"],
        color="#386cb0",
        linewidth=1.2,
        alpha=0.9,
        label="CV fusion",
    )
    rejected = estimates.loc[~estimates["accepted"].astype(bool)]
    if not rejected.empty:
        ax.scatter(
            rejected["east_m"],
            rejected["north_m"],
            s=22,
            marker="x",
            color="#7570b3",
            alpha=0.8,
            label="rejected update state",
        )
    ax.set_title(f"{flight_name} ENU trajectory sanity check")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.grid(True, color="#dddddd", linewidth=0.7)
    _set_trajectory_limits(ax, truth, radar, estimates)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", frameon=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _set_trajectory_limits(
    ax: Any,
    truth: pd.DataFrame,
    radar: pd.DataFrame,
    estimates: pd.DataFrame,
) -> None:
    frames = [truth, estimates]
    if not radar.empty:
        frames.append(radar)
    xy = np.vstack([frame[["east_m", "north_m"]].to_numpy(dtype=float) for frame in frames])
    xy = xy[np.isfinite(xy).all(axis=1)]
    if xy.size == 0:
        return
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    x_pad = max(25.0, 0.08 * (x_max - x_min))
    y_pad = max(25.0, 0.08 * (y_max - y_min))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def _print_modality_summary(modality: str, summary: dict[str, Any] | None) -> None:
    if summary is None:
        print(f"  {modality}: missing")
        return
    columns = ",".join(summary["columns"])
    print(f"  {modality}: file={summary['file']} rows={summary['rows']}")
    print(f"    columns={columns}")
    if "raw_time_min" in summary or "time_s_min" in summary:
        print(
            "    time="
            f"raw[{summary.get('raw_time_min')} -> {summary.get('raw_time_max')}] "
            f"s[{_fmt(summary.get('time_s_min'))} -> {_fmt(summary.get('time_s_max'))}]"
        )
    if modality == "radar" and summary.get("track_ids_count") is not None:
        print(
            f"    track_ids_count={summary['track_ids_count']} "
            f"track_ids_sample={summary['track_ids_sample']}"
        )


def _fmt(value: object) -> str:
    if value is None:
        return "None"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
