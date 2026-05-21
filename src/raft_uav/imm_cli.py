"""Standalone command-line entry point for IMM fusion experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.imm import run_async_imm_baseline
from raft_uav.baselines.imm_radar_association import (
    IMM_RADAR_ASSOCIATION_MODES,
    run_async_imm_baseline_with_radar_association,
)
from raft_uav.baselines.kalman import run_async_cv_baseline
from raft_uav.baselines.smoothing import SMOOTHER_MODES, smooth_tracking_records
from raft_uav.evaluation.diagnostics import build_diagnostic_summary
from raft_uav.evaluation.metrics import position_errors_m, summarize_errors
from raft_uav.io.aerpaw import (
    RADAR_SELECTION_MODES,
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
)
from raft_uav.numeric import optional_float as _optional_float


def main(argv: list[str] | None = None) -> int:
    """Run a CV-or-IMM baseline on one AERPAW flight."""

    parser = argparse.ArgumentParser(prog="raft-uav-imm")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/imm-baseline"))
    parser.add_argument("--tracker", choices=["cv", "imm"], default="imm")
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument(
        "--imm-mode-switch-time-constant",
        type=float,
        default=20.0,
        help="IMM Markov-mode switching time constant in seconds",
    )
    parser.add_argument(
        "--smoother",
        choices=SMOOTHER_MODES,
        default="none",
        help="post-filter smoothing mode applied before metrics are computed",
    )
    parser.add_argument(
        "--smoother-lag-s",
        type=float,
        default=20.0,
        help="future horizon for --smoother fixed-lag",
    )
    parser.add_argument(
        "--radar-selection",
        choices=RADAR_SELECTION_MODES,
        default="catprob",
        help="radar row selection before fusion",
    )
    parser.add_argument(
        "--radar-association",
        choices=["catprob", *IMM_RADAR_ASSOCIATION_MODES],
        default="catprob",
        help="IMM-native online radar association mode; catprob keeps legacy preselection",
    )
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.5)
    parser.add_argument("--truth-gate-m", type=float, default=150.0)
    parser.add_argument("--truth-time-gate-s", type=float, default=1.0)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument(
        "--robust-update",
        choices=["none", "nis-inflate"],
        default="none",
        help="robust update rule for plausible RF and radar updates",
    )
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument(
        "--disable-association-safety-gate",
        action="store_true",
        help="disable the hard RF/radar safety gate that turns impossible updates into misses",
    )
    parser.add_argument("--rf-safety-gate-prob", type=float, default=0.9999999)
    parser.add_argument("--radar-safety-gate-prob", type=float, default=0.9999999)
    parser.add_argument("--rf-max-residual-m", type=float, default=750.0)
    parser.add_argument("--radar-max-residual-m", type=float, default=0.0)
    parser.add_argument("--rf-inflation-alpha", type=float, default=1.0)
    parser.add_argument("--radar-inflation-alpha", type=float, default=1.0)
    args = parser.parse_args(argv)

    return run_experiment(
        dataset_root=args.dataset_root,
        flight_name=args.flight,
        output_dir=args.output_dir,
        tracker=args.tracker,
        acceleration_std=args.acceleration_std,
        imm_mode_switch_time_constant=args.imm_mode_switch_time_constant,
        smoother=args.smoother,
        smoother_lag_s=args.smoother_lag_s,
        radar_selection=args.radar_selection,
        radar_association=args.radar_association,
        radar_catprob_threshold=args.radar_catprob_threshold,
        truth_gate_m=args.truth_gate_m,
        truth_time_gate_s=args.truth_time_gate_s,
        max_eval_time_delta_s=args.max_eval_time_delta_s,
        robust_update=args.robust_update,
        rf_gate_prob=args.rf_gate_prob,
        radar_gate_prob=args.radar_gate_prob,
        enable_association_safety_gate=not args.disable_association_safety_gate,
        rf_safety_gate_prob=args.rf_safety_gate_prob,
        radar_safety_gate_prob=args.radar_safety_gate_prob,
        rf_max_residual_m=args.rf_max_residual_m,
        radar_max_residual_m=args.radar_max_residual_m,
        rf_inflation_alpha=args.rf_inflation_alpha,
        radar_inflation_alpha=args.radar_inflation_alpha,
    )


def run_experiment(
    *,
    dataset_root: Path,
    flight_name: str,
    output_dir: Path,
    tracker: str = "imm",
    acceleration_std: float = 4.0,
    imm_mode_switch_time_constant: float = 20.0,
    smoother: str = "none",
    smoother_lag_s: float = 20.0,
    radar_selection: str = "catprob",
    radar_association: str = "catprob",
    radar_catprob_threshold: float = 0.5,
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
    max_eval_time_delta_s: float = 2.0,
    robust_update: str = "none",
    rf_gate_prob: float = 0.99,
    radar_gate_prob: float = 0.99,
    enable_association_safety_gate: bool = True,
    rf_safety_gate_prob: float = 0.9999999,
    radar_safety_gate_prob: float = 0.9999999,
    rf_max_residual_m: float = 750.0,
    radar_max_residual_m: float = 0.0,
    rf_inflation_alpha: float = 1.0,
    radar_inflation_alpha: float = 1.0,
) -> int:
    """Run one flight and write estimates, diagnostics, and metrics."""

    if tracker not in {"cv", "imm"}:
        raise ValueError("tracker must be 'cv' or 'imm'")
    if imm_mode_switch_time_constant <= 0.0:
        raise ValueError("imm_mode_switch_time_constant must be positive")
    if smoother not in SMOOTHER_MODES:
        raise ValueError(f"smoother must be one of {SMOOTHER_MODES}; got {smoother!r}")
    if smoother in {"fixed-lag", "fixed-lag-map"} and smoother_lag_s < 0.0:
        raise ValueError(f"smoother_lag_s must be nonnegative for {smoother} smoothing")
    if radar_association not in {"catprob", *IMM_RADAR_ASSOCIATION_MODES}:
        raise ValueError(f"unknown radar_association {radar_association!r}")
    if tracker != "imm" and radar_association in IMM_RADAR_ASSOCIATION_MODES:
        raise ValueError("IMM-native radar association requires --tracker imm")
    if robust_update not in {"none", "nis-inflate"}:
        raise ValueError("robust_update must be 'none' or 'nis-inflate'")
    if rf_inflation_alpha <= 0.0 or radar_inflation_alpha <= 0.0:
        raise ValueError("inflation alphas must be positive")

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
    use_imm_radar_association = tracker == "imm" and radar_association in IMM_RADAR_ASSOCIATION_MODES
    if flight.rf_csv is not None:
        rf = _inside_truth_window(
            normalize_rf(read_rf_csv(flight.rf_csv), projector, truth_origin_time), truth
        )
        rf_measurements = rf_measurements_to_enu(rf)
        measurements.extend(rf_measurements)
    if flight.radar_json is not None:
        radar = _inside_truth_window(
            normalize_radar(read_radar_tracks_json(flight.radar_json), projector, truth_origin_time),
            truth,
        )
        if not use_imm_radar_association:
            selected_radar = select_radar_measurement_rows(
                radar,
                selection=radar_selection,
                truth=truth,
                catprob_threshold=radar_catprob_threshold,
                truth_gate_m=truth_gate_m,
                truth_time_gate_s=truth_time_gate_s,
            )
            measurements.extend(radar_measurements_to_enu(selected_radar))

    gate_probabilities = None
    safety_gate_probabilities = None
    max_residual_norms = None
    robust_updates = None
    inflation_alphas = None
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
        gate_probabilities = {"rf": rf_gate_prob, "radar": radar_gate_prob}
        robust_updates = {"rf": robust_update, "radar": robust_update}
        inflation_alphas = {"rf": rf_inflation_alpha, "radar": radar_inflation_alpha}

    if use_imm_radar_association:
        records, selected_radar = run_async_imm_baseline_with_radar_association(
            rf_measurements=rf_measurements,
            radar=radar,
            association=radar_association,
            acceleration_std_mps2=acceleration_std,
            gate_probabilities_by_source=gate_probabilities,
            safety_gate_probabilities_by_source=safety_gate_probabilities,
            robust_update_by_source=robust_updates,
            inflation_alpha_by_source=inflation_alphas,
            max_residual_norms_by_source=max_residual_norms,
            candidate_catprob_threshold=radar_catprob_threshold,
            mode_switch_time_constant_s=imm_mode_switch_time_constant,
        )
        measurements = [
            *rf_measurements,
            *radar_measurements_to_enu(selected_radar),
        ]
    elif tracker == "imm":
        records = run_async_imm_baseline(
            measurements,
            acceleration_std_mps2=acceleration_std,
            gate_probabilities_by_source=gate_probabilities,
            safety_gate_probabilities_by_source=safety_gate_probabilities,
            robust_update_by_source=robust_updates,
            inflation_alpha_by_source=inflation_alphas,
            max_residual_norms_by_source=max_residual_norms,
            mode_switch_time_constant_s=imm_mode_switch_time_constant,
        )
    else:
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
        raise RuntimeError(f"{flight.name} produced no posterior records")

    records = smooth_tracking_records(
        records,
        method=smoother,
        acceleration_std_mps2=acceleration_std,
        lag_s=smoother_lag_s,
        measurements=measurements,
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
    metrics_path = flight_output / "metrics.json"
    diagnostic_summary_path = flight_output / "diagnostic_summary.json"
    selected_radar_path = flight_output / "selected_radar.csv"

    estimate_frame.to_csv(estimates_path, index=False)
    diagnostics_frame.to_csv(diagnostics_path, index=False)
    selected_radar.to_csv(selected_radar_path, index=False)
    metrics = _metrics(
        flight_name=flight.name,
        truth=truth,
        rf=rf,
        radar=radar,
        selected_radar=selected_radar,
        estimate_frame=estimate_frame,
        tracker=tracker,
        acceleration_std=acceleration_std,
        imm_mode_switch_time_constant=imm_mode_switch_time_constant,
        smoother=smoother,
        smoother_lag_s=smoother_lag_s,
        radar_selection=radar_association if use_imm_radar_association else radar_selection,
        radar_catprob_threshold=radar_catprob_threshold,
        max_eval_time_delta_s=max_eval_time_delta_s,
        robust_update=robust_update,
        enable_association_safety_gate=enable_association_safety_gate,
        rf_safety_gate_prob=rf_safety_gate_prob,
        radar_safety_gate_prob=radar_safety_gate_prob,
        rf_max_residual_m=rf_max_residual_m,
        radar_max_residual_m=radar_max_residual_m,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    diagnostic_summary = build_diagnostic_summary(
        estimate_frame=estimate_frame,
        selected_radar=selected_radar,
        truth=truth,
        max_eval_time_delta_s=max_eval_time_delta_s,
    )
    diagnostic_summary_path.write_text(
        json.dumps(diagnostic_summary, indent=2),
        encoding="utf-8",
    )

    print(f"flight={flight.name}")
    print(f"tracker={tracker}")
    print(f"smoother={smoother}")
    print(f"posterior_records={len(records)}")
    print(f"selected_radar_rows={len(selected_radar)}")
    print(f"metrics_json={metrics_path}")
    print(f"diagnostic_summary_json={diagnostic_summary_path}")
    print(f"estimates_csv={estimates_path}")
    print(f"diagnostics_csv={diagnostics_path}")
    print(f"rmse_2d_m={metrics['position_error_2d']['rmse_m']:.3f}")
    print(f"rmse_3d_m={metrics['position_error_3d']['rmse_m']:.3f}")
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
        row = {
            "time_s": float(record["time_s"]),
            "source": str(record["source"]),
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
        probabilities = record.get("mode_probability_map")
        if isinstance(probabilities, dict):
            for mode_name, probability in probabilities.items():
                row[f"mode_probability_{str(mode_name).replace('-', '_')}"] = float(probability)
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values("time_s").reset_index(drop=True)


def _metrics(
    *,
    flight_name: str,
    truth: pd.DataFrame,
    rf: pd.DataFrame,
    radar: pd.DataFrame,
    selected_radar: pd.DataFrame,
    estimate_frame: pd.DataFrame,
    tracker: str,
    acceleration_std: float,
    imm_mode_switch_time_constant: float,
    smoother: str,
    smoother_lag_s: float,
    radar_selection: str,
    radar_catprob_threshold: float,
    max_eval_time_delta_s: float,
    robust_update: str,
    enable_association_safety_gate: bool,
    rf_safety_gate_prob: float,
    radar_safety_gate_prob: float,
    rf_max_residual_m: float,
    radar_max_residual_m: float,
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
    source_counts = Counter(str(value) for value in estimate_frame["source"])
    accepted_mask = estimate_frame["accepted"].astype(bool)
    return {
        "flight": flight_name,
        "tracker": {
            "method": tracker,
            "acceleration_std_mps2": float(acceleration_std),
            "imm_mode_switch_time_constant_s": float(imm_mode_switch_time_constant)
            if tracker == "imm"
            else None,
        },
        "smoother": {
            "method": smoother,
            "lag_s": float(smoother_lag_s) if smoother in {"fixed-lag", "fixed-lag-map"} else None,
        },
        "radar_selection": radar_selection,
        "radar_catprob_threshold": float(radar_catprob_threshold),
        "robust_update": None if robust_update == "none" else robust_update,
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
        "max_eval_time_delta_s": float(max_eval_time_delta_s),
        "truth_rows": int(len(truth)),
        "rf_rows": int(len(rf)),
        "radar_rows": int(len(radar)),
        "selected_radar_rows": int(len(selected_radar)),
        "posterior_records": int(len(estimate_frame)),
        "accepted_measurements": int(accepted_mask.sum()),
        "rejected_measurements": int((~accepted_mask).sum()),
        "source_counts": {key: int(value) for key, value in sorted(source_counts.items())},
        "position_error_2d": summarize_errors(error_2d),
        "position_error_3d": summarize_errors(error_3d),
    }


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    truth_min = float(truth["time_s"].min())
    truth_max = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= truth_min) & (frame["time_s"] <= truth_max)].copy()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
