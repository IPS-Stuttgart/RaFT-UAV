"""CLI for the top-k tracklet graph + weak-z smoother method row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.topk_weakz_tracklet import (
    TopKWeakZTrackletConfig,
    records_to_frame,
    run_topk_tracklet_graph_weakz_smoother,
)
from raft_uav.evaluation.metrics import position_errors_at_estimates_m, summarize_errors
from raft_uav.io.aerpaw import (
    DEFAULT_RADAR_CLOCK_OFFSET_S,
    DEFAULT_RF_CLOCK_OFFSET_S,
    flight_file_manifest,
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    rf_measurements_to_enu,
    select_flight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-topk-weakz-tracklet",
        description="run raw-stream top-k Fortem tracklet graph + weak-z smoother",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--variant", choices=("auto", "original", "rerun"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/topk-weakz-tracklet"))
    parser.add_argument("--rf-clock-offset-s", type=float, default=DEFAULT_RF_CLOCK_OFFSET_S)
    parser.add_argument("--radar-clock-offset-s", type=float, default=DEFAULT_RADAR_CLOCK_OFFSET_S)
    parser.add_argument("--top-k-paths", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--min-tracklet-length", type=int, default=3)
    parser.add_argument("--max-tracklets", type=int, default=256)
    parser.add_argument("--range-gate-m", type=float, default=900.0)
    parser.add_argument("--disable-range-gate", action="store_true")
    parser.add_argument("--max-transition-gap-s", type=float, default=30.0)
    parser.add_argument("--max-transition-speed-mps", type=float, default=80.0)
    parser.add_argument("--weakz-radar-xy-std-m", type=float, default=360.0)
    parser.add_argument("--weakz-radar-z-std-m", type=float, default=20000.0)
    parser.add_argument("--acceleration-std", type=float, default=14.0)
    parser.add_argument("--smoother", choices=("none", "fixed-lag", "rts"), default="fixed-lag")
    parser.add_argument("--smoother-lag-s", type=float, default=15.0)
    parser.add_argument("--smoother-acceleration-std", type=float, default=28.0)
    parser.add_argument("--rf-radar-consistency-std-m", type=float, default=160.0)
    parser.add_argument("--rf-max-covariance-scale", type=float, default=50.0)
    parser.add_argument("--rf-reject-distance-m", type=float, default=0.0)
    parser.add_argument("--disable-rf-soft-weight", action="store_true")
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    args = parser.parse_args(argv)

    return run_cli(args)


def run_cli(args: argparse.Namespace) -> int:
    if args.top_k_paths < 1:
        raise ValueError("--top-k-paths must be positive")
    if args.beam_width < args.top_k_paths:
        raise ValueError("--beam-width must be >= --top-k-paths")
    if args.disable_range_gate:
        range_gate_m = None
    else:
        range_gate_m = float(args.range_gate_m)
    config = TopKWeakZTrackletConfig(
        top_k_paths=int(args.top_k_paths),
        beam_width=int(args.beam_width),
        max_tracklets=int(args.max_tracklets),
        min_tracklet_length=int(args.min_tracklet_length),
        range_gate_m=range_gate_m,
        max_transition_gap_s=float(args.max_transition_gap_s),
        max_transition_speed_mps=float(args.max_transition_speed_mps),
        weakz_radar_xy_std_m=float(args.weakz_radar_xy_std_m),
        weakz_radar_z_std_m=float(args.weakz_radar_z_std_m),
        acceleration_std_mps2=float(args.acceleration_std),
        smoother=str(args.smoother),
        smoother_lag_s=float(args.smoother_lag_s),
        smoother_acceleration_std_mps2=float(args.smoother_acceleration_std),
        rf_soft_weight=not bool(args.disable_rf_soft_weight),
        rf_radar_consistency_std_m=float(args.rf_radar_consistency_std_m),
        rf_max_covariance_scale=float(args.rf_max_covariance_scale),
        rf_reject_distance_m=(
            None if float(args.rf_reject_distance_m) <= 0.0 else float(args.rf_reject_distance_m)
        ),
    )
    flight = select_flight(args.dataset_root, args.flight, variant=args.variant)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    if flight.rf_csv is None:
        raise FileNotFoundError(f"{flight.name} has no RF CSV file")
    if flight.radar_json is None:
        raise FileNotFoundError(f"{flight.name} has no radar JSON file")

    truth_raw = read_truth(flight.truth_txt)
    truth, projector, truth_origin_time = normalize_truth(truth_raw)
    rf = _inside_truth_window(
        normalize_rf(
            read_rf_csv(flight.rf_csv),
            projector,
            truth_origin_time,
            clock_offset_s=float(args.rf_clock_offset_s),
        ),
        truth,
    )
    radar = _inside_truth_window(
        normalize_radar(
            read_radar_tracks_json(flight.radar_json),
            projector,
            truth_origin_time,
            clock_offset_s=float(args.radar_clock_offset_s),
        ),
        truth,
    )
    rf_measurements = rf_measurements_to_enu(rf)
    result = run_topk_tracklet_graph_weakz_smoother(
        rf_measurements=rf_measurements,
        radar=radar,
        config=config,
    )
    if not result.records:
        raise RuntimeError(f"{flight.name} produced no top-k weak-z posterior records")

    output = Path(args.output_dir) / flight.name
    output.mkdir(parents=True, exist_ok=True)
    estimates_path = output / "estimates.csv"
    filtered_estimates_path = output / "filtered_estimates.csv"
    selected_radar_path = output / "selected_radar.csv"
    attempted_radar_path = output / "attempted_selected_radar.csv"
    path_diagnostics_path = output / "path_diagnostics.csv"
    tracklet_diagnostics_path = output / "tracklet_diagnostics.csv"
    metrics_path = output / "metrics.json"
    manifest_path = output / "manifest.json"

    estimates = records_to_frame(result.records)
    filtered_estimates = records_to_frame(result.filtered_records)
    estimates.to_csv(estimates_path, index=False)
    filtered_estimates.to_csv(filtered_estimates_path, index=False)
    result.selected_radar.to_csv(selected_radar_path, index=False)
    result.attempted_radar.to_csv(attempted_radar_path, index=False)
    result.path_diagnostics.to_csv(path_diagnostics_path, index=False)
    result.tracklet_diagnostics.to_csv(tracklet_diagnostics_path, index=False)

    metrics = _metrics(
        flight_name=flight.name,
        estimates=estimates,
        truth=truth,
        max_time_delta_s=float(args.max_eval_time_delta_s),
    )
    metrics["selected_path"] = _jsonable(result.selected_path_summary)
    metrics["radar_rows_raw"] = int(len(radar))
    metrics["radar_rows_selected"] = int(len(result.selected_radar))
    metrics["rf_rows"] = int(len(rf))
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    manifest = {
        "method": "topk_tracklet_graph_weakz_smoother",
        "flight": flight.name,
        "dataset_root": str(args.dataset_root),
        "file_manifest": flight_file_manifest(flight, dataset_root=args.dataset_root),
        "config": _jsonable(config.__dict__),
        "outputs": {
            "estimates_csv": str(estimates_path),
            "filtered_estimates_csv": str(filtered_estimates_path),
            "selected_radar_csv": str(selected_radar_path),
            "attempted_radar_csv": str(attempted_radar_path),
            "path_diagnostics_csv": str(path_diagnostics_path),
            "tracklet_diagnostics_csv": str(tracklet_diagnostics_path),
            "metrics_json": str(metrics_path),
        },
    }
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")

    print("topk_weakz_tracklet=ok")
    print(f"flight={flight.name}")
    print(f"posterior_records={len(estimates)}")
    print(f"selected_radar_rows={len(result.selected_radar)}")
    print(f"path_diagnostics_csv={path_diagnostics_path}")
    print(f"metrics_json={metrics_path}")
    return 0


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns or truth.empty:
        return frame.copy()
    start = float(np.nanmin(truth["time_s"].to_numpy(dtype=float)))
    end = float(np.nanmax(truth["time_s"].to_numpy(dtype=float)))
    time_s = pd.to_numeric(frame["time_s"], errors="coerce")
    return frame.loc[(time_s >= start) & (time_s <= end)].copy().reset_index(drop=True)


def _metrics(
    *,
    flight_name: str,
    estimates: pd.DataFrame,
    truth: pd.DataFrame,
    max_time_delta_s: float,
) -> dict[str, Any]:
    estimate_times = estimates["time_s"].to_numpy(dtype=float)
    estimate_positions = estimates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    errors_3d = position_errors_at_estimates_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=max_time_delta_s,
        dimensions=3,
    )
    errors_2d = position_errors_at_estimates_m(
        estimate_times,
        estimate_positions,
        truth_times,
        truth_positions,
        max_time_delta_s=max_time_delta_s,
        dimensions=2,
    )
    return {
        "flight": flight_name,
        "posterior_records": int(len(estimates)),
        "paper_sample_3d": summarize_errors(errors_3d),
        "paper_sample_2d": summarize_errors(errors_2d),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except ValueError:
            pass
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
