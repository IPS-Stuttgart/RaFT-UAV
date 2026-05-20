"""Command-line entry point for oracle candidate-retention diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.evaluation.oracle_coverage import build_oracle_candidate_coverage
from raft_uav.io.aerpaw import (
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
    parser = argparse.ArgumentParser(prog="raft-uav-oracle-coverage")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oracle_coverage"))
    parser.add_argument("--radar-catprob-threshold", type=_optional_threshold, default=0.5)
    parser.add_argument("--max-candidates-per-frame", type=_positive_int, default=8)
    parser.add_argument("--acceleration-std", type=_positive_float, default=4.0)
    parser.add_argument("--radar-xy-std-m", type=_positive_float, default=25.0)
    parser.add_argument("--radar-z-std-m", type=_positive_float, default=35.0)
    parser.add_argument(
        "--truth-time-gate-s",
        type=_optional_positive_float,
        default=1.0,
        help="nearest truth support tolerance; pass <=0 to disable the tolerance",
    )
    args = parser.parse_args(argv)

    result = _run_one(
        dataset_root=args.dataset_root,
        flight_name=args.flight,
        output_dir=args.output_dir,
        radar_catprob_threshold=args.radar_catprob_threshold,
        max_candidates_per_frame=args.max_candidates_per_frame,
        acceleration_std=args.acceleration_std,
        radar_xy_std_m=args.radar_xy_std_m,
        radar_z_std_m=args.radar_z_std_m,
        truth_time_gate_s=args.truth_time_gate_s,
    )
    print(f"flight={result['flight']}")
    print(f"oracle_available_frames={result['oracle_available_frames']}")
    print(f"oracle_retained_frames={result['oracle_retained_frames']}")
    print(f"oracle_retention_rate={result['oracle_retention_rate']:.6f}")
    print(f"catprob_threshold_drop_frames={result['catprob_threshold_drop_frames']}")
    print(f"top_k_drop_frames={result['top_k_drop_frames']}")
    print(f"frame_coverage_csv={result['frame_coverage_csv']}")
    print(f"bucket_summary_csv={result['bucket_summary_csv']}")
    print(f"summary_json={result['summary_json']}")
    return 0


def _run_one(
    *,
    dataset_root: Path,
    flight_name: str,
    output_dir: Path,
    radar_catprob_threshold: float | None,
    max_candidates_per_frame: int,
    acceleration_std: float,
    radar_xy_std_m: float,
    radar_z_std_m: float,
    truth_time_gate_s: float | None,
) -> dict[str, Any]:
    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")
    if flight.radar_json is None:
        raise FileNotFoundError(f"{flight.name} has no radar JSON file")

    truth_raw = read_truth(flight.truth_txt)
    truth, projector, truth_origin_time = normalize_truth(truth_raw)
    radar = _inside_truth_window(
        normalize_radar(read_radar_tracks_json(flight.radar_json), projector, truth_origin_time),
        truth,
    )
    rf = pd.DataFrame()
    if flight.rf_csv is not None:
        rf = _inside_truth_window(
            normalize_rf(read_rf_csv(flight.rf_csv), projector, truth_origin_time),
            truth,
        )

    result = build_oracle_candidate_coverage(
        radar=radar,
        truth=truth,
        rf_measurements=rf_measurements_to_enu(rf),
        candidate_catprob_threshold=radar_catprob_threshold,
        config=TrackletViterbiAssociationConfig(
            max_candidates_per_frame=int(max_candidates_per_frame),
        ),
        acceleration_std_mps2=acceleration_std,
        radar_xy_std_m=radar_xy_std_m,
        radar_z_std_m=radar_z_std_m,
        truth_time_gate_s=truth_time_gate_s,
    )

    flight_output = output_dir / flight.name
    flight_output.mkdir(parents=True, exist_ok=True)
    frame_path = flight_output / "oracle_candidate_coverage.csv"
    bucket_path = flight_output / "oracle_candidate_coverage_buckets.csv"
    summary_path = flight_output / "oracle_candidate_coverage_summary.json"
    result.frame_coverage.to_csv(frame_path, index=False)
    result.bucket_summary.to_csv(bucket_path, index=False)

    summary = {
        "flight": flight.name,
        "files": {
            "truth": flight.truth_txt.name if flight.truth_txt else None,
            "rf": flight.rf_csv.name if flight.rf_csv else None,
            "radar": flight.radar_json.name if flight.radar_json else None,
        },
        "radar_rows": int(len(radar)),
        "rf_rows": int(len(rf)),
        **result.summary,
        "outputs": {
            "frame_coverage_csv": str(frame_path),
            "bucket_summary_csv": str(bucket_path),
        },
    }
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    return {
        **summary,
        "frame_coverage_csv": str(frame_path),
        "bucket_summary_csv": str(bucket_path),
        "summary_json": str(summary_path),
    }


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "time_s" not in frame.columns:
        return frame
    truth_min = float(truth["time_s"].min())
    truth_max = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= truth_min) & (frame["time_s"] <= truth_max)].copy()


def _optional_threshold(value: str) -> float | None:
    parsed = float(value)
    return None if parsed < 0.0 else parsed


def _optional_positive_float(value: str) -> float | None:
    parsed = float(value)
    return None if parsed <= 0.0 else parsed


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


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
