"""Radar-track segment fingerprints for paper-parity debugging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from raft_uav.diagnostics.paper_strict import (
    PAPER_STRICT_RANGE_GATE_M,
    load_paper_strict_inputs,
)
from raft_uav.evaluation.metrics import position_errors_at_times_m
from raft_uav.io.aerpaw import DEFAULT_RADAR_CLOCK_OFFSET_S, DEFAULT_RF_CLOCK_OFFSET_S, discover_flights


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-radar-segment-fingerprint",
        description="write Fortem track-segment fingerprints before fusion tuning",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/radar-fingerprint"))
    parser.add_argument("--variant", choices=["auto", "original", "rerun"], default="auto")
    parser.add_argument("--range-gate-m", type=float, default=PAPER_STRICT_RANGE_GATE_M)
    parser.add_argument("--truth-time-gate-s", type=float, default=2.0)
    parser.add_argument("--enu-origin", choices=["truth-first", "lla", "lw1"], default="lw1")
    parser.add_argument("--enu-origin-lla", default=None)
    parser.add_argument("--lw1-origin-lla", default=None)
    parser.add_argument("--origin-config", type=Path, default=None)
    parser.add_argument("--rf-clock-offset-s", type=float, default=DEFAULT_RF_CLOCK_OFFSET_S)
    parser.add_argument("--radar-clock-offset-s", type=float, default=DEFAULT_RADAR_CLOCK_OFFSET_S)
    args = parser.parse_args(argv)

    result = run_radar_fingerprint(
        dataset_root=args.dataset_root,
        flights=args.flight,
        output_dir=args.output_dir,
        variant=args.variant,
        range_gate_m=args.range_gate_m,
        truth_time_gate_s=args.truth_time_gate_s,
        enu_origin=args.enu_origin,
        enu_origin_lla=args.enu_origin_lla,
        lw1_origin_lla=args.lw1_origin_lla,
        origin_config=args.origin_config,
        rf_clock_offset_s=args.rf_clock_offset_s,
        radar_clock_offset_s=args.radar_clock_offset_s,
    )
    print(f"output_dir={result['output_dir']}")
    print(f"summary_csv={result['summary_csv']}")
    print(f"summary_json={result['summary_json']}")
    return 0


def run_radar_fingerprint(
    *,
    dataset_root: Path,
    flights: Iterable[str] | None,
    output_dir: Path,
    variant: str = "auto",
    range_gate_m: float = PAPER_STRICT_RANGE_GATE_M,
    truth_time_gate_s: float = 2.0,
    enu_origin: str = "lw1",
    enu_origin_lla: str | None = None,
    lw1_origin_lla: str | None = None,
    origin_config: Path | None = None,
    rf_clock_offset_s: float = DEFAULT_RF_CLOCK_OFFSET_S,
    radar_clock_offset_s: float = DEFAULT_RADAR_CLOCK_OFFSET_S,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_flights = list(flights or [])
    if not selected_flights:
        selected_flights = [flight.name for flight in discover_flights(dataset_root, variant=variant)]

    all_rows: list[dict[str, Any]] = []
    per_flight_payloads: list[dict[str, Any]] = []
    for flight_name in selected_flights:
        inputs = load_paper_strict_inputs(
            dataset_root=Path(dataset_root),
            flight_name=flight_name,
            enu_origin=enu_origin,
            enu_origin_lla=enu_origin_lla,
            lw1_origin_lla=lw1_origin_lla,
            origin_config=origin_config,
            rf_default_std_m=75.0,
            variant=variant,
            rf_clock_offset_s=rf_clock_offset_s,
            radar_clock_offset_s=radar_clock_offset_s,
        )
        rows = radar_segment_fingerprint_rows(
            flight_name=inputs.flight_name,
            radar=inputs.radar,
            truth=inputs.truth,
            range_gate_m=range_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
        frame = pd.DataFrame.from_records(rows)
        flight_dir = output / inputs.flight_name
        flight_dir.mkdir(parents=True, exist_ok=True)
        flight_csv = flight_dir / "radar_segment_fingerprint.csv"
        frame.to_csv(flight_csv, index=False)
        all_rows.extend(rows)
        per_flight_payloads.append(
            {
                "flight": inputs.flight_name,
                "csv": str(flight_csv),
                "segments": int(len(frame)),
                "range_gate_m": float(range_gate_m),
                "truth_time_gate_s": float(truth_time_gate_s),
                "file_manifest": inputs.file_manifest,
            }
        )

    summary = pd.DataFrame.from_records(all_rows)
    summary_csv = output / "radar_segment_fingerprint_summary.csv"
    summary_json = output / "radar_segment_fingerprint_summary.json"
    summary.to_csv(summary_csv, index=False)
    payload = {
        "output_dir": str(output),
        "summary_csv": str(summary_csv),
        "flights": per_flight_payloads,
        "variant": variant,
        "range_gate_m": float(range_gate_m),
        "truth_time_gate_s": float(truth_time_gate_s),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**payload, "summary_json": str(summary_json)}


def radar_segment_fingerprint_rows(
    *,
    flight_name: str,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    range_gate_m: float,
    truth_time_gate_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(_continuous_track_segments(radar)):
        if segment.empty:
            continue
        track_id = _optional_int(segment["track_id"].iloc[0]) if "track_id" in segment.columns else None
        errors = _segment_errors(segment, truth, truth_time_gate_s=truth_time_gate_s)
        ranges = _segment_ranges(segment)
        catprob = (
            pd.to_numeric(segment["cat_prob_uav"], errors="coerce").to_numpy(dtype=float)
            if "cat_prob_uav" in segment.columns
            else np.full(len(segment), np.nan)
        )
        rows.append(
            {
                "flight": flight_name,
                "segment_index": int(segment_index),
                "track_id": track_id,
                "start_time_s": float(segment["time_s"].iloc[0]),
                "end_time_s": float(segment["time_s"].iloc[-1]),
                "duration_s": float(segment["time_s"].iloc[-1] - segment["time_s"].iloc[0]),
                "frames": int(len(segment)),
                "range_gate_m": float(range_gate_m),
                "range_gated_frames": int(np.count_nonzero(np.isfinite(ranges) & (ranges <= range_gate_m))),
                "range_source": "range_m" if "range_m" in segment.columns else "enu_norm",
                "mean_catprob": _nan_stat(catprob, np.nanmean),
                "median_catprob": _nan_stat(catprob, np.nanmedian),
                "min_range_m": _nan_stat(ranges, np.nanmin),
                "median_range_m": _nan_stat(ranges, np.nanmedian),
                "max_range_m": _nan_stat(ranges, np.nanmax),
                "matched_error_count": int(errors.size),
                "mean_3d_error_to_truth_m": _finite_stat(errors, np.mean),
                "p50_3d_error_to_truth_m": _percentile(errors, 50.0),
                "p95_3d_error_to_truth_m": _percentile(errors, 95.0),
                "max_3d_error_to_truth_m": _finite_stat(errors, np.max),
                "first_frame_index": _optional_int(segment["frame_index"].iloc[0])
                if "frame_index" in segment.columns
                else None,
                "last_frame_index": _optional_int(segment["frame_index"].iloc[-1])
                if "frame_index" in segment.columns
                else None,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["range_gated_frames"]),
            _none_as_inf(row["mean_3d_error_to_truth_m"]),
            -int(row["frames"]),
        ),
    )


def _continuous_track_segments(radar: pd.DataFrame) -> list[pd.DataFrame]:
    if radar.empty or "track_id" not in radar.columns:
        return []
    segments: list[pd.DataFrame] = []
    for _, track_rows in radar.groupby("track_id", sort=True):
        sort_key = _track_continuity_key(track_rows)
        ordered = track_rows.sort_values([sort_key, "time_s"]).reset_index(drop=True)
        values = pd.to_numeric(ordered[sort_key], errors="coerce").to_numpy(dtype=float)
        split_points = np.r_[
            0,
            np.where(np.diff(values) > _segment_gap_threshold(values))[0] + 1,
            len(ordered),
        ]
        for start, end in zip(split_points[:-1], split_points[1:]):
            segment = ordered.iloc[int(start) : int(end)].copy()
            if not segment.empty:
                segments.append(segment)
    return segments


def _track_continuity_key(track_rows: pd.DataFrame) -> str:
    if "frame_index" not in track_rows.columns:
        return "time_s"
    frame_indices = pd.to_numeric(track_rows["frame_index"], errors="coerce").to_numpy(dtype=float)
    return "frame_index" if np.isfinite(frame_indices).all() else "time_s"


def _segment_gap_threshold(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("inf")
    diffs = np.diff(np.sort(finite))
    positive = diffs[diffs > 1.0e-9]
    if positive.size == 0:
        return float("inf")
    if np.allclose(finite, np.round(finite)):
        return 1.5
    return 1.5 * float(np.median(positive))


def _segment_ranges(segment: pd.DataFrame) -> np.ndarray:
    if "range_m" in segment.columns:
        ranges = pd.to_numeric(segment["range_m"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(ranges).any():
            return ranges
    return np.linalg.norm(segment[["east_m", "north_m", "up_m"]].to_numpy(dtype=float), axis=1)


def _segment_errors(segment: pd.DataFrame, truth: pd.DataFrame, *, truth_time_gate_s: float) -> np.ndarray:
    return position_errors_at_times_m(
        estimate_times_s=segment["time_s"].to_numpy(dtype=float),
        estimate_positions_m=segment[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
        truth_times_s=truth["time_s"].to_numpy(dtype=float),
        truth_positions_m=truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float),
        max_time_delta_s=truth_time_gate_s,
        dimensions=3,
    )


def _finite_stat(values: np.ndarray, fn: Any) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(fn(finite))


def _nan_stat(values: np.ndarray, fn: Any) -> float | None:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return None
    return float(fn(values))


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.percentile(finite, percentile))


def _optional_int(value: Any) -> int | None:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return int(scalar) if np.isfinite(scalar) else None


def _none_as_inf(value: Any) -> float:
    if value is None:
        return float("inf")
    return float(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
