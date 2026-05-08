"""Command-line entry point for RF/radar bias-calibration models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from raft_uav.calibration.bias import bias_training_rows, fit_bias_correction_bank
from raft_uav.io.aerpaw import (
    discover_flights,
    normalize_radar,
    normalize_rf,
    normalize_truth,
    read_radar_tracks_json,
    read_rf_csv,
    read_truth,
    select_flight,
)


def main(argv: list[str] | None = None) -> int:
    """Train a JSON RF/radar residual-bias correction model."""

    parser = argparse.ArgumentParser(prog="raft-uav-train-bias-model")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--flight",
        action="append",
        help="training flight; omit to use all discovered flights with truth",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/calibration/bias_model.json"),
        help="path for the trained JSON bias model bundle",
    )
    parser.add_argument("--max-time-delta-s", type=float, default=2.0)
    parser.add_argument("--max-position-error-m", type=float, default=250.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-samples", type=int, default=5)
    args = parser.parse_args(argv)
    return train_bias_model(
        dataset_root=args.dataset_root,
        requested_flights=args.flight,
        output_path=args.output_path,
        max_time_delta_s=args.max_time_delta_s,
        max_position_error_m=args.max_position_error_m,
        ridge_alpha=args.ridge_alpha,
        min_samples=args.min_samples,
    )


def train_bias_model(
    *,
    dataset_root: Path,
    requested_flights: list[str] | None,
    output_path: Path,
    max_time_delta_s: float,
    max_position_error_m: float | None,
    ridge_alpha: float,
    min_samples: int,
) -> int:
    """Train RF/radar bias correction models from selected normalized flights."""

    if max_time_delta_s <= 0.0:
        raise ValueError("max_time_delta_s must be positive")
    if max_position_error_m is not None and max_position_error_m <= 0.0:
        raise ValueError("max_position_error_m must be positive")
    if ridge_alpha < 0.0:
        raise ValueError("ridge_alpha must be nonnegative")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")

    if requested_flights:
        flights = [select_flight(dataset_root, name) for name in requested_flights]
    else:
        flights = discover_flights(dataset_root)

    training_frames: dict[str, list[pd.DataFrame]] = {"rf": [], "radar": []}
    skipped: list[str] = []
    for flight in flights:
        if flight.truth_txt is None:
            skipped.append(f"{flight.name}: no truth")
            continue
        truth_raw = read_truth(flight.truth_txt)
        truth, projector, truth_origin_time = normalize_truth(truth_raw)

        if flight.rf_csv is not None:
            rf = _inside_truth_window(
                normalize_rf(read_rf_csv(flight.rf_csv), projector, truth_origin_time), truth
            )
            rows = bias_training_rows(
                rf,
                truth,
                source="rf",
                max_time_delta_s=max_time_delta_s,
                max_position_error_m=max_position_error_m,
            )
            if not rows.empty:
                rows["flight"] = flight.name
                training_frames["rf"].append(rows)

        if flight.radar_json is not None:
            radar = _inside_truth_window(
                normalize_radar(
                    read_radar_tracks_json(flight.radar_json), projector, truth_origin_time
                ),
                truth,
            )
            rows = bias_training_rows(
                radar,
                truth,
                source="radar",
                max_time_delta_s=max_time_delta_s,
                max_position_error_m=max_position_error_m,
            )
            if not rows.empty:
                rows["flight"] = flight.name
                training_frames["radar"].append(rows)

    combined = {
        source: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for source, frames in training_frames.items()
    }
    bank = fit_bias_correction_bank(combined, ridge_alpha=ridge_alpha, min_samples=min_samples)
    bank.save(output_path)

    print(f"bias_model_json={output_path}")
    for source, model in sorted(bank.models.items()):
        print(f"{source}_training_rows={model.training_rows}")
        print(f"{source}_features={','.join(model.feature_columns)}")
        print(
            f"{source}_residual_std_m="
            + ",".join(f"{value:.6g}" for value in model.residual_std)
        )
    for message in skipped:
        print(f"skipped={message}")
    return 0


def _inside_truth_window(frame: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or truth.empty or "time_s" not in frame.columns:
        return frame
    lower = float(truth["time_s"].min())
    upper = float(truth["time_s"].max())
    return frame.loc[(frame["time_s"] >= lower) & (frame["time_s"] <= upper)].copy()
