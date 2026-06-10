"""Run LOFO radar calibration diagnostics on selected radar rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raft_uav.evaluation.radar_calibration_audit import (  # noqa: E402
    IDENTITY_CALIBRATION,
    apply_spatial_calibration,
    concatenate_pairs,
    evaluate_calibrated_measurements,
    fit_constant_offset,
    fit_time_offset,
    fit_yaw_offset_altitude,
    pair_measurements_to_truth,
)
from raft_uav.io.aerpaw import normalize_truth, read_truth, select_flight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--selected-radar-root",
        type=Path,
        required=True,
        help="directory containing <flight>/selected_radar.csv outputs",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/radar_calibration_audit"))
    parser.add_argument("--flights", nargs="*", default=["Opt1", "Opt2", "Opt3"])
    parser.add_argument("--offset-min-s", type=float, default=-2.0)
    parser.add_argument("--offset-max-s", type=float, default=2.0)
    parser.add_argument("--offset-step-s", type=float, default=0.25)
    parser.add_argument("--max-time-delta-s", type=float, default=2.0)
    parser.add_argument("--write-corrected", action="store_true")
    args = parser.parse_args()

    if args.offset_step_s <= 0.0:
        raise ValueError("--offset-step-s must be positive")
    flights = list(dict.fromkeys(args.flights))
    if len(flights) < 2:
        raise ValueError("LOFO calibration audit needs at least two flights")

    truth_by_flight: dict[str, pd.DataFrame] = {}
    selected_by_flight: dict[str, pd.DataFrame] = {}
    for flight in flights:
        paths = select_flight(args.dataset_root, flight)
        truth, _, _ = normalize_truth(read_truth(paths.truth_txt))
        selected_path = args.selected_radar_root / flight / "selected_radar.csv"
        if not selected_path.exists():
            raise FileNotFoundError(f"missing selected radar CSV for {flight}: {selected_path}")
        truth_by_flight[flight] = truth
        selected_by_flight[flight] = pd.read_csv(selected_path)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets_s = _offset_grid(args.offset_min_s, args.offset_max_s, args.offset_step_s)
    rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {
        "selected_radar_root": str(args.selected_radar_root),
        "flights": flights,
        "max_time_delta_s": float(args.max_time_delta_s),
        "offset_grid_s": {
            "min": float(args.offset_min_s),
            "max": float(args.offset_max_s),
            "step": float(args.offset_step_s),
        },
        "folds": {},
    }

    for heldout in flights:
        training_flights = [flight for flight in flights if flight != heldout]
        training_measurements = {flight: selected_by_flight[flight] for flight in training_flights}
        training_truth = {flight: truth_by_flight[flight] for flight in training_flights}
        raw_pairs = concatenate_pairs(
            pair_measurements_to_truth(
                selected_by_flight[flight],
                truth_by_flight[flight],
                max_time_delta_s=args.max_time_delta_s,
            )
            for flight in training_flights
        )
        constant_model = fit_constant_offset(raw_pairs)
        yaw_model = fit_yaw_offset_altitude(raw_pairs)
        best_time_offset_s, time_sweep = fit_time_offset(
            training_measurements,
            training_truth,
            offsets_s,
            max_time_delta_s=args.max_time_delta_s,
        )
        time_shifted_pairs = concatenate_pairs(
            pair_measurements_to_truth(
                selected_by_flight[flight],
                truth_by_flight[flight],
                time_offset_s=best_time_offset_s,
                max_time_delta_s=args.max_time_delta_s,
            )
            for flight in training_flights
        )
        combined_model = fit_yaw_offset_altitude(time_shifted_pairs)

        fold_dir = output_dir / heldout
        fold_dir.mkdir(parents=True, exist_ok=True)
        time_sweep.to_csv(fold_dir / "training_time_offset_sweep.csv", index=False)

        fold_models = {
            "training_flights": training_flights,
            "constant_offset": constant_model.as_dict(),
            "yaw_offset_altitude": yaw_model.as_dict(),
            "best_time_offset_s": float(best_time_offset_s),
            "combined_yaw_time": combined_model.as_dict(),
        }
        models["folds"][heldout] = fold_models

        methods = [
            ("raw", IDENTITY_CALIBRATION, 0.0),
            ("constant_offset", constant_model, 0.0),
            ("yaw_offset_altitude", yaw_model, 0.0),
            ("time_offset", IDENTITY_CALIBRATION, best_time_offset_s),
            ("combined_yaw_time", combined_model, best_time_offset_s),
        ]
        for method_name, calibration, time_offset_s in methods:
            metrics = evaluate_calibrated_measurements(
                selected_by_flight[heldout],
                truth_by_flight[heldout],
                calibration=calibration,
                time_offset_s=time_offset_s,
                max_time_delta_s=args.max_time_delta_s,
            )
            row = {
                "heldout_flight": heldout,
                "method": method_name,
                "training_flights": " ".join(training_flights),
                "time_offset_s": float(time_offset_s),
                "yaw_rad": float(calibration.yaw_rad),
                "yaw_deg": float(np.degrees(calibration.yaw_rad)),
                "offset_east_m": float(calibration.offset_east_m),
                "offset_north_m": float(calibration.offset_north_m),
                "offset_up_m": float(calibration.offset_up_m),
                "matched_rows": metrics["matched_rows"],
                "truth_coverage_rows": metrics["truth_coverage_rows"],
                "rmse_3d_m": metrics["rmse_m"],
                "mae_3d_m": metrics["mae_m"],
                "p50_3d_m": metrics["p50_m"],
                "p95_3d_m": metrics["p95_m"],
                "max_3d_m": metrics.get("max_m", float("nan")),
            }
            rows.append(row)
            if args.write_corrected and method_name == "combined_yaw_time":
                corrected = apply_spatial_calibration(selected_by_flight[heldout], calibration)
                corrected["time_s"] = corrected["time_s"].astype(float) + float(time_offset_s)
                corrected.to_csv(fold_dir / "selected_radar_combined_corrected.csv", index=False)

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "radar_calibration_audit_summary.csv"
    models_path = output_dir / "radar_calibration_audit_models.json"
    summary.to_csv(summary_path, index=False)
    models_path.write_text(json.dumps(models, indent=2), encoding="utf-8")
    print(f"summary_csv={summary_path}")
    print(f"models_json={models_path}")
    best = summary.sort_values(["heldout_flight", "rmse_3d_m"]).groupby("heldout_flight").head(1)
    for row in best.to_dict(orient="records"):
        print(
            f"heldout={row['heldout_flight']} best={row['method']} "
            f"rmse_3d_m={_format_optional_metric(row.get('rmse_3d_m'))} "
            f"p95_3d_m={_format_optional_metric(row.get('p95_3d_m'))}"
        )
    return 0


def _offset_grid(min_s: float, max_s: float, step_s: float) -> np.ndarray:
    if max_s < min_s:
        raise ValueError("--offset-max-s must be >= --offset-min-s")
    count = int(np.floor((max_s - min_s) / step_s + 1.0e-9)) + 1
    grid = min_s + np.arange(count, dtype=float) * step_s
    if grid.size == 0 or grid[-1] < max_s - 1.0e-9:
        grid = np.append(grid, max_s)
    return grid


def _format_optional_metric(value: object) -> str:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(metric):
        return "nan"
    return f"{metric:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
