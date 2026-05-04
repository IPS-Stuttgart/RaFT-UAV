"""Command-line entry points for RaFT-UAV experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from raft_uav.baselines.kalman import run_async_cv_baseline
from raft_uav.io.aerpaw import (
    discover_flights,
    projector_from_truth,
    radar_measurements_to_enu,
    read_radar_json,
    read_rf_csv,
    read_truth,
    rf_measurements_to_enu,
    select_flight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raft-uav")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="list discovered AERPAW flights")
    inspect_parser.add_argument("dataset_root", type=Path)

    baseline_parser = subparsers.add_parser("run-baseline", help="run the initial CV fusion baseline")
    baseline_parser.add_argument("dataset_root", type=Path)
    baseline_parser.add_argument("--flight", required=True)
    baseline_parser.add_argument("--acceleration-std", type=float, default=4.0)

    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _inspect(args.dataset_root)
    if args.command == "run-baseline":
        return _run_baseline(args.dataset_root, args.flight, args.acceleration_std)
    raise ValueError(args.command)


def _inspect(dataset_root: Path) -> int:
    flights = discover_flights(dataset_root)
    print(f"discovered_flights={len(flights)}")
    for flight in flights:
        print(
            f"{flight.name}\t"
            f"rf={_yes_no(flight.rf_csv)}\t"
            f"radar={_yes_no(flight.radar_json)}\t"
            f"truth={_yes_no(flight.truth_txt)}"
        )
    return 0


def _run_baseline(dataset_root: Path, flight_name: str, acceleration_std: float) -> int:
    flight = select_flight(dataset_root, flight_name)
    if flight.truth_txt is None:
        raise FileNotFoundError(f"{flight.name} has no truth telemetry file")

    truth = read_truth(flight.truth_txt)
    projector = projector_from_truth(truth)
    measurements = []
    if flight.rf_csv is not None:
        measurements.extend(rf_measurements_to_enu(read_rf_csv(flight.rf_csv), projector))
    if flight.radar_json is not None:
        measurements.extend(radar_measurements_to_enu(read_radar_json(flight.radar_json), projector))

    records = run_async_cv_baseline(measurements, acceleration_std_mps2=acceleration_std)
    print(f"flight={flight.name}")
    print(f"measurements={len(measurements)}")
    print(f"posterior_records={len(records)}")
    if records:
        final_state = np.asarray(records[-1]["state"], dtype=float)
        print("final_state_enu_cv=" + ",".join(f"{value:.3f}" for value in final_state))
    return 0


def _yes_no(path: Path | None) -> str:
    return "yes" if path is not None else "no"


if __name__ == "__main__":
    raise SystemExit(main())
