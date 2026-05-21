#!/usr/bin/env python3
"""Train LOFO calibration/model bundles for stronger RaFT-UAV experiments.

For each holdout flight, this script trains on all other selected flights:

* a learned radar-candidate likelihood,
* RF/radar residual-bias correction models, and
* a heteroscedastic RF/radar uncertainty model.

It then writes a calibration bundle manifest that can be passed to
``raft-uav run-baseline --calibration-bundle`` or
``raft-uav-best-non-oracle --calibration-bundle``.  If a LOFO time-offset
summary CSV is supplied, the manifest also includes holdout-specific RF/radar
time offsets selected from training flights.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raft_uav.io.aerpaw import discover_flights, select_flight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lofo_calibration"))
    parser.add_argument("--flight", action="append", default=None)
    parser.add_argument(
        "--time-offset-summary",
        type=Path,
        default=None,
        help="optional lofo_time_offset_summary.csv with rf/radar offsets",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    flights = _selected_flights(args.dataset_root, args.flight)
    if len(flights) < 2:
        raise ValueError("LOFO calibration needs at least two flights")
    offsets = _load_time_offsets(args.time_offset_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for holdout in flights:
        training_flights = [flight for flight in flights if flight != holdout]
        holdout_dir = args.output_dir / holdout
        holdout_dir.mkdir(parents=True, exist_ok=True)

        association_model = holdout_dir / "radar_association_model.json"
        association_examples = holdout_dir / "radar_association_examples.csv"
        bias_model = holdout_dir / "bias_model.json"
        uncertainty_model = holdout_dir / "uncertainty_model.json"
        bundle = holdout_dir / "calibration_bundle.json"

        _run(
            [
                sys.executable,
                "-m",
                "raft_uav.train_radar_association_cli",
                str(args.dataset_root),
                "--exclude-flight",
                holdout,
                "--output-model",
                str(association_model),
                "--output-examples",
                str(association_examples),
                "--radar-catprob-threshold",
                "0.4",
            ],
            target=association_model,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        _run(
            [
                sys.executable,
                "-m",
                "raft_uav.bias_cli",
                str(args.dataset_root),
                *sum((["--flight", flight] for flight in training_flights), []),
                "--output-path",
                str(bias_model),
            ],
            target=bias_model,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "train_heteroscedastic_uncertainty.py"),
                str(args.dataset_root),
                "--exclude-flight",
                holdout,
                "--output",
                str(uncertainty_model),
            ],
            target=uncertainty_model,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )

        rf_offset, radar_offset = offsets.get(holdout, (0.0, 0.0))
        payload = {
            "schema_version": 1,
            "time_offsets": {"rf": rf_offset, "radar": radar_offset},
            "bias_model_path": bias_model.name,
            "uncertainty_model_path": uncertainty_model.name,
            "metadata": {
                "holdout_flight": holdout,
                "training_flights": training_flights,
                "association_model_path": association_model.name,
            },
        }
        if not args.dry_run:
            bundle.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        rows.append(
            {
                "holdout_flight": holdout,
                "training_flights": ";".join(training_flights),
                "calibration_bundle": str(bundle),
                "radar_association_model": str(association_model),
                "bias_model": str(bias_model),
                "uncertainty_model": str(uncertainty_model),
                "rf_time_offset_s": rf_offset,
                "radar_time_offset_s": radar_offset,
            }
        )

    summary = pd.DataFrame.from_records(rows)
    summary_path = args.output_dir / "lofo_calibration_bundles.csv"
    if not args.dry_run:
        summary.to_csv(summary_path, index=False)
    print(f"summary_csv={summary_path}")
    return 0


def _selected_flights(dataset_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return [select_flight(dataset_root, item).name for item in requested]
    return [flight.name for flight in discover_flights(dataset_root) if flight.truth_txt is not None]


def _load_time_offsets(path: Path | None) -> dict[str, tuple[float, float]]:
    if path is None or not path.exists():
        return {}
    frame = pd.read_csv(path)
    offsets: dict[str, tuple[float, float]] = {}
    for _, row in frame.iterrows():
        flight = str(row.get("flight") or row.get("holdout_flight") or "")
        if not flight:
            continue
        rf = _optional_float(row.get("rf_offset_s"), row.get("applied_rf_time_offset_s"), 0.0)
        radar = _optional_float(
            row.get("radar_offset_s"),
            row.get("applied_radar_time_offset_s"),
            0.0,
        )
        offsets[flight] = (rf, radar)
    return offsets


def _optional_float(*values: object) -> float:
    for value in values:
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if pd.notna(out):
            return out
    return 0.0


def _run(command: list[str], *, target: Path, skip_existing: bool, dry_run: bool) -> None:
    if skip_existing and target.exists():
        print(f"skip_existing={target}")
        return
    print(" ".join(command), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else os.pathsep.join([src_path, env["PYTHONPATH"]])
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
