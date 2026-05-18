"""CLI for LOFO time-offset plus bias calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.calibration.lofo_combined import (
    apply_time_offsets,
    fit_training_bias_bank,
    fit_training_time_offset,
)
from raft_uav.calibration.lofo_io import jsonable, load_flight_frames, requested_flights
from raft_uav.diagnostics.time_offset import OBJECTIVE_COLUMNS, offset_grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", action="append")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lofo_bias_time_offset"))
    parser.add_argument("--tau-min", type=float, default=-10.0)
    parser.add_argument("--tau-max", type=float, default=10.0)
    parser.add_argument("--tau-step", type=float, default=0.25)
    parser.add_argument("--offset-objective", choices=sorted(OBJECTIVE_COLUMNS), default="p95")
    parser.add_argument("--max-truth-time-delta-s", type=float, default=2.0)
    parser.add_argument("--radar-catprob-threshold", type=float, default=0.4)
    parser.add_argument("--bias-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--bias-min-samples", type=int, default=5)
    parser.add_argument("--bias-max-position-error-m", type=float, default=300.0)
    parser.add_argument("--skip-bias", action="store_true")
    args = parser.parse_args(argv)

    names = requested_flights(args.dataset_root, args.flight)
    if len(names) < 2:
        raise ValueError("LOFO calibration requires at least two flights")
    items = {name: load_flight_frames(args.dataset_root, name) for name in names}
    taus = offset_grid(args.tau_min, args.tau_max, args.tau_step)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for holdout in names:
        rows.append(_holdout(args, items, [n for n in names if n != holdout], holdout, taus))
    summary = pd.DataFrame.from_records(rows)
    path = args.output_dir / "lofo_bias_time_offset_summary.csv"
    summary.to_csv(path, index=False)
    print(f"summary_csv={path}")
    return 0


def _holdout(args: argparse.Namespace, items: dict, train: list[str], holdout: str, taus):
    out = args.output_dir / holdout
    out.mkdir(parents=True, exist_ok=True)
    rf_tau, rf_sweep = _fit_tau(args, items, train, "rf", taus)
    radar_tau, radar_sweep = _fit_tau(args, items, train, "radar", taus)
    rf_sweep.to_csv(out / "rf_time_offset_training_sweep.csv", index=False)
    radar_sweep.to_csv(out / "radar_time_offset_training_sweep.csv", index=False)
    shifted_train = {
        n: apply_time_offsets(items[n], rf_tau_s=rf_tau, radar_tau_s=radar_tau)
        for n in train
    }
    shifted = apply_time_offsets(items[holdout], rf_tau_s=rf_tau, radar_tau_s=radar_tau)
    bias_summary: dict[str, Any] = {"enabled": False}
    if not args.skip_bias:
        bank = fit_training_bias_bank(
            shifted_train,
            train,
            radar_catprob_threshold=args.radar_catprob_threshold,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
            max_position_error_m=args.bias_max_position_error_m,
            ridge_alpha=args.bias_ridge_alpha,
            min_samples=args.bias_min_samples,
        )
        bank.save(out / "bias_model.json")
        bias_summary = bank.summary(out / "bias_model.json")
        shifted["rf"] = bank.correct_frame(shifted["rf"], "rf")
        shifted["radar"] = bank.correct_frame(shifted["radar"], "radar")
    shifted["rf"].to_csv(out / "rf_time_bias_corrected.csv", index=False)
    shifted["radar"].to_csv(out / "radar_time_bias_corrected.csv", index=False)
    payload = {
        "holdout": holdout,
        "training_flights": train,
        "rf_tau_s": float(rf_tau),
        "radar_tau_s": float(radar_tau),
        "bias": bias_summary,
    }
    metrics = out / "lofo_bias_time_offset_metrics.json"
    metrics.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    return {
        "holdout": holdout,
        "training_flights": ",".join(train),
        "rf_tau_s": float(rf_tau),
        "radar_tau_s": float(radar_tau),
        "metrics_json": str(metrics),
    }


def _fit_tau(args, items, train, source, taus):
    return fit_training_time_offset(
        items,
        train,
        source=source,
        taus_s=taus,
        objective=args.offset_objective,
        max_truth_time_delta_s=args.max_truth_time_delta_s,
        radar_catprob_threshold=args.radar_catprob_threshold,
    )
