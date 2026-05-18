#!/usr/bin/env python3
"""LOFO tuning for range-angle radar covariance parameters.

The script trains covariance hyperparameters on non-held-out flights only and
then evaluates the chosen setting on the held-out flight.  It delegates actual
tracking to ``scripts/run_tracklet_viterbi_baseline.py`` while setting the
``RAFT_UAV_RADAR_*`` environment variables consumed by the runtime covariance
hook.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RadarCovarianceCandidate:
    """One range-angle radar covariance hyperparameter setting."""

    candidate_id: str
    range_std_m: float
    azimuth_std_deg: float
    elevation_std_deg: float
    min_std_m: float
    max_std_m: float

    def environment(self) -> dict[str, str]:
        """Return environment variables for the existing covariance hook."""

        return {
            "RAFT_UAV_RADAR_COVARIANCE_MODE": "range-angle",
            "RAFT_UAV_RADAR_RANGE_STD_M": _format_float(self.range_std_m),
            "RAFT_UAV_RADAR_AZIMUTH_STD_DEG": _format_float(self.azimuth_std_deg),
            "RAFT_UAV_RADAR_ELEVATION_STD_DEG": _format_float(self.elevation_std_deg),
            "RAFT_UAV_RADAR_COVARIANCE_MIN_STD_M": _format_float(self.min_std_m),
            "RAFT_UAV_RADAR_COVARIANCE_MAX_STD_M": _format_float(self.max_std_m),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lofo_radar_covariance"))
    parser.add_argument("--baseline-script", type=Path, default=Path("scripts/run_tracklet_viterbi_baseline.py"))
    parser.add_argument("--metric", default="position_error_3d.rmse_m")
    parser.add_argument("--aggregate", choices=["mean", "median", "max"], default="mean")
    parser.add_argument("--range-std-m", default="3,5,10,20")
    parser.add_argument("--azimuth-std-deg", default="1,2,3,4")
    parser.add_argument("--elevation-std-deg", default="1,2,3,4")
    parser.add_argument("--min-std-m", default="3")
    parser.add_argument("--max-std-m", default="150,250,400")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--baseline-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to run_tracklet_viterbi_baseline.py. "
        "Repeat for each token, e.g. --baseline-arg --smoother --baseline-arg fixed-lag.",
    )
    args = parser.parse_args(argv)

    flights = args.flight or ["Opt1", "Opt2", "Opt3"]
    if len(flights) < 2:
        raise ValueError("LOFO covariance tuning requires at least two flights")

    candidates = _candidate_grid(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for holdout in flights:
        train_flights = [flight for flight in flights if flight != holdout]
        train_rows = _evaluate_training_grid(
            args=args,
            candidates=candidates,
            train_flights=train_flights,
            holdout=holdout,
        )
        all_rows.extend(train_rows)
        sweep = pd.DataFrame.from_records(train_rows)
        sweep_path = args.output_dir / holdout / "training_covariance_sweep.csv"
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        sweep.to_csv(sweep_path, index=False)

        selected = _select_candidate(sweep, metric_column="metric_value", aggregate=args.aggregate)
        if selected is None:
            summary_rows.append(_failed_summary_row(holdout, train_flights, sweep_path))
            continue

        selected_candidate = next(
            candidate for candidate in candidates if candidate.candidate_id == selected["candidate_id"]
        )
        holdout_metrics_path = _run_baseline_for_candidate(
            args=args,
            candidate=selected_candidate,
            flight=holdout,
            split="holdout",
            holdout=holdout,
        )
        holdout_metric = _read_metric(holdout_metrics_path, args.metric)
        holdout_summary = _read_metrics_summary(holdout_metrics_path)
        selection_payload = {
            "holdout_flight": holdout,
            "training_flights": train_flights,
            "selection_metric": args.metric,
            "selection_aggregate": args.aggregate,
            "selected_candidate": asdict(selected_candidate),
            "training_metric_value": selected["aggregate_metric_value"],
            "training_sweep_csv": str(sweep_path),
            "holdout_metrics_json": str(holdout_metrics_path),
            "holdout_metric_value": holdout_metric,
            "holdout_summary": holdout_summary,
        }
        selection_path = args.output_dir / holdout / "selected_covariance.json"
        selection_path.write_text(json.dumps(selection_payload, indent=2), encoding="utf-8")
        summary_rows.append(
            {
                "holdout_flight": holdout,
                "training_flights": ",".join(train_flights),
                "candidate_id": selected_candidate.candidate_id,
                "range_std_m": selected_candidate.range_std_m,
                "azimuth_std_deg": selected_candidate.azimuth_std_deg,
                "elevation_std_deg": selected_candidate.elevation_std_deg,
                "min_std_m": selected_candidate.min_std_m,
                "max_std_m": selected_candidate.max_std_m,
                "selection_metric": args.metric,
                "selection_aggregate": args.aggregate,
                "training_metric_value": selected["aggregate_metric_value"],
                "holdout_metric_value": holdout_metric,
                **holdout_summary,
                "training_sweep_csv": str(sweep_path),
                "selected_covariance_json": str(selection_path),
                "holdout_metrics_json": str(holdout_metrics_path),
            }
        )

    all_sweeps_path = args.output_dir / "lofo_radar_covariance_all_training_rows.csv"
    summary_path = args.output_dir / "lofo_radar_covariance_summary.csv"
    pd.DataFrame.from_records(all_rows).to_csv(all_sweeps_path, index=False)
    pd.DataFrame.from_records(summary_rows).to_csv(summary_path, index=False)
    print(f"summary_csv={summary_path}")
    print(f"all_training_rows_csv={all_sweeps_path}")
    return 0


def _candidate_grid(args: argparse.Namespace) -> list[RadarCovarianceCandidate]:
    ranges = _parse_float_list(args.range_std_m)
    azimuths = _parse_float_list(args.azimuth_std_deg)
    elevations = _parse_float_list(args.elevation_std_deg)
    mins = _parse_float_list(args.min_std_m)
    maxes = _parse_float_list(args.max_std_m)
    candidates: list[RadarCovarianceCandidate] = []
    for idx, (range_std, azimuth, elevation, min_std, max_std) in enumerate(
        itertools.product(ranges, azimuths, elevations, mins, maxes)
    ):
        if max_std < min_std:
            continue
        candidates.append(
            RadarCovarianceCandidate(
                candidate_id=f"cov{idx:04d}",
                range_std_m=range_std,
                azimuth_std_deg=azimuth,
                elevation_std_deg=elevation,
                min_std_m=min_std,
                max_std_m=max_std,
            )
        )
    if not candidates:
        raise ValueError("covariance grid is empty")
    return candidates


def _evaluate_training_grid(
    *,
    args: argparse.Namespace,
    candidates: list[RadarCovarianceCandidate],
    train_flights: list[str],
    holdout: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for flight in train_flights:
            metrics_path = _run_baseline_for_candidate(
                args=args,
                candidate=candidate,
                flight=flight,
                split="train",
                holdout=holdout,
            )
            metric_value = _read_metric(metrics_path, args.metric)
            rows.append(
                {
                    "holdout_flight": holdout,
                    "train_flight": flight,
                    "candidate_id": candidate.candidate_id,
                    "range_std_m": candidate.range_std_m,
                    "azimuth_std_deg": candidate.azimuth_std_deg,
                    "elevation_std_deg": candidate.elevation_std_deg,
                    "min_std_m": candidate.min_std_m,
                    "max_std_m": candidate.max_std_m,
                    "metric": args.metric,
                    "metric_value": metric_value,
                    "metrics_json": str(metrics_path),
                }
            )
    return rows


def _run_baseline_for_candidate(
    *,
    args: argparse.Namespace,
    candidate: RadarCovarianceCandidate,
    flight: str,
    split: str,
    holdout: str,
) -> Path:
    output_dir = args.output_dir / holdout / split / candidate.candidate_id
    metrics_path = output_dir / flight / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        return metrics_path

    command = [
        sys.executable,
        str(args.baseline_script),
        str(args.dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(output_dir),
        *args.baseline_arg,
    ]
    env = os.environ.copy()
    env.update(candidate.environment())
    if args.dry_run:
        print(" ".join(command))
        return metrics_path
    subprocess.run(command, check=True, env=env)
    if not metrics_path.exists():
        raise FileNotFoundError(f"baseline did not write expected metrics file: {metrics_path}")
    return metrics_path


def _select_candidate(
    sweep: pd.DataFrame,
    *,
    metric_column: str,
    aggregate: str,
) -> dict[str, Any] | None:
    if sweep.empty or metric_column not in sweep.columns:
        return None
    grouped = []
    for candidate_id, group in sweep.groupby("candidate_id", sort=False):
        values = pd.to_numeric(group[metric_column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if aggregate == "mean":
            aggregate_value = float(np.mean(values))
        elif aggregate == "median":
            aggregate_value = float(np.median(values))
        elif aggregate == "max":
            aggregate_value = float(np.max(values))
        else:  # pragma: no cover - argparse prevents this
            raise ValueError(f"unknown aggregate {aggregate!r}")
        grouped.append(
            {
                "candidate_id": str(candidate_id),
                "aggregate_metric_value": aggregate_value,
                "finite_train_flights": int(values.size),
            }
        )
    if not grouped:
        return None
    grouped.sort(key=lambda row: (row["aggregate_metric_value"], row["candidate_id"]))
    return grouped[0]


def _read_metric(metrics_path: Path, dotted_key: str) -> float:
    if not metrics_path.exists():
        return float("nan")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    value: Any = data
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return float("nan")
        value = value[part]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _read_metrics_summary(metrics_path: Path) -> dict[str, float]:
    return {
        "holdout_rmse_3d_m": _read_metric(metrics_path, "position_error_3d.rmse_m"),
        "holdout_p95_3d_m": _read_metric(metrics_path, "position_error_3d.p95_m"),
        "holdout_mae_3d_m": _read_metric(metrics_path, "position_error_3d.mae_m"),
        "holdout_max_3d_m": _read_metric(metrics_path, "position_error_3d.max_m"),
        "holdout_rmse_2d_m": _read_metric(metrics_path, "position_error_2d.rmse_m"),
        "holdout_p95_2d_m": _read_metric(metrics_path, "position_error_2d.p95_m"),
        "selected_radar_rmse_3d_m": _read_metric(
            metrics_path, "selected_radar_position_error_3d.rmse_m"
        ),
        "selected_radar_p95_3d_m": _read_metric(
            metrics_path, "selected_radar_position_error_3d.p95_m"
        ),
    }


def _failed_summary_row(holdout: str, train_flights: list[str], sweep_path: Path) -> dict[str, Any]:
    return {
        "holdout_flight": holdout,
        "training_flights": ",".join(train_flights),
        "candidate_id": "",
        "selection_metric": "",
        "selection_aggregate": "",
        "training_metric_value": float("nan"),
        "holdout_metric_value": float("nan"),
        "training_sweep_csv": str(sweep_path),
        "selected_covariance_json": "",
        "holdout_metrics_json": "",
    }


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"empty numeric list: {raw!r}")
    for value in values:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"all values must be finite and positive: {raw!r}")
    return values


def _format_float(value: float) -> str:
    return f"{float(value):.12g}"


if __name__ == "__main__":
    raise SystemExit(main())
