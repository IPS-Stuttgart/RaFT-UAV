"""Nested leave-one-flight-out hyperparameter tuning runner.

The runner is intentionally generic: each candidate is an argument string that
is appended to a baseline command template.  Training folds are used to select a
candidate, then exactly one selected candidate is evaluated on the held-out
flight.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class Candidate:
    """One named command-line candidate."""

    name: str
    args: tuple[str, ...]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--flight", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nested_lofo_tuning"))
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="NAME=arg string, e.g. base='--radar-association prediction-nis'",
    )
    parser.add_argument("--metric", default="position_error_3d.rmse_m")
    parser.add_argument("--aggregate", choices=("mean", "median", "max"), default="mean")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--base-command",
        default="{python} -m raft_uav.cli run-baseline {dataset_root} --flight {flight} --output-dir {output_dir}",
        help="format string with {python},{dataset_root},{flight},{output_dir}",
    )
    args = parser.parse_args(argv)

    flights = [str(f) for f in args.flight]
    if len(flights) < 2:
        raise ValueError("nested LOFO needs at least two flights")
    _require_unique(flights, label="flight")
    candidates = [_parse_candidate(spec) for spec in args.candidate]
    _require_unique([candidate.name for candidate in candidates], label="candidate name")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for holdout in flights:
        train_flights = [flight for flight in flights if flight != holdout]
        training_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            for flight in train_flights:
                metrics = _run_candidate(args, candidate, flight, split=f"holdout_{holdout}/train")
                row = {
                    "holdout_flight": holdout,
                    "split": "train",
                    "flight": flight,
                    "candidate": candidate.name,
                    "metric": args.metric,
                    "metric_value": _read_metric(metrics, args.metric),
                    "metrics_json": str(metrics),
                }
                all_rows.append(row)
                training_rows.append(row)
        selected = _select_candidate(
            training_rows,
            aggregate=args.aggregate,
            expected_flights=train_flights,
        )
        if selected is None:
            summary_rows.append(
                {
                    "holdout_flight": holdout,
                    "selected_candidate": "",
                    "training_metric_value": float("nan"),
                    "holdout_metric_value": float("nan"),
                }
            )
            continue
        candidate = next(c for c in candidates if c.name == selected["candidate"])
        holdout_metrics = _run_candidate(args, candidate, holdout, split=f"holdout_{holdout}/test")
        holdout_value = _read_metric(holdout_metrics, args.metric)
        summary_rows.append(
            {
                "holdout_flight": holdout,
                "train_flights": ";".join(train_flights),
                "selected_candidate": candidate.name,
                "selected_args": " ".join(candidate.args),
                "selection_metric": args.metric,
                "selection_aggregate": args.aggregate,
                "training_metric_value": selected["metric_value"],
                "holdout_metric_value": holdout_value,
                "holdout_metrics_json": str(holdout_metrics),
            }
        )
        all_rows.append(
            {
                "holdout_flight": holdout,
                "split": "test",
                "flight": holdout,
                "candidate": candidate.name,
                "metric": args.metric,
                "metric_value": holdout_value,
                "metrics_json": str(holdout_metrics),
            }
        )

    _write_csv(args.output_dir / "nested_lofo_all_rows.csv", all_rows)
    _write_csv(args.output_dir / "nested_lofo_summary.csv", summary_rows)
    print(f"summary_csv={args.output_dir / 'nested_lofo_summary.csv'}")
    return 0


def _parse_candidate(spec: str) -> Candidate:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("candidate must be NAME=arg string")
    name, raw_args = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("candidate name must not be empty")
    return Candidate(name=name, args=tuple(shlex.split(raw_args)))


def _require_unique(values: Sequence[str], *, label: str) -> None:
    """Reject duplicate experiment identifiers before they share outputs or scores."""

    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        rendered = ", ".join(repr(value) for value in duplicates)
        raise ValueError(f"{label} values must be unique; duplicate values: {rendered}")


def _run_candidate(args: argparse.Namespace, candidate: Candidate, flight: str, *, split: str) -> Path:
    output_dir = args.output_dir / split / candidate.name
    metrics_path = output_dir / flight / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        return metrics_path
    command = shlex.split(
        args.base_command.format(
            python=sys.executable,
            dataset_root=str(args.dataset_root),
            flight=flight,
            output_dir=str(output_dir),
        )
    )
    command.extend(candidate.args)
    print(" ".join(shlex.quote(token) for token in command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)
    return metrics_path


def _read_metric(path: Path, dotted_key: str) -> float:
    if not path.exists():
        return float("nan")
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return float("nan")
        value = value[key]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _select_candidate(
    rows: list[dict[str, Any]],
    *,
    aggregate: str,
    expected_flights: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    if expected_flights is None:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            value = float(row.get("metric_value", float("nan")))
            if np.isfinite(value):
                grouped.setdefault(str(row["candidate"]), []).append(value)
    else:
        expected = tuple(str(flight) for flight in expected_flights)
        expected_set = set(expected)
        grouped_by_flight: dict[str, dict[str, float]] = {}
        duplicate_rows: set[str] = set()
        for row in rows:
            candidate = str(row["candidate"])
            flight = str(row.get("flight", ""))
            if flight not in expected_set:
                continue
            candidate_values = grouped_by_flight.setdefault(candidate, {})
            if flight in candidate_values:
                duplicate_rows.add(candidate)
            candidate_values[flight] = float(row.get("metric_value", float("nan")))
        grouped = {}
        for candidate, values_by_flight in grouped_by_flight.items():
            if candidate in duplicate_rows or set(values_by_flight) != expected_set:
                continue
            values = [values_by_flight[flight] for flight in expected]
            if all(np.isfinite(value) for value in values):
                grouped[candidate] = values

    scored = []
    for name, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        if aggregate == "mean":
            metric = float(np.mean(arr))
        elif aggregate == "median":
            metric = float(np.median(arr))
        else:
            metric = float(np.max(arr))
        scored.append({"candidate": name, "metric_value": metric})
    return min(scored, key=lambda row: (row["metric_value"], row["candidate"])) if scored else None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
