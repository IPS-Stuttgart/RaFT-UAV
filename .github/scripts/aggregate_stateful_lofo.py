#!/usr/bin/env python3
"""Aggregate stateful LOFO workflow artifacts and enforce workflow policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AggregateResult:
    summary: dict[str, Any]
    rows: list[dict[str, Any]]
    smoke_failures: list[str]
    threshold_failures: list[str]

    @property
    def should_fail(self) -> bool:
        return bool(self.smoke_failures or self.threshold_failures)


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def collect_rows(artifacts_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(artifacts_dir.glob("**/summary.json")):
        try:
            row = load_strict_json(summary_path)
            if not isinstance(row, dict):
                raise TypeError("summary payload is not a JSON object")
            row["_artifact_root"] = str(summary_path.parent)
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "flight": str(summary_path),
                    "status": f"failed_to_read_summary: {exc}",
                    "rmse_3d_m": None,
                    "p95_3d_m": None,
                    "_artifact_root": str(summary_path.parent),
                }
            )
    rows.sort(key=lambda item: str(item.get("flight", "")))
    return rows


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def compute_means(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float | None, float | None]:
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and is_number(row.get("rmse_3d_m"))
        and is_number(row.get("p95_3d_m"))
        and math.isfinite(float(row["rmse_3d_m"]))
        and math.isfinite(float(row["p95_3d_m"]))
    ]
    mean_rmse_3d = sum(float(row["rmse_3d_m"]) for row in ok_rows) / len(ok_rows) if ok_rows else None
    mean_p95_3d = sum(float(row["p95_3d_m"]) for row in ok_rows) / len(ok_rows) if ok_rows else None
    return ok_rows, mean_rmse_3d, mean_p95_3d


def required_smoke_artifacts(artifact_root: Path, flight: str) -> list[Path]:
    return [
        artifact_root / "radar_assoc.json",
        artifact_root / "radar_assoc_examples.csv",
        artifact_root / "run" / flight / "metrics.json",
        artifact_root / "run" / flight / "diagnostic_summary.json",
        artifact_root / "run" / flight / "diagnostics.csv",
        artifact_root / "run" / flight / "selected_radar.csv",
        artifact_root / "run" / flight / "estimates.csv",
        artifact_root / "run" / flight / "trajectory.png",
    ]


def validate_smoke(rows: list[dict[str, Any]], expected_flights: list[str]) -> list[str]:
    failures: list[str] = []
    by_flight = {str(row.get("flight")): row for row in rows}
    for flight in expected_flights:
        row = by_flight.get(str(flight))
        if row is None:
            failures.append(f"{flight}: missing summary artifact")
            continue
        if row.get("status") != "ok":
            failures.append(f"{flight}: summary status is {row.get('status')}")
        for count_name in ("selected_radar_rows", "posterior_records"):
            if not is_positive_int(row.get(count_name)):
                failures.append(f"{flight}: {count_name} is unavailable or empty")

        artifact_root = Path(str(row.get("_artifact_root", "")))
        required = required_smoke_artifacts(artifact_root, str(flight))
        missing = [str(path) for path in required if not path.exists() or path.stat().st_size <= 0]
        if missing:
            failures.append(f"{flight}: missing or empty required artifacts: {missing}")
            continue
        for json_path in (
            artifact_root / "radar_assoc.json",
            artifact_root / "run" / str(flight) / "metrics.json",
            artifact_root / "run" / str(flight) / "diagnostic_summary.json",
        ):
            try:
                load_strict_json(json_path)
            except Exception as exc:
                failures.append(f"{flight}: JSON artifact validation failed for {json_path}: {exc}")
    return failures


def validate_thresholds(
    ok_rows: list[dict[str, Any]],
    mean_rmse_3d: float | None,
    target_mean_rmse_3d_m: float,
    target_opt1_p95_3d_m: float,
) -> list[str]:
    failures: list[str] = []
    opt1 = next((row for row in ok_rows if str(row.get("flight")) == "Opt1"), None)
    if mean_rmse_3d is None:
        failures.append("mean_rmse_3d_m is unavailable")
    elif mean_rmse_3d > target_mean_rmse_3d_m:
        failures.append(f"mean_rmse_3d_m={mean_rmse_3d:.3f} exceeds target {target_mean_rmse_3d_m:.3f}")
    if opt1 is None:
        failures.append("Opt1 metrics are unavailable")
    elif float(opt1["p95_3d_m"]) > target_opt1_p95_3d_m:
        failures.append(f"Opt1 p95_3d_m={float(opt1['p95_3d_m']):.3f} exceeds target {target_opt1_p95_3d_m:.3f}")
    return failures


def aggregate_lofo_artifacts(
    artifacts_dir: Path,
    expected_flights: list[str],
    *,
    smoke_mode: bool,
    enforce_thresholds: bool,
    target_mean_rmse_3d_m: float,
    target_opt1_p95_3d_m: float,
) -> AggregateResult:
    rows = collect_rows(artifacts_dir)
    ok_rows, mean_rmse_3d, mean_p95_3d = compute_means(rows)
    smoke_failures = validate_smoke(rows, expected_flights) if smoke_mode else []
    threshold_failures = []
    if enforce_thresholds and not smoke_mode:
        threshold_failures = validate_thresholds(
            ok_rows,
            mean_rmse_3d,
            target_mean_rmse_3d_m,
            target_opt1_p95_3d_m,
        )
    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    summary = {
        "smoke_mode": smoke_mode,
        "expected_flights": expected_flights,
        "completed_flights": len(ok_rows),
        "mean_rmse_3d_m": mean_rmse_3d,
        "mean_p95_3d_m": mean_p95_3d,
        "smoke_failures": smoke_failures,
        "threshold_failures": threshold_failures,
        "flights": public_rows,
    }
    return AggregateResult(summary, rows, smoke_failures, threshold_failures)


def append_step_summary(path: Path, result: AggregateResult) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("## Stateful learned radar association LOFO summary\n\n")
        if result.summary["smoke_mode"]:
            handle.write("Smoke mode: **dataset/CLI/artifact validity only**. Metric targets are not enforced.\n\n")
        handle.write("| Flight | Status | 3D RMSE m | 3D P95 m | Selected radar rows |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in result.rows:
            handle.write(
                f"| {row.get('flight')} | {row.get('status')} | {row.get('rmse_3d_m', 'NA')} | "
                f"{row.get('p95_3d_m', 'NA')} | {row.get('selected_radar_rows', 'NA')} |\n"
            )
        mean_rmse_3d = result.summary["mean_rmse_3d_m"]
        mean_p95_3d = result.summary["mean_p95_3d_m"]
        handle.write(f"\nMean 3D RMSE: **{mean_rmse_3d if mean_rmse_3d is not None else 'NA'} m**\n\n")
        handle.write(f"Mean 3D P95: **{mean_p95_3d if mean_p95_3d is not None else 'NA'} m**\n\n")
        if result.smoke_failures:
            handle.write("### Smoke artifact validation failures\n\n")
            for failure in result.smoke_failures:
                handle.write(f"- {failure}\n")
        if result.threshold_failures:
            handle.write("### Threshold failures\n\n")
            for failure in result.threshold_failures:
                handle.write(f"- {failure}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("lofo_aggregate_summary.json"))
    parser.add_argument("--expected-flights-json", required=True)
    parser.add_argument("--smoke-mode", default="false")
    parser.add_argument("--enforce-thresholds", default="false")
    parser.add_argument("--target-mean-rmse-3d-m", type=float, required=True)
    parser.add_argument("--target-opt1-p95-3d-m", type=float, required=True)
    parser.add_argument("--github-step-summary", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_flights = json.loads(args.expected_flights_json)
    if not isinstance(expected_flights, list) or not all(isinstance(item, str) for item in expected_flights):
        raise SystemExit("--expected-flights-json must be a JSON array of strings")
    result = aggregate_lofo_artifacts(
        args.artifacts_dir,
        expected_flights,
        smoke_mode=parse_bool(args.smoke_mode),
        enforce_thresholds=parse_bool(args.enforce_thresholds),
        target_mean_rmse_3d_m=args.target_mean_rmse_3d_m,
        target_opt1_p95_3d_m=args.target_opt1_p95_3d_m,
    )
    args.output.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
    step_summary = args.github_step_summary or os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        append_step_summary(Path(step_summary), result)
    if result.smoke_failures:
        print("Smoke artifact validation failures:")
        for failure in result.smoke_failures:
            print(f"- {failure}")
    if result.threshold_failures:
        print("Threshold failures:")
        for failure in result.threshold_failures:
            print(f"- {failure}")
    return 1 if result.should_fail else 0


if __name__ == "__main__":
    sys.exit(main())
