#!/usr/bin/env python3
"""Compare two RaFT-UAV artifact directories for deterministic output."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd

from raft_uav.evaluation.fifth_wave_diagnostics import deterministic_artifact_summary


_ESTIMATE_REQUIRED_COLUMNS = ("time_s", "east_m", "north_m", "up_m")


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _record_estimate_schema(
    summary: dict[str, object],
    estimates_a: pd.DataFrame,
    estimates_b: pd.DataFrame,
) -> None:
    missing_a = [
        column for column in _ESTIMATE_REQUIRED_COLUMNS if column not in estimates_a.columns
    ]
    missing_b = [
        column for column in _ESTIMATE_REQUIRED_COLUMNS if column not in estimates_b.columns
    ]
    schema_equal = not missing_a and not missing_b
    summary["estimate_missing_columns_a"] = missing_a
    summary["estimate_missing_columns_b"] = missing_b
    summary["estimate_schema_equal"] = schema_equal
    if not schema_equal:
        summary["estimates_nearly_equal"] = False


def _record_optional_artifact_presence(
    summary: dict[str, object],
    *,
    label: str,
    artifact_a: pd.DataFrame | None,
    artifact_b: pd.DataFrame | None,
) -> None:
    present_a = artifact_a is not None
    present_b = artifact_b is not None
    presence_equal = present_a == present_b
    summary[f"{label}_artifact_present_a"] = present_a
    summary[f"{label}_artifact_present_b"] = present_b
    summary[f"{label}_artifact_presence_equal"] = presence_equal
    if not presence_equal:
        summary[f"{label}_rows_equal"] = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/determinism_check.json"))
    parser.add_argument("--atol", type=float, default=1.0e-9)
    parser.add_argument("--fail-on-difference", action="store_true")
    args = parser.parse_args(_normalize_negative_option_values(sys.argv[1:]))
    if not math.isfinite(args.atol) or args.atol < 0.0:
        parser.error("--atol must be finite and non-negative")

    estimates_a = pd.read_csv(args.run_a / "estimates.csv")
    estimates_b = pd.read_csv(args.run_b / "estimates.csv")
    selected_a = _read_optional_csv(args.run_a / "selected_radar.csv")
    selected_b = _read_optional_csv(args.run_b / "selected_radar.csv")
    summary = deterministic_artifact_summary(
        estimates_a,
        estimates_b,
        selected_a=selected_a,
        selected_b=selected_b,
        atol=args.atol,
    )
    _record_estimate_schema(summary, estimates_a, estimates_b)
    _record_optional_artifact_presence(
        summary,
        label="selected",
        artifact_a=selected_a,
        artifact_b=selected_b,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary_json={args.output_json}")
    if args.fail_on_difference:
        if not bool(summary.get("estimates_nearly_equal", False)):
            return 1
        if not bool(summary["selected_artifact_presence_equal"]):
            return 1
        if "selected_rows_equal" in summary and not bool(summary["selected_rows_equal"]):
            return 1
    return 0


def _normalize_negative_option_values(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--atol" and index + 1 < len(argv):
            value = argv[index + 1]
            try:
                float(value)
            except ValueError:
                pass
            else:
                normalized.append(f"--atol={value}")
                index += 2
                continue
        normalized.append(item)
        index += 1
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
