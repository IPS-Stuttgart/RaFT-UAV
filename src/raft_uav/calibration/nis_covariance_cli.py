"""Command-line interface for NIS-based covariance calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from raft_uav.calibration.nis_covariance import (
    NIS_COVARIANCE_CALIBRATION_METHODS,
    environment_assignment,
    fit_nis_covariance_calibration_from_paths,
    write_nis_covariance_calibration,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raft-uav-calibrate-nis-covariance")
    parser.add_argument(
        "diagnostics",
        nargs="+",
        type=Path,
        help="diagnostics.csv files or directories containing run-baseline outputs",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/nis_covariance_calibration.json"),
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=None,
        help="optional flat per-source summary CSV",
    )
    parser.add_argument(
        "--method",
        choices=NIS_COVARIANCE_CALIBRATION_METHODS,
        default="mean",
        help="match either mean NIS or a chi-square quantile",
    )
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-scale", type=float, default=0.25)
    parser.add_argument("--max-scale", type=float, default=25.0)
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="fit on accepted and rejected updates instead of accepted updates only",
    )
    args = parser.parse_args(argv)

    payload = fit_nis_covariance_calibration_from_paths(
        args.diagnostics,
        method=args.method,
        quantile=args.quantile,
        min_samples=args.min_samples,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        accepted_only=not args.include_rejected,
    )
    output_json = write_nis_covariance_calibration(payload, args.output_json)
    summary = _summary_frame(payload)
    if args.output_summary_csv is not None:
        args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_summary_csv, index=False)
        print(f"summary_csv={args.output_summary_csv}")
    print(f"calibration_json={output_json}")
    print(f"runtime_env={environment_assignment(output_json)}")
    if not summary.empty:
        enabled = int(summary["enabled"].sum())
        print(f"calibrated_groups={enabled}/{len(summary)}")
        for _, row in summary.sort_values(["source", "measurement_dim"]).iterrows():
            status = "enabled" if bool(row["enabled"]) else "disabled"
            print(
                "group="
                f"{row['source']}:{int(row['measurement_dim'])} "
                f"count={int(row['count'])} "
                f"scale={float(row['applied_scale']):.6g} "
                f"raw={float(row['raw_scale']):.6g} "
                f"{status}"
            )
    return 0


def _summary_frame(payload: dict) -> pd.DataFrame:
    groups = payload.get("groups", {})
    if not isinstance(groups, dict):
        return pd.DataFrame()
    rows = []
    for key, group in groups.items():
        if not isinstance(group, dict):
            continue
        row = {"group": key}
        row.update(group)
        rows.append(row)
    return pd.DataFrame.from_records(rows)


if __name__ == "__main__":
    raise SystemExit(main())
