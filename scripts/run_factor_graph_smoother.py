#!/usr/bin/env python3
"""Run the lightweight factor-graph smoother on normalized measurement CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from raft_uav.research.factor_graph import (
    LeastSquaresSmoothingConfig,
    coordinate_descent_association_and_smoothing,
    smooth_position_trajectory,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=None)
    parser.add_argument("--radar", type=Path, default=None)
    parser.add_argument("--rf", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--motion-std-mps2", type=float, default=4.0)
    parser.add_argument("--measurement-std-m", type=float, default=25.0)
    parser.add_argument("--rf-std-m", type=float, default=50.0)
    args = parser.parse_args(argv)

    config = LeastSquaresSmoothingConfig(
        motion_std_mps2=args.motion_std_mps2,
        measurement_std_m=args.measurement_std_m,
        rf_std_m=args.rf_std_m,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.measurements is not None:
        result = smooth_position_trajectory(pd.read_csv(args.measurements), config=config)
        selected = None
    elif args.radar is not None:
        rf = pd.read_csv(args.rf) if args.rf is not None else None
        estimates, selected = coordinate_descent_association_and_smoothing(
            pd.read_csv(args.radar),
            rf,
            iterations=args.iterations,
            config=config,
        )
        result = smooth_position_trajectory(estimates, config=config)
        result = result.__class__(estimates, result.cost, result.optimality, result.iterations, result.success, result.message)
    else:
        raise ValueError("provide either --measurements or --radar")
    estimates_path = args.output_dir / "factor_graph_estimates.csv"
    result.estimates.to_csv(estimates_path, index=False)
    if selected is not None:
        selected.to_csv(args.output_dir / "factor_graph_selected_radar.csv", index=False)
    summary = {
        "success": result.success,
        "cost": result.cost,
        "optimality": result.optimality,
        "iterations": result.iterations,
        "message": result.message,
        "estimates_csv": str(estimates_path),
    }
    summary_path = args.output_dir / "factor_graph_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary_json={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
