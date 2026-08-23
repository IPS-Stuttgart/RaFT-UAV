#!/usr/bin/env python3
"""Rank ablation rows under coverage, switch-count, and tail-error constraints."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from raft_uav.research.optimizer import pareto_front, select_constrained_configs


def _constrained_pareto_front(
    ranked: pd.DataFrame,
    *,
    minimize_columns: Sequence[str],
    maximize_columns: Sequence[str],
) -> pd.Series:
    """Return the Pareto front over feasible configurations only."""

    front = pd.Series(False, index=ranked.index, dtype=bool)
    feasible = ranked["constraint_feasible"].eq(True)
    if feasible.any():
        front.loc[feasible] = pareto_front(
            ranked.loc[feasible],
            minimize_columns=minimize_columns,
            maximize_columns=maximize_columns,
        )
    return front


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--objective", default="error_3d_rmse_m")
    parser.add_argument("--group-columns", nargs="*", default=["method"])
    parser.add_argument(
        "--constraint",
        action="append",
        default=[],
        help="Constraint as column:op:value, e.g. truth_coverage_rate:>=:0.95",
    )
    parser.add_argument("--pareto-minimize", nargs="*", default=[])
    parser.add_argument("--pareto-maximize", nargs="*", default=[])
    args = parser.parse_args(argv)

    rows = pd.read_csv(args.summary_csv)
    constraints = []
    for item in args.constraint:
        column, op, value = item.split(":", 2)
        constraints.append((column, op, float(value)))
    ranked = select_constrained_configs(
        rows,
        objective=args.objective,
        constraints=constraints,
        group_columns=args.group_columns,
    )
    if args.pareto_minimize or args.pareto_maximize:
        ranked["pareto_front"] = _constrained_pareto_front(
            ranked,
            minimize_columns=args.pareto_minimize,
            maximize_columns=args.pareto_maximize,
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(args.output_csv, index=False)
    print(f"ranked_csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
