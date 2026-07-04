"""Constrained experiment selection utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def select_constrained_configs(
    rows: pd.DataFrame,
    *,
    objective: str = "error_3d_rmse_m",
    minimize: bool = True,
    constraints: Mapping[str, tuple[str, float]] | None = None,
    group_columns: Sequence[str] = ("method",),
) -> pd.DataFrame:
    """Rank experiment configurations subject to numeric constraints.

    ``constraints`` maps column names to ``(operator, threshold)`` where operator
    is one of ``<=,<,>=,>,==``.  Rows are aggregated by ``group_columns`` using
    means for numeric columns before filtering.
    """

    if rows.empty:
        return rows.copy()
    constraints = constraints or {}
    group_columns = tuple(group_columns)
    if group_columns:
        grouped = rows.groupby(list(group_columns), dropna=False).mean(numeric_only=True).reset_index()
    else:
        grouped = rows.copy()
    keep = np.ones(len(grouped), dtype=bool)
    for column, (op, threshold) in constraints.items():
        values = pd.to_numeric(grouped[column], errors="coerce").to_numpy(dtype=float)
        keep &= _compare(values, op, float(threshold))
    feasible = grouped.loc[keep].copy()
    feasible["constraint_feasible"] = True
    infeasible = grouped.loc[~keep].copy()
    infeasible["constraint_feasible"] = False
    ranked = pd.concat([feasible, infeasible], ignore_index=True, sort=False)
    ranked = ranked.sort_values(
        ["constraint_feasible", objective],
        ascending=[False, bool(minimize)],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["constrained_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def pareto_front(
    rows: pd.DataFrame,
    *,
    minimize_columns: Sequence[str],
    maximize_columns: Sequence[str] = (),
) -> pd.Series:
    """Return a Boolean mask indicating Pareto-front rows."""

    if rows.empty:
        return pd.Series(dtype=bool, index=rows.index)
    values = []
    for column in minimize_columns:
        values.append(pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float))
    for column in maximize_columns:
        values.append(-pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=float))
    if not values:
        return pd.Series(True, index=rows.index, dtype=bool)
    matrix = np.column_stack(values)
    finite = np.isfinite(matrix).all(axis=1)
    front = finite.copy()
    for i in range(len(rows)):
        if not finite[i]:
            continue
        dominates = finite & np.all(matrix <= matrix[i], axis=1) & np.any(matrix < matrix[i], axis=1)
        dominates[i] = False
        if np.any(dominates):
            front[i] = False
    return pd.Series(front, index=rows.index)


def _compare(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    if op == "<=":
        return values <= threshold
    if op == "<":
        return values < threshold
    if op == ">=":
        return values >= threshold
    if op == ">":
        return values > threshold
    if op == "==":
        return values == threshold
    raise ValueError(f"unsupported constraint operator {op!r}")
