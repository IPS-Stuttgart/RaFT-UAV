"""Exact sparse assignment for proposal-graph tracklet links."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def solve_link_component(
    candidates: Mapping[tuple[int, int], float],
    max_link_cost: float,
) -> dict[int, int]:
    """Solve optional one-to-one links without constructing a dense dummy matrix.

    The dense proposal-graph formulation charges ``max_link_cost / 2`` for every
    unmatched left or right tracklet. A real link therefore replaces two such
    penalties and is useful exactly when its cost is below ``max_link_cost``.

    We keep the same objective but remove the explicit dummy rows. Every left
    tracklet is assigned either to a real right tracklet or to its own private
    dummy column. Unmatched-right penalties are a constant minus one half-cost
    for every real match, which yields the adjusted real-edge costs below. A
    common positive shift keeps all stored sparse weights nonzero without
    changing the minimizer.
    """
    if not math.isfinite(max_link_cost) or max_link_cost <= 0.0:
        raise ValueError("max_link_cost must be finite and positive")
    if not candidates:
        return {}

    left_ids = sorted({left for left, _right in candidates})
    right_ids = sorted({right for _left, right in candidates})
    if not left_ids or not right_ids:
        return {}

    left_position = {value: index for index, value in enumerate(left_ids)}
    right_position = {value: index for index, value in enumerate(right_ids)}
    left_count = len(left_ids)
    right_count = len(right_ids)

    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for (left, right), raw_cost in candidates.items():
        cost = float(raw_cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("candidate link costs must be finite and nonnegative")
        rows.append(left_position[left])
        columns.append(right_position[right])
        # Equivalent to cost - max_link_cost / 2 plus a per-row constant.
        weights.append(cost + 1.0)

    # Each left tracklet receives a private unmatched column. The shifted
    # unmatched-left cost is max_link_cost + 1, so every stored weight is
    # strictly positive and cannot disappear from the SciPy sparse matrix.
    for index in range(left_count):
        rows.append(index)
        columns.append(right_count + index)
        weights.append(max_link_cost + 1.0)

    matrix = coo_matrix(
        (
            np.asarray(weights, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(columns, dtype=np.int64),
            ),
        ),
        shape=(left_count, right_count + left_count),
    ).tocsr()
    matched_rows, matched_columns = min_weight_full_bipartite_matching(matrix)

    links: dict[int, int] = {}
    for row_index, column_index in zip(
        matched_rows,
        matched_columns,
        strict=True,
    ):
        if column_index >= right_count:
            continue
        left = left_ids[int(row_index)]
        right = right_ids[int(column_index)]
        if candidates.get((left, right), math.inf) < max_link_cost:
            links[left] = right
    return links
