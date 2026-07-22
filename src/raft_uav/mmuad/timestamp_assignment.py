"""Globally consistent one-to-one matching for timestamp grids."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def _solve_sparse_assignment(
    rows: list[int],
    columns: list[int],
    costs: list[float],
    *,
    shape: tuple[int, int],
    request_order: np.ndarray,
    prediction_order: np.ndarray,
    prediction_count: int,
) -> dict[int, int]:
    """Solve one sparse assignment and restore original array positions."""

    graph = coo_matrix(
        (np.asarray(costs, dtype=float), (rows, columns)),
        shape=shape,
    ).tocsr()
    matched_requests, matched_columns = min_weight_full_bipartite_matching(graph)

    assignment: dict[int, int] = {}
    for request_rank, column in zip(matched_requests, matched_columns, strict=True):
        if int(column) >= prediction_count:
            continue
        request_index = int(request_order[int(request_rank)])
        prediction_index = int(prediction_order[int(column)])
        assignment[request_index] = prediction_index
    return assignment


def _assignment_error(
    assignment: dict[int, int],
    requests: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """Return an accurately accumulated absolute timestamp error."""

    return math.fsum(
        abs(float(predictions[prediction_index] - requests[request_index]))
        for request_index, prediction_index in assignment.items()
    )


def optimal_timestamp_assignment(
    requested_times: Iterable[float],
    prediction_times: Iterable[float],
    *,
    tolerance_s: float,
) -> dict[int, int]:
    """Match timestamps globally, maximizing coverage then minimizing time error.

    Returned keys and values are positions in the original request and prediction
    arrays. Every request and prediction appears at most once in the assignment.
    Equal-error assignments preserve the stable chronological ordering of the
    request and prediction arrays.
    """

    requests = np.asarray(list(requested_times), dtype=float)
    predictions = np.asarray(list(prediction_times), dtype=float)
    if requests.ndim != 1 or predictions.ndim != 1:
        raise ValueError("timestamp arrays must be one-dimensional")
    if not np.isfinite(requests).all() or not np.isfinite(predictions).all():
        raise ValueError("timestamp arrays must contain only finite values")
    tolerance = float(tolerance_s)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_s must be non-negative and finite")
    if requests.size == 0 or predictions.size == 0:
        return {}

    request_order = np.argsort(requests, kind="stable")
    prediction_order = np.argsort(predictions, kind="stable")
    sorted_requests = requests[request_order]
    sorted_predictions = predictions[prediction_order]
    request_count = int(sorted_requests.size)
    prediction_count = int(sorted_predictions.size)

    rows: list[int] = []
    columns: list[int] = []
    primary_costs: list[float] = []
    stable_costs: list[float] = []
    scale = tolerance + 1.0
    tie_unit = 8.0 * np.finfo(float).eps
    positive_zero_guard = np.finfo(float).tiny
    for request_rank, request_time in enumerate(sorted_requests):
        # Widen the binary-search window by one representable value. Rounded
        # subtraction/addition can otherwise exclude an exact boundary match.
        lower = np.nextafter(request_time - tolerance, -np.inf)
        upper = np.nextafter(request_time + tolerance, np.inf)
        left = int(np.searchsorted(sorted_predictions, lower, side="left"))
        right = int(np.searchsorted(sorted_predictions, upper, side="right"))
        for prediction_rank in range(left, right):
            gap = abs(float(sorted_predictions[prediction_rank] - request_time))
            # The widened window can include a value just outside the tolerance,
            # so enforce the actual matching predicate explicitly.
            if gap > tolerance:
                continue
            primary_cost = gap / scale
            # A multi-ULP order penalty makes exact error ties deterministic in
            # SciPy's sparse matcher. The unperturbed solve below prevents that
            # secondary preference from ever increasing the primary time error.
            rank_distance = request_rank - prediction_rank
            tie_break = tie_unit * float(1 + rank_distance * rank_distance)
            rows.append(request_rank)
            columns.append(prediction_rank)
            # Sparse matching drops explicit zero-weight edges. Adding the smallest
            # positive float to every real edge preserves the primary objective for
            # a fixed cardinality while retaining exact timestamp matches.
            primary_costs.append(primary_cost + positive_zero_guard)
            stable_costs.append(primary_cost + tie_break)

        rows.append(request_rank)
        columns.append(prediction_count + request_rank)
        dummy_cost = float(min(request_count, prediction_count) + 1)
        primary_costs.append(dummy_cost)
        stable_costs.append(dummy_cost)

    shape = (request_count, prediction_count + request_count)
    primary_assignment = _solve_sparse_assignment(
        rows,
        columns,
        primary_costs,
        shape=shape,
        request_order=request_order,
        prediction_order=prediction_order,
        prediction_count=prediction_count,
    )
    stable_assignment = _solve_sparse_assignment(
        rows,
        columns,
        stable_costs,
        shape=shape,
        request_order=request_order,
        prediction_order=prediction_order,
        prediction_count=prediction_count,
    )

    if len(stable_assignment) != len(primary_assignment):
        return (
            stable_assignment
            if len(stable_assignment) > len(primary_assignment)
            else primary_assignment
        )
    if _assignment_error(stable_assignment, requests, predictions) <= _assignment_error(
        primary_assignment,
        requests,
        predictions,
    ):
        return stable_assignment
    return primary_assignment
