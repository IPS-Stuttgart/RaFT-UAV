"""Globally consistent one-to-one matching for timestamp grids."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def optimal_timestamp_assignment(
    requested_times: Iterable[float],
    prediction_times: Iterable[float],
    *,
    tolerance_s: float,
) -> dict[int, int]:
    """Match timestamps globally, maximizing coverage then minimizing time error.

    Returned keys and values are positions in the original request and prediction
    arrays. Every request and prediction appears at most once in the assignment.
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
    costs: list[float] = []
    scale = tolerance + 1.0
    epsilon = np.finfo(float).eps
    for request_rank, request_time in enumerate(sorted_requests):
        left = int(np.searchsorted(sorted_predictions, request_time - tolerance, side="left"))
        right = int(np.searchsorted(sorted_predictions, request_time + tolerance, side="right"))
        for prediction_rank in range(left, right):
            gap = abs(float(sorted_predictions[prediction_rank] - request_time))
            tie_break = epsilon * (
                1.0
                + prediction_rank / (prediction_count + 1.0)
                + request_rank / ((request_count + 1.0) * (prediction_count + 1.0))
            )
            rows.append(request_rank)
            columns.append(prediction_rank)
            costs.append(gap / scale + tie_break)

        rows.append(request_rank)
        columns.append(prediction_count + request_rank)
        costs.append(float(min(request_count, prediction_count) + 1))

    graph = coo_matrix(
        (np.asarray(costs, dtype=float), (rows, columns)),
        shape=(request_count, prediction_count + request_count),
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
