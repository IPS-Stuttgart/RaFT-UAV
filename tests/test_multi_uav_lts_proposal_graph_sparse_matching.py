from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from raft_uav.multi_uav_lts import _proposal_graph_core as graph_core
from raft_uav.multi_uav_lts import _proposal_graph_sparse_matching as sparse_matching
from raft_uav.multi_uav_lts import experimental_proposal_graph_tracker as experimental
from raft_uav.multi_uav_lts import proposal_graph_tracker


def _dense_reference(
    candidates: dict[tuple[int, int], float],
    max_link_cost: float,
) -> dict[int, int]:
    left_ids = sorted({left for left, _right in candidates})
    right_ids = sorted({right for _left, right in candidates})
    if not left_ids or not right_ids:
        return {}
    left_position = {value: index for index, value in enumerate(left_ids)}
    right_position = {value: index for index, value in enumerate(right_ids)}
    left_count = len(left_ids)
    right_count = len(right_ids)
    size = left_count + right_count
    matrix = np.full((size, size), 1e9)
    for (left, right), cost in candidates.items():
        matrix[left_position[left], right_position[right]] = cost
    unmatched = 0.5 * max_link_cost
    for index in range(left_count):
        matrix[index, right_count + index] = unmatched
    for index in range(right_count):
        matrix[left_count + index, index] = unmatched
    matrix[left_count:, right_count:] = 0.0
    rows, columns = linear_sum_assignment(matrix)
    links: dict[int, int] = {}
    for row_index, column_index in zip(rows, columns, strict=True):
        if row_index >= left_count or column_index >= right_count:
            continue
        left = left_ids[int(row_index)]
        right = right_ids[int(column_index)]
        if candidates.get((left, right), math.inf) < max_link_cost:
            links[left] = right
    return links


def _objective(
    links: dict[int, int],
    candidates: dict[tuple[int, int], float],
    max_link_cost: float,
) -> float:
    left_count = len({left for left, _right in candidates})
    right_count = len({right for _left, right in candidates})
    unmatched = left_count + right_count - 2 * len(links)
    return sum(candidates[(left, right)] for left, right in links.items()) + (
        0.5 * max_link_cost * unmatched
    )


def test_sparse_solver_matches_dense_objective_on_rectangular_components() -> None:
    generator = np.random.default_rng(7)
    for left_count, right_count in ((2, 5), (5, 2), (6, 6)):
        for _case in range(20):
            candidates = {
                (left, 100 + right): float(generator.uniform(0.0, 2.24))
                for left in range(left_count)
                for right in range(right_count)
                if generator.random() < 0.55
            }
            if not candidates:
                continue
            dense = _dense_reference(candidates, 2.25)
            sparse = sparse_matching.solve_link_component(candidates, 2.25)
            assert _objective(sparse, candidates, 2.25) == pytest.approx(
                _objective(dense, candidates, 2.25),
                abs=1e-10,
            )
            assert len(sparse) == len(set(sparse.values()))


def test_sparse_solver_materializes_only_edges_and_private_dummies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left_count = 1_000
    edges_per_left = 3
    candidates = {
        (left, 10_000 + (left + offset) % left_count): 0.25 + 0.1 * offset
        for left in range(left_count)
        for offset in range(edges_per_left)
    }

    def fake_solver(matrix):
        assert matrix.shape == (left_count, 2 * left_count)
        assert matrix.nnz == left_count * (edges_per_left + 1)
        rows = np.arange(left_count, dtype=np.int64)
        columns = left_count + rows
        return rows, columns

    monkeypatch.setattr(
        sparse_matching,
        "min_weight_full_bipartite_matching",
        fake_solver,
    )

    assert sparse_matching.solve_link_component(candidates, 2.25) == {}


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_sparse_solver_rejects_invalid_maximum_cost(value: float) -> None:
    with pytest.raises(ValueError, match="max_link_cost"):
        sparse_matching.solve_link_component({(0, 1): 0.5}, value)


def test_experimental_cli_installs_and_restores_sparse_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = graph_core._solve_link_component
    observed = []

    def fake_main(arguments):
        observed.append((tuple(arguments), graph_core._solve_link_component))
        return 0

    monkeypatch.setattr(proposal_graph_tracker, "main", fake_main)

    assert experimental.main(["--min-proposal-confidence", "0.01"]) == 0
    assert observed == [
        (
            ("--min-proposal-confidence", "0.01"),
            sparse_matching.solve_link_component,
        )
    ]
    assert graph_core._solve_link_component is original


def test_experimental_cli_restores_sparse_solver_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = graph_core._solve_link_component

    def fail(_arguments):
        assert graph_core._solve_link_component is sparse_matching.solve_link_component
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(proposal_graph_tracker, "main", fail)

    with pytest.raises(RuntimeError, match="synthetic failure"):
        experimental.main([])
    assert graph_core._solve_link_component is original
