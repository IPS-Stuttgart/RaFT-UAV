from __future__ import annotations

import pandas as pd

from raft_uav.research.optimizer import select_constrained_configs


def test_grouped_optimizer_keeps_numeric_string_metrics() -> None:
    rows = pd.DataFrame(
        {
            "method": ["a", "a", "b", "b"],
            "error_3d_rmse_m": ["10.0", "12.0", "8.0", "9.0"],
            "truth_coverage_rate": ["0.9", "0.8", "0.5", "0.6"],
        }
    )

    ranked = select_constrained_configs(
        rows,
        constraints={"truth_coverage_rate": (">=", 0.8)},
    )

    assert ranked["method"].tolist() == ["a", "b"]
    assert ranked["error_3d_rmse_m"].tolist() == [11.0, 8.5]
    assert ranked["truth_coverage_rate"].tolist() == [0.85, 0.55]
    assert ranked["constraint_feasible"].tolist() == [True, False]


def test_ungrouped_optimizer_sorts_numeric_string_objective_numerically() -> None:
    rows = pd.DataFrame(
        {
            "candidate": ["ten", "two"],
            "error_3d_rmse_m": ["10", "2"],
        }
    )

    ranked = select_constrained_configs(rows, group_columns=())

    assert ranked["candidate"].tolist() == ["two", "ten"]
    assert ranked["error_3d_rmse_m"].tolist() == [2, 10]
