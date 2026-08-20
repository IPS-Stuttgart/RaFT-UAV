from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.research.optimizer import pareto_front
from raft_uav.research.optimizer import select_constrained_configs


def test_pareto_front_ignores_nonfinite_minimize_rows_as_dominators() -> None:
    rows = pd.DataFrame({"rmse": [1.0, -np.inf, 2.0]})

    mask = pareto_front(rows, minimize_columns=["rmse"])

    assert mask.tolist() == [True, False, False]


def test_pareto_front_ignores_nonfinite_maximize_rows_as_dominators() -> None:
    rows = pd.DataFrame({"recall": [0.9, np.inf, 0.8]})

    mask = pareto_front(rows, minimize_columns=[], maximize_columns=["recall"])

    assert mask.tolist() == [True, False, False]


def test_pareto_front_without_objectives_preserves_index_and_marks_all_front() -> None:
    rows = pd.DataFrame({"method": ["a", "b"]}, index=[10, 20])

    mask = pareto_front(rows, minimize_columns=[])

    assert mask.tolist() == [True, True]
    assert mask.index.tolist() == [10, 20]


@pytest.mark.parametrize("minimize", ["False", 0, 1, None, [], {}])
def test_constrained_selection_rejects_non_boolean_minimize(minimize: object) -> None:
    rows = pd.DataFrame({"method": ["a", "b"], "error_3d_rmse_m": [1.0, 2.0]})

    with pytest.raises(TypeError, match="minimize must be a Boolean"):
        select_constrained_configs(rows, minimize=minimize)


def test_constrained_selection_accepts_numpy_boolean_minimize() -> None:
    rows = pd.DataFrame({"method": ["a", "b"], "error_3d_rmse_m": [1.0, 2.0]})

    ranked = select_constrained_configs(rows, minimize=np.bool_(False))

    assert ranked["method"].tolist() == ["b", "a"]


def test_constrained_selection_accepts_scalar_group_column_name() -> None:
    rows = pd.DataFrame(
        {
            "method": ["a", "a", "b", "b"],
            "error_3d_rmse_m": [1.0, 3.0, 4.0, 6.0],
        }
    )

    ranked = select_constrained_configs(rows, group_columns="method")

    assert ranked["method"].tolist() == ["a", "b"]
    assert ranked["error_3d_rmse_m"].tolist() == pytest.approx([2.0, 5.0])


def test_pareto_front_accepts_scalar_objective_column_name() -> None:
    rows = pd.DataFrame({"rmse": [1.0, 2.0, 0.5]})

    mask = pareto_front(rows, minimize_columns="rmse")

    assert mask.tolist() == [False, False, True]


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf, True])
def test_constrained_selection_rejects_invalid_thresholds(threshold: object) -> None:
    rows = pd.DataFrame(
        {
            "method": ["a", "b"],
            "error_3d_rmse_m": [1.0, 2.0],
            "coverage": [0.9, 0.8],
        }
    )

    with pytest.raises(ValueError, match="constraint threshold must be finite"):
        select_constrained_configs(
            rows,
            constraints={"coverage": (">=", threshold)},
        )
