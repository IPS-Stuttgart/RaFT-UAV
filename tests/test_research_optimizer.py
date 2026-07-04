from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.optimizer import pareto_front


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
