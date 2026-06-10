from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.comprehensive_improvements import time_bias_grid_search


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.1],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )


def test_time_bias_grid_search_handles_unsorted_truth_times() -> None:
    truth = pd.DataFrame(
        {
            "time_s": [2.0, 1.0, 0.0],
            "east_m": [200.0, 100.0, 0.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )

    table = time_bias_grid_search(
        _frame(),
        truth,
        source="radar",
        offsets_s=[0.0],
        max_time_delta_s=0.2,
        dimensions=3,
    )

    assert table.loc[0, "count"] == 1
    assert np.isclose(table.loc[0, "rmse_m"], 0.0)


def test_time_bias_grid_search_reports_zero_count_for_empty_truth() -> None:
    truth = pd.DataFrame(columns=["time_s", "east_m", "north_m", "up_m"])

    table = time_bias_grid_search(_frame(), truth, source="radar", offsets_s=[-1.0, 0.0])

    assert table["count"].tolist() == [0, 0]
    assert table["offset_s"].tolist() == [-1.0, 0.0]
