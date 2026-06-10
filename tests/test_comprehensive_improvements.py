from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.comprehensive_improvements import (
    candidate_recall_regret_table,
    time_bias_grid_search,
)


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


def test_candidate_recall_regret_ignores_invalid_candidate_positions() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "frame_index": [0, 0],
            "track_id": [99, 7],
            "east_m": [np.nan, 0.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
            "cat_prob_uav": [0.99, 0.1],
        }
    )
    truth = pd.DataFrame({"time_s": [0.0], "east_m": [0.0], "north_m": [0.0], "up_m": [0.0]})

    table = candidate_recall_regret_table(
        radar,
        truth,
        truth_gate_m=1.0,
        truth_time_gate_s=0.5,
        catprob_threshold=0.5,
    )

    assert table.loc[0, "best_candidate_error_m"] == 0.0
    assert bool(table.loc[0, "candidate_available"])
    assert bool(table.loc[0, "correct_candidate_lost_by_catprob"])
    assert table.loc[0, "failure_bucket"] == "missed_association"
