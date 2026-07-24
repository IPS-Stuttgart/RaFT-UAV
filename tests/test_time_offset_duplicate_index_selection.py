from __future__ import annotations

import pandas as pd

from raft_uav.diagnostics.time_offset import highest_catprob_candidate


def test_highest_catprob_candidate_returns_one_row_with_duplicate_index_labels() -> None:
    candidates = pd.DataFrame(
        {
            "cat_prob_uav": [0.2, 0.9, 0.4],
            "east_m": [10.0, 20.0, 30.0],
            "north_m": [1.0, 2.0, 3.0],
            "up_m": [100.0, 200.0, 300.0],
        },
        index=["shared", "shared", "other"],
    )

    selected = highest_catprob_candidate(candidates)

    assert isinstance(selected, pd.Series)
    assert float(selected["cat_prob_uav"]) == 0.9
    assert selected[["east_m", "north_m", "up_m"]].tolist() == [20.0, 2.0, 200.0]
