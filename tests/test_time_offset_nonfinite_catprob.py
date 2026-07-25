import numpy as np
import pandas as pd

from raft_uav.diagnostics.time_offset import (
    catprob_candidate_pool,
    highest_catprob_candidate,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": [1, 2, 3],
            "cat_prob_uav": [np.inf, 0.8, np.nan],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )


def test_highest_catprob_ignores_nonfinite_scores():
    selected = highest_catprob_candidate(_candidates())

    assert selected is not None
    assert int(selected["track_id"]) == 2


def test_catprob_pool_excludes_nonfinite_scores():
    pooled = catprob_candidate_pool(_candidates(), threshold=0.7)

    assert pooled["track_id"].tolist() == [2]
