import pandas as pd

from raft_uav.diagnostics.time_offset import catprob_candidate_pool


def test_catprob_candidate_pool_with_none_keeps_candidates():
    candidates = pd.DataFrame(
        {
            "cat_prob_uav": [0.1, 0.8],
            "east_m": [0.0, 1.0],
            "north_m": [0.0, 1.0],
            "up_m": [0.0, 1.0],
        }
    )

    result = catprob_candidate_pool(candidates, None)

    pd.testing.assert_frame_equal(result, candidates)
