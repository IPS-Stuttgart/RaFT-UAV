from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.diagnostics.time_offset import (
    catprob_candidate_pool,
    highest_catprob_candidate,
)


def _candidates(scores: list[float]) -> pd.DataFrame:
    count = len(scores)
    return pd.DataFrame(
        {
            "cat_prob_uav": scores,
            "east_m": np.arange(count, dtype=float),
            "north_m": np.zeros(count, dtype=float),
            "up_m": np.zeros(count, dtype=float),
        }
    )


def test_highest_catprob_candidate_ignores_nonfinite_scores() -> None:
    candidates = _candidates([np.inf, 0.9, -np.inf, np.nan])

    selected = highest_catprob_candidate(candidates)

    assert selected is not None
    assert float(selected["cat_prob_uav"]) == 0.9
    assert float(selected["east_m"]) == 1.0


def test_catprob_candidate_pool_does_not_treat_positive_infinity_as_confident() -> None:
    candidates = _candidates([np.inf, 0.9, 0.4])

    selected = catprob_candidate_pool(candidates, threshold=0.5)

    assert selected["cat_prob_uav"].tolist() == [0.9]
    assert selected["east_m"].tolist() == [1.0]


def test_nonfinite_only_scores_preserve_existing_fallback() -> None:
    candidates = _candidates([np.inf, -np.inf, np.nan])

    selected = highest_catprob_candidate(candidates)
    pooled = catprob_candidate_pool(candidates, threshold=0.5)

    assert selected is not None
    assert float(selected["east_m"]) == 0.0
    pd.testing.assert_frame_equal(pooled, candidates)
