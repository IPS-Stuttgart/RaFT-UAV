from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.diagnostics.time_offset import (
    catprob_candidate_pool,
    highest_catprob_candidate,
    nearest_candidate_to_truth,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "east_m": [0.1 + 4.0j, 2.0, np.complex128(3.0 + 0.0j)],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
            "cat_prob_uav": [0.99 + 2.0j, 0.8, np.complex128(0.7 + 0.0j)],
            "track_id": ["nonreal", "real", "promoted-real"],
        }
    )


def test_nearest_candidate_rejects_nonreal_position_without_losing_real_rows() -> None:
    selected = nearest_candidate_to_truth(_candidates(), np.zeros(3))

    assert selected is not None
    assert selected["track_id"] == "real"


def test_highest_catprob_rejects_nonreal_scores() -> None:
    selected = highest_catprob_candidate(_candidates())

    assert selected is not None
    assert selected["track_id"] == "real"


def test_catprob_pool_preserves_zero_imaginary_values_after_dtype_promotion() -> None:
    selected = catprob_candidate_pool(_candidates(), 0.6)

    assert selected["track_id"].tolist() == ["real", "promoted-real"]
