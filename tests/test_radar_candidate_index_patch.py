import numpy as np
import pandas as pd

from raft_uav.baselines import radar_association
from raft_uav.baselines._radar_candidate_index_patch import (
    _unique_scored_index,
    apply_radar_candidate_index_patch,
)
from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker


def test_unique_scored_index_preserves_ordinary_indices():
    scored = pd.DataFrame({"association_nis": [2.0, 1.0]}, index=[10, 20])

    normalized = _unique_scored_index(scored)

    assert normalized is scored
    assert normalized.index.tolist() == [10, 20]


def test_radar_scoring_normalizes_duplicate_candidate_indices():
    tracker = AsyncConstantVelocityKalmanTracker(
        initial_position=np.zeros(3),
        initial_time_s=0.0,
    )
    candidates = pd.DataFrame(
        [
            {
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 1.0,
                "north_m": 0.0,
                "up_m": 0.0,
            },
            {
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
            },
        ],
        index=[7, 7],
    )

    scored = radar_association._nis_scored_candidates(
        candidates,
        tracker,
        np.diag([25.0**2, 25.0**2, 35.0**2]),
    )
    selected = scored.loc[scored["association_nis"].idxmin()]

    assert scored.index.tolist() == [0, 1]
    assert isinstance(selected, pd.Series)
    assert int(selected["track_id"]) == 1


def test_radar_candidate_index_patch_is_idempotent():
    installed = radar_association._nis_scored_candidates

    apply_radar_candidate_index_patch(radar_association)

    assert radar_association._nis_scored_candidates is installed
