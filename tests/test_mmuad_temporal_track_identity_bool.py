from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.candidate_forward_backward import (
    CandidateForwardBackwardConfig,
    _transition_log_likelihood,
)
from raft_uav.mmuad.candidate_identity import canonical_track_id


@pytest.mark.parametrize(
    "value",
    [True, False, np.bool_(True), np.bool_(False), "true", "FALSE"],
)
def test_canonical_track_id_rejects_boolean_like_values(value: object) -> None:
    assert canonical_track_id(value) is None


def test_temporal_prior_does_not_reward_boolean_missing_track_ids() -> None:
    previous = {
        "time_s": 0.0,
        "positions": np.asarray([[0.0, 0.0, 0.0]]),
        "sources": np.asarray(["lidar_360"], dtype=object),
        "branches": np.asarray(["raw"], dtype=object),
        "track_ids": np.asarray([False], dtype=object),
    }
    current = {
        "time_s": 1.0,
        "positions": np.zeros((2, 3), dtype=float),
        "sources": np.asarray(["lidar_360", "lidar_360"], dtype=object),
        "branches": np.asarray(["raw", "raw"], dtype=object),
        "track_ids": np.asarray([False, "other"], dtype=object),
    }
    config = CandidateForwardBackwardConfig(
        transition_distance_std_m=1.0,
        transition_speed_std_mps=0.0,
        max_speed_mps=100.0,
        speed_gate_penalty=0.0,
        source_switch_penalty=0.0,
        branch_switch_penalty=0.0,
        track_continuation_bonus=2.0,
        time_gap_penalty=0.0,
    )

    log_likelihood = _transition_log_likelihood(previous, current, config)

    assert log_likelihood.shape == (1, 2)
    assert log_likelihood[0, 0] == pytest.approx(log_likelihood[0, 1])
