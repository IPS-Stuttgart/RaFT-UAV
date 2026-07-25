from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_forward_backward import (
    CandidateForwardBackwardConfig,
    _transition_log_likelihood as first_order_transition,
)
from raft_uav.mmuad.candidate_identity import canonical_track_id
from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
    _transition_log_likelihood as pair_state_transition,
)


def _object_array(values: list[object]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = value
    return result


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([7], id="list"),
        pytest.param((7,), id="tuple"),
        pytest.param({7}, id="set"),
        pytest.param({"track": 7}, id="mapping"),
        pytest.param(pd.Series([7]), id="series"),
    ],
)
def test_canonical_track_id_rejects_container_values(value: object) -> None:
    assert canonical_track_id(value) is None


@pytest.mark.parametrize(
    ("transition", "config"),
    [
        (
            first_order_transition,
            CandidateForwardBackwardConfig(
                transition_distance_std_m=1.0,
                transition_speed_std_mps=0.0,
                max_speed_mps=100.0,
                speed_gate_penalty=0.0,
                source_switch_penalty=0.0,
                branch_switch_penalty=0.0,
                track_continuation_bonus=2.0,
                time_gap_penalty=0.0,
            ),
        ),
        (
            pair_state_transition,
            CandidatePairForwardBackwardConfig(
                transition_distance_std_m=1.0,
                transition_speed_std_mps=0.0,
                max_speed_mps=100.0,
                speed_gate_penalty=0.0,
                source_switch_penalty=0.0,
                branch_switch_penalty=0.0,
                track_continuation_bonus=2.0,
                time_gap_penalty=0.0,
            ),
        ),
    ],
)
def test_temporal_priors_do_not_reward_repeated_container_track_ids(
    transition,
    config,
) -> None:
    previous = {
        "time_s": 0.0,
        "positions": np.asarray([[0.0, 0.0, 0.0]]),
        "sources": np.asarray(["lidar_360"], dtype=object),
        "branches": np.asarray(["raw"], dtype=object),
        "track_ids": _object_array([[7]]),
    }
    current = {
        "time_s": 1.0,
        "positions": np.zeros((2, 3), dtype=float),
        "sources": np.asarray(["lidar_360", "lidar_360"], dtype=object),
        "branches": np.asarray(["raw", "raw"], dtype=object),
        "track_ids": _object_array([[7], "other"]),
    }

    log_likelihood = transition(previous, current, config)

    assert log_likelihood.shape == (1, 2)
    assert log_likelihood[0, 0] == pytest.approx(log_likelihood[0, 1])
