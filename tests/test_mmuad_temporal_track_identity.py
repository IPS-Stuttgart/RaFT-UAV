from __future__ import annotations

from fractions import Fraction

import numpy as np
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


def _transition_frames(previous_track: object, current_tracks: list[object]) -> tuple[dict, dict]:
    previous = {
        "time_s": 0.0,
        "positions": np.asarray([[0.0, 0.0, 0.0]]),
        "sources": np.asarray(["lidar_360"], dtype=object),
        "branches": np.asarray(["raw"], dtype=object),
        "track_ids": np.asarray([previous_track], dtype=object),
    }
    current = {
        "time_s": 1.0,
        "positions": np.zeros((len(current_tracks), 3), dtype=float),
        "sources": np.asarray(["lidar_360"] * len(current_tracks), dtype=object),
        "branches": np.asarray(["raw"] * len(current_tracks), dtype=object),
        "track_ids": np.asarray(current_tracks, dtype=object),
    }
    return previous, current


def test_canonical_track_id_matches_numeric_csv_representations() -> None:
    assert canonical_track_id(491) == "491"
    assert canonical_track_id(491.0) == "491"
    assert canonical_track_id("491") == "491"
    assert canonical_track_id("491.0") == "491"
    assert canonical_track_id(" 491.000 ") == "491"
    assert canonical_track_id(-1) is None
    assert canonical_track_id("nan") is None


def test_canonical_track_id_preserves_exact_rationals_beyond_float_precision() -> None:
    large_integer = (2**53) + 1
    fractional = Fraction(large_integer, 2)

    assert canonical_track_id(Fraction(large_integer, 1)) == str(large_integer)
    assert canonical_track_id(fractional) == "4503599627370496.5"
    assert canonical_track_id(fractional) != canonical_track_id(4503599627370496)
    assert canonical_track_id(Fraction(1, 3)) == "1/3"


def test_canonical_track_id_preserves_opaque_leading_zeros() -> None:
    assert canonical_track_id("001") == "001"
    assert canonical_track_id("001.0") == "001.0"
    assert canonical_track_id("001") != canonical_track_id(1.0)


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
def test_temporal_priors_reward_same_numeric_track_across_csv_types(
    transition,
    config,
) -> None:
    previous, current = _transition_frames(491, ["491.0", "other"])

    log_likelihood = transition(previous, current, config)

    assert log_likelihood.shape == (1, 2)
    assert log_likelihood[0, 0] - log_likelihood[0, 1] == pytest.approx(2.0)


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
def test_temporal_priors_do_not_collapse_opaque_track_ids(
    transition,
    config,
) -> None:
    previous, current = _transition_frames("001", [1.0, "other"])

    log_likelihood = transition(previous, current, config)

    assert log_likelihood[0, 0] == pytest.approx(log_likelihood[0, 1])
