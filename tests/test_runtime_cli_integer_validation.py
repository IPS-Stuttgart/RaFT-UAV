from __future__ import annotations

import pytest

from raft_uav.runtime_cli_config import _nonnegative_int, _positive_int


@pytest.mark.parametrize("value", [1.5, "2.5", True, False])
def test_positive_int_rejects_fractional_and_boolean_values(value: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        _positive_int(value, "tracklet_max_candidates")


@pytest.mark.parametrize("value", [0.5, "3.25", True, False])
def test_nonnegative_int_rejects_fractional_and_boolean_values(value: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        _nonnegative_int(value, "tracklet_max_candidates_per_track_id")


def test_runtime_integer_validators_accept_integer_like_values() -> None:
    assert _positive_int("2.0", "tracklet_max_candidates") == 2
    assert _positive_int(3.0, "tracklet_max_candidates") == 3
    assert _nonnegative_int(0.0, "tracklet_max_candidates_per_track_id") == 0
