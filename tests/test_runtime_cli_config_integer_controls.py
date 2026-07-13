from __future__ import annotations

import argparse

import pytest

from raft_uav.runtime_cli_config import add_runtime_configuration_arguments
from raft_uav.runtime_cli_config import runtime_config_from_args


def _default_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_configuration_arguments(parser)
    return parser.parse_args([])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tracklet_max_candidates", 2.5, "positive integer"),
        ("tracklet_max_candidates", True, "positive integer"),
        ("tracklet_max_candidate_pool_per_frame", float("nan"), "positive integer"),
        ("tracklet_max_candidate_pool_per_frame", float("inf"), "positive integer"),
        ("tracklet_max_candidates_per_track_id", 1.5, "nonnegative integer"),
        ("tracklet_max_candidates_per_track_id", False, "nonnegative integer"),
    ],
)
def test_runtime_config_rejects_malformed_integer_controls(
    field: str,
    value: object,
    message: str,
) -> None:
    args = _default_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        runtime_config_from_args(args)


def test_runtime_config_accepts_integer_equivalent_numeric_controls() -> None:
    args = _default_args()
    args.tracklet_max_candidates = 12.0
    args.tracklet_max_candidate_pool_per_frame = 24.0
    args.tracklet_max_candidates_per_track_id = 0.0

    config = runtime_config_from_args(args)["tracklet_viterbi"]

    assert config["max_candidates"] == 12
    assert config["max_candidate_pool_per_frame"] == 24
    assert config["max_candidates_per_track_id"] == 0
