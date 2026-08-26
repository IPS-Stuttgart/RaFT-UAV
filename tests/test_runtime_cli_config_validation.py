from __future__ import annotations

import argparse

import pytest

from raft_uav.runtime_cli_config import (
    add_runtime_configuration_arguments,
    runtime_config_from_args,
)


def _default_runtime_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_configuration_arguments(parser)
    return parser.parse_args([])


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        (
            "tracklet_max_candidates",
            1.5,
            "tracklet_max_candidates must be a positive integer",
        ),
        (
            "tracklet_max_candidate_pool_per_frame",
            True,
            "tracklet_max_candidate_pool_per_frame must be a positive integer",
        ),
        (
            "tracklet_max_candidates_per_track_id",
            0.5,
            "tracklet_max_candidates_per_track_id must be a nonnegative integer",
        ),
    ],
)
def test_runtime_config_rejects_lossy_integer_coercions(
    attribute: str,
    value: object,
    message: str,
) -> None:
    args = _default_runtime_args()
    setattr(args, attribute, value)

    with pytest.raises(ValueError, match=message):
        runtime_config_from_args(args)


def test_runtime_config_rejects_boolean_float_controls() -> None:
    args = _default_runtime_args()
    args.radar_xy_std_m = True

    with pytest.raises(
        ValueError,
        match="radar_xy_std_m must be a finite real scalar",
    ):
        runtime_config_from_args(args)


def test_runtime_config_accepts_integral_scalar_controls() -> None:
    args = _default_runtime_args()
    args.tracklet_max_candidates = 8.0
    args.tracklet_max_candidate_pool_per_frame = "24.0"
    args.tracklet_max_candidates_per_track_id = 0.0

    config = runtime_config_from_args(args)

    tracklet = config["tracklet_viterbi"]
    assert tracklet["max_candidates"] == 8
    assert tracklet["max_candidate_pool_per_frame"] == 24
    assert tracklet["max_candidates_per_track_id"] == 0
