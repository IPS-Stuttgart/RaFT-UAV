from __future__ import annotations

import argparse

import numpy as np
import pytest

from raft_uav.runtime_cli_config import (
    add_runtime_configuration_arguments,
    runtime_config_from_args,
)


def _default_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_configuration_arguments(parser)
    return parser.parse_args([])


def _boxed(value: object) -> np.ndarray:
    out = np.empty((), dtype=object)
    out[()] = value
    return out


def _cyclic_scalar() -> np.ndarray:
    out = np.empty((), dtype=object)
    out[()] = out
    return out


@pytest.mark.parametrize(
    "value",
    [
        _boxed(_boxed(True)),
        _boxed(np.array([3.0])),
        _boxed(_boxed(1.0 + 2.0j)),
        _cyclic_scalar(),
    ],
)
def test_runtime_float_controls_reject_nested_pseudo_scalars(value: object) -> None:
    args = _default_args()
    args.radar_xy_std_m = value

    with pytest.raises(ValueError, match="radar_xy_std_m"):
        runtime_config_from_args(args)


@pytest.mark.parametrize(
    "field",
    [
        "tracklet_max_candidates",
        "tracklet_max_candidate_pool_per_frame",
        "tracklet_max_candidates_per_track_id",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        _boxed(_boxed(True)),
        _boxed(np.array([3.0])),
        _boxed(_boxed(1.0 + 2.0j)),
        _cyclic_scalar(),
    ],
)
def test_runtime_integer_controls_reject_nested_pseudo_scalars(
    field: str,
    value: object,
) -> None:
    args = _default_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match=field):
        runtime_config_from_args(args)


def test_runtime_controls_accept_recursively_boxed_real_scalars() -> None:
    args = _default_args()
    args.radar_xy_std_m = _boxed(_boxed(np.float64(12.5)))
    args.tracklet_max_candidates = _boxed(_boxed(np.float64(6.0)))
    args.tracklet_max_candidates_per_track_id = _boxed(_boxed(np.int64(2)))

    config = runtime_config_from_args(args)

    assert config["radar_covariance"]["xy_std_m"] == 12.5
    assert config["tracklet_viterbi"]["max_candidates"] == 6
    assert config["tracklet_viterbi"]["max_candidates_per_track_id"] == 2
