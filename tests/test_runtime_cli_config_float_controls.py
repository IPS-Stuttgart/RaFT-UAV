from __future__ import annotations

import argparse

import numpy as np
import pytest

from raft_uav.runtime_cli_config import add_runtime_configuration_arguments
from raft_uav.runtime_cli_config import runtime_config_from_args


def _default_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_configuration_arguments(parser)
    return parser.parse_args([])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("radar_xy_std_m", True),
        ("radar_z_std_m", np.bool_(False)),
        ("radar_range_std_m", np.array([5.0])),
        ("radar_azimuth_std_deg", np.ma.array(2.0, mask=True)),
        ("radar_elevation_std_deg", 2.0 + 0.0j),
        ("radar_origin_east_m", np.array([1.0])),
        ("tracklet_catprob_weight", False),
        ("tracklet_range_gate_m", np.complex128(1.0 + 0.0j)),
    ],
)
def test_runtime_config_rejects_malformed_float_controls(
    field: str,
    value: object,
) -> None:
    args = _default_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match="finite real scalar"):
        runtime_config_from_args(args)


def test_runtime_config_accepts_real_scalar_like_float_controls() -> None:
    args = _default_args()
    args.radar_xy_std_m = np.array(30.0)
    args.radar_origin_east_m = np.float32(12.5)
    args.tracklet_catprob_weight = "0.75"
    args.tracklet_range_gate_m = np.ma.array(0.0, mask=False)

    config = runtime_config_from_args(args)

    assert config["radar_covariance"]["xy_std_m"] == 30.0
    assert config["radar_covariance"]["origin_east_m"] == 12.5
    assert config["tracklet_viterbi"]["catprob_weight"] == 0.75
    assert config["tracklet_viterbi"]["range_gate_m"] is None
