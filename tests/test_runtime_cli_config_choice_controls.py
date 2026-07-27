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
    ("field", "value", "message"),
    [
        ("radar_covariance_mode", "bogus", "must be one of"),
        ("disable_tracklet_rf_anchor", "false", "Boolean scalar"),
        ("disable_tracklet_rf_anchor", 0, "Boolean scalar"),
        ("disable_tracklet_rf_anchor", np.array([False]), "Boolean scalar"),
    ],
)
def test_runtime_config_rejects_malformed_choice_controls(
    field: str,
    value: object,
    message: str,
) -> None:
    args = _default_args()
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        runtime_config_from_args(args)


def test_runtime_config_normalizes_mode_and_accepts_numpy_boolean() -> None:
    args = _default_args()
    args.radar_covariance_mode = " FIXED "
    args.disable_tracklet_rf_anchor = np.bool_(True)

    config = runtime_config_from_args(args)

    assert config["radar_covariance"]["mode"] == "fixed"
    assert config["tracklet_viterbi"]["use_rf_anchor"] is False
