import argparse
import os

import pytest

from raft_uav.runtime_cli_config import (
    add_runtime_configuration_arguments,
    apply_runtime_environment,
    parse_runtime_config,
    runtime_config_from_args,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_runtime_configuration_arguments(parser)
    return parser.parse_args(argv)


def test_runtime_config_from_args_records_tracklet_and_radar_settings():
    args = _parse_args(
        [
            "--radar-covariance-mode",
            "fixed",
            "--radar-range-std-m",
            "8",
            "--radar-origin-east-m",
            "12.5",
            "--tracklet-track-switch-cost",
            "14",
            "--tracklet-anchor-nis-weight",
            "0.6",
            "--tracklet-range-gate-m",
            "0",
            "--disable-tracklet-rf-anchor",
        ]
    )

    config = runtime_config_from_args(args)

    assert config["radar_covariance"]["mode"] == "fixed"
    assert config["radar_covariance"]["range_std_m"] == 8.0
    assert config["radar_covariance"]["origin_east_m"] == 12.5
    assert config["tracklet_viterbi"]["track_switch_cost"] == 14.0
    assert config["tracklet_viterbi"]["anchor_nis_weight"] == 0.6
    assert config["tracklet_viterbi"]["range_gate_m"] is None
    assert config["tracklet_viterbi"]["use_rf_anchor"] is False


def test_parse_runtime_config_preserves_standard_cli_arguments():
    config, remaining = parse_runtime_config(
        [
            "run-baseline",
            "/data/aerpaw",
            "--flight",
            "Opt1",
            "--tracklet-track-switch-cost",
            "16",
            "--radar-covariance-mode",
            "fixed",
            "--smoother",
            "fixed-lag",
        ]
    )

    assert remaining == [
        "run-baseline",
        "/data/aerpaw",
        "--flight",
        "Opt1",
        "--smoother",
        "fixed-lag",
    ]
    assert config["tracklet_viterbi"]["track_switch_cost"] == 16.0
    assert config["radar_covariance"]["mode"] == "fixed"


def test_apply_runtime_environment_sets_expected_variables(monkeypatch):
    args = _parse_args(
        [
            "--radar-covariance-mode",
            "range-angle",
            "--radar-azimuth-std-deg",
            "3",
            "--tracklet-max-candidates",
            "12",
            "--tracklet-track-switch-cost",
            "16",
        ]
    )
    config = runtime_config_from_args(args)

    apply_runtime_environment(config)

    assert os.environ["RAFT_UAV_RADAR_COVARIANCE_MODE"] == "range-angle"
    assert os.environ["RAFT_UAV_RADAR_AZIMUTH_STD_DEG"] == "3.0"
    assert os.environ["RAFT_UAV_TRACKLET_MAX_CANDIDATES"] == "12"
    assert os.environ["RAFT_UAV_TRACKLET_TRACK_SWITCH_COST"] == "16.0"


def test_runtime_config_rejects_invalid_radar_covariance_range():
    args = _parse_args(
        [
            "--radar-covariance-min-std-m",
            "10",
            "--radar-covariance-max-std-m",
            "5",
        ]
    )

    with pytest.raises(ValueError, match="max_std"):
        runtime_config_from_args(args)
