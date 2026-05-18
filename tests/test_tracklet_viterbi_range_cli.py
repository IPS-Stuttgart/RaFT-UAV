from __future__ import annotations

import os

from raft_uav import tracklet_viterbi_cli
from raft_uav import tracklet_viterbi_range_cli


def test_range_cli_extracts_range_args_and_forwards_base_args() -> None:
    forwarded, overrides = tracklet_viterbi_range_cli._extract_range_args(
        [
            "--tracklet-radar-range-xy-scale",
            "0.041",
            "--tracklet-radar-range-z-scale",
            "0.063",
            "run-baseline",
            "dataset",
            "--flight",
            "Opt1",
        ]
    )

    assert forwarded == ["run-baseline", "dataset", "--flight", "Opt1"]
    assert overrides["radar_range_xy_scale"] == 0.041
    assert overrides["radar_range_z_scale"] == 0.063


def test_range_cli_overlays_tracklet_config(monkeypatch) -> None:
    monkeypatch.delenv(tracklet_viterbi_range_cli._RANGE_VARIANT_ENV, raising=False)

    with tracklet_viterbi_range_cli._temporary_range_configuration(
        {
            "use_range_adaptive_radar_covariance": False,
            "radar_range_xy_floor_std_m": 21.0,
            "radar_range_z_floor_std_m": 33.0,
            "radar_range_xy_scale": 0.044,
            "radar_range_z_scale": 0.066,
        }
    ):
        config = tracklet_viterbi_cli._tracklet_config_from_environment()
        assert os.environ[tracklet_viterbi_range_cli._RANGE_VARIANT_ENV] == "range-covariance"
        assert config.use_range_adaptive_radar_covariance is False
        assert config.radar_range_xy_floor_std_m == 21.0
        assert config.radar_range_z_floor_std_m == 33.0
        assert config.radar_range_xy_scale == 0.044
        assert config.radar_range_z_scale == 0.066

    assert tracklet_viterbi_range_cli._RANGE_VARIANT_ENV not in os.environ


def test_range_cli_preserves_existing_variant_env(monkeypatch) -> None:
    monkeypatch.setenv(tracklet_viterbi_range_cli._RANGE_VARIANT_ENV, "retention")

    with tracklet_viterbi_range_cli._temporary_range_configuration({}):
        assert os.environ[tracklet_viterbi_range_cli._RANGE_VARIANT_ENV] == "range-covariance"

    assert os.environ[tracklet_viterbi_range_cli._RANGE_VARIANT_ENV] == "retention"


def test_range_cli_main_forwards_to_tracklet_cli(monkeypatch) -> None:
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        config = tracklet_viterbi_cli._tracklet_config_from_environment()
        seen["xy_scale"] = config.radar_range_xy_scale
        seen["variant"] = os.environ[tracklet_viterbi_range_cli._RANGE_VARIANT_ENV]
        return 0

    monkeypatch.setattr(tracklet_viterbi_cli, "main", fake_main)

    status = tracklet_viterbi_range_cli.main(
        [
            "--tracklet-radar-range-xy-scale",
            "0.039",
            "run-baseline",
            "dataset",
        ]
    )

    assert status == 0
    assert seen["argv"] == ["run-baseline", "dataset"]
    assert seen["xy_scale"] == 0.039
    assert seen["variant"] == "range-covariance"
