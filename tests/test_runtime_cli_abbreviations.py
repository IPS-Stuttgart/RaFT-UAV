from __future__ import annotations

import os

import pytest

from raft_uav.runtime_cli_config import (
    apply_runtime_environment,
    parse_runtime_config,
    runtime_environment_names_from_argv,
)
from raft_uav.runtime_cli_patch import _runtime_aware_command


@pytest.mark.parametrize(
    ("argv_tail", "restored"),
    [
        (["--radar-range-std", "8.5"], ["--radar-range-std-m", "8.5"]),
        (["--radar-range-std=8.5"], ["--radar-range-std-m=8.5"]),
    ],
)
def test_abbreviated_runtime_passthrough_is_canonicalized(
    argv_tail: list[str],
    restored: list[str],
) -> None:
    argv = ["run-baseline", "/data/aerpaw", *argv_tail]

    config, remaining = parse_runtime_config(argv)

    assert config["radar_covariance"]["range_std_m"] == 8.5
    assert remaining == ["run-baseline", "/data/aerpaw", *restored]
    assert runtime_environment_names_from_argv(argv) == {
        "RAFT_UAV_RADAR_RANGE_STD_M"
    }


def test_abbreviated_runtime_flag_overrides_existing_environment(monkeypatch) -> None:
    monkeypatch.setenv("RAFT_UAV_RADAR_RANGE_STD_M", "5.0")
    argv = [
        "run-baseline",
        "/data/aerpaw",
        "--radar-range-std",
        "8.5",
    ]
    config, _ = parse_runtime_config(argv)

    apply_runtime_environment(
        config,
        overwrite_existing_env_names=runtime_environment_names_from_argv(argv),
    )

    assert os.environ["RAFT_UAV_RADAR_RANGE_STD_M"] == "8.5"


@pytest.mark.parametrize(
    "argv",
    [
        ["--radar-range-std", "8.5", "run-baseline", "/data/aerpaw"],
        ["--radar-range-std=8.5", "run-baseline", "/data/aerpaw"],
        ["--disable-tracklet-rf", "run-baseline", "/data/aerpaw"],
    ],
)
def test_leading_abbreviated_runtime_flag_keeps_runtime_dispatch(
    argv: list[str],
) -> None:
    assert _runtime_aware_command(argv) == "run-baseline"
