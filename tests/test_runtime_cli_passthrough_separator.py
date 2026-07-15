import pytest

from raft_uav.runtime_cli_config import parse_runtime_config


@pytest.mark.parametrize(
    "passthrough",
    [
        ["--radar-range-std-m", "8.5"],
        ["--radar-range-std-m=8.5"],
    ],
)
def test_parse_runtime_config_restores_passthrough_before_argument_separator(
    passthrough: list[str],
):
    config, remaining = parse_runtime_config(
        [
            "run-baseline",
            "/data/aerpaw",
            *passthrough,
            "--",
            "--literal-dataset-fragment",
        ]
    )

    assert remaining == [
        "run-baseline",
        "/data/aerpaw",
        *passthrough,
        "--",
        "--literal-dataset-fragment",
    ]
    assert config["radar_covariance"]["range_std_m"] == 8.5
