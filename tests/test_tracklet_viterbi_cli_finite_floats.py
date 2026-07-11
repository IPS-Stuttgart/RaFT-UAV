from __future__ import annotations

import pytest

from raft_uav import tracklet_viterbi_cli


@pytest.mark.parametrize(
    "option",
    [
        "--tracklet-viterbi-lag-s",
        "--tracklet-soft-path-temperature",
    ],
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_tracklet_cli_rejects_nonfinite_positive_float_options(
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        tracklet_viterbi_cli._extract_tracklet_args([option, value])


@pytest.mark.parametrize(
    "option",
    [
        "--tracklet-learned-unary-weight",
        "--tracklet-hand-unary-weight",
    ],
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_tracklet_cli_rejects_nonfinite_nonnegative_float_options(
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        tracklet_viterbi_cli._extract_tracklet_args([option, value])


def test_tracklet_cli_accepts_finite_float_options() -> None:
    remaining, updates = tracklet_viterbi_cli._extract_tracklet_args(
        [
            "run-baseline",
            "dataset",
            "--tracklet-viterbi-lag-s",
            "2.5",
            "--tracklet-learned-unary-weight",
            "0",
        ]
    )

    assert remaining == ["run-baseline", "dataset"]
    assert updates["RAFT_UAV_TRACKLET_VITERBI_LAG_S"] == "2.5"
    assert updates["RAFT_UAV_TRACKLET_LEARNED_UNARY_WEIGHT"] == "0.0"
