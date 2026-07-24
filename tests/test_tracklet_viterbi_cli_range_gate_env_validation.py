from __future__ import annotations

import pytest

from raft_uav import tracklet_viterbi_cli


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
def test_canonical_tracklet_cli_rejects_invalid_range_gate_env(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("RAFT_UAV_TRACKLET_RANGE_GATE_M", value)

    with pytest.raises(
        ValueError,
        match="RAFT_UAV_TRACKLET_RANGE_GATE_M must be finite and nonnegative",
    ):
        tracklet_viterbi_cli._tracklet_config_from_environment()


def test_canonical_tracklet_cli_keeps_zero_as_disabled_range_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAFT_UAV_TRACKLET_RANGE_GATE_M", "0")

    config = tracklet_viterbi_cli._tracklet_config_from_environment()

    assert config.range_gate_m is None


def test_canonical_tracklet_cli_preserves_positive_range_gate_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAFT_UAV_TRACKLET_RANGE_GATE_M", "725.5")

    config = tracklet_viterbi_cli._tracklet_config_from_environment()

    assert config.range_gate_m == pytest.approx(725.5)
