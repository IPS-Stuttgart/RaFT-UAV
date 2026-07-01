from __future__ import annotations

import pytest

from raft_uav.baselines.tracklet_viterbi_runtime import _config_from_environment


def test_tracklet_runtime_rejects_negative_range_gate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAFT_UAV_TRACKLET_RANGE_GATE_M", "-1")

    with pytest.raises(ValueError, match="RAFT_UAV_TRACKLET_RANGE_GATE_M"):
        _config_from_environment()
