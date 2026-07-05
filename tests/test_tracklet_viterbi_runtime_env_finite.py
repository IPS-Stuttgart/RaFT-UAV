from __future__ import annotations

import pytest

from raft_uav.baselines.tracklet_viterbi_runtime import _config_from_environment


def test_tracklet_runtime_rejects_nan_env_float(monkeypatch) -> None:
    monkeypatch.setenv("RAFT_UAV_TRACKLET_SUPPORT_WEIGHT", "nan")

    with pytest.raises(ValueError, match="RAFT_UAV_TRACKLET_SUPPORT_WEIGHT must be finite"):
        _config_from_environment()
