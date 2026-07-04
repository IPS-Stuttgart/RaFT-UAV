import pytest

from raft_uav.tracklet_viterbi_fixed_lag_safe_cli import _validated_fixed_lag_s_from_env


def test_fixed_lag_env_rejects_nonfinite_values(monkeypatch):
    for value in ["nan", "inf", "-inf"]:
        monkeypatch.setenv("RAFT_UAV_TRACKLET_VITERBI_LAG_S", value)
        with pytest.raises(ValueError, match="finite and positive"):
            _validated_fixed_lag_s_from_env()


def test_fixed_lag_env_accepts_positive_finite_value(monkeypatch):
    monkeypatch.setenv("RAFT_UAV_TRACKLET_VITERBI_LAG_S", "3.5")

    assert _validated_fixed_lag_s_from_env() == 3.5
