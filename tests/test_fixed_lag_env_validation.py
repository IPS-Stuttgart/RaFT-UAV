import pytest

from raft_uav.tracklet_viterbi_fixed_lag_cli import _fixed_lag_s_from_env
from raft_uav.tracklet_viterbi_fixed_lag_safe_cli import (
    _validated_fixed_lag_s_from_env,
    main as fixed_lag_safe_main,
)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_fixed_lag_env_rejects_nonfinite_values(monkeypatch, value):
    monkeypatch.setenv("RAFT_UAV_TRACKLET_VITERBI_LAG_S", value)
    with pytest.raises(ValueError, match="finite and positive"):
        _validated_fixed_lag_s_from_env()


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_fixed_lag_core_env_rejects_nonfinite_values(monkeypatch, value):
    monkeypatch.setenv("RAFT_UAV_TRACKLET_VITERBI_LAG_S", value)
    with pytest.raises(ValueError, match="finite and positive"):
        _fixed_lag_s_from_env()


def test_fixed_lag_env_accepts_positive_finite_value(monkeypatch):
    monkeypatch.setenv("RAFT_UAV_TRACKLET_VITERBI_LAG_S", "3.5")

    assert _validated_fixed_lag_s_from_env() == 3.5
    assert _fixed_lag_s_from_env() == 3.5


def test_fixed_lag_safe_cli_help_defers_lag_validation(monkeypatch, capsys):
    monkeypatch.setenv("RAFT_UAV_TRACKLET_VITERBI_LAG_S", "nan")

    with pytest.raises(SystemExit) as exc_info:
        fixed_lag_safe_main(["--help"])

    assert exc_info.value.code == 0
    assert "run-baseline" in capsys.readouterr().out
