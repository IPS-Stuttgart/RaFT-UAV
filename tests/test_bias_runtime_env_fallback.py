from pathlib import Path

from raft_uav.calibration import bias_runtime
from raft_uav.calibration.bias_runtime import BIAS_MODEL_ENV, BIAS_MODEL_PATH_ENV


def test_bias_model_path_falls_back_when_primary_env_is_whitespace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fallback = tmp_path / "fallback-bias-model.json"
    monkeypatch.setenv(BIAS_MODEL_ENV, " \t ")
    monkeypatch.setenv(BIAS_MODEL_PATH_ENV, str(fallback))

    assert bias_runtime.configured_bias_model_path() == fallback
    assert bias_runtime.bias_correction_enabled()


def test_bias_model_path_prefers_nonempty_primary_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary = tmp_path / "primary-bias-model.json"
    fallback = tmp_path / "fallback-bias-model.json"
    monkeypatch.setenv(BIAS_MODEL_ENV, str(primary))
    monkeypatch.setenv(BIAS_MODEL_PATH_ENV, str(fallback))

    assert bias_runtime.configured_bias_model_path() == primary
