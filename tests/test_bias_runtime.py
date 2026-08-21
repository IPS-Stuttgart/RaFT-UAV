from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.bias_model_cli import _extract_bias_model
from raft_uav.calibration.bias import BiasCorrectionBank, SensorBiasCorrectionModel
from raft_uav.calibration import bias_runtime
from raft_uav.calibration.bias_runtime import BIAS_MODEL_ENV


def _constant_rf_model(intercept: tuple[float, float] = (10.0, -5.0)) -> SensorBiasCorrectionModel:
    return SensorBiasCorrectionModel(
        source="rf",
        target_columns=("east_m", "north_m"),
        feature_columns=(),
        intercept=np.asarray(intercept, dtype=float),
        coefficients=np.empty((0, 2), dtype=float),
        feature_mean=np.empty(0, dtype=float),
        feature_scale=np.empty(0, dtype=float),
        residual_std=np.zeros(2, dtype=float),
        training_rows=12,
        ridge_alpha=0.0,
        time_gate_s=2.0,
    )


def _reset_runtime_bias_cache(monkeypatch) -> None:
    monkeypatch.setattr(bias_runtime, "_CACHED_MODEL_PATH", None)
    monkeypatch.setattr(bias_runtime, "_CACHED_MODEL_SIGNATURE", None, raising=False)
    monkeypatch.setattr(bias_runtime, "_CACHED_BANK", None)


def test_runtime_bias_correction_applies_configured_model(tmp_path, monkeypatch):
    path = tmp_path / "bias_model.json"
    BiasCorrectionBank({"rf": _constant_rf_model()}).save(path)
    monkeypatch.setenv(BIAS_MODEL_ENV, str(path))
    _reset_runtime_bias_cache(monkeypatch)
    frame = pd.DataFrame({"east_m": [12.0], "north_m": [3.0], "std_m": [75.0]})

    corrected = bias_runtime._apply_runtime_bias(frame, "rf")

    assert corrected["east_m"].tolist() == [2.0]
    assert corrected["north_m"].tolist() == [8.0]
    assert corrected["raw_east_m"].tolist() == [12.0]
    assert corrected["raw_north_m"].tolist() == [3.0]
    assert corrected["bias_model_path"].tolist() == [str(path)]


def test_runtime_bias_cache_reloads_model_replaced_at_same_path(tmp_path, monkeypatch):
    path = tmp_path / "bias_model.json"
    BiasCorrectionBank({"rf": _constant_rf_model()}).save(path)
    monkeypatch.setenv(BIAS_MODEL_ENV, str(path))
    _reset_runtime_bias_cache(monkeypatch)
    frame = pd.DataFrame({"east_m": [12.0], "north_m": [3.0]})

    first = bias_runtime._apply_runtime_bias(frame, "rf")
    replacement = tmp_path / "replacement.json"
    BiasCorrectionBank({"rf": _constant_rf_model((1.0, 2.0))}).save(replacement)
    replacement.replace(path)
    second = bias_runtime._apply_runtime_bias(frame, "rf")

    assert first[["east_m", "north_m"]].to_numpy().tolist() == [[2.0, 8.0]]
    assert second[["east_m", "north_m"]].to_numpy().tolist() == [[11.0, 1.0]]


def test_runtime_bias_cache_does_not_mask_deleted_model(tmp_path, monkeypatch):
    path = tmp_path / "bias_model.json"
    BiasCorrectionBank({"rf": _constant_rf_model()}).save(path)
    monkeypatch.setenv(BIAS_MODEL_ENV, str(path))
    _reset_runtime_bias_cache(monkeypatch)
    frame = pd.DataFrame({"east_m": [12.0], "north_m": [3.0]})

    bias_runtime._apply_runtime_bias(frame, "rf")
    path.unlink()

    with pytest.raises(FileNotFoundError):
        bias_runtime._apply_runtime_bias(frame, "rf")


def test_runtime_bias_correction_is_noop_without_model(monkeypatch):
    monkeypatch.delenv(BIAS_MODEL_ENV, raising=False)
    frame = pd.DataFrame({"east_m": [1.0], "north_m": [2.0]})

    corrected = bias_runtime._apply_runtime_bias(frame, "rf")

    assert corrected is frame


def test_bias_wrapper_extracts_model_path_and_preserves_remaining_args():
    path, remaining = _extract_bias_model(
        ["--bias-model", "model.json", "run-baseline", "data", "--flight", "Opt1"]
    )

    assert path == Path("model.json")
    assert remaining == ["run-baseline", "data", "--flight", "Opt1"]


def test_bias_wrapper_supports_equals_form():
    path, remaining = _extract_bias_model(
        ["--bias-model=model.json", "run-baseline", "data", "--flight", "Opt1"]
    )

    assert path == Path("model.json")
    assert remaining == ["run-baseline", "data", "--flight", "Opt1"]


def test_bias_wrapper_rejects_empty_model_path():
    for argv in (["--bias-model", ""], ["--bias-model=   "]):
        with pytest.raises(SystemExit, match="non-empty path"):
            _extract_bias_model(argv)
