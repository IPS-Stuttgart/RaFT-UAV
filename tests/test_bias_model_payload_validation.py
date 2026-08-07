from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from raft_uav.calibration.bias import (
    RF_TARGET_COLUMNS,
    SensorBiasCorrectionModel,
    load_bias_correction_models,
)


def _model_payload() -> dict[str, object]:
    return {
        "version": 1,
        "source": "rf",
        "target_columns": list(RF_TARGET_COLUMNS),
        "feature_columns": ["time_s"],
        "intercept": [0.0, 0.0],
        "coefficients": [[0.0, 0.0]],
        "feature_mean": [0.0],
        "feature_scale": [1.0],
        "residual_std": [1.0, 1.0],
        "training_rows": 8,
        "ridge_alpha": 0.0,
        "time_gate_s": 2.0,
    }


def _write_bundle(
    tmp_path: Path,
    model: dict[str, object],
    *,
    source: str = "rf",
    version: object = 1,
) -> Path:
    path = tmp_path / "bias.json"
    path.write_text(
        json.dumps(
            {"version": version, "models": {source: model}},
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("intercept", [float("nan"), 0.0]),
        ("intercept", [True, 0.0]),
        ("coefficients", [[0.0, float("inf")]]),
        ("feature_mean", [float("nan")]),
        ("feature_scale", [-1.0]),
        ("feature_scale", [0.0]),
        ("residual_std", [1.0, float("nan")]),
        ("training_rows", 1.5),
        ("ridge_alpha", True),
        ("time_gate_s", float("inf")),
    ],
)
def test_model_payload_rejects_malformed_state(field: str, value: object) -> None:
    payload = _model_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        SensorBiasCorrectionModel.from_dict(payload)


def test_model_payload_rejects_fractional_version() -> None:
    payload = _model_payload()
    payload["version"] = 1.5

    with pytest.raises(ValueError, match="version"):
        SensorBiasCorrectionModel.from_dict(payload)


def test_direct_constructor_rejects_fractional_training_rows() -> None:
    with pytest.raises(ValueError, match="training_rows"):
        SensorBiasCorrectionModel(
            source="rf",
            target_columns=RF_TARGET_COLUMNS,
            feature_columns=("time_s",),
            intercept=np.zeros(2),
            coefficients=np.zeros((1, 2)),
            feature_mean=np.zeros(1),
            feature_scale=np.ones(1),
            residual_std=np.ones(2),
            training_rows=3.5,
            ridge_alpha=0.0,
            time_gate_s=2.0,
        )


def test_direct_constructor_rejects_zero_feature_scale() -> None:
    with pytest.raises(ValueError, match="feature_scale"):
        SensorBiasCorrectionModel(
            source="rf",
            target_columns=RF_TARGET_COLUMNS,
            feature_columns=("time_s",),
            intercept=np.zeros(2),
            coefficients=np.zeros((1, 2)),
            feature_mean=np.zeros(1),
            feature_scale=np.zeros(1),
            residual_std=np.ones(2),
            training_rows=8,
            ridge_alpha=0.0,
            time_gate_s=2.0,
        )


def test_bundle_rejects_fractional_version(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bundle version"):
        load_bias_correction_models(
            _write_bundle(tmp_path, _model_payload(), version=1.5)
        )


def test_bundle_rejects_source_mismatch(tmp_path: Path) -> None:
    payload = _model_payload()
    payload["source"] = "radar"
    payload["target_columns"] = ["east_m", "north_m", "up_m"]
    payload["intercept"] = [0.0, 0.0, 0.0]
    payload["coefficients"] = [[0.0, 0.0, 0.0]]
    payload["residual_std"] = [1.0, 1.0, 1.0]

    with pytest.raises(ValueError, match="does not match model source"):
        load_bias_correction_models(_write_bundle(tmp_path, payload))


def test_bundle_rejects_wrong_targets_for_known_source(tmp_path: Path) -> None:
    payload = deepcopy(_model_payload())
    payload["target_columns"] = ["north_m", "east_m"]

    with pytest.raises(ValueError, match="must target"):
        load_bias_correction_models(_write_bundle(tmp_path, payload))


def test_bundle_loads_valid_numeric_string_controls(tmp_path: Path) -> None:
    payload = _model_payload()
    payload["version"] = "1"
    payload["training_rows"] = "8"
    payload["ridge_alpha"] = "0.0"
    payload["time_gate_s"] = "2.0"

    models = load_bias_correction_models(_write_bundle(tmp_path, payload))

    assert models["rf"].source == "rf"
    assert models["rf"].training_rows == 8
    assert models["rf"].ridge_alpha == 0.0
    assert models["rf"].time_gate_s == 2.0
