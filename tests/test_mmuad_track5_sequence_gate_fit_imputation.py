from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import raft_uav.mmuad.track5_sequence_gate_fit as sequence_gate_fit


class _FirstFeatureRegressor:
    def __init__(self) -> None:
        self.fit_matrix: np.ndarray | None = None
        self.predict_matrix: np.ndarray | None = None

    def fit(self, features: np.ndarray, target: Any) -> "_FirstFeatureRegressor":
        del target
        self.fit_matrix = np.asarray(features, dtype=float).copy()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.predict_matrix = np.asarray(features, dtype=float).copy()
        return self.predict_matrix[:, 0]


def _model_factory(models: list[_FirstFeatureRegressor]):
    def make_model(model_name: str, *, random_state: int) -> _FirstFeatureRegressor:
        del model_name, random_state
        model = _FirstFeatureRegressor()
        models.append(model)
        return model

    return make_model


def test_apply_prediction_uses_training_feature_imputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models: list[_FirstFeatureRegressor] = []
    monkeypatch.setattr(
        sequence_gate_fit,
        "_make_model",
        _model_factory(models),
    )
    train_rows = pd.DataFrame(
        {
            "sequence_id": ["train-a", "train-b"],
            "feature": [10.0, 20.0],
            "oracle_weight": [0.1, 0.2],
        }
    )
    apply_rows = pd.DataFrame(
        {
            "sequence_id": ["apply-missing", "apply-outlier"],
            "feature": [np.nan, 1000.0],
        }
    )

    weights = sequence_gate_fit._predict_apply_weights(
        "ridge",
        train_rows,
        apply_rows,
        ["feature"],
        random_state=7,
        min_weight=0.0,
        max_weight=2000.0,
    ).set_index("sequence_id")

    assert weights.loc["apply-missing", "blend_weight"] == pytest.approx(15.0)
    assert weights.loc["apply-outlier", "blend_weight"] == pytest.approx(1000.0)
    assert models[0].fit_matrix[:, 0].tolist() == [10.0, 20.0]
    assert models[0].predict_matrix[:, 0].tolist() == [15.0, 1000.0]


def test_loso_prediction_uses_training_fold_imputation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models: list[_FirstFeatureRegressor] = []
    monkeypatch.setattr(
        sequence_gate_fit,
        "_make_model",
        _model_factory(models),
    )
    rows = pd.DataFrame(
        {
            "sequence_id": ["a", "b", "missing"],
            "feature": [1.0, 3.0, np.nan],
            "oracle_weight": [0.1, 0.2, 0.3],
        }
    )

    weights = sequence_gate_fit._predict_loso_weights(
        "ridge",
        rows,
        ["feature"],
        random_state=7,
        min_weight=0.0,
        max_weight=10.0,
    ).set_index("sequence_id")

    assert weights.loc["missing", "blend_weight"] == pytest.approx(2.0)
    assert models[2].fit_matrix[:, 0].tolist() == [1.0, 3.0]
    assert models[2].predict_matrix[:, 0].tolist() == [2.0]


def test_loso_prediction_rejects_single_sequence_oracle_leakage() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["only"],
            "feature": [np.nan],
            "oracle_weight": [0.37],
        }
    )

    with pytest.raises(ValueError, match="at least two sequences"):
        sequence_gate_fit._predict_loso_weights(
            "ridge",
            rows,
            ["feature"],
            random_state=7,
            min_weight=0.0,
            max_weight=1.0,
        )


def test_public_main_forwards_the_pandas_compatibility_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    proxy = object()
    original_legacy_pandas = sequence_gate_fit._IMPL.pd

    def fake_main(argv: list[str] | None) -> int:
        observed["argv"] = argv
        observed["legacy_pandas"] = sequence_gate_fit._IMPL.pd
        return 23

    monkeypatch.setattr(sequence_gate_fit, "_ORIGINAL_MAIN", fake_main)
    monkeypatch.setattr(sequence_gate_fit, "pd", proxy)

    assert sequence_gate_fit.main(["--example"]) == 23
    assert observed["argv"] == ["--example"]
    assert observed["legacy_pandas"] is proxy
    assert sequence_gate_fit._IMPL.pd is original_legacy_pandas
