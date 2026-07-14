from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_uncertainty import (
    CandidateUncertaintyModel,
    apply_candidate_uncertainty,
    load_candidate_uncertainty_model,
    predict_candidate_sigma,
    save_candidate_uncertainty_model,
    train_candidate_uncertainty,
)


def _valid_model() -> CandidateUncertaintyModel:
    return CandidateUncertaintyModel(
        model_type="ridge",
        feature_columns=["cluster_size"],
        feature_means=[0.0],
        feature_scales=[1.0],
        source_values=[],
        target_transform="identity",
        sigma_min_m=1.0,
        sigma_max_m=30.0,
        fallback_sigma_m=5.0,
        weights=[1.0],
        bias=0.0,
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"feature_means": []}, "feature_means must contain 1 finite values"),
        ({"feature_scales": [0.0]}, "feature_scales must contain positive"),
        ({"weights": [np.inf]}, "weights must contain 1 finite values"),
        ({"sigma_min_m": np.nan}, "sigma_min_m must be finite"),
        ({"fallback_sigma_m": 31.0}, "fallback_sigma_m must lie within"),
        ({"bias": np.inf}, "bias must be finite"),
    ],
)
def test_prediction_rejects_malformed_model_payloads(
    updates: dict[str, object],
    message: str,
) -> None:
    model = replace(_valid_model(), **updates)

    with pytest.raises(ValueError, match=message):
        predict_candidate_sigma(pd.DataFrame({"cluster_size": [2.0]}), model)


def test_prediction_rejects_missing_sklearn_payload() -> None:
    model = replace(_valid_model(), model_type="random-forest")

    with pytest.raises(ValueError, match="requires sklearn_estimator_base64"):
        predict_candidate_sigma(pd.DataFrame({"cluster_size": [2.0]}), model)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sigma_min_m": np.nan}, "sigma_min_m must be finite"),
        ({"sigma_max_m": np.inf}, "sigma_max_m must be finite"),
        ({"model_type": "ridge", "ridge_alpha": np.nan}, "ridge_alpha must be finite"),
        (
            {"model_type": "random-forest", "n_estimators": 1.5},
            "n_estimators must be a positive integer",
        ),
    ],
)
def test_training_rejects_invalid_controls_before_reading_features(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        train_candidate_uncertainty(pd.DataFrame(), **kwargs)


def test_load_rejects_malformed_saved_model(tmp_path: Path) -> None:
    payload = asdict(replace(_valid_model(), feature_scales=[0.0]))
    model_json = tmp_path / "model.json"
    model_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_scales must contain positive"):
        load_candidate_uncertainty_model(model_json)


def test_save_rejects_malformed_model_without_writing(tmp_path: Path) -> None:
    model_json = tmp_path / "model.json"

    with pytest.raises(ValueError, match="bias must be finite"):
        save_candidate_uncertainty_model(
            replace(_valid_model(), bias=np.inf),
            model_json,
        )

    assert not model_json.exists()


@pytest.mark.parametrize("z_scale", [0.0, np.nan, np.inf])
def test_covariance_replacement_rejects_invalid_z_scale(z_scale: float) -> None:
    with pytest.raises(ValueError, match="z_scale must be finite and positive"):
        apply_candidate_uncertainty(
            pd.DataFrame(),
            _valid_model(),
            replace_covariance=True,
            z_scale=z_scale,
        )
