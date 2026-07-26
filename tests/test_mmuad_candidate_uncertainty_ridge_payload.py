from __future__ import annotations

import base64
import pickle

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_uncertainty import (
    CandidateUncertaintyModel,
    predict_candidate_sigma,
)


class _ConstantEstimator:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), 7.0, dtype=float)


def _ridge_model_with_sklearn_payload() -> CandidateUncertaintyModel:
    payload = base64.b64encode(pickle.dumps(_ConstantEstimator())).decode("ascii")
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
        sklearn_estimator_base64=payload,
    )


def test_prediction_rejects_sklearn_payload_on_ridge_model() -> None:
    model = _ridge_model_with_sklearn_payload()

    with pytest.raises(
        ValueError,
        match="ridge uncertainty model must not define sklearn_estimator_base64",
    ):
        predict_candidate_sigma(pd.DataFrame({"cluster_size": [2.0]}), model)
