from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_uncertainty import (
    CandidateUncertaintyModel,
    candidate_uncertainty_training_summary,
)


def test_uncertainty_summary_excludes_nonfinite_truth_distances() -> None:
    model = CandidateUncertaintyModel(
        model_type="ridge",
        feature_columns=["confidence"],
        feature_means=[0.0],
        feature_scales=[1.0],
        source_values=[],
        target_transform="identity",
        sigma_min_m=1.0,
        sigma_max_m=30.0,
        fallback_sigma_m=10.0,
        weights=[1.0],
        bias=0.0,
    )
    features = pd.DataFrame(
        {
            "confidence": [2.0, 3.0, 4.0, 5.0],
            "truth_distance_3d_m": [2.0, 3.0, np.inf, -np.inf],
        }
    )

    summary = candidate_uncertainty_training_summary(features, model)

    assert summary["row_count"] == 2
    assert summary["mae_m"] == 0.0
    assert summary["rmse_m"] == 0.0
    assert summary["truth_mean_m"] == 2.5
    assert summary["predicted_mean_m"] == 2.5
    assert summary["correlation"] == pytest.approx(1.0)
    json.dumps(summary, allow_nan=False)
