from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_group_spatial_topk import (
    SpatialHypothesisGroupTopKConfig,
    select_spatial_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import HypothesisGroupConfig


@pytest.mark.parametrize("invalid_sigma", [-5.0, 0.0])
def test_spatial_group_topk_treats_nonpositive_sigma_as_missing(
    invalid_sigma: float,
) -> None:
    candidates = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "invalid-sigma",
                "candidate_branch": "raw",
                "mmuad_calibration_origin_row": 1,
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": invalid_sigma,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "livox_avia",
                "track_id": "valid-sigma",
                "candidate_branch": "raw",
                "mmuad_calibration_origin_row": 2,
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 2.0,
            },
        ]
    )

    selected, _ = select_spatial_hypothesis_group_topk(
        candidates,
        mixture_config=CandidateMixtureMapConfig(
            score_column="ranker_score",
            score_normalization="none",
            score_weight=0.0,
            sigma_column="predicted_sigma_m",
            default_sigma_m=10.0,
            sigma_min_m=1.0,
            sigma_max_m=30.0,
            sigma_log_weight=3.0,
        ),
        group_config=HypothesisGroupConfig(correction_strength=1.0),
        selection_config=SpatialHypothesisGroupTopKConfig(
            group_top_k=1,
            max_siblings_per_group=1,
            diversity_weight=0.0,
        ),
    )

    assert selected["track_id"].tolist() == ["valid-sigma"]
    assert selected["predicted_sigma_m"].tolist() == [2.0]
