from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
)
from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
)
from raft_uav.mmuad.candidate_mixture_group_multi_anchor_coverage import (
    COVERAGE_RESCUE_ANCHORS,
    COVERAGE_RESCUED,
    AnchorGroupCoverageConfig,
    select_multi_anchor_coverage_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import HypothesisGroupConfig


def _anchor(offset_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seqA"],
            "Timestamp": [0.0],
            "x_m": [offset_m],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    )


def test_shared_rescue_group_keeps_all_anchor_provenance() -> None:
    candidates = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "left",
                "origin_row": "left-origin",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "livox_avia",
                "track_id": "right",
                "origin_row": "right-origin",
                "candidate_branch": "translated",
                "x_m": 10.0,
                "y_m": 0.0,
                "z_m": 1.0,
                "ranker_score": 0.0,
                "predicted_sigma_m": 1.0,
            },
        ]
    )

    _, selected, _, frames, _ = select_multi_anchor_coverage_hypothesis_group_topk(
        candidates,
        anchor_estimates={
            "left": _anchor(0.0),
            "right_a": _anchor(10.0),
            "right_b": _anchor(10.0),
        },
        mixture_config=CandidateMixtureMapConfig(
            top_k=0,
            score_column="ranker_score",
            fallback_score_columns=(),
            sigma_column="predicted_sigma_m",
            score_normalization="minmax",
            score_weight=1.0,
            temperature=1.0,
            sigma_log_weight=0.0,
            smoothness_weight=0.0,
            iterations=1,
        ),
        group_config=HypothesisGroupConfig(group_column="origin_row"),
        selection_config=PosteriorMassGroupTopKConfig(
            min_group_top_k=1,
            max_group_top_k=1,
            target_posterior_mass=0.95,
            posterior_temperature=1.0,
            uniform_posterior_blend=0.0,
            max_siblings_per_group=1,
            group_score_mode="max",
            diversity_weight=0.0,
            diversity_scale_m=5.0,
            diversity_cap_m=30.0,
        ),
        anchor_config=AnchorConditioningConfig(
            anchor_selection_weight=5.0,
            anchor_scale_m=1.0,
            anchor_huber_delta=1.0,
            anchor_cost_cap=10.0,
            anchor_time_tolerance_s=0.01,
        ),
        coverage_config=AnchorGroupCoverageConfig(
            max_anchor_distance_m=0.1,
            max_extra_groups_per_frame=1,
            max_siblings_per_rescued_group=1,
        ),
    )

    rescued = selected.loc[selected[COVERAGE_RESCUED]].iloc[0]
    assert rescued["track_id"] == "right"
    assert rescued[COVERAGE_RESCUE_ANCHORS] == "right_a;right_b"
    assert frames.loc[0, "covered_anchors_before"] == 1
    assert frames.loc[0, "covered_anchors_by_rescue"] == 2
