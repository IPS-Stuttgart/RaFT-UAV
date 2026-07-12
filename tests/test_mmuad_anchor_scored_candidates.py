from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
    run_anchor_posterior_mass_group_topk_candidate_mixture_map,
)
from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_mixture_map_grouped import HypothesisGroupConfig


def test_run_result_keeps_rejected_anchor_scored_candidates() -> None:
    candidate_rows = []
    for time_s in range(3):
        candidate_rows.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"anchor-{time_s}",
                    "origin_row": f"anchor-origin-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.0,
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "livox_avia",
                    "track_id": f"rejected-{time_s}",
                    "origin_row": f"rejected-origin-{time_s}",
                    "candidate_branch": "translated",
                    "x_m": float(time_s + 10),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 1.0,
                    "predicted_sigma_m": 1.0,
                },
            ]
        )
    candidates = pd.DataFrame.from_records(candidate_rows)
    anchors = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [1.0, 1.0, 1.0],
        }
    )

    result = run_anchor_posterior_mass_group_topk_candidate_mixture_map(
        candidates,
        initial_estimates=anchors,
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
            anchor_cost_cap=4.0,
            anchor_time_tolerance_s=0.01,
        ),
    )

    assert len(result.scored_candidates) == len(candidates)
    assert len(result.selected_candidates) == 3
    assert result.selected_candidates["track_id"].astype(str).str.startswith("anchor-").all()
    rejected = result.scored_candidates[
        result.scored_candidates["track_id"].astype(str).str.startswith("rejected-")
    ]
    assert len(rejected) == 3
    assert rejected["mixture_anchor_matched"].all()
    assert (rejected["mixture_anchor_distance_m"] == 10.0).all()
    assert "mixture_anchor_conditioned_selection_utility" in result.scored_candidates
