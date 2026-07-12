from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk import (
    AnchorConditioningConfig,
)
from raft_uav.mmuad.candidate_mixture_group_multi_anchor_mass_topk import (
    MULTI_ANCHOR_UTILITY_COLUMN,
    add_multi_anchor_conditioned_selection_utility,
)
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sequence_id",
            "time_s",
            "source",
            "track_id",
            "origin_row",
            "candidate_branch",
            "x_m",
            "y_m",
            "z_m",
            "ranker_score",
            "predicted_sigma_m",
        ]
    )


def _anchor(offset_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "state_x_m": [offset_m, offset_m + 1.0],
            "state_y_m": [0.0, 0.0],
            "state_z_m": [1.0, 1.0],
        }
    )


def test_multi_anchor_scoring_accepts_empty_candidate_table() -> None:
    scored, normalized_anchors, summary = (
        add_multi_anchor_conditioned_selection_utility(
            _empty_candidates(),
            {
                "left": _anchor(0.0),
                "right": _anchor(10.0),
            },
            mixture_config=CandidateMixtureMapConfig(
                score_column="ranker_score",
                fallback_score_columns=(),
                sigma_column="predicted_sigma_m",
            ),
            anchor_config=AnchorConditioningConfig(
                anchor_time_tolerance_s=0.01,
            ),
        )
    )

    assert scored.empty
    assert MULTI_ANCHOR_UTILITY_COLUMN in scored.columns
    assert "mixture_multi_anchor_left_cost" in scored.columns
    assert "mixture_multi_anchor_right_cost" in scored.columns
    assert set(normalized_anchors["anchor_name"]) == {"left", "right"}
    assert summary["candidate_rows"] == 0
    assert summary["frame_count"] == 0
    assert summary["anchor_count"] == 2
