from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import raft_uav.mmuad.candidate_assignment_action_plan as action_plan
from raft_uav.mmuad.candidate_assignment_action_plan import (
    build_candidate_assignment_action_plan,
)


WEIGHT_FIELDS = (
    "duration_weight",
    "frame_weight",
    "error_weight",
    "regret_weight",
    "buried_weight",
)


@pytest.mark.parametrize("field", WEIGHT_FIELDS)
@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(False),
        np.array([1.0]),
        "not-a-weight",
    ],
)
def test_action_plan_rejects_invalid_weights_before_empty_return(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be a finite scalar"):
        build_candidate_assignment_action_plan(
            pd.DataFrame(),
            **{field: value},
        )


def test_action_plan_accepts_finite_numpy_and_negative_weights() -> None:
    blocks = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "assignment_failure_mode": [
                "good_candidate_buried",
                "smoothing_assignment_gap",
            ],
            "frame_count": [20, 10],
            "duration_s": [8.0, 4.0],
            "state_error_3d_m_max": [30.0, 15.0],
            "state_regret_m_p95": [20.0, 10.0],
            "oracle_in_topk_by_weight_rate": [0.2, 1.0],
            "dominant_matches_oracle_rate": [0.3, 0.9],
        }
    )

    actions, summary = build_candidate_assignment_action_plan(
        blocks,
        duration_weight=np.float64(1.0),
        frame_weight=np.array(1.0),
        error_weight=np.float32(1.0),
        regret_weight=-0.25,
        buried_weight=np.int64(1),
    )

    assert len(actions) == 2
    assert len(summary) == 2
    assert np.isfinite(actions["assignment_action_priority_score"]).all()


def test_action_plan_cli_resolves_validated_public_builder() -> None:
    assert (
        action_plan.main.__globals__["build_candidate_assignment_action_plan"]
        is build_candidate_assignment_action_plan
    )
