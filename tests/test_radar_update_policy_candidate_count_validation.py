from __future__ import annotations

import math

import pytest

from raft_uav.baselines.radar_update_policy import (
    RadarUpdatePolicy,
    classify_radar_update_row,
)


def _candidate_count_only_policy() -> RadarUpdatePolicy:
    return RadarUpdatePolicy(
        entropy_soften=10.0,
        entropy_defer=11.0,
        effective_candidates_soften=2.0,
        effective_candidates_defer=3.0,
    )


@pytest.mark.parametrize("invalid_count", [0.0, -1.0])
def test_invalid_explicit_count_does_not_override_entropy(
    invalid_count: float,
) -> None:
    plan = classify_radar_update_row(
        {
            "association_effective_candidates": invalid_count,
            "association_soft_path_weight_entropy": math.log(4.0),
        },
        _candidate_count_only_policy(),
    )

    assert plan.action == "skip"
    assert plan.effective_candidates == pytest.approx(4.0)
    assert "effective_candidates>=3" in plan.reason


def test_invalid_first_alias_does_not_override_valid_second_alias() -> None:
    plan = classify_radar_update_row(
        {
            "association_effective_candidates": 0.0,
            "association_soft_path_effective_candidates": 2.5,
        },
        _candidate_count_only_policy(),
    )

    assert plan.action == "soften"
    assert plan.effective_candidates == pytest.approx(2.5)
    assert "effective_candidates>=2" in plan.reason


def test_negative_entropy_falls_back_to_raw_candidate_count() -> None:
    plan = classify_radar_update_row(
        {
            "association_soft_path_weight_entropy": -1.0,
            "association_soft_path_count": 4,
        },
        _candidate_count_only_policy(),
    )

    assert plan.action == "skip"
    assert plan.effective_candidates == pytest.approx(4.0)
