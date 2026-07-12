from __future__ import annotations

import math

import pytest

from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig


_NUMERIC_CONFIG_FIELDS = (
    "max_candidates_per_frame",
    "missed_detection_cost",
    "consecutive_miss_cost",
    "track_switch_cost",
    "missing_track_id_cost",
    "catprob_weight",
    "anchor_nis_weight",
    "transition_nis_weight",
    "velocity_nis_weight",
    "transition_position_std_m",
    "transition_speed_std_mps",
    "velocity_std_mps",
    "max_speed_mps",
    "max_speed_penalty",
    "range_gate_m",
    "range_gate_slack_m",
    "range_penalty",
    "reacquisition_miss_streak_threshold",
    "reacquisition_gate_nis",
    "reacquisition_gate_growth",
    "reacquisition_reward",
    "reacquisition_outside_gate_penalty",
    "min_learned_candidate_probability",
    "min_catprob",
    "soft_top_k_paths",
    "soft_path_temperature",
)


@pytest.mark.parametrize("field", _NUMERIC_CONFIG_FIELDS)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_tracklet_viterbi_config_rejects_nonfinite_controls(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be finite"):
        TrackletViterbiAssociationConfig(**{field: value})


def test_tracklet_viterbi_config_keeps_optional_range_gate() -> None:
    config = TrackletViterbiAssociationConfig(range_gate_m=None)

    assert config.range_gate_m is None
