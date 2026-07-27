from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.topk_weakz_tracklet import TopKWeakZTrackletConfig


_REAL_CONTROL_FIELDS = (
    "max_intra_tracklet_gap_s",
    "max_transition_gap_s",
    "max_transition_speed_mps",
    "max_transition_altitude_jump_m",
    "range_gate_m",
    "range_slack_m",
    "track_switch_cost",
    "gap_cost_per_s",
    "speed_cost_weight",
    "altitude_jump_cost_weight",
    "tracklet_length_reward",
    "catprob_reward_weight",
    "confidence_reward_weight",
    "range_penalty_weight",
    "weakz_radar_xy_std_m",
    "weakz_radar_z_std_m",
    "acceleration_std_mps2",
    "smoother_lag_s",
    "smoother_acceleration_std_mps2",
    "rf_radar_consistency_std_m",
    "rf_min_reliability",
    "rf_max_covariance_scale",
    "rf_outside_radar_scale",
    "rf_reject_distance_m",
    "replay_nis_weight",
    "replay_rejection_penalty",
)


@pytest.mark.parametrize("field_name", _REAL_CONTROL_FIELDS)
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_topk_config_rejects_nonfinite_real_controls(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        TopKWeakZTrackletConfig(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("top_k_paths", 2.5),
        ("beam_width", np.nan),
        ("max_tracklets", True),
        ("min_tracklet_length", np.array([3])),
    ],
)
def test_topk_config_rejects_non_integer_count_controls(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        TopKWeakZTrackletConfig(**{field_name: value})


def test_topk_config_accepts_valid_numpy_scalars() -> None:
    config = TopKWeakZTrackletConfig(
        top_k_paths=np.int64(2),
        beam_width=np.int64(4),
        replay_nis_weight=np.float64(0.1),
        rf_reject_distance_m=np.float64(250.0),
    )

    assert config.top_k_paths == 2
    assert config.beam_width == 4
    assert config.replay_nis_weight == 0.1
    assert config.rf_reject_distance_m == 250.0
