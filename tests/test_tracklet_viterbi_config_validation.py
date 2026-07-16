from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_candidates_per_frame", 1.5, "finite integer scalar"),
        ("soft_top_k_paths", True, "finite integer scalar"),
        (
            "reacquisition_miss_streak_threshold",
            np.array([2]),
            "finite integer scalar",
        ),
        ("missed_detection_cost", np.nan, "finite real scalar"),
        ("transition_position_std_m", np.inf, "finite real scalar"),
        ("range_gate_m", np.array([100.0]), "finite real scalar"),
        ("min_catprob", np.bool_(True), "finite real scalar"),
        ("soft_path_temperature", 1.0 + 0.0j, "finite real scalar"),
    ],
)
def test_tracklet_config_rejects_malformed_numeric_controls(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TrackletViterbiAssociationConfig(**{field: value})


def test_tracklet_config_normalizes_valid_scalar_like_controls() -> None:
    config = TrackletViterbiAssociationConfig(
        max_candidates_per_frame=np.array(4),
        missed_detection_cost=np.float64(7.5),
        range_gate_m=np.array(120.0),
        min_catprob=np.float64(0.2),
        reacquisition_miss_streak_threshold=np.int64(3),
        soft_top_k_paths=np.array(2),
        soft_path_temperature=np.float32(0.5),
    )

    assert config.max_candidates_per_frame == 4
    assert isinstance(config.max_candidates_per_frame, int)
    assert config.missed_detection_cost == 7.5
    assert isinstance(config.missed_detection_cost, float)
    assert config.range_gate_m == 120.0
    assert isinstance(config.range_gate_m, float)
    assert config.min_catprob == pytest.approx(0.2)
    assert config.reacquisition_miss_streak_threshold == 3
    assert config.soft_top_k_paths == 2
    assert config.soft_path_temperature == pytest.approx(0.5)
