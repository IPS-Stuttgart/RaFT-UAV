import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.mot import MultiObjectTrackerConfig, run_mmuad_multi_object_tracker
from raft_uav.mmuad.schema import CandidateFrame


def test_mot_accepts_minimal_valid_candidate_frame_without_optional_columns() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["s1", "s1", "s1"],
                "time_s": [0.0, 1.0, 2.0],
                "source": ["radar", "radar", "radar"],
                "x_m": [0.0, 1.0, 2.0],
                "y_m": [0.0, 0.0, 0.0],
                "z_m": [2.0, 2.0, 2.0],
            }
        )
    )

    output = run_mmuad_multi_object_tracker(
        candidates,
        config=MultiObjectTrackerConfig(max_association_distance_m=5.0),
    )

    assert output.estimates["time_s"].tolist() == [0.0, 1.0, 2.0]
    assert output.estimates["update_action"].tolist() == [
        "new_track",
        "matched_update",
        "matched_update",
    ]
    assert output.estimates["output_track_id"].tolist() == ["mot_1", "mot_1", "mot_1"]
    assert output.metrics["pooled"] == {"count": 3, "track_count": 1}
    assert output.selected_tracklets["output_track_id"].tolist() == [
        "mot_1",
        "mot_1",
        "mot_1",
    ]


def test_mot_normalizes_confidence_before_sorting_and_thresholding() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["s1", "s1"],
                "time_s": [0.0, 0.0],
                "source": ["radar", "radar"],
                "x_m": [0.0, 1.0],
                "y_m": [0.0, 0.0],
                "z_m": [2.0, 2.0],
                "confidence": ["not-a-number", "0.7"],
            }
        )
    )

    output = run_mmuad_multi_object_tracker(
        candidates,
        config=MultiObjectTrackerConfig(min_new_track_confidence=0.05),
    )

    assert output.estimates["time_s"].tolist() == [0.0]
    assert output.estimates["state_x_m"].tolist() == [1.0]
    assert output.estimates["output_track_id"].tolist() == ["mot_1"]
    assert output.metrics["pooled"] == {"count": 1, "track_count": 1}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"acceleration_std_mps2": float("nan")}, "acceleration_std_mps2 must be finite"),
        (
            {"max_association_distance_m": -1.0},
            "max_association_distance_m must be nonnegative",
        ),
        ({"max_track_age_s": float("inf")}, "max_track_age_s must be finite"),
        (
            {"min_new_track_confidence": -0.1},
            "min_new_track_confidence must be nonnegative",
        ),
        ({"covariance_scale": 0.0}, "covariance_scale must be positive"),
        ({"covariance_scale": -1.0}, "covariance_scale must be positive"),
    ],
)
def test_mot_config_rejects_invalid_numeric_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        MultiObjectTrackerConfig(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("acceleration_std_mps2", True),
        ("max_association_distance_m", np.bool_(False)),
        ("max_track_age_s", np.array(True)),
        ("min_new_track_confidence", np.array([0.05])),
        ("covariance_scale", 1.0 + 0.0j),
        ("covariance_scale", np.ma.masked),
    ],
)
def test_mot_config_rejects_non_real_scalar_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        MultiObjectTrackerConfig(**{field: value})


def test_mot_config_accepts_zero_dimensional_real_scalars() -> None:
    config = MultiObjectTrackerConfig(
        acceleration_std_mps2=np.array(8.0),
        max_association_distance_m=np.float64(15.0),
        max_track_age_s=np.int64(2),
        min_new_track_confidence=np.array(0.05),
        covariance_scale=np.array(1.0),
    )

    assert config == MultiObjectTrackerConfig(
        acceleration_std_mps2=8.0,
        max_association_distance_m=15.0,
        max_track_age_s=2.0,
        min_new_track_confidence=0.05,
        covariance_scale=1.0,
    )
